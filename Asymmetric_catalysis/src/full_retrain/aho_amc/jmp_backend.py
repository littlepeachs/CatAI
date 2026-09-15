from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from ase import Atoms
from torch import nn
from torch_geometric.data import Batch, Data


MOMA_ROOT = Path(os.environ.get("CATAI_MOMA_ROOT", "/root/autodl-tmp/MoMa"))
PICFM_ROOT = Path(os.environ.get("CATAI_PICFM_ROOT", "/root/autodl-tmp/Pi-CFM"))


def prepare_moma_runtime() -> None:
    """Apply local compatibility shims without modifying the MoMa checkout."""
    moma_path = str(MOMA_ROOT)
    if moma_path not in sys.path:
        sys.path.insert(0, moma_path)

    # MoMa's spherical-basis implementation predates NumPy 2.0.
    if not hasattr(np, "math"):
        np.math = math  # type: ignore[attr-defined]

    # PyG 2.8 only dispatches radius_graph through pyg-lib, while this
    # environment provides the matching torch-cluster CUDA/CPU extension.
    from torch_cluster import radius_graph as torch_cluster_radius_graph

    import ocpmodels.models.gemnet_oc_mt.goc_graph as goc_graph

    goc_graph.radius_graph = torch_cluster_radius_graph


def strip_distributed_prefix(key: str) -> str:
    prefix = "module.module."
    if not key.startswith(prefix):
        raise RuntimeError(f"Unexpected Pi-CFM state key prefix: {key}")
    return key.removeprefix(prefix)


def checkpoint_backbone_state(
    checkpoint: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("Pi-CFM checkpoint has no state_dict mapping")
    output = {
        strip_distributed_prefix(str(key)): value.detach().cpu()
        for key, value in state_dict.items()
        if "module_dict." not in str(key)
    }
    if not output:
        raise RuntimeError("Pi-CFM checkpoint yielded an empty backbone")
    return output


def sanitized_model_spec(
    checkpoint: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError("Pi-CFM checkpoint has no config mapping")
    model_name = str(config.get("model"))
    if model_name != "gemnet_oc_mt":
        raise RuntimeError(f"Unsupported Pi-CFM model: {model_name}")
    attributes = copy.deepcopy(dict(config.get("model_attributes", {})))
    # Scaling tensors are in the checkpoint; do not depend on the unavailable
    # training-host-relative scale file during construction.
    attributes["scale_file"] = None
    return model_name, attributes


def build_jmp_backbone(
    *,
    model_name: str,
    model_attributes: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    device: torch.device | str = "cpu",
) -> nn.Module:
    prepare_moma_runtime()
    from pydantic import TypeAdapter

    from ocpmodels.common.registry import registry
    from ocpmodels.models.gemnet_oc_mt.backbone import GemNetOCBackbone  # noqa: F401
    from ocpmodels.trainers.mt.config import ModelConfig

    attributes = copy.deepcopy(dict(model_attributes))
    attributes["scale_file"] = None
    model_config = TypeAdapter(ModelConfig).validate_python(
        {**attributes, "name": model_name}
    )
    model = registry.get_model_class(model_name)(
        {},
        model_config,
        **attributes,
    )
    model.load_state_dict(dict(state_dict), strict=True)
    return model.to(device)


class JMPGraphAdapter:
    """Convert non-periodic AHO ASE molecules to the JMP GemNet-OC graph."""

    def __init__(self, model_name: str, model_attributes: Mapping[str, Any]):
        prepare_moma_runtime()
        from pydantic import TypeAdapter

        from ocpmodels.trainers.mt.config import ModelConfig, TransformConfigs

        attributes = copy.deepcopy(dict(model_attributes))
        attributes["scale_file"] = None
        self.model_config = TypeAdapter(ModelConfig).validate_python(
            {**attributes, "name": model_name}
        )
        self.transform_configs = TransformConfigs(
            mt=None,
            model=self.model_config,
        )
        self.cutoff = float(
            attributes.get("absolute_rbf_cutoff")
            or attributes.get("cutoff")
            or 12.0
        )

    def from_ase_atoms_list(
        self,
        atoms_list: Sequence[Atoms],
        *,
        device: torch.device | str,
    ) -> Batch:
        from ocpmodels.trainers.ft.transform import qm9_transform

        data_list: list[Data] = []
        real_node_counts: list[int] = []
        virtual_node_counts: list[int] = []
        for graph_index, atoms in enumerate(atoms_list):
            numbers = torch.as_tensor(
                atoms.get_atomic_numbers(),
                dtype=torch.long,
            )
            positions = torch.as_tensor(
                atoms.get_positions(),
                dtype=torch.float32,
            )
            real_num_atoms = int(len(atoms))
            virtual_num_atoms = 0
            if real_num_atoms == 1:
                # GemNet-OC rejects an edge-free image. Add one deterministic
                # same-element virtual node just inside the radial cutoff, where
                # the polynomial envelope makes its message nearly zero. The
                # caller retains only the leading real-atom embedding.
                numbers = torch.cat([numbers, numbers[:1]], dim=0)
                virtual_position = positions[:1].clone()
                virtual_position[0, 0] += self.cutoff * (1.0 - 1e-4)
                positions = torch.cat(
                    [positions, virtual_position],
                    dim=0,
                )
                virtual_num_atoms = 1
            num_atoms = real_num_atoms + virtual_num_atoms
            data = Data(
                pos=positions,
                atomic_numbers=numbers,
                natoms=num_atoms,
                tags=torch.ones(num_atoms, dtype=torch.long),
                fixed=torch.zeros(num_atoms, dtype=torch.long),
                cell=(torch.eye(3, dtype=torch.float32) * 1000.0).unsqueeze(0),
                pbc=torch.zeros(3, dtype=torch.bool),
                y=torch.zeros(1, dtype=torch.float32),
                sid=torch.tensor([graph_index], dtype=torch.long),
            )
            data = qm9_transform(
                data,
                config=self.transform_configs,
                training=False,
            )
            data_list.append(data)
            real_node_counts.append(real_num_atoms)
            virtual_node_counts.append(virtual_num_atoms)
        batch = Batch.from_data_list(data_list)
        # Preserve the small interface used by the existing ORRA assembly.
        batch.n_node = batch.natoms
        batch.real_n_node = torch.tensor(
            real_node_counts,
            dtype=torch.long,
        )
        batch.virtual_n_node = torch.tensor(
            virtual_node_counts,
            dtype=torch.long,
        )
        return batch.to(device)


class JMPNodeBackbone(nn.Module):
    """Expose JMP node embeddings through the Orb-compatible output key."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        graph: Batch,
        *,
        return_final_interaction: bool = False,
    ) -> dict[str, torch.Tensor]:
        if not return_final_interaction:
            output = self.backbone._forward(graph)
            return {"node_features": output["node_embedding"]}

        interaction_blocks = getattr(self.backbone, "int_blocks", None)
        if interaction_blocks is None or len(interaction_blocks) == 0:
            raise RuntimeError(
                "JMP backbone has no interaction blocks to capture"
            )
        captured: list[torch.Tensor] = []

        def capture_final_interaction(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if (
                not isinstance(output, (tuple, list))
                or len(output) < 1
                or not isinstance(output[0], torch.Tensor)
            ):
                raise RuntimeError(
                    "Unexpected final JMP interaction-block output schema"
                )
            captured.append(output[0])

        handle = interaction_blocks[-1].register_forward_hook(
            capture_final_interaction
        )
        try:
            output = self.backbone._forward(graph)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError(
                "Expected exactly one final interaction feature capture; "
                f"got {len(captured)}"
            )
        final_features = output["node_embedding"]
        interaction_features = captured[0]
        if (
            not isinstance(final_features, torch.Tensor)
            or final_features.ndim != 2
            or interaction_features.ndim != 2
            or interaction_features.shape != final_features.shape
        ):
            raise RuntimeError(
                "Final interaction and node_embedding shapes do not match: "
                f"{tuple(interaction_features.shape)} vs "
                f"{tuple(getattr(final_features, 'shape', ()))}"
            )
        return {
            "node_features": final_features,
            "secondary_node_features": interaction_features,
        }


def load_fused_jmp_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
) -> tuple[JMPNodeBackbone, JMPGraphAdapter, dict[str, Any]]:
    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if payload.get("format") not in {
        "aho_jmp_amc_fused_backbone_v1",
        "cpa_jmp_amc_fused_backbone_v1",
    }:
        raise RuntimeError("Unsupported fused JMP checkpoint format")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("Fused JMP checkpoint has no state_dict")
    model_name = str(payload["model_name"])
    model_attributes = dict(payload["model_attributes"])
    backbone = build_jmp_backbone(
        model_name=model_name,
        model_attributes=model_attributes,
        state_dict=state_dict,
        device=device,
    )
    adapter = JMPGraphAdapter(model_name, model_attributes)
    return JMPNodeBackbone(backbone), adapter, payload
