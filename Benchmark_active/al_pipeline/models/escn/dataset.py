"""Dataset wrapper used by ESCN/DPP pipelines."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from cgcnn_model.cgcnn.data_process import CIFData

try:
    from torch_geometric.data import Data
except ImportError as exc:  # pragma: no cover - runtime safeguard
    raise RuntimeError("ESCN and DPP backends require torch_geometric dependencies.") from exc

from ocpmodels_catai.common.registry import registry
from ocpmodels_catai.models.gemnet_oc_mt.backbone import GemNetOCBackbone
from ocpmodels_catai.trainers.mt.config import GemNetOCModelConfig

from torch_geometric.data import Data, Batch
from ocpmodels_catai.models.gemnet_oc_mt.goc_graph import Cutoffs,MaxNeighbors
from ocpmodels_catai.trainers.mt.transform import _generate_graphs
from ocpmodels_catai.trainers.mt.config import TransformConfigs

from typing import Mapping
import fnmatch
from logging import getLogger
log = getLogger(__name__)

def _common_transform(data: Data):
    if hasattr(data, "energy"):
        if not torch.is_tensor(data.energy):
            data.energy = torch.tensor(data.energy, dtype=torch.float)
        data.energy = data.energy.view(-1).float()

    if not hasattr(data, "sid"):
        if hasattr(data, "absolute_idx"):
            data.sid = data.absolute_idx
    if not hasattr(data, "natoms"):
        data.natoms = data.num_nodes

    # data.fixed = torch.ones(data.natoms)
    if not hasattr(data, "fixed"):
        data.fixed = torch.zeros(data.natoms, dtype=torch.float)

    if not hasattr(data, "tags"):
        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()
    if not hasattr(data, "cell"):
        max_length = 1000.0
        data.cell = (torch.eye(3) * max_length).unsqueeze(dim=0)

    return data

def _common_transform_all(data: Data):
    data.fixed = data.fixed.long()
    data.atomic_numbers = data.atomic_numbers.long()
    data.tags = data.tags.long()

    data.natoms = int(
        data.natoms.item() if torch.is_tensor(data) else data.natoms
    )

    return data

def matbench_transform(
    data: Data, *, config: TransformConfigs, training: bool
):
    # Decrease max neighbors for very large systems to avoid memory issues.
    if data.natoms > 300:
        max_neighbors = 5
    elif data.natoms > 200:
        max_neighbors = 10
    else:
        max_neighbors = 30
    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
        pbc=True,
        training=training,
    )
    data = _common_transform(data)
    data = _common_transform_all(data)
    return data

model_configs = GemNetOCModelConfig(
        num_targets=1,
        num_spherical=7,
        num_radial=128,
        num_blocks=6,
        emb_size_atom=256,
        emb_size_edge=1024,
        emb_size_trip_in=64,
        emb_size_trip_out=128,
        emb_size_quad_in=64,
        emb_size_quad_out=32,
        emb_size_aint_in=64,
        emb_size_aint_out=64,
        emb_size_rbf=32,
        emb_size_cbf=16,
        emb_size_sbf=64,
        num_before_skip=2,
        num_after_skip=2,
        num_concat=4,
        num_atom=3,
        num_output_afteratom=3,
        num_atom_emb_layers=2,
        num_global_out_layers=2,
        regress_forces=False,
        regress_energy=True,
        direct_forces=False,
        use_pbc=True,
        scale_backprop_forces=False,
        rbf={"name": "gaussian"},
        rbf_spherical=None,
        envelope={"name": "polynomial", "exponent": 5},
        cbf={"name": "spherical_harmonics"},
        sbf={"name": "legendre_outer"},
        extensive=True,
        forces_coupled=False,
        activation="silu",
        quad_interaction=True,
        atom_edge_interaction=True,
        edge_atom_interaction=True,
        atom_interaction=True,
        scale_basis=False,
        qint_tags=[1, 2],
        num_elements=120,
        otf_graph=True,
        ln=False,
        dropout=0.0,
        scale_file="configs/fm_scale_files/large.pt",
        absolute_rbf_cutoff=12.0,
        learnable_rbf=False,
        learnable_rbf_stds=False,
        unique_basis_per_layer=False,
        old_gaussian_implementation=False,
        edge_dropout=0.0,
        name="gemnet_oc_mt",
    )

class EscnCIFDataset(torch.utils.data.Dataset):
    """Wrap CIFData to provide PyG-compatible graphs for ESCN/DPP backends."""

    def __init__(
        self,
        label_dir: str,
        structure_dir: str,
        properties: List[str],
        aug: bool = False,
        process_dir: Optional[str] = None,
        cache_filename: str = "processed.pt",
        rebuild_processed: bool = False,
        load_processed: bool = True,
        process_on_init: bool = True,
    ):
        self.base = CIFData(
            label_dir=label_dir,
            properties=properties,
            struct_dir=structure_dir,
            aug=aug,
        )
        self.properties = properties
        self.id_prop_data = self.base.id_prop_data
        self.legacy_mode = self.base.legacy_mode
        self.simple_mode = self.base.simple_mode
        self.transform_config = TransformConfigs(mt=None,model=model_configs)
        self.process_dir = process_dir
        self.cache_filename = cache_filename
        self.process_path = (
            os.path.join(self.process_dir, self.cache_filename)
            if self.process_dir
            else None
        )
        self._processed_data: Optional[List[Data]] = None

        if self.process_dir:
            os.makedirs(self.process_dir, exist_ok=True)

            if (
                self.process_path
                and load_processed
                and not rebuild_processed
                and os.path.exists(self.process_path)
            ):
                self._processed_data = torch.load(
                    self.process_path, map_location="cpu"
                )
                log.info(
                    "Loaded processed ESCN data from %s (n=%d)",
                    self.process_path,
                    len(self._processed_data),
                )
            elif self.process_path and process_on_init:
                self._process_and_cache()

    def __len__(self) -> int:
        return len(self.base)

    def _resolve_structure(self, idx: int):
        if self.legacy_mode:
            row = self.id_prop_data.iloc[idx]
            cif_id = row["cif_id"]
            prime_struct = row["prime_struct"]
            struct_path = os.path.join(self.base.struct_dir, prime_struct) + ".cif"
            structure = self.base.get_structure(struct_path)
            identifier = str(cif_id)
        else:
            row = self.id_prop_data.iloc[idx]
            cif_name = row["Crystals"]
            cand1 = os.path.join(self.base.struct_dir, "structures", cif_name)
            cand2 = os.path.join(self.base.struct_dir, cif_name)
            struct_path = cand1 if os.path.exists(cand1) else cand2
            if not struct_path.endswith(".cif"):
                struct_path += ".cif"
            if not os.path.exists(struct_path):
                raise FileNotFoundError(f"Structure file not found: {struct_path}")
            structure = self.base.get_structure(struct_path)
            identifier = str(cif_name)
        targets = torch.tensor(row[self.properties].to_numpy(dtype=float), dtype=torch.float32)
        return structure, identifier, targets

    def _surface_mask(self, structure) -> Tuple[torch.Tensor, torch.Tensor]:
        coords = np.asarray(structure.frac_coords[:, 2])
        if coords.size == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
        max_z = float(coords.max())
        mask = coords >= max_z * 0.95
        if not mask.any():
            mask = coords >= max_z - 1e-3
        idxs = torch.from_numpy(np.where(mask)[0]).long()
        tags = torch.zeros(coords.size, dtype=torch.long)
        fixed = torch.ones(coords.size, dtype=torch.long)
        if idxs.numel() > 0:
            tags[idxs] = 1
            fixed[idxs] = 0
        return tags, fixed

    def _build_item(self, idx: int) -> Data:
        structure, identifier, target = self._resolve_structure(idx)
        pos = torch.tensor(structure.cart_coords, dtype=torch.float32)
        atomic_numbers = torch.tensor([site.specie.number for site in structure], dtype=torch.long)
        cell = torch.tensor(structure.lattice.matrix, dtype=torch.float32).unsqueeze(0)
        natoms = torch.tensor([len(structure)], dtype=torch.long)
        tags, fixed = self._surface_mask(structure)
        data = Data(
            pos=pos,
            cell=cell,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            y=target,
        )
        data.pbc = torch.ones((1, 3), dtype=torch.bool)
        data.tags = tags
        data.fixed = fixed
        data.sample_idx = torch.tensor([idx], dtype=torch.long)
        data.identifier = identifier
        data.y_relaxed = data.y.clone()
        data = matbench_transform(data , config=self.transform_config, training=True)
        
        return data

    def _process_and_cache(self):
        if not self.process_path:
            return
        processed: List[Data] = []
        for idx in range(len(self.base)):
            processed.append(self._build_item(idx))
        torch.save(processed, self.process_path)
        self._processed_data = processed
        log.info(
            "Processed and cached ESCN data to %s (n=%d)",
            self.process_path,
            len(processed),
        )

    def __getitem__(self, idx: int):
        if self._processed_data is not None:
            return self._processed_data[idx]
        return self._build_item(idx)

    def get_identifier(self, idx: int) -> str:
        if self.legacy_mode:
            return str(self.id_prop_data.iloc[idx]["cif_id"])
        return str(self.id_prop_data.iloc[idx]["Crystals"])

__all__ = ["EscnCIFDataset"]
