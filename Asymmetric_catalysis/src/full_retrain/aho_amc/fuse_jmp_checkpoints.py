from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from full_retrain.aho_amc.common import (
    AMC_OUTPUT_ROOT,
    CONFORMER_CACHE,
    sha256_file,
    tensor_schema,
    write_json,
)
from full_retrain.aho_amc.jmp_backend import (
    PICFM_ROOT,
    JMPGraphAdapter,
    JMPNodeBackbone,
    build_jmp_backbone,
    checkpoint_backbone_state,
    prepare_moma_runtime,
    sanitized_model_spec,
)
from full_retrain.train_unified_orra_orb_finetune import payload_to_atoms


TASKS = (
    "aho_major",
    "aho_ddg",
    "cpa_main",
    "cpa_transformation",
    "cpa_imine",
    "cpa_thiol",
    "cpa_catalyst",
)
RUNTIME_GRAPH_CONFIG_KEYS = {
    "absolute_rbf_cutoff",
    "cutoff",
    "cutoff_qint",
    "cutoff_aeaint",
    "cutoff_aint",
    "max_neighbors",
    "max_neighbors_qint",
    "max_neighbors_aeaint",
    "max_neighbors_aint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize one parameter-space JMP AMC-Fuse backbone per AHO task. "
            "Task heads are never fused and no prediction ensemble is created."
        )
    )
    parser.add_argument(
        "--task",
        action="append",
        choices=TASKS,
        default=[],
        help="Optional task; repeat for both. Default: both.",
    )
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=AMC_OUTPUT_ROOT / "selections",
    )
    parser.add_argument("--picfm-root", type=Path, default=PICFM_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=AMC_OUTPUT_ROOT / "jmp" / "fused_checkpoints",
    )
    parser.add_argument(
        "--conformer-cache",
        type=Path,
        default=CONFORMER_CACHE,
        help=(
            "Task-local component conformers used only for strict reload and "
            "single-backbone forward verification."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=8,
        help=(
            "Bound CPU BLAS/torch parallelism during JMP construction so "
            "multi-GPU materialization waves do not oversubscribe the host."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(
            value.reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def fuse_jmp_backbones(
    *,
    picfm_root: Path,
    module_names: list[str],
    weights: list[float],
) -> tuple[
    dict[str, torch.Tensor],
    str,
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    weight_array = np.asarray(weights, dtype=np.float64)
    if (
        not module_names
        or len(module_names) != len(weight_array)
        or not np.isfinite(weight_array).all()
        or np.any(weight_array < 0.0)
        or abs(float(weight_array.sum()) - 1.0) >= 1e-6
    ):
        raise RuntimeError("JMP AMC weights fail simplex constraints")

    expected_schema: dict[str, dict[str, Any]] | None = None
    expected_model_name: str | None = None
    expected_attributes: dict[str, Any] | None = None
    accumulators: dict[str, torch.Tensor] = {}
    output_dtypes: dict[str, torch.dtype] = {}
    fixed_tensors: dict[str, torch.Tensor] = {}
    sources: dict[str, dict[str, Any]] = {}

    for module_name, weight in zip(module_names, weight_array, strict=True):
        checkpoint_path = picfm_root / module_name / "best_checkpoint.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_sha = sha256_file(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        model_name, attributes = sanitized_model_spec(checkpoint)
        state = checkpoint_backbone_state(checkpoint)
        schema = tensor_schema(state)
        if expected_schema is None:
            expected_schema = schema
            expected_model_name = model_name
            expected_attributes = attributes
            for key, value in state.items():
                output_dtypes[key] = value.dtype
                if torch.is_floating_point(value):
                    accumulators[key] = torch.zeros_like(
                        value,
                        dtype=torch.float64,
                    )
                else:
                    fixed_tensors[key] = value.clone()
            config_drift: dict[str, dict[str, Any]] = {}
        else:
            if schema != expected_schema:
                raise RuntimeError(f"JMP backbone schema drift: {module_name}")
            if model_name != expected_model_name:
                raise RuntimeError(f"JMP model config drift: {module_name}")
            config_drift = {
                key: {
                    "canonical": expected_attributes.get(key),
                    "source": attributes.get(key),
                }
                for key in sorted(set(expected_attributes) | set(attributes))
                if expected_attributes.get(key) != attributes.get(key)
            }
            unsupported_drift = set(config_drift).difference(
                RUNTIME_GRAPH_CONFIG_KEYS
            )
            if unsupported_drift:
                raise RuntimeError(
                    f"JMP structural model config drift in {module_name}: "
                    f"{sorted(unsupported_drift)}"
                )

        for key, value in state.items():
            if torch.is_floating_point(value):
                accumulators[key].add_(
                    value.to(torch.float64),
                    alpha=float(weight),
                )
            elif not torch.equal(value, fixed_tensors[key]):
                raise RuntimeError(
                    f"Non-floating JMP tensor differs across modules: {key}"
                )
        sources[module_name] = {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
            "weight": float(weight),
            "backbone_tensor_count": len(state),
            "runtime_graph_config_drift": config_drift,
        }
        del checkpoint, state

    assert expected_schema is not None
    assert expected_model_name is not None
    assert expected_attributes is not None
    fused = {
        key: accumulators[key].to(output_dtypes[key])
        if key in accumulators
        else fixed_tensors[key]
        for key in expected_schema
    }
    if tensor_schema(fused) != expected_schema:
        raise RuntimeError("Fused JMP backbone schema changed")
    if not all(
        torch.isfinite(value).all().item()
        for value in fused.values()
        if torch.is_floating_point(value)
    ):
        raise RuntimeError("Fused JMP backbone contains NaN/Inf")
    return fused, expected_model_name, expected_attributes, sources


def verify_fused_checkpoint(
    *,
    checkpoint_path: Path,
    expected_state: Mapping[str, torch.Tensor],
    payloads: dict[int, dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = build_jmp_backbone(
        model_name=checkpoint["model_name"],
        model_attributes=checkpoint["model_attributes"],
        state_dict=checkpoint["state_dict"],
        device="cpu",
    )
    loaded_state = model.state_dict()
    state_exact = (
        set(loaded_state) == set(expected_state)
        and all(
            torch.equal(loaded_state[key], expected_state[key])
            for key in expected_state
        )
    )
    if not state_exact:
        raise RuntimeError("Reloaded JMP backbone differs from fused tensors")

    wrapper = JMPNodeBackbone(model.to(device).eval())
    adapter = JMPGraphAdapter(
        checkpoint["model_name"],
        checkpoint["model_attributes"],
    )
    sample_id = min(payloads)
    sample_positions = np.asarray(payloads[sample_id]["positions"])
    sample_conformer_count = (
        1 if sample_positions.ndim == 2 else len(sample_positions)
    )
    sample_atoms = [
        payload_to_atoms(payloads[sample_id], conformer_index)
        for conformer_index in range(sample_conformer_count)
    ]
    sample_graph = adapter.from_ase_atoms_list(sample_atoms, device=device)
    batch_ids = sorted(payloads)[:4]
    batch_atoms = [
        payload_to_atoms(payloads[component_id], conformer_index)
        for component_id in batch_ids
        for conformer_index in range(
            1
            if np.asarray(payloads[component_id]["positions"]).ndim == 2
            else len(np.asarray(payloads[component_id]["positions"]))
        )
    ]
    batch_graph = adapter.from_ase_atoms_list(batch_atoms, device=device)
    with torch.inference_mode():
        sample_first = wrapper(sample_graph)["node_features"]
        sample_second = wrapper(sample_graph)["node_features"]
        batch_output = wrapper(batch_graph)["node_features"]
    expected_sample_nodes = int(sample_graph.n_node.sum())
    expected_batch_nodes = int(batch_graph.n_node.sum())
    if tuple(sample_first.shape) != (expected_sample_nodes, 256):
        raise RuntimeError(
            f"Unexpected JMP sample output shape: {tuple(sample_first.shape)}"
        )
    if tuple(batch_output.shape) != (expected_batch_nodes, 256):
        raise RuntimeError(
            f"Unexpected JMP batch output shape: {tuple(batch_output.shape)}"
        )
    if not (
        torch.isfinite(sample_first).all()
        and torch.isfinite(sample_second).all()
        and torch.isfinite(batch_output).all()
    ):
        raise RuntimeError("Non-finite fused JMP forward")

    difference = (sample_first - sample_second).abs()
    cosine = F.cosine_similarity(
        sample_first.float(),
        sample_second.float(),
        dim=1,
    )
    relative_l2 = (
        (sample_first.float() - sample_second.float()).norm(dim=1)
        / sample_first.float().norm(dim=1).clamp_min(1e-12)
    )
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    minimum_cosine = float(cosine.min())
    maximum_relative_l2 = float(relative_l2.max())
    repeat_pass = (
        max_abs <= 5e-3
        and mean_abs <= 5e-4
        and minimum_cosine >= 0.99999
        and maximum_relative_l2 <= 2e-3
    )
    if not repeat_pass:
        raise RuntimeError(
            "JMP repeat-forward tolerance failed: "
            f"max_abs={max_abs}, mean_abs={mean_abs}, "
            f"min_cosine={minimum_cosine}, "
            f"max_relative_l2={maximum_relative_l2}"
        )
    return {
        "strict_reload_pass": True,
        "backbone_state_exact_after_reload": state_exact,
        "backbone_tensor_count": len(loaded_state),
        "sample_component_id": sample_id,
        "sample_conformer_count": sample_conformer_count,
        "sample_output_shape": list(sample_first.shape),
        "batch_component_ids": batch_ids,
        "batch_conformer_graph_count": len(batch_atoms),
        "batch_output_shape": list(batch_output.shape),
        "batch_real_node_count": int(batch_graph.real_n_node.sum()),
        "batch_virtual_node_count": int(batch_graph.virtual_n_node.sum()),
        "outputs_finite": True,
        "repeat_forward_max_abs_difference": max_abs,
        "repeat_forward_mean_abs_difference": mean_abs,
        "repeat_forward_minimum_cosine_similarity": minimum_cosine,
        "repeat_forward_maximum_relative_l2": maximum_relative_l2,
        "repeat_forward_numerically_equivalent": repeat_pass,
        "runtime_jmp_backbone_instances": 1,
    }


def main() -> None:
    args = parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    prepare_moma_runtime()
    tasks = args.task or list(TASKS)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    payloads = joblib.load(args.conformer_cache)

    for task in tasks:
        selection_path = args.selection_dir / f"{task}.json"
        selection = read_json(selection_path)
        if selection.get("task") != task:
            raise RuntimeError(f"Selection task mismatch: {task}")
        if selection.get("test_predictions_used") is not False:
            raise RuntimeError(f"Selection for {task} is not test-sealed")
        selection_family = selection.get("backbone_family")
        selection_lineage = selection.get("selection_lineage")
        if selection_family not in {None, "jmp"}:
            raise RuntimeError(
                "JMP fusion only accepts an explicitly JMP-native selection "
                "or a legacy transfer seed without a family label"
            )
        if selection_family == "jmp":
            accepted_native_lineages = {
                "family_native_30_module_embedding_reselection",
                "catalyst_nested_group_cv_30_module_reselection",
            }
            if selection_lineage not in accepted_native_lineages:
                raise RuntimeError(
                    "JMP-native selection is missing the required 30-module "
                    "re-selection lineage"
                )
            if (
                selection_lineage
                == "catalyst_nested_group_cv_30_module_reselection"
                and task != "cpa_catalyst"
            ):
                raise RuntimeError(
                    "Catalyst nested-group selection lineage is only valid "
                    "for cpa_catalyst"
                )
            dataset_name = "AHO" if task.startswith("aho_") else "CPA"
            if (
                selection_lineage
                == "catalyst_nested_group_cv_30_module_reselection"
            ):
                initialization_note = (
                    "Weights were fitted from all 30 Pi-CFM JMP backbone "
                    "embeddings using nested Catalyst-group OOF over CPA "
                    "train+validation only. The sealed test was not used and "
                    "no Orb AMC weights were reused."
                )
            else:
                initialization_note = (
                    "Weights were fitted from all 30 Pi-CFM JMP backbone "
                    f"embeddings on {dataset_name} train only; pooling and K "
                    f"were selected on {dataset_name} validation. No Orb AMC "
                    "weights were reused."
                )
        else:
            initialization_note = (
                "AHO-causal AMC module names and weights transferred by "
                "exact task name from the Orb family into homologous "
                "Pi-CFM JMP task checkpoints; this checkpoint is a "
                "transfer-seeded preflight, not JMP-family re-selection."
            )
        module_names = list(selection["module_names"])
        weights = [float(value) for value in selection["weights"]]
        fused_state, model_name, model_attributes, sources = (
            fuse_jmp_backbones(
                picfm_root=args.picfm_root,
                module_names=module_names,
                weights=weights,
            )
        )
        output_path = (
            args.output_dir / f"{task}_jmp_amc_fused.pt"
        ).resolve()
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"{output_path} exists; use --overwrite to replace it"
            )
        checkpoint_payload = {
            "format": (
                "aho_jmp_amc_fused_backbone_v1"
                if task.startswith("aho_")
                else "cpa_jmp_amc_fused_backbone_v1"
            ),
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "task": task,
            "model_name": model_name,
            "model_attributes": model_attributes,
            "state_dict": fused_state,
            "fusion": {
                "type": "parameter_space_weighted_average",
                "module_names": module_names,
                "weights": weights,
                "weight_sum": float(sum(weights)),
                "task_heads_fused": False,
                "prediction_space_ensemble": False,
                "ensemble_size": 1,
                "single_checkpoint": True,
                "selection_source": str(selection_path.resolve()),
                "selection_source_sha256": sha256_file(selection_path),
                "selection_backbone_family": selection_family,
                "selection_lineage": selection_lineage,
                "selection_test_predictions_used": False,
                "initialization_note": initialization_note,
            },
            "sources": sources,
        }
        atomic_torch_save(checkpoint_payload, output_path)
        verification = verify_fused_checkpoint(
            checkpoint_path=output_path,
            expected_state=fused_state,
            payloads=payloads,
            device=device,
        )
        manifest = {
            "format_version": 1,
            "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "task": task,
            "checkpoint_path": str(output_path),
            "checkpoint_sha256": sha256_file(output_path),
            "checkpoint_format": checkpoint_payload["format"],
            "architecture": "JMP-large/GemNet-OC",
            "model_name": model_name,
            "parameter_space_fusion": True,
            "task_heads_fused": False,
            "prediction_space_ensemble": False,
            "ensemble_size": 1,
            "single_checkpoint": True,
            "selection_path": str(selection_path.resolve()),
            "selection_sha256": sha256_file(selection_path),
            "selection_backbone_family": selection_family,
            "selection_lineage": selection_lineage,
            "selection_test_predictions_used": False,
            "module_names": module_names,
            "weights": weights,
            "weight_sum": float(sum(weights)),
            "sources": sources,
            "fused_backbone_tensor_count": len(fused_state),
            "fused_backbone_state_sha256": state_hash(fused_state),
            "verification": verification,
            "all_pass": (
                verification["strict_reload_pass"]
                and verification["backbone_state_exact_after_reload"]
                and verification["repeat_forward_numerically_equivalent"]
                and verification["runtime_jmp_backbone_instances"] == 1
            ),
        }
        write_json(output_path.with_suffix(".manifest.json"), manifest)
        if not manifest["all_pass"]:
            raise RuntimeError(f"Fused JMP checkpoint audit failed: {task}")
        print(
            f"{task}: wrote {output_path}; modules={module_names}; "
            f"weights={[round(value, 6) for value in weights]}; "
            "strict/reload/forward PASS",
            flush=True,
        )
        del fused_state, checkpoint_payload
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
