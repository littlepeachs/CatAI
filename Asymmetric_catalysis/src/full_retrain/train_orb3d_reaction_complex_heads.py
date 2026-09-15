#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from full_retrain.data import ROOT, load_task
from full_retrain.metrics import best_threshold_accuracy, classification_metrics
from full_retrain.train_orb3d_classical_heads import build_model, prediction_values
from full_retrain.train_orb3d_site_attention import TOKEN_TYPE_IDS


ROLE_IDS = {
    "reactant": 0,
    "solvent": 2,
    "ligand": 8,
    "metal": 9,
    "catalyst": 10,
}

POOL_NAMES = (
    "all_atoms",
    "reactant_all",
    "reactant_site",
    "reactant_shell1",
    "reactant_shell2",
    "reactant_site_shell2",
    "solvent_all",
    "ligand_all",
    "ligand_donor",
    "ligand_chiral",
    "metal_all",
)

PAIR_NAMES = (
    ("reactant_site", "ligand_donor"),
    ("reactant_site_shell2", "ligand_donor"),
    ("reactant_site", "ligand_all"),
    ("reactant_site", "metal_all"),
    ("reactant_site_shell2", "metal_all"),
    ("ligand_donor", "metal_all"),
    ("reactant_all", "ligand_all"),
    ("solvent_all", "ligand_all"),
    ("solvent_all", "metal_all"),
)


@dataclass
class ReactionComplexStore:
    node_features: np.ndarray
    positions: np.ndarray
    atom_mask: np.ndarray
    conformer_mask: np.ndarray
    role_ids: np.ndarray
    token_type_ids: np.ndarray
    component_slots: np.ndarray
    component_ids: np.ndarray
    local_atom_indices: np.ndarray
    atomic_numbers: np.ndarray
    atom_descriptors: np.ndarray
    metadata: pd.DataFrame
    mapping: pd.DataFrame


def load_reaction_complex_store(complex_token_dir: Path) -> ReactionComplexStore:
    data = np.load(complex_token_dir / "reaction_complex_tokens.npz")
    return ReactionComplexStore(
        node_features=data["node_features"].astype(np.float16),
        positions=data["positions"].astype(np.float32),
        atom_mask=data["atom_mask"].astype(bool),
        conformer_mask=data["conformer_mask"].astype(bool),
        role_ids=data["role_ids"].astype(np.int64),
        token_type_ids=data["token_type_ids"].astype(np.int64),
        component_slots=data["component_slots"].astype(np.int16),
        component_ids=data["component_ids"].astype(np.int32),
        local_atom_indices=data["local_atom_indices"].astype(np.int16),
        atomic_numbers=data["atomic_numbers"].astype(np.int16),
        atom_descriptors=data["atom_descriptors"].astype(np.float32),
        metadata=pd.read_csv(complex_token_dir / "reaction_complex_metadata.csv"),
        mapping=pd.read_csv(complex_token_dir / "reaction_component_map.csv"),
    )


def conformer_feature_dim(store: ReactionComplexStore, conformer_feature_mode: str) -> int:
    base_dim = int(store.node_features.shape[-1])
    if conformer_feature_mode == "mean":
        return base_dim
    if conformer_feature_mode == "mean_std":
        return base_dim * 2
    if conformer_feature_mode == "mean_std_max":
        return base_dim * 3
    raise ValueError(f"Unsupported conformer feature mode: {conformer_feature_mode}")


def reaction_atom_features(
    store: ReactionComplexStore,
    reaction_id: int,
    conformer_feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    atom_mask = store.atom_mask[reaction_id]
    conf_mask = store.conformer_mask[reaction_id]
    active_atoms = int(atom_mask.sum())
    dim = conformer_feature_dim(store, conformer_feature_mode)
    if active_atoms == 0 or not conf_mask.any():
        return (
            np.zeros((active_atoms, dim), dtype=np.float32),
            np.zeros((active_atoms, 3), dtype=np.float32),
            atom_mask,
            False,
        )

    selected = store.node_features[reaction_id][conf_mask][:, atom_mask, :].astype(np.float32)
    mean = selected.mean(axis=0)
    if conformer_feature_mode == "mean":
        features = mean
    elif conformer_feature_mode == "mean_std":
        features = np.concatenate([mean, selected.std(axis=0)], axis=-1)
    elif conformer_feature_mode == "mean_std_max":
        features = np.concatenate([mean, selected.std(axis=0), selected.max(axis=0)], axis=-1)
    else:
        raise ValueError(f"Unsupported conformer feature mode: {conformer_feature_mode}")

    xyz = store.positions[reaction_id][conf_mask][:, atom_mask, :].astype(np.float32).mean(axis=0)
    return features.astype(np.float32), xyz.astype(np.float32), atom_mask, True


def pool_stats(features: np.ndarray, mask: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    if features.size == 0 or not np.any(mask):
        return np.zeros((dim * 3,), dtype=np.float32), np.zeros((dim,), dtype=np.float32)
    selected = features[mask].astype(np.float32)
    stats = np.concatenate([selected.mean(axis=0), selected.std(axis=0), selected.max(axis=0)], axis=0)
    return stats.astype(np.float32), selected.mean(axis=0).astype(np.float32)


def pair_geometry(xyz: np.ndarray, left_mask: np.ndarray, right_mask: np.ndarray) -> np.ndarray:
    if len(xyz) == 0 or not left_mask.any() or not right_mask.any():
        return np.zeros((5,), dtype=np.float32)
    left = xyz[left_mask].astype(np.float32)
    right = xyz[right_mask].astype(np.float32)
    centroid_distance = float(np.linalg.norm(left.mean(axis=0) - right.mean(axis=0)))
    distances = np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
    return np.array(
        [
            centroid_distance,
            float(distances.min()),
            float(distances.mean()),
            float(distances.max()),
            1.0,
        ],
        dtype=np.float32,
    )


def build_pool_masks(role_ids: np.ndarray, token_type_ids: np.ndarray) -> dict[str, np.ndarray]:
    masks = {
        "all_atoms": np.ones_like(role_ids, dtype=bool),
        "reactant_all": role_ids == ROLE_IDS["reactant"],
        "solvent_all": role_ids == ROLE_IDS["solvent"],
        "ligand_all": role_ids == ROLE_IDS["ligand"],
        "metal_all": role_ids == ROLE_IDS["metal"],
        "reactant_site": token_type_ids == TOKEN_TYPE_IDS["reactant_site"],
        "reactant_shell1": token_type_ids == TOKEN_TYPE_IDS["reactant_neighbor1"],
        "reactant_shell2": token_type_ids == TOKEN_TYPE_IDS["reactant_neighbor2"],
        "ligand_donor": token_type_ids == TOKEN_TYPE_IDS["ligand_donor"],
        "ligand_chiral": token_type_ids == TOKEN_TYPE_IDS["ligand_chiral"],
    }
    masks["reactant_site_shell2"] = (
        masks["reactant_site"] | masks["reactant_shell1"] | masks["reactant_shell2"]
    )
    return masks


def feature_row(store: ReactionComplexStore, reaction_id: int, conformer_feature_mode: str) -> np.ndarray:
    dim = conformer_feature_dim(store, conformer_feature_mode)
    features, xyz, atom_mask, has_3d = reaction_atom_features(store, reaction_id, conformer_feature_mode)
    active_role_ids = store.role_ids[reaction_id][atom_mask]
    active_token_type_ids = store.token_type_ids[reaction_id][atom_mask]
    active_atomic_numbers = store.atomic_numbers[reaction_id][atom_mask]
    masks = build_pool_masks(active_role_ids, active_token_type_ids)

    pools: dict[str, np.ndarray] = {}
    means: dict[str, np.ndarray] = {}
    scalars: list[float] = [float(atom_mask.sum()), float(has_3d), float(store.conformer_mask[reaction_id].sum())]
    for name in POOL_NAMES:
        stats, mean = pool_stats(features, masks[name], dim)
        pools[name] = stats
        means[name] = mean
        scalars.extend([float(masks[name].sum()), float(masks[name].any())])

    blocks: list[np.ndarray] = [pools[name] for name in POOL_NAMES]
    geometry_blocks: list[np.ndarray] = []
    for left_name, right_name in PAIR_NAMES:
        left = means[left_name]
        right = means[right_name]
        blocks.extend([left * right, np.abs(left - right)])
        geometry_blocks.append(pair_geometry(xyz, masks[left_name], masks[right_name]))

    token_type_counts = np.bincount(
        np.clip(active_token_type_ids, 0, max(TOKEN_TYPE_IDS.values())),
        minlength=max(TOKEN_TYPE_IDS.values()) + 1,
    ).astype(np.float32)
    role_counts = np.array(
        [
            float((active_role_ids == ROLE_IDS["reactant"]).sum()),
            float((active_role_ids == ROLE_IDS["solvent"]).sum()),
            float((active_role_ids == ROLE_IDS["ligand"]).sum()),
            float((active_role_ids == ROLE_IDS["metal"]).sum()),
        ],
        dtype=np.float32,
    )
    atomic_summary = np.array(
        [
            float(active_atomic_numbers.mean()) if len(active_atomic_numbers) else 0.0,
            float(active_atomic_numbers.std()) if len(active_atomic_numbers) else 0.0,
            float(active_atomic_numbers.max()) if len(active_atomic_numbers) else 0.0,
            float((active_atomic_numbers > 18).sum()),
        ],
        dtype=np.float32,
    )
    blocks.append(np.concatenate(geometry_blocks, axis=0).astype(np.float32))
    blocks.append(token_type_counts)
    blocks.append(role_counts)
    blocks.append(atomic_summary)
    blocks.append(np.asarray(scalars, dtype=np.float32))
    return np.concatenate(blocks, axis=0).astype(np.float32)


def build_feature_frame(
    *,
    store: ReactionComplexStore,
    part: str,
    conformer_feature_mode: str,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = store.metadata[store.metadata["split_part"].astype(str) == part].sort_values("reaction_id").copy()
    if max_rows > 0:
        metadata = metadata.iloc[:max_rows].copy()
    features: list[np.ndarray] = []
    targets: list[float] = []
    row_indices: list[int] = []
    for meta in metadata.itertuples(index=False):
        y_value = pd.to_numeric(meta.y_true, errors="coerce")
        if not np.isfinite(y_value):
            continue
        reaction_id = int(meta.reaction_id)
        features.append(feature_row(store, reaction_id, conformer_feature_mode))
        targets.append(float(y_value))
        row_indices.append(int(meta.source_index))
    if not features:
        raise ValueError(f"No valid rows for {part}")
    x = np.stack(features, axis=0).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    idx = np.asarray(row_indices, dtype=int)
    print(f"[features] part={part} rows={len(y)} dim={x.shape[1]}", flush=True)
    return x, y, idx


def train_eval_one(model_name: str, args, bundle, feature_sets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    train_x, train_y, _ = feature_sets["train"]
    val_x, val_y, _ = feature_sets["val"]
    test_x, test_y, test_row_index = feature_sets["test"]

    model = build_model(model_name, bundle.task_type, args.seed)
    model.fit(train_x, train_y)
    val_pred = prediction_values(model, model_name, bundle.task_type, val_x)
    threshold = best_threshold_accuracy(val_y, val_pred)["best_threshold"]
    test_pred = prediction_values(model, model_name, bundle.task_type, test_x)

    model_tag = f"orbmol_v2_reaction_complex_{args.conformer_feature_mode}_{model_name}"
    row = {
        "task": bundle.name,
        "split": args.split,
        "model": model_tag,
        "feature": f"pseudo_complex_{args.conformer_feature_mode}_atom_tokens_orbmol_v2",
        "conformer_feature_mode": args.conformer_feature_mode,
        "complex_token_dir": str(args.complex_token_dir),
        "n_train": int(len(train_y)),
        "n_val": int(len(val_y)),
    }
    row.update(classification_metrics(test_y, test_pred, threshold=threshold))
    row.update(best_threshold_accuracy(test_y, test_pred))

    pred_df = pd.DataFrame(
        {
            "row_index": test_row_index,
            "y_true": test_y,
            "y_prob": test_pred,
            "y_pred": (test_pred >= threshold).astype(int),
        }
    )
    model_path = args.out_dir / f"{bundle.name}__{model_tag}.joblib"
    pred_path = args.out_dir / f"{bundle.name}__{model_tag}__predictions.csv"
    joblib.dump(model, model_path)
    pred_df.to_csv(pred_path, index=False)
    row["model_path"] = str(model_path)
    row["prediction_path"] = str(pred_path)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical heads on standalone OrbMol-v2 reaction pseudo-complex tokens.")
    parser.add_argument("--split", default="paper_exact_test", choices=["paper_exact_test", "paper_random42", "example"])
    parser.add_argument(
        "--complex-token-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_reaction_complex_tokens_aho_major",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_reaction_complex_classical_aho_major",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "logreg", "extra_trees", "hgb", "mlp", "xgb"],
        choices=["ridge", "logreg", "knn", "rf", "extra_trees", "hgb", "mlp", "xgb"],
    )
    parser.add_argument(
        "--conformer-feature-mode",
        default="mean",
        choices=["mean", "mean_std", "mean_std_max"],
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    store = load_reaction_complex_store(args.complex_token_dir)
    bundle = load_task("aho_major", split=args.split)
    if bundle.task_type != "classification":
        raise ValueError("Reaction complex heads currently target AHO major classification.")
    feature_sets = {
        part: build_feature_frame(
            store=store,
            part=part,
            conformer_feature_mode=args.conformer_feature_mode,
            max_rows=args.max_train_rows if part == "train" else args.max_eval_rows,
        )
        for part in ("train", "val", "test")
    }

    rows = []
    for model_name in args.models:
        print(f"[train] model=orbmol_v2_reaction_complex_{args.conformer_feature_mode}_{model_name}", flush=True)
        rows.append(train_eval_one(model_name, args, bundle, feature_sets))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    with open(args.out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(metrics.to_string(index=False))
    print(f"Wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
