#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from full_retrain.data import ROOT, load_task
from full_retrain.metrics import best_threshold_accuracy, classification_metrics, regression_metrics
from full_retrain.train_orb3d_classical_heads import build_model, prediction_values
from full_retrain.train_orb3d_site_attention import (
    load_atom_token_store,
    neighbor_shells,
    site_indices_for_reactant,
    split_frames,
)


RDLogger.DisableLog("rdApp.*")


POOL_NAMES = (
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
    ("reactant_site_shell2", "ligand_all"),
    ("reactant_site", "metal_all"),
    ("reactant_all", "ligand_all"),
    ("solvent_all", "ligand_all"),
)


def conformer_feature_dim(store, conformer_feature_mode: str) -> int:
    base_dim = int(store.node_features.shape[-1])
    if conformer_feature_mode == "mean":
        return base_dim
    if conformer_feature_mode == "mean_std":
        return base_dim * 2
    if conformer_feature_mode == "mean_std_max":
        return base_dim * 3
    raise ValueError(f"Unsupported conformer feature mode: {conformer_feature_mode}")


def conformer_pooled_atom_features(store, component_id: int, conformer_feature_mode: str) -> tuple[np.ndarray, bool]:
    atom_mask = store.atom_mask[component_id]
    conf_mask = store.conformer_mask[component_id]
    n_atoms = int(atom_mask.sum())
    dim = conformer_feature_dim(store, conformer_feature_mode)
    if not conf_mask.any() or not atom_mask.any():
        return np.zeros((n_atoms, dim), dtype=np.float32), False

    selected = store.node_features[component_id][conf_mask][:, atom_mask, :].astype(np.float32)
    mean = selected.mean(axis=0)
    if conformer_feature_mode == "mean":
        return mean.astype(np.float32), True
    std = selected.std(axis=0)
    if conformer_feature_mode == "mean_std":
        return np.concatenate([mean, std], axis=-1).astype(np.float32), True
    if conformer_feature_mode == "mean_std_max":
        max_values = selected.max(axis=0)
        return np.concatenate([mean, std, max_values], axis=-1).astype(np.float32), True
    raise ValueError(f"Unsupported conformer feature mode: {conformer_feature_mode}")


def pool_stats(features: np.ndarray, mask: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    if features.size == 0 or not np.any(mask):
        return np.zeros((dim * 3,), dtype=np.float32), np.zeros((dim,), dtype=np.float32)
    selected = features[mask].astype(np.float32)
    stats = np.concatenate(
        [
            selected.mean(axis=0),
            selected.std(axis=0),
            selected.max(axis=0),
        ],
        axis=0,
    )
    return stats.astype(np.float32), selected.mean(axis=0).astype(np.float32)


def component_cache_entry(
    store,
    metadata: pd.DataFrame,
    component_id: int,
    cache: dict[int, dict],
    conformer_feature_mode: str,
) -> dict:
    if component_id in cache:
        return cache[component_id]
    if component_id < 0 or component_id not in metadata.index:
        dim = conformer_feature_dim(store, conformer_feature_mode)
        entry = {
            "mol": None,
            "features": np.zeros((0, dim), dtype=np.float32),
            "descriptors": np.zeros((0, 4), dtype=np.float32),
            "has_3d": False,
            "canonical_smiles": "",
        }
        cache[component_id] = entry
        return entry
    meta = metadata.loc[component_id]
    canonical_smiles = str(meta["canonical_smiles"])
    mol = Chem.MolFromSmiles(canonical_smiles)
    features, has_3d = conformer_pooled_atom_features(store, component_id, conformer_feature_mode)
    descriptors = store.atom_descriptors[component_id][store.atom_mask[component_id]]
    entry = {
        "mol": mol,
        "features": features.astype(np.float32),
        "descriptors": descriptors.astype(np.float32),
        "has_3d": bool(has_3d),
        "canonical_smiles": canonical_smiles,
    }
    cache[component_id] = entry
    return entry


def add_pool(
    pools: dict[str, np.ndarray],
    means: dict[str, np.ndarray],
    scalars: list[float],
    name: str,
    features: np.ndarray,
    mask: np.ndarray,
    dim: int,
) -> None:
    stats, mean = pool_stats(features, mask, dim)
    pools[name] = stats
    means[name] = mean
    scalars.extend([float(mask.sum()), float(np.any(mask))])


def feature_row(
    store,
    metadata: pd.DataFrame,
    cache: dict[int, dict],
    row: pd.Series,
    map_rows: pd.DataFrame,
    conformer_feature_mode: str,
) -> np.ndarray:
    dim = conformer_feature_dim(store, conformer_feature_mode)
    pools = {name: np.zeros((dim * 3,), dtype=np.float32) for name in POOL_NAMES}
    means = {name: np.zeros((dim,), dtype=np.float32) for name in POOL_NAMES}
    scalars: list[float] = []

    for map_row in map_rows.itertuples(index=False):
        component_id = int(map_row.component_id)
        role = str(map_row.role)
        entry = component_cache_entry(store, metadata, component_id, cache, conformer_feature_mode)
        features = entry["features"]
        descriptors = entry["descriptors"]
        n_atoms = len(features)
        all_mask = np.ones((n_atoms,), dtype=bool) if entry["has_3d"] and n_atoms else np.zeros((n_atoms,), dtype=bool)
        scalars.extend([float(n_atoms), float(entry["has_3d"])])

        if role == "reactant":
            site_indices = site_indices_for_reactant(row, entry["canonical_smiles"])
            shells = neighbor_shells(entry["mol"], site_indices, max_radius=2)
            site_mask = np.zeros((n_atoms,), dtype=bool)
            shell1_mask = np.zeros((n_atoms,), dtype=bool)
            shell2_mask = np.zeros((n_atoms,), dtype=bool)
            shell02_mask = np.zeros((n_atoms,), dtype=bool)
            for idx, radius in shells.items():
                if idx >= n_atoms:
                    continue
                if radius == 0:
                    site_mask[idx] = True
                if radius == 1:
                    shell1_mask[idx] = True
                if radius == 2:
                    shell2_mask[idx] = True
                if radius <= 2:
                    shell02_mask[idx] = True
            add_pool(pools, means, scalars, "reactant_all", features, all_mask, dim)
            add_pool(pools, means, scalars, "reactant_site", features, site_mask, dim)
            add_pool(pools, means, scalars, "reactant_shell1", features, shell1_mask, dim)
            add_pool(pools, means, scalars, "reactant_shell2", features, shell2_mask, dim)
            add_pool(pools, means, scalars, "reactant_site_shell2", features, shell02_mask, dim)
        elif role == "solvent":
            add_pool(pools, means, scalars, "solvent_all", features, all_mask, dim)
        elif role == "ligand":
            donor_mask = descriptors[:, 3] > 0.5 if len(descriptors) else np.zeros((n_atoms,), dtype=bool)
            chiral_mask = descriptors[:, 2] > 0.5 if len(descriptors) else np.zeros((n_atoms,), dtype=bool)
            add_pool(pools, means, scalars, "ligand_all", features, all_mask, dim)
            add_pool(pools, means, scalars, "ligand_donor", features, donor_mask, dim)
            add_pool(pools, means, scalars, "ligand_chiral", features, chiral_mask, dim)
        elif role == "metal":
            add_pool(pools, means, scalars, "metal_all", features, all_mask, dim)

    blocks = [pools[name] for name in POOL_NAMES]
    for left_name, right_name in PAIR_NAMES:
        left = means[left_name]
        right = means[right_name]
        blocks.extend([left * right, np.abs(left - right)])
    blocks.append(np.asarray(scalars, dtype=np.float32))
    return np.concatenate(blocks, axis=0).astype(np.float32)


def build_feature_frame(
    *,
    store,
    mapping_task: str,
    part: str,
    target_frame: pd.DataFrame,
    target_column: str,
    max_rows: int,
    conformer_feature_mode: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapping = store.mapping[(store.mapping["task"].astype(str) == mapping_task) & (store.mapping["split_part"] == part)].copy()
    if max_rows > 0:
        keep_rows = set(range(min(max_rows, len(target_frame))))
        target_frame = target_frame.iloc[:max_rows].copy()
        mapping = mapping[mapping["split_row_pos"].isin(keep_rows)].copy()

    metadata = store.metadata.set_index("component_id")
    map_by_row = {
        int(row_pos): frame.sort_values("slot")
        for row_pos, frame in mapping.groupby("split_row_pos", sort=False)
    }
    cache: dict[int, dict] = {}
    features = []
    targets = []
    row_indices = []
    for row_pos, (_, row) in enumerate(target_frame.iterrows()):
        y_value = pd.to_numeric(row.get(target_column), errors="coerce")
        if not np.isfinite(y_value):
            continue
        map_rows = map_by_row.get(row_pos, pd.DataFrame())
        features.append(feature_row(store, metadata, cache, row, map_rows, conformer_feature_mode))
        targets.append(float(y_value))
        row_indices.append(int(row["_source_index"]))
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
    test_pred = prediction_values(model, model_name, bundle.task_type, test_x)

    model_tag = f"orbmol_v2_site_{model_name}"
    if args.conformer_feature_mode != "mean":
        model_tag = f"orbmol_v2_site_{args.conformer_feature_mode}_{model_name}"
    row = {
        "task": bundle.name,
        "split": args.split,
        "model": model_tag,
        "feature": f"site_pooled_{args.conformer_feature_mode}_atom_tokens_orbmol_v2",
        "conformer_feature_mode": args.conformer_feature_mode,
        "atom_token_dir": str(args.atom_token_dir),
        "mapping_task": str(args.mapping_task),
        "n_train": int(len(train_y)),
        "n_val": int(len(val_y)),
    }
    pred_df = pd.DataFrame({"row_index": test_row_index, "y_true": test_y})
    if bundle.task_type == "classification":
        val_pred = prediction_values(model, model_name, bundle.task_type, val_x)
        threshold = best_threshold_accuracy(val_y, val_pred)["best_threshold"]
        row.update(classification_metrics(test_y, test_pred, threshold=threshold))
        row.update(best_threshold_accuracy(test_y, test_pred))
        pred_df["y_prob"] = test_pred
        pred_df["y_pred"] = (test_pred >= threshold).astype(int)
    else:
        row.update(regression_metrics(test_y, test_pred))
        pred_df["y_pred"] = test_pred

    model_path = args.out_dir / f"{bundle.name}__{model_tag}.joblib"
    pred_path = args.out_dir / f"{bundle.name}__{model_tag}__predictions.csv"
    joblib.dump(model, model_path)
    pred_df.to_csv(pred_path, index=False)
    row["model_path"] = str(model_path)
    row["prediction_path"] = str(pred_path)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical heads on site-aware OrbMol-v2 atom-token pools.")
    parser.add_argument("--task", default="aho_major")
    parser.add_argument("--split", default="paper_exact_test", choices=["paper_exact_test", "paper_random42", "example"])
    parser.add_argument(
        "--atom-token-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_atom_tokens_aho_major_ligand_metal_h100",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_site_classical_aho_major_ligand_metal_h100",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "mlp", "xgb", "logreg", "knn", "rf", "extra_trees", "hgb"],
        choices=["ridge", "mlp", "xgb", "logreg", "knn", "rf", "extra_trees", "hgb"],
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument(
        "--conformer-feature-mode",
        default="mean",
        choices=["mean", "mean_std", "mean_std_max"],
        help="How to combine per-atom OrbMol-v2 node features across conformers before site pooling.",
    )
    parser.add_argument(
        "--mapping-task",
        default="aho_major",
        help=(
            "Task name stored in the atom-token reaction_component_map. Use aho_major to reuse "
            "the AHO major token cache for the row-aligned aho_ddg regression task."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    store = load_atom_token_store(args.atom_token_dir)
    bundle = load_task(args.task, split=args.split)
    frames = split_frames(bundle)
    feature_sets = {
        part: build_feature_frame(
            store=store,
            mapping_task=args.mapping_task,
            part=part,
            target_frame=frames[part],
            target_column=bundle.target_column,
            max_rows=args.max_train_rows if part == "train" else args.max_eval_rows,
            conformer_feature_mode=args.conformer_feature_mode,
        )
        for part in ("train", "val", "test")
    }
    rows = []
    for model_name in args.models:
        print(f"[train] model=orbmol_v2_site_{model_name}", flush=True)
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
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
