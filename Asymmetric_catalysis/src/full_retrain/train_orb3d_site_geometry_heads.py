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
from full_retrain.metrics import best_threshold_accuracy, classification_metrics
from full_retrain.train_orb3d_classical_heads import build_model, prediction_values
from full_retrain.train_orb3d_site_attention import (
    load_atom_token_store,
    neighbor_shells,
    site_indices_for_reactant,
    split_frames,
)
from full_retrain.train_orb3d_site_classical_heads import (
    PAIR_NAMES,
    POOL_NAMES,
    conformer_feature_dim,
    feature_row as base_site_feature_row,
)


RDLogger.DisableLog("rdApp.*")


GEOMETRY_POOL_PAIRS = (
    ("reactant_site", "reactant_shell1"),
    ("reactant_site", "reactant_shell2"),
    ("reactant_site", "reactant_site_shell2"),
    ("reactant_site", "reactant_all"),
    ("ligand_donor", "ligand_chiral"),
    ("ligand_donor", "ligand_all"),
    ("ligand_chiral", "ligand_all"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standalone OrbMol-v2 site-pooled heads augmented with intrinsic 3D geometry statistics."
    )
    parser.add_argument("--split", default="paper_exact_test", choices=["paper_exact_test", "paper_random42", "example"])
    parser.add_argument(
        "--atom-token-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_atom_tokens_aho_major_ligand_metal_h100_c3",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_site_geometry_aho_major",
    )
    parser.add_argument(
        "--conformer-feature-mode",
        default="mean_std_max",
        choices=["mean", "mean_std", "mean_std_max"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "logreg", "extra_trees", "hgb", "mlp", "xgb"],
        choices=["ridge", "logreg", "knn", "rf", "extra_trees", "hgb", "mlp", "xgb"],
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def mean_positions(store, component_id: int) -> tuple[np.ndarray, bool]:
    atom_mask = store.atom_mask[component_id]
    conf_mask = store.conformer_mask[component_id]
    n_atoms = int(atom_mask.sum())
    if not conf_mask.any() or not atom_mask.any():
        return np.zeros((n_atoms, 3), dtype=np.float32), False
    weights = conf_mask.astype(np.float32)[:, None, None]
    denom = weights.sum(axis=0).clip(min=1.0)
    xyz = (store.positions[component_id].astype(np.float32) * weights).sum(axis=0) / denom
    return xyz[atom_mask].astype(np.float32), True


def safe_stats(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros((8,), dtype=np.float32)
    return np.asarray(
        [
            float(values.mean()),
            float(values.std()),
            float(values.min()),
            float(np.quantile(values, 0.25)),
            float(np.median(values)),
            float(np.quantile(values, 0.75)),
            float(values.max()),
            float(values.size),
        ],
        dtype=np.float32,
    )


def pairwise_distances(left: np.ndarray, right: np.ndarray | None = None) -> np.ndarray:
    left = np.asarray(left, dtype=np.float32)
    if right is None:
        if len(left) < 2:
            return np.zeros((0,), dtype=np.float32)
        delta = left[:, None, :] - left[None, :, :]
        dist = np.sqrt(np.maximum((delta**2).sum(axis=-1), 0.0))
        tri = np.triu_indices(len(left), k=1)
        return dist[tri].astype(np.float32)
    right = np.asarray(right, dtype=np.float32)
    if len(left) == 0 or len(right) == 0:
        return np.zeros((0,), dtype=np.float32)
    delta = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.maximum((delta**2).sum(axis=-1), 0.0)).reshape(-1).astype(np.float32)


def covariance_eigenvalues(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    if len(xyz) < 2:
        return np.zeros((3,), dtype=np.float32)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(xyz) - 1, 1)
    eig = np.linalg.eigvalsh(cov).astype(np.float32)
    eig = np.sort(np.maximum(eig, 0.0))[::-1]
    denom = float(eig.sum())
    if denom > 1e-8:
        eig = eig / denom
    return eig.astype(np.float32)


def pool_geometry_stats(xyz: np.ndarray, mask: np.ndarray, component_centroid: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if len(xyz) == 0 or not np.any(mask):
        return np.zeros((30,), dtype=np.float32)
    selected = xyz[mask].astype(np.float32)
    centroid = selected.mean(axis=0)
    local_radius = np.sqrt(np.maximum(((selected - centroid[None, :]) ** 2).sum(axis=1), 0.0))
    component_radius = np.sqrt(np.maximum(((selected - component_centroid[None, :]) ** 2).sum(axis=1), 0.0))
    internal_dist = pairwise_distances(selected)
    eig = covariance_eigenvalues(selected)
    centroid_radius = np.asarray([float(np.linalg.norm(centroid - component_centroid))], dtype=np.float32)
    count_has = np.asarray([float(mask.sum()), 1.0], dtype=np.float32)
    return np.concatenate(
        [
            count_has,
            centroid_radius,
            safe_stats(local_radius),
            safe_stats(component_radius),
            safe_stats(internal_dist),
            eig,
        ],
        axis=0,
    ).astype(np.float32)


def pool_pair_geometry(left_xyz: np.ndarray, right_xyz: np.ndarray) -> np.ndarray:
    if len(left_xyz) == 0 or len(right_xyz) == 0:
        return np.zeros((12,), dtype=np.float32)
    left_centroid = left_xyz.mean(axis=0)
    right_centroid = right_xyz.mean(axis=0)
    centroid_distance = np.asarray([float(np.linalg.norm(left_centroid - right_centroid))], dtype=np.float32)
    distances = pairwise_distances(left_xyz, right_xyz)
    return np.concatenate([centroid_distance, safe_stats(distances), covariance_eigenvalues(np.vstack([left_xyz, right_xyz]))])


def component_geometry_entry(store, metadata: pd.DataFrame, component_id: int, cache: dict[int, dict]) -> dict:
    if component_id in cache:
        return cache[component_id]
    if component_id < 0 or component_id not in metadata.index:
        entry = {
            "mol": None,
            "xyz": np.zeros((0, 3), dtype=np.float32),
            "descriptors": np.zeros((0, 4), dtype=np.float32),
            "atomic_numbers": np.zeros((0,), dtype=np.float32),
            "has_3d": False,
            "canonical_smiles": "",
        }
        cache[component_id] = entry
        return entry
    meta = metadata.loc[component_id]
    canonical_smiles = str(meta["canonical_smiles"])
    xyz, has_3d = mean_positions(store, component_id)
    atom_mask = store.atom_mask[component_id]
    entry = {
        "mol": Chem.MolFromSmiles(canonical_smiles),
        "xyz": xyz,
        "descriptors": store.atom_descriptors[component_id][atom_mask].astype(np.float32),
        "atomic_numbers": store.atomic_numbers[component_id][atom_mask].astype(np.float32),
        "has_3d": bool(has_3d),
        "canonical_smiles": canonical_smiles,
    }
    cache[component_id] = entry
    return entry


def geometry_feature_row(
    store,
    metadata: pd.DataFrame,
    cache: dict[int, dict],
    row: pd.Series,
    map_rows: pd.DataFrame,
) -> np.ndarray:
    pool_xyz: dict[str, np.ndarray] = {name: np.zeros((0, 3), dtype=np.float32) for name in POOL_NAMES}
    pool_geom: dict[str, np.ndarray] = {name: np.zeros((30,), dtype=np.float32) for name in POOL_NAMES}
    slot_scalars = np.zeros((4, 6), dtype=np.float32)

    for map_row in map_rows.itertuples(index=False):
        component_id = int(map_row.component_id)
        role = str(map_row.role)
        slot = int(map_row.slot)
        entry = component_geometry_entry(store, metadata, component_id, cache)
        xyz = entry["xyz"]
        n_atoms = len(xyz)
        all_mask = np.ones((n_atoms,), dtype=bool) if entry["has_3d"] and n_atoms else np.zeros((n_atoms,), dtype=bool)
        component_centroid = xyz.mean(axis=0) if len(xyz) else np.zeros((3,), dtype=np.float32)
        descriptors = entry["descriptors"]
        atom_nums = entry["atomic_numbers"]
        if 0 <= slot < slot_scalars.shape[0]:
            slot_scalars[slot] = np.asarray(
                [
                    float(n_atoms),
                    float(entry["has_3d"]),
                    float(atom_nums.mean()) if len(atom_nums) else 0.0,
                    float(atom_nums.std()) if len(atom_nums) else 0.0,
                    float((descriptors[:, 2] > 0.5).sum()) if len(descriptors) else 0.0,
                    float((descriptors[:, 3] > 0.5).sum()) if len(descriptors) else 0.0,
                ],
                dtype=np.float32,
            )

        def set_pool(name: str, mask: np.ndarray) -> None:
            pool_xyz[name] = xyz[mask].astype(np.float32) if len(xyz) and np.any(mask) else np.zeros((0, 3), dtype=np.float32)
            pool_geom[name] = pool_geometry_stats(xyz, mask, component_centroid)

        if role == "reactant":
            site_indices = site_indices_for_reactant(row, entry["canonical_smiles"])
            shells = neighbor_shells(entry["mol"], site_indices, max_radius=2)
            site_mask = np.zeros((n_atoms,), dtype=bool)
            shell1_mask = np.zeros((n_atoms,), dtype=bool)
            shell2_mask = np.zeros((n_atoms,), dtype=bool)
            shell02_mask = np.zeros((n_atoms,), dtype=bool)
            for atom_idx, radius in shells.items():
                if atom_idx >= n_atoms:
                    continue
                if radius == 0:
                    site_mask[atom_idx] = True
                if radius == 1:
                    shell1_mask[atom_idx] = True
                if radius == 2:
                    shell2_mask[atom_idx] = True
                if radius <= 2:
                    shell02_mask[atom_idx] = True
            set_pool("reactant_all", all_mask)
            set_pool("reactant_site", site_mask)
            set_pool("reactant_shell1", shell1_mask)
            set_pool("reactant_shell2", shell2_mask)
            set_pool("reactant_site_shell2", shell02_mask)
        elif role == "solvent":
            set_pool("solvent_all", all_mask)
        elif role == "ligand":
            donor_mask = descriptors[:, 3] > 0.5 if len(descriptors) else np.zeros((n_atoms,), dtype=bool)
            chiral_mask = descriptors[:, 2] > 0.5 if len(descriptors) else np.zeros((n_atoms,), dtype=bool)
            set_pool("ligand_all", all_mask)
            set_pool("ligand_donor", donor_mask)
            set_pool("ligand_chiral", chiral_mask)
        elif role == "metal":
            set_pool("metal_all", all_mask)

    blocks = [pool_geom[name] for name in POOL_NAMES]
    for left_name, right_name in GEOMETRY_POOL_PAIRS:
        blocks.append(pool_pair_geometry(pool_xyz[left_name], pool_xyz[right_name]))
    blocks.append(slot_scalars.reshape(-1).astype(np.float32))
    return np.concatenate(blocks, axis=0).astype(np.float32)


def build_feature_frame(
    *,
    store,
    part: str,
    target_frame: pd.DataFrame,
    target_column: str,
    max_rows: int,
    conformer_feature_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mapping = store.mapping[(store.mapping["task"].astype(str) == "aho_major") & (store.mapping["split_part"] == part)].copy()
    if max_rows > 0:
        keep_rows = set(range(min(max_rows, len(target_frame))))
        target_frame = target_frame.iloc[:max_rows].copy()
        mapping = mapping[mapping["split_row_pos"].isin(keep_rows)].copy()

    metadata = store.metadata.set_index("component_id")
    map_by_row = {
        int(row_pos): frame.sort_values("slot")
        for row_pos, frame in mapping.groupby("split_row_pos", sort=False)
    }
    base_cache: dict[int, dict] = {}
    geometry_cache: dict[int, dict] = {}
    rows = []
    targets = []
    row_indices = []
    for row_pos, (_, row) in enumerate(target_frame.iterrows()):
        y_value = pd.to_numeric(row.get(target_column), errors="coerce")
        if not np.isfinite(y_value):
            continue
        map_rows = map_by_row.get(row_pos, pd.DataFrame())
        base_features = base_site_feature_row(
            store,
            metadata,
            base_cache,
            row,
            map_rows,
            conformer_feature_mode,
        )
        dim = conformer_feature_dim(store, conformer_feature_mode)
        base_prefix_dim = len(POOL_NAMES) * dim * 3 + len(PAIR_NAMES) * dim * 2
        base_scalars = base_features[base_prefix_dim:]
        fixed_base_scalars = np.zeros((8,), dtype=np.float32)
        fixed_base_scalars[: min(len(base_scalars), len(fixed_base_scalars))] = base_scalars[
            : min(len(base_scalars), len(fixed_base_scalars))
        ]
        base_features = np.concatenate(
            [base_features[:base_prefix_dim], fixed_base_scalars],
            axis=0,
        ).astype(np.float32)
        geometry_features = geometry_feature_row(store, metadata, geometry_cache, row, map_rows)
        rows.append(np.concatenate([base_features, geometry_features], axis=0).astype(np.float32))
        targets.append(float(y_value))
        row_indices.append(int(row["_source_index"]))
    if not rows:
        raise ValueError(f"No valid rows for {part}")
    x = np.stack(rows, axis=0).astype(np.float32)
    y = np.asarray(targets, dtype=np.float32)
    idx = np.asarray(row_indices, dtype=int)
    print(
        f"[features] part={part} rows={len(y)} dim={x.shape[1]} "
        f"base_dim={conformer_feature_dim(store, conformer_feature_mode)}",
        flush=True,
    )
    return x, y, idx


def model_tag(model_name: str, conformer_feature_mode: str) -> str:
    if conformer_feature_mode == "mean":
        return f"orbmol_v2_site_geometry_{model_name}"
    return f"orbmol_v2_site_geometry_{conformer_feature_mode}_{model_name}"


def train_eval_one(model_name: str, args, bundle, feature_sets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> dict:
    train_x, train_y, _ = feature_sets["train"]
    val_x, val_y, _ = feature_sets["val"]
    test_x, test_y, test_row_index = feature_sets["test"]
    model = build_model(model_name, "classification", args.seed)
    model.fit(train_x, train_y)
    val_prob = prediction_values(model, model_name, "classification", val_x)
    threshold = best_threshold_accuracy(val_y, val_prob)["best_threshold"]
    test_prob = prediction_values(model, model_name, "classification", test_x)

    tag = model_tag(model_name, args.conformer_feature_mode)
    pred_path = args.out_dir / f"aho_major__{tag}__predictions.csv"
    pd.DataFrame(
        {
            "row_index": test_row_index,
            "y_true": test_y,
            "y_prob": test_prob,
            "y_pred": (test_prob >= threshold).astype(int),
        }
    ).to_csv(pred_path, index=False)
    row = {
        "task": bundle.name,
        "split": args.split,
        "model": tag,
        "feature": f"site_pooled_plus_intrinsic_geometry_{args.conformer_feature_mode}_atom_tokens_orbmol_v2",
        "conformer_feature_mode": args.conformer_feature_mode,
        "atom_token_dir": str(args.atom_token_dir),
        "n_train": int(len(train_y)),
        "n_val": int(len(val_y)),
    }
    row.update(classification_metrics(test_y, test_prob, threshold=threshold))
    row.update(best_threshold_accuracy(test_y, test_prob))
    row.update(
        {
            "model_path": str(args.out_dir / f"aho_major__{tag}.joblib"),
            "prediction_path": str(pred_path),
            "notes": (
                "Standalone OrbMol-v2 site-pooled atom-token features plus intrinsic "
                "component geometry statistics; no ChemAHNet features, logits, labels, "
                "residuals, or predictions."
            ),
        }
    )
    joblib.dump(model, args.out_dir / f"aho_major__{tag}.joblib")
    return row


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    store = load_atom_token_store(args.atom_token_dir)
    bundle = load_task("aho_major", split=args.split)
    frames = split_frames(bundle)
    feature_sets = {
        part: build_feature_frame(
            store=store,
            part=part,
            target_frame=frames[part],
            target_column=bundle.target_column,
            max_rows=args.max_train_rows if part == "train" else args.max_eval_rows,
            conformer_feature_mode=args.conformer_feature_mode,
        )
        for part in ["train", "val", "test"]
    }

    rows = []
    for model_name in args.models:
        print(f"[train] {model_name}", flush=True)
        rows.append(train_eval_one(model_name, args, bundle, feature_sets))
        print(pd.DataFrame([rows[-1]]).to_string(index=False), flush=True)

    metrics = pd.DataFrame(rows).sort_values(["accuracy", "auroc"], ascending=[False, False])
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
