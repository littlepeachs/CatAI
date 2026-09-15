#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from full_retrain.data import ROOT, load_task
from full_retrain.metrics import best_threshold_accuracy, classification_metrics, regression_metrics
from full_retrain.train_orb3d_reaction_attention import load_embedding_store, split_frames


def _pool_components(embeddings: np.ndarray, conformer_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = conformer_mask.astype(bool)
    weights = mask.astype(np.float32)[..., None]
    denom = weights.sum(axis=1).clip(min=1.0)
    pooled = (embeddings * weights).sum(axis=1) / denom
    has_3d = mask.any(axis=1)
    pooled[~has_3d] = 0.0
    return pooled.astype(np.float32), has_3d


def build_feature_frame(
    *,
    task_name: str,
    part: str,
    target_frame: pd.DataFrame,
    target_column: str,
    embeddings: np.ndarray,
    conformer_mask: np.ndarray,
    mapping: pd.DataFrame,
    max_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task_map = mapping[(mapping["task"].astype(str) == task_name) & (mapping["split_part"] == part)].copy()
    if max_rows > 0:
        keep_rows = set(range(min(max_rows, len(target_frame))))
        target_frame = target_frame.iloc[:max_rows].copy()
        task_map = task_map[task_map["split_row_pos"].isin(keep_rows)].copy()

    target_frame = target_frame.copy()
    n_rows = len(target_frame)
    if n_rows == 0:
        raise ValueError(f"No rows for {task_name}/{part}")
    n_slots = int(task_map["slot"].max() + 1) if len(task_map) else 1
    dim = embeddings.shape[-1]
    component_pool, component_has_3d = _pool_components(embeddings, conformer_mask)

    slot_vectors = np.zeros((n_rows, n_slots, dim), dtype=np.float32)
    slot_present = np.zeros((n_rows, n_slots), dtype=np.float32)
    slot_has_3d = np.zeros((n_rows, n_slots), dtype=np.float32)
    role_ids = np.zeros((n_rows, n_slots), dtype=np.float32)
    for row in task_map.itertuples(index=False):
        row_pos = int(row.split_row_pos)
        if row_pos >= n_rows:
            continue
        slot = int(row.slot)
        component_id = int(row.component_id)
        slot_present[row_pos, slot] = 1.0
        role_ids[row_pos, slot] = float(int(row.role_id))
        if 0 <= component_id < len(component_pool) and component_has_3d[component_id]:
            slot_vectors[row_pos, slot] = component_pool[component_id]
            slot_has_3d[row_pos, slot] = 1.0

    blocks: list[np.ndarray] = [
        slot_vectors.reshape(n_rows, n_slots * dim),
        slot_present,
        slot_has_3d,
        role_ids / 10.0,
    ]

    if n_slots > 1:
        pair_blocks = []
        for i in range(n_slots):
            for j in range(i + 1, n_slots):
                left = slot_vectors[:, i, :]
                right = slot_vectors[:, j, :]
                pair_blocks.extend([left * right, np.abs(left - right)])
        blocks.append(np.concatenate(pair_blocks, axis=1))

    present_denom = slot_has_3d.sum(axis=1, keepdims=True).clip(min=1.0)
    mean_vec = slot_vectors.sum(axis=1) / present_denom
    max_vec = slot_vectors.max(axis=1)
    blocks.extend([mean_vec, max_vec])

    x = np.concatenate(blocks, axis=1).astype(np.float32)
    y = pd.to_numeric(target_frame[target_column], errors="coerce").to_numpy(dtype=np.float32)
    row_index = target_frame["_source_index"].to_numpy(dtype=int)
    valid = np.isfinite(y)
    return x[valid], y[valid], row_index[valid]


def build_model(model_name: str, task_type: str, seed: int):
    if model_name == "ridge":
        if task_type == "classification":
            return make_pipeline(StandardScaler(), RidgeClassifier(alpha=1.0))
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))

    if model_name == "mlp":
        if task_type == "classification":
            return make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(256, 128),
                    activation="relu",
                    alpha=1e-3,
                    batch_size=64,
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    random_state=seed,
                ),
            )
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(256, 128),
                activation="relu",
                alpha=1e-3,
                batch_size=64,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                random_state=seed,
            ),
        )

    if model_name == "xgb":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise RuntimeError("Install xgboost to use Orb3D-XGB: pip install xgboost") from exc
        common = {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "max_depth": 4,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        if task_type == "classification":
            return XGBClassifier(**common, eval_metric="logloss")
        return XGBRegressor(**common, objective="reg:squarederror")

    if model_name == "logreg":
        if task_type != "classification":
            raise ValueError("logreg is classification-only; use ridge for regression.")
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=0.1, class_weight="balanced", random_state=seed),
        )

    if model_name == "knn":
        if task_type == "classification":
            return make_pipeline(
                StandardScaler(),
                KNeighborsClassifier(n_neighbors=15, weights="distance", metric="cosine", n_jobs=-1),
            )
        return make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=15, weights="distance", metric="cosine", n_jobs=-1),
        )

    if model_name == "rf":
        common = {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": seed,
        }
        if task_type == "classification":
            return RandomForestClassifier(**common, class_weight="balanced_subsample")
        return RandomForestRegressor(**common)

    if model_name == "extra_trees":
        common = {
            "n_estimators": 500,
            "max_features": "sqrt",
            "min_samples_leaf": 1,
            "n_jobs": -1,
            "random_state": seed,
        }
        if task_type == "classification":
            return ExtraTreesClassifier(**common, class_weight="balanced")
        return ExtraTreesRegressor(**common)

    if model_name == "hgb":
        if task_type == "classification":
            return make_pipeline(
                StandardScaler(),
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.04,
                    l2_regularization=1e-3,
                    random_state=seed,
                ),
            )
        return make_pipeline(
            StandardScaler(),
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.04,
                l2_regularization=1e-3,
                random_state=seed,
            ),
        )

    raise ValueError(f"Unknown model: {model_name}")


def prediction_values(model, model_name: str, task_type: str, x: np.ndarray) -> np.ndarray:
    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
        decision = np.asarray(model.decision_function(x), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    return np.asarray(model.predict(x), dtype=float)


def train_eval_one(model_name: str, args, bundle, embeddings, conformer_mask, mapping) -> dict:
    if bundle.cv_folds:
        raise NotImplementedError("Orb3D classical heads currently require train/val/test embedding maps; CV support should build features per fold.")

    frames = split_frames(bundle)
    train_x, train_y, _ = build_feature_frame(
        task_name=bundle.name,
        part="train",
        target_frame=frames["train"],
        target_column=bundle.target_column,
        embeddings=embeddings,
        conformer_mask=conformer_mask,
        mapping=mapping,
        max_rows=args.max_train_rows,
    )
    val_x = val_y = None
    if "val" in frames:
        val_x, val_y, _ = build_feature_frame(
            task_name=bundle.name,
            part="val",
            target_frame=frames["val"],
            target_column=bundle.target_column,
            embeddings=embeddings,
            conformer_mask=conformer_mask,
            mapping=mapping,
            max_rows=args.max_eval_rows,
        )
    test_x, test_y, test_row_index = build_feature_frame(
        task_name=bundle.name,
        part="test",
        target_frame=frames["test"],
        target_column=bundle.target_column,
        embeddings=embeddings,
        conformer_mask=conformer_mask,
        mapping=mapping,
        max_rows=args.max_eval_rows,
    )

    model = build_model(model_name, bundle.task_type, args.seed)
    model.fit(train_x, train_y)
    test_pred = prediction_values(model, model_name, bundle.task_type, test_x)

    row = {
        "task": bundle.name,
        "split": args.split,
        "model": f"orbmol_v2_{model_name}",
        "feature": "slot_pair_pooled_orbmol_v2",
        "embedding_dir": str(args.embedding_dir),
        "n_train": int(len(train_y)),
        "n_val": int(0 if val_y is None else len(val_y)),
    }
    pred_df = pd.DataFrame({"row_index": test_row_index, "y_true": test_y})
    if bundle.task_type == "classification":
        threshold = 0.5
        if val_x is not None and val_y is not None and len(val_y):
            val_pred = prediction_values(model, model_name, bundle.task_type, val_x)
            threshold = best_threshold_accuracy(val_y, val_pred)["best_threshold"]
        row.update(classification_metrics(test_y, test_pred, threshold=threshold))
        row.update(best_threshold_accuracy(test_y, test_pred))
        pred_df["y_prob"] = test_pred
        pred_df["y_pred"] = (test_pred >= threshold).astype(int)
    else:
        row.update(regression_metrics(test_y, test_pred))
        pred_df["y_pred"] = test_pred

    model_path = args.out_dir / f"{bundle.name}__orbmol_v2_{model_name}.joblib"
    pred_path = args.out_dir / f"{bundle.name}__orbmol_v2_{model_name}__predictions.csv"
    joblib.dump(model, model_path)
    pred_df.to_csv(pred_path, index=False)
    row["model_path"] = str(model_path)
    row["prediction_path"] = str(pred_path)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical heads on frozen OrbMol-v2 component embeddings.")
    parser.add_argument("--task", default="aho_ddg")
    parser.add_argument("--split", default="paper_exact_test", choices=["paper_exact_test", "paper_random42", "example"])
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_embeddings",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_classical",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ridge", "mlp", "xgb"],
        choices=["ridge", "mlp", "xgb", "logreg", "knn", "rf", "extra_trees", "hgb"],
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    embeddings, conformer_mask, mapping = load_embedding_store(args.embedding_dir)
    bundle = load_task(args.task, split=args.split)
    rows = []
    for model_name in args.models:
        print(f"[train] task={bundle.name} model=orbmol_v2_{model_name}", flush=True)
        rows.append(train_eval_one(model_name, args, bundle, embeddings, conformer_mask, mapping))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.out_dir / "metrics.csv", index=False)
    config = vars(args).copy()
    config["embedding_dir"] = str(config["embedding_dir"])
    config["out_dir"] = str(config["out_dir"])
    with open(args.out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(metrics.to_string(index=False))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
