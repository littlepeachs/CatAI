#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from full_retrain.data import ROOT, load_task
from full_retrain.metrics import best_threshold_accuracy, classification_metrics
from full_retrain.orb3d_attention_model import Orb3DReactionAttentionModel


TASK_IDS = {
    "aho_major": 0,
    "aho_ddg": 1,
    "rh_binol": 2,
    "cpa_main": 3,
    "cpa_imine": 4,
    "cpa_thiol": 5,
    "cpa_catalyst": 6,
    "cpa_transformation": 7,
    "oca": 8,
    "pca": 9,
    "oe": 10,
}


@dataclass
class TensorBundle:
    x: torch.Tensor
    conformer_mask: torch.Tensor
    component_mask: torch.Tensor
    role_ids: torch.Tensor
    task_ids: torch.Tensor
    y: torch.Tensor
    row_index: np.ndarray


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y_true)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def load_embedding_store(embedding_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data = np.load(embedding_dir / "component_conformer_embeddings.npz")
    embeddings = data["embeddings"].astype(np.float32)
    conformer_mask = data["conformer_mask"].astype(bool)
    mapping = pd.read_csv(embedding_dir / "reaction_component_map.csv")
    return embeddings, conformer_mask, mapping


def split_frames(bundle) -> dict[str, pd.DataFrame]:
    frames = {}
    for part, frame in [("train", bundle.train), ("val", bundle.val), ("test", bundle.test)]:
        if frame is not None:
            tmp = frame.copy()
            tmp["_split_row_pos"] = np.arange(len(tmp), dtype=int)
            tmp["_source_index"] = tmp.index.to_numpy()
            frames[part] = tmp
    if not frames and bundle.train is not None:
        tmp = bundle.train.copy()
        tmp["_split_row_pos"] = np.arange(len(tmp), dtype=int)
        tmp["_source_index"] = tmp.index.to_numpy()
        frames["train"] = tmp
    return frames


def build_tensor_bundle(
    *,
    task: str,
    part: str,
    target_frame: pd.DataFrame,
    target_column: str,
    task_type: str,
    embeddings: np.ndarray,
    embedding_mask: np.ndarray,
    mapping: pd.DataFrame,
    max_rows: int,
) -> TensorBundle:
    task_map = mapping[(mapping["task"].astype(str) == load_task(task).name) & (mapping["split_part"] == part)].copy()
    if max_rows > 0:
        keep_rows = set(range(min(max_rows, len(target_frame))))
        target_frame = target_frame.iloc[:max_rows].copy()
        task_map = task_map[task_map["split_row_pos"].isin(keep_rows)].copy()

    n_rows = len(target_frame)
    n_slots = int(task_map["slot"].max() + 1) if len(task_map) else 1
    n_conformers = embeddings.shape[1]
    dim = embeddings.shape[2]
    x = np.zeros((n_rows, n_slots, n_conformers, dim), dtype=np.float32)
    conformer_mask = np.zeros((n_rows, n_slots, n_conformers), dtype=bool)
    component_mask = np.zeros((n_rows, n_slots), dtype=bool)
    role_ids = np.zeros((n_rows, n_slots), dtype=np.int64)

    for row in task_map.itertuples(index=False):
        row_pos = int(row.split_row_pos)
        slot = int(row.slot)
        if row_pos >= n_rows:
            continue
        component_mask[row_pos, slot] = True
        role_ids[row_pos, slot] = int(row.role_id)
        component_id = int(row.component_id)
        if component_id >= 0 and component_id < embeddings.shape[0]:
            x[row_pos, slot] = embeddings[component_id]
            conformer_mask[row_pos, slot] = embedding_mask[component_id]

    y = pd.to_numeric(target_frame[target_column], errors="coerce").to_numpy(dtype=np.float32)
    valid = np.isfinite(y)
    x = x[valid]
    conformer_mask = conformer_mask[valid]
    component_mask = component_mask[valid]
    role_ids = role_ids[valid]
    y = y[valid]
    row_index = target_frame["_source_index"].to_numpy(dtype=int)[valid]
    if task_type == "classification":
        y = y.astype(np.float32)
    return TensorBundle(
        x=torch.from_numpy(x),
        conformer_mask=torch.from_numpy(conformer_mask),
        component_mask=torch.from_numpy(component_mask),
        role_ids=torch.from_numpy(role_ids),
        task_ids=torch.full((len(y),), TASK_IDS[task], dtype=torch.long),
        y=torch.from_numpy(y),
        row_index=row_index,
    )


def make_loader(bundle: TensorBundle, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        bundle.x,
        bundle.conformer_mask,
        bundle.component_mask,
        bundle.role_ids,
        bundle.task_ids,
        bundle.y,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, device: str, task_type: str, y_mean: float, y_std: float) -> float:
    model.train()
    losses = []
    for x, conf_mask, comp_mask, role_ids, task_ids, y in loader:
        x = x.to(device)
        conf_mask = conf_mask.to(device)
        comp_mask = comp_mask.to(device)
        role_ids = role_ids.to(device)
        task_ids = task_ids.to(device)
        y = y.to(device)
        pred = model(x, conf_mask, comp_mask, role_ids, task_ids)
        if task_type == "classification":
            loss = F.binary_cross_entropy_with_logits(pred, y)
        else:
            target = (y - y_mean) / y_std
            loss = F.huber_loss(pred, target, delta=1.0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(model: nn.Module, bundle: TensorBundle, batch_size: int, device: str, task_type: str, y_mean: float, y_std: float) -> np.ndarray:
    model.eval()
    preds = []
    for x, conf_mask, comp_mask, role_ids, task_ids, _ in make_loader(bundle, batch_size, shuffle=False):
        raw = model(
            x.to(device),
            conf_mask.to(device),
            comp_mask.to(device),
            role_ids.to(device),
            task_ids.to(device),
        )
        if task_type == "classification":
            pred = torch.sigmoid(raw)
        else:
            pred = raw * y_std + y_mean
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=float)


def metric_row(task_type: str, y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    if task_type == "classification":
        row = classification_metrics(y_true, y_pred, threshold=threshold)
        row.update(best_threshold_accuracy(y_true, y_pred))
        return row
    return regression_metrics(y_true, y_pred)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Orb3D reaction attention model from cached component embeddings.")
    parser.add_argument("--task", default="aho_ddg", choices=list(TASK_IDS))
    parser.add_argument("--split", default="paper_exact_test", choices=["paper_exact_test", "paper_random42", "example"])
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_embeddings",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_attention",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    embeddings, embedding_mask, mapping = load_embedding_store(args.embedding_dir)
    bundle = load_task(args.task, split=args.split)
    frames = split_frames(bundle)
    if "test" not in frames:
        raise NotImplementedError("This training entry currently supports holdout tasks; CV support will use the same tensors per fold.")
    train_bundle = build_tensor_bundle(
        task=args.task,
        part="train",
        target_frame=frames["train"],
        target_column=bundle.target_column,
        task_type=bundle.task_type,
        embeddings=embeddings,
        embedding_mask=embedding_mask,
        mapping=mapping,
        max_rows=args.max_train_rows,
    )
    eval_part = "val" if "val" in frames else "test"
    eval_bundle = build_tensor_bundle(
        task=args.task,
        part=eval_part,
        target_frame=frames[eval_part],
        target_column=bundle.target_column,
        task_type=bundle.task_type,
        embeddings=embeddings,
        embedding_mask=embedding_mask,
        mapping=mapping,
        max_rows=args.max_eval_rows,
    )
    test_bundle = build_tensor_bundle(
        task=args.task,
        part="test",
        target_frame=frames["test"],
        target_column=bundle.target_column,
        task_type=bundle.task_type,
        embeddings=embeddings,
        embedding_mask=embedding_mask,
        mapping=mapping,
        max_rows=args.max_eval_rows,
    )

    y_train = train_bundle.y.numpy().astype(float)
    y_mean = float(y_train.mean()) if bundle.task_type == "regression" else 0.0
    y_std = float(y_train.std()) if bundle.task_type == "regression" else 1.0
    y_std = y_std if y_std > 1e-8 else 1.0

    model = Orb3DReactionAttentionModel(
        input_dim=int(embeddings.shape[2]),
        hidden_dim=args.hidden_dim,
        num_roles=16,
        num_tasks=16,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        output_dim=1,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(train_bundle, args.batch_size, shuffle=True)

    best_state = None
    best_eval = float("inf")
    history = []
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, loader, optimizer, args.device, bundle.task_type, y_mean, y_std)
        eval_pred = predict(model, eval_bundle, args.batch_size, args.device, bundle.task_type, y_mean, y_std)
        eval_true = eval_bundle.y.numpy().astype(float)
        if bundle.task_type == "classification":
            eval_best = best_threshold_accuracy(eval_true, eval_pred)
            eval_score = 1.0 - eval_best["best_accuracy"]
        else:
            eval_score = mean_absolute_error(eval_true, eval_pred)
            eval_best = {"best_accuracy": float("nan"), "best_threshold": float("nan")}
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_score": float(eval_score),
                "eval_best_accuracy": eval_best["best_accuracy"],
                "eval_best_threshold": eval_best["best_threshold"],
            }
        )
        if eval_score < best_eval:
            best_eval = float(eval_score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[epoch] {epoch + 1}/{args.epochs} train_loss={train_loss:.6f} eval_score={eval_score:.6f}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    threshold = 0.5
    if bundle.task_type == "classification":
        eval_pred = predict(model, eval_bundle, args.batch_size, args.device, bundle.task_type, y_mean, y_std)
        eval_true = eval_bundle.y.numpy().astype(float)
        threshold = best_threshold_accuracy(eval_true, eval_pred)["best_threshold"]
    test_pred = predict(model, test_bundle, args.batch_size, args.device, bundle.task_type, y_mean, y_std)
    test_true = test_bundle.y.numpy().astype(float)
    metrics = metric_row(bundle.task_type, test_true, test_pred, threshold=threshold)

    pred_df = pd.DataFrame({"row_index": test_bundle.row_index, "y_true": test_true})
    if bundle.task_type == "classification":
        pred_df["y_prob"] = test_pred
        pred_df["y_pred"] = (test_pred >= threshold).astype(int)
    else:
        pred_df["y_pred"] = test_pred
    pred_path = args.out_dir / f"{bundle.name}__orb3d_attention__predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    metrics.update(
        {
            "task": bundle.name,
            "split": args.split,
            "model": "orb3d_attention",
            "prediction_path": str(pred_path),
            "embedding_dir": str(args.embedding_dir),
        }
    )
    pd.DataFrame([metrics]).to_csv(args.out_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "args": vars(args), "y_mean": y_mean, "y_std": y_std}, args.out_dir / "model.pt")
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    with open(args.out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
