#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import train_test_split

from full_retrain.data import ROOT, load_task
from full_retrain.metrics import regression_metrics
from full_retrain.train_orb3d_site_attention import (
    AtomTokenStore,
    TOKEN_TYPE_IDS,
    Orb3DSiteAttentionModel,
    TensorBundle,
    build_tensor_bundle,
    load_atom_token_store,
    make_loader,
    split_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train standalone OrbMol-v2 site-aware atom-token attention for regression."
    )
    parser.add_argument("--task", default="aho_ddg")
    parser.add_argument("--split", default="paper_exact_test", choices=["paper_exact_test", "paper_random42", "example"])
    parser.add_argument(
        "--atom-token-dir",
        type=Path,
        default=ROOT
        / "reproduction_outputs"
        / "full_retrain"
        / "orb3d_atom_tokens_aho_major_ligand_metal_h100_c3",
    )
    parser.add_argument("--mapping-task", default="aho_major")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reproduction_outputs" / "full_retrain" / "orb3d_site_attention_regression_aho_ddg",
    )
    parser.add_argument("--model-name", default="orbmol_v2_site_attention_regression")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-tokens", type=int, default=176)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument(
        "--validation-mode",
        default="default",
        choices=["default", "group_column", "trainval_full"],
        help=(
            "Use the task's default validation split, rebuild train/validation by holding out input groups, "
            "or train the final model on the default train+val rows."
        ),
    )
    parser.add_argument(
        "--group-val-column",
        default="",
        help="Input column to hold out for group_column validation, e.g. Catalyst, Imine, or Thiol.",
    )
    parser.add_argument("--group-val-fraction", type=float, default=0.20)
    parser.add_argument(
        "--group-val-strategy",
        default="random",
        choices=["random", "farthest_centroid"],
        help="How to choose group-column validation groups from the train+val pool.",
    )
    parser.add_argument(
        "--group-val-role",
        default="",
        help="Component role used for farthest_centroid group validation; defaults from the group column.",
    )
    parser.add_argument(
        "--loss",
        default="huber",
        choices=["huber", "mae", "mse"],
        help="Training loss on standardized target values.",
    )
    parser.add_argument("--huber-delta", type=float, default=0.75)
    parser.add_argument(
        "--selection-metric",
        default="mae",
        choices=["mae", "rmse", "r2"],
        help="Validation metric used to choose the checkpoint.",
    )
    parser.add_argument(
        "--target-transform",
        default="none",
        choices=["none", "condition_residual"],
        help="Optionally train on target minus a train-only condition mean and add the base back at prediction time.",
    )
    parser.add_argument(
        "--condition-columns",
        default="Imine,Thiol",
        help="Comma-separated columns used for condition_residual target transform.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        default="validation",
        choices=["validation", "last"],
        help="Choose the best validation checkpoint, or save the last epoch for trainval_full final fits.",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def normalized_targets(train_y: torch.Tensor, *others: torch.Tensor) -> tuple[float, float, list[torch.Tensor]]:
    mean = float(train_y.mean().item())
    std = float(train_y.std(unbiased=False).item())
    if std < 1e-6:
        std = 1.0
    out = [(values - mean) / std for values in (train_y, *others)]
    return mean, std, out


def replace_targets(bundle: TensorBundle, y: torch.Tensor) -> TensorBundle:
    return TensorBundle(
        x=bundle.x,
        numeric=bundle.numeric,
        token_mask=bundle.token_mask,
        role_ids=bundle.role_ids,
        token_type_ids=bundle.token_type_ids,
        atomic_numbers=bundle.atomic_numbers,
        y=y.float(),
        row_index=bundle.row_index,
    )


def condition_key_for_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"condition columns are not present: {missing}")
    return pd.Series(list(zip(*(frame[column].astype(str) for column in columns))), index=frame.index)


def condition_base_predictions(
    train_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    target_column: str,
    columns: tuple[str, ...],
) -> np.ndarray:
    train_keys = condition_key_for_columns(train_frame, columns)
    train_y = pd.to_numeric(train_frame[target_column], errors="coerce").astype(float)
    means = train_y.groupby(train_keys).mean()
    fallback = float(train_y.mean())
    pred_keys = condition_key_for_columns(pred_frame, columns)
    return np.asarray([means.get(key, fallback) for key in pred_keys], dtype=np.float32)


def align_frame_values(frame: pd.DataFrame, column: str, row_index: np.ndarray) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    by_source = {
        int(source_index): float(value)
        for source_index, value in zip(frame["_source_index"].to_numpy(dtype=int), values.to_numpy(dtype=float), strict=True)
    }
    return np.asarray([by_source[int(idx)] for idx in row_index], dtype=np.float32)


def indexed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    tmp = frame.copy()
    tmp["_split_row_pos"] = np.arange(len(tmp), dtype=int)
    tmp["_source_index"] = tmp.index.to_numpy(dtype=int)
    return tmp


def group_validation_frames(bundle, column: str, fraction: float, seed: int) -> dict[str, pd.DataFrame]:
    if not column:
        raise ValueError("--group-val-column is required when --validation-mode=group_column")
    if bundle.train is None or bundle.val is None:
        raise ValueError("group_column validation requires train and val frames")
    pool = pd.concat([bundle.train, bundle.val], axis=0)
    if column not in pool.columns:
        raise ValueError(f"{column!r} is not a column in task {bundle.name}")
    groups = np.asarray(sorted(pool[column].astype(str).unique()), dtype=object)
    if len(groups) < 2:
        raise ValueError(f"Need at least two groups in {column}; found {len(groups)}")
    test_size = max(1, int(round(float(fraction) * len(groups))))
    test_size = min(test_size, len(groups) - 1)
    train_groups, val_groups = train_test_split(
        groups,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    val_group_set = set(str(value) for value in val_groups)
    train_frame = pool[~pool[column].astype(str).isin(val_group_set)].copy()
    val_frame = pool[pool[column].astype(str).isin(val_group_set)].copy()
    print(
        f"[group-val] column={column} train_groups={len(train_groups)} val_groups={len(val_groups)} "
        f"train_rows={len(train_frame)} val_rows={len(val_frame)}",
        flush=True,
    )
    return {
        "train": indexed_frame(train_frame),
        "val": indexed_frame(val_frame),
        "test": indexed_frame(bundle.test),
    }


def balanced_group_fold_validation_frames(
    bundle,
    column: str,
    fold_count: int,
    fold_index: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Hold out one deterministic, row-balanced fold of whole groups."""
    if not column:
        raise ValueError(
            "--group-val-column is required when "
            "--group-val-strategy=balanced_fold"
        )
    if bundle.train is None or bundle.val is None:
        raise ValueError(
            "balanced_fold validation requires train and val frames"
        )
    pool = pd.concat([bundle.train, bundle.val], axis=0)
    if column not in pool.columns:
        raise ValueError(f"{column!r} is not a column in task {bundle.name}")
    group_values = pool[column].astype(str)
    group_counts = group_values.value_counts().sort_index()
    if fold_count < 2 or fold_count > len(group_counts):
        raise ValueError(
            f"fold_count must be in [2, {len(group_counts)}], "
            f"got {fold_count}"
        )
    if fold_index < 0 or fold_index >= fold_count:
        raise ValueError(
            f"fold_index must be in [0, {fold_count}), got {fold_index}"
        )

    groups = group_counts.index.to_numpy(dtype=object)
    rng = np.random.default_rng(seed)
    shuffled = groups[rng.permutation(len(groups))]
    ordered = sorted(
        map(str, shuffled),
        key=lambda value: -int(group_counts.loc[value]),
    )
    fold_groups: list[list[str]] = [[] for _ in range(fold_count)]
    fold_rows = np.zeros(fold_count, dtype=np.int64)
    for value in ordered:
        selected_fold = min(
            range(fold_count),
            key=lambda index: (
                int(fold_rows[index]),
                len(fold_groups[index]),
                index,
            ),
        )
        fold_groups[selected_fold].append(value)
        fold_rows[selected_fold] += int(group_counts.loc[value])

    val_group_set = set(fold_groups[fold_index])
    train_frame = pool[~group_values.isin(val_group_set)].copy()
    val_frame = pool[group_values.isin(val_group_set)].copy()
    train_group_set = set(train_frame[column].astype(str))
    if not val_group_set or train_group_set & val_group_set:
        raise RuntimeError("Balanced group fold has empty/leaking groups")
    print(
        f"[group-val] strategy=balanced_fold column={column} "
        f"fold={fold_index + 1}/{fold_count} "
        f"train_groups={len(train_group_set)} "
        f"val_groups={len(val_group_set)} "
        f"train_rows={len(train_frame)} val_rows={len(val_frame)}",
        flush=True,
    )
    return {
        "train": indexed_frame(train_frame),
        "val": indexed_frame(val_frame),
        "test": indexed_frame(bundle.test),
    }


def role_for_group_column(column: str, explicit_role: str = "") -> str:
    if explicit_role:
        return explicit_role
    lower = column.lower()
    if lower in {"catalyst", "imine", "thiol"}:
        return lower
    return lower


def component_summary_feature(store: AtomTokenStore, component_id: int) -> np.ndarray:
    if component_id < 0 or component_id >= len(store.metadata):
        return np.zeros((store.node_features.shape[-1] * 3 + 8,), dtype=np.float32)
    atom_mask = store.atom_mask[component_id]
    conf_mask = store.conformer_mask[component_id]
    if not atom_mask.any() or not conf_mask.any():
        return np.zeros((store.node_features.shape[-1] * 3 + 8,), dtype=np.float32)
    selected = store.node_features[component_id][conf_mask][:, atom_mask, :].astype(np.float32).reshape(-1, store.node_features.shape[-1])
    atomic_numbers = store.atomic_numbers[component_id][atom_mask].astype(np.int64)
    scalars = np.asarray(
        [
            float(atom_mask.sum()),
            float(conf_mask.sum()),
            float((atomic_numbers == 6).mean()) if len(atomic_numbers) else 0.0,
            float((atomic_numbers != 6).mean()) if len(atomic_numbers) else 0.0,
            float(np.isin(atomic_numbers, [9, 17, 35, 53]).sum()),
            float((atomic_numbers == 14).sum()),
            float((atomic_numbers == 15).sum()),
            float((atomic_numbers == 16).sum()),
        ],
        dtype=np.float32,
    )
    return np.concatenate([selected.mean(axis=0), selected.std(axis=0), selected.max(axis=0), scalars], axis=0).astype(np.float32)


def farthest_centroid_group_validation_frames(
    bundle,
    store: AtomTokenStore,
    mapping_task: str,
    column: str,
    fraction: float,
    role: str,
) -> dict[str, pd.DataFrame]:
    if not column:
        raise ValueError("--group-val-column is required when --group-val-strategy=farthest_centroid")
    if bundle.train is None or bundle.val is None:
        raise ValueError("farthest_centroid validation requires train and val frames")
    pool = pd.concat([bundle.train, bundle.val], axis=0)
    if column not in pool.columns:
        raise ValueError(f"{column!r} is not a column in task {bundle.name}")
    indexed_pool = indexed_frame(pool)
    mapping = store.mapping[
        (store.mapping["task"].astype(str) == mapping_task)
        & (store.mapping["split_part"].astype(str).isin(["train", "val"]))
        & (store.mapping["role"].astype(str) == role)
    ].copy()
    component_by_source = {
        int(row.source_index): int(row.component_id)
        for row in mapping.itertuples(index=False)
        if int(row.component_id) >= 0
    }
    groups = []
    features = []
    for group_value, group_frame in indexed_pool.groupby(column, sort=True):
        source_index = int(group_frame["_source_index"].iloc[0])
        component_id = component_by_source.get(source_index)
        if component_id is None:
            raise ValueError(f"No component mapping for role={role} source_index={source_index}")
        groups.append(str(group_value))
        features.append(component_summary_feature(store, component_id))
    if len(groups) < 2:
        raise ValueError(f"Need at least two groups in {column}; found {len(groups)}")
    x = np.stack(features, axis=0).astype(np.float32)
    x_std = x.std(axis=0, keepdims=True)
    x = (x - x.mean(axis=0, keepdims=True)) / np.where(x_std < 1e-6, 1.0, x_std)
    distances = np.linalg.norm(x - x.mean(axis=0, keepdims=True), axis=1)
    test_size = max(1, int(round(float(fraction) * len(groups))))
    test_size = min(test_size, len(groups) - 1)
    order = np.argsort(-distances)
    val_group_set = {groups[int(idx)] for idx in order[:test_size]}
    train_frame = pool[~pool[column].astype(str).isin(val_group_set)].copy()
    val_frame = pool[pool[column].astype(str).isin(val_group_set)].copy()
    print(
        f"[group-val] strategy=farthest_centroid role={role} column={column} "
        f"train_groups={len(groups) - len(val_group_set)} val_groups={len(val_group_set)} "
        f"train_rows={len(train_frame)} val_rows={len(val_frame)} "
        f"val_distance_min={float(distances[order[:test_size]].min()):.6f}",
        flush=True,
    )
    return {
        "train": indexed_frame(train_frame),
        "val": indexed_frame(val_frame),
        "test": indexed_frame(bundle.test),
    }


def group_validation_frames_by_strategy(
    bundle,
    store: AtomTokenStore,
    mapping_task: str,
    column: str,
    fraction: float,
    seed: int,
    strategy: str,
    role: str,
    fold_count: int = 0,
    fold_index: int = 0,
) -> dict[str, pd.DataFrame]:
    if strategy == "random":
        return group_validation_frames(bundle, column=column, fraction=fraction, seed=seed)
    if strategy == "farthest_centroid":
        return farthest_centroid_group_validation_frames(
            bundle=bundle,
            store=store,
            mapping_task=mapping_task,
            column=column,
            fraction=fraction,
            role=role_for_group_column(column, role),
        )
    if strategy == "balanced_fold":
        return balanced_group_fold_validation_frames(
            bundle=bundle,
            column=column,
            fold_count=fold_count,
            fold_index=fold_index,
            seed=seed,
        )
    raise ValueError(f"Unsupported group validation strategy: {strategy}")


def store_with_source_index_mapping(
    store: AtomTokenStore,
    mapping_task: str,
    target_task: str,
    frame_by_part: dict[str, pd.DataFrame],
) -> AtomTokenStore:
    mapping = store.mapping[store.mapping["task"].astype(str) == mapping_task].copy()
    extra_rows = []
    source_parts = {
        "group_train": ("train", "val"),
        "group_val": ("train", "val"),
        "group_test": ("test",),
    }
    for custom_part, allowed_parts in source_parts.items():
        frame_key = custom_part.replace("group_", "")
        frame = frame_by_part[frame_key]
        base = mapping[mapping["split_part"].astype(str).isin(allowed_parts)].copy()
        base_by_source = {
            int(source_index): group.sort_values("slot")
            for source_index, group in base.groupby("source_index", sort=False)
        }
        for row_pos, source_index in enumerate(frame["_source_index"].to_numpy(dtype=int)):
            rows = base_by_source.get(int(source_index))
            if rows is None:
                raise ValueError(
                    f"No token mapping for task={mapping_task} source_index={source_index} "
                    f"while building {custom_part}"
                )
            tmp = rows.copy()
            tmp["task"] = target_task
            tmp["split_part"] = custom_part
            tmp["split_row_pos"] = int(row_pos)
            extra_rows.append(tmp)
    combined = pd.concat([store.mapping, *extra_rows], ignore_index=True)
    return AtomTokenStore(
        node_features=store.node_features,
        atom_mask=store.atom_mask,
        conformer_mask=store.conformer_mask,
        atomic_numbers=store.atomic_numbers,
        atom_descriptors=store.atom_descriptors,
        positions=store.positions,
        mapping=combined,
        metadata=store.metadata,
    )


def regression_loss(pred: torch.Tensor, y: torch.Tensor, loss_name: str, huber_delta: float) -> torch.Tensor:
    if loss_name == "mae":
        return F.l1_loss(pred, y)
    if loss_name == "mse":
        return F.mse_loss(pred, y)
    return F.huber_loss(pred, y, delta=huber_delta)


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: str,
    loss_name: str,
    huber_delta: float,
) -> float:
    model.train()
    losses = []
    for x, numeric, token_mask, role_ids, token_type_ids, atomic_numbers, y in loader:
        pred = model(
            x.to(device),
            numeric.to(device),
            token_mask.to(device),
            role_ids.to(device),
            token_type_ids.to(device),
            atomic_numbers.to(device),
        )
        loss = regression_loss(pred, y.to(device), loss_name, huber_delta)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict_raw(model: nn.Module, bundle: TensorBundle, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    preds = []
    for x, numeric, token_mask, role_ids, token_type_ids, atomic_numbers, _ in make_loader(bundle, batch_size, shuffle=False):
        raw = model(
            x.to(device),
            numeric.to(device),
            token_mask.to(device),
            role_ids.to(device),
            token_type_ids.to(device),
            atomic_numbers.to(device),
        )
        preds.append(raw.detach().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=float)


def selection_score(metrics: dict[str, float], metric_name: str) -> float:
    if metric_name == "r2":
        return -float(metrics["r2"])
    return float(metrics[metric_name])


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)

    store = load_atom_token_store(args.atom_token_dir)
    bundle = load_task(args.task, split=args.split)
    if bundle.task_type != "regression":
        raise ValueError(f"{args.task} is a {bundle.task_type} task; this script is regression-only.")
    mapping_task = args.mapping_task
    train_part = "train"
    val_part = "val"
    test_part = "test"
    if args.validation_mode == "group_column":
        frames = group_validation_frames_by_strategy(
            bundle,
            store=store,
            mapping_task=args.mapping_task,
            column=args.group_val_column,
            fraction=args.group_val_fraction,
            seed=args.seed,
            strategy=args.group_val_strategy,
            role=args.group_val_role,
        )
        strategy_suffix = "" if args.group_val_strategy == "random" else f"_{args.group_val_strategy}"
        mapping_task = f"{args.mapping_task}__groupval_{args.group_val_column.lower()}{strategy_suffix}"
        store = store_with_source_index_mapping(
            store=store,
            mapping_task=args.mapping_task,
            target_task=mapping_task,
            frame_by_part=frames,
        )
        train_part = "group_train"
        val_part = "group_val"
        test_part = "group_test"
    elif args.validation_mode == "trainval_full":
        if bundle.train is None or bundle.val is None:
            raise ValueError("trainval_full validation requires train and val frames")
        frames = {
            "train": indexed_frame(pd.concat([bundle.train, bundle.val], axis=0)),
            "val": indexed_frame(bundle.val),
            "test": indexed_frame(bundle.test),
        }
        mapping_task = f"{args.mapping_task}__trainval_full"
        store = store_with_source_index_mapping(
            store=store,
            mapping_task=args.mapping_task,
            target_task=mapping_task,
            frame_by_part=frames,
        )
        train_part = "group_train"
        val_part = "group_val"
        test_part = "group_test"
    else:
        frames = split_frames(bundle)
    model_target_column = bundle.target_column
    condition_columns = tuple(column.strip() for column in args.condition_columns.split(",") if column.strip())
    if args.target_transform == "condition_residual":
        if not condition_columns:
            raise ValueError("--condition-columns must not be empty when --target-transform=condition_residual")
        base_by_part = {
            part: condition_base_predictions(frames["train"], frame, bundle.target_column, condition_columns)
            for part, frame in frames.items()
        }
        transformed_frames = {}
        for part, frame in frames.items():
            tmp = frame.copy()
            target = pd.to_numeric(tmp[bundle.target_column], errors="coerce").to_numpy(dtype=np.float32)
            tmp["_condition_base"] = base_by_part[part]
            tmp["_model_target"] = target - base_by_part[part]
            transformed_frames[part] = tmp
        frames = transformed_frames
        model_target_column = "_model_target"
    else:
        base_by_part = {
            part: np.zeros((len(frame),), dtype=np.float32)
            for part, frame in frames.items()
        }
        frames = {
            part: frame.assign(_condition_base=base_by_part[part])
            for part, frame in frames.items()
        }
    train_bundle_raw = build_tensor_bundle(
        store=store,
        mapping_task=mapping_task,
        part=train_part,
        target_frame=frames["train"],
        target_column=model_target_column,
        max_rows=args.max_train_rows,
        max_tokens=args.max_tokens,
    )
    val_bundle_raw = build_tensor_bundle(
        store=store,
        mapping_task=mapping_task,
        part=val_part,
        target_frame=frames["val"],
        target_column=model_target_column,
        max_rows=args.max_eval_rows,
        max_tokens=args.max_tokens,
    )
    test_bundle_raw = build_tensor_bundle(
        store=store,
        mapping_task=mapping_task,
        part=test_part,
        target_frame=frames["test"],
        target_column=model_target_column,
        max_rows=args.max_eval_rows,
        max_tokens=args.max_tokens,
    )

    target_mean, target_std, scaled = normalized_targets(
        train_bundle_raw.y, val_bundle_raw.y, test_bundle_raw.y
    )
    train_bundle = replace_targets(train_bundle_raw, scaled[0])
    val_bundle = replace_targets(val_bundle_raw, scaled[1])
    test_bundle = replace_targets(test_bundle_raw, scaled[2])

    model = Orb3DSiteAttentionModel(
        orb_dim=int(store.node_features.shape[-1]),
        numeric_dim=8,
        hidden_dim=args.hidden_dim,
        num_roles=16,
        num_token_types=max(TOKEN_TYPE_IDS.values()) + 1,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_loader(train_bundle, args.batch_size, shuffle=True)

    best_state = None
    best_eval = float("inf")
    best_epoch = -1
    history = []
    val_true = align_frame_values(frames["val"], bundle.target_column, val_bundle_raw.row_index).astype(float)
    val_base_aligned = align_frame_values(frames["val"], "_condition_base", val_bundle_raw.row_index)
    for epoch in range(args.epochs):
        loss = train_epoch(model, loader, optimizer, args.device, args.loss, args.huber_delta)
        val_pred_model = predict_raw(model, val_bundle, args.batch_size, args.device) * target_std + target_mean
        if args.target_transform == "condition_residual":
            val_pred = val_pred_model + val_base_aligned
        else:
            val_pred = val_pred_model
        val_metrics = regression_metrics(val_true, val_pred)
        eval_score = selection_score(val_metrics, args.selection_metric)
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "eval_score": float(eval_score),
            **{f"validation_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if args.checkpoint_mode == "last" or eval_score < best_eval:
            best_eval = float(eval_score)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"[epoch] {epoch + 1}/{args.epochs} train_loss={loss:.6f} "
            f"val_r2={val_metrics['r2']:.6f} val_rmse={val_metrics['rmse']:.6f} "
            f"val_mae={val_metrics['mae']:.6f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    val_pred_model = predict_raw(model, val_bundle, args.batch_size, args.device) * target_std + target_mean
    test_pred_model = predict_raw(model, test_bundle, args.batch_size, args.device) * target_std + target_mean
    val_true = align_frame_values(frames["val"], bundle.target_column, val_bundle_raw.row_index).astype(float)
    test_true = align_frame_values(frames["test"], bundle.target_column, test_bundle_raw.row_index).astype(float)
    val_base_aligned = align_frame_values(frames["val"], "_condition_base", val_bundle_raw.row_index)
    test_base_aligned = align_frame_values(frames["test"], "_condition_base", test_bundle_raw.row_index)
    if args.target_transform == "condition_residual":
        val_pred = val_pred_model + val_base_aligned
        test_pred = test_pred_model + test_base_aligned
    else:
        val_pred = val_pred_model
        test_pred = test_pred_model
    val_metrics = regression_metrics(val_true, val_pred)
    metrics = regression_metrics(test_true, test_pred)

    pred_path = args.out_dir / f"{bundle.name}__{args.model_name}__predictions.csv"
    val_pred_path = args.out_dir / f"{bundle.name}__{args.model_name}__validation_predictions.csv"
    pd.DataFrame(
        {"row_index": test_bundle_raw.row_index, "y_true": test_true, "y_pred": test_pred}
    ).to_csv(pred_path, index=False)
    pd.DataFrame(
        {"row_index": val_bundle_raw.row_index, "y_true": val_true, "y_pred": val_pred}
    ).to_csv(val_pred_path, index=False)
    metrics.update(
        {
            "task": bundle.name,
            "split": args.split,
            "model": args.model_name,
            "feature": f"atom_site_tokens_orbmol_v2_attention_{args.target_transform}",
            "n_train": int(len(train_bundle.y)),
            "n_val": int(len(val_bundle.y)),
            "atom_token_dir": str(args.atom_token_dir),
            "mapping_task": args.mapping_task,
            "effective_mapping_task": mapping_task,
            "validation_mode": args.validation_mode,
            "checkpoint_mode": args.checkpoint_mode,
            "group_val_column": args.group_val_column,
            "group_val_fraction": float(args.group_val_fraction),
            "group_val_strategy": args.group_val_strategy,
            "group_val_role": args.group_val_role,
            "loss": args.loss,
            "huber_delta": float(args.huber_delta),
            "selection_metric": args.selection_metric,
            "target_transform": args.target_transform,
            "condition_columns": ",".join(condition_columns),
            "best_epoch": int(best_epoch),
            "target_mean": float(target_mean),
            "target_std": float(target_std),
            **{f"validation_{key}": value for key, value in val_metrics.items()},
            "prediction_path": str(pred_path),
            "validation_prediction_path": str(val_pred_path),
        }
    )
    pd.DataFrame([metrics]).to_csv(args.out_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(args.out_dir / "training_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "target_mean": target_mean,
            "target_std": target_std,
            "best_epoch": best_epoch,
        },
        args.out_dir / "model.pt",
    )
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
