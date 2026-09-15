#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from full_retrain.data import ROOT, load_task
from full_retrain.metrics import (
    best_threshold_accuracy,
    classification_metrics,
    regression_metrics,
)
from full_retrain.train_orb3d_atom_relation_attention import (
    AtomRelationTensorBundle,
    build_tensor_bundle,
)
from full_retrain.train_orb3d_site_attention import (
    TOKEN_TYPE_IDS,
    load_atom_token_store,
    split_frames,
)
from full_retrain.train_orb3d_site_attention_regression import (
    group_validation_frames_by_strategy,
    store_with_source_index_mapping,
)
from full_retrain.unified_orra_model import UnifiedORRAModel


TASKS = (
    "aho_major",
    "aho_ddg",
    "cpa_main",
    "cpa_transformation",
    "cpa_imine",
    "cpa_thiol",
    "cpa_catalyst",
)

TOKEN_DIR_NAMES = {
    "aho_major": "orb3d_atom_tokens_aho_major_ligand_metal_h100_c3",
    "aho_ddg": "orb3d_atom_tokens_aho_major_ligand_metal_h100_c3",
    "cpa_main": "orb3d_atom_tokens_cpa_main_h100_c3",
    "cpa_transformation": "orb3d_atom_tokens_cpa_transformation_h100_c3",
    "cpa_imine": "orb3d_atom_tokens_cpa_imine_h100_c3",
    "cpa_thiol": "orb3d_atom_tokens_cpa_thiol_h100_c3",
    "cpa_catalyst": "orb3d_atom_tokens_cpa_catalyst_h100_c3",
}


@dataclass
class UnifiedTensorBundle:
    x: torch.Tensor
    numeric: torch.Tensor
    token_mask: torch.Tensor
    role_ids: torch.Tensor
    token_type_ids: torch.Tensor
    atomic_numbers: torch.Tensor
    component_ids: torch.Tensor
    density: torch.Tensor
    sample_prior: torch.Tensor
    y: torch.Tensor
    row_index: np.ndarray


@dataclass
class RoleDensityState:
    mean: np.ndarray
    std: np.ndarray
    prototypes: np.ndarray
    support_radius: float


class ComponentDensityProfile:
    """Train-only component support used by the shared confidence gate."""

    def __init__(
        self,
        num_roles: int = 16,
        quantile: float = 0.95,
        unseen_floor: float = 0.0,
        ratio_cap: float = 10.0,
    ) -> None:
        self.num_roles = int(num_roles)
        self.quantile = float(quantile)
        self.unseen_floor = float(unseen_floor)
        self.ratio_cap = float(ratio_cap)
        if self.unseen_floor < 0.0:
            raise ValueError("unseen_floor must be non-negative")
        if not np.isfinite(self.ratio_cap) or self.ratio_cap <= 0.0:
            raise ValueError("ratio_cap must be finite and positive")
        self.roles: dict[int, RoleDensityState] = {}

    @staticmethod
    def _global_features_by_role(
        bundle: AtomRelationTensorBundle,
        role: int,
    ) -> np.ndarray:
        is_global = (
            bundle.token_mask.numpy().astype(bool)
            & (bundle.token_type_ids.numpy() == TOKEN_TYPE_IDS["global"])
            & (bundle.role_ids.numpy() == role)
        )
        if not is_global.any():
            return np.zeros((0, bundle.x.shape[-1]), dtype=np.float32)
        return bundle.x.numpy()[is_global].astype(np.float32)

    def fit(self, train_bundle: AtomRelationTensorBundle) -> ComponentDensityProfile:
        for role in range(self.num_roles):
            raw = self._global_features_by_role(train_bundle, role)
            if len(raw) == 0:
                continue
            prototypes_raw = np.unique(raw, axis=0).astype(np.float32)
            mean = prototypes_raw.mean(axis=0).astype(np.float32)
            std = prototypes_raw.std(axis=0).astype(np.float32)
            std = np.where(std < 1e-5, 1.0, std).astype(np.float32)
            prototypes = ((prototypes_raw - mean) / std).astype(np.float32)

            if len(prototypes) == 1:
                support_radius = 1.0
            else:
                neighbors = NearestNeighbors(n_neighbors=2, metric="euclidean", algorithm="brute")
                neighbors.fit(prototypes)
                distances, _ = neighbors.kneighbors(prototypes)
                nearest_other = distances[:, 1]
                support_radius = float(np.quantile(nearest_other, self.quantile))
                if not np.isfinite(support_radius) or support_radius < 1e-5:
                    support_radius = float(np.max(nearest_other))
                if not np.isfinite(support_radius) or support_radius < 1e-5:
                    support_radius = 1.0
            self.roles[role] = RoleDensityState(
                mean=mean,
                std=std,
                prototypes=prototypes,
                support_radius=support_radius,
            )
            print(
                f"[density-fit] role={role} unique_components={len(prototypes)} "
                f"q{int(round(self.quantile * 100))}_radius={support_radius:.6f}",
                flush=True,
            )
        return self

    @staticmethod
    def _nearest_distances(
        query: np.ndarray,
        state: RoleDensityState,
        *,
        training: bool,
    ) -> np.ndarray:
        normalized = ((query.astype(np.float32) - state.mean) / state.std).astype(np.float32)
        neighbor_count = 2 if training and len(state.prototypes) > 1 else 1
        neighbors = NearestNeighbors(
            n_neighbors=neighbor_count,
            metric="euclidean",
            algorithm="brute",
        )
        neighbors.fit(state.prototypes)
        distances, _ = neighbors.kneighbors(normalized)
        if training and neighbor_count == 2:
            exact = distances[:, 0] < 1e-5
            selected = np.where(exact, distances[:, 1], distances[:, 0])
        else:
            selected = distances[:, 0]
        return selected.astype(np.float32)

    def transform(
        self,
        bundle: AtomRelationTensorBundle,
        *,
        training: bool,
    ) -> torch.Tensor:
        role_ids = bundle.role_ids.numpy()
        token_types = bundle.token_type_ids.numpy()
        token_mask = bundle.token_mask.numpy().astype(bool)
        x = bundle.x.numpy()
        n_rows = len(bundle.y)
        ratios = np.zeros((n_rows, self.num_roles), dtype=np.float32)
        present = np.zeros((n_rows, self.num_roles), dtype=bool)

        for role, state in self.roles.items():
            role_global = (
                token_mask
                & (token_types == TOKEN_TYPE_IDS["global"])
                & (role_ids == role)
            )
            row_ids, token_ids = np.nonzero(role_global)
            if len(row_ids) == 0:
                continue
            distances = self._nearest_distances(x[row_ids, token_ids], state, training=training)
            normalized = np.clip(
                distances / state.support_radius,
                0.0,
                self.ratio_cap,
            )
            if self.unseen_floor > 0.0:
                unseen = distances >= 1e-5
                normalized = np.where(
                    unseen,
                    np.maximum(normalized, self.unseen_floor),
                    normalized,
                )
            np.maximum.at(ratios[:, role], row_ids, normalized)
            present[row_ids, role] = True

        present_count = present.sum(axis=1).clip(min=1)
        density_mean = (ratios * present).sum(axis=1) / present_count
        density_max = ratios.max(axis=1)
        outside_fraction = ((ratios > 1.0) & present).sum(axis=1) / present_count
        features = np.concatenate(
            [
                ratios,
                density_mean[:, None].astype(np.float32),
                density_max[:, None].astype(np.float32),
                outside_fraction[:, None].astype(np.float32),
            ],
            axis=1,
        )
        return torch.from_numpy(features.astype(np.float32))


def parse_regression_head_hidden_dims(
    value: str,
) -> tuple[int, ...] | None:
    stripped = value.strip()
    if not stripped:
        return None
    tokens = [token.strip() for token in stripped.split(",")]
    if any(not token for token in tokens):
        raise argparse.ArgumentTypeError(
            "Regression-head hidden dimensions must be comma-separated "
            "positive integers"
        )
    try:
        hidden_dims = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Regression-head hidden dimensions must be comma-separated "
            "positive integers"
        ) from exc
    if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
        raise argparse.ArgumentTypeError(
            "Regression-head hidden dimensions must be positive"
        )
    return hidden_dims


def parse_regression_head_dropout(value: str) -> float | None:
    try:
        dropout = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Regression-head dropout must be a number"
        ) from exc
    if not math.isfinite(dropout):
        raise argparse.ArgumentTypeError(
            "Regression-head dropout must be finite"
        )
    if dropout < 0.0:
        return None
    if dropout >= 1.0:
        raise argparse.ArgumentTypeError(
            "Regression-head dropout must be negative or in [0, 1)"
        )
    return dropout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the same single-model self-attention + role cross-attention + "
            "density-gated ORRA architecture on any AHO/CPA task."
        )
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument(
        "--split",
        default="paper_exact_test",
        choices=["paper_exact_test", "paper_random42", "example"],
    )
    parser.add_argument("--atom-token-dir", type=Path, default=None)
    parser.add_argument("--mapping-task", default="")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize from a shape-compatible Unified ORRA checkpoint. "
            "The task prior is never transferred."
        ),
    )
    parser.add_argument(
        "--init-mode",
        default="full",
        choices=["full", "features"],
        help=(
            "full also transfers the prediction head; features excludes "
            "prediction and confidence heads."
        ),
    )
    parser.add_argument(
        "--conformer-feature-mode",
        default="mean_std_max",
        choices=["mean", "mean_std", "mean_std_max"],
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-self-layers", type=int, default=2)
    parser.add_argument("--num-role-layers", type=int, default=2)
    parser.add_argument(
        "--cross-attention-view",
        default="roles_types",
        choices=["roles", "roles_types"],
    )
    parser.add_argument(
        "--grouped-statistics",
        action="store_true",
        help=(
            "Add the same per-group mean/std/max token summary to every role "
            "and token-type cross-attention query."
        ),
    )
    parser.add_argument(
        "--cross-view-interactions",
        action="store_true",
        help=(
            "Attention-pool all valid role-token by token-type-token pair "
            "interactions using one shared projection."
        ),
    )
    parser.add_argument(
        "--ddg-sign-magnitude",
        action="store_true",
        help=(
            "Enable the existing same-network sign/magnitude structured "
            "regression branch. Despite the historical option name, this "
            "default-off branch can be used by any regression task."
        ),
    )
    parser.add_argument(
        "--ddg-sign-magnitude-hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--ddg-sign-magnitude-residual-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--ddg-residual-calibrator",
        action="store_true",
        help=(
            "Enable the existing same-network post-gate residual calibrator; "
            "requires --ddg-sign-magnitude and is disabled by default."
        ),
    )
    parser.add_argument(
        "--ddg-residual-calibrator-nonlinear-hidden-dim",
        type=int,
        default=0,
        help=(
            "Optional nonlinear residual width after the linear calibrator. "
            "Zero exactly selects the locked J185 linear structure."
        ),
    )
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument(
        "--regression-head-hidden-dims",
        type=parse_regression_head_hidden_dims,
        default=None,
        help=(
            "Comma-separated hidden dimensions for an opt-in regression "
            "head. Omit or pass an empty string to preserve the legacy "
            "fusion->2H->H->1 head exactly."
        ),
    )
    parser.add_argument(
        "--regression-head-dropout",
        type=parse_regression_head_dropout,
        default=None,
        help=(
            "Dropout for the regression head. A negative value or omission "
            "follows --dropout."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lr-patience", type=int, default=10)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=5.0,
        help="Global gradient-norm cap; use 0 to disable clipping.",
    )
    parser.add_argument("--max-tokens", type=int, default=176)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help=(
            "Train and write validation predictions without constructing or "
            "evaluating the test split; intended for OOF hyperparameter selection."
        ),
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help=(
            "Load --init-checkpoint, apply the requested gate configuration, "
            "and evaluate it without an optimizer update."
        ),
    )
    parser.add_argument("--density-quantile", type=float, default=0.95)
    parser.add_argument(
        "--density-ratio-cap",
        type=float,
        default=10.0,
        help=(
            "Maximum normalized component distance exposed to the confidence "
            "gate. The historical default 10 preserves the original path."
        ),
    )
    parser.add_argument(
        "--unseen-density-floor",
        type=float,
        default=0.0,
        help=(
            "Minimum normalized density ratio assigned to a component whose "
            "global embedding is not an exact member of the train support."
        ),
    )
    parser.add_argument(
        "--density-train-mode",
        default="exact_seen",
        choices=["exact_seen", "leave_one_component_out"],
    )
    parser.add_argument("--ood-threshold", type=float, default=1.0)
    parser.add_argument("--ood-scale-init", type=float, default=0.2)
    parser.add_argument(
        "--ood-gate-mode",
        default="exponential",
        choices=["exponential", "logit"],
    )
    parser.add_argument(
        "--gate-residual-mode",
        default="convex",
        choices=["convex", "centered_ood"],
        help=(
            "convex shrinks every prediction toward its prior; centered_ood "
            "leaves supported samples raw and shifts only OOD samples while "
            "retaining confidence-scaled residuals around the train centre."
        ),
    )
    parser.add_argument(
        "--ood-shift-scale",
        type=float,
        default=1.0,
        help="Monotonic strength of the OOD-only centre shift.",
    )
    parser.add_argument(
        "--ood-base-shift",
        type=float,
        default=0.0,
        help=(
            "Supported-sample mixture in the same centred-residual gate: "
            "0 leaves supported samples raw; 1 applies the residual gate everywhere."
        ),
    )
    parser.add_argument(
        "--residual-global-center-mix",
        type=float,
        default=1.0,
        help=(
            "Residual centre interpolation: 1 uses the global train centre; "
            "0 uses the sample prior and exactly recovers ordinary convex shrinkage "
            "when --ood-base-shift=1."
        ),
    )
    parser.add_argument("--gate-regularization", type=float, default=0.005)
    parser.add_argument(
        "--freeze-gate",
        action="store_true",
        help="Keep the train-only density gate deterministic while fitting the attention readout.",
    )
    parser.add_argument(
        "--sample-prior-mode",
        default="global",
        choices=["global", "train_quantile", "condition_quantile"],
    )
    parser.add_argument(
        "--prior-condition-columns",
        default="",
        help="Comma-separated train-only condition columns for the per-row gate prior.",
    )
    parser.add_argument("--prior-quantile", type=float, default=0.0)
    parser.add_argument("--loss", default="huber", choices=["huber", "mae", "mse"])
    parser.add_argument("--huber-delta", type=float, default=0.75)
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help=(
            "Binary-label smoothing used only during classification training. "
            "The default 0 exactly preserves the original BCE objective."
        ),
    )
    parser.add_argument(
        "--positive-class-weight-scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier on the train-derived positive-class BCE weight. "
            "The default 1 exactly preserves the original class weighting."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        default="auto",
        choices=["auto", "accuracy", "auroc", "mae", "rmse", "r2"],
    )
    parser.add_argument(
        "--selection-output",
        default="gated",
        choices=["gated", "raw"],
        help=(
            "Select checkpoints on the final gated output, or on the raw "
            "attention prediction for later OOF gate calibration."
        ),
    )
    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Fixed in advance; it is never selected using test predictions.",
    )
    parser.add_argument(
        "--classification-threshold-mode",
        default="validation",
        choices=["validation", "fixed"],
        help="Select one threshold on validation predictions, or use the fixed value.",
    )
    parser.add_argument(
        "--validation-mode",
        default="default",
        choices=["default", "group_column"],
    )
    parser.add_argument("--group-val-column", default="")
    parser.add_argument("--group-val-fraction", type=float, default=0.20)
    parser.add_argument(
        "--group-val-strategy",
        default="random",
        choices=["random", "farthest_centroid"],
    )
    parser.add_argument("--group-val-role", default="")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def indexed_true_values(
    frame: pd.DataFrame,
    target_column: str,
    row_index: np.ndarray,
) -> np.ndarray:
    by_source = {
        int(source): float(value)
        for source, value in zip(
            frame["_source_index"].to_numpy(dtype=int),
            pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=float),
            strict=True,
        )
    }
    return np.asarray([by_source[int(index)] for index in row_index], dtype=np.float32)


def train_only_condition_prior(
    *,
    train_frame: pd.DataFrame,
    pred_frame: pd.DataFrame,
    target_column: str,
    condition_columns: tuple[str, ...],
    quantile: float,
    row_index: np.ndarray,
) -> np.ndarray:
    missing = [column for column in condition_columns if column not in train_frame.columns]
    if missing:
        raise ValueError(f"Prior condition columns are not present: {missing}")
    train_keys = train_frame[list(condition_columns)].astype(str).agg("||".join, axis=1)
    pred_keys = pred_frame[list(condition_columns)].astype(str).agg("||".join, axis=1)
    train_targets = pd.to_numeric(train_frame[target_column], errors="coerce").astype(float)
    values_by_key = train_targets.groupby(train_keys).quantile(float(quantile))
    fallback = float(train_targets.quantile(float(quantile)))
    by_source = {
        int(source): float(values_by_key.get(key, fallback))
        for source, key in zip(
            pred_frame["_source_index"].to_numpy(dtype=int),
            pred_keys.to_numpy(dtype=str),
            strict=True,
        )
    }
    return np.asarray([by_source[int(index)] for index in row_index], dtype=np.float32)


def attach_density(
    bundle: AtomRelationTensorBundle,
    density: torch.Tensor,
    sample_prior: torch.Tensor,
    y: torch.Tensor | None = None,
) -> UnifiedTensorBundle:
    return UnifiedTensorBundle(
        x=bundle.x,
        numeric=bundle.numeric,
        token_mask=bundle.token_mask,
        role_ids=bundle.role_ids,
        token_type_ids=bundle.token_type_ids,
        atomic_numbers=bundle.atomic_numbers,
        component_ids=bundle.component_ids,
        density=density,
        sample_prior=sample_prior.float(),
        y=bundle.y if y is None else y.float(),
        row_index=bundle.row_index,
    )


def make_loader(
    bundle: UnifiedTensorBundle,
    batch_size: int,
    *,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        bundle.x,
        bundle.numeric,
        bundle.token_mask,
        bundle.role_ids,
        bundle.token_type_ids,
        bundle.atomic_numbers,
        bundle.density,
        bundle.sample_prior,
        bundle.y,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=0,
    )


def model_call(
    model: nn.Module,
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    *,
    diagnostics: bool,
):
    (
        x,
        numeric,
        token_mask,
        role_ids,
        token_type_ids,
        atomic_numbers,
        density,
        sample_prior,
        _,
    ) = batch
    return model(
        x.to(device, non_blocking=True),
        numeric.to(device, non_blocking=True),
        token_mask.to(device, non_blocking=True),
        role_ids.to(device, non_blocking=True),
        token_type_ids.to(device, non_blocking=True),
        atomic_numbers.to(device, non_blocking=True),
        density.to(device, non_blocking=True),
        sample_prior=sample_prior.to(device, non_blocking=True),
        return_diagnostics=diagnostics,
    )


def prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    task_type: str,
    loss_name: str,
    huber_delta: float,
    pos_weight: torch.Tensor | None,
    label_smoothing: float,
) -> torch.Tensor:
    if task_type == "classification":
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("--label-smoothing must be in [0, 1)")
        smoothed_target = (
            target
            if label_smoothing == 0.0
            else target * (1.0 - label_smoothing) + 0.5 * label_smoothing
        )
        return F.binary_cross_entropy_with_logits(
            prediction,
            smoothed_target,
            pos_weight=pos_weight,
        )
    if loss_name == "mae":
        return F.l1_loss(prediction, target)
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    return F.huber_loss(prediction, target, delta=huber_delta)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    task_type: str,
    loss_name: str,
    huber_delta: float,
    pos_weight: torch.Tensor | None,
    label_smoothing: float,
    gate_regularization: float,
    gradient_clip_norm: float,
    use_amp: bool,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    confidence_values: list[float] = []
    for batch in loader:
        target = batch[-1].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            output = model_call(model, batch, device, diagnostics=True)
            primary = prediction_loss(
                output["prediction"],
                target,
                task_type=task_type,
                loss_name=loss_name,
                huber_delta=huber_delta,
                pos_weight=pos_weight,
                label_smoothing=label_smoothing,
            )
            gate_penalty = torch.mean((1.0 - output["confidence"]) ** 2)
            loss = primary + gate_regularization * gate_penalty
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        confidence_values.append(float(output["confidence"].detach().mean().cpu()))
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "confidence": float(np.mean(confidence_values)) if confidence_values else float("nan"),
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    bundle: UnifiedTensorBundle,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        "prediction": [],
        "raw_prediction": [],
        "confidence": [],
        "ood_excess": [],
        "ood_shift": [],
        "prior": [],
    }
    base_output_keys = tuple(collected)
    loader = make_loader(
        bundle,
        batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    for batch in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            output = model_call(model, batch, device, diagnostics=True)
        for key in base_output_keys:
            collected[key].append(output[key].float().cpu().numpy())
        for key in (
            "direct_raw_prediction",
            "ddg_sign_logit",
            "ddg_sign_probability",
            "ddg_magnitude_prediction",
            "ddg_structured_prediction",
            "ddg_structured_residual",
            "pre_calibration_prediction",
            "ddg_calibration_residual",
        ):
            if key in output and output[key] is not None:
                collected.setdefault(key, []).append(
                    output[key].float().cpu().numpy()
                )
    return {
        key: np.concatenate(values, axis=0) if values else np.array([], dtype=np.float32)
        for key, values in collected.items()
    }


def task_metrics(
    y_true: np.ndarray,
    model_output: np.ndarray,
    *,
    task_type: str,
    threshold: float,
    target_mean: float,
    target_std: float,
) -> tuple[dict[str, float], np.ndarray]:
    if task_type == "classification":
        predictions = 1.0 / (1.0 + np.exp(-np.clip(model_output, -30.0, 30.0)))
        return classification_metrics(y_true, predictions, threshold=threshold), predictions
    predictions = model_output * target_std + target_mean
    return regression_metrics(y_true, predictions), predictions


def metric_score(metrics: dict[str, float], metric: str) -> float:
    if metric in {"accuracy", "auroc", "r2"}:
        return -float(metrics[metric])
    return float(metrics[metric])


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def initialize_unified_model(
    model: UnifiedORRAModel,
    checkpoint_path: Path,
    *,
    mode: str,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in checkpoint:
        source_state = checkpoint["model_state_dict"]
    elif "readout_state_dict" in checkpoint:
        source_state = checkpoint["readout_state_dict"]
    else:
        raise ValueError(
            f"{checkpoint_path} has neither model_state_dict nor readout_state_dict"
        )
    target_state = model.state_dict()
    # Task-level priors and confidence-gate scalars are run configuration,
    # not transferable learned features.  Loading any source gate buffer
    # would silently make the requested CLI configuration disagree with the
    # saved metrics while reusing a parent feature/readout checkpoint.
    excluded_prefixes = (
        "prior",
        "ood_scale_raw",
        "ood_threshold",
        "ood_shift_scale",
        "ood_base_shift",
        "residual_global_center_mix",
    )
    if mode == "features":
        excluded_prefixes = (
            "prior",
            "raw_head.",
            "gate_head.",
            "ood_scale_raw",
            "ood_threshold",
            "ood_shift_scale",
            "ood_base_shift",
            "residual_global_center_mix",
        )
    candidate_state = {
        key: value
        for key, value in source_state.items()
        if not any(
            key == prefix or key.startswith(prefix)
            for prefix in excluded_prefixes
        )
    }
    incompatible = sorted(
        key
        for key, value in candidate_state.items()
        if key not in target_state or target_state[key].shape != value.shape
    )
    if incompatible:
        raise ValueError(
            "Initialization checkpoint does not match the requested architecture; "
            f"incompatible keys include: {', '.join(incompatible[:8])}"
        )
    transferable = {
        key: value
        for key, value in candidate_state.items()
        if key in target_state and target_state[key].shape == value.shape
    }
    model.load_state_dict(transferable, strict=False)
    transferred_parameters = int(
        sum(target_state[key].numel() for key in transferable)
    )
    print(
        f"[initialize] path={checkpoint_path} mode={mode} "
        f"tensors={len(transferable)} parameters={transferred_parameters}",
        flush=True,
    )
    return {
        "path": str(checkpoint_path),
        "mode": mode,
        "transferred_tensors": len(transferable),
        "transferred_parameters": transferred_parameters,
        "source_best_epoch": int(checkpoint.get("best_epoch", -1)),
    }


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    token_dir = args.atom_token_dir
    if token_dir is None:
        token_dir = ROOT / "reproduction_outputs" / "full_retrain" / TOKEN_DIR_NAMES[args.task]
    output_dir = args.out_dir
    if output_dir is None:
        output_dir = (
            ROOT
            / "reproduction_outputs"
            / "unified_orra"
            / f"{args.task}_{args.conformer_feature_mode}_seed{args.seed}"
        )
    mapping_task = args.mapping_task or ("aho_major" if args.task == "aho_ddg" else args.task)
    return token_dir, output_dir, mapping_task


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.adam_beta1 < 1.0:
        raise ValueError("--adam-beta1 must be in [0, 1)")
    if not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("--adam-beta2 must be in [0, 1)")
    if args.adam_eps <= 0.0:
        raise ValueError("--adam-eps must be positive")
    if args.gradient_clip_norm < 0.0:
        raise ValueError("--gradient-clip-norm must be non-negative")
    token_dir, output_dir, mapping_task = resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    store = load_atom_token_store(token_dir)
    task = load_task(args.task, split=args.split)
    if task.task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported task type: {task.task_type}")
    if args.ddg_sign_magnitude and task.task_type != "regression":
        raise ValueError(
            "--ddg-sign-magnitude requires a regression task"
        )
    if args.ddg_sign_magnitude_hidden_dim <= 0:
        raise ValueError(
            "--ddg-sign-magnitude-hidden-dim must be positive"
        )
    if args.ddg_sign_magnitude_residual_multiplier < 0.0:
        raise ValueError(
            "--ddg-sign-magnitude-residual-multiplier must be non-negative"
        )
    if args.ddg_residual_calibrator and not args.ddg_sign_magnitude:
        raise ValueError(
            "--ddg-residual-calibrator requires --ddg-sign-magnitude"
        )
    if args.ddg_residual_calibrator_nonlinear_hidden_dim < 0:
        raise ValueError(
            "--ddg-residual-calibrator-nonlinear-hidden-dim must be "
            "non-negative"
        )
    if (
        args.ddg_residual_calibrator_nonlinear_hidden_dim > 0
        and not args.ddg_residual_calibrator
    ):
        raise ValueError(
            "A nonlinear residual calibrator requires "
            "--ddg-residual-calibrator"
        )

    effective_mapping_task = mapping_task
    part_names = {"train": "train", "val": "val", "test": "test"}
    if args.validation_mode == "group_column":
        frames = group_validation_frames_by_strategy(
            task,
            store=store,
            mapping_task=mapping_task,
            column=args.group_val_column,
            fraction=args.group_val_fraction,
            seed=args.seed,
            strategy=args.group_val_strategy,
            role=args.group_val_role,
        )
        effective_mapping_task = (
            f"{mapping_task}__unified_groupval_{args.group_val_column.lower()}_"
            f"{args.group_val_strategy}"
        )
        store = store_with_source_index_mapping(
            store=store,
            mapping_task=mapping_task,
            target_task=effective_mapping_task,
            frame_by_part=frames,
        )
        part_names = {
            "train": "group_train",
            "val": "group_val",
            "test": "group_test",
        }
    else:
        frames = split_frames(task)

    parts = (
        ("train", "val")
        if args.skip_test_evaluation
        else ("train", "val", "test")
    )
    raw_bundles: dict[str, AtomRelationTensorBundle] = {}
    for part in parts:
        raw_bundles[part] = build_tensor_bundle(
            store=store,
            mapping_task=effective_mapping_task,
            part=part_names[part],
            target_frame=frames[part],
            target_column=task.target_column,
            max_rows=args.max_train_rows if part == "train" else args.max_eval_rows,
            max_tokens=args.max_tokens,
            conformer_feature_mode=args.conformer_feature_mode,
        )

    density_profile = ComponentDensityProfile(
        num_roles=16,
        quantile=args.density_quantile,
        unseen_floor=args.unseen_density_floor,
        ratio_cap=args.density_ratio_cap,
    ).fit(raw_bundles["train"])
    density_by_part = {
        part: density_profile.transform(
            raw_bundles[part],
            training=(
                part == "train"
                and args.density_train_mode == "leave_one_component_out"
            ),
        )
        for part in parts
    }

    if task.task_type == "regression":
        target_mean = float(raw_bundles["train"].y.mean().item())
        target_std = float(raw_bundles["train"].y.std(unbiased=False).item())
        if target_std < 1e-6:
            target_std = 1.0
        scaled_targets = {
            part: (raw_bundles[part].y - target_mean) / target_std
            for part in parts
        }
        prior = 0.0
    else:
        target_mean = 0.0
        target_std = 1.0
        scaled_targets = {part: raw_bundles[part].y for part in parts}
        positive_rate = float(raw_bundles["train"].y.mean().item())
        positive_rate = min(max(positive_rate, 1e-4), 1.0 - 1e-4)
        prior = math.log(positive_rate / (1.0 - positive_rate))
    ddg_sign_magnitude_zero_point = (
        -target_mean / target_std
        if args.ddg_sign_magnitude
        else 0.0
    )

    prior_columns = tuple(
        column.strip()
        for column in args.prior_condition_columns.split(",")
        if column.strip()
    )
    if args.sample_prior_mode == "condition_quantile":
        if task.task_type != "regression":
            raise ValueError("condition_quantile sample priors currently require regression")
        if not prior_columns:
            raise ValueError(
                "--prior-condition-columns is required for condition_quantile sample priors"
            )
        sample_prior_by_part = {}
        for part in parts:
            raw_prior = train_only_condition_prior(
                train_frame=frames["train"],
                pred_frame=frames[part],
                target_column=task.target_column,
                condition_columns=prior_columns,
                quantile=args.prior_quantile,
                row_index=raw_bundles[part].row_index,
            )
            sample_prior_by_part[part] = torch.from_numpy(
                ((raw_prior - target_mean) / target_std).astype(np.float32)
            )
    elif args.sample_prior_mode == "train_quantile":
        if task.task_type != "regression":
            raise ValueError("train_quantile sample priors currently require regression")
        raw_prior = float(
            pd.to_numeric(
                frames["train"][task.target_column],
                errors="coerce",
            ).quantile(float(args.prior_quantile))
        )
        scaled_prior = (raw_prior - target_mean) / target_std
        sample_prior_by_part = {
            part: torch.full(
                (len(raw_bundles[part].y),),
                float(scaled_prior),
                dtype=torch.float32,
            )
            for part in parts
        }
    else:
        sample_prior_by_part = {
            part: torch.full(
                (len(raw_bundles[part].y),),
                float(prior),
                dtype=torch.float32,
            )
            for part in parts
        }

    bundles = {
        part: attach_density(
            raw_bundles[part],
            density_by_part[part],
            sample_prior_by_part[part],
            y=scaled_targets[part],
        )
        for part in parts
    }
    true_by_part = {
        part: indexed_true_values(
            frames[part],
            task.target_column,
            raw_bundles[part].row_index,
        )
        for part in parts
        if part != "train"
    }

    selection_metric = args.selection_metric
    if selection_metric == "auto":
        selection_metric = "accuracy" if task.task_type == "classification" else "mae"
    valid_metrics = (
        {"accuracy", "auroc"} if task.task_type == "classification" else {"mae", "rmse", "r2"}
    )
    if selection_metric not in valid_metrics:
        raise ValueError(
            f"selection metric {selection_metric!r} is invalid for {task.task_type}; "
            f"choose from {sorted(valid_metrics)}"
        )

    model = UnifiedORRAModel(
        orb_dim=int(raw_bundles["train"].x.shape[-1]),
        numeric_dim=int(raw_bundles["train"].numeric.shape[-1]),
        density_dim=16 + 3,
        hidden_dim=args.hidden_dim,
        num_roles=16,
        num_token_types=max(TOKEN_TYPE_IDS.values()) + 1,
        num_self_layers=args.num_self_layers,
        num_role_layers=args.num_role_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        prior=prior,
        regression_head_hidden_dims=args.regression_head_hidden_dims,
        regression_head_dropout=args.regression_head_dropout,
        use_token_type_cross_attention=args.cross_attention_view == "roles_types",
        ood_threshold=args.ood_threshold,
        ood_scale_init=args.ood_scale_init,
        exponential_ood_gate=args.ood_gate_mode == "exponential",
        gate_residual_mode=args.gate_residual_mode,
        ood_shift_scale=args.ood_shift_scale,
        ood_base_shift=args.ood_base_shift,
        residual_global_center_mix=args.residual_global_center_mix,
        use_grouped_statistics=args.grouped_statistics,
        use_cross_view_interactions=args.cross_view_interactions,
        use_ddg_sign_magnitude=args.ddg_sign_magnitude,
        ddg_sign_magnitude_hidden_dim=(
            args.ddg_sign_magnitude_hidden_dim
        ),
        ddg_sign_magnitude_target_zero_point=(
            ddg_sign_magnitude_zero_point
        ),
        ddg_sign_magnitude_residual_multiplier=(
            args.ddg_sign_magnitude_residual_multiplier
        ),
        use_ddg_residual_calibrator=args.ddg_residual_calibrator,
        ddg_residual_calibrator_nonlinear_hidden_dim=(
            args.ddg_residual_calibrator_nonlinear_hidden_dim
        ),
    ).to(device)
    init_info: dict[str, object] | None = None
    if args.init_checkpoint is not None:
        init_info = initialize_unified_model(
            model,
            args.init_checkpoint,
            mode=args.init_mode,
        )
    if args.freeze_gate:
        for parameter in model.gate_head.parameters():
            parameter.requires_grad = False
        model.ood_scale_raw.requires_grad = False
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )
    train_loader = make_loader(
        bundles["train"],
        args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    if task.task_type == "classification":
        if args.positive_class_weight_scale <= 0.0:
            raise ValueError("--positive-class-weight-scale must be positive")
        positives = float(raw_bundles["train"].y.sum().item())
        negatives = float(len(raw_bundles["train"].y) - positives)
        pos_weight = torch.tensor(
            (negatives / max(positives, 1.0))
            * args.positive_class_weight_scale,
            dtype=torch.float32,
            device=device,
        )
    else:
        pos_weight = None

    best_score = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_classification_threshold = float(args.classification_threshold)
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    if args.evaluation_only:
        if args.init_checkpoint is None:
            raise ValueError("--evaluation-only requires --init-checkpoint")
        val_output = predict(
            model,
            bundles["val"],
            batch_size=args.batch_size,
            device=device,
            use_amp=use_amp,
        )
        selection_values = (
            val_output["raw_prediction"]
            if args.selection_output == "raw"
            else val_output["prediction"]
        )
        val_metrics, val_predictions = task_metrics(
            true_by_part["val"],
            selection_values,
            task_type=task.task_type,
            threshold=args.classification_threshold,
            target_mean=target_mean,
            target_std=target_std,
        )
        if (
            task.task_type == "classification"
            and args.classification_threshold_mode == "validation"
        ):
            threshold_result = best_threshold_accuracy(
                true_by_part["val"],
                val_predictions,
            )
            best_classification_threshold = float(
                threshold_result["best_threshold"]
            )
            val_metrics = classification_metrics(
                true_by_part["val"],
                val_predictions,
                threshold=best_classification_threshold,
            )
        best_score = metric_score(val_metrics, selection_metric)
        best_epoch = int(init_info["source_best_epoch"]) if init_info else -1
        best_state = clone_state_dict(model)
        history.append(
            {
                "epoch": best_epoch,
                "train_loss": float("nan"),
                "train_confidence": float("nan"),
                "validation_score": best_score,
                "learning_rate": 0.0,
                "validation_confidence": float(
                    val_output["confidence"].mean()
                ),
                "validation_ood_excess": float(
                    val_output["ood_excess"].mean()
                ),
                **{
                    f"validation_{key}": value
                    for key, value in val_metrics.items()
                },
            }
        )
        print(
            f"[evaluation-only] source_best_epoch={best_epoch + 1} "
            + " ".join(
                f"{key}={value:.6f}"
                for key, value in val_metrics.items()
                if key not in {"n", "threshold"}
            ),
            flush=True,
        )

    for epoch in range(0 if args.evaluation_only else args.epochs):
        train_stats = train_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            task_type=task.task_type,
            loss_name=args.loss,
            huber_delta=args.huber_delta,
            pos_weight=pos_weight,
            label_smoothing=args.label_smoothing,
            gate_regularization=args.gate_regularization,
            gradient_clip_norm=args.gradient_clip_norm,
            use_amp=use_amp,
        )
        val_output = predict(
            model,
            bundles["val"],
            batch_size=args.batch_size,
            device=device,
            use_amp=use_amp,
        )
        selection_values = (
            val_output["raw_prediction"]
            if args.selection_output == "raw"
            else val_output["prediction"]
        )
        val_metrics, val_predictions = task_metrics(
            true_by_part["val"],
            selection_values,
            task_type=task.task_type,
            threshold=args.classification_threshold,
            target_mean=target_mean,
            target_std=target_std,
        )
        epoch_threshold = float(args.classification_threshold)
        if (
            task.task_type == "classification"
            and args.classification_threshold_mode == "validation"
        ):
            threshold_result = best_threshold_accuracy(
                true_by_part["val"],
                val_predictions,
            )
            epoch_threshold = float(threshold_result["best_threshold"])
            val_metrics = classification_metrics(
                true_by_part["val"],
                val_predictions,
                threshold=epoch_threshold,
            )
        score = metric_score(val_metrics, selection_metric)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_confidence": train_stats["confidence"],
            "validation_score": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_confidence": float(val_output["confidence"].mean()),
            "validation_ood_excess": float(val_output["ood_excess"].mean()),
        }
        row.update({f"validation_{key}": value for key, value in val_metrics.items()})
        history.append(row)
        if score < best_score - 1e-8:
            best_score = score
            best_epoch = epoch
            best_state = clone_state_dict(model)
            best_classification_threshold = epoch_threshold
            stale_epochs = 0
        else:
            stale_epochs += 1
        scheduler.step(score)
        metric_text = " ".join(
            f"{key}={value:.6f}"
            for key, value in val_metrics.items()
            if key not in {"n", "threshold"}
        )
        print(
            f"[epoch] {epoch + 1}/{args.epochs} loss={train_stats['loss']:.6f} "
            f"train_gate={train_stats['confidence']:.4f} val_gate={val_output['confidence'].mean():.4f} "
            f"{metric_text}",
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(
                f"[early-stop] epoch={epoch + 1} best_epoch={best_epoch + 1}",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    final_threshold = (
        best_classification_threshold
        if task.task_type == "classification"
        else float(args.classification_threshold)
    )

    final_outputs: dict[str, dict[str, np.ndarray]] = {}
    final_metrics: dict[str, dict[str, float]] = {}
    final_predictions: dict[str, np.ndarray] = {}
    evaluation_parts = tuple(part for part in parts if part != "train")
    for part in evaluation_parts:
        final_outputs[part] = predict(
            model,
            bundles[part],
            batch_size=args.batch_size,
            device=device,
            use_amp=use_amp,
        )
        final_metrics[part], final_predictions[part] = task_metrics(
            true_by_part[part],
            final_outputs[part]["prediction"],
            task_type=task.task_type,
            threshold=final_threshold,
            target_mean=target_mean,
            target_std=target_std,
        )
        prediction_columns = {
            "row_index": raw_bundles[part].row_index,
            "y_true": true_by_part[part],
            "y_pred": final_predictions[part],
            "raw_model_output": final_outputs[part]["raw_prediction"],
            "confidence": final_outputs[part]["confidence"],
            "ood_excess": final_outputs[part]["ood_excess"],
            "ood_shift": final_outputs[part]["ood_shift"],
            "sample_prior": (
                1.0
                / (
                    1.0
                    + np.exp(
                        -np.clip(final_outputs[part]["prior"], -30.0, 30.0)
                    )
                )
                if task.task_type == "classification"
                else final_outputs[part]["prior"] * target_std + target_mean
            ),
        }
        for key in (
            "direct_raw_prediction",
            "ddg_sign_logit",
            "ddg_sign_probability",
            "ddg_magnitude_prediction",
            "ddg_structured_prediction",
            "ddg_structured_residual",
            "pre_calibration_prediction",
            "ddg_calibration_residual",
        ):
            if key in final_outputs[part]:
                prediction_columns[key] = final_outputs[part][key]
        pd.DataFrame(prediction_columns).to_csv(
            output_dir / f"{part}_predictions.csv",
            index=False,
        )

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    metrics_row: dict[str, object] = {
        "task": task.name,
        "task_type": task.task_type,
        "split": args.split,
        "model": "unified_orra_single",
        "architecture": (
            "atom_self_attention+role_and_token_type_cross_attention+"
            "shared_set_self_attention+density_gate"
            if args.cross_attention_view == "roles_types"
            else "atom_self_attention+role_cross_attention+role_self_attention+density_gate"
        )
        + ("+grouped_mean_std_max" if args.grouped_statistics else "")
        + ("+cross_view_pair_attention" if args.cross_view_interactions else "")
        + ("+ddg_sign_magnitude" if args.ddg_sign_magnitude else "")
        + (
            "+ddg_residual_calibrator"
            if args.ddg_residual_calibrator
            else ""
        )
        + (
            "+ddg_residual_nonlinear_adapter"
            if args.ddg_residual_calibrator_nonlinear_hidden_dim > 0
            else ""
        ),
        "single_checkpoint": True,
        "single_forward": True,
        "ensemble_size": 1,
        "parameter_count": parameter_count,
        "n_train": int(len(bundles["train"].y)),
        "n_val": int(len(bundles["val"].y)),
        "n_test": int(len(frames["test"])),
        "test_evaluated": not args.skip_test_evaluation,
        "best_epoch": int(best_epoch),
        "selection_metric": selection_metric,
        "selection_output": args.selection_output,
        "classification_threshold": float(final_threshold),
        "classification_threshold_mode": args.classification_threshold_mode,
        "target_mean": target_mean,
        "target_std": target_std,
        "atom_token_dir": str(token_dir),
        "mapping_task": mapping_task,
        "effective_mapping_task": effective_mapping_task,
        "validation_mode": args.validation_mode,
        "sample_prior_mode": args.sample_prior_mode,
        "prior_condition_columns": ",".join(prior_columns),
        "prior_quantile": float(args.prior_quantile),
        "ood_threshold": float(args.ood_threshold),
        "ood_scale_init": float(args.ood_scale_init),
        "ood_gate_mode": args.ood_gate_mode,
        "gate_residual_mode": args.gate_residual_mode,
        "ood_shift_scale": float(args.ood_shift_scale),
        "ood_base_shift": float(args.ood_base_shift),
        "residual_global_center_mix": float(args.residual_global_center_mix),
        "density_train_mode": args.density_train_mode,
        "density_ratio_cap": float(args.density_ratio_cap),
        "unseen_density_floor": float(args.unseen_density_floor),
        "freeze_gate": bool(args.freeze_gate),
        "label_smoothing": float(args.label_smoothing),
        "positive_class_weight_scale": float(args.positive_class_weight_scale),
        "adam_beta1": float(args.adam_beta1),
        "adam_beta2": float(args.adam_beta2),
        "adam_eps": float(args.adam_eps),
        "gradient_clip_norm": float(args.gradient_clip_norm),
        "ddg_sign_magnitude": bool(args.ddg_sign_magnitude),
        "ddg_sign_magnitude_hidden_dim": int(
            args.ddg_sign_magnitude_hidden_dim
        ),
        "ddg_sign_magnitude_target_zero_point": float(
            ddg_sign_magnitude_zero_point
        ),
        "ddg_sign_magnitude_residual_multiplier": float(
            args.ddg_sign_magnitude_residual_multiplier
        ),
        "ddg_residual_calibrator": bool(
            args.ddg_residual_calibrator
        ),
        "ddg_residual_calibrator_nonlinear_hidden_dim": int(
            args.ddg_residual_calibrator_nonlinear_hidden_dim
        ),
        "init_checkpoint": (
            str(args.init_checkpoint)
            if args.init_checkpoint is not None
            else ""
        ),
        "init_mode": args.init_mode if init_info is not None else "",
        "init_transferred_tensors": (
            int(init_info["transferred_tensors"])
            if init_info is not None
            else 0
        ),
        "init_transferred_parameters": (
            int(init_info["transferred_parameters"])
            if init_info is not None
            else 0
        ),
    }
    if "test" in final_metrics:
        metrics_row.update(final_metrics["test"])
    metrics_row.update(
        {f"validation_{key}": value for key, value in final_metrics["val"].items()}
    )
    metrics_row["validation_mean_confidence"] = float(
        final_outputs["val"]["confidence"].mean()
    )
    metrics_row["validation_mean_ood_excess"] = float(
        final_outputs["val"]["ood_excess"].mean()
    )
    metrics_row["validation_mean_ood_shift"] = float(
        final_outputs["val"]["ood_shift"].mean()
    )
    if "test" in final_outputs:
        metrics_row["test_mean_confidence"] = float(
            final_outputs["test"]["confidence"].mean()
        )
        metrics_row["test_mean_ood_excess"] = float(
            final_outputs["test"]["ood_excess"].mean()
        )
        metrics_row["test_mean_ood_shift"] = float(
            final_outputs["test"]["ood_shift"].mean()
        )
    pd.DataFrame([metrics_row]).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "orb_dim": int(raw_bundles["train"].x.shape[-1]),
            "numeric_dim": int(raw_bundles["train"].numeric.shape[-1]),
            "density_dim": 19,
            "density_ratio_cap": args.density_ratio_cap,
            "hidden_dim": args.hidden_dim,
            "num_roles": 16,
            "num_token_types": max(TOKEN_TYPE_IDS.values()) + 1,
            "num_self_layers": args.num_self_layers,
            "num_role_layers": args.num_role_layers,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "prior": prior,
            "regression_head_hidden_dims": (
                args.regression_head_hidden_dims
            ),
            "regression_head_dropout": args.regression_head_dropout,
            "use_token_type_cross_attention": args.cross_attention_view == "roles_types",
            "ood_threshold": args.ood_threshold,
            "ood_scale_init": args.ood_scale_init,
            "exponential_ood_gate": args.ood_gate_mode == "exponential",
            "gate_residual_mode": args.gate_residual_mode,
            "ood_shift_scale": args.ood_shift_scale,
            "ood_base_shift": args.ood_base_shift,
            "residual_global_center_mix": args.residual_global_center_mix,
            "use_grouped_statistics": args.grouped_statistics,
            "use_cross_view_interactions": args.cross_view_interactions,
            "use_ddg_sign_magnitude": args.ddg_sign_magnitude,
            "ddg_sign_magnitude_hidden_dim": (
                args.ddg_sign_magnitude_hidden_dim
            ),
            "ddg_sign_magnitude_target_zero_point": (
                ddg_sign_magnitude_zero_point
            ),
            "ddg_sign_magnitude_residual_multiplier": (
                args.ddg_sign_magnitude_residual_multiplier
            ),
            "use_ddg_residual_calibrator": (
                args.ddg_residual_calibrator
            ),
            "ddg_residual_calibrator_nonlinear_hidden_dim": (
                args.ddg_residual_calibrator_nonlinear_hidden_dim
            ),
        },
        "target_mean": target_mean,
        "target_std": target_std,
        "best_epoch": best_epoch,
        "task": task.name,
        "task_type": task.task_type,
        "classification_threshold": final_threshold,
        "classification_threshold_mode": args.classification_threshold_mode,
        "initialization": init_info,
    }
    torch.save(checkpoint, output_dir / "model.pt")
    joblib.dump(density_profile, output_dir / "density_profile.joblib")

    config = vars(args).copy()
    config.update(
        {
            "atom_token_dir": str(token_dir),
            "out_dir": str(output_dir),
            "mapping_task": mapping_task,
            "effective_mapping_task": effective_mapping_task,
            "use_amp": use_amp,
        }
    )
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print(pd.DataFrame([metrics_row]).to_string(index=False), flush=True)
    print(f"Wrote {output_dir}", flush=True)


if __name__ == "__main__":
    main()
