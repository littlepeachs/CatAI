from __future__ import annotations

import math

import torch
from torch import nn


def masked_attention_pool(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    scorer: nn.Module,
) -> torch.Tensor:
    """Attention-pool a padded token sequence without producing NaNs."""
    mask = mask.bool()
    scores = scorer(tokens).squeeze(-1)
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=1)
    weights = weights * mask.to(weights.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return (tokens * weights.unsqueeze(-1)).sum(dim=1)


class RoleCrossAttention(nn.Module):
    """Use the same learned query mechanism for every chemical component role."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_roles: int,
        dropout: float,
        use_grouped_statistics: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_roles = num_roles
        self.use_grouped_statistics = bool(use_grouped_statistics)

        self.role_queries = nn.Parameter(torch.empty(num_roles, hidden_dim))
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)
        if self.use_grouped_statistics:
            self.stat_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        nn.init.normal_(self.role_queries, mean=0.0, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        role_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_tokens, _ = tokens.shape
        queries = self.role_queries.unsqueeze(0).expand(batch_size, -1, -1)
        q = self.q_proj(self.query_norm(queries))
        k = self.k_proj(self.token_norm(tokens))
        v = self.v_proj(self.token_norm(tokens))

        q = q.view(batch_size, self.num_roles, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        role_range = torch.arange(self.num_roles, device=tokens.device)
        role_token_mask = (
            token_mask[:, None, :].bool()
            & (role_ids[:, None, :].long() == role_range[None, :, None])
        )
        role_present = role_token_mask.any(dim=-1)
        scores = scores.masked_fill(
            ~role_token_mask[:, None, :, :],
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = weights * role_token_mask[:, None, :, :].to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = self.attn_dropout(weights)

        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(batch_size, self.num_roles, self.hidden_dim)
        role_tokens = queries + self.residual_dropout(self.out_proj(attended))
        if self.use_grouped_statistics:
            stat_mask = role_token_mask.to(tokens.dtype)
            count = stat_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean = torch.einsum("brt,bth->brh", stat_mask, tokens) / count
            second = torch.einsum("brt,bth->brh", stat_mask, tokens.square()) / count
            std = torch.sqrt(torch.clamp(second - mean.square(), min=0.0) + 1e-8)
            group_index = role_ids.long().unsqueeze(-1).expand(
                -1,
                -1,
                self.hidden_dim,
            )
            maximum = torch.full(
                (
                    batch_size,
                    self.num_roles,
                    self.hidden_dim,
                ),
                torch.finfo(tokens.dtype).min,
                dtype=tokens.dtype,
                device=tokens.device,
            )
            maximum.scatter_reduce_(
                1,
                group_index,
                tokens.masked_fill(
                    ~token_mask.bool().unsqueeze(-1),
                    torch.finfo(tokens.dtype).min,
                ),
                reduce="amax",
                include_self=True,
            )
            maximum = torch.where(
                role_present.unsqueeze(-1),
                maximum,
                torch.zeros_like(maximum),
            )
            role_tokens = role_tokens + self.stat_proj(
                torch.cat([mean, std, maximum], dim=-1)
            )
        role_tokens = role_tokens + self.ffn(self.output_norm(role_tokens))
        role_tokens = role_tokens * role_present.unsqueeze(-1).to(role_tokens.dtype)
        return role_tokens, role_present


class ConformerAttentionAggregator(nn.Module):
    """Shared per-atom attention over aligned Orb conformer features.

    The final scorer is zero-initialized, so a newly enabled aggregator starts
    as an exact uniform conformer pool before learning task-specific weights.
    """

    def __init__(
        self,
        *,
        input_dim: int = 256,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("conformer attention dimensions must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(self.input_dim)
        self.scorer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(
        self,
        conformer_features: torch.Tensor,
        feature_mode: str,
    ) -> torch.Tensor:
        if conformer_features.ndim != 3:
            raise ValueError(
                "conformer_features must have shape "
                "[num_conformers, num_atoms, feature_dim]"
            )
        if conformer_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected conformer feature dim {self.input_dim}, got "
                f"{conformer_features.shape[-1]}"
            )
        if feature_mode not in {"mean", "mean_std", "mean_std_max"}:
            raise ValueError(f"Unsupported conformer feature mode: {feature_mode}")
        scores = self.scorer(self.norm(conformer_features)).squeeze(-1)
        weights = torch.softmax(scores, dim=0)
        attention_mean = (
            conformer_features * weights.unsqueeze(-1).to(conformer_features.dtype)
        ).sum(dim=0)
        parts = [attention_mean]
        if feature_mode in {"mean_std", "mean_std_max"}:
            parts.append(conformer_features.std(dim=0, unbiased=False))
        if feature_mode == "mean_std_max":
            parts.append(conformer_features.max(dim=0).values)
        return torch.cat(parts, dim=-1)


class MultiLevelOrbFeatureAttention(nn.Module):
    """Read early/middle/final Orb features while preserving the final level.

    A shared scorer learns atom-conditional weights across Orb levels.  The
    per-channel residual gate starts at exactly zero, so a newly enabled A4
    module returns the original final-level feature exactly before training.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        num_levels: int,
        score_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or num_levels < 2 or score_hidden_dim <= 0:
            raise ValueError("multi-level Orb attention dimensions are invalid")
        self.input_dim = int(input_dim)
        self.num_levels = int(num_levels)
        self.score_hidden_dim = int(score_hidden_dim)
        self.level_norm = nn.LayerNorm(self.input_dim)
        self.score_hidden = nn.Linear(self.input_dim, self.score_hidden_dim)
        self.score_output = nn.Linear(self.score_hidden_dim, 1)
        self.level_logits = nn.Parameter(torch.zeros(self.num_levels))
        self.residual_gate = nn.Parameter(torch.zeros(self.input_dim))
        nn.init.zeros_(self.score_output.weight)
        nn.init.zeros_(self.score_output.bias)

    def forward(self, level_features: torch.Tensor) -> torch.Tensor:
        if level_features.ndim < 3:
            raise ValueError(
                "level_features must have shape [..., num_levels, input_dim]"
            )
        if level_features.shape[-2] != self.num_levels:
            raise ValueError(
                f"Expected {self.num_levels} Orb levels, got "
                f"{level_features.shape[-2]}"
            )
        if level_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected Orb level feature dim {self.input_dim}, got "
                f"{level_features.shape[-1]}"
            )
        normalized = self.level_norm(level_features.float())
        scores = self.score_output(
            torch.nn.functional.gelu(self.score_hidden(normalized))
        ).squeeze(-1)
        scores = scores + self.level_logits
        weights = torch.softmax(scores, dim=-1)
        float_features = level_features.float()
        context = (
            float_features * weights.unsqueeze(-1).to(float_features.dtype)
        ).sum(dim=-2)
        final_features = float_features[..., -1, :]
        gate = torch.tanh(self.residual_gate).to(final_features.dtype)
        return final_features + gate * (context - final_features)


class DualBackboneFeatureResidualFusion(nn.Module):
    """Add a zero-initialized residual from two aligned backbone views.

    The primary view remains the exact parent input at initialization.  The
    final projection is initialized to zero, so enabling this incremental
    module cannot perturb an existing readout before the adapter is trained.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 64,
        num_roles: int | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                "dual backbone feature fusion dimensions must be positive"
            )
        if num_roles is not None and num_roles <= 0:
            raise ValueError(
                "dual backbone feature role count must be positive"
            )
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_roles = int(num_roles) if num_roles is not None else None
        self.primary_norm = nn.LayerNorm(self.input_dim)
        self.secondary_norm = nn.LayerNorm(self.input_dim)
        self.hidden = nn.Linear(self.input_dim * 2, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, self.input_dim)
        if self.num_roles is None:
            self.register_parameter("role_scale_delta", None)
        else:
            self.role_scale_delta = nn.Parameter(
                torch.zeros(self.num_roles)
            )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        primary_features: torch.Tensor,
        secondary_features: torch.Tensor,
        *,
        role_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if primary_features.shape != secondary_features.shape:
            raise ValueError(
                "dual backbone feature views must have identical shapes; "
                f"got {tuple(primary_features.shape)} and "
                f"{tuple(secondary_features.shape)}"
            )
        if primary_features.ndim < 2:
            raise ValueError(
                "dual backbone feature views must have at least two dimensions"
            )
        if primary_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected dual-view feature dim {self.input_dim}, got "
                f"{primary_features.shape[-1]}"
            )
        primary = primary_features.float()
        secondary = secondary_features.float()
        joined = torch.cat(
            [
                self.primary_norm(primary),
                self.secondary_norm(secondary),
            ],
            dim=-1,
        )
        residual = self.output(
            torch.nn.functional.gelu(self.hidden(joined))
        )
        if self.role_scale_delta is not None:
            if role_ids is None:
                raise ValueError(
                    "role-conditioned dual feature fusion requires role_ids"
                )
            if tuple(role_ids.shape) != tuple(primary_features.shape[:-1]):
                raise ValueError(
                    "dual feature role_ids shape must match token dimensions; "
                    f"got {tuple(role_ids.shape)} and "
                    f"{tuple(primary_features.shape[:-1])}"
                )
            integer_roles = role_ids.long()
            if integer_roles.numel() > 0 and (
                int(integer_roles.min()) < 0
                or int(integer_roles.max()) >= self.num_roles
            ):
                raise ValueError(
                    "dual feature role_ids are outside the configured range"
                )
            scales = 1.0 + torch.tanh(
                self.role_scale_delta[integer_roles]
            )
            residual = residual * scales.unsqueeze(-1).to(residual.dtype)
        elif role_ids is not None:
            raise ValueError(
                "role_ids were supplied while dual feature role scaling "
                "is disabled"
            )
        return primary + residual


class AHOPhysicalConsistencyAdapter(nn.Module):
    """Couple AHO Major and signed ΔΔG‡ inside one task-specific network.

    The auxiliary head learns the complementary physical target. A scalar
    zero-initialized coupling turns that prediction into a residual on the
    primary output, so enabling the adapter preserves the parent function
    exactly and never averages predictions from separate checkpoints.
    """

    def __init__(
        self,
        *,
        fusion_dim: int,
        hidden_dim: int,
        mode: str,
        auxiliary_zero_point: float = 0.0,
    ) -> None:
        super().__init__()
        if fusion_dim <= 0 or hidden_dim <= 0:
            raise ValueError("AHO consistency dimensions must be positive")
        if mode not in {"major_with_ddg", "ddg_with_major"}:
            raise ValueError(
                "AHO consistency mode must be major_with_ddg or ddg_with_major"
            )
        self.fusion_dim = int(fusion_dim)
        self.hidden_dim = int(hidden_dim)
        self.mode = str(mode)
        self.norm = nn.LayerNorm(self.fusion_dim)
        self.hidden = nn.Linear(self.fusion_dim, self.hidden_dim)
        self.output = nn.Linear(self.hidden_dim, 1)
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "auxiliary_zero_point",
            torch.tensor(float(auxiliary_zero_point), dtype=torch.float32),
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        fusion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        auxiliary_prediction = self.output(
            torch.nn.functional.gelu(self.hidden(self.norm(fusion)))
        ).squeeze(-1)
        if self.mode == "major_with_ddg":
            aligned_auxiliary = (
                auxiliary_prediction - self.auxiliary_zero_point
            )
        else:
            aligned_auxiliary = torch.tanh(auxiliary_prediction)
        primary_residual = (
            torch.tanh(self.residual_scale) * aligned_auxiliary
        )
        return auxiliary_prediction, primary_residual


class DDGSignMagnitudeAdapter(nn.Module):
    """Factor one ΔΔG‡ prediction into a learned sign and positive magnitude.

    A zero-initialized residual gate preserves the direct-regression parent
    exactly. The structured branch is internal to the same readout and does
    not consume predictions from another task checkpoint.
    """

    def __init__(
        self,
        *,
        fusion_dim: int,
        hidden_dim: int,
        target_zero_point: float,
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if fusion_dim <= 0 or hidden_dim <= 0:
            raise ValueError("DDG sign-magnitude dimensions must be positive")
        if residual_multiplier < 0.0:
            raise ValueError(
                "DDG sign-magnitude residual multiplier must be non-negative"
            )
        self.fusion_dim = int(fusion_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_multiplier = float(residual_multiplier)
        self.norm = nn.LayerNorm(self.fusion_dim)
        self.hidden = nn.Linear(self.fusion_dim, self.hidden_dim)
        self.sign_output = nn.Linear(self.hidden_dim, 1)
        self.magnitude_output = nn.Linear(self.hidden_dim, 1)
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "target_zero_point",
            torch.tensor(float(target_zero_point), dtype=torch.float32),
        )
        nn.init.zeros_(self.sign_output.weight)
        nn.init.zeros_(self.sign_output.bias)
        nn.init.zeros_(self.magnitude_output.weight)
        nn.init.zeros_(self.magnitude_output.bias)

    def forward(
        self,
        fusion: torch.Tensor,
        direct_prediction: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        hidden = torch.nn.functional.gelu(
            self.hidden(self.norm(fusion))
        )
        sign_logit = self.sign_output(hidden).squeeze(-1)
        magnitude = torch.nn.functional.softplus(
            self.magnitude_output(hidden).squeeze(-1)
        )
        structured_prediction = (
            self.target_zero_point
            + torch.tanh(sign_logit) * magnitude
        )
        structured_residual = (
            self.residual_multiplier
            * torch.tanh(self.residual_scale)
            * (structured_prediction - direct_prediction)
        )
        return (
            sign_logit,
            magnitude,
            structured_prediction,
            structured_residual,
        )


class DDGResidualCalibrator(nn.Module):
    """Small post-gate residual head using only same-network A6 diagnostics.

    The 13 scalar context features are normalized with train-batch statistics
    and passed through one linear layer.  Optionally, a zero-initialized
    nonlinear residual branch can be added on top of the historical linear
    A7 output.  Both paths live in the same readout checkpoint; no second
    predictor or checkpoint is consulted.
    """

    feature_dim = 13

    def __init__(self, *, nonlinear_hidden_dim: int = 0) -> None:
        super().__init__()
        if int(nonlinear_hidden_dim) < 0:
            raise ValueError(
                "DDG nonlinear calibrator hidden dimension must be "
                "non-negative"
            )
        self.nonlinear_hidden_dim = int(nonlinear_hidden_dim)
        self.momentum = 0.05
        self.eps = 1.0e-5
        self.register_buffer(
            "running_mean",
            torch.zeros(self.feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "running_variance",
            torch.ones(self.feature_dim, dtype=torch.float32),
        )
        self.register_buffer(
            "batches_seen",
            torch.zeros((), dtype=torch.long),
        )
        self.output = nn.Linear(self.feature_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.nonlinear_adapter: nn.Sequential | None
        if self.nonlinear_hidden_dim > 0:
            self.nonlinear_adapter = nn.Sequential(
                nn.Linear(
                    self.feature_dim,
                    self.nonlinear_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(
                    self.nonlinear_hidden_dim,
                    self.nonlinear_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(self.nonlinear_hidden_dim, 1),
            )
            nn.init.zeros_(self.nonlinear_adapter[-1].weight)
            nn.init.zeros_(self.nonlinear_adapter[-1].bias)
        else:
            self.nonlinear_adapter = None

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        if self.training:
            mean = features.mean(dim=0)
            variance = features.var(dim=0, unbiased=False)
            with torch.no_grad():
                if int(self.batches_seen) == 0:
                    self.running_mean.copy_(mean)
                    self.running_variance.copy_(variance)
                else:
                    self.running_mean.lerp_(mean, self.momentum)
                    self.running_variance.lerp_(variance, self.momentum)
                self.batches_seen.add_(1)
        else:
            mean = self.running_mean
            variance = self.running_variance
        return (features - mean) / torch.sqrt(variance + self.eps)

    def forward(
        self,
        *,
        direct_prediction: torch.Tensor,
        confidence: torch.Tensor,
        ood_excess: torch.Tensor,
        sign_logit: torch.Tensor,
        magnitude_prediction: torch.Tensor,
        structured_prediction: torch.Tensor,
        structured_output_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sign_probability = torch.sigmoid(sign_logit)
        disagreement = structured_prediction - direct_prediction
        features = torch.stack(
            [
                direct_prediction,
                direct_prediction.abs(),
                confidence,
                ood_excess,
                sign_logit.clamp(min=-20.0, max=20.0),
                sign_probability,
                4.0 * sign_probability * (1.0 - sign_probability),
                magnitude_prediction,
                structured_prediction,
                disagreement,
                disagreement.abs(),
                structured_output_residual,
                structured_output_residual.abs(),
            ],
            dim=-1,
        ).float()
        normalized_features = self._normalize(features)
        calibrated_residual = self.output(
            normalized_features
        ).squeeze(-1)
        if self.nonlinear_adapter is not None:
            calibrated_residual = calibrated_residual + (
                self.nonlinear_adapter(normalized_features).squeeze(-1)
            )
        return calibrated_residual, features


class DDGMajorSummaryAdapter(nn.Module):
    """Use same-network role-level Major evidence to correct signed ΔΔG‡.

    Five internal classifiers inspect the complete fusion plus the reactant,
    solvent, ligand, and metal role tokens.  Their logits are summarized
    together with the existing A6/A7 diagnostics by one bounded residual
    head.  The residual output is zero-initialized, so adding this default-off
    module to an existing checkpoint preserves its prediction exactly.
    """

    def __init__(
        self,
        *,
        fusion_dim: int,
        role_hidden_dim: int,
        num_roles: int,
        classifier_hidden_dim: int = 64,
        residual_hidden_dim: int = 32,
        role_ids: tuple[int, ...] = (0, 2, 8, 9),
        residual_multiplier: float = 1.0,
    ) -> None:
        super().__init__()
        if (
            fusion_dim <= 0
            or role_hidden_dim <= 0
            or num_roles <= 0
            or classifier_hidden_dim <= 0
            or residual_hidden_dim <= 0
        ):
            raise ValueError(
                "DDG Major-summary dimensions must be positive"
            )
        normalized_role_ids = tuple(int(value) for value in role_ids)
        if not normalized_role_ids:
            raise ValueError(
                "DDG Major-summary role IDs must not be empty"
            )
        if len(set(normalized_role_ids)) != len(normalized_role_ids):
            raise ValueError(
                "DDG Major-summary role IDs must be unique"
            )
        if any(
            value < 0 or value >= int(num_roles)
            for value in normalized_role_ids
        ):
            raise ValueError(
                "A DDG Major-summary role ID is outside the configured range"
            )
        if residual_multiplier < 0.0:
            raise ValueError(
                "DDG Major-summary residual multiplier must be non-negative"
            )

        self.fusion_dim = int(fusion_dim)
        self.role_hidden_dim = int(role_hidden_dim)
        self.num_roles = int(num_roles)
        self.classifier_hidden_dim = int(classifier_hidden_dim)
        self.residual_hidden_dim = int(residual_hidden_dim)
        self.role_ids = normalized_role_ids
        self.residual_multiplier = float(residual_multiplier)
        self.num_views = 1 + len(self.role_ids)
        self.feature_dim = 7 + 3 * self.num_views + 6

        def classifier(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, self.classifier_hidden_dim),
                nn.GELU(),
                nn.Linear(self.classifier_hidden_dim, 1),
            )

        self.fusion_classifier = classifier(self.fusion_dim)
        self.role_classifiers = nn.ModuleList(
            classifier(self.role_hidden_dim) for _ in self.role_ids
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.residual_hidden_dim),
            nn.GELU(),
            nn.Linear(self.residual_hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def forward(
        self,
        *,
        fusion: torch.Tensor,
        role_tokens: torch.Tensor,
        role_present: torch.Tensor,
        base_prediction: torch.Tensor,
        confidence: torch.Tensor,
        ood_excess: torch.Tensor,
        sign_logit: torch.Tensor,
        magnitude_prediction: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if fusion.ndim != 2 or fusion.shape[-1] != self.fusion_dim:
            raise ValueError(
                "fusion shape is incompatible with DDG Major summary"
            )
        if (
            role_tokens.ndim != 3
            or tuple(role_tokens.shape[1:])
            != (self.num_roles, self.role_hidden_dim)
        ):
            raise ValueError(
                "role_tokens shape is incompatible with DDG Major summary"
            )
        if tuple(role_present.shape) != tuple(role_tokens.shape[:2]):
            raise ValueError(
                "role_present shape is incompatible with DDG Major summary"
            )

        fusion_logit = self.fusion_classifier(fusion).squeeze(-1)
        role_logits = torch.stack(
            [
                classifier(role_tokens[:, role_id]).squeeze(-1)
                for classifier, role_id in zip(
                    self.role_classifiers,
                    self.role_ids,
                    strict=True,
                )
            ],
            dim=-1,
        )
        role_mask = role_present[:, self.role_ids].bool()
        view_logits = torch.cat(
            [fusion_logit.unsqueeze(-1), role_logits],
            dim=-1,
        )
        view_mask = torch.cat(
            [
                torch.ones_like(
                    fusion_logit.unsqueeze(-1),
                    dtype=torch.bool,
                ),
                role_mask,
            ],
            dim=-1,
        )
        mask_float = view_mask.to(view_logits.dtype)
        view_count = mask_float.sum(dim=-1).clamp_min(1.0)
        masked_logits = view_logits * mask_float
        aggregate_logit = masked_logits.sum(dim=-1) / view_count
        view_probabilities = torch.sigmoid(view_logits)
        aggregate_probability = (
            view_probabilities * mask_float
        ).sum(dim=-1) / view_count
        centered_probability = (
            view_probabilities - aggregate_probability.unsqueeze(-1)
        ) * mask_float
        probability_std = torch.sqrt(
            centered_probability.square().sum(dim=-1) / view_count
        )
        probability_min = view_probabilities.masked_fill(
            ~view_mask,
            1.0,
        ).min(dim=-1).values
        probability_max = view_probabilities.masked_fill(
            ~view_mask,
            0.0,
        ).max(dim=-1).values
        clipped_sign_logit = sign_logit.clamp(min=-20.0, max=20.0)
        sign_probability = torch.sigmoid(clipped_sign_logit)
        residual_features = torch.cat(
            [
                torch.stack(
                    [
                        base_prediction,
                        base_prediction.abs(),
                        confidence,
                        ood_excess,
                        clipped_sign_logit,
                        sign_probability,
                        magnitude_prediction,
                    ],
                    dim=-1,
                ),
                view_logits.clamp(min=-20.0, max=20.0),
                view_probabilities,
                mask_float,
                torch.stack(
                    [
                        aggregate_logit,
                        aggregate_probability,
                        probability_std,
                        probability_min,
                        probability_max,
                        aggregate_probability - sign_probability,
                    ],
                    dim=-1,
                ),
            ],
            dim=-1,
        ).float()
        if residual_features.shape[-1] != self.feature_dim:
            raise RuntimeError(
                "DDG Major-summary residual feature accounting changed"
            )
        residual = self.residual_multiplier * torch.tanh(
            self.residual_head(residual_features).squeeze(-1)
        )
        return (
            view_logits,
            view_mask,
            aggregate_logit,
            probability_std,
            residual,
        )


class DirectedRoleContextAttention(nn.Module):
    """Let one AHO role explicitly read a fixed set of partner roles.

    The residual output projection is zero-initialized. Enabling this module
    therefore preserves the parent model's function exactly at initialization,
    while making the directed interaction available as an incremental Stage-2
    design.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_roles: int,
        query_role_id: int = 0,
        context_role_ids: tuple[int, ...] = (2, 8, 9),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or num_roles <= 0:
            raise ValueError("directed role-attention dimensions must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= int(query_role_id) < int(num_roles):
            raise ValueError("query_role_id is outside the configured role range")
        normalized_context_ids = tuple(int(value) for value in context_role_ids)
        if not normalized_context_ids:
            raise ValueError("context_role_ids must contain at least one role")
        if len(set(normalized_context_ids)) != len(normalized_context_ids):
            raise ValueError("context_role_ids must be unique")
        if int(query_role_id) in normalized_context_ids:
            raise ValueError("query_role_id cannot also be a context role")
        if any(value < 0 or value >= int(num_roles) for value in normalized_context_ids):
            raise ValueError("a context role ID is outside the configured role range")

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_roles = int(num_roles)
        self.query_role_id = int(query_role_id)
        self.context_role_ids = normalized_context_ids

        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.context_norm = nn.LayerNorm(self.hidden_dim)
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        role_tokens: torch.Tensor,
        role_present: torch.Tensor,
    ) -> torch.Tensor:
        if role_tokens.ndim != 3:
            raise ValueError("role_tokens must have shape [batch, roles, hidden_dim]")
        if role_tokens.shape[1:] != (self.num_roles, self.hidden_dim):
            raise ValueError(
                "role_tokens shape is incompatible with directed role attention"
            )
        if role_present.shape != role_tokens.shape[:2]:
            raise ValueError("role_present must have shape [batch, roles]")

        batch_size = role_tokens.shape[0]
        role_present = role_present.bool()
        query_token = role_tokens[:, self.query_role_id]
        context_index = torch.tensor(
            self.context_role_ids,
            dtype=torch.long,
            device=role_tokens.device,
        )
        context_tokens = role_tokens.index_select(1, context_index)
        context_present = role_present.index_select(1, context_index)

        query = self.q_proj(self.query_norm(query_token))
        normalized_context = self.context_norm(context_tokens)
        key = self.k_proj(normalized_context)
        value = self.v_proj(normalized_context)
        query = query.view(batch_size, self.num_heads, 1, self.head_dim)
        key = key.view(
            batch_size,
            len(self.context_role_ids),
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            len(self.context_role_ids),
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)).squeeze(-2)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~context_present[:, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = weights * context_present[:, None, :].to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = self.attn_dropout(weights)

        attended = torch.matmul(weights.unsqueeze(-2), value).squeeze(-2)
        attended = attended.reshape(batch_size, self.hidden_dim)
        valid_interaction = (
            role_present[:, self.query_role_id] & context_present.any(dim=1)
        )
        delta = self.residual_dropout(self.out_proj(attended))
        delta = delta * valid_interaction.unsqueeze(-1).to(delta.dtype)
        query_selector = (
            torch.arange(self.num_roles, device=role_tokens.device)
            == self.query_role_id
        ).to(role_tokens.dtype)
        return role_tokens + delta[:, None, :] * query_selector[None, :, None]


class ReactionPairTransformerResidualAdapter(nn.Module):
    """Model explicit AHO role pairs after the shared role Transformer.

    The adapter builds the six unordered pair tokens among reactant, solvent,
    ligand, and metal, then lets one compact Transformer jointly contextualize
    the four role tokens and six pair tokens. Its final projection is exactly
    zero initialized, so enabling this default-off Stage-2 head preserves the
    parent prediction while leaving the fused backbone untouched.
    """

    pair_indices = (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    )

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_roles: int,
        role_ids: tuple[int, ...] = (0, 2, 8, 9),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        normalized_role_ids = tuple(int(value) for value in role_ids)
        if hidden_dim <= 0 or num_heads <= 0 or num_layers <= 0:
            raise ValueError(
                "reaction-pair Transformer dimensions must be positive"
            )
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if len(normalized_role_ids) != 4:
            raise ValueError(
                "reaction-pair Transformer requires exactly four role IDs"
            )
        if len(set(normalized_role_ids)) != len(normalized_role_ids):
            raise ValueError("reaction-pair Transformer role IDs must be unique")
        if any(
            value < 0 or value >= int(num_roles)
            for value in normalized_role_ids
        ):
            raise ValueError(
                "a reaction-pair Transformer role ID is out of range"
            )
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.role_ids = normalized_role_ids
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, self.hidden_dim)
        )
        self.role_slot_embedding = nn.Parameter(
            torch.empty(1, len(self.role_ids), self.hidden_dim)
        )
        self.pair_type_embedding = nn.Parameter(
            torch.empty(1, len(self.pair_indices), self.hidden_dim)
        )
        nn.init.normal_(self.role_slot_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.pair_type_embedding, mean=0.0, std=0.02)
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=self.num_layers,
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(
        self,
        role_tokens: torch.Tensor,
        role_present: torch.Tensor,
    ) -> torch.Tensor:
        if role_tokens.ndim != 3:
            raise ValueError(
                "role_tokens must have shape [batch, num_roles, hidden_dim]"
            )
        if role_present.shape != role_tokens.shape[:2]:
            raise ValueError("role presence and tokens are not aligned")
        if role_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected hidden dim {self.hidden_dim}, got "
                f"{role_tokens.shape[-1]}"
            )
        role_index = torch.tensor(
            self.role_ids,
            dtype=torch.long,
            device=role_tokens.device,
        )
        selected = role_tokens.index_select(1, role_index)
        selected_present = role_present.index_select(1, role_index).bool()
        role_sequence = selected + self.role_slot_embedding.to(
            selected.dtype
        )

        pair_parts: list[torch.Tensor] = []
        pair_presence: list[torch.Tensor] = []
        for left_index, right_index in self.pair_indices:
            left = selected[:, left_index]
            right = selected[:, right_index]
            pair_parts.append(
                torch.cat(
                    [
                        left,
                        right,
                        left * right,
                        torch.abs(left - right),
                    ],
                    dim=-1,
                )
            )
            pair_presence.append(
                selected_present[:, left_index]
                & selected_present[:, right_index]
            )
        pair_sequence = self.pair_projection(
            torch.stack(pair_parts, dim=1)
        )
        pair_sequence = pair_sequence + self.pair_type_embedding.to(
            pair_sequence.dtype
        )
        pair_present = torch.stack(pair_presence, dim=1)
        cls = self.cls_token.expand(role_tokens.shape[0], -1, -1).to(
            role_tokens.dtype
        )
        sequence = torch.cat(
            [cls, role_sequence, pair_sequence],
            dim=1,
        )
        sequence_present = torch.cat(
            [
                torch.ones(
                    (role_tokens.shape[0], 1),
                    dtype=torch.bool,
                    device=role_tokens.device,
                ),
                selected_present,
                pair_present,
            ],
            dim=1,
        )
        encoded = self.encoder(
            sequence,
            src_key_padding_mask=~sequence_present,
        )
        return self.output_head(encoded[:, 0]).squeeze(-1)


class SiteLigandMetalInteractionAttention(nn.Module):
    """Build an explicit site -> ligand -> metal interaction residual.

    Reactant site/shell atom tokens first read ligand donor/chiral tokens. The
    resulting site-ligand context then reads metal tokens. A shared interaction
    projection produces a residual only for the site/shell tokens. The final
    projection is zero-initialized so enabling A3 exactly preserves its A2
    parent at initialization.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_token_types: int,
        site_token_type_ids: tuple[int, ...] = (1, 2, 3),
        ligand_token_type_ids: tuple[int, ...] = (6, 7),
        metal_token_type_ids: tuple[int, ...] = (9,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or num_token_types <= 0:
            raise ValueError("site interaction dimensions must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        def normalize_ids(name: str, values: tuple[int, ...]) -> tuple[int, ...]:
            normalized = tuple(int(value) for value in values)
            if not normalized:
                raise ValueError(f"{name} must contain at least one token type")
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{name} must contain unique token types")
            if any(value < 0 or value >= int(num_token_types) for value in normalized):
                raise ValueError(f"{name} contains an out-of-range token type")
            return normalized

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads
        self.num_token_types = int(num_token_types)
        self.site_token_type_ids = normalize_ids(
            "site_token_type_ids",
            site_token_type_ids,
        )
        self.ligand_token_type_ids = normalize_ids(
            "ligand_token_type_ids",
            ligand_token_type_ids,
        )
        self.metal_token_type_ids = normalize_ids(
            "metal_token_type_ids",
            metal_token_type_ids,
        )
        all_focus_ids = (
            self.site_token_type_ids
            + self.ligand_token_type_ids
            + self.metal_token_type_ids
        )
        if len(set(all_focus_ids)) != len(all_focus_ids):
            raise ValueError("site, ligand, and metal token-type sets must be disjoint")

        self.site_norm = nn.LayerNorm(self.hidden_dim)
        self.ligand_norm = nn.LayerNorm(self.hidden_dim)
        self.metal_query_norm = nn.LayerNorm(self.hidden_dim)
        self.metal_norm = nn.LayerNorm(self.hidden_dim)
        self.ligand_q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ligand_k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.ligand_v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.metal_q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.metal_k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.metal_v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.interaction_norm = nn.LayerNorm(self.hidden_dim * 4)
        self.interaction_proj = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.interaction_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _type_mask(
        token_type_ids: torch.Tensor,
        allowed_ids: tuple[int, ...],
    ) -> torch.Tensor:
        allowed = torch.tensor(
            allowed_ids,
            dtype=torch.long,
            device=token_type_ids.device,
        )
        return (token_type_ids.long().unsqueeze(-1) == allowed).any(dim=-1)

    def _cross_attention(
        self,
        *,
        query_input: torch.Tensor,
        context_input: torch.Tensor,
        context_mask: torch.Tensor,
        query_projection: nn.Linear,
        key_projection: nn.Linear,
        value_projection: nn.Linear,
    ) -> torch.Tensor:
        batch_size, num_tokens, _ = query_input.shape
        query = query_projection(query_input).view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key_projection(context_input).view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value_projection(context_input).view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~context_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = weights * context_mask[:, None, None, :].to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = self.attn_dropout(weights)
        attended = torch.matmul(weights, value)
        return attended.transpose(1, 2).reshape(
            batch_size,
            num_tokens,
            self.hidden_dim,
        )

    def forward(
        self,
        atom_tokens: torch.Tensor,
        token_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if atom_tokens.ndim != 3:
            raise ValueError("atom_tokens must have shape [batch, tokens, hidden_dim]")
        if atom_tokens.shape[-1] != self.hidden_dim:
            raise ValueError("atom_tokens has the wrong hidden dimension")
        if token_mask.shape != atom_tokens.shape[:2]:
            raise ValueError("token_mask must have shape [batch, tokens]")
        if token_type_ids.shape != atom_tokens.shape[:2]:
            raise ValueError("token_type_ids must have shape [batch, tokens]")

        token_mask = token_mask.bool()
        site_mask = (
            self._type_mask(token_type_ids, self.site_token_type_ids) & token_mask
        )
        ligand_mask = (
            self._type_mask(token_type_ids, self.ligand_token_type_ids) & token_mask
        )
        metal_mask = (
            self._type_mask(token_type_ids, self.metal_token_type_ids) & token_mask
        )

        ligand_context = self._cross_attention(
            query_input=self.site_norm(atom_tokens),
            context_input=self.ligand_norm(atom_tokens),
            context_mask=ligand_mask,
            query_projection=self.ligand_q_proj,
            key_projection=self.ligand_k_proj,
            value_projection=self.ligand_v_proj,
        )
        metal_query = self.metal_query_norm(atom_tokens + ligand_context)
        metal_context = self._cross_attention(
            query_input=metal_query,
            context_input=self.metal_norm(atom_tokens),
            context_mask=metal_mask,
            query_projection=self.metal_q_proj,
            key_projection=self.metal_k_proj,
            value_projection=self.metal_v_proj,
        )
        interaction = torch.cat(
            [
                atom_tokens * ligand_context,
                torch.abs(atom_tokens - ligand_context),
                ligand_context * metal_context,
                torch.abs(ligand_context - metal_context),
            ],
            dim=-1,
        )
        interaction = torch.nn.functional.gelu(
            self.interaction_proj(self.interaction_norm(interaction))
        )
        interaction = self.interaction_dropout(interaction)
        valid_interaction = (
            site_mask
            & ligand_mask.any(dim=1, keepdim=True)
            & metal_mask.any(dim=1, keepdim=True)
        )
        delta = self.residual_dropout(self.out_proj(interaction))
        delta = delta * valid_interaction.unsqueeze(-1).to(delta.dtype)
        return atom_tokens + delta


class SiteDonorRelationEncoderLayer(nn.Module):
    """Self-attention layer with an explicit per-pair chemistry bias."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        relation_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or relation_dim <= 0:
            raise ValueError("relation-attention dimensions must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_dim // self.num_heads

        self.attention_norm = nn.LayerNorm(self.hidden_dim)
        self.qkv = nn.Linear(self.hidden_dim, self.hidden_dim * 3)
        self.relation_bias = nn.Sequential(
            nn.LayerNorm(int(relation_dim)),
            nn.Linear(int(relation_dim), self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.num_heads),
        )
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        relation: torch.Tensor,
        pair_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("relation tokens must have shape [batch, tokens, hidden]")
        if relation.shape[:3] != (
            tokens.shape[0],
            tokens.shape[1],
            tokens.shape[1],
        ):
            raise ValueError("relation tensor is not aligned to relation tokens")
        if pair_mask.shape != relation.shape[:3]:
            raise ValueError("relation pair mask has an incompatible shape")
        if token_mask.shape != tokens.shape[:2]:
            raise ValueError("relation token mask has an incompatible shape")

        batch_size, sequence_length, _ = tokens.shape
        normalized = self.attention_norm(tokens)
        qkv = self.qkv(normalized).view(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        bias = self.relation_bias(relation).permute(0, 3, 1, 2)
        scores = scores + bias
        pair_mask = pair_mask.bool()
        scores = scores.masked_fill(
            ~pair_mask[:, None, :, :],
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores, dim=-1)
        weights = weights * pair_mask[:, None, :, :].to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = self.attention_dropout(weights)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            self.hidden_dim,
        )
        tokens = tokens + self.residual_dropout(self.out_proj(attended))
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens * token_mask.unsqueeze(-1).to(tokens.dtype)


class SiteDonorRelationResidualAdapter(nn.Module):
    """Single-network residual from chemistry-constrained atom relations.

    The branch uses same-component 3D distance bias and permits cross-component
    attention only between the reactant site/shell, ligand donor/chiral atoms,
    metal, and solvent contexts.  Its final scalar projection is zero
    initialized, so enabling the adapter is exactly function preserving before
    training and does not alter the existing ORRA fusion topology.
    """

    relation_dim = 33

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        num_roles: int,
        num_token_types: int,
        dropout: float,
        reactant_role_id: int = 0,
        solvent_role_id: int = 2,
        ligand_role_id: int = 8,
        metal_role_id: int = 9,
        global_token_type_id: int = 0,
        site_token_type_ids: tuple[int, ...] = (1, 2, 3),
        ligand_focus_token_type_ids: tuple[int, ...] = (6, 7),
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("site-donor relation attention needs at least one layer")
        role_values = (
            int(reactant_role_id),
            int(solvent_role_id),
            int(ligand_role_id),
            int(metal_role_id),
        )
        if len(set(role_values)) != len(role_values):
            raise ValueError("site-donor relation role IDs must be distinct")
        if any(value < 0 or value >= int(num_roles) for value in role_values):
            raise ValueError("a site-donor relation role ID is out of range")
        site_values = tuple(int(value) for value in site_token_type_ids)
        ligand_values = tuple(int(value) for value in ligand_focus_token_type_ids)
        if not site_values or not ligand_values:
            raise ValueError("site and ligand-focus token-type sets must be nonempty")
        all_type_values = (
            int(global_token_type_id),
            *site_values,
            *ligand_values,
        )
        if any(
            value < 0 or value >= int(num_token_types)
            for value in all_type_values
        ):
            raise ValueError("a site-donor relation token-type ID is out of range")

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.reactant_role_id = role_values[0]
        self.solvent_role_id = role_values[1]
        self.ligand_role_id = role_values[2]
        self.metal_role_id = role_values[3]
        self.global_token_type_id = int(global_token_type_id)
        self.site_token_type_ids = site_values
        self.ligand_focus_token_type_ids = ligand_values
        self.rbf_width = 0.45
        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.0, 3.0, 8),
            persistent=False,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.layers = nn.ModuleList(
            [
                SiteDonorRelationEncoderLayer(
                    hidden_dim=self.hidden_dim,
                    num_heads=num_heads,
                    relation_dim=self.relation_dim,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.pool_score = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)
        self.residual_coupling_logit = nn.Parameter(torch.zeros(()))

    @staticmethod
    def _matches_any(
        values: torch.Tensor,
        allowed_values: tuple[int, ...],
    ) -> torch.Tensor:
        allowed = torch.tensor(
            allowed_values,
            dtype=torch.long,
            device=values.device,
        )
        return (values.long().unsqueeze(-1) == allowed).any(dim=-1)

    @staticmethod
    def _unordered_pair(
        left: torch.Tensor,
        right: torch.Tensor,
        valid_pair: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (left[:, :, None] & right[:, None, :])
            | (right[:, :, None] & left[:, None, :])
        ) & valid_pair

    def relation_features(
        self,
        numeric: torch.Tensor,
        token_mask: torch.Tensor,
        role_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        component_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if numeric.shape[:2] != token_mask.shape:
            raise ValueError("numeric and token mask are not aligned")
        for name, values in (
            ("role IDs", role_ids),
            ("token-type IDs", token_type_ids),
            ("component IDs", component_ids),
        ):
            if values.shape != token_mask.shape:
                raise ValueError(f"{name} and token mask are not aligned")

        token_mask = token_mask.bool()
        role_ids = role_ids.long()
        token_type_ids = token_type_ids.long()
        component_ids = component_ids.long()
        valid_pair = token_mask[:, :, None] & token_mask[:, None, :]
        same_component = (
            (component_ids[:, :, None] == component_ids[:, None, :])
            & (component_ids[:, :, None] >= 0)
            & valid_pair
        )
        cross_component = valid_pair & ~same_component
        same_role = (
            role_ids[:, :, None] == role_ids[:, None, :]
        ) & valid_pair
        same_token_type = (
            token_type_ids[:, :, None] == token_type_ids[:, None, :]
        ) & valid_pair

        reactant = role_ids == self.reactant_role_id
        solvent = role_ids == self.solvent_role_id
        ligand = role_ids == self.ligand_role_id
        metal = role_ids == self.metal_role_id
        global_token = token_type_ids == self.global_token_type_id
        site = token_type_ids == self.site_token_type_ids[0]
        site_shell = self._matches_any(
            token_type_ids,
            self.site_token_type_ids,
        )
        ligand_focus = self._matches_any(
            token_type_ids,
            self.ligand_focus_token_type_ids,
        )
        reactant_context = site_shell | (reactant & global_token)
        ligand_context = ligand_focus | (ligand & global_token)
        metal_context = metal
        solvent_context = solvent
        focus = (
            reactant_context
            | ligand_context
            | metal_context
            | solvent_context
        ) & token_mask

        def pair(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
            return self._unordered_pair(left, right, valid_pair)

        role_pair_flags = [
            pair(site, ligand_focus),
            pair(site_shell, ligand_focus),
            pair(site, ligand),
            pair(site_shell, ligand),
            pair(site, metal),
            pair(site_shell, metal),
            pair(ligand_focus, metal),
            pair(
                token_type_ids == self.ligand_focus_token_type_ids[-1],
                site,
            ),
            pair(solvent, ligand),
            pair(solvent, metal),
            pair(reactant, ligand),
            pair(reactant, metal),
        ]
        constrained_cross = (
            pair(reactant_context, ligand_context)
            | pair(reactant_context, metal_context)
            | pair(reactant_context, solvent_context)
            | pair(ligand_context, metal_context)
            | pair(ligand_context, solvent_context)
            | pair(metal_context, solvent_context)
        ) & cross_component
        allowed_pair = same_component | constrained_cross

        position = numeric[..., :3].float()
        delta = position[:, :, None, :] - position[:, None, :, :]
        distance = torch.sqrt(
            torch.clamp((delta * delta).sum(dim=-1), min=0.0) + 1e-8
        )
        distance = distance * same_component.to(distance.dtype)
        rbf = torch.exp(
            -0.5
            * (
                (
                    distance[..., None]
                    - self.rbf_centers.to(distance.device)
                )
                / self.rbf_width
            )
            ** 2
        )
        rbf = rbf * same_component[..., None].to(rbf.dtype)
        eye = torch.eye(
            token_mask.shape[1],
            dtype=torch.bool,
            device=token_mask.device,
        ).unsqueeze(0)
        relation_body = torch.cat(
            [
                valid_pair[..., None].float(),
                same_component[..., None].float(),
                cross_component[..., None].float(),
                allowed_pair[..., None].float(),
                same_role[..., None].float(),
                same_token_type[..., None].float(),
                eye.expand_as(valid_pair)[..., None].float(),
                pair(focus, focus)[..., None].float(),
                pair(reactant_context, ligand_context)[..., None].float(),
                pair(ligand_context, metal_context)[..., None].float(),
                pair(reactant_context, metal_context)[..., None].float(),
                pair(solvent_context, focus)[..., None].float(),
                *[flag[..., None].float() for flag in role_pair_flags],
                distance[..., None],
                rbf,
            ],
            dim=-1,
        )
        if relation_body.shape[-1] != self.relation_dim:
            raise RuntimeError("site-donor relation feature dimension changed")

        batch_size, num_tokens = token_mask.shape
        full_relation = relation_body.new_zeros(
            batch_size,
            num_tokens + 1,
            num_tokens + 1,
            self.relation_dim,
        )
        full_relation[:, 1:, 1:, :] = relation_body
        full_token_mask = torch.cat(
            [
                torch.ones(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=token_mask.device,
                ),
                token_mask,
            ],
            dim=1,
        )
        full_pair_mask = torch.zeros(
            batch_size,
            num_tokens + 1,
            num_tokens + 1,
            dtype=torch.bool,
            device=token_mask.device,
        )
        full_pair_mask[:, 0, :] = full_token_mask
        full_pair_mask[:, :, 0] = full_token_mask
        full_pair_mask[:, 1:, 1:] = allowed_pair
        return full_relation, full_pair_mask, full_token_mask, focus

    def forward(
        self,
        atom_tokens: torch.Tensor,
        numeric: torch.Tensor,
        token_mask: torch.Tensor,
        role_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        component_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        relation, pair_mask, full_token_mask, focus = self.relation_features(
            numeric,
            token_mask,
            role_ids,
            token_type_ids,
            component_ids,
        )
        cls = self.cls_token.expand(atom_tokens.shape[0], -1, -1)
        encoded = torch.cat([cls, atom_tokens], dim=1)
        for layer in self.layers:
            encoded = layer(
                encoded,
                relation,
                pair_mask,
                full_token_mask,
            )
        body = encoded[:, 1:]
        has_focus = focus.any(dim=1, keepdim=True)
        pool_mask = torch.where(has_focus, focus, token_mask.bool())
        pooled = masked_attention_pool(body, pool_mask, self.pool_score)
        relation_logit = self.output_head(
            torch.cat([encoded[:, 0], pooled], dim=-1)
        ).squeeze(-1)
        coupling = 2.0 * torch.sigmoid(self.residual_coupling_logit)
        residual = coupling * relation_logit
        return (
            relation_logit,
            residual,
            coupling.expand_as(relation_logit),
        )


class SMILESSequenceResidualAdapter(nn.Module):
    """Default-off byte-SMILES complement to the atomistic ORRA fusion.

    The adapter is intentionally self contained: it receives one deterministic
    byte-token sequence and the fusion produced by the existing AMC/ORRA path.
    Its last projection is zero initialized, so adding the module to a parent
    checkpoint preserves that parent's prediction exactly before training.
    """

    def __init__(
        self,
        *,
        fusion_dim: int,
        vocab_size: int = 259,
        max_length: int = 320,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        residual_multiplier: float = 3.0,
        use_auxiliary_head: bool = False,
    ) -> None:
        super().__init__()
        if fusion_dim <= 0:
            raise ValueError("SMILES sequence fusion dimension must be positive")
        if vocab_size <= 258:
            raise ValueError(
                "SMILES byte vocabulary must contain PAD, bytes, CLS, and SEP"
            )
        if max_length <= 3:
            raise ValueError("SMILES sequence max length must exceed three")
        if hidden_dim <= 0 or hidden_dim % num_heads != 0:
            raise ValueError(
                "SMILES sequence hidden dimension must be positive and "
                "divisible by the number of heads"
            )
        if num_layers <= 0:
            raise ValueError("SMILES sequence Transformer needs at least one layer")
        if residual_multiplier < 0.0:
            raise ValueError(
                "SMILES sequence residual multiplier must be non-negative"
            )
        self.fusion_dim = int(fusion_dim)
        self.vocab_size = int(vocab_size)
        self.max_length = int(max_length)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.residual_multiplier = float(residual_multiplier)
        self.use_auxiliary_head = bool(use_auxiliary_head)

        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.hidden_dim,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(
            self.max_length,
            self.hidden_dim,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )
        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        pooled_dim = self.hidden_dim * 6
        self.sequence_projection = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.fusion_projection = nn.Sequential(
            nn.LayerNorm(self.fusion_dim),
            nn.Linear(self.fusion_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.auxiliary_head: nn.Sequential | None
        if self.use_auxiliary_head:
            self.auxiliary_head = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, 1),
            )
        else:
            self.auxiliary_head = None

    def auxiliary_prediction(
        self,
        sequence_summary: torch.Tensor,
    ) -> torch.Tensor | None:
        """Predict the training-only task target from the sequence view."""
        if self.auxiliary_head is None:
            return None
        return self.auxiliary_head(sequence_summary).squeeze(-1)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        fusion: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2:
            raise ValueError(
                "SMILES sequence token IDs must have shape [batch, length]"
            )
        if token_mask.shape != token_ids.shape:
            raise ValueError(
                "SMILES sequence mask must match the token-ID tensor"
            )
        if token_ids.shape[1] > self.max_length:
            raise ValueError(
                "SMILES sequence length exceeds the configured maximum"
            )
        if fusion.ndim != 2 or fusion.shape != (
            token_ids.shape[0],
            self.fusion_dim,
        ):
            raise ValueError(
                "SMILES sequence fusion tensor has an incompatible shape"
            )
        token_mask = token_mask.bool()
        if torch.any(~token_mask.any(dim=1)):
            raise ValueError("Every SMILES sequence must contain at least one token")
        token_ids = token_ids.long()
        if torch.any(token_ids < 0) or torch.any(token_ids >= self.vocab_size):
            raise ValueError("SMILES sequence token ID is outside the vocabulary")
        if torch.any((token_ids == 0) & token_mask):
            raise ValueError("PAD token cannot be marked as present")
        if torch.any((token_ids != 0) & ~token_mask):
            raise ValueError("Masked SMILES positions must contain PAD")

        positions = torch.arange(
            token_ids.shape[1],
            device=token_ids.device,
        ).unsqueeze(0)
        encoded = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )
        encoded = encoded * token_mask.unsqueeze(-1).to(encoded.dtype)
        encoded = self.encoder(
            encoded,
            src_key_padding_mask=~token_mask,
        )
        encoded = encoded * token_mask.unsqueeze(-1).to(encoded.dtype)

        convolution_input = encoded.transpose(1, 2)
        pooled = []
        valid_count = token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        for convolution in self.convolutions:
            values = torch.nn.functional.gelu(
                convolution(convolution_input)
            ).transpose(1, 2)
            values = values * token_mask.unsqueeze(-1).to(values.dtype)
            mean_pool = values.sum(dim=1) / valid_count.to(values.dtype)
            max_pool = values.masked_fill(
                ~token_mask.unsqueeze(-1),
                torch.finfo(values.dtype).min,
            ).amax(dim=1)
            pooled.extend([mean_pool, max_pool])
        sequence_summary = self.sequence_projection(
            torch.cat(pooled, dim=-1)
        )
        fusion_summary = self.fusion_projection(fusion)
        interaction = torch.cat(
            [
                sequence_summary,
                fusion_summary,
                sequence_summary * fusion_summary,
                torch.abs(sequence_summary - fusion_summary),
            ],
            dim=-1,
        )
        residual_logit = self.residual_head(interaction).squeeze(-1)
        residual = self.residual_multiplier * torch.tanh(residual_logit)
        return residual, sequence_summary


class UnifiedORRAModel(nn.Module):
    """One ORRA readout used unchanged for AHO classification and all regressions.

    The default output is a confidence-gated residual around a train-only prior:

        prediction = prior + confidence * (raw_prediction - prior)

    The centered-OOD variant leaves supported samples untouched and, only
    outside train support, shifts the prediction centre toward a train-only
    prior while retaining a confidence-scaled residual around the global
    training centre.  It therefore corrects group-level OOD offsets without
    discarding the ordering learned by the attention readout.

    A fixed monotonic OOD term lowers confidence when component density moves
    outside the training support.  No ensemble or task-specific expert exists.
    """

    def __init__(
        self,
        *,
        orb_dim: int,
        numeric_dim: int,
        density_dim: int,
        hidden_dim: int,
        num_roles: int,
        num_token_types: int,
        num_self_layers: int,
        num_role_layers: int,
        num_heads: int,
        dropout: float,
        prior: float,
        regression_head_hidden_dims: tuple[int, ...] | None = None,
        regression_head_dropout: float | None = None,
        use_token_type_cross_attention: bool = False,
        ood_threshold: float = 1.0,
        ood_scale_init: float = 0.2,
        exponential_ood_gate: bool = False,
        gate_residual_mode: str = "convex",
        ood_shift_scale: float = 1.0,
        ood_base_shift: float = 0.0,
        residual_global_center_mix: float = 1.0,
        use_grouped_statistics: bool = False,
        use_cross_view_interactions: bool = False,
        use_conformer_attention: bool = False,
        conformer_attention_hidden_dim: int = 64,
        use_directed_role_attention: bool = False,
        directed_query_role_id: int = 0,
        directed_context_role_ids: tuple[int, ...] = (2, 8, 9),
        use_reaction_pair_transformer: bool = False,
        reaction_pair_transformer_num_layers: int = 2,
        reaction_pair_role_ids: tuple[int, ...] = (0, 2, 8, 9),
        use_site_ligand_metal_attention: bool = False,
        site_token_type_ids: tuple[int, ...] = (1, 2, 3),
        ligand_token_type_ids: tuple[int, ...] = (6, 7),
        metal_token_type_ids: tuple[int, ...] = (9,),
        use_site_donor_relation_attention: bool = False,
        site_donor_relation_num_layers: int = 2,
        use_multi_level_orb_attention: bool = False,
        multi_level_orb_num_levels: int = 3,
        multi_level_orb_score_hidden_dim: int = 64,
        multi_level_orb_stack_indices: tuple[int, ...] = (4, 9, 14),
        use_dual_backbone_feature_fusion: bool = False,
        dual_backbone_feature_fusion_hidden_dim: int = 64,
        use_dual_backbone_feature_role_gate: bool = False,
        use_cip_orientation_embedding: bool = False,
        aho_consistency_mode: str = "none",
        aho_consistency_hidden_dim: int = 128,
        aho_consistency_auxiliary_zero_point: float = 0.0,
        use_ddg_sign_magnitude: bool = False,
        ddg_sign_magnitude_hidden_dim: int = 128,
        ddg_sign_magnitude_target_zero_point: float = 0.0,
        ddg_sign_magnitude_residual_multiplier: float = 1.0,
        use_ddg_residual_calibrator: bool = False,
        ddg_residual_calibrator_nonlinear_hidden_dim: int = 0,
        use_ddg_major_summary: bool = False,
        ddg_major_summary_classifier_hidden_dim: int = 64,
        ddg_major_summary_residual_hidden_dim: int = 32,
        ddg_major_summary_role_ids: tuple[int, ...] = (0, 2, 8, 9),
        ddg_major_summary_residual_multiplier: float = 1.0,
        use_smiles_sequence_adapter: bool = False,
        smiles_sequence_vocab_size: int = 259,
        smiles_sequence_max_length: int = 320,
        smiles_sequence_hidden_dim: int = 128,
        smiles_sequence_num_layers: int = 2,
        smiles_sequence_num_heads: int = 4,
        smiles_sequence_dropout: float = 0.1,
        smiles_sequence_residual_multiplier: float = 3.0,
        use_smiles_sequence_auxiliary_head: bool = False,
    ) -> None:
        super().__init__()
        self.num_roles = num_roles
        self.num_token_types = num_token_types
        self.density_dim = density_dim
        self.regression_head_hidden_dims = (
            None
            if regression_head_hidden_dims is None
            else tuple(regression_head_hidden_dims)
        )
        if (
            self.regression_head_hidden_dims is not None
            and any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.regression_head_hidden_dims
            )
        ):
            raise ValueError(
                "Regression-head hidden dimensions must be positive integers"
            )
        self.regression_head_dropout = (
            float(dropout)
            if regression_head_dropout is None
            else float(regression_head_dropout)
        )
        if not 0.0 <= self.regression_head_dropout < 1.0:
            raise ValueError("Regression-head dropout must be in [0, 1)")
        self.use_token_type_cross_attention = bool(use_token_type_cross_attention)
        self.exponential_ood_gate = bool(exponential_ood_gate)
        if gate_residual_mode not in {"convex", "centered_ood"}:
            raise ValueError(
                "gate_residual_mode must be one of {'convex', 'centered_ood'}"
            )
        if float(ood_shift_scale) < 0.0:
            raise ValueError("ood_shift_scale must be non-negative")
        if not 0.0 <= float(ood_base_shift) <= 1.0:
            raise ValueError("ood_base_shift must be in [0, 1]")
        if not 0.0 <= float(residual_global_center_mix) <= 1.0:
            raise ValueError("residual_global_center_mix must be in [0, 1]")
        self.gate_residual_mode = str(gate_residual_mode)
        self.use_grouped_statistics = bool(use_grouped_statistics)
        self.use_cross_view_interactions = bool(use_cross_view_interactions)
        self.use_conformer_attention = bool(use_conformer_attention)
        self.conformer_attention_hidden_dim = int(conformer_attention_hidden_dim)
        self.use_directed_role_attention = bool(use_directed_role_attention)
        self.directed_query_role_id = int(directed_query_role_id)
        self.directed_context_role_ids = tuple(
            int(value) for value in directed_context_role_ids
        )
        self.use_reaction_pair_transformer = bool(
            use_reaction_pair_transformer
        )
        self.reaction_pair_transformer_num_layers = int(
            reaction_pair_transformer_num_layers
        )
        self.reaction_pair_role_ids = tuple(
            int(value) for value in reaction_pair_role_ids
        )
        if (
            self.use_reaction_pair_transformer
            and self.reaction_pair_transformer_num_layers <= 0
        ):
            raise ValueError(
                "reaction_pair_transformer_num_layers must be positive"
            )
        self.use_site_ligand_metal_attention = bool(
            use_site_ligand_metal_attention
        )
        self.site_token_type_ids = tuple(int(value) for value in site_token_type_ids)
        self.ligand_token_type_ids = tuple(
            int(value) for value in ligand_token_type_ids
        )
        self.metal_token_type_ids = tuple(
            int(value) for value in metal_token_type_ids
        )
        self.use_site_donor_relation_attention = bool(
            use_site_donor_relation_attention
        )
        self.site_donor_relation_num_layers = int(
            site_donor_relation_num_layers
        )
        if (
            self.use_site_donor_relation_attention
            and self.site_donor_relation_num_layers <= 0
        ):
            raise ValueError(
                "site_donor_relation_num_layers must be positive"
            )
        self.use_multi_level_orb_attention = bool(
            use_multi_level_orb_attention
        )
        self.multi_level_orb_num_levels = int(multi_level_orb_num_levels)
        self.multi_level_orb_score_hidden_dim = int(
            multi_level_orb_score_hidden_dim
        )
        self.multi_level_orb_stack_indices = tuple(
            int(value) for value in multi_level_orb_stack_indices
        )
        self.use_dual_backbone_feature_fusion = bool(
            use_dual_backbone_feature_fusion
        )
        self.dual_backbone_feature_fusion_hidden_dim = int(
            dual_backbone_feature_fusion_hidden_dim
        )
        self.use_dual_backbone_feature_role_gate = bool(
            use_dual_backbone_feature_role_gate
        )
        self.use_cip_orientation_embedding = bool(
            use_cip_orientation_embedding
        )
        if (
            self.use_multi_level_orb_attention
            and len(self.multi_level_orb_stack_indices)
            != self.multi_level_orb_num_levels
        ):
            raise ValueError(
                "multi_level_orb_stack_indices length must equal "
                "multi_level_orb_num_levels"
            )
        if (
            self.use_dual_backbone_feature_fusion
            and self.dual_backbone_feature_fusion_hidden_dim <= 0
        ):
            raise ValueError(
                "dual backbone feature fusion hidden dimension must be positive"
            )
        if (
            self.use_dual_backbone_feature_role_gate
            and not self.use_dual_backbone_feature_fusion
        ):
            raise ValueError(
                "dual backbone feature role gate requires dual backbone "
                "feature fusion"
            )
        if (
            self.use_dual_backbone_feature_fusion
            and self.use_multi_level_orb_attention
        ):
            raise ValueError(
                "dual backbone feature fusion and multi-level Orb attention "
                "are mutually exclusive"
            )
        if aho_consistency_mode not in {
            "none",
            "major_with_ddg",
            "ddg_with_major",
        }:
            raise ValueError(f"Unsupported AHO consistency mode: {aho_consistency_mode}")
        self.aho_consistency_mode = str(aho_consistency_mode)
        self.aho_consistency_hidden_dim = int(aho_consistency_hidden_dim)
        self.aho_consistency_auxiliary_zero_point = float(
            aho_consistency_auxiliary_zero_point
        )
        self.use_ddg_sign_magnitude = bool(use_ddg_sign_magnitude)
        self.ddg_sign_magnitude_hidden_dim = int(
            ddg_sign_magnitude_hidden_dim
        )
        self.ddg_sign_magnitude_target_zero_point = float(
            ddg_sign_magnitude_target_zero_point
        )
        self.ddg_sign_magnitude_residual_multiplier = float(
            ddg_sign_magnitude_residual_multiplier
        )
        self.use_ddg_residual_calibrator = bool(
            use_ddg_residual_calibrator
        )
        self.ddg_residual_calibrator_nonlinear_hidden_dim = int(
            ddg_residual_calibrator_nonlinear_hidden_dim
        )
        self.use_ddg_major_summary = bool(use_ddg_major_summary)
        self.ddg_major_summary_classifier_hidden_dim = int(
            ddg_major_summary_classifier_hidden_dim
        )
        self.ddg_major_summary_residual_hidden_dim = int(
            ddg_major_summary_residual_hidden_dim
        )
        self.ddg_major_summary_role_ids = tuple(
            int(value) for value in ddg_major_summary_role_ids
        )
        self.ddg_major_summary_residual_multiplier = float(
            ddg_major_summary_residual_multiplier
        )
        self.use_smiles_sequence_adapter = bool(
            use_smiles_sequence_adapter
        )
        self.smiles_sequence_vocab_size = int(
            smiles_sequence_vocab_size
        )
        self.smiles_sequence_max_length = int(
            smiles_sequence_max_length
        )
        self.smiles_sequence_hidden_dim = int(
            smiles_sequence_hidden_dim
        )
        self.smiles_sequence_num_layers = int(
            smiles_sequence_num_layers
        )
        self.smiles_sequence_num_heads = int(
            smiles_sequence_num_heads
        )
        self.smiles_sequence_dropout = float(
            smiles_sequence_dropout
        )
        self.smiles_sequence_residual_multiplier = float(
            smiles_sequence_residual_multiplier
        )
        self.use_smiles_sequence_auxiliary_head = bool(
            use_smiles_sequence_auxiliary_head
        )
        if (
            self.use_smiles_sequence_auxiliary_head
            and not self.use_smiles_sequence_adapter
        ):
            raise ValueError(
                "SMILES sequence auxiliary head requires the sequence adapter"
            )
        if self.ddg_residual_calibrator_nonlinear_hidden_dim < 0:
            raise ValueError(
                "DDG nonlinear calibrator hidden dimension must be "
                "non-negative"
            )
        if self.ddg_sign_magnitude_residual_multiplier < 0.0:
            raise ValueError(
                "DDG sign-magnitude residual multiplier must be non-negative"
            )
        if (
            self.use_ddg_sign_magnitude
            and self.aho_consistency_mode != "none"
        ):
            raise ValueError(
                "DDG sign-magnitude and AHO consistency adapters are "
                "mutually exclusive incremental heads"
            )
        if (
            self.use_ddg_residual_calibrator
            and not self.use_ddg_sign_magnitude
        ):
            raise ValueError(
                "DDG residual calibration requires the A6 sign-magnitude "
                "adapter"
            )
        if (
            self.ddg_residual_calibrator_nonlinear_hidden_dim > 0
            and not self.use_ddg_residual_calibrator
        ):
            raise ValueError(
                "DDG nonlinear residual calibration requires the A7 "
                "residual calibrator"
            )
        if self.use_ddg_major_summary and (
            not self.use_ddg_sign_magnitude
            or not self.use_ddg_residual_calibrator
        ):
            raise ValueError(
                "DDG Major summary requires the A6 sign-magnitude and A7 "
                "residual-calibrator parent"
            )
        if (
            self.ddg_major_summary_classifier_hidden_dim <= 0
            or self.ddg_major_summary_residual_hidden_dim <= 0
        ):
            raise ValueError(
                "DDG Major-summary hidden dimensions must be positive"
            )
        if self.ddg_major_summary_residual_multiplier < 0.0:
            raise ValueError(
                "DDG Major-summary residual multiplier must be non-negative"
            )
        self.conformer_attention_aggregator: ConformerAttentionAggregator | None
        if self.use_conformer_attention:
            self.conformer_attention_aggregator = ConformerAttentionAggregator(
                input_dim=256,
                hidden_dim=self.conformer_attention_hidden_dim,
            )
        else:
            self.conformer_attention_aggregator = None
        self.multi_level_orb_feature_attention: (
            MultiLevelOrbFeatureAttention | None
        )
        if self.use_multi_level_orb_attention:
            self.multi_level_orb_feature_attention = (
                MultiLevelOrbFeatureAttention(
                    input_dim=orb_dim,
                    num_levels=self.multi_level_orb_num_levels,
                    score_hidden_dim=self.multi_level_orb_score_hidden_dim,
                )
            )
        else:
            self.multi_level_orb_feature_attention = None
        self.dual_backbone_feature_fusion: (
            DualBackboneFeatureResidualFusion | None
        )
        if self.use_dual_backbone_feature_fusion:
            self.dual_backbone_feature_fusion = (
                DualBackboneFeatureResidualFusion(
                    input_dim=orb_dim,
                    hidden_dim=(
                        self.dual_backbone_feature_fusion_hidden_dim
                    ),
                    num_roles=(
                        self.num_roles
                        if self.use_dual_backbone_feature_role_gate
                        else None
                    ),
                )
            )
        else:
            self.dual_backbone_feature_fusion = None
        if self.use_cross_view_interactions and not self.use_token_type_cross_attention:
            raise ValueError(
                "cross-view interactions require token-type cross-attention"
            )

        self.input_norm = nn.LayerNorm(orb_dim)
        self.input_proj = nn.Linear(orb_dim, hidden_dim)
        self.numeric_proj = nn.Linear(numeric_dim, hidden_dim)
        self.role_embedding = nn.Embedding(num_roles, hidden_dim)
        self.token_type_embedding = nn.Embedding(num_token_types, hidden_dim)
        self.atomic_number_embedding = nn.Embedding(119, hidden_dim)
        self.cip_orientation_embedding: nn.Embedding | None
        if self.use_cip_orientation_embedding:
            self.cip_orientation_embedding = nn.Embedding(
                4,
                hidden_dim,
                padding_idx=0,
            )
            nn.init.zeros_(self.cip_orientation_embedding.weight)
        else:
            self.cip_orientation_embedding = None
        self.input_dropout = nn.Dropout(dropout)

        atom_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.atom_encoder = nn.TransformerEncoder(atom_layer, num_layers=num_self_layers)
        self.site_ligand_metal_attention: (
            SiteLigandMetalInteractionAttention | None
        )
        if self.use_site_ligand_metal_attention:
            self.site_ligand_metal_attention = SiteLigandMetalInteractionAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_token_types=num_token_types,
                site_token_type_ids=self.site_token_type_ids,
                ligand_token_type_ids=self.ligand_token_type_ids,
                metal_token_type_ids=self.metal_token_type_ids,
                dropout=dropout,
            )
        else:
            self.site_ligand_metal_attention = None
        self.site_donor_relation_attention: (
            SiteDonorRelationResidualAdapter | None
        )
        if self.use_site_donor_relation_attention:
            self.site_donor_relation_attention = (
                SiteDonorRelationResidualAdapter(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_layers=self.site_donor_relation_num_layers,
                    num_roles=num_roles,
                    num_token_types=num_token_types,
                    dropout=dropout,
                    reactant_role_id=0,
                    solvent_role_id=2,
                    ligand_role_id=8,
                    metal_role_id=9,
                    global_token_type_id=0,
                    site_token_type_ids=self.site_token_type_ids,
                    ligand_focus_token_type_ids=(
                        self.ligand_token_type_ids
                    ),
                )
            )
        else:
            self.site_donor_relation_attention = None
        self.role_cross_attention = RoleCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_roles=num_roles,
            dropout=dropout,
            use_grouped_statistics=use_grouped_statistics,
        )
        if self.use_token_type_cross_attention:
            self.token_type_cross_attention = RoleCrossAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_roles=num_token_types,
                dropout=dropout,
                use_grouped_statistics=use_grouped_statistics,
            )
        role_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.role_encoder = nn.TransformerEncoder(role_layer, num_layers=num_role_layers)
        self.directed_role_attention: DirectedRoleContextAttention | None
        if self.use_directed_role_attention:
            self.directed_role_attention = DirectedRoleContextAttention(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_roles=num_roles,
                query_role_id=self.directed_query_role_id,
                context_role_ids=self.directed_context_role_ids,
                dropout=dropout,
            )
        else:
            self.directed_role_attention = None
        self.reaction_pair_transformer: (
            ReactionPairTransformerResidualAdapter | None
        )
        if self.use_reaction_pair_transformer:
            self.reaction_pair_transformer = (
                ReactionPairTransformerResidualAdapter(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_layers=(
                        self.reaction_pair_transformer_num_layers
                    ),
                    num_roles=num_roles,
                    role_ids=self.reaction_pair_role_ids,
                    dropout=dropout,
                )
            )
        else:
            self.reaction_pair_transformer = None

        def pool_scorer() -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

        self.atom_pool_score = pool_scorer()
        self.role_pool_score = pool_scorer()
        if self.use_cross_view_interactions:
            self.cross_view_interaction_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim * 2),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.cross_view_interaction_pool_score = pool_scorer()
        fusion_blocks = 9 if self.use_token_type_cross_attention else 4
        if self.use_cross_view_interactions:
            fusion_blocks += 1
        fusion_dim = hidden_dim * fusion_blocks
        self.aho_consistency_adapter: AHOPhysicalConsistencyAdapter | None
        if self.aho_consistency_mode != "none":
            self.aho_consistency_adapter = AHOPhysicalConsistencyAdapter(
                fusion_dim=fusion_dim,
                hidden_dim=self.aho_consistency_hidden_dim,
                mode=self.aho_consistency_mode,
                auxiliary_zero_point=(
                    self.aho_consistency_auxiliary_zero_point
                ),
            )
        else:
            self.aho_consistency_adapter = None
        self.ddg_sign_magnitude_adapter: DDGSignMagnitudeAdapter | None
        if self.use_ddg_sign_magnitude:
            self.ddg_sign_magnitude_adapter = DDGSignMagnitudeAdapter(
                fusion_dim=fusion_dim,
                hidden_dim=self.ddg_sign_magnitude_hidden_dim,
                target_zero_point=(
                    self.ddg_sign_magnitude_target_zero_point
                ),
                residual_multiplier=(
                    self.ddg_sign_magnitude_residual_multiplier
                ),
            )
        else:
            self.ddg_sign_magnitude_adapter = None
        self.ddg_residual_calibrator: DDGResidualCalibrator | None
        if self.use_ddg_residual_calibrator:
            self.ddg_residual_calibrator = DDGResidualCalibrator(
                nonlinear_hidden_dim=(
                    self.ddg_residual_calibrator_nonlinear_hidden_dim
                ),
            )
        else:
            self.ddg_residual_calibrator = None
        self.ddg_major_summary_adapter: DDGMajorSummaryAdapter | None
        if self.use_ddg_major_summary:
            self.ddg_major_summary_adapter = DDGMajorSummaryAdapter(
                fusion_dim=fusion_dim,
                role_hidden_dim=hidden_dim,
                num_roles=num_roles,
                classifier_hidden_dim=(
                    self.ddg_major_summary_classifier_hidden_dim
                ),
                residual_hidden_dim=(
                    self.ddg_major_summary_residual_hidden_dim
                ),
                role_ids=self.ddg_major_summary_role_ids,
                residual_multiplier=(
                    self.ddg_major_summary_residual_multiplier
                ),
            )
        else:
            self.ddg_major_summary_adapter = None
        self.smiles_sequence_adapter: (
            SMILESSequenceResidualAdapter | None
        )
        if self.use_smiles_sequence_adapter:
            self.smiles_sequence_adapter = (
                SMILESSequenceResidualAdapter(
                    fusion_dim=fusion_dim,
                    vocab_size=self.smiles_sequence_vocab_size,
                    max_length=self.smiles_sequence_max_length,
                    hidden_dim=self.smiles_sequence_hidden_dim,
                    num_layers=self.smiles_sequence_num_layers,
                    num_heads=self.smiles_sequence_num_heads,
                    dropout=self.smiles_sequence_dropout,
                    residual_multiplier=(
                        self.smiles_sequence_residual_multiplier
                    ),
                    use_auxiliary_head=(
                        self.use_smiles_sequence_auxiliary_head
                    ),
                )
            )
        else:
            self.smiles_sequence_adapter = None
        if self.regression_head_hidden_dims is None:
            # Keep this literal legacy branch intact: old checkpoints depend on
            # the Linear modules remaining raw_head.1, raw_head.4, raw_head.7.
            self.raw_head = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(self.regression_head_dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.regression_head_dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            regression_head_layers: list[nn.Module] = [
                nn.LayerNorm(fusion_dim)
            ]
            regression_input_dim = fusion_dim
            for regression_hidden_dim in self.regression_head_hidden_dims:
                regression_head_layers.extend(
                    [
                        nn.Linear(
                            regression_input_dim,
                            regression_hidden_dim,
                        ),
                        nn.GELU(),
                        nn.Dropout(self.regression_head_dropout),
                    ]
                )
                regression_input_dim = regression_hidden_dim
            regression_head_layers.append(
                nn.Linear(regression_input_dim, 1)
            )
            self.raw_head = nn.Sequential(*regression_head_layers)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(fusion_dim + density_dim),
            nn.Linear(fusion_dim + density_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, 3.0)
        positive_scale = max(float(ood_scale_init), 1e-6)
        inverse_softplus = math.log(math.expm1(positive_scale))
        self.ood_scale_raw = nn.Parameter(torch.tensor(inverse_softplus))
        self.register_buffer(
            "ood_threshold",
            torch.tensor(float(ood_threshold), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ood_shift_scale",
            torch.tensor(float(ood_shift_scale), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "ood_base_shift",
            torch.tensor(float(ood_base_shift), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "residual_global_center_mix",
            torch.tensor(float(residual_global_center_mix), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("prior", torch.tensor(float(prior), dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        numeric: torch.Tensor,
        token_mask: torch.Tensor,
        role_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        atomic_numbers: torch.Tensor,
        density: torch.Tensor,
        *,
        component_ids: torch.Tensor | None = None,
        cip_orientation_ids: torch.Tensor | None = None,
        sample_prior: torch.Tensor | None = None,
        orb_level_features: torch.Tensor | None = None,
        secondary_orb_features: torch.Tensor | None = None,
        smiles_sequence_token_ids: torch.Tensor | None = None,
        smiles_sequence_token_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        token_mask = token_mask.bool()
        if self.multi_level_orb_feature_attention is not None:
            if orb_level_features is None:
                raise ValueError(
                    "A4 multi-level Orb attention requires orb_level_features"
                )
            x = self.multi_level_orb_feature_attention(orb_level_features)
        elif orb_level_features is not None:
            raise ValueError(
                "orb_level_features were provided while A4 is disabled"
            )
        if self.dual_backbone_feature_fusion is not None:
            if secondary_orb_features is None:
                raise ValueError(
                    "dual backbone feature fusion requires "
                    "secondary_orb_features"
                )
            x = self.dual_backbone_feature_fusion(
                x,
                secondary_orb_features,
                role_ids=(
                    role_ids
                    if self.use_dual_backbone_feature_role_gate
                    else None
                ),
            )
        elif secondary_orb_features is not None:
            raise ValueError(
                "secondary_orb_features were provided while dual backbone "
                "feature fusion is disabled"
            )
        cip_orientation_features: torch.Tensor | float = 0.0
        if self.cip_orientation_embedding is not None:
            if cip_orientation_ids is None:
                raise ValueError(
                    "CIP orientation embedding requires "
                    "cip_orientation_ids"
                )
            if cip_orientation_ids.shape != token_mask.shape:
                raise ValueError(
                    "cip_orientation_ids must match token_mask shape"
                )
            cip_orientation_ids = cip_orientation_ids.long()
            if (
                torch.any(cip_orientation_ids < 0)
                or torch.any(cip_orientation_ids > 3)
            ):
                raise ValueError(
                    "cip_orientation_ids must be in [0, 3]"
                )
            if torch.any(
                (cip_orientation_ids != 0) & ~token_mask
            ):
                raise ValueError(
                    "Padded tokens cannot carry a CIP orientation"
                )
            cip_orientation_features = self.cip_orientation_embedding(
                cip_orientation_ids
            )
        elif cip_orientation_ids is not None:
            raise ValueError(
                "cip_orientation_ids were provided while the CIP "
                "orientation embedding is disabled"
            )
        h = (
            self.input_proj(self.input_norm(x.float()))
            + self.numeric_proj(numeric.float())
            + self.role_embedding(role_ids.long())
            + self.token_type_embedding(token_type_ids.long())
            + self.atomic_number_embedding(atomic_numbers.long().clamp(min=0, max=118))
            + cip_orientation_features
        )
        h = self.input_dropout(h) * token_mask.unsqueeze(-1).to(h.dtype)
        atom_tokens = self.atom_encoder(h, src_key_padding_mask=~token_mask)
        atom_tokens = atom_tokens * token_mask.unsqueeze(-1).to(atom_tokens.dtype)
        if self.site_ligand_metal_attention is not None:
            atom_tokens = self.site_ligand_metal_attention(
                atom_tokens,
                token_mask,
                token_type_ids,
            )

        role_tokens, role_present = self.role_cross_attention(atom_tokens, token_mask, role_ids)
        safe_role_present = role_present.clone()
        empty_rows = ~safe_role_present.any(dim=1)
        if empty_rows.any():
            safe_role_present[empty_rows, 0] = True
        role_tokens = self.role_encoder(role_tokens, src_key_padding_mask=~safe_role_present)
        role_tokens = role_tokens * role_present.unsqueeze(-1).to(role_tokens.dtype)
        if self.directed_role_attention is not None:
            role_tokens = self.directed_role_attention(role_tokens, role_present)

        atom_summary = masked_attention_pool(atom_tokens, token_mask, self.atom_pool_score)
        role_summary = masked_attention_pool(role_tokens, safe_role_present, self.role_pool_score)
        if self.use_token_type_cross_attention:
            type_tokens, type_present = self.token_type_cross_attention(
                atom_tokens,
                token_mask,
                token_type_ids,
            )
            safe_type_present = type_present.clone()
            empty_type_rows = ~safe_type_present.any(dim=1)
            if empty_type_rows.any():
                safe_type_present[empty_type_rows, 0] = True
            type_tokens = self.role_encoder(
                type_tokens,
                src_key_padding_mask=~safe_type_present,
            )
            type_tokens = type_tokens * type_present.unsqueeze(-1).to(type_tokens.dtype)
            type_summary = masked_attention_pool(
                type_tokens,
                safe_type_present,
                self.role_pool_score,
            )
            fusion_parts = [
                atom_summary,
                role_summary,
                type_summary,
                atom_summary * role_summary,
                torch.abs(atom_summary - role_summary),
                atom_summary * type_summary,
                torch.abs(atom_summary - type_summary),
                role_summary * type_summary,
                torch.abs(role_summary - type_summary),
            ]
            if self.use_cross_view_interactions:
                role_view = role_tokens.unsqueeze(2)
                type_view = type_tokens.unsqueeze(1)
                interaction_tokens = self.cross_view_interaction_proj(
                    torch.cat(
                        [
                            role_view * type_view,
                            torch.abs(role_view - type_view),
                        ],
                        dim=-1,
                    )
                )
                interaction_mask = (
                    role_present.unsqueeze(2) & type_present.unsqueeze(1)
                )
                interaction_summary = masked_attention_pool(
                    interaction_tokens.flatten(1, 2),
                    interaction_mask.flatten(1, 2),
                    self.cross_view_interaction_pool_score,
                )
                fusion_parts.append(interaction_summary)
            fusion = torch.cat(fusion_parts, dim=-1)
        else:
            fusion = torch.cat(
                [
                    atom_summary,
                    role_summary,
                    atom_summary * role_summary,
                    torch.abs(atom_summary - role_summary),
                ],
                dim=-1,
            )
        site_donor_relation_residual = None
        site_donor_relation_logit = None
        site_donor_relation_coupling = None
        reaction_pair_transformer_residual = None
        direct_raw_prediction = self.raw_head(fusion).squeeze(-1)
        if self.reaction_pair_transformer is not None:
            reaction_pair_transformer_residual = (
                self.reaction_pair_transformer(
                    role_tokens,
                    role_present,
                )
            )
            direct_raw_prediction = (
                direct_raw_prediction
                + reaction_pair_transformer_residual
            )
        if self.site_donor_relation_attention is not None:
            if component_ids is None:
                raise ValueError(
                    "site-donor relation attention requires component_ids"
                )
            (
                site_donor_relation_logit,
                site_donor_relation_residual,
                site_donor_relation_coupling,
            ) = (
                self.site_donor_relation_attention(
                    atom_tokens,
                    numeric,
                    token_mask,
                    role_ids,
                    token_type_ids,
                    component_ids,
                )
            )
            direct_raw_prediction = (
                direct_raw_prediction + site_donor_relation_residual
            )
        raw_prediction = direct_raw_prediction
        auxiliary_prediction = None
        consistency_residual = None
        sign_logit = None
        magnitude_prediction = None
        structured_prediction = None
        structured_residual = None
        if self.aho_consistency_adapter is not None:
            auxiliary_prediction, consistency_residual = (
                self.aho_consistency_adapter(fusion)
            )
            raw_prediction = raw_prediction + consistency_residual
        if self.ddg_sign_magnitude_adapter is not None:
            (
                sign_logit,
                magnitude_prediction,
                structured_prediction,
                structured_residual,
            ) = self.ddg_sign_magnitude_adapter(
                fusion,
                direct_raw_prediction,
            )
            raw_prediction = raw_prediction + structured_residual

        density = density.float()
        base_gate_logit = self.gate_head(torch.cat([fusion, density], dim=-1)).squeeze(-1)
        density_max = density[:, self.num_roles + 1]
        ood_excess = torch.relu(density_max - self.ood_threshold)
        ood_scale = torch.nn.functional.softplus(self.ood_scale_raw)
        if self.exponential_ood_gate:
            confidence = torch.sigmoid(base_gate_logit) * torch.exp(-ood_scale * ood_excess)
        else:
            confidence = torch.sigmoid(base_gate_logit - ood_scale * ood_excess)
        prior = (
            self.prior.expand_as(raw_prediction)
            if sample_prior is None
            else sample_prior.to(raw_prediction.dtype)
        )
        if self.gate_residual_mode == "centered_ood":
            residual_center = (
                self.residual_global_center_mix * self.prior
                + (1.0 - self.residual_global_center_mix) * prior
            )
            centered_prediction = prior + confidence * (
                raw_prediction - residual_center
            )
            ood_shift = self.ood_base_shift + (1.0 - self.ood_base_shift) * (
                1.0 - torch.exp(-self.ood_shift_scale * ood_excess)
            )
            prediction = raw_prediction + ood_shift * (
                centered_prediction - raw_prediction
            )
        else:
            ood_shift = torch.ones_like(raw_prediction)
            prediction = prior + confidence * (raw_prediction - prior)

        pre_calibration_prediction = None
        calibration_residual = None
        calibration_features = None
        if self.ddg_residual_calibrator is not None:
            if (
                sign_logit is None
                or magnitude_prediction is None
                or structured_prediction is None
                or structured_residual is None
            ):
                raise RuntimeError(
                    "DDG residual calibrator has no A6 diagnostics"
                )
            if self.gate_residual_mode == "centered_ood":
                raw_sensitivity = 1.0 + ood_shift * (confidence - 1.0)
            else:
                raw_sensitivity = confidence
            structured_output_residual = (
                raw_sensitivity * structured_residual
            )
            direct_prediction = prediction - structured_output_residual
            pre_calibration_prediction = prediction
            (
                calibration_residual,
                calibration_features,
            ) = self.ddg_residual_calibrator(
                direct_prediction=direct_prediction,
                confidence=confidence,
                ood_excess=ood_excess,
                sign_logit=sign_logit,
                magnitude_prediction=magnitude_prediction,
                structured_prediction=structured_prediction,
                structured_output_residual=structured_output_residual,
            )
            prediction = prediction + calibration_residual

        major_summary_view_logits = None
        major_summary_view_mask = None
        major_summary_logit = None
        major_summary_disagreement = None
        major_summary_residual = None
        if self.ddg_major_summary_adapter is not None:
            if sign_logit is None or magnitude_prediction is None:
                raise RuntimeError(
                    "DDG Major summary has no A6 diagnostics"
                )
            (
                major_summary_view_logits,
                major_summary_view_mask,
                major_summary_logit,
                major_summary_disagreement,
                major_summary_residual,
            ) = self.ddg_major_summary_adapter(
                fusion=fusion,
                role_tokens=role_tokens,
                role_present=role_present,
                base_prediction=prediction,
                confidence=confidence,
                ood_excess=ood_excess,
                sign_logit=sign_logit,
                magnitude_prediction=magnitude_prediction,
            )
            prediction = prediction + major_summary_residual

        smiles_sequence_residual = None
        smiles_sequence_summary = None
        smiles_sequence_auxiliary_prediction = None
        if self.smiles_sequence_adapter is not None:
            if (
                smiles_sequence_token_ids is None
                or smiles_sequence_token_mask is None
            ):
                raise ValueError(
                    "SMILES sequence adapter requires token IDs and mask"
                )
            (
                smiles_sequence_residual,
                smiles_sequence_summary,
            ) = self.smiles_sequence_adapter(
                smiles_sequence_token_ids,
                smiles_sequence_token_mask,
                fusion,
            )
            smiles_sequence_auxiliary_prediction = (
                self.smiles_sequence_adapter.auxiliary_prediction(
                    smiles_sequence_summary
                )
            )
            prediction = prediction + smiles_sequence_residual
        elif (
            smiles_sequence_token_ids is not None
            or smiles_sequence_token_mask is not None
        ):
            raise ValueError(
                "SMILES sequence tensors were provided while the adapter "
                "is disabled"
            )

        if return_diagnostics:
            diagnostics = {
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "confidence": confidence,
                "ood_excess": ood_excess,
                "ood_shift": ood_shift,
                "prior": prior,
                "fusion": fusion,
            }
            if auxiliary_prediction is not None:
                diagnostics["auxiliary_prediction"] = auxiliary_prediction
                diagnostics["consistency_residual"] = consistency_residual
            if site_donor_relation_residual is not None:
                diagnostics["site_donor_relation_logit"] = (
                    site_donor_relation_logit
                )
                diagnostics["site_donor_relation_residual"] = (
                    site_donor_relation_residual
                )
                diagnostics["site_donor_relation_coupling"] = (
                    site_donor_relation_coupling
                )
            if reaction_pair_transformer_residual is not None:
                diagnostics["reaction_pair_transformer_residual"] = (
                    reaction_pair_transformer_residual
                )
            if sign_logit is not None:
                diagnostics["direct_raw_prediction"] = (
                    direct_raw_prediction
                )
                diagnostics["ddg_sign_logit"] = sign_logit
                diagnostics["ddg_magnitude_prediction"] = (
                    magnitude_prediction
                )
                diagnostics["ddg_structured_prediction"] = (
                    structured_prediction
                )
                diagnostics["ddg_structured_residual"] = (
                    structured_residual
                )
            if calibration_residual is not None:
                diagnostics["pre_calibration_prediction"] = (
                    pre_calibration_prediction
                )
                diagnostics["ddg_calibration_residual"] = (
                    calibration_residual
                )
                diagnostics["ddg_calibration_features"] = (
                    calibration_features
                )
            if major_summary_residual is not None:
                diagnostics["ddg_major_summary_view_logits"] = (
                    major_summary_view_logits
                )
                diagnostics["ddg_major_summary_view_mask"] = (
                    major_summary_view_mask
                )
                diagnostics["ddg_major_summary_logit"] = (
                    major_summary_logit
                )
                diagnostics["ddg_major_summary_disagreement"] = (
                    major_summary_disagreement
                )
                diagnostics["ddg_major_summary_residual"] = (
                    major_summary_residual
                )
            if smiles_sequence_residual is not None:
                diagnostics["smiles_sequence_residual"] = (
                    smiles_sequence_residual
                )
                diagnostics["smiles_sequence_summary"] = (
                    smiles_sequence_summary
                )
                if smiles_sequence_auxiliary_prediction is not None:
                    diagnostics[
                        "smiles_sequence_auxiliary_prediction"
                    ] = smiles_sequence_auxiliary_prediction
            return diagnostics
        return prediction
