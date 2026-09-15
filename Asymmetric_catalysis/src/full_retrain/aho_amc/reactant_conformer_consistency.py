"""Train-only utilities for reactant conformer-view consistency.

The helpers are intentionally independent of the trainer so the historical
fixed three-conformer path remains untouched unless a caller explicitly asks
for an augmented view.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


FEATURE_MODES = {"mean", "mean_std", "mean_std_max"}


def aggregate_fixed_conformer_view(
    stacked: torch.Tensor,
    feature_mode: str,
    *,
    selected_index: int | None = None,
) -> torch.Tensor:
    """Aggregate ``[conformer, atom, channel]`` features.

    ``selected_index=None`` is exactly the historical fixed aggregation.
    A selected single-conformer view preserves the output width by using zero
    dispersion and the selected features for both mean and maximum channels.
    """
    if feature_mode not in FEATURE_MODES:
        raise ValueError(f"Unsupported conformer feature mode: {feature_mode}")
    if stacked.ndim != 3 or stacked.shape[0] < 1:
        raise ValueError(
            "stacked conformer features must have shape "
            "[conformer, atom, channel] with at least one conformer"
        )
    if selected_index is None:
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0, unbiased=False)
        maximum = stacked.max(dim=0).values
    else:
        if selected_index < 0 or selected_index >= stacked.shape[0]:
            raise IndexError(
                f"selected conformer {selected_index} is outside "
                f"0..{stacked.shape[0] - 1}"
            )
        mean = stacked[selected_index]
        std = torch.zeros_like(mean)
        maximum = mean
    parts = [mean]
    if feature_mode in {"mean_std", "mean_std_max"}:
        parts.append(std)
    if feature_mode == "mean_std_max":
        parts.append(maximum)
    return torch.cat(parts, dim=-1)


def mix_role_token_view(
    clean_tokens: torch.Tensor,
    augmented_tokens: torch.Tensor,
    role_ids: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    role_id: int,
) -> torch.Tensor:
    """Replace only valid tokens belonging to one requested role."""
    if clean_tokens.shape != augmented_tokens.shape:
        raise ValueError("clean and augmented token shapes differ")
    if role_ids.shape != clean_tokens.shape[:2]:
        raise ValueError("role_ids shape does not match token axes")
    if token_mask.shape != clean_tokens.shape[:2]:
        raise ValueError("token_mask shape does not match token axes")
    replace = (
        token_mask.bool() & role_ids.long().eq(int(role_id))
    ).to(device=clean_tokens.device)
    return torch.where(
        replace.unsqueeze(-1),
        augmented_tokens,
        clean_tokens,
    )


def component_ids_for_role(
    *,
    role_ids: torch.Tensor,
    token_mask: torch.Tensor,
    component_local_ids: torch.Tensor,
    global_components: torch.Tensor,
    role_id: int,
) -> tuple[int, ...]:
    """Resolve batch-local role tokens to unique global component IDs."""
    token_shape = role_ids.shape
    if (
        token_mask.shape != token_shape
        or component_local_ids.shape != token_shape
    ):
        raise ValueError("token role/mask/component-local shapes differ")
    if global_components.ndim != 2 or global_components.shape[0] != token_shape[0]:
        raise ValueError("global_components must have shape [batch, components]")
    selected: set[int] = set()
    role_mask = token_mask.bool() & role_ids.long().eq(int(role_id))
    for row in range(token_shape[0]):
        local_ids = torch.unique(component_local_ids[row, role_mask[row]].long())
        for local_id_tensor in local_ids:
            local_id = int(local_id_tensor)
            if local_id < 0 or local_id >= global_components.shape[1]:
                continue
            component_id = int(global_components[row, local_id])
            if component_id >= 0:
                selected.add(component_id)
    return tuple(sorted(selected))


def sample_single_conformer_indices(
    component_ids: Sequence[int],
    conformer_counts: Mapping[int, int],
    *,
    probability: float,
    generator: torch.Generator | None = None,
) -> dict[int, int]:
    """Sample at most one valid conformer for each requested component."""
    if probability < 0.0 or probability > 1.0:
        raise ValueError("conformer-view probability must be in [0, 1]")
    selected: dict[int, int] = {}
    for raw_component_id in component_ids:
        component_id = int(raw_component_id)
        count = int(conformer_counts.get(component_id, 0))
        if count <= 1:
            continue
        use_view = bool(
            torch.rand((), generator=generator).item() < probability
        )
        if use_view:
            selected[component_id] = int(
                torch.randint(
                    count,
                    (),
                    generator=generator,
                ).item()
            )
    return selected


def prediction_consistency_loss(
    clean_prediction: torch.Tensor,
    augmented_prediction: torch.Tensor,
    *,
    huber_delta: float,
) -> torch.Tensor:
    """Huber consistency with a stop-gradient clean-view teacher."""
    if clean_prediction.shape != augmented_prediction.shape:
        raise ValueError("clean and augmented prediction shapes differ")
    if huber_delta <= 0.0:
        raise ValueError("consistency Huber delta must be positive")
    return F.huber_loss(
        augmented_prediction,
        clean_prediction.detach(),
        delta=huber_delta,
    )
