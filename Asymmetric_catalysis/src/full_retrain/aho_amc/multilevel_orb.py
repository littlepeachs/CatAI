from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn


def validate_orb_stack_indices(
    stack_indices: Sequence[int],
    *,
    num_stacks: int,
) -> tuple[int, ...]:
    """Validate an early/middle/final stack selection for one Orb forward."""
    indices = tuple(int(index) for index in stack_indices)
    if len(indices) < 2:
        raise ValueError("Multi-level Orb attention requires at least two stack indices")
    if tuple(sorted(set(indices))) != indices:
        raise ValueError("Orb stack indices must be unique and strictly increasing")
    if indices[0] < 0 or indices[-1] >= int(num_stacks):
        raise ValueError(
            f"Orb stack indices {indices} are outside [0, {int(num_stacks) - 1}]"
        )
    if indices[-1] != int(num_stacks) - 1:
        raise ValueError(
            "The final selected Orb level must be the last GNN stack so the "
            "parent node_features path is preserved exactly"
        )
    return indices


def forward_orb_with_node_feature_levels(
    orb_backbone: nn.Module,
    graph: Any,
    *,
    stack_indices: Sequence[int],
) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
    """Run Orb exactly once and return selected intermediate node features.

    Temporary forward hooks observe early/middle GNN stack outputs without
    changing the Orb-v2 module topology or state dict. The last level is taken
    directly from the normal ``node_features`` return value.
    """
    if not hasattr(orb_backbone, "gnn_stacks"):
        raise TypeError("Orb backbone does not expose gnn_stacks")
    gnn_stacks = orb_backbone.gnn_stacks
    indices = validate_orb_stack_indices(
        stack_indices,
        num_stacks=len(gnn_stacks),
    )
    captured: dict[int, torch.Tensor] = {}
    handles: list[Any] = []

    def make_hook(index: int):
        def capture(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if (
                not isinstance(output, (tuple, list))
                or not output
                or not torch.is_tensor(output[0])
            ):
                raise RuntimeError(
                    f"Orb GNN stack {index} did not return a node-feature tensor"
                )
            captured[index] = output[0]

        return capture

    for index in indices[:-1]:
        handles.append(gnn_stacks[index].register_forward_hook(make_hook(index)))
    try:
        output = orb_backbone(graph)
    finally:
        for handle in handles:
            handle.remove()

    if not isinstance(output, dict) or not torch.is_tensor(
        output.get("node_features")
    ):
        raise RuntimeError("Orb backbone did not return node_features")
    final_features = output["node_features"]
    missing = [index for index in indices[:-1] if index not in captured]
    if missing:
        raise RuntimeError(f"Orb hooks did not observe selected stacks: {missing}")
    levels = tuple(
        final_features if index == indices[-1] else captured[index]
        for index in indices
    )
    if any(level.shape != final_features.shape for level in levels):
        shapes = [tuple(level.shape) for level in levels]
        raise RuntimeError(
            "Selected Orb node-feature levels have inconsistent shapes: "
            f"{shapes}"
        )
    return output, levels
