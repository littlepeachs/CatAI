"""Utility helpers shared across active learning backends."""

from __future__ import annotations

from typing import Iterator, List

import torch
from torch.utils.data import Sampler


class Normalizer:
    """Perform mean/std normalization with safe standard deviation floor."""

    def __init__(self, tensor: torch.Tensor, eps: float = 1e-12):
        self.mean = torch.mean(tensor, dim=0)
        self.std = torch.std(tensor, dim=0)
        self.std = torch.where(self.std < eps, torch.ones_like(self.std) * eps, self.std)

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)
        return (tensor - mean) / std

    def denorm(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)
        return tensor * std + mean


class SubsetSequentialSampler(Sampler):
    """Iterate a fixed list of indices without replacement."""

    def __init__(self, indices: List[int]):
        self.indices = list(indices)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
