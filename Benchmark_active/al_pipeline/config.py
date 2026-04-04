"""Configuration objects for the active learning pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ALConfig:
    label_dir: str
    structure_dir: str
    properties: List[str]
    cuda: bool = True
    workers: int = 4
    batch_size: int = 16
    epochs: int = 100
    lr: float = 8e-5
    lr_milestones: Optional[List[int]] = None
    weight_decay: float = 0.0
    dropout: float = 0.1
    n_conv: int = 3
    n_h: int = 2
    h_fea_len: int = 64
    atom_fea_len: int = 128
    out_size: int = 1
    random_seed: int = 23
    init_size: int = 40
    rounds: int = 10
    per_round_best_k: int = 20
    per_round_uncert_k: int = 20
    maximize_best: bool = False
    eval_times: int = 20
    save_dir: str = "./al_runs"
    model_type: str = "cgcnn"
    model_kwargs: Optional[Dict[str, Any]] = None
    val_size: int = 40
    early_stop_patience: Optional[int] = 10
