"""Training utilities for the DimeNet++ (DPP) active learning backend."""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Subset
from tqdm import tqdm

from ...config import ALConfig
from ...utils import Normalizer
from ..escn import EscnCIFDataset
import sys
import os
AL_PIPELINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if AL_PIPELINE_ROOT not in sys.path:
    sys.path.append(AL_PIPELINE_ROOT)
import time
try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import global_mean_pool
except ImportError as exc:  # pragma: no cover - runtime safeguard
    raise RuntimeError("DPP backend requires torch_geometric dependencies.") from exc

from models.encoders.dpp import DppEncoder

class DppRegressor(nn.Module):
    def __init__(self, out_dim: int, dropout: float, encoder_kwargs: Optional[dict] = None):
        super().__init__()
        encoder_kwargs = encoder_kwargs or {}
        
        
            
            
        self.encoder = DppEncoder(**encoder_kwargs)
        linear_layer = getattr(getattr(self.encoder, "output_block", None), "lin", None)
        if linear_layer is not None and hasattr(linear_layer, "out_features"):
            hidden_dim = int(linear_layer.out_features)
        else:
            hidden_dim = int(encoder_kwargs.get("out_channels", 128))
        mid_dim = max(1, hidden_dim // 2)
        self.readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, mid_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, batch):  # type: ignore[override]
        node_emb = self.encoder(batch)
        pooled = global_mean_pool(node_emb, batch.batch)
        output = self.readout(pooled)
        return output


class DPPSubsetTrainer:
    """Train DimeNet++ on a labeled subset and keep the best validation checkpoint."""

    def __init__(self, cfg: ALConfig, dataset: EscnCIFDataset, labeled_idxs: List[int], val_idxs: List[int]):
        self.cfg = cfg
        self.dataset = dataset
        self.labeled_idxs = list(labeled_idxs)
        self.val_idxs = list(val_idxs)
        self.device = torch.device("cuda" if cfg.cuda and torch.cuda.is_available() else "cpu")

        encoder_kwargs = cfg.model_kwargs or {}
        self.model = DppRegressor(
            out_dim=cfg.out_size,
            dropout=cfg.dropout,
            encoder_kwargs=encoder_kwargs,
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        milestones = self.cfg.lr_milestones or []
        self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1) if milestones else None

        subset = Subset(self.dataset, self.labeled_idxs)
        
        self.train_loader = PyGDataLoader(
            subset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.workers,
            pin_memory=(self.device.type == "cuda"),
        )

        self.val_loader = None
        if self.val_idxs:
            val_subset = Subset(self.dataset, self.val_idxs)
            self.val_loader = PyGDataLoader(
                val_subset,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.workers,
                pin_memory=(self.device.type == "cuda"),
            )

        sample_indices = self.labeled_idxs[: min(200, len(self.labeled_idxs))]
        if sample_indices:
            sample_targets = torch.stack([self.dataset[i].y for i in sample_indices], dim=0)
        else:
            sample_targets = torch.zeros((1, self.cfg.out_size), dtype=torch.float32)
        if sample_targets.ndim == 1:
            sample_targets = sample_targets.unsqueeze(-1)
        self.normalizer = Normalizer(sample_targets.to(self.device))

    def fit(self, log_path: Optional[str] = None):
        if not self.labeled_idxs:
            return
        history_rows = []
        best_state = copy.deepcopy(self.model.state_dict())
        best_val_loss = math.inf
        epochs_without_improve = 0
        patience = self.cfg.early_stop_patience if (self.val_loader and self.cfg.early_stop_patience is not None) else None
        for epoch in tqdm(range(self.cfg.epochs), desc="Train epochs", leave=False, dynamic_ncols=True):
            self.model.train()
            train_loss_sum = 0.0
            train_count = 0
            for batch in self.train_loader:
                
                batch = batch.to(self.device)
                target = batch.y.to(self.device)
                if target.ndim == 1:
                    target = target.unsqueeze(-1)
                target_norm = self.normalizer.norm(target)
                output = self.model(batch)
                loss = self.criterion(output, target_norm)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss_sum += float(loss) * target_norm.size(0)
                train_count += target_norm.size(0)
            train_loss_avg = train_loss_sum / max(train_count, 1)
            if self.scheduler is not None:
                self.scheduler.step()

            val_loss_avg = float("nan")
            improved = False
            if self.val_loader:
                self.model.eval()
                val_loss_sum = 0.0
                val_count = 0
                with torch.no_grad():
                    for batch in self.val_loader:
                        batch = batch.to(self.device)
                        target = batch.y.to(self.device)
                        if target.ndim == 1:
                            target = target.unsqueeze(-1)
                        target_norm = self.normalizer.norm(target)
                        output = self.model(batch)
                        batch_loss = self.criterion(output, target_norm)
                        val_loss_sum += float(batch_loss) * target_norm.size(0)
                        val_count += target_norm.size(0)
                if val_count > 0:
                    val_loss_avg = val_loss_sum / val_count
                if math.isfinite(val_loss_avg) and val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    best_state = copy.deepcopy(self.model.state_dict())
                    improved = True
            else:
                best_state = copy.deepcopy(self.model.state_dict())
                improved = True
            history_rows.append((epoch, train_loss_avg, val_loss_avg))

            if patience is not None:
                if improved:
                    epochs_without_improve = 0
                else:
                    epochs_without_improve += 1
                    if epochs_without_improve >= patience:
                        print(f"Early stopping after {epoch + 1} epochs (no val improvement for {patience} epochs).")
                        break

        self.model.load_state_dict(best_state)

        if log_path:
            import os

            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", encoding="utf-8") as f:
                f.write("epoch,train_loss,val_loss\n")
                for epoch_idx, train_loss, val_loss in history_rows:
                    train_repr = f"{train_loss:.6f}" if math.isfinite(train_loss) else "nan"
                    val_repr = f"{val_loss:.6f}" if math.isfinite(val_loss) else "nan"
                    f.write(f"{epoch_idx},{train_repr},{val_repr}\n")

    @torch.no_grad()
    def predict_subset_mc(self, indices: List[int]) -> pd.DataFrame:
        if not indices:
            cols = ["dataset_idx", "cif_id"]
            for prop in self.cfg.properties:
                cols.extend([f"v_{prop}_mean", f"v_{prop}_std"])
            return pd.DataFrame(columns=cols)
        subset = Subset(self.dataset, indices)
        loader = PyGDataLoader(
            subset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.workers,
            pin_memory=(self.device.type == "cuda"),
        )
        self.model.eval()
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
        order_map = {idx: pos for pos, idx in enumerate(indices)}
        rows = []
        for batch in loader:
            batch = batch.to(self.device)
            outputs = []
            for _ in range(self.cfg.eval_times):
                out = self.model(batch)
                out = self.normalizer.denorm(out)
                outputs.append(out)
            
            out_stack = torch.stack(outputs, dim=0)
            means = out_stack.mean(dim=0).cpu().numpy()
            stds = out_stack.std(dim=0, unbiased=False).cpu().numpy()
            batch_indices = batch.sample_idx.view(-1).cpu().numpy()
            for j, idx in enumerate(batch_indices):
                identifier = self.dataset.get_identifier(int(idx))
                row = {"dataset_idx": int(idx), "cif_id": identifier}
                for p_idx, prop in enumerate(self.cfg.properties):
                    row[f"v_{prop}_mean"] = means[j, p_idx]
                    row[f"v_{prop}_std"] = stds[j, p_idx]
                rows.append(row)
        rows.sort(key=lambda r: order_map.get(r["dataset_idx"], -1))
        return pd.DataFrame(rows)


__all__ = ["DppRegressor", "DPPSubsetTrainer"]
