"""Training utilities for the CGCNN active learning backend."""

from __future__ import annotations

import copy
import math
from typing import List, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm

from cgcnn_model.cgcnn.data_process import CIFData, collate_pool
from cgcnn_model.cgcnn.model import CrystalGraphConvNet

from ...config import ALConfig
from ...utils import Normalizer, SubsetSequentialSampler


class CGCNNSubsetTrainer:
    """Train CGCNN on a labeled subset and keep the best validation checkpoint."""

    def __init__(self, cfg: ALConfig, dataset: CIFData, labeled_idxs: List[int], val_idxs: List[int]):
        self.cfg = cfg
        self.dataset = dataset
        self.labeled_idxs = list(labeled_idxs)
        self.val_idxs = list(val_idxs)
        self.device = torch.device("cuda" if cfg.cuda and torch.cuda.is_available() else "cpu")

        self.train_loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.workers,
            sampler=SubsetRandomSampler(self.labeled_idxs),
            collate_fn=collate_pool,
            pin_memory=(self.device.type == "cuda"),
        )

        self.val_loader = None
        if self.val_idxs:
            self.val_loader = DataLoader(
                dataset=self.dataset,
                batch_size=self.cfg.batch_size,
                num_workers=self.cfg.workers,
                sampler=SubsetSequentialSampler(self.val_idxs),
                collate_fn=collate_pool,
                pin_memory=(self.device.type == "cuda"),
            )

        sample_indices = self.labeled_idxs[: min(200, len(self.labeled_idxs))]
        sample_data_list = [self.dataset[i] for i in sample_indices] if sample_indices else []
        if sample_data_list:
            _, sample_target, _ = collate_pool(sample_data_list)
        else:
            sample_target = torch.zeros((1, self.cfg.out_size), dtype=torch.float32)
        self.normalizer = Normalizer(sample_target.to(self.device))

        structures, _, _ = self.dataset[0]
        orig_atom_fea_len = structures[0].shape[-1]
        nbr_fea_len = structures[1].shape[-1]

        self.model = CrystalGraphConvNet(
            orig_atom_fea_len,
            nbr_fea_len,
            atom_fea_len=self.cfg.atom_fea_len,
            n_conv=self.cfg.n_conv,
            h_fea_len=self.cfg.h_fea_len,
            n_h=self.cfg.n_h,
            outsize=self.cfg.out_size,
            dropout=self.cfg.dropout,
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        milestones = self.cfg.lr_milestones or []
        self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1) if milestones else None

    def _move_to_device(self, input_tuple, target):
        atom_fea = input_tuple[0].to(self.device, non_blocking=True)
        nbr_fea = input_tuple[1].to(self.device, non_blocking=True)
        nbr_fea_idx = input_tuple[2].to(self.device, non_blocking=True)
        crystal_atom_idx = [x.to(self.device, non_blocking=True) for x in input_tuple[3]]
        pooling_idx = [x.to(self.device, non_blocking=True) for x in input_tuple[4]]
        target = target.to(self.device, non_blocking=True)
        return (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, pooling_idx), target

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
            for input_batch, target_batch, _ in self.train_loader:
                (input_dev, target_dev) = self._move_to_device(input_batch, target_batch)
                target_norm = self.normalizer.norm(target_dev)
                output = self.model(*input_dev)
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
                    for input_batch, target_batch, _ in self.val_loader:
                        (input_dev, target_dev) = self._move_to_device(input_batch, target_batch)
                        target_norm = self.normalizer.norm(target_dev)
                        output = self.model(*input_dev)
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
            return pd.DataFrame()
        loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.workers,
            sampler=SubsetSequentialSampler(indices),
            collate_fn=collate_pool,
            pin_memory=(self.device.type == "cuda"),
        )
        self.model.eval()
        for module in self.model.modules():
            if module.__class__.__name__.lower().startswith("dropout"):
                module.train()  # enable MC dropout
        rows = []
        for input_batch, target_batch, batch_ids in loader:
            input_dev, _ = self._move_to_device(input_batch, target_batch)
            outputs = []
            for _ in range(self.cfg.eval_times):
                out = self.model(*input_dev)
                out = self.normalizer.denorm(out)
                outputs.append(out)
            out_stack = torch.stack(outputs, dim=0)
            means = out_stack.mean(dim=0).cpu().numpy()
            stds = out_stack.std(dim=0, unbiased=False).cpu().numpy()
            for j, cif_id in enumerate(batch_ids):
                row = {"dataset_idx": None, "cif_id": cif_id}
                for p_idx, prop in enumerate(self.cfg.properties):
                    row[f"v_{prop}_mean"] = means[j, p_idx]
                    row[f"v_{prop}_std"] = stds[j, p_idx]
                rows.append(row)
        for k, idx in enumerate(indices):
            rows[k]["dataset_idx"] = idx
        return pd.DataFrame(rows)


__all__ = ["CGCNNSubsetTrainer"]
