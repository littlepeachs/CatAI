"""Training utilities for the DimeNet++ (CatAI) active learning backend."""

from __future__ import annotations

import copy
import math
from typing import List, Optional
from pathlib import Path

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time

try:
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import global_mean_pool
except ImportError as exc:  # pragma: no cover - runtime safeguard
    raise RuntimeError("CatAI backend requires torch_geometric dependencies.") from exc

BENCHMARK_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(BENCHMARK_ROOT))
from ocpmodels_catai.common.registry import registry
from ocpmodels_catai.models.gemnet_oc_mt.backbone import GemNetOCBackbone
from ocpmodels_catai.trainers.mt.config import GemNetOCModelConfig

from torch_geometric.data import Data, Batch
from ocpmodels_catai.models.gemnet_oc_mt.goc_graph import Cutoffs,MaxNeighbors
from ocpmodels_catai.trainers.mt.transform import _generate_graphs
from ocpmodels_catai.trainers.mt.config import TransformConfigs

from typing import Mapping
import fnmatch
from logging import getLogger
log = getLogger(__name__)

def load_state_dict(
    module: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    ignore_keys_patterns: list[str] = [],
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    ignore_keys_patterns.append("*module_dict.y*")
    print("ignore_keys_patterns: ", ignore_keys_patterns)
    updated_state_dict: dict[str, torch.Tensor] = {}

    for k, v in state_dict.items():
        matched_pattern = None
        for p in ignore_keys_patterns:
            if fnmatch.fnmatch(k, p):
                matched_pattern = p
                break

        if matched_pattern is not None:
            log.info(
                f"Ignoring existing key {k} due to matching pattern: {matched_pattern}"
            )
            continue  # ignore this key

        updated_state_dict[k] = v

    incompat_keys = module.load_state_dict(updated_state_dict, strict=False)

def _common_transform(data: Data):
    if hasattr(data, "energy"):
        if not torch.is_tensor(data.energy):
            data.energy = torch.tensor(data.energy, dtype=torch.float)
        data.energy = data.energy.view(-1).float()

    if not hasattr(data, "sid"):
        if hasattr(data, "absolute_idx"):
            data.sid = data.absolute_idx
    if not hasattr(data, "natoms"):
        data.natoms = data.num_nodes

    # data.fixed = torch.ones(data.natoms)
    if not hasattr(data, "fixed"):
        data.fixed = torch.zeros(data.natoms, dtype=torch.float)

    if not hasattr(data, "tags"):
        data.tags = 2 * torch.ones(data.natoms)
        data.tags = data.tags.long()
    if not hasattr(data, "cell"):
        max_length = 1000.0
        data.cell = (torch.eye(3) * max_length).unsqueeze(dim=0)

    return data

def _common_transform_all(data: Data):
    data.fixed = data.fixed.long()
    data.atomic_numbers = data.atomic_numbers.long()
    data.tags = data.tags.long()

    data.natoms = int(
        data.natoms.item() if torch.is_tensor(data) else data.natoms
    )

    return data

def matbench_transform(
    data: Data, *, config: TransformConfigs, training: bool
):
    # Decrease max neighbors for very large systems to avoid memory issues.
    if data.natoms > 300:
        max_neighbors = 5
    elif data.natoms > 200:
        max_neighbors = 10
    else:
        max_neighbors = 30
    data = _generate_graphs(
        data,
        config,
        cutoffs=Cutoffs.from_constant(12.0),
        max_neighbors=MaxNeighbors.from_goc_base_proportions(max_neighbors),
        pbc=True,
        training=training,
    )
    data = _common_transform(data)
    data = _common_transform_all(data)
    return data

class CatAIRegressor(nn.Module):
    def __init__(self, out_dim: int, dropout: float, encoder_kwargs: Optional[dict] = None):
        super().__init__()
        
        model_cls = registry.get_model_class('gemnet_oc_mt')
        model_configs = GemNetOCModelConfig(
                num_targets=1,
                num_spherical=7,
                num_radial=128,
                num_blocks=6,
                emb_size_atom=256,
                emb_size_edge=1024,
                emb_size_trip_in=64,
                emb_size_trip_out=128,
                emb_size_quad_in=64,
                emb_size_quad_out=32,
                emb_size_aint_in=64,
                emb_size_aint_out=64,
                emb_size_rbf=32,
                emb_size_cbf=16,
                emb_size_sbf=64,
                num_before_skip=2,
                num_after_skip=2,
                num_concat=4,
                num_atom=3,
                num_output_afteratom=3,
                num_atom_emb_layers=2,
                num_global_out_layers=2,
                regress_forces=False,
                regress_energy=True,
                direct_forces=False,
                use_pbc=True,
                scale_backprop_forces=False,
                rbf={"name": "gaussian"},
                rbf_spherical=None,
                envelope={"name": "polynomial", "exponent": 5},
                cbf={"name": "spherical_harmonics"},
                sbf={"name": "legendre_outer"},
                extensive=True,
                forces_coupled=False,
                activation="silu",
                quad_interaction=True,
                atom_edge_interaction=True,
                edge_atom_interaction=True,
                atom_interaction=True,
                scale_basis=False,
                qint_tags=[1, 2],
                num_elements=120,
                otf_graph=True,
                ln=False,
                dropout=0.0,
                scale_file="configs/fm_scale_files/large.pt",
                absolute_rbf_cutoff=12.0,
                learnable_rbf=False,
                learnable_rbf_stds=False,
                unique_basis_per_layer=False,
                old_gaussian_implementation=False,
                edge_dropout=0.0,
                name="gemnet_oc_mt",
            )
        output_targets = {
            "y": {
            "shape": 1,
            "level": "system",
            "custom_head": True,
            "num_layers": 4,
            "aggregate": "mean",
            }
        }
        model_attributes = {
            "num_spherical": 7,
            "num_radial": 128,
            "num_blocks": 6,
            "emb_size_atom": 256,
            "emb_size_edge": 1024,
            "emb_size_trip_in": 64,
            "emb_size_trip_out": 128,
            "emb_size_quad_in": 64,
            "emb_size_quad_out": 32,
            "emb_size_aint_in": 64,
            "emb_size_aint_out": 64,
            "emb_size_rbf": 32,
            "emb_size_cbf": 16,
            "emb_size_sbf": 64,
            "num_before_skip": 2,
            "num_after_skip": 2,
            "num_concat": 4,
            "num_atom": 3,
            "num_output_afteratom": 3,
            "cutoff": 12.0,
            "cutoff_qint": 12.0,
            "cutoff_aeaint": 12.0,
            "cutoff_aint": 12.0,
            "max_neighbors": 30,
            "max_neighbors_qint": 8,
            "max_neighbors_aeaint": 20,
            "max_neighbors_aint": 1000,
            "rbf": {"name": "gaussian"},
            "envelope": {"name": "polynomial", "exponent": 5},
            "cbf": {"name": "spherical_harmonics"},
            "sbf": {"name": "legendre_outer"},
            "extensive": True,
            "output_init": "HeOrthogonal",
            "activation": "silu",
            "scale_file": "configs/fm_scale_files/large.pt",
            "regress_forces": False,
            "direct_forces": False,
            "forces_coupled": False,
            "quad_interaction": True,
            "atom_edge_interaction": True,
            "edge_atom_interaction": True,
            "atom_interaction": True,
            "num_atom_emb_layers": 2,
            "num_global_out_layers": 2,
            "qint_tags": [1, 2],
            "absolute_rbf_cutoff": 12.0,
            "otf_graph": True,
            "ln": False,
            "edge_dropout": 0.0,
            "dropout": 0.0,
        }

        self.encoder = model_cls(output_targets,model_configs, **model_attributes)
        self.transform_config = TransformConfigs(mt=None,model=model_configs)
        checkpoint_root = BENCHMARK_ROOT / "al_ckpt"
        task_name = encoder_kwargs.get('task_name', None)
        if task_name == '[C]':
            checkpoint = torch.load(
                checkpoint_root / "al_C.pt",
                map_location="cpu",
            )
        elif task_name == '[O]':
            checkpoint = torch.load(
                checkpoint_root / "al_O.pt",
                map_location="cpu",
            )
        else:
            raise ValueError(f"Unknown task name: {task_name}. Only [C] and [O] are supported.")
        state_dict = checkpoint.get("state_dict", checkpoint)
        load_state_dict(self.encoder, state_dict, strict=False)

        self.readout = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
        )

        
    def forward(self, batch):  # type: ignore[override]
        
        # device = batch.y.device
        # graph_list = []
        
        # batch_size = batch.natoms.shape[0]
        # for idx in range(batch_size):
        #     data = matbench_transform(batch[idx] , config=self.transform_config, training=True)
        #     graph_list.append(data)
        # batch = Batch.from_data_list(graph_list).to(device)
        
        node_emb = self.encoder(batch)
        graph_ebds = node_emb[1]
        output = self.readout(graph_ebds)
        return output


class CatAISubsetTrainer:
    """Train DimeNet++ on a labeled subset and keep the best validation checkpoint."""

    def __init__(self, cfg: ALConfig, dataset: EscnCIFDataset, labeled_idxs: List[int], val_idxs: List[int]):
        self.cfg = cfg
        self.dataset = dataset
        
        self.labeled_idxs = list(labeled_idxs)
        self.val_idxs = list(val_idxs)
        self.device = torch.device("cuda" if cfg.cuda and torch.cuda.is_available() else "cpu")
        
        
        encoder_kwargs = cfg.model_kwargs or {}
        encoder_kwargs['task_name'] = self.cfg.label_dir.split('/')[-1]

        self.model = CatAIRegressor(
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
            # start_time = time.time()
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
            # print(f"Prediction batch time: {time.time() - start_time:.2f} seconds")
        rows.sort(key=lambda r: order_map.get(r["dataset_idx"], -1))
        return pd.DataFrame(rows)


__all__ = ["CatAIRegressor", "CatAISubsetTrainer"]
