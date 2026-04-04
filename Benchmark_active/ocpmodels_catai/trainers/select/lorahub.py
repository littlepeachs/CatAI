import os
import json
import logging
from collections import OrderedDict
from functools import cached_property, partial

import numpy as np
import torch
import nevergrad as ng
from torch.nn.parallel.distributed import DistributedDataParallel
from typing_extensions import override

from ocpmodels_catai.common import distutils
from ocpmodels_catai.common.data_parallel import OCPDataParallel
from ocpmodels_catai.common.registry import registry
from ocpmodels_catai.common.typed_config import TypeAdapter
from ocpmodels_catai.common.utils import scatter_det
from ocpmodels_catai.modules.scaling.util import ensure_fitted
from ocpmodels_catai.trainers.mt.collate import ParallelCollater
from ocpmodels_catai.trainers.mt.config import ModelConfig
from ocpmodels_catai.trainers.ocp_trainer import OCPTrainer

from ..ft.dataset import FTDatasetsConfig, create_ft_datasets
from ..ft.util import load_state_dict

log = logging.getLogger(__name__)


@registry.register_trainer("lorahub")
class LoraHubTrainer(OCPTrainer):
    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_weights(w: np.ndarray) -> np.ndarray:
        """Soft‑max style normalisation so that Σw = 1 (within fp tolerance)."""
        # w = np.exp(w - np.max(w))
        w = np.exp(w)
        w /= np.sum(w) + 1e-12
        return w
    
    
    """Trainer that optimises a linear combination of full modules using
    Nevergrad + k‑NN (cosine, label propagation) evaluated with MAE.
    """

    # ---------------------------------------------------------------------
    # Boilerplate / ctor
    # ---------------------------------------------------------------------
    def __init__(
        self,
        task,
        model,
        outputs,
        dataset,
        optimizer,
        loss_fns,
        eval_metrics,
        identifier,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm=None,
        noddp=False,
        name="ocp",
    ):
        if slurm is None:
            slurm = {}
        super().__init__(
            task,
            model,
            outputs,
            FTDatasetsConfig.from_dict(dataset),
            optimizer,
            loss_fns,
            eval_metrics,
            identifier,
            timestamp_id,
            run_dir,
            is_debug,
            print_every,
            seed,
            logger,
            local_rank,
            amp,
            cpu,
            slurm,
            noddp,
            name,
        )

    @override
    def load(self) -> None:
        self.load_seed_from_config()
        self.load_logger()
        self.load_datasets()
        self.load_task()
        print("=" * 20 + "Start loading model", "=" * 20)
        self.load_model()
        print("=" * 20 + "Finish loading model", "=" * 20)
        print("=" * 20 + "Start loading checkpoint", "=" * 20)
        self.load_checkpoint(self.config["task"]["base_checkpoint"]["src"])
        print("=" * 20 + "Finish loading checkpoint", "=" * 20)
        # print("=" * 20 + "Start loading loss", "=" * 20)
        self.load_loss()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    @property
    def dataset_config(self):
        return self.config["dataset"]

    def load_datasets(self) -> None:
        log.info("Loading datasets")
        self.parallel_collater = ParallelCollater(
            0 if self.cpu else 1,
            otf_graph=self.config["model_attributes"].get("otf_graph", False),
        )

        (self.train_dataset, self.val_dataset, self.test_dataset) = create_ft_datasets(
            self.dataset_config, self.model_config
        )

        # loaders -------------------------------------------------------
        self.flag_loader = None
        if self.train_dataset is not None:
            flag_bs = self.config["task"].get("flag_batchsize", 32)
            sampler = self.get_sampler(self.train_dataset, flag_bs, shuffle=True)
            self.flag_loader = self.get_dataloader(self.train_dataset, sampler)

    # ------------------------------------------------------------------
    # Expert embeddings
    # ------------------------------------------------------------------
    def embed_gen(self, save_outputs: bool = False):
        """Forward pass over flag_loader to obtain node‑pooled expert embeddings
        and labels. Returns (embeddings, labels).
        """
        assert self.flag_loader is not None, "flag_loader unavailable"
        self.model.eval()

        all_embeds, all_labels = [], []
        for batch in self.flag_loader:
            with torch.no_grad():
                _, expert_out = self._forward(batch)  # (T, N, D)
                # print("expert_out.shape:", expert_out.shape)

                index = batch[0].batch
                natoms = batch[0].natoms.shape[0]
                tgt_name = list(self.output_targets.keys())[0]
                labels = batch[0][tgt_name].unsqueeze(1)  # (M,1)
          
                token_feat = expert_out.squeeze(1)
                pooled = scatter_det(token_feat, index, dim=0, dim_size=natoms, reduce="mean")
                # print(pooled.shape)

                all_embeds.append(pooled)
                all_labels.append(labels)

        embeds = torch.cat(all_embeds, dim=0)          # (N, D)
        labels = torch.cat(all_labels, dim=0)          # (N,1)
        # print("embeds.shape:", embeds.shape)
        # print("labels.shape:", labels.shape)
        return embeds, labels

    # ------------------------------------------------------------------
    # k‑NN metric (cosine + label propagation + MAE)
    # ------------------------------------------------------------------
    @staticmethod
    def _knn_metric(embeds: torch.Tensor, labels: torch.Tensor, k: int = 5, tau: float = 0.2) -> float:
        x = torch.nn.functional.normalize(embeds.float(), dim=1)
        # print("x.shape:", x.shape) (bsz, dim)
        y = labels.float()
        sim = x @ x.t()
        adj = torch.exp(sim / tau) - torch.diag_embed(torch.diag(sim))
        _, topk_idx = torch.topk(adj, k=k, dim=1)
        mask = torch.zeros_like(adj).scatter_(1, topk_idx, 1.0)
        adj = adj * mask
        adj = adj / (adj.sum(dim=1, keepdim=True) + 1e-12)
        pred = adj @ y
        mae = torch.mean(torch.abs(pred - y)).item()
        return mae

    # ------------------------------------------------------------------
    # Nevergrad objective
    # ------------------------------------------------------------------
    def _score_module_weights(self, weights: np.ndarray, step: int | None = None) -> float:
        # --- 1. normalise on simplex ---
        # w = self._normalize_weights(weights)
        w = weights
        if step is not None:
            log.info(f"NG step {step}: weights = {w.round(4)}  (Σ={w.sum():.4f})")

        # --- 2. evaluate ---
        self._load_weighted_checkpoint(w)
        embeds, labels = self.embed_gen(save_outputs=False)
        loss = self._knn_metric(embeds, labels, k=5)
        reg = 0.05 * np.mean(np.abs(w))
        return loss + reg

    # ------------------------------------------------------------------
    # Weight optimisation
    # ------------------------------------------------------------------
    def _optimize_module_weights(self):
        """Run Nevergrad optimisation with weights constrained to sum to 1.
        If `task.enforce_simplex` in config, use NG's `Simplex` so the
        optimizer *natively* works on the Σ=1 simplex; otherwise fall back to
        unconstrained array + soft‑max inside the objective.
        """
        mod_cnt = 18
        budget = self.config["task"].get("opt_budget", 40)
        # Nevergrad does not provide a native Simplex parameter (Σw = 1),
        # so we always search in R^d and rely on the soft‑max in the objective.
        instrum = ng.p.Array(init=[0.0] * mod_cnt, lower=[-1.5] * mod_cnt, upper=[1.5] * mod_cnt)

        opt = ng.optimizers.NGOpt(parametrization=instrum, budget=budget)

        # Wrap score fn to inject iteration counter for logging
        counter = {"i": 0}
        def score_with_step(x):
            val = self._score_module_weights(x, step=counter["i"])
            counter["i"] += 1
            return val

        rec = opt.minimize(score_with_step, verbosity=1)
        best = self._normalize_weights(rec.value)
        log.info(f"Best weights (Σ={best.sum():.4f}): {best}")
        self._load_weighted_checkpoint(best)

        # save the results
        task_name = self.config["task"].get("data_name", "expt_bandgap")
        task_split = self.config["task"].get("split", "0")
        
        # update JSON for this split
        # TEST
        json_path = f'json/lorahub/split_{task_split}_default.json'
        # read or init
        try:
            with open(json_path) as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}
        # write this task
        data[task_name] = best.tolist()
        # overwrite
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_weighted_checkpoint(self, weights):
        module_names = [
            'mp_eform','mp_bandgap','n_Seebeck','n_avg_eff_mass','n_th_cond',
            'p_Seebeck','p_avg_eff_mass','p_e_cond','n_e_cond','p_th_cond',
            'mp_gvrh','castelli_eform','jarvis_gvrh','jarvis_bandgap','jarvis_eform',
            'jarvis_kvrh','mp_kvrh', 'jarvis_dielectric_opt'
        ]
        assert len(weights) == len(module_names)
        module_base_path=self.config["task"].get("module_base_path", None)
        merged = None
        for name, w in zip(module_names, weights):
            # path = f'/path/to/catai_data/catai_hub/{name}/best_checkpoint.pt' # TEST
            path = f'{module_base_path}/{name}/best_checkpoint.pt'
            ckpt = torch.load(path, map_location='cpu')
            sd = ckpt['state_dict']
            if merged is None:
                merged = OrderedDict({k: torch.zeros_like(v) for k, v in sd.items()})
            for k in merged:
                if 'scale_factor' in k:
                    merged[k] = sd[k]
                    continue
                if 'module_dict.y' in k:
                    continue
                merged[k] += sd[k] * w
        # print(merged["module.module.int_blocks.0.trip_interaction.scale_rbf.scale_factor"])
        _ = load_state_dict(self.model, merged)

    # ------------------------------------------------------------------
    # Train entry point
    # ------------------------------------------------------------------
    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)
        if self.config["task"].get("optimize_modules", False):
            print("Optimizing module weights with lorahub")
            self._optimize_module_weights()
        elif self.config["task"].get("expert_select", False):
            self.embed_gen(save_outputs=True)
        else:
            log.warning("Nothing to do – set 'optimize_modules' or 'expert_select' in config.")

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    @cached_property
    def model_config(self):
        cfg = self.config["model_attributes"].copy()
        cfg["name"] = self.config["model"]
        return TypeAdapter(ModelConfig).validate_python(cfg)

    def load_model(self):
        if distutils.is_master():
            log.info(f"Loading model: {self.config['model']}")
        self.model = registry.get_model_class(self.config["model"])(
            self.output_targets, self.model_config, **self.config["model_attributes"]
        ).to(self.device)
        self.model = OCPDataParallel(self.model, output_device=self.device, num_gpus=1 if not self.cpu else 0)
        if distutils.initialized() and not self.config["noddp"]:
            self.model = DistributedDataParallel(self.model, device_ids=[self.device], find_unused_parameters=True)