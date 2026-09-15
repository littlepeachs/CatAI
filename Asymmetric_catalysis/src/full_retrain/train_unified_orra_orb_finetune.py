#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from full_retrain.aho_amc.common import sha256_file
from full_retrain.aho_amc.multilevel_orb import (
    forward_orb_with_node_feature_levels,
    validate_orb_stack_indices,
)
from full_retrain.aho_amc.reactant_conformer_consistency import (
    aggregate_fixed_conformer_view,
    component_ids_for_role,
    mix_role_token_view,
    prediction_consistency_loss,
    sample_single_conformer_indices,
)
from full_retrain.cpa_amc.catalyst_scaffold_augmentation import (
    catalyst_h8_scaffold_view,
)
from full_retrain.data import ROOT, load_task
from full_retrain.train_orb3d_atom_relation_attention import (
    AtomRelationTensorBundle,
    build_tensor_bundle,
)
from full_retrain.train_orb3d_site_attention import TOKEN_TYPE_IDS, load_atom_token_store, split_frames
from full_retrain.train_orb3d_site_attention_regression import (
    group_validation_frames_by_strategy,
    store_with_source_index_mapping,
)
from full_retrain.train_unified_orra import (
    ComponentDensityProfile,
    TASKS,
    TOKEN_DIR_NAMES,
    initialize_unified_model,
    indexed_true_values,
    metric_score,
    task_metrics,
    train_only_condition_prior,
)
from full_retrain.unified_orra_model import UnifiedORRAModel


CIP_ORIENTATION_IDS = {
    "none": 0,
    "R": 1,
    "S": 2,
    "other": 3,
}

SMILES_BYTE_TOKEN_IDS = {
    "pad": 0,
    "byte_offset": 1,
    "cls": 257,
    "sep": 258,
}
SMILES_BYTE_VOCAB_SIZE = 259


def concatenate_atom_relation_bundles(
    *bundles: AtomRelationTensorBundle,
) -> AtomRelationTensorBundle:
    """Concatenate split bundles for leakage-safe final train+val fitting."""
    if not bundles:
        raise ValueError("At least one tensor bundle is required")
    return AtomRelationTensorBundle(
        x=torch.cat([bundle.x for bundle in bundles], dim=0),
        numeric=torch.cat(
            [bundle.numeric for bundle in bundles],
            dim=0,
        ),
        token_mask=torch.cat(
            [bundle.token_mask for bundle in bundles],
            dim=0,
        ),
        role_ids=torch.cat(
            [bundle.role_ids for bundle in bundles],
            dim=0,
        ),
        token_type_ids=torch.cat(
            [bundle.token_type_ids for bundle in bundles],
            dim=0,
        ),
        atomic_numbers=torch.cat(
            [bundle.atomic_numbers for bundle in bundles],
            dim=0,
        ),
        component_ids=torch.cat(
            [bundle.component_ids for bundle in bundles],
            dim=0,
        ),
        y=torch.cat([bundle.y for bundle in bundles], dim=0),
        row_index=np.concatenate(
            [bundle.row_index for bundle in bundles],
            axis=0,
        ),
    )


def safe_learning_rate_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Return a reporting-only LR ratio without failing on a frozen group."""
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def validate_smiles_sequence_adapter_config(
    *,
    enabled: bool,
    auxiliary_enabled: bool,
    auxiliary_weight: float,
    auxiliary_warmup_epochs: int,
    auxiliary_post_warmup_lr: float,
    readout_train_scope: str,
    vocab_size: int,
    max_length: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    dropout: float,
    residual_multiplier: float,
) -> bool:
    """Fail closed on partial or malformed sequence-adapter configurations."""
    expected_scope = "smiles_sequence_adapter"
    if not enabled:
        if readout_train_scope == expected_scope:
            raise ValueError(
                f"--readout-train-scope {expected_scope} requires "
                "--smiles-sequence-adapter"
            )
        if (
            auxiliary_enabled
            or auxiliary_weight != 0.0
            or auxiliary_warmup_epochs != 0
            or auxiliary_post_warmup_lr != 0.0
        ):
            raise ValueError(
                "SMILES sequence auxiliary settings require "
                "--smiles-sequence-adapter"
            )
        return False
    if vocab_size != SMILES_BYTE_VOCAB_SIZE:
        raise ValueError(
            "The fixed byte-SMILES tokenizer requires vocabulary size "
            f"{SMILES_BYTE_VOCAB_SIZE}"
        )
    if max_length <= 3:
        raise ValueError("--smiles-sequence-max-length must exceed three")
    if (
        hidden_dim <= 0
        or num_heads <= 0
        or hidden_dim % num_heads != 0
    ):
        raise ValueError(
            "SMILES sequence hidden dimension must be positive and "
            "divisible by the number of heads"
        )
    if num_layers <= 0:
        raise ValueError("--smiles-sequence-num-layers must be positive")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("--smiles-sequence-dropout must be in [0, 1)")
    if residual_multiplier < 0.0:
        raise ValueError(
            "--smiles-sequence-residual-multiplier must be non-negative"
        )
    if auxiliary_enabled:
        if readout_train_scope != expected_scope:
            raise ValueError(
                "--smiles-sequence-auxiliary requires "
                "--readout-train-scope smiles_sequence_adapter"
            )
        if auxiliary_weight <= 0.0:
            raise ValueError(
                "--smiles-sequence-auxiliary-weight must be positive"
            )
        if auxiliary_warmup_epochs < 0:
            raise ValueError(
                "--smiles-sequence-auxiliary-warmup-epochs must be "
                "non-negative"
            )
        if auxiliary_post_warmup_lr < 0.0:
            raise ValueError(
                "--smiles-sequence-auxiliary-post-warmup-lr must be "
                "non-negative"
            )
    elif (
        auxiliary_weight != 0.0
        or auxiliary_warmup_epochs != 0
        or auxiliary_post_warmup_lr != 0.0
    ):
        raise ValueError(
            "Non-default SMILES sequence auxiliary settings require "
            "--smiles-sequence-auxiliary"
        )
    return True


def validate_dual_backbone_feature_fusion_config(
    *,
    enabled: bool,
    secondary_atom_token_dir: Path | None,
    backbone_feature_source: str,
    backbone_family: str,
    conformer_feature_mode: str,
    conformer_aggregation: str,
    multi_level_orb_attention: bool,
    reactant_conformer_consistency_enabled: bool,
    hidden_dim: int,
    role_gate_enabled: bool = False,
) -> bool:
    """Validate the default-off dual-view cache or online path."""
    if not enabled:
        if role_gate_enabled:
            raise ValueError(
                "--dual-backbone-feature-role-gate requires "
                "--dual-backbone-feature-fusion"
            )
        if secondary_atom_token_dir is not None:
            raise ValueError(
                "--secondary-atom-token-dir requires "
                "--dual-backbone-feature-fusion"
            )
        return False
    if backbone_feature_source not in {"token_cache", "online"}:
        raise ValueError(
            "dual backbone feature fusion requires token_cache or online "
            "backbone features"
        )
    if backbone_family != "jmp":
        raise ValueError(
            "dual backbone feature fusion currently supports only the JMP "
            "AMC-Fuse cache screen"
        )
    if (
        backbone_feature_source == "token_cache"
        and secondary_atom_token_dir is None
    ):
        raise ValueError(
            "cached dual backbone feature fusion requires "
            "--secondary-atom-token-dir"
        )
    if (
        backbone_feature_source == "online"
        and secondary_atom_token_dir is not None
    ):
        raise ValueError(
            "online dual backbone feature fusion captures both views in one "
            "JMP traversal and must not receive --secondary-atom-token-dir"
        )
    if conformer_feature_mode != "mean_std_max":
        raise ValueError(
            "dual backbone feature fusion requires "
            "--conformer-feature-mode mean_std_max"
        )
    if conformer_aggregation != "fixed":
        raise ValueError(
            "dual backbone feature fusion requires fixed conformer aggregation"
        )
    if multi_level_orb_attention:
        raise ValueError(
            "dual backbone feature fusion is incompatible with multi-level "
            "Orb attention"
        )
    if reactant_conformer_consistency_enabled:
        raise ValueError(
            "dual backbone feature fusion is incompatible with reactant "
            "conformer consistency in the frozen-cache screen"
        )
    if hidden_dim <= 0:
        raise ValueError(
            "--dual-backbone-feature-fusion-hidden-dim must be positive"
        )
    return True


def validate_dual_atom_token_stores(
    *,
    primary_store,
    secondary_store,
    primary_manifest: dict[str, object],
    secondary_manifest: dict[str, object],
    task: str,
) -> dict[str, object]:
    """Fail closed unless two cache views have identical provenance/schema."""
    for label, manifest in (
        ("primary", primary_manifest),
        ("secondary", secondary_manifest),
    ):
        if manifest.get("all_pass") is not True:
            raise RuntimeError(f"{label} token-cache manifest did not pass")
        if manifest.get("task") != task:
            raise RuntimeError(
                f"{label} token-cache task mismatch: "
                f"{manifest.get('task')} != {task}"
            )
        if manifest.get("jmp_checkpoint_task") != task:
            raise RuntimeError(
                f"{label} JMP checkpoint task mismatch: "
                f"{manifest.get('jmp_checkpoint_task')} != {task}"
            )
        if manifest.get("test_labels_or_predictions_used") is not False:
            raise RuntimeError(f"{label} token cache is not test-sealed")
        if manifest.get("ensemble_size") != 1:
            raise RuntimeError(
                f"{label} token cache does not declare ensemble_size=1"
            )
        if manifest.get("prediction_space_ensemble") is not False:
            raise RuntimeError(
                f"{label} token cache used prediction-space ensembling"
            )
        if manifest.get("single_materialized_jmp_checkpoint") is not True:
            raise RuntimeError(
                f"{label} token cache is not from one materialized JMP checkpoint"
            )
    primary_view = primary_manifest.get(
        "node_feature_view",
        "node_embedding",
    )
    secondary_view = secondary_manifest.get("node_feature_view")
    if primary_view != "node_embedding":
        raise RuntimeError(
            f"Primary token cache must use node_embedding, got {primary_view}"
        )
    if secondary_view != "final_interaction":
        raise RuntimeError(
            "Secondary token cache must use final_interaction, got "
            f"{secondary_view}"
        )
    primary_checkpoint_hash = primary_manifest.get(
        "jmp_checkpoint_sha256"
    )
    secondary_checkpoint_hash = secondary_manifest.get(
        "jmp_checkpoint_sha256"
    )
    if (
        not primary_checkpoint_hash
        or primary_checkpoint_hash != secondary_checkpoint_hash
    ):
        raise RuntimeError(
            "Dual token caches do not share the same JMP checkpoint hash"
        )
    if primary_store.node_features.shape != secondary_store.node_features.shape:
        raise RuntimeError(
            "Dual token-cache node feature shapes do not match"
        )
    aligned_array_names = (
        "atom_mask",
        "conformer_mask",
        "atomic_numbers",
        "atom_descriptors",
        "positions",
    )
    for name in aligned_array_names:
        primary_value = np.asarray(getattr(primary_store, name))
        secondary_value = np.asarray(getattr(secondary_store, name))
        if not np.array_equal(primary_value, secondary_value):
            raise RuntimeError(
                f"Dual token-cache alignment mismatch for {name}"
            )
    if not primary_store.metadata.equals(secondary_store.metadata):
        raise RuntimeError("Dual token-cache component metadata do not match")
    if not primary_store.mapping.equals(secondary_store.mapping):
        raise RuntimeError("Dual token-cache reaction mappings do not match")
    return {
        "task": task,
        "primary_node_feature_view": str(primary_view),
        "secondary_node_feature_view": str(secondary_view),
        "jmp_checkpoint_sha256": str(primary_checkpoint_hash),
        "node_feature_shape": list(primary_store.node_features.shape),
        "aligned_array_names": list(aligned_array_names),
        "metadata_equal": True,
        "mapping_equal": True,
        "test_labels_or_predictions_used": False,
        "ensemble_size": 1,
    }


def configure_backbone_trainability(
    backbone: nn.Module,
    *,
    family: str,
    scope: str,
    unfreeze_orb_stacks: int,
) -> list[str]:
    """Select trainable backbone tensors without changing its architecture."""
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    if scope == "frozen":
        return []
    if scope == "full_backbone":
        for parameter in backbone.parameters():
            parameter.requires_grad = True
    elif family == "orb" and scope == "stacks":
        stacks = getattr(backbone, "gnn_stacks", None)
        if stacks is None:
            raise ValueError("Orb stacks scope requires backbone.gnn_stacks")
        if unfreeze_orb_stacks < 1:
            raise ValueError("--unfreeze-orb-stacks must be at least 1")
        if unfreeze_orb_stacks > len(stacks):
            raise ValueError(
                f"Cannot unfreeze {unfreeze_orb_stacks} stacks; "
                f"backbone has {len(stacks)}"
            )
        for stack in stacks[-unfreeze_orb_stacks:]:
            for parameter in stack.parameters():
                parameter.requires_grad = True
    elif family == "jmp" and scope in {
        "jmp_output",
        "jmp_last_block",
        "jmp_last_two_blocks",
    }:
        jmp = getattr(backbone, "backbone", None)
        if jmp is None:
            raise ValueError(
                "JMP output scopes require a JMPNodeBackbone wrapper"
            )
        required = ("out_mlp_E", "out_blocks", "int_blocks")
        missing = [
            name for name in required if not hasattr(jmp, name)
        ]
        if missing:
            raise ValueError(
                "JMP backbone is missing trainability modules: "
                f"{missing}"
            )
        modules = [jmp.out_mlp_E, jmp.out_blocks[-1]]
        if scope == "jmp_last_block":
            modules.append(jmp.int_blocks[-1])
        elif scope == "jmp_last_two_blocks":
            if len(jmp.out_blocks) < 2 or len(jmp.int_blocks) < 2:
                raise ValueError(
                    "JMP last-two-block scope requires at least two output "
                    "and interaction blocks"
                )
            modules.extend(
                [
                    jmp.out_blocks[-2],
                    jmp.int_blocks[-2],
                    jmp.int_blocks[-1],
                ]
            )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad = True
    else:
        raise ValueError(
            f"Backbone scope {scope!r} is incompatible with family "
            f"{family!r}"
        )

    return [
        name
        for name, parameter in backbone.named_parameters()
        if parameter.requires_grad
    ]


def configure_ddg_joint_adapter_trainability(
    readout: UnifiedORRAModel,
) -> list[str]:
    """Train A6 and A7 together while leaving their ORRA parent unchanged."""
    if readout.ddg_sign_magnitude_adapter is None:
        raise ValueError(
            "joint DDG adapter scope requires --ddg-sign-magnitude"
        )
    if readout.ddg_residual_calibrator is None:
        raise ValueError(
            "joint DDG adapter scope requires --ddg-residual-calibrator"
        )
    for parameter in readout.parameters():
        parameter.requires_grad = False
    for module in (
        readout.ddg_sign_magnitude_adapter,
        readout.ddg_residual_calibrator,
    ):
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [
        name
        for name, parameter in readout.named_parameters()
        if parameter.requires_grad
    ]


def configure_ddg_nonlinear_calibrator_trainability(
    readout: UnifiedORRAModel,
) -> list[str]:
    """Train only the incremental nonlinear A8 branch over a frozen A7."""
    calibrator = readout.ddg_residual_calibrator
    if calibrator is None:
        raise ValueError(
            "nonlinear DDG calibrator scope requires "
            "--ddg-residual-calibrator"
        )
    if calibrator.nonlinear_adapter is None:
        raise ValueError(
            "nonlinear DDG calibrator scope requires a positive "
            "--ddg-residual-calibrator-nonlinear-hidden-dim"
        )
    for parameter in readout.parameters():
        parameter.requires_grad = False
    for parameter in calibrator.nonlinear_adapter.parameters():
        parameter.requires_grad = True
    return [
        name
        for name, parameter in readout.named_parameters()
        if parameter.requires_grad
    ]


def configure_ddg_major_summary_trainability(
    readout: UnifiedORRAModel,
) -> list[str]:
    """Train only the default-off A9 classifier-summary residual."""
    adapter = readout.ddg_major_summary_adapter
    if adapter is None:
        raise ValueError(
            "DDG Major-summary scope requires --ddg-major-summary"
        )
    for parameter in readout.parameters():
        parameter.requires_grad = False
    for parameter in adapter.parameters():
        parameter.requires_grad = True
    return [
        name
        for name, parameter in readout.named_parameters()
        if parameter.requires_grad
    ]


def configure_ddg_major_summary_training_phase(
    readout: UnifiedORRAModel,
    *,
    residual_active: bool,
) -> list[str]:
    """Warm up A9 classifiers while its zero residual remains frozen."""
    adapter = readout.ddg_major_summary_adapter
    if adapter is None:
        raise ValueError(
            "DDG Major-summary training phase requires "
            "--ddg-major-summary"
        )
    for parameter in adapter.fusion_classifier.parameters():
        parameter.requires_grad = True
    for classifier in adapter.role_classifiers:
        for parameter in classifier.parameters():
            parameter.requires_grad = True
    for parameter in adapter.residual_head.parameters():
        parameter.requires_grad = bool(residual_active)
    return [
        name
        for name, parameter in readout.named_parameters()
        if parameter.requires_grad
    ]


def configure_smiles_sequence_training_phase(
    readout: UnifiedORRAModel,
    *,
    residual_active: bool,
) -> list[str]:
    """Warm up the sequence representation before enabling its residual."""
    adapter = readout.smiles_sequence_adapter
    if adapter is None or adapter.auxiliary_head is None:
        raise ValueError(
            "SMILES auxiliary training phase requires the sequence adapter "
            "and auxiliary head"
        )
    for parameter in adapter.parameters():
        parameter.requires_grad = False
    warmup_modules = (
        adapter.token_embedding,
        adapter.position_embedding,
        adapter.encoder,
        adapter.convolutions,
        adapter.sequence_projection,
        adapter.auxiliary_head,
    )
    for module in warmup_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True
    if residual_active:
        for module in (
            adapter.fusion_projection,
            adapter.residual_head,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
    return [
        name
        for name, parameter in readout.named_parameters()
        if parameter.requires_grad
    ]


def configure_readout_training_mode(
    readout: UnifiedORRAModel,
    *,
    readout_train_scope: str,
    module_mode: str,
) -> None:
    """Configure readout module modes without changing gradient scopes.

    ``train`` preserves the historical behavior, including the special
    deterministic-parent handling used by the A7/A8 scopes.  ``eval`` keeps
    every readout module and running statistic fixed while still allowing
    gradients to pass through the readout into an online backbone.
    """
    if module_mode not in {"train", "eval"}:
        raise ValueError(
            "readout module mode must be one of {'train', 'eval'}"
        )
    if module_mode == "eval":
        readout.eval()
        return

    if readout_train_scope == "cip_orientation_embedding":
        readout.eval()
        if readout.cip_orientation_embedding is None:
            raise RuntimeError("CIP training scope has no embedding")
        readout.cip_orientation_embedding.train()
        return

    if readout_train_scope == "smiles_sequence_adapter":
        readout.eval()
        if readout.smiles_sequence_adapter is None:
            raise RuntimeError("SMILES training scope has no adapter")
        readout.smiles_sequence_adapter.train()
        return

    if readout_train_scope == "regression_head":
        readout.eval()
        readout.raw_head.train()
        return

    if readout_train_scope == "regression_output":
        readout.eval()
        return

    readout.train()
    if readout_train_scope not in {
        "ddg_residual_calibrator",
        "ddg_residual_nonlinear_adapter",
        "ddg_sign_magnitude_and_residual_calibrator",
        "ddg_major_summary_adapter",
    }:
        return

    # A7 calibrates deterministic parent diagnostics. Keep every frozen
    # parent dropout layer in evaluation mode while allowing the selected
    # A6/A7 modules and A7 running feature statistics to update.
    readout.eval()
    if readout_train_scope == "ddg_major_summary_adapter":
        if readout.ddg_major_summary_adapter is None:
            raise RuntimeError("A9 training scope has no adapter")
        readout.ddg_major_summary_adapter.train()
        return
    if readout.ddg_residual_calibrator is None:
        raise RuntimeError("A7 training scope has no calibrator")
    if readout_train_scope == "ddg_residual_nonlinear_adapter":
        nonlinear_adapter = (
            readout.ddg_residual_calibrator.nonlinear_adapter
        )
        if nonlinear_adapter is None:
            raise RuntimeError("A8 training scope has no nonlinear adapter")
        # Keep A7 running statistics and linear coefficients fixed while
        # training only the default-off nonlinear residual branch.
        nonlinear_adapter.train()
    else:
        readout.ddg_residual_calibrator.train()
    if (
        readout_train_scope
        == "ddg_sign_magnitude_and_residual_calibrator"
    ):
        if readout.ddg_sign_magnitude_adapter is None:
            raise RuntimeError("joint DDG scope has no A6 adapter")
        readout.ddg_sign_magnitude_adapter.train()


def validation_selection_score(
    metrics: dict[str, float],
    metric: str,
    *,
    major_accuracy_gate: float,
    major_auroc_gate: float,
    ddg_r2_gate: float = 0.58,
    ddg_mae_gate: float = 0.80,
) -> float:
    """Return a minimized validation score while preserving legacy metrics."""
    if metric == "major_gate_balance":
        if major_accuracy_gate <= 0.0 or major_auroc_gate <= 0.0:
            raise ValueError("Major validation gates must be positive")
        accuracy_gap = 1.0 - float(metrics["accuracy"]) / major_accuracy_gate
        auroc_gap = 1.0 - float(metrics["auroc"]) / major_auroc_gate
        return max(accuracy_gap, auroc_gap)
    if metric == "ddg_gate_balance":
        if ddg_r2_gate <= 0.0 or ddg_mae_gate <= 0.0:
            raise ValueError("DDG validation gates must be positive")
        r2_gap = 1.0 - float(metrics["r2"]) / ddg_r2_gate
        mae_gap = float(metrics["mae"]) / ddg_mae_gate - 1.0
        return max(r2_gap, mae_gap)
    return metric_score(metrics, metric)


def validate_checkpoint_selection_split(
    *,
    checkpoint_selection_split: str,
    skip_test_evaluation: bool,
    acknowledge_observed_test_selection: bool,
) -> str:
    """Fail closed before allowing explicitly test-guided checkpoint choice."""
    if checkpoint_selection_split not in {"validation", "test"}:
        raise ValueError(
            "--checkpoint-selection-split must be validation or test"
        )
    if checkpoint_selection_split == "test":
        if skip_test_evaluation:
            raise ValueError(
                "--checkpoint-selection-split test is incompatible with "
                "--skip-test-evaluation"
            )
        if not acknowledge_observed_test_selection:
            raise ValueError(
                "--checkpoint-selection-split test requires "
                "--acknowledge-observed-test-selection"
            )
    return checkpoint_selection_split


def validate_final_retrain_train_plus_val_config(
    *,
    enabled: bool,
    checkpoint_selection_split: str,
    export_fusion_features: bool,
) -> bool:
    """Fail closed for the optional fixed-epoch final-retrain protocol."""
    if not enabled:
        return False
    if checkpoint_selection_split != "validation":
        raise ValueError(
            "--final-retrain-train-plus-val is incompatible with "
            "--checkpoint-selection-split test"
        )
    if export_fusion_features:
        raise ValueError(
            "--final-retrain-train-plus-val is incompatible with "
            "--export-fusion-features"
        )
    return True


def validate_epoch_ema_decay(decay: float) -> bool:
    """Validate the default-off epoch-level parameter EMA."""
    if not 0.0 <= decay < 1.0:
        raise ValueError("--epoch-ema-decay must be in [0, 1)")
    return decay > 0.0


def validate_cip_orientation_embedding_config(
    *,
    enabled: bool,
    task: str,
    backbone_feature_source: str,
    backbone_train_scope: str,
    backbone_lr: float,
    readout_train_scope: str,
    readout_module_mode: str,
    resume_finetune_checkpoint: Path | None,
    resume_readout_load_mode: str,
    resume_backbone_load_mode: str,
) -> bool:
    """Fail closed around the single J199 embedding-only causal screen."""
    expected_scope = "cip_orientation_embedding"
    if not enabled:
        if readout_train_scope == expected_scope:
            raise ValueError(
                f"--readout-train-scope {expected_scope} requires "
                "--cip-orientation-embedding"
            )
        return False
    if task != "aho_major":
        raise ValueError(
            "--cip-orientation-embedding is only defined for aho_major"
        )
    if backbone_feature_source != "token_cache":
        raise ValueError(
            "--cip-orientation-embedding currently requires the sealed "
            "token-cache causal screen"
        )
    if backbone_train_scope != "frozen" or backbone_lr != 0.0:
        raise ValueError(
            "--cip-orientation-embedding requires a frozen backbone and "
            "--orb-lr 0"
        )
    if readout_train_scope != expected_scope:
        raise ValueError(
            "--cip-orientation-embedding requires "
            f"--readout-train-scope {expected_scope}"
        )
    if readout_module_mode != "eval":
        raise ValueError(
            "--cip-orientation-embedding requires "
            "--readout-module-mode eval"
        )
    if resume_finetune_checkpoint is None:
        raise ValueError(
            "--cip-orientation-embedding requires a parent "
            "--resume-finetune-checkpoint"
        )
    if resume_readout_load_mode != "compatible":
        raise ValueError(
            "--cip-orientation-embedding requires "
            "--resume-readout-load-mode compatible"
        )
    if resume_backbone_load_mode != "skip":
        raise ValueError(
            "--cip-orientation-embedding requires "
            "--resume-backbone-load-mode skip"
        )
    return True


def update_epoch_ema_state(
    ema_state: dict[str, torch.Tensor],
    live_state: dict[str, torch.Tensor],
    *,
    decay: float,
) -> None:
    """Update one CPU state dict in place without averaging integer buffers."""
    validate_epoch_ema_decay(decay)
    if set(ema_state) != set(live_state):
        raise ValueError("EMA/live state keys differ")
    with torch.no_grad():
        for key, ema_value in ema_state.items():
            live_value = live_state[key]
            if (
                ema_value.shape != live_value.shape
                or ema_value.dtype != live_value.dtype
            ):
                raise ValueError(f"EMA/live tensor schema differs for {key}")
            if ema_value.is_floating_point() or ema_value.is_complex():
                ema_value.mul_(decay).add_(live_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(live_value)


def resolve_scheduler_metric(
    requested_metric: str,
    *,
    selection_metric: str,
    task_type: str,
) -> str:
    """Resolve a scheduler monitor while keeping selection-coupled default."""
    if requested_metric == "selection":
        return selection_metric
    if requested_metric == "auto":
        return "accuracy" if task_type == "classification" else "mae"
    return requested_metric


def parse_stack_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Orb stack indices must be comma-separated integers"
        ) from error
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError(
            "Orb stack indices must be non-negative comma-separated integers"
        )
    if tuple(sorted(set(indices))) != indices:
        raise argparse.ArgumentTypeError(
            "Orb stack indices must be unique and strictly increasing"
        )
    return indices


def validate_reactant_conformer_consistency_config(
    *,
    probability: float,
    weight: float,
    huber_delta: float,
    conformer_aggregation: str,
    multi_level_orb_attention: bool,
) -> bool:
    """Validate the default-off train-only conformer consistency branch."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "--reactant-conformer-view-probability must be in [0, 1]"
        )
    if weight < 0.0:
        raise ValueError(
            "--reactant-conformer-consistency-weight must be non-negative"
        )
    if huber_delta <= 0.0:
        raise ValueError(
            "--reactant-conformer-consistency-huber-delta must be positive"
        )
    probability_enabled = probability > 0.0
    weight_enabled = weight > 0.0
    if probability_enabled != weight_enabled:
        raise ValueError(
            "Reactant conformer consistency requires both a positive "
            "--reactant-conformer-view-probability and a positive "
            "--reactant-conformer-consistency-weight, or both zero"
        )
    if probability_enabled and conformer_aggregation != "fixed":
        raise ValueError(
            "Reactant conformer consistency requires "
            "--conformer-aggregation fixed"
        )
    if probability_enabled and multi_level_orb_attention:
        raise ValueError(
            "Reactant conformer consistency is incompatible with "
            "--multi-level-orb-attention"
        )
    return probability_enabled


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
            "End-to-end Unified ORRA with a locally loaded Orb-v2 backbone. "
            "The pretrained Orb layers use a lower learning rate than the new readout."
        )
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument(
        "--split",
        default="paper_exact_test",
        choices=["paper_exact_test", "paper_random42", "example"],
    )
    parser.add_argument("--atom-token-dir", type=Path, default=None)
    parser.add_argument("--secondary-atom-token-dir", type=Path, default=None)
    parser.add_argument("--mapping-task", default="")
    parser.add_argument(
        "--orb-checkpoint",
        type=Path,
        default=Path("/root/autodl-tmp/checkpoints_orb/orb-v2-20241011.ckpt"),
    )
    parser.add_argument(
        "--backbone-family",
        default="orb",
        choices=["orb", "jmp"],
        help=(
            "Provenance label for the single atomistic backbone. The default "
            "keeps the historical Orb path unchanged."
        ),
    )
    parser.add_argument(
        "--backbone-feature-source",
        default="online",
        choices=["online", "token_cache"],
        help=(
            "Use the historical differentiable online backbone forward, or "
            "a label-free atom-token cache extracted from one materialized "
            "checkpoint. Cache mode is default-off and readout-only."
        ),
    )
    parser.add_argument("--conformer-cache", type=Path, default=None)
    parser.add_argument(
        "--conformer-seed",
        type=int,
        default=20260722,
        help="Fixed geometry seed; keep equal to the atom-token extraction seed.",
    )
    parser.add_argument(
        "--conformer-feature-mode",
        default="mean",
        choices=["mean", "mean_std", "mean_std_max"],
        help="Aggregate online Orb atom features across the same three conformers.",
    )
    parser.add_argument(
        "--conformer-aggregation",
        default="fixed",
        choices=["fixed", "attention"],
        help=(
            "Use the original fixed conformer mean, or a shared trainable "
            "per-atom conformer attention. The default preserves A0 exactly."
        ),
    )
    parser.add_argument(
        "--conformer-attention-hidden-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--dual-backbone-feature-fusion",
        action="store_true",
        help=(
            "Default-off zero-initialized residual fusion of aligned JMP "
            "node-embedding and final-interaction token caches."
        ),
    )
    parser.add_argument(
        "--dual-backbone-feature-fusion-hidden-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--dual-backbone-feature-role-gate",
        action="store_true",
        help=(
            "Default-off per-role scaling of an enabled dual-backbone "
            "residual. All scales initialize to exactly one."
        ),
    )
    parser.add_argument(
        "--cip-orientation-embedding",
        action="store_true",
        help=(
            "Default-off zero-initialized Stage-2 atom embedding for "
            "0=none, 1=R, 2=S, 3=other CIP orientation IDs."
        ),
    )
    parser.add_argument(
        "--smiles-sequence-adapter",
        action="store_true",
        help=(
            "Default-off zero-initialized byte-SMILES Transformer/CNN "
            "residual over the existing AMC-ORRA fusion."
        ),
    )
    parser.add_argument(
        "--smiles-sequence-vocab-size",
        type=int,
        default=SMILES_BYTE_VOCAB_SIZE,
        help="Fixed byte-token vocabulary size; changing it is rejected.",
    )
    parser.add_argument(
        "--smiles-sequence-max-length",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--smiles-sequence-hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--smiles-sequence-num-layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--smiles-sequence-num-heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--smiles-sequence-dropout",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--smiles-sequence-residual-multiplier",
        type=float,
        default=3.0,
        help=(
            "Bound on the normalized/logit residual contributed by the "
            "default-off SMILES branch."
        ),
    )
    parser.add_argument(
        "--smiles-sequence-catalyst-h8-augmentation",
        action="store_true",
        help=(
            "Default-off, train-only CPA Catalyst augmentation: duplicate "
            "each training example with the two outer catalyst scaffold "
            "rings partially hydrogenated while preserving the original "
            "AMC-Fuse tokens, target, and all substituents. Validation and "
            "test inputs are never augmented."
        ),
    )
    parser.add_argument(
        "--smiles-sequence-catalyst-h8-target-mode",
        choices=["copy_target", "condition_quantile"],
        default="copy_target",
        help=(
            "Training target for the default-off Catalyst H8 sequence view. "
            "copy_target preserves the original augmentation behavior. "
            "condition_quantile uses a train-only lower-tail target grouped "
            "by the configured reaction-condition columns."
        ),
    )
    parser.add_argument(
        "--smiles-sequence-catalyst-h8-target-quantile",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--smiles-sequence-catalyst-h8-target-mix",
        type=float,
        default=1.0,
        help=(
            "For condition_quantile H8 targets, interpolate from the "
            "original label (0) to the condition quantile (1). This is "
            "training-only and does not alter inference."
        ),
    )
    parser.add_argument(
        "--smiles-sequence-catalyst-h8-target-condition-columns",
        default="Imine,Thiol",
        help=(
            "Comma-separated train-frame condition columns used only when "
            "the H8 target mode is condition_quantile."
        ),
    )
    parser.add_argument(
        "--smiles-sequence-catalyst-h8-randomized-views",
        type=int,
        default=0,
        help=(
            "Number of additional train-only, molecule-equivalent rooted "
            "SMILES enumerations made from each H8 sequence view. This "
            "changes no validation/test input or inference module."
        ),
    )
    parser.add_argument(
        "--training-role-feature-dropout-probability",
        type=float,
        default=0.0,
        help=(
            "Default-off training-only probability of zeroing backbone "
            "features for one complete role in a row. Inference is "
            "unchanged."
        ),
    )
    parser.add_argument(
        "--training-role-feature-dropout-id",
        type=int,
        default=-1,
        help=(
            "Role ID affected by training-only feature dropout; CPA "
            "Catalyst is role 10."
        ),
    )
    parser.add_argument(
        "--smiles-sequence-auxiliary",
        action="store_true",
        help=(
            "Default-off training-only task head over the byte-SMILES "
            "sequence summary. It never replaces or enters the final "
            "prediction."
        ),
    )
    parser.add_argument(
        "--smiles-sequence-auxiliary-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--smiles-sequence-auxiliary-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Train the sequence encoder and auxiliary head for this many "
            "epochs before enabling the zero-initialized fusion residual."
        ),
    )
    parser.add_argument(
        "--smiles-sequence-auxiliary-post-warmup-lr",
        type=float,
        default=0.0,
        help=(
            "Optional readout LR applied exactly when the sequence residual "
            "is enabled; zero retains the warmup LR."
        ),
    )
    parser.add_argument(
        "--reactant-conformer-view-probability",
        type=float,
        default=0.0,
        help=(
            "Default-off train-only probability of replacing role-0 tokens "
            "with one sampled conformer view. Validation and inference always "
            "retain the clean fixed three-conformer aggregation."
        ),
    )
    parser.add_argument(
        "--reactant-conformer-consistency-weight",
        type=float,
        default=0.0,
        help=(
            "Default-off stop-gradient clean-view Huber consistency weight. "
            "A positive value requires a positive conformer-view probability."
        ),
    )
    parser.add_argument(
        "--reactant-conformer-consistency-huber-delta",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--directed-role-attention",
        action="store_true",
        help=(
            "Incrementally let the AHO reactant role (ID 0) attend only to "
            "solvent/ligand/metal roles (IDs 2/8/9). The zero-initialized "
            "residual is disabled by default and preserves the parent function."
        ),
    )
    parser.add_argument(
        "--reaction-pair-transformer",
        action="store_true",
        help=(
            "Incrementally add a Stage-2-only Transformer over the four AHO "
            "role tokens and their six explicit pair tokens. The scalar "
            "residual head is zero initialized and disabled by default."
        ),
    )
    parser.add_argument(
        "--reaction-pair-transformer-num-layers",
        type=int,
        default=2,
        help=(
            "Number of layers in the default-off reaction-pair Transformer."
        ),
    )
    parser.add_argument(
        "--site-ligand-metal-attention",
        action="store_true",
        help=(
            "Incrementally enable A3 site-to-ligand-to-metal interaction "
            "attention. Its zero-initialized residual is disabled by default."
        ),
    )
    parser.add_argument(
        "--site-donor-relation-attention",
        action="store_true",
        help=(
            "Incrementally enable one in-network site/donor/metal/solvent "
            "relation-attention residual. Cross-component attention is "
            "chemically constrained, the final residual is zero initialized, "
            "and the branch is disabled by default."
        ),
    )
    parser.add_argument(
        "--site-donor-relation-num-layers",
        type=int,
        default=2,
        help=(
            "Number of chemistry-constrained relation layers in the "
            "default-off residual adapter."
        ),
    )
    parser.add_argument(
        "--site-donor-relation-auxiliary-weight",
        type=float,
        default=0.0,
        help=(
            "Default-off BCE weight that teaches the in-network relation "
            "logit directly while its bounded coupling is learned through "
            "the primary Major objective."
        ),
    )
    parser.add_argument(
        "--multi-level-orb-attention",
        action="store_true",
        help=(
            "Incrementally enable A4 attention over selected early/middle/final "
            "Orb GNN stack features captured from one backbone forward. The "
            "zero residual gate is disabled by default."
        ),
    )
    parser.add_argument(
        "--multi-level-orb-stack-indices",
        type=parse_stack_indices,
        default=(4, 9, 14),
        help=(
            "Zero-based Orb GNN stack indices used by A4. The final index must "
            "be the last stack; default 4,9,14 for Orb-v2."
        ),
    )
    parser.add_argument(
        "--multi-level-orb-score-hidden-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--aho-consistency-auxiliary",
        action="store_true",
        help=(
            "Enable one in-network AHO physical-consistency adapter: Major "
            "uses signed ΔΔG‡ as an auxiliary target and ΔΔG‡ uses Major as "
            "an auxiliary target. Disabled by default; no cross-checkpoint "
            "prediction combination is performed."
        ),
    )
    parser.add_argument(
        "--aho-consistency-hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--aho-consistency-weight",
        type=float,
        default=0.25,
        help="Training weight for the complementary AHO auxiliary loss.",
    )
    parser.add_argument(
        "--aho-consistency-huber-delta",
        type=float,
        default=0.75,
        help="Huber delta for the signed-ΔΔG‡ auxiliary target on Major.",
    )
    parser.add_argument(
        "--ddg-sign-magnitude",
        action="store_true",
        help=(
            "Incrementally factor AHO ΔΔG‡ into an internal sign classifier "
            "and positive magnitude regressor. A zero-initialized residual "
            "gate preserves the parent prediction exactly; disabled by default."
        ),
    )
    parser.add_argument(
        "--ddg-sign-magnitude-hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--ddg-sign-magnitude-weight",
        type=float,
        default=0.25,
        help="Weight for the sign BCE plus magnitude Huber auxiliary loss.",
    )
    parser.add_argument(
        "--ddg-sign-only-auxiliary-weight",
        type=float,
        default=0.0,
        help=(
            "Default-off additional weight on only the A6 sign BCE. This "
            "allows sign supervision without simultaneously increasing the "
            "magnitude Huber objective; zero exactly preserves the legacy "
            "combined A6 loss."
        ),
    )
    parser.add_argument(
        "--ddg-sign-magnitude-huber-delta",
        type=float,
        default=0.75,
        help="Huber delta for normalized absolute ΔΔG‡ magnitude.",
    )
    parser.add_argument(
        "--ddg-sign-magnitude-residual-multiplier",
        type=float,
        default=1.0,
        help=(
            "Default-neutral multiplier on the learned A6 structured "
            "residual. A value of one exactly preserves historical behavior."
        ),
    )
    parser.add_argument(
        "--ddg-residual-calibrator",
        action="store_true",
        help=(
            "Incrementally enable A7: one zero-initialized post-gate linear "
            "residual using same-network A6 scalar diagnostics. Disabled by "
            "default and requires --ddg-sign-magnitude."
        ),
    )
    parser.add_argument(
        "--ddg-residual-calibrator-nonlinear-hidden-dim",
        type=int,
        default=0,
        help=(
            "Default-off A8 nonlinear residual width added after the "
            "historical A7 linear calibrator. Zero creates no new state and "
            "exactly preserves A7; a positive value requires "
            "--ddg-residual-calibrator."
        ),
    )
    parser.add_argument(
        "--ddg-major-summary",
        action="store_true",
        help=(
            "Incrementally enable A9: same-network Major classifiers over "
            "the fusion and reactant/solvent/ligand/metal role tokens feed "
            "one zero-initialized bounded ΔΔG‡ residual. Disabled by default "
            "and requires the A6+A7 parent."
        ),
    )
    parser.add_argument(
        "--ddg-major-summary-classifier-hidden-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--ddg-major-summary-residual-hidden-dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--ddg-major-summary-weight",
        type=float,
        default=0.25,
        help=(
            "Weight for the same-network role-view Major BCE used by A9."
        ),
    )
    parser.add_argument(
        "--ddg-major-summary-residual-multiplier",
        type=float,
        default=1.0,
        help=(
            "Physical-capacity multiplier on the bounded normalized A9 "
            "residual; one is the default."
        ),
    )
    parser.add_argument(
        "--ddg-major-summary-classifier-warmup-epochs",
        type=int,
        default=0,
        help=(
            "Default-zero number of A9 classifier-only epochs. During this "
            "warmup the zero-initialized residual head remains frozen and "
            "checkpoint patience does not advance; afterwards the same "
            "single-network residual is enabled."
        ),
    )
    parser.add_argument(
        "--ddg-major-summary-post-warmup-lr",
        type=float,
        default=0.0,
        help=(
            "Optional A9 readout learning rate applied when the residual "
            "phase begins. Zero keeps --lr unchanged and exactly preserves "
            "the default optimizer schedule."
        ),
    )
    parser.add_argument(
        "--ddg-sign-bce-magnitude-strength",
        type=float,
        default=0.0,
        help=(
            "Default-off strength for smoothly upweighting high-|ΔΔG‡| "
            "samples inside the A6 sign BCE. Per-batch weights are normalized "
            "to mean one so this does not change the total auxiliary scale."
        ),
    )
    parser.add_argument(
        "--ddg-sign-bce-magnitude-scale",
        type=float,
        default=1.0,
        help=(
            "Physical |ΔΔG‡| scale for the magnitude-aware A6 sign BCE. "
            "Ignored when strength is zero."
        ),
    )
    parser.add_argument(
        "--ddg-output-sign-margin-weight",
        type=float,
        default=0.0,
        help=(
            "Default-off weight for a magnitude-aware sign-margin loss on "
            "the final AHO ΔΔG‡ prediction."
        ),
    )
    parser.add_argument(
        "--ddg-output-sign-margin-temperature",
        type=float,
        default=2.0,
        help="Softplus sign-margin temperature in normalized target units.",
    )
    parser.add_argument(
        "--ddg-output-sign-margin-dead-zone",
        type=float,
        default=0.10,
        help=(
            "Ignore physical |ΔΔG‡| values at or below this dead zone when "
            "forming the sign-margin objective."
        ),
    )
    parser.add_argument(
        "--ddg-output-sign-margin-magnitude-scale",
        type=float,
        default=1.0,
        help=(
            "Physical |ΔΔG‡| scale used to smoothly upweight high-magnitude "
            "sign mistakes."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--init-readout-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--resume-finetune-checkpoint",
        type=Path,
        default=None,
        help=(
            "Initialize both the ORRA readout and Orb backbone from a previous "
            "finetuning model.pt. The optimizer and schedule are intentionally reset."
        ),
    )
    parser.add_argument(
        "--resume-readout-load-mode",
        default="strict",
        choices=["strict", "compatible", "intersection"],
        help=(
            "Strictly restore the parent readout, or transfer every compatible "
            "readout tensor while allowing newly enabled Stage-2 modules. "
            "Intersection mode explicitly permits dropping family-specific "
            "parent modules during a cross-family transfer. "
            "Backbone restoration is controlled separately and remains strict "
            "by default."
        ),
    )
    parser.add_argument(
        "--resume-backbone-load-mode",
        default="strict",
        choices=["strict", "skip"],
        help=(
            "Strictly restore the parent backbone as before, or transfer only "
            "the Stage-2 readout when changing backbone families. The "
            "default preserves historical resume behavior."
        ),
    )
    parser.add_argument(
        "--init-readout-mode",
        default="full",
        choices=["full", "features"],
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
    parser.add_argument("--max-tokens", type=int, default=176)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--epoch-ema-decay",
        type=float,
        default=0.0,
        help=(
            "Epoch-level parameter EMA decay in [0,1); 0 disables EMA and "
            "preserves the historical checkpoint-selection path."
        ),
    )
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--orb-lr", type=float, default=1e-5)
    parser.add_argument(
        "--orb-warmup-epochs",
        type=int,
        default=5,
        help="Train the new readout first; enable the low-LR Orb parameter group afterward.",
    )
    parser.add_argument("--weight-decay", type=float, default=3e-4)
    parser.add_argument("--orb-weight-decay", type=float, default=1e-5)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument(
        "--readout-gradient-clip-norm",
        type=float,
        default=5.0,
        help="Readout gradient-norm cap; use 0 to disable clipping.",
    )
    parser.add_argument(
        "--orb-gradient-clip-norm",
        type=float,
        default=1.0,
        help="Unfrozen Orb gradient-norm cap; use 0 to disable clipping.",
    )
    parser.add_argument(
        "--orb-module-mode",
        default="eval",
        choices=["eval", "train"],
        help=(
            "Module mode used while differentiating through Orb. Eval mode still "
            "updates unfrozen weights but keeps pretrained dropout/statistics fixed."
        ),
    )
    parser.add_argument(
        "--readout-module-mode",
        default="train",
        choices=["train", "eval"],
        help=(
            "Module mode used while differentiating through the ORRA readout. "
            "The historical default is train. Eval mode keeps readout dropout "
            "and running statistics fixed while gradients may still update an "
            "online backbone."
        ),
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use bfloat16 autocast on CUDA while retaining float32 optimizer states.",
    )
    parser.add_argument("--lr-patience", type=int, default=7)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--unfreeze-orb-stacks", type=int, default=1)
    parser.add_argument(
        "--orb-train-scope",
        default="stacks",
        choices=[
            "frozen",
            "stacks",
            "full_backbone",
            "jmp_output",
            "jmp_last_block",
            "jmp_last_two_blocks",
        ],
        help=(
            "Keep Orb frozen, train only the requested final GNN stacks, or "
            "train the complete backbone. JMP additionally supports its output "
            "path, output path plus final interaction block, or the final two "
            "output/interaction blocks. The default exactly preserves the "
            "original Orb stacks-only behavior."
        ),
    )
    parser.add_argument(
        "--readout-train-scope",
        default="all",
        choices=[
            "all",
            "conformer_attention",
            "directed_role_attention",
            "reaction_pair_transformer",
            "site_ligand_metal_attention",
            "site_donor_relation_attention",
            "multi_level_orb_attention",
            "dual_backbone_feature_fusion",
            "dual_backbone_feature_role_gate",
            "cip_orientation_embedding",
            "smiles_sequence_adapter",
            "regression_head",
            "regression_output",
            "aho_consistency_adapter",
            "ddg_sign_magnitude_adapter",
            "ddg_residual_calibrator",
            "ddg_residual_nonlinear_adapter",
            "ddg_sign_magnitude_and_residual_calibrator",
            "ddg_major_summary_adapter",
        ],
        help=(
            "Train the complete ORRA readout, or only the explicitly enabled "
            "A1 conformer-attention, A2 directed-role-attention, A3 "
            "site-ligand-metal interaction, reaction-pair Transformer, "
            "site-donor relation residual, A4 multi-level Orb attention, or "
            "the default-off dual-backbone feature residual, one default-off "
            "CIP R/S atom embedding, one default-off byte-SMILES residual, "
            "the existing regression head or only its output layer, "
            "one default-off single-network output "
            "adapter, A7 residual "
            "calibrator, its A8 nonlinear residual branch, or the explicit "
            "A6+A7 joint scope, or the default-off A9 same-network "
            "role-level Major summary. The default preserves the original "
            "complete-readout optimization."
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
    parser.add_argument("--unseen-density-floor", type=float, default=0.0)
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
    )
    parser.add_argument("--ood-shift-scale", type=float, default=1.0)
    parser.add_argument("--ood-base-shift", type=float, default=0.0)
    parser.add_argument("--residual-global-center-mix", type=float, default=1.0)
    parser.add_argument(
        "--sample-prior-mode",
        default="global",
        choices=["global", "train_quantile", "condition_quantile"],
        help=(
            "Use the historical global train prior, one train-label quantile, "
            "or train-only condition-group quantiles as the existing gate's "
            "per-sample fallback. The default preserves the original path."
        ),
    )
    parser.add_argument(
        "--prior-condition-columns",
        default="",
        help=(
            "Comma-separated train-frame columns for condition_quantile "
            "sample priors."
        ),
    )
    parser.add_argument("--prior-quantile", type=float, default=0.10)
    parser.add_argument(
        "--freeze-gate",
        action="store_true",
        help="Keep the train-only density gate deterministic while fitting the model.",
    )
    parser.add_argument("--gate-regularization", type=float, default=0.005)
    parser.add_argument("--loss", default="huber", choices=["huber", "mae", "mse"])
    parser.add_argument("--huber-delta", type=float, default=0.75)
    parser.add_argument(
        "--regression-group-mean-weight",
        type=float,
        default=0.0,
        help=(
            "Default-off training-only auxiliary weight on per-group mean "
            "prediction error. It changes no inference module."
        ),
    )
    parser.add_argument(
        "--regression-group-mean-column",
        default="",
        help=(
            "Training-frame column defining groups for the default-off "
            "regression group-mean objective."
        ),
    )
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
        "--classification-focal-gamma",
        type=float,
        default=0.0,
        help=(
            "Default-off focal exponent for the primary AHO Major BCE "
            "objective. Gamma 0 follows the historical BCE path exactly."
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
        "--classification-pairwise-ranking-weight",
        type=float,
        default=0.0,
        help=(
            "Default-off within-batch positive-vs-negative logistic ranking "
            "loss added only to the primary classification objective. A "
            "weight of 0 exactly preserves the existing BCE training path."
        ),
    )
    parser.add_argument(
        "--classification-pairwise-ranking-temperature",
        type=float,
        default=1.0,
        help=(
            "Positive temperature for the optional classification pairwise "
            "ranking loss. It has no effect while the ranking weight is 0."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        default="auto",
        choices=[
            "auto",
            "accuracy",
            "auroc",
            "major_gate_balance",
            "ddg_gate_balance",
            "mae",
            "rmse",
            "r2",
        ],
        help=(
            "Checkpoint criterion on --checkpoint-selection-split. "
            "major_gate_balance minimizes "
            "the worse relative shortfall to the declared Major accuracy "
            "and AUROC gates; ddg_gate_balance does the same for the DDG R2 "
            "and MAE gates; all legacy choices are unchanged."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection-split",
        default="validation",
        choices=["validation", "test"],
        help=(
            "Select saved checkpoints on validation by default. The test "
            "option is intentionally fail-closed and marks the run as "
            "observed-test/test-guided."
        ),
    )
    parser.add_argument(
        "--acknowledge-observed-test-selection",
        action="store_true",
        help=(
            "Required acknowledgement that selecting checkpoints on test "
            "invalidates blind-test interpretation."
        ),
    )
    parser.add_argument("--major-accuracy-gate", type=float, default=0.86)
    parser.add_argument("--major-auroc-gate", type=float, default=0.925)
    parser.add_argument("--ddg-r2-gate", type=float, default=0.58)
    parser.add_argument("--ddg-mae-gate", type=float, default=0.80)
    parser.add_argument(
        "--scheduler-metric",
        default="selection",
        choices=[
            "selection",
            "auto",
            "accuracy",
            "auroc",
            "major_gate_balance",
            "ddg_gate_balance",
            "mae",
            "rmse",
            "r2",
        ],
        help=(
            "ReduceLROnPlateau monitor. The default 'selection' exactly "
            "preserves the legacy coupling to --selection-metric."
        ),
    )
    parser.add_argument("--classification-threshold", type=float, default=0.5)
    parser.add_argument(
        "--validation-mode",
        default="default",
        choices=["default", "group_column"],
        help=(
            "Use the task's existing validation split by default, or rebuild "
            "train/validation from train+validation by held-out groups."
        ),
    )
    parser.add_argument("--group-val-column", default="")
    parser.add_argument("--group-val-fraction", type=float, default=0.20)
    parser.add_argument(
        "--group-val-strategy",
        default="random",
        choices=["random", "farthest_centroid", "balanced_fold"],
    )
    parser.add_argument("--group-val-role", default="")
    parser.add_argument(
        "--group-val-fold-count",
        type=int,
        default=0,
        help=(
            "Number of deterministic whole-group folds when "
            "--group-val-strategy=balanced_fold."
        ),
    )
    parser.add_argument(
        "--group-val-fold-index",
        type=int,
        default=0,
        help=(
            "Zero-based held-out fold when "
            "--group-val-strategy=balanced_fold."
        ),
    )
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help="Write validation-only candidates without constructing or reading test predictions.",
    )
    parser.add_argument(
        "--final-retrain-train-plus-val",
        action="store_true",
        help=(
            "Default-off final-retrain protocol: append validation rows to "
            "the training loader and always save the fixed final epoch. "
            "Validation metrics become training-set diagnostics and are not "
            "used for checkpoint selection or scheduling."
        ),
    )
    parser.add_argument(
        "--export-fusion-features",
        action="store_true",
        help=(
            "Diagnostic-only: evaluate train as well as validation and export "
            "the frozen Stage-2 fusion tensor. Disabled by default."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, str]:
    token_dir = args.atom_token_dir
    if token_dir is None:
        token_dir = ROOT / "reproduction_outputs" / "full_retrain" / TOKEN_DIR_NAMES[args.task]
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = ROOT / "reproduction_outputs" / "unified_orra_finetune" / args.task
    conformer_cache = args.conformer_cache
    if conformer_cache is None:
        conformer_cache = token_dir / "orb_finetune_conformers_v2_three.joblib"
    mapping_task = args.mapping_task or ("aho_major" if args.task == "aho_ddg" else args.task)
    return token_dir, out_dir, conformer_cache, mapping_task


def atoms_payload(
    canonical_smiles: str,
    charge: int,
    spin: int,
    seed: int,
    fallback_numbers: np.ndarray,
    fallback_positions: np.ndarray,
) -> dict:
    heavy_mol = Chem.MolFromSmiles(canonical_smiles)
    if heavy_mol is not None:
        num_heavy = int(heavy_mol.GetNumAtoms())
        molecule = Chem.AddHs(heavy_mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.useRandomCoords = True
        try:
            conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=3, params=params))
        except Exception:
            conformer_ids = []
        if conformer_ids:
            return {
                "numbers": np.asarray(
                    [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                    dtype=np.int64,
                ),
                "positions": np.stack(
                    [
                        np.asarray(
                            molecule.GetConformer(int(conformer_id)).GetPositions(),
                            dtype=np.float32,
                        )
                        for conformer_id in conformer_ids
                    ],
                    axis=0,
                ),
                "charge": int(charge),
                "spin": int(spin),
                "num_heavy": num_heavy,
                "source": "rdkit_etkdgv3_three_conformers",
            }

    fallback_positions = np.asarray(fallback_positions, dtype=np.float32)
    if fallback_positions.ndim == 2:
        fallback_positions = fallback_positions[None, ...]
    return {
        "numbers": fallback_numbers.astype(np.int64),
        "positions": fallback_positions,
        "charge": int(charge),
        "spin": int(spin),
        "num_heavy": int(len(fallback_numbers)),
        "source": "cached_heavy_atom_fallback",
    }


def prepare_conformer_payloads(store, cache_path: Path, seed: int) -> dict[int, dict]:
    if cache_path.exists():
        payloads = joblib.load(cache_path)
        print(f"[conformers] loaded {len(payloads)} components from {cache_path}", flush=True)
        return payloads

    metadata = store.metadata.set_index("component_id")
    payloads: dict[int, dict] = {}
    for component_id in metadata.index.to_numpy(dtype=int):
        row = metadata.loc[component_id]
        atom_mask = store.atom_mask[component_id]
        conformer_ids = np.flatnonzero(store.conformer_mask[component_id])
        if len(conformer_ids):
            fallback_positions = store.positions[
                component_id,
                conformer_ids,
            ][:, atom_mask]
        else:
            fallback_positions = store.positions[
                component_id,
                :1,
            ][:, atom_mask]
        payloads[int(component_id)] = atoms_payload(
            canonical_smiles=str(row["canonical_smiles"]),
            charge=int(row.get("formal_charge", 0)),
            spin=int(row.get("spin_multiplicity", 1)),
            seed=seed + int(component_id),
            fallback_numbers=store.atomic_numbers[component_id][atom_mask],
            fallback_positions=fallback_positions,
        )
        if (len(payloads) % 100 == 0) or len(payloads) == len(metadata):
            print(
                f"[conformers] prepared {len(payloads)}/{len(metadata)}",
                flush=True,
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payloads, cache_path)
    print(f"[conformers] wrote {cache_path}", flush=True)
    return payloads


def payload_to_atoms(payload: dict, conformer_index: int) -> Atoms:
    positions = np.asarray(payload["positions"], dtype=np.float32)
    if positions.ndim == 2:
        positions = positions[None, ...]
    atoms = Atoms(
        numbers=np.asarray(payload["numbers"], dtype=np.int64),
        positions=positions[int(conformer_index)],
    )
    atoms.info["charge"] = int(payload["charge"])
    atoms.info["spin"] = int(payload["spin"])
    return atoms


def global_component_matrix(
    store,
    *,
    mapping_task: str,
    part: str,
    row_index: np.ndarray,
) -> torch.Tensor:
    mapping = store.mapping[
        (store.mapping["task"].astype(str) == mapping_task)
        & (store.mapping["split_part"].astype(str) == part)
    ].copy()
    grouped = {
        int(source): frame.sort_values("slot")
        for source, frame in mapping.groupby("source_index", sort=False)
    }
    max_components = max((len(grouped.get(int(index), ())) for index in row_index), default=1)
    result = np.full((len(row_index), max_components), -1, dtype=np.int64)
    for row_pos, source_index in enumerate(row_index):
        frame = grouped.get(int(source_index))
        if frame is None:
            continue
        component_ids = frame["component_id"].to_numpy(dtype=np.int64)
        result[row_pos, : len(component_ids)] = component_ids
    return torch.from_numpy(result)


def build_cip_orientation_lookup(
    store,
    component_ids: list[int] | tuple[int, ...] | np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    """Build atom-aligned CIP IDs from canonical input-component SMILES."""
    metadata = store.metadata.set_index("component_id")
    if not metadata.index.is_unique:
        raise ValueError("Component metadata IDs are not unique")
    lookup: dict[int, np.ndarray] = {}
    r_atoms = 0
    s_atoms = 0
    other_atoms = 0
    components_with_cip = 0
    for component_id_raw in sorted(
        {int(value) for value in component_ids if int(value) >= 0}
    ):
        component_id = int(component_id_raw)
        if component_id not in metadata.index:
            raise ValueError(
                f"Missing metadata for CIP component {component_id}"
            )
        canonical_smiles = str(
            metadata.loc[component_id, "canonical_smiles"]
        )
        molecule = Chem.MolFromSmiles(canonical_smiles)
        if molecule is None:
            raise ValueError(
                f"Cannot parse CIP component {component_id}: "
                f"{canonical_smiles}"
            )
        Chem.AssignStereochemistry(
            molecule,
            cleanIt=True,
            force=True,
            flagPossibleStereoCenters=True,
        )
        ids = np.zeros(molecule.GetNumAtoms(), dtype=np.int64)
        for atom in molecule.GetAtoms():
            if not atom.HasProp("_CIPCode"):
                continue
            code = atom.GetProp("_CIPCode")
            ids[atom.GetIdx()] = CIP_ORIENTATION_IDS.get(
                code,
                CIP_ORIENTATION_IDS["other"],
            )
        atom_mask = np.asarray(
            store.atom_mask[component_id],
            dtype=bool,
        )
        num_atoms = len(ids)
        if (
            int(atom_mask.sum()) != num_atoms
            or not atom_mask[:num_atoms].all()
            or atom_mask[num_atoms:].any()
        ):
            raise ValueError(
                f"CIP atom-mask mismatch for component {component_id}: "
                f"RDKit={num_atoms}, cache={int(atom_mask.sum())}"
            )
        rdkit_numbers = np.asarray(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
            dtype=np.int64,
        )
        cache_numbers = np.asarray(
            store.atomic_numbers[component_id, :num_atoms],
            dtype=np.int64,
        )
        if not np.array_equal(rdkit_numbers, cache_numbers):
            raise ValueError(
                "CIP atomic-number alignment mismatch for component "
                f"{component_id}"
            )
        lookup[component_id] = ids
        n_r = int(np.count_nonzero(ids == CIP_ORIENTATION_IDS["R"]))
        n_s = int(np.count_nonzero(ids == CIP_ORIENTATION_IDS["S"]))
        n_other = int(
            np.count_nonzero(ids == CIP_ORIENTATION_IDS["other"])
        )
        r_atoms += n_r
        s_atoms += n_s
        other_atoms += n_other
        components_with_cip += int(n_r + n_s + n_other > 0)
    return lookup, {
        "enabled": True,
        "id_schema": dict(CIP_ORIENTATION_IDS),
        "aligned_components": len(lookup),
        "components_with_cip": components_with_cip,
        "r_atoms": r_atoms,
        "s_atoms": s_atoms,
        "other_atoms": other_atoms,
        "atom_count_mask_atomic_number_alignment_pass": True,
    }


def assemble_cip_orientation_ids(
    *,
    raw_bundle,
    global_components: torch.Tensor,
    component_lookup: dict[int, np.ndarray],
) -> torch.Tensor:
    """Align component-local atom CIP IDs to existing ORRA token positions."""
    result = torch.zeros_like(raw_bundle.token_type_ids, dtype=torch.long)
    if global_components.shape[0] != result.shape[0]:
        raise ValueError(
            "CIP global-component rows do not match the token bundle"
        )
    for row in range(result.shape[0]):
        for local_id in range(global_components.shape[1]):
            component_id = int(global_components[row, local_id])
            if component_id < 0:
                continue
            if component_id not in component_lookup:
                raise ValueError(
                    f"CIP lookup lacks component {component_id}"
                )
            component_positions = torch.nonzero(
                raw_bundle.token_mask[row].bool()
                & (raw_bundle.component_ids[row].long() == local_id),
                as_tuple=False,
            ).squeeze(-1)
            if len(component_positions) == 0:
                continue
            component_types = raw_bundle.token_type_ids[
                row,
                component_positions,
            ]
            atom_positions = component_positions[
                (component_types != TOKEN_TYPE_IDS["global"])
                & (component_types != TOKEN_TYPE_IDS["missing"])
            ]
            component_ids = component_lookup[component_id]
            atom_count = min(len(atom_positions), len(component_ids))
            if atom_count:
                result[row, atom_positions[:atom_count]] = (
                    torch.from_numpy(component_ids[:atom_count])
                )
    if torch.any((result != 0) & ~raw_bundle.token_mask.bool()):
        raise RuntimeError("A padded token received a CIP orientation")
    if torch.any(result < 0) or torch.any(result > 3):
        raise RuntimeError("Assembled CIP IDs are outside [0, 3]")
    return result


def encode_byte_smiles_sequence(
    values: tuple[object, ...],
    *,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ordered model inputs with a deterministic byte vocabulary."""
    token_ids = [SMILES_BYTE_TOKEN_IDS["cls"]]
    for value in values:
        text = "" if pd.isna(value) else str(value)
        token_ids.extend(
            int(byte) + SMILES_BYTE_TOKEN_IDS["byte_offset"]
            for byte in text.encode("utf-8")
        )
        token_ids.append(SMILES_BYTE_TOKEN_IDS["sep"])
    if len(token_ids) > max_length:
        raise ValueError(
            "A byte-SMILES input exceeds --smiles-sequence-max-length: "
            f"{len(token_ids)} > {max_length}"
        )
    ids = torch.zeros(max_length, dtype=torch.long)
    ids[: len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
    mask = ids != SMILES_BYTE_TOKEN_IDS["pad"]
    return ids, mask


def deterministic_rooted_smiles_view(
    value: object,
    *,
    source_index: int,
    column_index: int,
    view_index: int,
    seed: int,
) -> str:
    """Return a deterministic, molecule-equivalent noncanonical SMILES."""
    text = "" if pd.isna(value) else str(value)
    molecule = Chem.MolFromSmiles(text)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError(
            "Cannot build a randomized training SMILES view for "
            f"{text!r}"
        )
    root = (
        int(seed)
        + int(source_index) * 104729
        + int(column_index) * 1009
        + int(view_index) * 9176
    ) % molecule.GetNumAtoms()
    return Chem.MolToSmiles(
        molecule,
        canonical=False,
        rootedAtAtom=root,
        isomericSmiles=True,
    )


def build_byte_smiles_sequences(
    *,
    frame: pd.DataFrame,
    row_index: np.ndarray,
    input_columns: tuple[str, ...],
    max_length: int,
    catalyst_h8_view: bool = False,
    randomized_view_index: int = -1,
    randomized_seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Align byte-SMILES inputs to the existing ORRA source-row order."""
    missing_columns = [
        column for column in input_columns if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(
            "SMILES sequence input columns are missing: "
            + ", ".join(missing_columns)
        )
    by_source: dict[int, tuple[object, ...]] = {}
    augmented_rows = 0
    augmented_rings = 0
    for source, (_, row) in zip(
        frame["_source_index"].to_numpy(dtype=int),
        frame.iterrows(),
        strict=True,
    ):
        values: list[object] = []
        row_changed_rings = 0
        for column_index, column in enumerate(input_columns):
            value = row[column]
            if catalyst_h8_view and column == "Catalyst":
                value, changed_rings = catalyst_h8_scaffold_view(value)
                row_changed_rings += changed_rings
            if randomized_view_index >= 0:
                value = deterministic_rooted_smiles_view(
                    value,
                    source_index=int(source),
                    column_index=column_index,
                    view_index=randomized_view_index,
                    seed=randomized_seed,
                )
            values.append(value)
        by_source[int(source)] = tuple(values)
        if row_changed_rings:
            augmented_rows += 1
            augmented_rings += row_changed_rings
    encoded = []
    for source in np.asarray(row_index, dtype=int):
        if int(source) not in by_source:
            raise ValueError(
                f"SMILES sequence frame lacks source row {int(source)}"
            )
        encoded.append(
            encode_byte_smiles_sequence(
                by_source[int(source)],
                max_length=max_length,
            )
        )
    token_ids = torch.stack([item[0] for item in encoded], dim=0)
    token_mask = torch.stack([item[1] for item in encoded], dim=0)
    non_padding = token_mask.sum(dim=1)
    metadata = {
        "rows": int(len(token_ids)),
        "input_columns": list(input_columns),
        "max_length": int(max_length),
        "max_observed_tokens": int(non_padding.max()) if len(token_ids) else 0,
        "mean_observed_tokens": (
            float(non_padding.float().mean()) if len(token_ids) else 0.0
        ),
        "truncated_rows": 0,
        "catalyst_h8_view": bool(catalyst_h8_view),
        "catalyst_h8_augmented_rows": int(augmented_rows),
        "catalyst_h8_augmented_rings": int(augmented_rings),
        "randomized_view_index": int(randomized_view_index),
        "randomized_seed": int(randomized_seed),
    }
    return token_ids, token_mask, metadata


def make_loader(
    raw_bundle,
    density: torch.Tensor,
    target: torch.Tensor,
    global_components: torch.Tensor,
    sample_prior: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    auxiliary_target: torch.Tensor | None = None,
    cip_orientation_ids: torch.Tensor | None = None,
    smiles_sequence_token_ids: torch.Tensor | None = None,
    smiles_sequence_token_mask: torch.Tensor | None = None,
    training_group_ids: torch.Tensor | None = None,
) -> DataLoader:
    tensors = (
        raw_bundle.numeric,
        raw_bundle.token_mask,
        raw_bundle.role_ids,
        raw_bundle.token_type_ids,
        raw_bundle.atomic_numbers,
        raw_bundle.component_ids,
        density,
        target.float(),
        global_components,
        sample_prior.float(),
    )
    if auxiliary_target is not None:
        tensors = (*tensors, auxiliary_target.float())
    if cip_orientation_ids is not None:
        if cip_orientation_ids.shape != raw_bundle.token_mask.shape:
            raise ValueError(
                "CIP orientation tensor must match token-mask shape"
            )
        tensors = (*tensors, cip_orientation_ids.long())
    if (
        (smiles_sequence_token_ids is None)
        != (smiles_sequence_token_mask is None)
    ):
        raise ValueError(
            "SMILES sequence token IDs and mask must be provided together"
        )
    if smiles_sequence_token_ids is not None:
        if (
            smiles_sequence_token_ids.ndim != 2
            or smiles_sequence_token_mask.shape
            != smiles_sequence_token_ids.shape
            or smiles_sequence_token_ids.shape[0] != len(target)
        ):
            raise ValueError(
                "SMILES sequence tensors are not aligned to the loader rows"
            )
        tensors = (
            *tensors,
            smiles_sequence_token_ids.long(),
            smiles_sequence_token_mask.bool(),
        )
    if training_group_ids is not None:
        if (
            training_group_ids.ndim != 1
            or len(training_group_ids) != len(target)
        ):
            raise ValueError(
                "Training group IDs must be a row-aligned vector"
            )
        tensors = (*tensors, training_group_ids.long())
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def aligned_training_group_ids(
    frame: pd.DataFrame,
    *,
    column: str,
    row_index: np.ndarray,
) -> torch.Tensor:
    if column not in frame.columns:
        raise ValueError(f"Training group column is missing: {column}")
    source_column = (
        frame["_source_index"].to_numpy(dtype=np.int64)
        if "_source_index" in frame.columns
        else frame.index.to_numpy(dtype=np.int64)
    )
    values = frame[column].astype(str).to_numpy(dtype=object)
    categories = {
        value: index
        for index, value in enumerate(sorted(set(map(str, values))))
    }
    by_source = {
        int(source): categories[str(value)]
        for source, value in zip(source_column, values, strict=True)
    }
    try:
        aligned = [by_source[int(source)] for source in row_index]
    except KeyError as error:
        raise ValueError(
            "Training group IDs are not aligned to tensor rows"
        ) from error
    return torch.as_tensor(aligned, dtype=torch.long)


def orb_component_features(
    *,
    orb_backbone: nn.Module,
    adapter,
    payloads: dict[int, dict],
    global_components: torch.Tensor,
    device: torch.device,
    conformer_feature_mode: str,
    conformer_attention_aggregator: nn.Module | None,
    single_conformer_indices: dict[int, int] | None = None,
) -> (
    dict[int, torch.Tensor]
    | tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]
):
    component_ids = sorted(
        {
            int(value)
            for value in global_components.reshape(-1).tolist()
            if int(value) >= 0
        }
    )
    if not component_ids:
        return {}
    records: list[tuple[int, int]] = []
    atoms_list: list[Atoms] = []
    for component_id in component_ids:
        positions = np.asarray(payloads[component_id]["positions"])
        num_conformers = 1 if positions.ndim == 2 else int(len(positions))
        for conformer_index in range(num_conformers):
            records.append((component_id, conformer_index))
            atoms_list.append(
                payload_to_atoms(payloads[component_id], conformer_index)
            )
    graph = adapter.from_ase_atoms_list(atoms_list, device=device)
    output = orb_backbone(graph)["node_features"]
    graph_sizes = graph.n_node.detach().cpu().tolist()
    split_features = torch.split(output, [int(size) for size in graph_sizes], dim=0)
    by_component: dict[int, list[torch.Tensor]] = {
        component_id: [] for component_id in component_ids
    }
    for (component_id, _), features in zip(
        records,
        split_features,
        strict=True,
    ):
        by_component[component_id].append(
            features[: int(payloads[component_id]["num_heavy"])]
        )
    aggregated: dict[int, torch.Tensor] = {}
    augmented: dict[int, torch.Tensor] | None = (
        {} if single_conformer_indices is not None else None
    )
    for component_id, conformer_features in by_component.items():
        stacked = torch.stack(conformer_features, dim=0)
        if conformer_attention_aggregator is not None:
            aggregated[component_id] = conformer_attention_aggregator(
                stacked,
                conformer_feature_mode,
            )
        else:
            parts = [stacked.mean(dim=0)]
            if conformer_feature_mode in {"mean_std", "mean_std_max"}:
                parts.append(stacked.std(dim=0, unbiased=False))
            if conformer_feature_mode == "mean_std_max":
                parts.append(stacked.max(dim=0).values)
            aggregated[component_id] = torch.cat(parts, dim=-1)
        if augmented is not None:
            selected_index = single_conformer_indices.get(component_id)
            if selected_index is None:
                augmented[component_id] = aggregated[component_id]
            elif conformer_attention_aggregator is not None:
                raise RuntimeError(
                    "Reactant single-conformer consistency requires fixed "
                    "conformer aggregation"
                )
            else:
                augmented[component_id] = aggregate_fixed_conformer_view(
                    stacked,
                    conformer_feature_mode,
                    selected_index=selected_index,
                )
    if augmented is not None:
        return aggregated, augmented
    return aggregated


def jmp_component_dual_features(
    *,
    orb_backbone: nn.Module,
    adapter,
    payloads: dict[int, dict],
    global_components: torch.Tensor,
    device: torch.device,
    conformer_feature_mode: str,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Aggregate final and interaction JMP views from one traversal."""
    component_ids = sorted(
        {
            int(value)
            for value in global_components.reshape(-1).tolist()
            if int(value) >= 0
        }
    )
    if not component_ids:
        return {}, {}
    records: list[tuple[int, int]] = []
    atoms_list: list[Atoms] = []
    for component_id in component_ids:
        positions = np.asarray(payloads[component_id]["positions"])
        num_conformers = 1 if positions.ndim == 2 else int(len(positions))
        for conformer_index in range(num_conformers):
            records.append((component_id, conformer_index))
            atoms_list.append(
                payload_to_atoms(payloads[component_id], conformer_index)
            )
    graph = adapter.from_ase_atoms_list(atoms_list, device=device)
    try:
        output = orb_backbone(
            graph,
            return_final_interaction=True,
        )
    except TypeError as error:
        raise RuntimeError(
            "Online dual backbone feature fusion requires a JMP wrapper "
            "that supports one-pass final-interaction capture"
        ) from error
    primary_output = output.get("node_features")
    secondary_output = output.get("secondary_node_features")
    if (
        not isinstance(primary_output, torch.Tensor)
        or not isinstance(secondary_output, torch.Tensor)
        or primary_output.shape != secondary_output.shape
    ):
        raise RuntimeError(
            "One-pass JMP dual feature output is missing or misaligned"
        )
    graph_sizes = graph.n_node.detach().cpu().tolist()
    split_primary = torch.split(
        primary_output,
        [int(size) for size in graph_sizes],
        dim=0,
    )
    split_secondary = torch.split(
        secondary_output,
        [int(size) for size in graph_sizes],
        dim=0,
    )
    primary_by_component: dict[int, list[torch.Tensor]] = {
        component_id: [] for component_id in component_ids
    }
    secondary_by_component: dict[int, list[torch.Tensor]] = {
        component_id: [] for component_id in component_ids
    }
    for (
        (component_id, _),
        primary_features,
        secondary_features,
    ) in zip(
        records,
        split_primary,
        split_secondary,
        strict=True,
    ):
        num_heavy = int(payloads[component_id]["num_heavy"])
        primary_by_component[component_id].append(
            primary_features[:num_heavy]
        )
        secondary_by_component[component_id].append(
            secondary_features[:num_heavy]
        )

    def aggregate(
        conformer_features: list[torch.Tensor],
    ) -> torch.Tensor:
        stacked = torch.stack(conformer_features, dim=0)
        parts = [stacked.mean(dim=0)]
        if conformer_feature_mode in {"mean_std", "mean_std_max"}:
            parts.append(stacked.std(dim=0, unbiased=False))
        if conformer_feature_mode == "mean_std_max":
            parts.append(stacked.max(dim=0).values)
        return torch.cat(parts, dim=-1)

    return (
        {
            component_id: aggregate(features)
            for component_id, features in primary_by_component.items()
        },
        {
            component_id: aggregate(features)
            for component_id, features in secondary_by_component.items()
        },
    )


def cached_component_features(
    *,
    store,
    global_components: torch.Tensor,
    device: torch.device,
    conformer_feature_mode: str,
    conformer_attention_aggregator: nn.Module | None,
    single_conformer_indices: dict[int, int] | None = None,
) -> (
    dict[int, torch.Tensor]
    | tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]
):
    """Aggregate atom tokens extracted from one frozen backbone checkpoint."""
    component_ids = sorted(
        {
            int(value)
            for value in global_components.reshape(-1).tolist()
            if int(value) >= 0
        }
    )
    aggregated: dict[int, torch.Tensor] = {}
    augmented: dict[int, torch.Tensor] | None = (
        {} if single_conformer_indices is not None else None
    )
    for component_id in component_ids:
        conformer_indices = np.flatnonzero(
            store.conformer_mask[component_id]
        )
        num_atoms = int(store.atom_mask[component_id].sum())
        if len(conformer_indices) == 0 or num_atoms == 0:
            continue
        stacked = torch.as_tensor(
            store.node_features[
                component_id,
                conformer_indices,
                :num_atoms,
            ],
            dtype=torch.float32,
            device=device,
        )
        if (
            stacked.ndim != 3
            or stacked.shape[0] != len(conformer_indices)
            or tuple(stacked.shape[1:]) != (num_atoms, 256)
        ):
            raise RuntimeError(
                f"Unexpected cached component feature shape for "
                f"{component_id}: {tuple(stacked.shape)}"
            )
        if not torch.isfinite(stacked).all().item():
            raise RuntimeError(
                f"Non-finite cached component features: {component_id}"
            )
        if conformer_attention_aggregator is not None:
            aggregated[component_id] = conformer_attention_aggregator(
                stacked,
                conformer_feature_mode,
            )
        else:
            parts = [stacked.mean(dim=0)]
            if conformer_feature_mode in {"mean_std", "mean_std_max"}:
                parts.append(stacked.std(dim=0, unbiased=False))
            if conformer_feature_mode == "mean_std_max":
                parts.append(stacked.max(dim=0).values)
            aggregated[component_id] = torch.cat(parts, dim=-1)
        if augmented is not None:
            selected_index = single_conformer_indices.get(component_id)
            if selected_index is None:
                augmented[component_id] = aggregated[component_id]
            elif conformer_attention_aggregator is not None:
                raise RuntimeError(
                    "Reactant single-conformer consistency requires fixed "
                    "conformer aggregation"
                )
            else:
                augmented[component_id] = aggregate_fixed_conformer_view(
                    stacked,
                    conformer_feature_mode,
                    selected_index=selected_index,
                )
    if augmented is not None:
        return aggregated, augmented
    return aggregated


def orb_component_feature_levels(
    *,
    orb_backbone: nn.Module,
    adapter,
    payloads: dict[int, dict],
    global_components: torch.Tensor,
    device: torch.device,
    conformer_feature_mode: str,
    conformer_attention_aggregator: nn.Module | None,
    stack_indices: tuple[int, ...],
) -> tuple[dict[int, torch.Tensor], ...]:
    """Aggregate selected Orb stack features after one backbone traversal."""
    component_ids = sorted(
        {
            int(value)
            for value in global_components.reshape(-1).tolist()
            if int(value) >= 0
        }
    )
    if not component_ids:
        return tuple({} for _ in stack_indices)
    records: list[tuple[int, int]] = []
    atoms_list: list[Atoms] = []
    for component_id in component_ids:
        positions = np.asarray(payloads[component_id]["positions"])
        num_conformers = 1 if positions.ndim == 2 else int(len(positions))
        for conformer_index in range(num_conformers):
            records.append((component_id, conformer_index))
            atoms_list.append(
                payload_to_atoms(payloads[component_id], conformer_index)
            )
    graph = adapter.from_ase_atoms_list(atoms_list, device=device)
    _, node_feature_levels = forward_orb_with_node_feature_levels(
        orb_backbone,
        graph,
        stack_indices=stack_indices,
    )
    graph_sizes = [int(size) for size in graph.n_node.detach().cpu().tolist()]
    split_feature_levels = tuple(
        torch.split(level, graph_sizes, dim=0)
        for level in node_feature_levels
    )
    by_level: list[dict[int, list[torch.Tensor]]] = [
        {component_id: [] for component_id in component_ids}
        for _ in stack_indices
    ]
    for level_index, split_features in enumerate(split_feature_levels):
        for (component_id, _), features in zip(
            records,
            split_features,
            strict=True,
        ):
            by_level[level_index][component_id].append(
                features[: int(payloads[component_id]["num_heavy"])]
            )

    aggregated_levels: list[dict[int, torch.Tensor]] = []
    for level_components in by_level:
        aggregated: dict[int, torch.Tensor] = {}
        for component_id, conformer_features in level_components.items():
            stacked = torch.stack(conformer_features, dim=0)
            if conformer_attention_aggregator is not None:
                aggregated[component_id] = conformer_attention_aggregator(
                    stacked,
                    conformer_feature_mode,
                )
            else:
                parts = [stacked.mean(dim=0)]
                if conformer_feature_mode in {"mean_std", "mean_std_max"}:
                    parts.append(stacked.std(dim=0, unbiased=False))
                if conformer_feature_mode == "mean_std_max":
                    parts.append(stacked.max(dim=0).values)
                aggregated[component_id] = torch.cat(parts, dim=-1)
        aggregated_levels.append(aggregated)
    return tuple(aggregated_levels)


def assemble_orb_tokens(
    *,
    component_features: dict[int, torch.Tensor],
    token_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    component_local_ids: torch.Tensor,
    global_components: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch_size, num_tokens = token_mask.shape
    sample_feature = next(iter(component_features.values()), None)
    if sample_feature is None:
        return torch.zeros((batch_size, num_tokens, 256), device=device)
    x = sample_feature.new_zeros((batch_size, num_tokens, sample_feature.shape[-1]))
    for row in range(batch_size):
        for local_id in range(global_components.shape[1]):
            component_id = int(global_components[row, local_id])
            features = component_features.get(component_id)
            if features is None or len(features) == 0:
                continue
            component_token_mask = (
                token_mask[row].bool()
                & (component_local_ids[row].long() == local_id)
            )
            positions = torch.nonzero(component_token_mask, as_tuple=False).squeeze(-1)
            if len(positions) == 0:
                continue
            types = token_type_ids[row, positions]
            global_positions = positions[types == TOKEN_TYPE_IDS["global"]]
            if len(global_positions):
                x[row, global_positions] = features.mean(dim=0)
            atom_positions = positions[
                (types != TOKEN_TYPE_IDS["global"])
                & (types != TOKEN_TYPE_IDS["missing"])
            ]
            atom_count = min(len(atom_positions), len(features))
            if atom_count:
                x[row, atom_positions[:atom_count]] = features[:atom_count]
    return x


def assemble_orb_token_levels(
    *,
    component_feature_levels: tuple[dict[int, torch.Tensor], ...],
    token_mask: torch.Tensor,
    token_type_ids: torch.Tensor,
    component_local_ids: torch.Tensor,
    global_components: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Assemble all A4 levels directly, avoiding duplicate padded token copies."""
    batch_size, num_tokens = token_mask.shape
    sample_feature = next(
        (
            feature
            for component_features in component_feature_levels
            for feature in component_features.values()
        ),
        None,
    )
    num_levels = len(component_feature_levels)
    if sample_feature is None:
        return torch.zeros(
            (batch_size, num_tokens, num_levels, 256),
            device=device,
        )
    x = sample_feature.new_zeros(
        (
            batch_size,
            num_tokens,
            num_levels,
            sample_feature.shape[-1],
        )
    )
    for row in range(batch_size):
        for local_id in range(global_components.shape[1]):
            component_id = int(global_components[row, local_id])
            features_by_level = tuple(
                component_features.get(component_id)
                for component_features in component_feature_levels
            )
            if any(features is None for features in features_by_level):
                continue
            features = torch.stack(features_by_level, dim=1)
            if len(features) == 0:
                continue
            component_token_mask = (
                token_mask[row].bool()
                & (component_local_ids[row].long() == local_id)
            )
            positions = torch.nonzero(
                component_token_mask,
                as_tuple=False,
            ).squeeze(-1)
            if len(positions) == 0:
                continue
            types = token_type_ids[row, positions]
            global_positions = positions[types == TOKEN_TYPE_IDS["global"]]
            if len(global_positions):
                x[row, global_positions] = features.mean(dim=0)
            atom_positions = positions[
                (types != TOKEN_TYPE_IDS["global"])
                & (types != TOKEN_TYPE_IDS["missing"])
            ]
            atom_count = min(len(atom_positions), len(features))
            if atom_count:
                x[row, atom_positions[:atom_count]] = features[:atom_count]
    return x


def apply_training_role_feature_dropout(
    x: torch.Tensor,
    *,
    role_ids: torch.Tensor,
    token_mask: torch.Tensor,
    role_id: int,
    probability: float,
) -> torch.Tensor:
    """Zero one role's backbone features for sampled rows during training."""
    if probability <= 0.0:
        return x
    if probability > 1.0:
        raise ValueError("Role feature dropout probability must be <= 1")
    selected_rows = (
        torch.rand(x.shape[0], device=x.device) < probability
    )
    selected_tokens = (
        selected_rows.unsqueeze(1)
        & token_mask.to(device=x.device, dtype=torch.bool)
        & role_ids.to(device=x.device).eq(role_id)
    )
    return x.masked_fill(selected_tokens.unsqueeze(-1), 0.0)


def forward_batch(
    *,
    readout: UnifiedORRAModel,
    orb_backbone: nn.Module,
    adapter,
    payloads: dict[int, dict],
    token_store=None,
    secondary_token_store=None,
    backbone_feature_source: str = "online",
    batch: tuple[torch.Tensor, ...],
    device: torch.device,
    conformer_feature_mode: str,
    reactant_conformer_indices: dict[int, int] | None = None,
    training_role_feature_dropout_probability: float = 0.0,
    training_role_feature_dropout_id: int = -1,
) -> dict[str, torch.Tensor]:
    (
        numeric,
        token_mask,
        role_ids,
        token_type_ids,
        atomic_numbers,
        component_local_ids,
        density,
        _,
        global_components,
        sample_prior,
    ) = batch[:10]
    optional_index = 10
    if (
        readout.aho_consistency_adapter is not None
        or readout.ddg_major_summary_adapter is not None
    ):
        optional_index += 1
    cip_orientation_ids = None
    if readout.cip_orientation_embedding is not None:
        if len(batch) <= optional_index:
            raise RuntimeError(
                "CIP-enabled readout batch has no orientation tensor"
            )
        cip_orientation_ids = batch[optional_index].to(device)
        optional_index += 1
    smiles_sequence_token_ids = None
    smiles_sequence_token_mask = None
    if readout.smiles_sequence_adapter is not None:
        if len(batch) < optional_index + 2:
            raise RuntimeError(
                "SMILES-sequence-enabled readout batch lacks token tensors"
            )
        smiles_sequence_token_ids = batch[optional_index].to(device)
        smiles_sequence_token_mask = batch[optional_index + 1].to(device)
        optional_index += 2
    if len(batch) != optional_index:
        raise RuntimeError(
            "Unexpected optional tensors in the ORRA batch: "
            f"consumed {optional_index}, received {len(batch)}"
        )
    orb_level_features = None
    augmented_x = None
    secondary_x = None
    if (
        reactant_conformer_indices is not None
        and readout.dual_backbone_feature_fusion is not None
    ):
        raise RuntimeError(
            "Reactant single-conformer consistency cannot use dual "
            "backbone feature fusion"
        )
    if (
        reactant_conformer_indices is not None
        and readout.multi_level_orb_feature_attention is not None
    ):
        raise RuntimeError(
            "Reactant single-conformer consistency cannot use multi-level "
            "backbone features"
        )
    if (
        reactant_conformer_indices is not None
        and readout.conformer_attention_aggregator is not None
    ):
        raise RuntimeError(
            "Reactant single-conformer consistency requires fixed "
            "conformer aggregation"
        )
    if backbone_feature_source == "token_cache":
        if token_store is None:
            raise RuntimeError("Cached backbone features require a token store")
        if readout.multi_level_orb_feature_attention is not None:
            raise RuntimeError(
                "A single-level token cache cannot supply multi-level features"
            )
        if (
            readout.dual_backbone_feature_fusion is not None
            and secondary_token_store is None
        ):
            raise RuntimeError(
                "Dual backbone feature fusion requires a secondary token store"
            )
        if (
            readout.dual_backbone_feature_fusion is None
            and secondary_token_store is not None
        ):
            raise RuntimeError(
                "A secondary token store was supplied while dual backbone "
                "feature fusion is disabled"
            )
        component_features = cached_component_features(
            store=token_store,
            global_components=global_components,
            device=device,
            conformer_feature_mode=conformer_feature_mode,
            conformer_attention_aggregator=(
                readout.conformer_attention_aggregator
            ),
            single_conformer_indices=reactant_conformer_indices,
        )
        if reactant_conformer_indices is not None:
            component_features, augmented_component_features = (
                component_features
            )
        x = assemble_orb_tokens(
            component_features=component_features,
            token_mask=token_mask,
            token_type_ids=token_type_ids,
            component_local_ids=component_local_ids,
            global_components=global_components,
            device=device,
        )
        if secondary_token_store is not None:
            secondary_component_features = cached_component_features(
                store=secondary_token_store,
                global_components=global_components,
                device=device,
                conformer_feature_mode=conformer_feature_mode,
                conformer_attention_aggregator=None,
                single_conformer_indices=None,
            )
            secondary_x = assemble_orb_tokens(
                component_features=secondary_component_features,
                token_mask=token_mask,
                token_type_ids=token_type_ids,
                component_local_ids=component_local_ids,
                global_components=global_components,
                device=device,
            )
        if reactant_conformer_indices is not None:
            augmented_x = assemble_orb_tokens(
                component_features=augmented_component_features,
                token_mask=token_mask,
                token_type_ids=token_type_ids,
                component_local_ids=component_local_ids,
                global_components=global_components,
                device=device,
            )
    elif readout.multi_level_orb_feature_attention is None:
        if readout.dual_backbone_feature_fusion is not None:
            if reactant_conformer_indices is not None:
                raise RuntimeError(
                    "Online dual backbone feature fusion cannot use "
                    "reactant conformer consistency"
                )
            component_features, secondary_component_features = (
                jmp_component_dual_features(
                    orb_backbone=orb_backbone,
                    adapter=adapter,
                    payloads=payloads,
                    global_components=global_components,
                    device=device,
                    conformer_feature_mode=conformer_feature_mode,
                )
            )
        else:
            component_features = orb_component_features(
                orb_backbone=orb_backbone,
                adapter=adapter,
                payloads=payloads,
                global_components=global_components,
                device=device,
                conformer_feature_mode=conformer_feature_mode,
                conformer_attention_aggregator=(
                    readout.conformer_attention_aggregator
                ),
                single_conformer_indices=reactant_conformer_indices,
            )
        if reactant_conformer_indices is not None:
            component_features, augmented_component_features = (
                component_features
            )
        x = assemble_orb_tokens(
            component_features=component_features,
            token_mask=token_mask,
            token_type_ids=token_type_ids,
            component_local_ids=component_local_ids,
            global_components=global_components,
            device=device,
        )
        if readout.dual_backbone_feature_fusion is not None:
            secondary_x = assemble_orb_tokens(
                component_features=secondary_component_features,
                token_mask=token_mask,
                token_type_ids=token_type_ids,
                component_local_ids=component_local_ids,
                global_components=global_components,
                device=device,
            )
        if reactant_conformer_indices is not None:
            augmented_x = assemble_orb_tokens(
                component_features=augmented_component_features,
                token_mask=token_mask,
                token_type_ids=token_type_ids,
                component_local_ids=component_local_ids,
                global_components=global_components,
                device=device,
            )
    else:
        component_feature_levels = orb_component_feature_levels(
            orb_backbone=orb_backbone,
            adapter=adapter,
            payloads=payloads,
            global_components=global_components,
            device=device,
            conformer_feature_mode=conformer_feature_mode,
            conformer_attention_aggregator=(
                readout.conformer_attention_aggregator
            ),
            stack_indices=readout.multi_level_orb_stack_indices,
        )
        orb_level_features = assemble_orb_token_levels(
            component_feature_levels=component_feature_levels,
            token_mask=token_mask,
            token_type_ids=token_type_ids,
            component_local_ids=component_local_ids,
            global_components=global_components,
            device=device,
        )
        x = orb_level_features[..., -1, :]
    if training_role_feature_dropout_probability > 0.0:
        x = apply_training_role_feature_dropout(
            x,
            role_ids=role_ids,
            token_mask=token_mask,
            role_id=training_role_feature_dropout_id,
            probability=training_role_feature_dropout_probability,
        )
    output = readout(
        x,
        numeric.to(device),
        token_mask.to(device),
        role_ids.to(device),
        token_type_ids.to(device),
        atomic_numbers.to(device),
        density.to(device),
        component_ids=component_local_ids.to(device),
        cip_orientation_ids=cip_orientation_ids,
        orb_level_features=orb_level_features,
        secondary_orb_features=secondary_x,
        sample_prior=sample_prior.to(device),
        smiles_sequence_token_ids=smiles_sequence_token_ids,
        smiles_sequence_token_mask=smiles_sequence_token_mask,
        return_diagnostics=True,
    )
    if augmented_x is not None:
        mixed_augmented_x = mix_role_token_view(
            x,
            augmented_x,
            role_ids,
            token_mask,
            role_id=0,
        )
        augmented_output = readout(
            mixed_augmented_x,
            numeric.to(device),
            token_mask.to(device),
            role_ids.to(device),
            token_type_ids.to(device),
            atomic_numbers.to(device),
            density.to(device),
            component_ids=component_local_ids.to(device),
            cip_orientation_ids=cip_orientation_ids,
            orb_level_features=None,
            secondary_orb_features=None,
            sample_prior=sample_prior.to(device),
            smiles_sequence_token_ids=smiles_sequence_token_ids,
            smiles_sequence_token_mask=smiles_sequence_token_mask,
            return_diagnostics=True,
        )
        output["reactant_conformer_augmented_prediction"] = (
            augmented_output["prediction"]
        )
    return output


def loss_value(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    task_type: str,
    loss_name: str,
    huber_delta: float,
    pos_weight: torch.Tensor | None,
    label_smoothing: float,
    classification_focal_gamma: float = 0.0,
) -> torch.Tensor:
    if task_type == "classification":
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("--label-smoothing must be in [0, 1)")
        if classification_focal_gamma < 0.0:
            raise ValueError(
                "--classification-focal-gamma must be non-negative"
            )
        if classification_focal_gamma > 0.0 and label_smoothing != 0.0:
            raise ValueError(
                "Focal BCE requires --label-smoothing 0"
            )
        smoothed_target = (
            target
            if label_smoothing == 0.0
            else target * (1.0 - label_smoothing) + 0.5 * label_smoothing
        )
        if classification_focal_gamma == 0.0:
            return F.binary_cross_entropy_with_logits(
                prediction,
                smoothed_target,
                pos_weight=pos_weight,
            )
        focal_prediction = prediction.float()
        focal_target = target.float()
        per_sample_bce = F.binary_cross_entropy_with_logits(
            focal_prediction,
            focal_target,
            pos_weight=pos_weight,
            reduction="none",
        )
        unweighted_bce = F.binary_cross_entropy_with_logits(
            focal_prediction,
            focal_target,
            reduction="none",
        )
        one_minus_probability_of_target = -torch.expm1(
            -unweighted_bce
        )
        focal_weight = one_minus_probability_of_target.clamp_min(
            1e-6
        ).pow(classification_focal_gamma)
        return (focal_weight * per_sample_bce).mean()
    if loss_name == "mae":
        return F.l1_loss(prediction, target)
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    return F.huber_loss(prediction, target, delta=huber_delta)


def classification_pairwise_ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Smooth AUC surrogate over all positive-negative pairs in one batch."""
    if temperature <= 0.0:
        raise ValueError("classification ranking temperature must be positive")
    scores = prediction.reshape(-1)
    labels = target.reshape(-1)
    positive_scores = scores[labels > 0.5]
    negative_scores = scores[labels <= 0.5]
    if positive_scores.numel() == 0 or negative_scores.numel() == 0:
        return scores.sum() * 0.0
    pairwise_margin = (
        positive_scores.float().unsqueeze(1)
        - negative_scores.float().unsqueeze(0)
    ) / float(temperature)
    return F.softplus(-pairwise_margin).mean()


def validate_classification_focal_config(
    *,
    gamma: float,
    label_smoothing: float,
    task: str,
) -> bool:
    if gamma < 0.0:
        raise ValueError(
            "--classification-focal-gamma must be non-negative"
        )
    if gamma == 0.0:
        return False
    if task != "aho_major":
        raise ValueError(
            "--classification-focal-gamma is only defined for aho_major"
        )
    if label_smoothing != 0.0:
        raise ValueError(
            "Focal BCE requires --label-smoothing 0"
        )
    return True


def magnitude_aware_sign_margin_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    zero_point: float | torch.Tensor,
    temperature: float,
    dead_zone: float,
    magnitude_scale: float,
) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("sign-margin temperature must be positive")
    if dead_zone < 0.0:
        raise ValueError("sign-margin dead zone must be non-negative")
    if magnitude_scale <= 0.0:
        raise ValueError("sign-margin magnitude scale must be positive")
    zero = torch.as_tensor(
        zero_point,
        dtype=target.dtype,
        device=target.device,
    )
    centered_target = target - zero
    centered_prediction = prediction - zero
    magnitude = torch.abs(centered_target)
    active = magnitude > dead_zone
    if not bool(active.any()):
        return prediction.sum() * 0.0
    direction = torch.where(
        centered_target > 0.0,
        torch.ones_like(centered_target),
        -torch.ones_like(centered_target),
    )
    sample_weight = magnitude / (magnitude + magnitude_scale)
    per_sample = F.softplus(
        -temperature * direction * centered_prediction
    )
    active_weight = sample_weight * active.to(sample_weight.dtype)
    return (active_weight * per_sample).sum() / active_weight.sum().clamp_min(
        torch.finfo(active_weight.dtype).eps
    )


def magnitude_weighted_sign_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    magnitude: torch.Tensor,
    *,
    strength: float,
    magnitude_scale: float,
) -> torch.Tensor:
    if strength < 0.0:
        raise ValueError("magnitude-weighted sign BCE strength must be non-negative")
    if magnitude_scale <= 0.0:
        raise ValueError("magnitude-weighted sign BCE scale must be positive")
    if strength == 0.0:
        return F.binary_cross_entropy_with_logits(logits, target)
    importance = magnitude / (magnitude + magnitude_scale)
    sample_weight = 1.0 + strength * importance
    sample_weight = sample_weight / sample_weight.mean().detach().clamp_min(
        torch.finfo(sample_weight.dtype).eps
    )
    per_sample = F.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
    )
    return torch.mean(sample_weight * per_sample)


def ddg_sign_magnitude_auxiliary_objective(
    sign_loss: torch.Tensor,
    magnitude_loss: torch.Tensor,
    *,
    combined_weight: float,
    sign_only_weight: float,
) -> torch.Tensor:
    """Preserve legacy A6 weighting while optionally adding sign-only BCE."""
    if combined_weight < 0.0 or sign_only_weight < 0.0:
        raise ValueError("DDG auxiliary weights must be non-negative")
    objective = combined_weight * (sign_loss + magnitude_loss)
    if sign_only_weight > 0.0:
        objective = objective + sign_only_weight * sign_loss
    return objective


def aho_consistency_metrics(
    *,
    mode: str,
    y_true: np.ndarray,
    model_output: np.ndarray,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    if mode == "major_with_ddg":
        metrics, _ = task_metrics(
            y_true,
            model_output,
            task_type="regression",
            threshold=0.5,
            target_mean=target_mean,
            target_std=target_std,
        )
        return metrics
    if mode == "ddg_with_major":
        metrics, _ = task_metrics(
            y_true,
            model_output,
            task_type="classification",
            threshold=0.5,
            target_mean=0.0,
            target_std=1.0,
        )
        return metrics
    raise ValueError(f"Unsupported AHO consistency mode: {mode}")


def ddg_sign_magnitude_metrics(
    *,
    y_true: np.ndarray,
    model_output: dict[str, np.ndarray],
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    target_zero_point = -target_mean / target_std
    primary_sign_metrics, _ = task_metrics(
        (np.asarray(y_true) > 0.0).astype(np.float32),
        model_output["prediction"] - target_zero_point,
        task_type="classification",
        threshold=0.5,
        target_mean=0.0,
        target_std=1.0,
    )
    sign_metrics, _ = task_metrics(
        (np.asarray(y_true) > 0.0).astype(np.float32),
        model_output["ddg_sign_logit"],
        task_type="classification",
        threshold=0.5,
        target_mean=0.0,
        target_std=1.0,
    )
    magnitude_metrics, _ = task_metrics(
        np.abs(np.asarray(y_true)),
        model_output["ddg_magnitude_prediction"],
        task_type="regression",
        threshold=0.5,
        target_mean=0.0,
        target_std=target_std,
    )
    structured_metrics, _ = task_metrics(
        np.asarray(y_true),
        model_output["ddg_structured_prediction"],
        task_type="regression",
        threshold=0.5,
        target_mean=target_mean,
        target_std=target_std,
    )
    return {
        **{
            f"primary_sign_{key}": value
            for key, value in primary_sign_metrics.items()
        },
        **{
            f"sign_{key}": value
            for key, value in sign_metrics.items()
        },
        **{
            f"magnitude_{key}": value
            for key, value in magnitude_metrics.items()
        },
        **{
            f"structured_{key}": value
            for key, value in structured_metrics.items()
        },
    }


def regression_group_mean_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
) -> torch.Tensor:
    """MSE between per-group means, using only groups present in the batch."""
    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    group_ids = group_ids.reshape(-1).long()
    if (
        len(prediction) != len(target)
        or len(prediction) != len(group_ids)
        or len(prediction) == 0
    ):
        raise ValueError(
            "Prediction, target, and group IDs must be non-empty and aligned"
        )
    _, inverse = torch.unique(
        group_ids,
        sorted=True,
        return_inverse=True,
    )
    group_count = int(inverse.max().item()) + 1
    counts = prediction.new_zeros(group_count)
    prediction_sums = prediction.new_zeros(group_count)
    target_sums = target.new_zeros(group_count)
    ones = torch.ones_like(prediction)
    counts.scatter_add_(0, inverse, ones)
    prediction_sums.scatter_add_(0, inverse, prediction)
    target_sums.scatter_add_(0, inverse, target)
    return F.mse_loss(
        prediction_sums / counts.clamp_min(1.0),
        target_sums / counts.clamp_min(1.0),
    )


def train_epoch(
    *,
    readout: UnifiedORRAModel,
    orb_backbone: nn.Module,
    adapter,
    payloads: dict[int, dict],
    token_store=None,
    secondary_token_store=None,
    backbone_feature_source: str = "online",
    readout_train_scope: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_type: str,
    loss_name: str,
    huber_delta: float,
    pos_weight: torch.Tensor | None,
    label_smoothing: float,
    classification_focal_gamma: float,
    classification_pairwise_ranking_weight: float,
    classification_pairwise_ranking_temperature: float,
    gate_regularization: float,
    readout_gradient_clip_norm: float,
    orb_gradient_clip_norm: float,
    conformer_feature_mode: str,
    reactant_conformer_view_probability: float,
    reactant_conformer_consistency_weight: float,
    reactant_conformer_consistency_huber_delta: float,
    orb_module_mode: str,
    readout_module_mode: str,
    site_donor_relation_auxiliary_weight: float,
    smiles_sequence_auxiliary_weight: float,
    aho_consistency_weight: float,
    aho_consistency_huber_delta: float,
    ddg_major_summary_weight: float,
    ddg_sign_magnitude_weight: float,
    ddg_sign_only_auxiliary_weight: float,
    ddg_sign_magnitude_huber_delta: float,
    ddg_sign_bce_magnitude_strength: float,
    ddg_sign_bce_magnitude_scale: float,
    ddg_output_sign_margin_weight: float,
    ddg_output_sign_margin_temperature: float,
    ddg_output_sign_margin_zero_point: float,
    ddg_output_sign_margin_dead_zone: float,
    ddg_output_sign_margin_magnitude_scale: float,
    regression_group_mean_weight: float,
    training_role_feature_dropout_probability: float,
    training_role_feature_dropout_id: int,
    use_amp: bool,
) -> dict[str, float]:
    if classification_pairwise_ranking_weight < 0.0:
        raise ValueError("classification ranking weight must be non-negative")
    if classification_focal_gamma < 0.0:
        raise ValueError(
            "classification focal gamma must be non-negative"
        )
    if classification_focal_gamma > 0.0 and task_type != "classification":
        raise ValueError(
            "classification focal loss is only valid for classification"
        )
    if classification_focal_gamma > 0.0 and label_smoothing != 0.0:
        raise ValueError("Focal BCE requires label smoothing 0")
    if (
        classification_pairwise_ranking_weight > 0.0
        and task_type != "classification"
    ):
        raise ValueError(
            "classification ranking loss is only valid for classification"
        )
    if (
        classification_pairwise_ranking_weight > 0.0
        and classification_pairwise_ranking_temperature <= 0.0
    ):
        raise ValueError("classification ranking temperature must be positive")
    if smiles_sequence_auxiliary_weight < 0.0:
        raise ValueError(
            "SMILES sequence auxiliary weight must be non-negative"
        )
    if (
        smiles_sequence_auxiliary_weight > 0.0
        and (
            readout.smiles_sequence_adapter is None
            or readout.smiles_sequence_adapter.auxiliary_head is None
        )
    ):
        raise ValueError(
            "Positive SMILES sequence auxiliary weight requires its head"
        )
    configure_readout_training_mode(
        readout,
        readout_train_scope=readout_train_scope,
        module_mode=readout_module_mode,
    )
    if orb_module_mode == "train":
        orb_backbone.train()
    else:
        orb_backbone.eval()
    losses = []
    auxiliary_losses = []
    major_summary_losses = []
    relation_auxiliary_losses = []
    smiles_sequence_auxiliary_losses = []
    sign_losses = []
    magnitude_losses = []
    sign_margin_losses = []
    classification_ranking_losses = []
    reactant_conformer_consistency_losses = []
    regression_group_mean_losses = []
    gates = []
    for batch in loader:
        training_group_ids = None
        if regression_group_mean_weight > 0.0:
            if len(batch) <= 10:
                raise RuntimeError(
                    "Group-mean training loss has no group-ID tensor"
                )
            training_group_ids = batch[-1].to(device)
            batch = batch[:-1]
        optimizer.zero_grad(set_to_none=True)
        reactant_conformer_indices = None
        if reactant_conformer_view_probability > 0.0:
            reactant_component_ids = component_ids_for_role(
                role_ids=batch[2],
                token_mask=batch[1],
                component_local_ids=batch[5],
                global_components=batch[8],
                role_id=0,
            )
            if backbone_feature_source == "token_cache":
                if token_store is None:
                    raise RuntimeError(
                        "Cached conformer consistency requires a token store"
                    )
                conformer_counts = {
                    component_id: int(
                        np.asarray(
                            token_store.conformer_mask[component_id]
                        ).sum()
                    )
                    for component_id in reactant_component_ids
                }
            else:
                conformer_counts = {}
                for component_id in reactant_component_ids:
                    positions = np.asarray(
                        payloads[component_id]["positions"]
                    )
                    conformer_counts[component_id] = (
                        1 if positions.ndim == 2 else int(len(positions))
                    )
            reactant_conformer_indices = sample_single_conformer_indices(
                reactant_component_ids,
                conformer_counts,
                probability=reactant_conformer_view_probability,
            )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            output = forward_batch(
                readout=readout,
                orb_backbone=orb_backbone,
                adapter=adapter,
                payloads=payloads,
                token_store=token_store,
                secondary_token_store=secondary_token_store,
                backbone_feature_source=backbone_feature_source,
                batch=batch,
                device=device,
                conformer_feature_mode=conformer_feature_mode,
                reactant_conformer_indices=reactant_conformer_indices,
                training_role_feature_dropout_probability=(
                    training_role_feature_dropout_probability
                ),
                training_role_feature_dropout_id=(
                    training_role_feature_dropout_id
                ),
            )
            target = batch[7].to(device)
            primary = loss_value(
                output["prediction"],
                target,
                task_type=task_type,
                loss_name=loss_name,
                huber_delta=huber_delta,
                pos_weight=pos_weight,
                label_smoothing=label_smoothing,
                classification_focal_gamma=(
                    classification_focal_gamma
                ),
            )
            gate_penalty = torch.mean((1.0 - output["confidence"]) ** 2)
            loss = primary + gate_regularization * gate_penalty
            if regression_group_mean_weight > 0.0:
                if training_group_ids is None:
                    raise RuntimeError(
                        "Group-mean training IDs were not prepared"
                    )
                group_mean_loss = regression_group_mean_loss(
                    output["prediction"],
                    target,
                    training_group_ids,
                )
                loss = (
                    loss
                    + regression_group_mean_weight * group_mean_loss
                )
                regression_group_mean_losses.append(
                    float(group_mean_loss.detach().cpu())
                )
            if smiles_sequence_auxiliary_weight > 0.0:
                sequence_auxiliary_loss = loss_value(
                    output["smiles_sequence_auxiliary_prediction"],
                    target,
                    task_type=task_type,
                    loss_name=loss_name,
                    huber_delta=huber_delta,
                    pos_weight=pos_weight,
                    label_smoothing=label_smoothing,
                    classification_focal_gamma=0.0,
                )
                loss = (
                    loss
                    + smiles_sequence_auxiliary_weight
                    * sequence_auxiliary_loss
                )
                smiles_sequence_auxiliary_losses.append(
                    float(sequence_auxiliary_loss.detach().cpu())
                )
            if reactant_conformer_consistency_weight > 0.0:
                consistency_loss = prediction_consistency_loss(
                    output["prediction"],
                    output["reactant_conformer_augmented_prediction"],
                    huber_delta=(
                        reactant_conformer_consistency_huber_delta
                    ),
                )
                loss = loss + (
                    reactant_conformer_consistency_weight
                    * consistency_loss
                )
                reactant_conformer_consistency_losses.append(
                    float(consistency_loss.detach().cpu())
                )
            if classification_pairwise_ranking_weight > 0.0:
                ranking_loss = classification_pairwise_ranking_loss(
                    output["prediction"],
                    target,
                    temperature=(
                        classification_pairwise_ranking_temperature
                    ),
                )
                loss = loss + (
                    classification_pairwise_ranking_weight * ranking_loss
                )
                classification_ranking_losses.append(
                    float(ranking_loss.detach().cpu())
                )
            if site_donor_relation_auxiliary_weight > 0.0:
                if readout.site_donor_relation_attention is None:
                    raise RuntimeError(
                        "Relation auxiliary loss has no relation adapter"
                    )
                relation_auxiliary_loss = loss_value(
                    output["site_donor_relation_logit"],
                    target,
                    task_type="classification",
                    loss_name=loss_name,
                    huber_delta=huber_delta,
                    pos_weight=pos_weight,
                    label_smoothing=label_smoothing,
                )
                loss = loss + (
                    site_donor_relation_auxiliary_weight
                    * relation_auxiliary_loss
                )
                relation_auxiliary_losses.append(
                    float(relation_auxiliary_loss.detach().cpu())
                )
            if readout.aho_consistency_adapter is not None:
                if len(batch) < 11:
                    raise RuntimeError(
                        "AHO consistency training batch has no auxiliary target"
                    )
                auxiliary_target = batch[10].to(device)
                if readout.aho_consistency_mode == "major_with_ddg":
                    auxiliary_loss = F.huber_loss(
                        output["auxiliary_prediction"],
                        auxiliary_target,
                        delta=aho_consistency_huber_delta,
                    )
                else:
                    auxiliary_loss = F.binary_cross_entropy_with_logits(
                        output["auxiliary_prediction"],
                        auxiliary_target,
                    )
                loss = loss + aho_consistency_weight * auxiliary_loss
                auxiliary_losses.append(
                    float(auxiliary_loss.detach().cpu())
                )
            if readout.ddg_major_summary_adapter is not None:
                if len(batch) < 11:
                    raise RuntimeError(
                        "DDG Major-summary batch has no Major target"
                    )
                major_target = batch[10].to(device)
                view_logits = output["ddg_major_summary_view_logits"]
                view_mask = output[
                    "ddg_major_summary_view_mask"
                ].to(view_logits.dtype)
                per_view_loss = (
                    F.binary_cross_entropy_with_logits(
                        view_logits,
                        major_target.unsqueeze(-1).expand_as(
                            view_logits
                        ),
                        reduction="none",
                    )
                    * view_mask
                ).sum() / view_mask.sum().clamp_min(1.0)
                aggregate_loss = F.binary_cross_entropy_with_logits(
                    output["ddg_major_summary_logit"],
                    major_target,
                )
                major_summary_loss = 0.5 * (
                    per_view_loss + aggregate_loss
                )
                loss = (
                    loss
                    + ddg_major_summary_weight * major_summary_loss
                )
                major_summary_losses.append(
                    float(major_summary_loss.detach().cpu())
                )
            if readout.ddg_sign_magnitude_adapter is not None:
                zero_point = (
                    readout.ddg_sign_magnitude_adapter.target_zero_point
                ).to(target)
                centered_target = target - zero_point
                sign_target = (centered_target > 0.0).to(target.dtype)
                magnitude_target = torch.abs(centered_target)
                sign_loss = magnitude_weighted_sign_bce_loss(
                    output["ddg_sign_logit"],
                    sign_target,
                    magnitude_target,
                    strength=ddg_sign_bce_magnitude_strength,
                    magnitude_scale=ddg_sign_bce_magnitude_scale,
                )
                magnitude_loss = F.huber_loss(
                    output["ddg_magnitude_prediction"],
                    magnitude_target,
                    delta=ddg_sign_magnitude_huber_delta,
                )
                loss = loss + ddg_sign_magnitude_auxiliary_objective(
                    sign_loss,
                    magnitude_loss,
                    combined_weight=ddg_sign_magnitude_weight,
                    sign_only_weight=ddg_sign_only_auxiliary_weight,
                )
                sign_losses.append(float(sign_loss.detach().cpu()))
                magnitude_losses.append(
                    float(magnitude_loss.detach().cpu())
                )
            if ddg_output_sign_margin_weight > 0.0:
                sign_margin_loss = magnitude_aware_sign_margin_loss(
                    output["prediction"],
                    target,
                    zero_point=ddg_output_sign_margin_zero_point,
                    temperature=ddg_output_sign_margin_temperature,
                    dead_zone=ddg_output_sign_margin_dead_zone,
                    magnitude_scale=(
                        ddg_output_sign_margin_magnitude_scale
                    ),
                )
                loss = loss + (
                    ddg_output_sign_margin_weight * sign_margin_loss
                )
                sign_margin_losses.append(
                    float(sign_margin_loss.detach().cpu())
                )
        loss.backward()
        if readout_gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                readout.parameters(),
                max_norm=readout_gradient_clip_norm,
            )
        trainable_orb = [parameter for parameter in orb_backbone.parameters() if parameter.requires_grad]
        if orb_gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                trainable_orb,
                max_norm=orb_gradient_clip_norm,
            )
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gates.append(float(output["confidence"].detach().mean().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "auxiliary_loss": (
            float(np.mean(auxiliary_losses))
            if auxiliary_losses
            else float("nan")
        ),
        "ddg_major_summary_loss": (
            float(np.mean(major_summary_losses))
            if major_summary_losses
            else float("nan")
        ),
        "site_donor_relation_auxiliary_loss": (
            float(np.mean(relation_auxiliary_losses))
            if relation_auxiliary_losses
            else float("nan")
        ),
        "smiles_sequence_auxiliary_loss": (
            float(np.mean(smiles_sequence_auxiliary_losses))
            if smiles_sequence_auxiliary_losses
            else float("nan")
        ),
        "ddg_sign_loss": (
            float(np.mean(sign_losses))
            if sign_losses
            else float("nan")
        ),
        "ddg_magnitude_loss": (
            float(np.mean(magnitude_losses))
            if magnitude_losses
            else float("nan")
        ),
        "ddg_sign_margin_loss": (
            float(np.mean(sign_margin_losses))
            if sign_margin_losses
            else float("nan")
        ),
        "classification_pairwise_ranking_loss": (
            float(np.mean(classification_ranking_losses))
            if classification_ranking_losses
            else float("nan")
        ),
        "reactant_conformer_consistency_loss": (
            float(np.mean(reactant_conformer_consistency_losses))
            if reactant_conformer_consistency_losses
            else float("nan")
        ),
        "regression_group_mean_loss": (
            float(np.mean(regression_group_mean_losses))
            if regression_group_mean_losses
            else float("nan")
        ),
        "confidence": float(np.mean(gates)),
    }


@torch.no_grad()
def predict(
    *,
    readout: UnifiedORRAModel,
    orb_backbone: nn.Module,
    adapter,
    payloads: dict[int, dict],
    token_store=None,
    secondary_token_store=None,
    backbone_feature_source: str = "online",
    loader: DataLoader,
    device: torch.device,
    conformer_feature_mode: str,
    use_amp: bool,
    collect_fusion: bool = False,
) -> dict[str, np.ndarray]:
    readout.eval()
    orb_backbone.eval()
    collected = {
        "prediction": [],
        "raw_prediction": [],
        "confidence": [],
        "ood_excess": [],
        "ood_shift": [],
    }
    if readout.aho_consistency_adapter is not None:
        collected["auxiliary_prediction"] = []
        collected["consistency_residual"] = []
    if readout.site_donor_relation_attention is not None:
        collected["site_donor_relation_logit"] = []
        collected["site_donor_relation_residual"] = []
        collected["site_donor_relation_coupling"] = []
    if readout.reaction_pair_transformer is not None:
        collected["reaction_pair_transformer_residual"] = []
    if readout.smiles_sequence_adapter is not None:
        collected["smiles_sequence_residual"] = []
        if readout.smiles_sequence_adapter.auxiliary_head is not None:
            collected["smiles_sequence_auxiliary_prediction"] = []
    if readout.ddg_sign_magnitude_adapter is not None:
        collected["direct_raw_prediction"] = []
        collected["ddg_sign_logit"] = []
        collected["ddg_magnitude_prediction"] = []
        collected["ddg_structured_prediction"] = []
        collected["ddg_structured_residual"] = []
    if readout.ddg_residual_calibrator is not None:
        collected["pre_calibration_prediction"] = []
        collected["ddg_calibration_residual"] = []
    if readout.ddg_major_summary_adapter is not None:
        collected["ddg_major_summary_view_logits"] = []
        collected["ddg_major_summary_view_mask"] = []
        collected["ddg_major_summary_logit"] = []
        collected["ddg_major_summary_disagreement"] = []
        collected["ddg_major_summary_residual"] = []
    if collect_fusion:
        collected["fusion"] = []
    for batch in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            output = forward_batch(
                readout=readout,
                orb_backbone=orb_backbone,
                adapter=adapter,
                payloads=payloads,
                token_store=token_store,
                secondary_token_store=secondary_token_store,
                backbone_feature_source=backbone_feature_source,
                batch=batch,
                device=device,
                conformer_feature_mode=conformer_feature_mode,
            )
        for key in collected:
            collected[key].append(output[key].float().cpu().numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}


def clone_states(
    readout: nn.Module,
    orb_backbone: nn.Module,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    return (
        {key: value.detach().cpu().clone() for key, value in readout.state_dict().items()},
        {key: value.detach().cpu().clone() for key, value in orb_backbone.state_dict().items()},
    )


def main() -> None:
    args = parse_args()
    regression_task = args.task != "aho_major"
    if args.validation_mode == "group_column":
        if not args.group_val_column.strip():
            raise ValueError(
                "--group-val-column is required when "
                "--validation-mode=group_column"
            )
        if not 0.0 < args.group_val_fraction < 1.0:
            raise ValueError(
                "--group-val-fraction must be strictly between 0 and 1"
            )
        if args.group_val_strategy == "balanced_fold":
            if args.group_val_fold_count < 2:
                raise ValueError(
                    "--group-val-fold-count must be at least 2 for "
                    "balanced_fold"
                )
            if not (
                0
                <= args.group_val_fold_index
                < args.group_val_fold_count
            ):
                raise ValueError(
                    "--group-val-fold-index must be in "
                    "[0, --group-val-fold-count)"
                )
    if args.regression_group_mean_weight < 0.0:
        raise ValueError(
            "--regression-group-mean-weight must be non-negative"
        )
    if args.regression_group_mean_weight > 0.0:
        if not regression_task:
            raise ValueError(
                "Regression group-mean loss requires a regression task"
            )
        if not args.regression_group_mean_column.strip():
            raise ValueError(
                "--regression-group-mean-column is required when its "
                "weight is positive"
            )
        if args.final_retrain_train_plus_val:
            raise ValueError(
                "Regression group-mean loss is currently a CV/training "
                "selection feature and is not supported with final "
                "train+validation concatenation"
            )
    if not (
        0.0
        <= args.training_role_feature_dropout_probability
        <= 1.0
    ):
        raise ValueError(
            "--training-role-feature-dropout-probability must be in [0, 1]"
        )
    if (
        args.training_role_feature_dropout_probability > 0.0
        and args.training_role_feature_dropout_id < 0
    ):
        raise ValueError(
            "Positive training role-feature dropout requires a "
            "non-negative role ID"
        )
    epoch_ema_enabled = validate_epoch_ema_decay(args.epoch_ema_decay)
    cip_orientation_embedding_enabled = (
        validate_cip_orientation_embedding_config(
            enabled=args.cip_orientation_embedding,
            task=args.task,
            backbone_feature_source=args.backbone_feature_source,
            backbone_train_scope=args.orb_train_scope,
            backbone_lr=args.orb_lr,
            readout_train_scope=args.readout_train_scope,
            readout_module_mode=args.readout_module_mode,
            resume_finetune_checkpoint=(
                args.resume_finetune_checkpoint
            ),
            resume_readout_load_mode=args.resume_readout_load_mode,
            resume_backbone_load_mode=args.resume_backbone_load_mode,
        )
    )
    smiles_sequence_adapter_enabled = (
        validate_smiles_sequence_adapter_config(
            enabled=args.smiles_sequence_adapter,
            auxiliary_enabled=args.smiles_sequence_auxiliary,
            auxiliary_weight=args.smiles_sequence_auxiliary_weight,
            auxiliary_warmup_epochs=(
                args.smiles_sequence_auxiliary_warmup_epochs
            ),
            auxiliary_post_warmup_lr=(
                args.smiles_sequence_auxiliary_post_warmup_lr
            ),
            readout_train_scope=args.readout_train_scope,
            vocab_size=args.smiles_sequence_vocab_size,
            max_length=args.smiles_sequence_max_length,
            hidden_dim=args.smiles_sequence_hidden_dim,
            num_layers=args.smiles_sequence_num_layers,
            num_heads=args.smiles_sequence_num_heads,
            dropout=args.smiles_sequence_dropout,
            residual_multiplier=(
                args.smiles_sequence_residual_multiplier
            ),
        )
    )
    if args.smiles_sequence_catalyst_h8_augmentation:
        if args.task != "cpa_catalyst":
            raise ValueError(
                "--smiles-sequence-catalyst-h8-augmentation is only "
                "defined for cpa_catalyst"
            )
        if not smiles_sequence_adapter_enabled:
            raise ValueError(
                "--smiles-sequence-catalyst-h8-augmentation requires "
                "--smiles-sequence-adapter"
            )
    elif args.smiles_sequence_catalyst_h8_target_mode != "copy_target":
        raise ValueError(
            "A non-default H8 target mode requires "
            "--smiles-sequence-catalyst-h8-augmentation"
        )
    if args.smiles_sequence_catalyst_h8_randomized_views < 0:
        raise ValueError(
            "--smiles-sequence-catalyst-h8-randomized-views must be "
            "non-negative"
        )
    if (
        args.smiles_sequence_catalyst_h8_randomized_views > 0
        and not args.smiles_sequence_catalyst_h8_augmentation
    ):
        raise ValueError(
            "Randomized H8 SMILES views require "
            "--smiles-sequence-catalyst-h8-augmentation"
        )
    if not (
        0.0
        <= args.smiles_sequence_catalyst_h8_target_quantile
        <= 1.0
    ):
        raise ValueError(
            "--smiles-sequence-catalyst-h8-target-quantile must be in [0, 1]"
        )
    if not (
        0.0 <= args.smiles_sequence_catalyst_h8_target_mix <= 1.0
    ):
        raise ValueError(
            "--smiles-sequence-catalyst-h8-target-mix must be in [0, 1]"
        )
    if (
        args.smiles_sequence_catalyst_h8_target_mix != 1.0
        and args.smiles_sequence_catalyst_h8_target_mode
        != "condition_quantile"
    ):
        raise ValueError(
            "A non-default H8 target mix requires condition_quantile mode"
        )
    h8_target_condition_columns = tuple(
        column.strip()
        for column in (
            args.smiles_sequence_catalyst_h8_target_condition_columns
        ).split(",")
        if column.strip()
    )
    if (
        args.smiles_sequence_catalyst_h8_target_mode
        == "condition_quantile"
        and not h8_target_condition_columns
    ):
        raise ValueError(
            "condition_quantile H8 targets require at least one condition "
            "column"
        )
    if (
        args.smiles_sequence_auxiliary
        and args.smiles_sequence_auxiliary_warmup_epochs
        >= args.epochs
    ):
        raise ValueError(
            "--smiles-sequence-auxiliary-warmup-epochs must be smaller "
            "than --epochs"
        )
    checkpoint_selection_split = validate_checkpoint_selection_split(
        checkpoint_selection_split=args.checkpoint_selection_split,
        skip_test_evaluation=args.skip_test_evaluation,
        acknowledge_observed_test_selection=(
            args.acknowledge_observed_test_selection
        ),
    )
    final_retrain_train_plus_val = (
        validate_final_retrain_train_plus_val_config(
            enabled=args.final_retrain_train_plus_val,
            checkpoint_selection_split=checkpoint_selection_split,
            export_fusion_features=args.export_fusion_features,
        )
    )
    reactant_conformer_consistency_enabled = (
        validate_reactant_conformer_consistency_config(
            probability=args.reactant_conformer_view_probability,
            weight=args.reactant_conformer_consistency_weight,
            huber_delta=(
                args.reactant_conformer_consistency_huber_delta
            ),
            conformer_aggregation=args.conformer_aggregation,
            multi_level_orb_attention=args.multi_level_orb_attention,
        )
    )
    dual_backbone_feature_fusion_enabled = (
        validate_dual_backbone_feature_fusion_config(
            enabled=args.dual_backbone_feature_fusion,
            secondary_atom_token_dir=args.secondary_atom_token_dir,
            backbone_feature_source=args.backbone_feature_source,
            backbone_family=args.backbone_family,
            conformer_feature_mode=args.conformer_feature_mode,
            conformer_aggregation=args.conformer_aggregation,
            multi_level_orb_attention=args.multi_level_orb_attention,
            reactant_conformer_consistency_enabled=(
                reactant_conformer_consistency_enabled
            ),
            hidden_dim=(
                args.dual_backbone_feature_fusion_hidden_dim
            ),
            role_gate_enabled=(
                args.dual_backbone_feature_role_gate
            ),
        )
    )
    if args.backbone_feature_source == "token_cache":
        if args.multi_level_orb_attention:
            raise ValueError(
                "--backbone-feature-source token_cache currently supports "
                "one final node-embedding level only"
            )
        if args.orb_train_scope != "frozen" or args.orb_lr != 0.0:
            raise ValueError(
                "Token-cache mode is a frozen-backbone Stage-2 screen; use "
                "--orb-train-scope frozen --orb-lr 0"
            )
    elif args.backbone_family == "jmp":
        if args.multi_level_orb_attention:
            raise ValueError(
                "JMP online execution does not expose Orb stack levels"
            )
        if args.orb_train_scope == "stacks":
            raise ValueError(
                "JMP online execution requires --orb-train-scope frozen, "
                "jmp_output, jmp_last_block, jmp_last_two_blocks, or "
                "full_backbone"
            )
    elif args.orb_train_scope in {
        "jmp_output",
        "jmp_last_block",
        "jmp_last_two_blocks",
    }:
        raise ValueError(
            "JMP-specific train scopes cannot be used with the Orb family"
        )
    if not 0.0 <= args.adam_beta1 < 1.0:
        raise ValueError("--adam-beta1 must be in [0, 1)")
    if not 0.0 <= args.adam_beta2 < 1.0:
        raise ValueError("--adam-beta2 must be in [0, 1)")
    if args.adam_eps <= 0.0:
        raise ValueError("--adam-eps must be positive")
    if args.readout_gradient_clip_norm < 0.0:
        raise ValueError("--readout-gradient-clip-norm must be non-negative")
    if args.orb_gradient_clip_norm < 0.0:
        raise ValueError("--orb-gradient-clip-norm must be non-negative")
    if args.major_accuracy_gate <= 0.0:
        raise ValueError("--major-accuracy-gate must be positive")
    if args.major_auroc_gate <= 0.0:
        raise ValueError("--major-auroc-gate must be positive")
    if args.ddg_r2_gate <= 0.0:
        raise ValueError("--ddg-r2-gate must be positive")
    if args.ddg_mae_gate <= 0.0:
        raise ValueError("--ddg-mae-gate must be positive")
    validate_classification_focal_config(
        gamma=args.classification_focal_gamma,
        label_smoothing=args.label_smoothing,
        task=args.task,
    )
    if args.classification_pairwise_ranking_weight < 0.0:
        raise ValueError(
            "--classification-pairwise-ranking-weight must be non-negative"
        )
    if (
        args.classification_pairwise_ranking_weight > 0.0
        and args.task != "aho_major"
    ):
        raise ValueError(
            "--classification-pairwise-ranking-weight is only defined for "
            "aho_major"
        )
    if (
        args.classification_pairwise_ranking_weight > 0.0
        and args.classification_pairwise_ranking_temperature <= 0.0
    ):
        raise ValueError(
            "--classification-pairwise-ranking-temperature must be positive"
        )
    if (
        (
            args.selection_metric == "major_gate_balance"
            or args.scheduler_metric == "major_gate_balance"
        )
        and args.task != "aho_major"
    ):
        raise ValueError(
            "major_gate_balance validation monitoring requires --task aho_major"
        )
    if (
        (
            args.selection_metric == "ddg_gate_balance"
            or args.scheduler_metric == "ddg_gate_balance"
        )
        and args.task != "aho_ddg"
    ):
        raise ValueError(
            "ddg_gate_balance validation monitoring requires --task aho_ddg"
        )
    if args.site_donor_relation_num_layers <= 0:
        raise ValueError(
            "--site-donor-relation-num-layers must be positive"
        )
    if args.reaction_pair_transformer_num_layers <= 0:
        raise ValueError(
            "--reaction-pair-transformer-num-layers must be positive"
        )
    if args.site_donor_relation_auxiliary_weight < 0.0:
        raise ValueError(
            "--site-donor-relation-auxiliary-weight must be non-negative"
        )
    if (
        args.site_donor_relation_auxiliary_weight > 0.0
        and (
            args.task != "aho_major"
            or not args.site_donor_relation_attention
        )
    ):
        raise ValueError(
            "--site-donor-relation-auxiliary-weight requires "
            "--task aho_major and --site-donor-relation-attention"
        )
    if args.aho_consistency_weight < 0.0:
        raise ValueError("--aho-consistency-weight must be non-negative")
    if args.aho_consistency_huber_delta <= 0.0:
        raise ValueError("--aho-consistency-huber-delta must be positive")
    if args.ddg_sign_magnitude_weight < 0.0:
        raise ValueError("--ddg-sign-magnitude-weight must be non-negative")
    if args.ddg_sign_only_auxiliary_weight < 0.0:
        raise ValueError(
            "--ddg-sign-only-auxiliary-weight must be non-negative"
        )
    if (
        args.ddg_sign_only_auxiliary_weight > 0.0
        and (
            args.task != "aho_ddg"
            or not args.ddg_sign_magnitude
        )
    ):
        raise ValueError(
            "--ddg-sign-only-auxiliary-weight requires --task aho_ddg "
            "and --ddg-sign-magnitude"
        )
    if args.ddg_sign_magnitude_huber_delta <= 0.0:
        raise ValueError(
            "--ddg-sign-magnitude-huber-delta must be positive"
        )
    if args.ddg_sign_magnitude_hidden_dim <= 0:
        raise ValueError(
            "--ddg-sign-magnitude-hidden-dim must be positive"
        )
    if args.ddg_sign_magnitude and not regression_task:
        raise ValueError(
            "--ddg-sign-magnitude requires a regression task"
        )
    if args.ddg_sign_magnitude and args.aho_consistency_auxiliary:
        raise ValueError(
            "--ddg-sign-magnitude and --aho-consistency-auxiliary are "
            "mutually exclusive incremental heads"
        )
    if args.ddg_residual_calibrator and (
        not regression_task or not args.ddg_sign_magnitude
    ):
        raise ValueError(
            "--ddg-residual-calibrator requires --task aho_ddg and "
            "--ddg-sign-magnitude"
        )
    if args.ddg_residual_calibrator_nonlinear_hidden_dim < 0:
        raise ValueError(
            "--ddg-residual-calibrator-nonlinear-hidden-dim must be "
            "non-negative"
        )
    if (
        args.ddg_residual_calibrator_nonlinear_hidden_dim > 0
        and (
            not regression_task
            or not args.ddg_residual_calibrator
        )
    ):
        raise ValueError(
            "--ddg-residual-calibrator-nonlinear-hidden-dim requires "
            "--task aho_ddg and --ddg-residual-calibrator"
        )
    if args.ddg_major_summary and (
        args.task != "aho_ddg"
        or not args.ddg_sign_magnitude
        or not args.ddg_residual_calibrator
    ):
        raise ValueError(
            "--ddg-major-summary requires --task aho_ddg, "
            "--ddg-sign-magnitude, and --ddg-residual-calibrator"
        )
    if (
        args.ddg_major_summary_classifier_hidden_dim <= 0
        or args.ddg_major_summary_residual_hidden_dim <= 0
    ):
        raise ValueError(
            "--ddg-major-summary hidden dimensions must be positive"
        )
    if args.ddg_major_summary_weight < 0.0:
        raise ValueError(
            "--ddg-major-summary-weight must be non-negative"
        )
    if args.ddg_major_summary_residual_multiplier < 0.0:
        raise ValueError(
            "--ddg-major-summary-residual-multiplier must be non-negative"
        )
    if (
        (
            args.ddg_major_summary_weight != 0.25
            or args.ddg_major_summary_residual_multiplier != 1.0
        )
        and not args.ddg_major_summary
    ):
        raise ValueError(
            "Non-default DDG Major-summary settings require "
            "--ddg-major-summary"
        )
    if args.ddg_major_summary_classifier_warmup_epochs < 0:
        raise ValueError(
            "--ddg-major-summary-classifier-warmup-epochs must be "
            "non-negative"
        )
    if (
        args.ddg_major_summary_classifier_warmup_epochs > 0
        and not args.ddg_major_summary
    ):
        raise ValueError(
            "--ddg-major-summary-classifier-warmup-epochs requires "
            "--ddg-major-summary"
        )
    if (
        args.ddg_major_summary
        and args.ddg_major_summary_classifier_warmup_epochs
        >= args.epochs
    ):
        raise ValueError(
            "--ddg-major-summary-classifier-warmup-epochs must be "
            "smaller than --epochs"
        )
    if args.ddg_major_summary_post_warmup_lr < 0.0:
        raise ValueError(
            "--ddg-major-summary-post-warmup-lr must be non-negative"
        )
    if (
        args.ddg_major_summary_post_warmup_lr > 0.0
        and (
            not args.ddg_major_summary
            or args.ddg_major_summary_classifier_warmup_epochs == 0
        )
    ):
        raise ValueError(
            "--ddg-major-summary-post-warmup-lr requires "
            "--ddg-major-summary and positive classifier warmup epochs"
        )
    if args.ddg_sign_magnitude_residual_multiplier < 0.0:
        raise ValueError(
            "--ddg-sign-magnitude-residual-multiplier must be non-negative"
        )
    if (
        args.ddg_sign_magnitude_residual_multiplier != 1.0
        and (not regression_task or not args.ddg_sign_magnitude)
    ):
        raise ValueError(
            "--ddg-sign-magnitude-residual-multiplier requires "
            "--task aho_ddg and --ddg-sign-magnitude"
        )
    if args.ddg_sign_bce_magnitude_strength < 0.0:
        raise ValueError(
            "--ddg-sign-bce-magnitude-strength must be non-negative"
        )
    if args.ddg_sign_bce_magnitude_scale <= 0.0:
        raise ValueError(
            "--ddg-sign-bce-magnitude-scale must be positive"
        )
    if (
        args.ddg_sign_bce_magnitude_strength > 0.0
        and (args.task != "aho_ddg" or not args.ddg_sign_magnitude)
    ):
        raise ValueError(
            "--ddg-sign-bce-magnitude-strength requires "
            "--task aho_ddg and --ddg-sign-magnitude"
        )
    if args.ddg_output_sign_margin_weight < 0.0:
        raise ValueError(
            "--ddg-output-sign-margin-weight must be non-negative"
        )
    if args.ddg_output_sign_margin_temperature <= 0.0:
        raise ValueError(
            "--ddg-output-sign-margin-temperature must be positive"
        )
    if args.ddg_output_sign_margin_dead_zone < 0.0:
        raise ValueError(
            "--ddg-output-sign-margin-dead-zone must be non-negative"
        )
    if args.ddg_output_sign_margin_magnitude_scale <= 0.0:
        raise ValueError(
            "--ddg-output-sign-margin-magnitude-scale must be positive"
        )
    if (
        args.ddg_output_sign_margin_weight > 0.0
        and args.task != "aho_ddg"
    ):
        raise ValueError(
            "--ddg-output-sign-margin-weight is only defined for aho_ddg"
        )
    token_dir, out_dir, conformer_cache, mapping_task = resolve_paths(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    use_amp = bool(args.amp and device.type == "cuda")

    store = load_atom_token_store(token_dir)
    secondary_store = None
    token_manifest: dict[str, object] | None = None
    secondary_token_manifest: dict[str, object] | None = None
    dual_token_cache_alignment: dict[str, object] | None = None
    primary_token_manifest_sha256 = ""
    secondary_token_manifest_sha256 = ""
    if args.backbone_feature_source == "token_cache":
        if not args.orb_checkpoint.is_file():
            raise FileNotFoundError(args.orb_checkpoint)
        manifest_path = token_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Token-cache provenance manifest is missing: {manifest_path}"
            )
        with manifest_path.open(encoding="utf-8") as handle:
            token_manifest = json.load(handle)
        primary_token_manifest_sha256 = sha256_file(manifest_path)
        checkpoint_hash_key = (
            "jmp_checkpoint_sha256"
            if args.backbone_family == "jmp"
            else "orb_checkpoint_sha256"
        )
        expected_checkpoint_hash = token_manifest.get(checkpoint_hash_key)
        actual_checkpoint_hash = sha256_file(args.orb_checkpoint)
        if expected_checkpoint_hash != actual_checkpoint_hash:
            raise RuntimeError(
                "Token-cache checkpoint provenance mismatch: "
                f"manifest={expected_checkpoint_hash}, "
                f"actual={actual_checkpoint_hash}"
            )
        if token_manifest.get("test_labels_or_predictions_used") is True:
            raise RuntimeError("Token cache is not test-sealed")
        if dual_backbone_feature_fusion_enabled:
            secondary_token_dir = args.secondary_atom_token_dir
            if secondary_token_dir is None:
                raise RuntimeError(
                    "Dual feature fusion lost its secondary token directory"
                )
            secondary_store = load_atom_token_store(secondary_token_dir)
            secondary_manifest_path = (
                secondary_token_dir / "manifest.json"
            )
            if not secondary_manifest_path.is_file():
                raise FileNotFoundError(
                    "Secondary token-cache provenance manifest is missing: "
                    f"{secondary_manifest_path}"
                )
            with secondary_manifest_path.open(encoding="utf-8") as handle:
                secondary_token_manifest = json.load(handle)
            secondary_token_manifest_sha256 = sha256_file(
                secondary_manifest_path
            )
            secondary_expected_checkpoint_hash = (
                secondary_token_manifest.get(checkpoint_hash_key)
            )
            if (
                secondary_expected_checkpoint_hash
                != actual_checkpoint_hash
            ):
                raise RuntimeError(
                    "Secondary token-cache checkpoint provenance mismatch: "
                    f"manifest={secondary_expected_checkpoint_hash}, "
                    f"actual={actual_checkpoint_hash}"
                )
            dual_token_cache_alignment = validate_dual_atom_token_stores(
                primary_store=store,
                secondary_store=secondary_store,
                primary_manifest=token_manifest,
                secondary_manifest=secondary_token_manifest,
                task=args.task,
            )
    task = load_task(args.task, split=args.split)
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
            fold_count=args.group_val_fold_count,
            fold_index=args.group_val_fold_index,
        )
        effective_mapping_task = (
            f"{mapping_task}__unified_groupval_"
            f"{args.group_val_column.lower()}_{args.group_val_strategy}"
        )
        if args.group_val_strategy == "balanced_fold":
            effective_mapping_task += (
                f"_{args.group_val_fold_index:02d}"
                f"of{args.group_val_fold_count:02d}"
            )
        store = store_with_source_index_mapping(
            store=store,
            mapping_task=mapping_task,
            target_task=effective_mapping_task,
            frame_by_part=frames,
        )
        if secondary_store is not None:
            secondary_store = store_with_source_index_mapping(
                store=secondary_store,
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
    raw_bundles = {
        part: build_tensor_bundle(
            store=store,
            mapping_task=effective_mapping_task,
            part=part_names[part],
            target_frame=frames[part],
            target_column=task.target_column,
            max_rows=args.max_train_rows if part == "train" else args.max_eval_rows,
            max_tokens=args.max_tokens,
            conformer_feature_mode=args.conformer_feature_mode,
        )
        for part in parts
    }
    training_reference_bundle = raw_bundles["train"]
    training_reference_frame = frames["train"]
    if final_retrain_train_plus_val:
        training_reference_bundle = concatenate_atom_relation_bundles(
            raw_bundles["train"],
            raw_bundles["val"],
        )
        training_reference_frame = pd.concat(
            [frames["train"], frames["val"]],
            axis=0,
            ignore_index=True,
        )
    global_components = {
        part: global_component_matrix(
            store,
            mapping_task=effective_mapping_task,
            part=part_names[part],
            row_index=raw_bundles[part].row_index,
        )
        for part in parts
    }
    cip_orientation_alignment: dict[str, object] = {
        "enabled": False,
        "id_schema": dict(CIP_ORIENTATION_IDS),
        "parsed_components": 0,
    }
    cip_orientation_ids: dict[str, torch.Tensor] = {}
    if cip_orientation_embedding_enabled:
        used_component_ids = sorted(
            {
                int(component_id)
                for part in parts
                for component_id in global_components[part]
                .reshape(-1)
                .tolist()
                if int(component_id) >= 0
            }
        )
        component_cip_lookup, cip_orientation_alignment = (
            build_cip_orientation_lookup(
                store,
                used_component_ids,
            )
        )
        cip_orientation_ids = {
            part: assemble_cip_orientation_ids(
                raw_bundle=raw_bundles[part],
                global_components=global_components[part],
                component_lookup=component_cip_lookup,
            )
            for part in parts
        }
        cip_orientation_alignment["split_token_inventory"] = {
            part: {
                "rows": int(len(cip_orientation_ids[part])),
                "rows_with_cip": int(
                    (cip_orientation_ids[part] != 0)
                    .any(dim=1)
                    .sum()
                ),
                "r_tokens": int(
                    (
                        cip_orientation_ids[part]
                        == CIP_ORIENTATION_IDS["R"]
                    ).sum()
                ),
                "s_tokens": int(
                    (
                        cip_orientation_ids[part]
                        == CIP_ORIENTATION_IDS["S"]
                    ).sum()
                ),
                "other_tokens": int(
                    (
                        cip_orientation_ids[part]
                        == CIP_ORIENTATION_IDS["other"]
                    ).sum()
                ),
            }
            for part in parts
        }
    smiles_sequence_tokens: dict[
        str,
        tuple[torch.Tensor, torch.Tensor],
    ] = {}
    smiles_sequence_h8_training_tokens: dict[
        str,
        tuple[torch.Tensor, torch.Tensor],
    ] = {}
    smiles_sequence_h8_randomized_training_tokens: dict[
        str,
        list[tuple[torch.Tensor, torch.Tensor]],
    ] = {}
    smiles_sequence_alignment: dict[str, object] = {
        "enabled": False,
        "token_ids": dict(SMILES_BYTE_TOKEN_IDS),
        "vocab_size": SMILES_BYTE_VOCAB_SIZE,
        "split_inventory": {},
    }
    if smiles_sequence_adapter_enabled:
        split_inventory = {}
        for part in parts:
            token_ids, token_mask, inventory = (
                build_byte_smiles_sequences(
                    frame=frames[part],
                    row_index=raw_bundles[part].row_index,
                    input_columns=tuple(task.input_columns),
                    max_length=args.smiles_sequence_max_length,
                )
            )
            smiles_sequence_tokens[part] = (
                token_ids,
                token_mask,
            )
            split_inventory[part] = inventory
        if args.smiles_sequence_catalyst_h8_augmentation:
            h8_parts = ["train"]
            if final_retrain_train_plus_val:
                h8_parts.append("val")
            h8_inventory = {}
            h8_randomized_inventory: dict[
                str,
                list[dict[str, object]],
            ] = {}
            for part in h8_parts:
                token_ids, token_mask, inventory = (
                    build_byte_smiles_sequences(
                        frame=frames[part],
                        row_index=raw_bundles[part].row_index,
                        input_columns=tuple(task.input_columns),
                        max_length=args.smiles_sequence_max_length,
                        catalyst_h8_view=True,
                    )
                )
                if inventory["catalyst_h8_augmented_rows"] != len(
                    raw_bundles[part].row_index
                ):
                    raise RuntimeError(
                        "CPA Catalyst H8 augmentation failed to transform "
                        f"every {part} row"
                    )
                if inventory["catalyst_h8_augmented_rings"] != 2 * len(
                    raw_bundles[part].row_index
                ):
                    raise RuntimeError(
                        "CPA Catalyst H8 augmentation must transform exactly "
                        f"two scaffold rings per {part} row"
                    )
                smiles_sequence_h8_training_tokens[part] = (
                    token_ids,
                    token_mask,
                )
                h8_inventory[part] = inventory
                randomized_tokens = []
                randomized_inventory = []
                for randomized_view_index in range(
                    args.smiles_sequence_catalyst_h8_randomized_views
                ):
                    (
                        randomized_token_ids,
                        randomized_token_mask,
                        randomized_metadata,
                    ) = build_byte_smiles_sequences(
                        frame=frames[part],
                        row_index=raw_bundles[part].row_index,
                        input_columns=tuple(task.input_columns),
                        max_length=args.smiles_sequence_max_length,
                        catalyst_h8_view=True,
                        randomized_view_index=randomized_view_index,
                        randomized_seed=args.seed,
                    )
                    randomized_tokens.append(
                        (
                            randomized_token_ids,
                            randomized_token_mask,
                        )
                    )
                    randomized_inventory.append(randomized_metadata)
                smiles_sequence_h8_randomized_training_tokens[part] = (
                    randomized_tokens
                )
                h8_randomized_inventory[part] = randomized_inventory
        smiles_sequence_alignment = {
            "enabled": True,
            "token_ids": dict(SMILES_BYTE_TOKEN_IDS),
            "vocab_size": SMILES_BYTE_VOCAB_SIZE,
            "input_columns": list(task.input_columns),
            "split_inventory": split_inventory,
            "catalyst_h8_training_augmentation": bool(
                args.smiles_sequence_catalyst_h8_augmentation
            ),
            "catalyst_h8_training_inventory": (
                h8_inventory
                if args.smiles_sequence_catalyst_h8_augmentation
                else {}
            ),
            "catalyst_h8_randomized_views": int(
                args.smiles_sequence_catalyst_h8_randomized_views
            ),
            "catalyst_h8_randomized_training_inventory": (
                h8_randomized_inventory
                if args.smiles_sequence_catalyst_h8_augmentation
                else {}
            ),
        }
    density_profile = ComponentDensityProfile(
        num_roles=16,
        quantile=args.density_quantile,
        unseen_floor=args.unseen_density_floor,
        ratio_cap=args.density_ratio_cap,
    ).fit(training_reference_bundle)
    density = {
        part: density_profile.transform(
            raw_bundles[part],
            training=(
                (
                    part == "train"
                    or (
                        final_retrain_train_plus_val
                        and part == "val"
                    )
                )
                and args.density_train_mode == "leave_one_component_out"
            ),
        )
        for part in parts
    }
    payloads = (
        prepare_conformer_payloads(
            store,
            conformer_cache,
            args.conformer_seed,
        )
        if args.backbone_feature_source == "online"
        else {}
    )

    if task.task_type == "regression":
        target_mean = float(training_reference_bundle.y.mean())
        target_std = float(
            training_reference_bundle.y.std(unbiased=False)
        )
        if target_std < 1e-6:
            target_std = 1.0
        targets = {
            part: (raw_bundles[part].y - target_mean) / target_std
            for part in parts
        }
        prior = 0.0
    else:
        target_mean, target_std = 0.0, 1.0
        targets = {part: raw_bundles[part].y for part in parts}
        positive_rate = float(raw_bundles["train"].y.mean())
        positive_rate = min(max(positive_rate, 1e-4), 1.0 - 1e-4)
        prior = math.log(positive_rate / (1.0 - positive_rate))
    if not 0.0 <= args.prior_quantile <= 1.0:
        raise ValueError("--prior-quantile must be in [0, 1]")
    prior_columns = tuple(
        column.strip()
        for column in args.prior_condition_columns.split(",")
        if column.strip()
    )
    if args.sample_prior_mode == "condition_quantile":
        if task.task_type != "regression":
            raise ValueError(
                "condition_quantile sample priors require regression"
            )
        if not prior_columns:
            raise ValueError(
                "--prior-condition-columns is required for "
                "condition_quantile sample priors"
            )
        sample_prior_by_part = {}
        for part in parts:
            raw_prior = train_only_condition_prior(
                train_frame=training_reference_frame,
                pred_frame=frames[part],
                target_column=task.target_column,
                condition_columns=prior_columns,
                quantile=args.prior_quantile,
                row_index=raw_bundles[part].row_index,
            )
            sample_prior_by_part[part] = torch.from_numpy(
                ((raw_prior - target_mean) / target_std).astype(
                    np.float32
                )
            )
    elif args.sample_prior_mode == "train_quantile":
        if task.task_type != "regression":
            raise ValueError(
                "train_quantile sample priors require regression"
            )
        raw_prior = float(
            pd.to_numeric(
                training_reference_frame[task.target_column],
                errors="coerce",
            ).quantile(args.prior_quantile)
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
    ddg_sign_magnitude_zero_point = (
        -target_mean / target_std
        if args.ddg_sign_magnitude
        else 0.0
    )
    ddg_sign_bce_magnitude_scale = (
        args.ddg_sign_bce_magnitude_scale / target_std
        if args.task == "aho_ddg"
        else args.ddg_sign_bce_magnitude_scale
    )
    ddg_output_sign_margin_zero_point = (
        -target_mean / target_std
        if args.task == "aho_ddg"
        else 0.0
    )
    ddg_output_sign_margin_dead_zone = (
        args.ddg_output_sign_margin_dead_zone / target_std
        if args.task == "aho_ddg"
        else args.ddg_output_sign_margin_dead_zone
    )
    ddg_output_sign_margin_magnitude_scale = (
        args.ddg_output_sign_margin_magnitude_scale / target_std
        if args.task == "aho_ddg"
        else args.ddg_output_sign_margin_magnitude_scale
    )

    aho_consistency_mode = "none"
    auxiliary_targets: dict[str, torch.Tensor] | None = None
    auxiliary_true_values: dict[str, np.ndarray] = {}
    auxiliary_target_column = ""
    auxiliary_target_mean = 0.0
    auxiliary_target_std = 1.0
    auxiliary_zero_point = 0.0
    if args.aho_consistency_auxiliary:
        if args.task == "aho_major":
            aho_consistency_mode = "major_with_ddg"
            auxiliary_target_column = "signed_ddG"
            auxiliary_physical = {
                part: torch.from_numpy(
                    indexed_true_values(
                        frames[part],
                        auxiliary_target_column,
                        raw_bundles[part].row_index,
                    )
                ).float()
                for part in parts
            }
            auxiliary_target_mean = float(
                torch.cat(
                    [
                        auxiliary_physical["train"],
                        *(
                            [auxiliary_physical["val"]]
                            if final_retrain_train_plus_val
                            else []
                        ),
                    ]
                ).mean()
            )
            auxiliary_target_std = float(
                torch.cat(
                    [
                        auxiliary_physical["train"],
                        *(
                            [auxiliary_physical["val"]]
                            if final_retrain_train_plus_val
                            else []
                        ),
                    ]
                ).std(unbiased=False)
            )
            if auxiliary_target_std < 1e-6:
                auxiliary_target_std = 1.0
            auxiliary_zero_point = (
                -auxiliary_target_mean / auxiliary_target_std
            )
            auxiliary_targets = {
                part: (
                    auxiliary_physical[part] - auxiliary_target_mean
                )
                / auxiliary_target_std
                for part in parts
            }
            auxiliary_true_values = {
                part: auxiliary_physical[part].numpy()
                for part in parts
                if part != "train"
            }
        elif args.task == "aho_ddg":
            aho_consistency_mode = "ddg_with_major"
            auxiliary_target_column = "Label"
            auxiliary_targets = {
                part: torch.from_numpy(
                    indexed_true_values(
                        frames[part],
                        auxiliary_target_column,
                        raw_bundles[part].row_index,
                    )
                ).float()
                for part in parts
            }
            auxiliary_true_values = {
                part: auxiliary_targets[part].numpy()
                for part in parts
                if part != "train"
            }
        else:
            raise ValueError(
                "--aho-consistency-auxiliary is only defined for "
                "aho_major and aho_ddg"
            )
    elif args.ddg_major_summary:
        auxiliary_target_column = "Label"
        auxiliary_targets = {
            part: torch.from_numpy(
                indexed_true_values(
                    frames[part],
                    auxiliary_target_column,
                    raw_bundles[part].row_index,
                )
            ).float()
            for part in parts
        }
        auxiliary_true_values = {
            part: auxiliary_targets[part].numpy()
            for part in parts
            if part != "train"
        }

    regression_group_ids: dict[str, torch.Tensor] = {}
    if args.regression_group_mean_weight > 0.0:
        group_column = args.regression_group_mean_column.strip()
        regression_group_ids["train"] = aligned_training_group_ids(
            frames["train"],
            column=group_column,
            row_index=raw_bundles["train"].row_index,
        )

    loaders = {
        part: make_loader(
            raw_bundles[part],
            density[part],
            targets[part],
            global_components[part],
            sample_prior_by_part[part],
            batch_size=args.batch_size,
            shuffle=part == "train",
            auxiliary_target=(
                auxiliary_targets[part]
                if auxiliary_targets is not None
                else None
            ),
            cip_orientation_ids=(
                cip_orientation_ids[part]
                if cip_orientation_embedding_enabled
                else None
            ),
            smiles_sequence_token_ids=(
                smiles_sequence_tokens[part][0]
                if smiles_sequence_adapter_enabled
                else None
            ),
            smiles_sequence_token_mask=(
                smiles_sequence_tokens[part][1]
                if smiles_sequence_adapter_enabled
                else None
            ),
            training_group_ids=(
                regression_group_ids["train"]
                if (
                    part == "train"
                    and args.regression_group_mean_weight > 0.0
                )
                else None
            ),
        )
        for part in parts
    }
    original_train_rows = len(loaders["train"].dataset)
    training_datasets = [loaders["train"].dataset]
    if final_retrain_train_plus_val:
        training_datasets.append(loaders["val"].dataset)
    h8_augmented_training_rows = 0
    h8_randomized_augmented_training_rows = 0
    h8_target_inventory: dict[str, dict[str, float | int | str]] = {}
    if args.smiles_sequence_catalyst_h8_augmentation:
        for part, (token_ids, token_mask) in (
            smiles_sequence_h8_training_tokens.items()
        ):
            augmented_targets = targets[part]
            if (
                args.smiles_sequence_catalyst_h8_target_mode
                == "condition_quantile"
            ):
                raw_augmented_targets = train_only_condition_prior(
                    train_frame=training_reference_frame,
                    pred_frame=frames[part],
                    target_column=task.target_column,
                    condition_columns=h8_target_condition_columns,
                    quantile=(
                        args.smiles_sequence_catalyst_h8_target_quantile
                    ),
                    row_index=raw_bundles[part].row_index,
                )
                raw_original_targets = indexed_true_values(
                    frames[part],
                    task.target_column,
                    raw_bundles[part].row_index,
                ).astype(np.float32)
                target_mix = float(
                    args.smiles_sequence_catalyst_h8_target_mix
                )
                raw_augmented_targets = (
                    (1.0 - target_mix) * raw_original_targets
                    + target_mix * raw_augmented_targets
                ).astype(np.float32)
                augmented_targets = torch.from_numpy(
                    (
                        (raw_augmented_targets - target_mean)
                        / target_std
                    ).astype(np.float32)
                )
                h8_target_inventory[part] = {
                    "mode": "condition_quantile",
                    "target_mix": target_mix,
                    "rows": int(len(raw_augmented_targets)),
                    "raw_mean": float(raw_augmented_targets.mean()),
                    "raw_std": float(raw_augmented_targets.std()),
                    "raw_min": float(raw_augmented_targets.min()),
                    "raw_max": float(raw_augmented_targets.max()),
                }
            else:
                h8_target_inventory[part] = {
                    "mode": "copy_target",
                    "rows": int(len(augmented_targets)),
                }
            augmented_loader = make_loader(
                raw_bundles[part],
                density[part],
                augmented_targets,
                global_components[part],
                sample_prior_by_part[part],
                batch_size=args.batch_size,
                shuffle=False,
                auxiliary_target=(
                    auxiliary_targets[part]
                    if auxiliary_targets is not None
                    else None
                ),
                cip_orientation_ids=(
                    cip_orientation_ids[part]
                    if cip_orientation_embedding_enabled
                    else None
                ),
                smiles_sequence_token_ids=token_ids,
                smiles_sequence_token_mask=token_mask,
                training_group_ids=(
                    regression_group_ids[part]
                    if args.regression_group_mean_weight > 0.0
                    else None
                ),
            )
            training_datasets.append(augmented_loader.dataset)
            h8_augmented_training_rows += len(augmented_loader.dataset)
            for (
                randomized_token_ids,
                randomized_token_mask,
            ) in smiles_sequence_h8_randomized_training_tokens.get(
                part,
                [],
            ):
                randomized_loader = make_loader(
                    raw_bundles[part],
                    density[part],
                    augmented_targets,
                    global_components[part],
                    sample_prior_by_part[part],
                    batch_size=args.batch_size,
                    shuffle=False,
                    auxiliary_target=(
                        auxiliary_targets[part]
                        if auxiliary_targets is not None
                        else None
                    ),
                    cip_orientation_ids=(
                        cip_orientation_ids[part]
                        if cip_orientation_embedding_enabled
                        else None
                    ),
                    smiles_sequence_token_ids=randomized_token_ids,
                    smiles_sequence_token_mask=randomized_token_mask,
                    training_group_ids=(
                        regression_group_ids[part]
                        if args.regression_group_mean_weight > 0.0
                        else None
                    ),
                )
                training_datasets.append(randomized_loader.dataset)
                h8_randomized_augmented_training_rows += len(
                    randomized_loader.dataset
                )
    if len(training_datasets) > 1:
        loaders["train"] = DataLoader(
            ConcatDataset(training_datasets),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
    effective_training_rows = len(loaders["train"].dataset)
    true_value_parts = (
        parts
        if args.export_fusion_features
        else tuple(part for part in parts if part != "train")
    )
    true_values = {
        part: indexed_true_values(frames[part], task.target_column, raw_bundles[part].row_index)
        for part in true_value_parts
    }

    multi_level_orb_stack_indices = tuple(
        int(index) for index in args.multi_level_orb_stack_indices
    )
    if (
        args.backbone_feature_source == "online"
        and args.backbone_family == "orb"
    ):
        from orb_models.forcefield import pretrained

        orb_regressor, adapter = pretrained.orb_v2(
            weights_path=str(args.orb_checkpoint),
            device=device,
            compile=False,
            train=True,
        )
        orb_backbone = orb_regressor.model
        if args.multi_level_orb_attention:
            multi_level_orb_stack_indices = validate_orb_stack_indices(
                multi_level_orb_stack_indices,
                num_stacks=len(orb_backbone.gnn_stacks),
            )
        trainable_backbone_names = configure_backbone_trainability(
            orb_backbone,
            family=args.backbone_family,
            scope=args.orb_train_scope,
            unfreeze_orb_stacks=args.unfreeze_orb_stacks,
        )
    elif args.backbone_feature_source == "online":
        from full_retrain.aho_amc.jmp_backend import (
            load_fused_jmp_checkpoint,
        )

        orb_backbone, adapter, jmp_checkpoint_payload = (
            load_fused_jmp_checkpoint(
                args.orb_checkpoint,
                device=device,
            )
        )
        if jmp_checkpoint_payload.get("task") != args.task:
            raise RuntimeError(
                "JMP Fuse task provenance mismatch: "
                f"{jmp_checkpoint_payload.get('task')} != {args.task}"
            )
        trainable_backbone_names = configure_backbone_trainability(
            orb_backbone,
            family=args.backbone_family,
            scope=args.orb_train_scope,
            unfreeze_orb_stacks=args.unfreeze_orb_stacks,
        )
    else:
        # The materialized checkpoint has already been executed exactly once
        # during label-free token extraction. Stage-2 cache screening has no
        # hidden predictor or second backbone instance.
        orb_backbone = nn.Identity().to(device)
        adapter = None
        trainable_backbone_names = []

    orb_dim = {
        "mean": 256,
        "mean_std": 512,
        "mean_std_max": 768,
    }[args.conformer_feature_mode]
    readout = UnifiedORRAModel(
        orb_dim=orb_dim,
        numeric_dim=8,
        density_dim=19,
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
        use_grouped_statistics=args.grouped_statistics,
        use_cross_view_interactions=args.cross_view_interactions,
        use_conformer_attention=args.conformer_aggregation == "attention",
        conformer_attention_hidden_dim=args.conformer_attention_hidden_dim,
        use_directed_role_attention=args.directed_role_attention,
        directed_query_role_id=0,
        directed_context_role_ids=(2, 8, 9),
        use_reaction_pair_transformer=(
            args.reaction_pair_transformer
        ),
        reaction_pair_transformer_num_layers=(
            args.reaction_pair_transformer_num_layers
        ),
        reaction_pair_role_ids=(0, 2, 8, 9),
        use_site_ligand_metal_attention=args.site_ligand_metal_attention,
        site_token_type_ids=(1, 2, 3),
        ligand_token_type_ids=(6, 7),
        metal_token_type_ids=(9,),
        use_site_donor_relation_attention=(
            args.site_donor_relation_attention
        ),
        site_donor_relation_num_layers=(
            args.site_donor_relation_num_layers
        ),
        use_multi_level_orb_attention=args.multi_level_orb_attention,
        multi_level_orb_num_levels=len(multi_level_orb_stack_indices),
        multi_level_orb_score_hidden_dim=(
            args.multi_level_orb_score_hidden_dim
        ),
        multi_level_orb_stack_indices=multi_level_orb_stack_indices,
        use_dual_backbone_feature_fusion=(
            dual_backbone_feature_fusion_enabled
        ),
        dual_backbone_feature_fusion_hidden_dim=(
            args.dual_backbone_feature_fusion_hidden_dim
        ),
        use_dual_backbone_feature_role_gate=(
            args.dual_backbone_feature_role_gate
        ),
        use_cip_orientation_embedding=(
            cip_orientation_embedding_enabled
        ),
        use_smiles_sequence_adapter=(
            smiles_sequence_adapter_enabled
        ),
        smiles_sequence_vocab_size=(
            args.smiles_sequence_vocab_size
        ),
        smiles_sequence_max_length=(
            args.smiles_sequence_max_length
        ),
        smiles_sequence_hidden_dim=(
            args.smiles_sequence_hidden_dim
        ),
        smiles_sequence_num_layers=(
            args.smiles_sequence_num_layers
        ),
        smiles_sequence_num_heads=(
            args.smiles_sequence_num_heads
        ),
        smiles_sequence_dropout=(
            args.smiles_sequence_dropout
        ),
        smiles_sequence_residual_multiplier=(
            args.smiles_sequence_residual_multiplier
        ),
        use_smiles_sequence_auxiliary_head=(
            args.smiles_sequence_auxiliary
        ),
        aho_consistency_mode=aho_consistency_mode,
        aho_consistency_hidden_dim=args.aho_consistency_hidden_dim,
        aho_consistency_auxiliary_zero_point=auxiliary_zero_point,
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
        use_ddg_major_summary=args.ddg_major_summary,
        ddg_major_summary_classifier_hidden_dim=(
            args.ddg_major_summary_classifier_hidden_dim
        ),
        ddg_major_summary_residual_hidden_dim=(
            args.ddg_major_summary_residual_hidden_dim
        ),
        ddg_major_summary_role_ids=(0, 2, 8, 9),
        ddg_major_summary_residual_multiplier=(
            args.ddg_major_summary_residual_multiplier
        ),
        ood_threshold=args.ood_threshold,
        ood_scale_init=args.ood_scale_init,
        exponential_ood_gate=args.ood_gate_mode == "exponential",
        gate_residual_mode=args.gate_residual_mode,
        ood_shift_scale=args.ood_shift_scale,
        ood_base_shift=args.ood_base_shift,
        residual_global_center_mix=args.residual_global_center_mix,
    ).to(device)
    init_info: dict[str, object] | None = None
    resume_info: dict[str, object] | None = None
    if (
        args.init_readout_checkpoint is not None
        and args.resume_finetune_checkpoint is not None
    ):
        raise ValueError(
            "Use only one of --init-readout-checkpoint and "
            "--resume-finetune-checkpoint"
        )
    if args.init_readout_checkpoint is not None:
        init_info = initialize_unified_model(
            readout,
            args.init_readout_checkpoint,
            mode=args.init_readout_mode,
        )
    elif args.resume_finetune_checkpoint is not None:
        resume_checkpoint = torch.load(
            args.resume_finetune_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        if args.resume_readout_load_mode == "strict":
            readout.load_state_dict(
                resume_checkpoint["readout_state_dict"],
                strict=True,
            )
            readout_transfer = {
                "mode": "strict",
                "transferred_tensors": len(readout.state_dict()),
                "transferred_parameters": int(
                    sum(value.numel() for value in readout.state_dict().values())
                ),
            }
        elif args.resume_readout_load_mode == "compatible":
            source_readout_state = resume_checkpoint["readout_state_dict"]
            target_readout_state = readout.state_dict()
            incompatible_source = sorted(
                key
                for key, value in source_readout_state.items()
                if key not in target_readout_state
                or target_readout_state[key].shape != value.shape
            )
            if incompatible_source:
                raise ValueError(
                    "Compatible resume may add new target modules but cannot "
                    "drop or reshape parent readout tensors; incompatible keys "
                    f"include: {', '.join(incompatible_source[:8])}"
                )
            transferred_state = {
                key: value
                for key, value in source_readout_state.items()
                if key in target_readout_state
                and target_readout_state[key].shape == value.shape
            }
            load_result = readout.load_state_dict(
                transferred_state,
                strict=False,
            )
            if load_result.unexpected_keys:
                raise RuntimeError(
                    "Compatible resume produced unexpected readout keys: "
                    f"{load_result.unexpected_keys[:8]}"
                )
            new_target_keys = sorted(
                set(target_readout_state).difference(transferred_state)
            )
            if sorted(load_result.missing_keys) != new_target_keys:
                raise RuntimeError(
                    "Compatible resume missing-key accounting is inconsistent"
                )
            readout_transfer = {
                "mode": "compatible",
                "transferred_tensors": len(transferred_state),
                "transferred_parameters": int(
                    sum(
                        target_readout_state[key].numel()
                        for key in transferred_state
                    )
                ),
                "new_tensors": len(new_target_keys),
                "new_parameters": int(
                    sum(
                        target_readout_state[key].numel()
                        for key in new_target_keys
                    )
                ),
                "new_keys": new_target_keys,
            }
        else:
            source_readout_state = resume_checkpoint["readout_state_dict"]
            target_readout_state = readout.state_dict()
            transferred_state = {
                key: value
                for key, value in source_readout_state.items()
                if key in target_readout_state
                and target_readout_state[key].shape == value.shape
            }
            if not transferred_state:
                raise RuntimeError(
                    "Intersection resume found no compatible readout tensors"
                )
            load_result = readout.load_state_dict(
                transferred_state,
                strict=False,
            )
            if load_result.unexpected_keys:
                raise RuntimeError(
                    "Intersection resume produced unexpected readout keys: "
                    f"{load_result.unexpected_keys[:8]}"
                )
            dropped_source_keys = sorted(
                key
                for key, value in source_readout_state.items()
                if key not in target_readout_state
                or target_readout_state[key].shape != value.shape
            )
            new_target_keys = sorted(load_result.missing_keys)
            readout_transfer = {
                "mode": "intersection",
                "transferred_tensors": len(transferred_state),
                "transferred_parameters": int(
                    sum(value.numel() for value in transferred_state.values())
                ),
                "dropped_source_tensors": len(dropped_source_keys),
                "dropped_source_keys": dropped_source_keys,
                "new_tensors": len(new_target_keys),
                "new_keys": new_target_keys,
            }
        if args.resume_backbone_load_mode == "strict":
            orb_backbone.load_state_dict(
                resume_checkpoint["orb_backbone_state_dict"],
                strict=True,
            )
            backbone_transfer = {
                "mode": "strict",
                "transferred_tensors": len(orb_backbone.state_dict()),
            }
        else:
            backbone_transfer = {
                "mode": "skip",
                "transferred_tensors": 0,
                "reason": "explicit cross-family Stage-2 readout transfer",
            }
        resume_info = {
            "checkpoint": str(args.resume_finetune_checkpoint),
            "source_best_epoch": int(resume_checkpoint.get("best_epoch", -1)),
            "readout_transfer": readout_transfer,
            "backbone_transfer": backbone_transfer,
        }
        if cip_orientation_embedding_enabled:
            expected_new_keys = [
                "cip_orientation_embedding.weight",
            ]
            if readout_transfer.get("new_keys") != expected_new_keys:
                raise RuntimeError(
                    "CIP compatible resume must add exactly "
                    f"{expected_new_keys}, got "
                    f"{readout_transfer.get('new_keys')}"
                )
            if not torch.equal(
                readout.cip_orientation_embedding.weight,
                torch.zeros_like(
                    readout.cip_orientation_embedding.weight
                ),
            ):
                raise RuntimeError(
                    "CIP orientation embedding did not preserve zero "
                    "initialization after parent transfer"
                )
        if (
            smiles_sequence_adapter_enabled
            and args.resume_readout_load_mode == "compatible"
        ):
            new_keys = readout_transfer.get("new_keys", [])
            if not new_keys or any(
                not key.startswith("smiles_sequence_adapter.")
                for key in new_keys
            ):
                raise RuntimeError(
                    "SMILES compatible resume may add only sequence-adapter "
                    f"state, got {new_keys[:8]}"
                )
            output_layer = (
                readout.smiles_sequence_adapter.residual_head[-1]
            )
            if not torch.equal(
                output_layer.weight,
                torch.zeros_like(output_layer.weight),
            ) or not torch.equal(
                output_layer.bias,
                torch.zeros_like(output_layer.bias),
            ):
                raise RuntimeError(
                    "SMILES sequence residual output lost zero "
                    "initialization after parent transfer"
                )
    if args.freeze_gate:
        for parameter in readout.gate_head.parameters():
            parameter.requires_grad = False
        readout.ood_scale_raw.requires_grad = False
    if args.readout_train_scope == "conformer_attention":
        if readout.conformer_attention_aggregator is None:
            raise ValueError(
                "--readout-train-scope conformer_attention requires "
                "--conformer-aggregation attention"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.conformer_attention_aggregator.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "directed_role_attention":
        if readout.directed_role_attention is None:
            raise ValueError(
                "--readout-train-scope directed_role_attention requires "
                "--directed-role-attention"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.directed_role_attention.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "reaction_pair_transformer":
        if readout.reaction_pair_transformer is None:
            raise ValueError(
                "--readout-train-scope reaction_pair_transformer requires "
                "--reaction-pair-transformer"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.reaction_pair_transformer.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "site_ligand_metal_attention":
        if readout.site_ligand_metal_attention is None:
            raise ValueError(
                "--readout-train-scope site_ligand_metal_attention requires "
                "--site-ligand-metal-attention"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.site_ligand_metal_attention.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "site_donor_relation_attention":
        if readout.site_donor_relation_attention is None:
            raise ValueError(
                "--readout-train-scope site_donor_relation_attention "
                "requires --site-donor-relation-attention"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.site_donor_relation_attention.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "multi_level_orb_attention":
        if readout.multi_level_orb_feature_attention is None:
            raise ValueError(
                "--readout-train-scope multi_level_orb_attention requires "
                "--multi-level-orb-attention"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.multi_level_orb_feature_attention.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "dual_backbone_feature_fusion":
        if readout.dual_backbone_feature_fusion is None:
            raise ValueError(
                "--readout-train-scope dual_backbone_feature_fusion "
                "requires --dual-backbone-feature-fusion"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.dual_backbone_feature_fusion.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "dual_backbone_feature_role_gate":
        adapter = readout.dual_backbone_feature_fusion
        if (
            adapter is None
            or adapter.role_scale_delta is None
        ):
            raise ValueError(
                "--readout-train-scope dual_backbone_feature_role_gate "
                "requires --dual-backbone-feature-fusion and "
                "--dual-backbone-feature-role-gate"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        adapter.role_scale_delta.requires_grad = True
    elif args.readout_train_scope == "cip_orientation_embedding":
        if readout.cip_orientation_embedding is None:
            raise ValueError(
                "--readout-train-scope cip_orientation_embedding "
                "requires --cip-orientation-embedding"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.cip_orientation_embedding.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "smiles_sequence_adapter":
        if readout.smiles_sequence_adapter is None:
            raise ValueError(
                "--readout-train-scope smiles_sequence_adapter "
                "requires --smiles-sequence-adapter"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.smiles_sequence_adapter.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "regression_head":
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.raw_head.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "regression_output":
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.raw_head[-1].parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "aho_consistency_adapter":
        if readout.aho_consistency_adapter is None:
            raise ValueError(
                "--readout-train-scope aho_consistency_adapter requires "
                "--aho-consistency-auxiliary"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.aho_consistency_adapter.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "ddg_sign_magnitude_adapter":
        if readout.ddg_sign_magnitude_adapter is None:
            raise ValueError(
                "--readout-train-scope ddg_sign_magnitude_adapter requires "
                "--ddg-sign-magnitude"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.ddg_sign_magnitude_adapter.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "ddg_residual_calibrator":
        if readout.ddg_residual_calibrator is None:
            raise ValueError(
                "--readout-train-scope ddg_residual_calibrator requires "
                "--ddg-residual-calibrator"
            )
        for parameter in readout.parameters():
            parameter.requires_grad = False
        for parameter in readout.ddg_residual_calibrator.parameters():
            parameter.requires_grad = True
    elif args.readout_train_scope == "ddg_residual_nonlinear_adapter":
        configure_ddg_nonlinear_calibrator_trainability(readout)
    elif (
        args.readout_train_scope
        == "ddg_sign_magnitude_and_residual_calibrator"
    ):
        configure_ddg_joint_adapter_trainability(readout)
    elif args.readout_train_scope == "ddg_major_summary_adapter":
        configure_ddg_major_summary_trainability(readout)
    trainable_readout_names = [
        name
        for name, parameter in readout.named_parameters()
        if parameter.requires_grad
    ]
    trainable_readout = [
        parameter for parameter in readout.parameters() if parameter.requires_grad
    ]
    if not trainable_readout:
        raise RuntimeError("No trainable readout parameters were selected")
    trainable_orb = [parameter for parameter in orb_backbone.parameters() if parameter.requires_grad]
    trainable_readout_parameters = int(
        sum(parameter.numel() for parameter in trainable_readout)
    )
    trainable_orb_parameters = int(
        sum(parameter.numel() for parameter in trainable_orb)
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": trainable_readout,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "name": args.readout_train_scope,
            },
            {
                "params": trainable_orb,
                "lr": 0.0 if args.orb_warmup_epochs > 0 else args.orb_lr,
                "weight_decay": args.orb_weight_decay,
                "name": f"pretrained_{args.backbone_family}",
            },
        ],
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
    if task.task_type == "classification":
        if args.positive_class_weight_scale <= 0.0:
            raise ValueError("--positive-class-weight-scale must be positive")
        classification_training_target = raw_bundles["train"].y
        if final_retrain_train_plus_val:
            classification_training_target = torch.cat(
                [
                    classification_training_target,
                    raw_bundles["val"].y,
                ]
            )
        positives = float(classification_training_target.sum())
        negatives = float(
            len(classification_training_target) - positives
        )
        pos_weight = torch.tensor(
            (negatives / max(positives, 1.0))
            * args.positive_class_weight_scale,
            device=device,
        )
    else:
        pos_weight = None
    selection_metric = args.selection_metric
    if selection_metric == "auto":
        selection_metric = "accuracy" if task.task_type == "classification" else "mae"
    scheduler_metric = resolve_scheduler_metric(
        args.scheduler_metric,
        selection_metric=selection_metric,
        task_type=task.task_type,
    )

    best_score = float("inf")
    best_epoch = -1
    best_states = None
    stale = 0
    history = []
    epoch_ema_states = (
        clone_states(readout, orb_backbone)
        if epoch_ema_enabled
        else None
    )
    epoch_ema_updates = 0
    major_summary_classifier_warmup_epochs = (
        args.ddg_major_summary_classifier_warmup_epochs
        if readout.ddg_major_summary_adapter is not None
        else 0
    )
    smiles_sequence_auxiliary_warmup_epochs = (
        args.smiles_sequence_auxiliary_warmup_epochs
        if (
            readout.smiles_sequence_adapter is not None
            and readout.smiles_sequence_adapter.auxiliary_head is not None
        )
        else 0
    )
    if resume_info is not None or init_info is not None:
        initial_val_output = predict(
            readout=readout,
            orb_backbone=orb_backbone,
            adapter=adapter,
            payloads=payloads,
            token_store=store,
            secondary_token_store=secondary_store,
            backbone_feature_source=args.backbone_feature_source,
            loader=loaders["val"],
            device=device,
            conformer_feature_mode=args.conformer_feature_mode,
            use_amp=use_amp,
        )
        initial_val_metrics, _ = task_metrics(
            true_values["val"],
            initial_val_output["prediction"],
            task_type=task.task_type,
            threshold=args.classification_threshold,
            target_mean=target_mean,
            target_std=target_std,
        )
        initial_test_selection_metrics = {}
        initial_selection_metrics = initial_val_metrics
        if checkpoint_selection_split == "test":
            initial_test_output = predict(
                readout=readout,
                orb_backbone=orb_backbone,
                adapter=adapter,
                payloads=payloads,
                token_store=store,
                secondary_token_store=secondary_store,
                backbone_feature_source=args.backbone_feature_source,
                loader=loaders["test"],
                device=device,
                conformer_feature_mode=args.conformer_feature_mode,
                use_amp=use_amp,
            )
            initial_test_selection_metrics, _ = task_metrics(
                true_values["test"],
                initial_test_output["prediction"],
                task_type=task.task_type,
                threshold=args.classification_threshold,
                target_mean=target_mean,
                target_std=target_std,
            )
            initial_selection_metrics = initial_test_selection_metrics
        initial_auxiliary_metrics = {}
        if readout.aho_consistency_adapter is not None:
            initial_auxiliary_metrics = aho_consistency_metrics(
                mode=readout.aho_consistency_mode,
                y_true=auxiliary_true_values["val"],
                model_output=initial_val_output["auxiliary_prediction"],
                target_mean=auxiliary_target_mean,
                target_std=auxiliary_target_std,
            )
        elif readout.ddg_major_summary_adapter is not None:
            initial_auxiliary_metrics = aho_consistency_metrics(
                mode="ddg_with_major",
                y_true=auxiliary_true_values["val"],
                model_output=initial_val_output[
                    "ddg_major_summary_logit"
                ],
                target_mean=0.0,
                target_std=1.0,
            )
        initial_smiles_sequence_auxiliary_metrics = {}
        if (
            readout.smiles_sequence_adapter is not None
            and readout.smiles_sequence_adapter.auxiliary_head is not None
        ):
            (
                initial_smiles_sequence_auxiliary_metrics,
                _,
            ) = task_metrics(
                true_values["val"],
                initial_val_output[
                    "smiles_sequence_auxiliary_prediction"
                ],
                task_type=task.task_type,
                threshold=args.classification_threshold,
                target_mean=target_mean,
                target_std=target_std,
            )
        initial_sign_magnitude_metrics = {}
        if readout.ddg_sign_magnitude_adapter is not None:
            initial_sign_magnitude_metrics = ddg_sign_magnitude_metrics(
                y_true=true_values["val"],
                model_output=initial_val_output,
                target_mean=target_mean,
                target_std=target_std,
            )
        best_score = validation_selection_score(
            initial_selection_metrics,
            selection_metric,
            major_accuracy_gate=args.major_accuracy_gate,
            major_auroc_gate=args.major_auroc_gate,
            ddg_r2_gate=args.ddg_r2_gate,
            ddg_mae_gate=args.ddg_mae_gate,
        )
        initial_validation_selection_score = validation_selection_score(
            initial_val_metrics,
            selection_metric,
            major_accuracy_gate=args.major_accuracy_gate,
            major_auroc_gate=args.major_auroc_gate,
            ddg_r2_gate=args.ddg_r2_gate,
            ddg_mae_gate=args.ddg_mae_gate,
        )
        best_states = clone_states(readout, orb_backbone)
        history.append(
            {
                "epoch": -1,
                "train_loss": float("nan"),
                "train_auxiliary_loss": float("nan"),
                "train_ddg_major_summary_loss": float("nan"),
                "train_smiles_sequence_auxiliary_loss": float("nan"),
                "train_ddg_sign_loss": float("nan"),
                "train_ddg_magnitude_loss": float("nan"),
                "train_ddg_sign_margin_loss": float("nan"),
                "train_classification_pairwise_ranking_loss": float("nan"),
                "train_reactant_conformer_consistency_loss": float("nan"),
                "train_regression_group_mean_loss": float("nan"),
                "train_confidence": float("nan"),
                "ddg_major_summary_residual_active": (
                    major_summary_classifier_warmup_epochs == 0
                ),
                "smiles_sequence_residual_active": (
                    smiles_sequence_auxiliary_warmup_epochs == 0
                ),
                "epoch_ema_enabled": epoch_ema_enabled,
                "epoch_ema_decay": float(args.epoch_ema_decay),
                "epoch_ema_updates": 0,
                "readout_lr": args.lr,
                "orb_lr": 0.0 if args.orb_warmup_epochs > 0 else args.orb_lr,
                "checkpoint_selection_score": best_score,
                "validation_selection_score": (
                    initial_validation_selection_score
                ),
                "validation_scheduler_score": validation_selection_score(
                    initial_val_metrics,
                    scheduler_metric,
                    major_accuracy_gate=args.major_accuracy_gate,
                    major_auroc_gate=args.major_auroc_gate,
                    ddg_r2_gate=args.ddg_r2_gate,
                    ddg_mae_gate=args.ddg_mae_gate,
                ),
                **{
                    f"validation_{key}": value
                    for key, value in initial_val_metrics.items()
                },
                **{
                    f"test_selection_{key}": value
                    for key, value in initial_test_selection_metrics.items()
                },
                **{
                    f"validation_auxiliary_{key}": value
                    for key, value in initial_auxiliary_metrics.items()
                },
                **{
                    f"validation_smiles_sequence_auxiliary_{key}": value
                    for key, value in (
                        initial_smiles_sequence_auxiliary_metrics.items()
                    )
                },
                **{
                    f"validation_ddg_{key}": value
                    for key, value in initial_sign_magnitude_metrics.items()
                },
            }
        )
        print(
            "[initial-validation] "
            + " ".join(
                f"{key}={value:.6f}"
                for key, value in initial_val_metrics.items()
                if key not in {"n", "threshold"}
            ),
            flush=True,
        )
        if initial_auxiliary_metrics:
            print(
                "[resume-auxiliary-validation] "
                + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in initial_auxiliary_metrics.items()
                    if key not in {"n", "threshold"}
                ),
                flush=True,
            )
        if initial_smiles_sequence_auxiliary_metrics:
            print(
                "[resume-smiles-sequence-auxiliary-validation] "
                + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in (
                        initial_smiles_sequence_auxiliary_metrics.items()
                    )
                    if key not in {"n", "threshold"}
                ),
                flush=True,
            )
        if initial_test_selection_metrics:
            print(
                "[resume-test-selection] "
                + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in initial_test_selection_metrics.items()
                    if key not in {"n", "threshold"}
                ),
                flush=True,
            )
        if initial_sign_magnitude_metrics:
            print(
                "[resume-sign-magnitude-validation] "
                + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in initial_sign_magnitude_metrics.items()
                    if key not in {"sign_n", "sign_threshold", "magnitude_n", "structured_n"}
                ),
                flush=True,
            )
    for epoch in range(args.epochs):
        major_summary_residual_active = (
            readout.ddg_major_summary_adapter is None
            or epoch >= major_summary_classifier_warmup_epochs
        )
        if readout.ddg_major_summary_adapter is not None:
            configure_ddg_major_summary_training_phase(
                readout,
                residual_active=major_summary_residual_active,
            )
            if (
                major_summary_classifier_warmup_epochs > 0
                and epoch == major_summary_classifier_warmup_epochs
            ):
                if args.ddg_major_summary_post_warmup_lr > 0.0:
                    optimizer.param_groups[0]["lr"] = (
                        args.ddg_major_summary_post_warmup_lr
                    )
                print(
                    "[ddg-major-summary-residual-enable] "
                    f"epoch={epoch + 1} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
        smiles_sequence_residual_active = (
            readout.smiles_sequence_adapter is None
            or readout.smiles_sequence_adapter.auxiliary_head is None
            or epoch >= smiles_sequence_auxiliary_warmup_epochs
        )
        if (
            readout.smiles_sequence_adapter is not None
            and readout.smiles_sequence_adapter.auxiliary_head is not None
        ):
            configure_smiles_sequence_training_phase(
                readout,
                residual_active=smiles_sequence_residual_active,
            )
            if (
                smiles_sequence_auxiliary_warmup_epochs > 0
                and epoch == smiles_sequence_auxiliary_warmup_epochs
            ):
                if (
                    args.smiles_sequence_auxiliary_post_warmup_lr
                    > 0.0
                ):
                    optimizer.param_groups[0]["lr"] = (
                        args.smiles_sequence_auxiliary_post_warmup_lr
                    )
                print(
                    "[smiles-sequence-residual-enable] "
                    f"epoch={epoch + 1} "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )
        if epoch == args.orb_warmup_epochs:
            optimizer.param_groups[1]["lr"] = args.orb_lr
            print(
                f"[orb-unfreeze] epoch={epoch + 1} lr={args.orb_lr:.2e} "
                f"scope={args.orb_train_scope} "
                f"stacks={args.unfreeze_orb_stacks}",
                flush=True,
            )
        train_stats = train_epoch(
            readout=readout,
            orb_backbone=orb_backbone,
            adapter=adapter,
            payloads=payloads,
            token_store=store,
            secondary_token_store=secondary_store,
            backbone_feature_source=args.backbone_feature_source,
            readout_train_scope=args.readout_train_scope,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            task_type=task.task_type,
            loss_name=args.loss,
            huber_delta=args.huber_delta,
            pos_weight=pos_weight,
            label_smoothing=args.label_smoothing,
            classification_focal_gamma=(
                args.classification_focal_gamma
            ),
            classification_pairwise_ranking_weight=(
                args.classification_pairwise_ranking_weight
            ),
            classification_pairwise_ranking_temperature=(
                args.classification_pairwise_ranking_temperature
            ),
            gate_regularization=args.gate_regularization,
            readout_gradient_clip_norm=args.readout_gradient_clip_norm,
            orb_gradient_clip_norm=args.orb_gradient_clip_norm,
            conformer_feature_mode=args.conformer_feature_mode,
            reactant_conformer_view_probability=(
                args.reactant_conformer_view_probability
            ),
            reactant_conformer_consistency_weight=(
                args.reactant_conformer_consistency_weight
            ),
            reactant_conformer_consistency_huber_delta=(
                args.reactant_conformer_consistency_huber_delta
            ),
            orb_module_mode=args.orb_module_mode,
            readout_module_mode=args.readout_module_mode,
            site_donor_relation_auxiliary_weight=(
                args.site_donor_relation_auxiliary_weight
            ),
            smiles_sequence_auxiliary_weight=(
                args.smiles_sequence_auxiliary_weight
            ),
            aho_consistency_weight=args.aho_consistency_weight,
            aho_consistency_huber_delta=(
                args.aho_consistency_huber_delta
            ),
            ddg_major_summary_weight=(
                args.ddg_major_summary_weight
            ),
            ddg_sign_magnitude_weight=(
                args.ddg_sign_magnitude_weight
            ),
            ddg_sign_only_auxiliary_weight=(
                args.ddg_sign_only_auxiliary_weight
            ),
            ddg_sign_magnitude_huber_delta=(
                args.ddg_sign_magnitude_huber_delta
            ),
            ddg_sign_bce_magnitude_strength=(
                args.ddg_sign_bce_magnitude_strength
            ),
            ddg_sign_bce_magnitude_scale=(
                ddg_sign_bce_magnitude_scale
            ),
            ddg_output_sign_margin_weight=(
                args.ddg_output_sign_margin_weight
            ),
            ddg_output_sign_margin_temperature=(
                args.ddg_output_sign_margin_temperature
            ),
            ddg_output_sign_margin_zero_point=(
                ddg_output_sign_margin_zero_point
            ),
            ddg_output_sign_margin_dead_zone=(
                ddg_output_sign_margin_dead_zone
            ),
            ddg_output_sign_margin_magnitude_scale=(
                ddg_output_sign_margin_magnitude_scale
            ),
            regression_group_mean_weight=(
                args.regression_group_mean_weight
            ),
            training_role_feature_dropout_probability=(
                args.training_role_feature_dropout_probability
            ),
            training_role_feature_dropout_id=(
                args.training_role_feature_dropout_id
            ),
            use_amp=use_amp,
        )
        live_states = None
        if epoch_ema_enabled:
            if epoch_ema_states is None:
                raise RuntimeError("Epoch EMA state was not initialized")
            live_states = clone_states(readout, orb_backbone)
            update_epoch_ema_state(
                epoch_ema_states[0],
                live_states[0],
                decay=args.epoch_ema_decay,
            )
            update_epoch_ema_state(
                epoch_ema_states[1],
                live_states[1],
                decay=args.epoch_ema_decay,
            )
            epoch_ema_updates += 1
            readout.load_state_dict(epoch_ema_states[0])
            orb_backbone.load_state_dict(epoch_ema_states[1])
        val_output = predict(
            readout=readout,
            orb_backbone=orb_backbone,
            adapter=adapter,
            payloads=payloads,
            token_store=store,
            secondary_token_store=secondary_store,
            backbone_feature_source=args.backbone_feature_source,
            loader=loaders["val"],
            device=device,
            conformer_feature_mode=args.conformer_feature_mode,
            use_amp=use_amp,
        )
        val_metrics, _ = task_metrics(
            true_values["val"],
            val_output["prediction"],
            task_type=task.task_type,
            threshold=args.classification_threshold,
            target_mean=target_mean,
            target_std=target_std,
        )
        epoch_test_selection_metrics = {}
        checkpoint_selection_metrics = val_metrics
        if checkpoint_selection_split == "test":
            epoch_test_output = predict(
                readout=readout,
                orb_backbone=orb_backbone,
                adapter=adapter,
                payloads=payloads,
                token_store=store,
                secondary_token_store=secondary_store,
                backbone_feature_source=args.backbone_feature_source,
                loader=loaders["test"],
                device=device,
                conformer_feature_mode=args.conformer_feature_mode,
                use_amp=use_amp,
            )
            epoch_test_selection_metrics, _ = task_metrics(
                true_values["test"],
                epoch_test_output["prediction"],
                task_type=task.task_type,
                threshold=args.classification_threshold,
                target_mean=target_mean,
                target_std=target_std,
            )
            checkpoint_selection_metrics = (
                epoch_test_selection_metrics
            )
        val_auxiliary_metrics = {}
        if readout.aho_consistency_adapter is not None:
            val_auxiliary_metrics = aho_consistency_metrics(
                mode=readout.aho_consistency_mode,
                y_true=auxiliary_true_values["val"],
                model_output=val_output["auxiliary_prediction"],
                target_mean=auxiliary_target_mean,
                target_std=auxiliary_target_std,
            )
        elif readout.ddg_major_summary_adapter is not None:
            val_auxiliary_metrics = aho_consistency_metrics(
                mode="ddg_with_major",
                y_true=auxiliary_true_values["val"],
                model_output=val_output[
                    "ddg_major_summary_logit"
                ],
                target_mean=0.0,
                target_std=1.0,
            )
        val_smiles_sequence_auxiliary_metrics = {}
        if (
            readout.smiles_sequence_adapter is not None
            and readout.smiles_sequence_adapter.auxiliary_head is not None
        ):
            (
                val_smiles_sequence_auxiliary_metrics,
                _,
            ) = task_metrics(
                true_values["val"],
                val_output["smiles_sequence_auxiliary_prediction"],
                task_type=task.task_type,
                threshold=args.classification_threshold,
                target_mean=target_mean,
                target_std=target_std,
            )
        val_sign_magnitude_metrics = {}
        if readout.ddg_sign_magnitude_adapter is not None:
            val_sign_magnitude_metrics = ddg_sign_magnitude_metrics(
                y_true=true_values["val"],
                model_output=val_output,
                target_mean=target_mean,
                target_std=target_std,
            )
        score = validation_selection_score(
            checkpoint_selection_metrics,
            selection_metric,
            major_accuracy_gate=args.major_accuracy_gate,
            major_auroc_gate=args.major_auroc_gate,
            ddg_r2_gate=args.ddg_r2_gate,
            ddg_mae_gate=args.ddg_mae_gate,
        )
        validation_score = validation_selection_score(
            val_metrics,
            selection_metric,
            major_accuracy_gate=args.major_accuracy_gate,
            major_auroc_gate=args.major_auroc_gate,
            ddg_r2_gate=args.ddg_r2_gate,
            ddg_mae_gate=args.ddg_mae_gate,
        )
        scheduler_score = validation_selection_score(
            val_metrics,
            scheduler_metric,
            major_accuracy_gate=args.major_accuracy_gate,
            major_auroc_gate=args.major_auroc_gate,
            ddg_r2_gate=args.ddg_r2_gate,
            ddg_mae_gate=args.ddg_mae_gate,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_auxiliary_loss": train_stats["auxiliary_loss"],
                "train_ddg_major_summary_loss": train_stats[
                    "ddg_major_summary_loss"
                ],
                "train_smiles_sequence_auxiliary_loss": train_stats[
                    "smiles_sequence_auxiliary_loss"
                ],
                "train_site_donor_relation_auxiliary_loss": train_stats[
                    "site_donor_relation_auxiliary_loss"
                ],
                "train_ddg_sign_loss": train_stats["ddg_sign_loss"],
                "train_ddg_magnitude_loss": train_stats[
                    "ddg_magnitude_loss"
                ],
                "train_ddg_sign_margin_loss": train_stats[
                    "ddg_sign_margin_loss"
                ],
                "train_classification_pairwise_ranking_loss": train_stats[
                    "classification_pairwise_ranking_loss"
                ],
                "train_reactant_conformer_consistency_loss": train_stats[
                    "reactant_conformer_consistency_loss"
                ],
                "train_regression_group_mean_loss": train_stats[
                    "regression_group_mean_loss"
                ],
                "train_confidence": train_stats["confidence"],
                "ddg_major_summary_residual_active": (
                    major_summary_residual_active
                ),
                "smiles_sequence_residual_active": (
                    smiles_sequence_residual_active
                ),
                "epoch_ema_enabled": epoch_ema_enabled,
                "epoch_ema_decay": float(args.epoch_ema_decay),
                "epoch_ema_updates": epoch_ema_updates,
                "readout_lr": optimizer.param_groups[0]["lr"],
                "orb_lr": optimizer.param_groups[1]["lr"],
                "checkpoint_selection_score": score,
                "validation_selection_score": validation_score,
                "validation_scheduler_score": scheduler_score,
                **{f"validation_{key}": value for key, value in val_metrics.items()},
                **{
                    f"test_selection_{key}": value
                    for key, value in epoch_test_selection_metrics.items()
                },
                **{
                    f"validation_auxiliary_{key}": value
                    for key, value in val_auxiliary_metrics.items()
                },
                **{
                    f"validation_smiles_sequence_auxiliary_{key}": value
                    for key, value in (
                        val_smiles_sequence_auxiliary_metrics.items()
                    )
                },
                **{
                    f"validation_ddg_{key}": value
                    for key, value in val_sign_magnitude_metrics.items()
                },
            }
        )
        checkpoint_eligible = epoch >= max(
            args.orb_warmup_epochs,
            major_summary_classifier_warmup_epochs,
            smiles_sequence_auxiliary_warmup_epochs,
        )
        if final_retrain_train_plus_val:
            best_score = score
            best_epoch = epoch
            best_states = clone_states(readout, orb_backbone)
            stale = 0
        else:
            if checkpoint_eligible and score < best_score - 1e-8:
                best_score = score
                best_epoch = epoch
                best_states = clone_states(readout, orb_backbone)
                stale = 0
            elif checkpoint_eligible:
                stale += 1
            if (
                major_summary_residual_active
                and smiles_sequence_residual_active
            ):
                scheduler.step(scheduler_score)
        if epoch_ema_enabled:
            if live_states is None:
                raise RuntimeError("Live state was not retained for EMA restore")
            readout.load_state_dict(live_states[0])
            orb_backbone.load_state_dict(live_states[1])
        print(
            f"[epoch] {epoch + 1}/{args.epochs} loss={train_stats['loss']:.6f} "
            f"readout_lr={optimizer.param_groups[0]['lr']:.2e} "
            f"orb_lr={optimizer.param_groups[1]['lr']:.2e} "
            + " ".join(
                f"{key}={value:.6f}"
                for key, value in val_metrics.items()
                if key not in {"n", "threshold"}
            ),
            flush=True,
        )
        if val_auxiliary_metrics:
            print(
                "[epoch-auxiliary] "
                + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in val_auxiliary_metrics.items()
                    if key not in {"n", "threshold"}
                ),
                flush=True,
            )
        if val_smiles_sequence_auxiliary_metrics:
            print(
                "[epoch-smiles-sequence-auxiliary] "
                + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in (
                        val_smiles_sequence_auxiliary_metrics.items()
                    )
                    if key not in {"n", "threshold"}
                ),
                flush=True,
            )
        if epoch_test_selection_metrics:
            print(
                "[epoch-test-selection] "
                + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in epoch_test_selection_metrics.items()
                    if key not in {"n", "threshold"}
                ),
                flush=True,
            )
        if val_sign_magnitude_metrics:
            print(
                "[epoch-sign-magnitude] "
                + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in val_sign_magnitude_metrics.items()
                    if key not in {"sign_n", "sign_threshold", "magnitude_n", "structured_n"}
                ),
                flush=True,
            )
        if (
            not final_retrain_train_plus_val
            and stale >= args.patience
        ):
            break

    if best_states is None:
        raise RuntimeError("No checkpoint selected")
    readout.load_state_dict(best_states[0])
    orb_backbone.load_state_dict(best_states[1])
    checkpoint_selection_mode = (
        "fixed_last_epoch_train_plus_val"
        if final_retrain_train_plus_val
        else "metric_best"
    )
    if epoch_ema_enabled:
        checkpoint_selection_mode += "_epoch_ema"

    final_metrics = {}
    final_auxiliary_metrics = {}
    final_smiles_sequence_auxiliary_metrics = {}
    final_sign_magnitude_metrics = {}
    final_outputs = {}
    evaluation_parts = (
        parts
        if args.export_fusion_features
        else tuple(part for part in parts if part != "train")
    )
    for part in evaluation_parts:
        evaluation_loader = loaders[part]
        if args.export_fusion_features and part == "train":
            evaluation_loader = make_loader(
                raw_bundles[part],
                density[part],
                targets[part],
                global_components[part],
                sample_prior_by_part[part],
                batch_size=args.batch_size,
                shuffle=False,
                auxiliary_target=(
                    auxiliary_targets[part]
                    if auxiliary_targets is not None
                    else None
                ),
                cip_orientation_ids=(
                    cip_orientation_ids[part]
                    if cip_orientation_embedding_enabled
                    else None
                ),
                smiles_sequence_token_ids=(
                    smiles_sequence_tokens[part][0]
                    if smiles_sequence_adapter_enabled
                    else None
                ),
                smiles_sequence_token_mask=(
                    smiles_sequence_tokens[part][1]
                    if smiles_sequence_adapter_enabled
                    else None
                ),
            )
        output = predict(
            readout=readout,
            orb_backbone=orb_backbone,
            adapter=adapter,
            payloads=payloads,
            token_store=store,
            secondary_token_store=secondary_store,
            backbone_feature_source=args.backbone_feature_source,
            loader=evaluation_loader,
            device=device,
            conformer_feature_mode=args.conformer_feature_mode,
            use_amp=use_amp,
            collect_fusion=args.export_fusion_features,
        )
        metrics, predictions = task_metrics(
            true_values[part],
            output["prediction"],
            task_type=task.task_type,
            threshold=args.classification_threshold,
            target_mean=target_mean,
            target_std=target_std,
        )
        final_metrics[part] = metrics
        if readout.aho_consistency_adapter is not None:
            final_auxiliary_metrics[part] = aho_consistency_metrics(
                mode=readout.aho_consistency_mode,
                y_true=auxiliary_true_values[part],
                model_output=output["auxiliary_prediction"],
                target_mean=auxiliary_target_mean,
                target_std=auxiliary_target_std,
            )
        elif readout.ddg_major_summary_adapter is not None:
            final_auxiliary_metrics[part] = aho_consistency_metrics(
                mode="ddg_with_major",
                y_true=auxiliary_true_values[part],
                model_output=output["ddg_major_summary_logit"],
                target_mean=0.0,
                target_std=1.0,
            )
        if (
            readout.smiles_sequence_adapter is not None
            and readout.smiles_sequence_adapter.auxiliary_head is not None
        ):
            (
                final_smiles_sequence_auxiliary_metrics[part],
                _,
            ) = task_metrics(
                true_values[part],
                output["smiles_sequence_auxiliary_prediction"],
                task_type=task.task_type,
                threshold=args.classification_threshold,
                target_mean=target_mean,
                target_std=target_std,
            )
        if readout.ddg_sign_magnitude_adapter is not None:
            final_sign_magnitude_metrics[part] = (
                ddg_sign_magnitude_metrics(
                    y_true=true_values[part],
                    model_output=output,
                    target_mean=target_mean,
                    target_std=target_std,
                )
            )
        final_outputs[part] = output
        prediction_columns = {
            "row_index": raw_bundles[part].row_index,
            "y_true": true_values[part],
            "y_pred": predictions,
            "raw_model_output": output["raw_prediction"],
            "confidence": output["confidence"],
            "ood_excess": output["ood_excess"],
        }
        if readout.aho_consistency_adapter is not None:
            prediction_columns.update(
                {
                    "auxiliary_y_true": auxiliary_true_values[part],
                    "auxiliary_model_output": output[
                        "auxiliary_prediction"
                    ],
                    "consistency_residual": output[
                        "consistency_residual"
                    ],
                }
            )
        if readout.site_donor_relation_attention is not None:
            prediction_columns.update(
                {
                    "site_donor_relation_logit": output[
                        "site_donor_relation_logit"
                    ],
                    "site_donor_relation_probability": 1.0 / (
                        1.0
                        + np.exp(
                            -output["site_donor_relation_logit"]
                        )
                    ),
                    "site_donor_relation_residual": output[
                        "site_donor_relation_residual"
                    ],
                    "site_donor_relation_coupling": output[
                        "site_donor_relation_coupling"
                    ],
                }
            )
        if readout.reaction_pair_transformer is not None:
            prediction_columns[
                "reaction_pair_transformer_residual"
            ] = output["reaction_pair_transformer_residual"]
        if readout.smiles_sequence_adapter is not None:
            prediction_columns["smiles_sequence_residual"] = (
                output["smiles_sequence_residual"]
                * (target_std if task.task_type == "regression" else 1.0)
            )
            if readout.smiles_sequence_adapter.auxiliary_head is not None:
                auxiliary_model_output = output[
                    "smiles_sequence_auxiliary_prediction"
                ]
                prediction_columns[
                    "smiles_sequence_auxiliary_model_output"
                ] = auxiliary_model_output
                if task.task_type == "classification":
                    prediction_columns[
                        "smiles_sequence_auxiliary_probability"
                    ] = 1.0 / (1.0 + np.exp(-auxiliary_model_output))
                else:
                    prediction_columns[
                        "smiles_sequence_auxiliary_y_pred"
                    ] = auxiliary_model_output * target_std + target_mean
        if readout.ddg_sign_magnitude_adapter is not None:
            prediction_columns.update(
                {
                    "direct_raw_model_output": output[
                        "direct_raw_prediction"
                    ],
                    "ddg_sign_logit": output["ddg_sign_logit"],
                    "ddg_sign_probability": 1.0 / (
                        1.0 + np.exp(-output["ddg_sign_logit"])
                    ),
                    "ddg_magnitude_prediction": output[
                        "ddg_magnitude_prediction"
                    ] * target_std,
                    "ddg_structured_prediction": (
                        output["ddg_structured_prediction"] * target_std
                        + target_mean
                    ),
                    "ddg_structured_residual": output[
                        "ddg_structured_residual"
                    ] * target_std,
                }
            )
        if readout.ddg_residual_calibrator is not None:
            prediction_columns.update(
                {
                    "pre_calibration_y_pred": (
                        output["pre_calibration_prediction"] * target_std
                        + target_mean
                    ),
                    "ddg_calibration_residual": output[
                        "ddg_calibration_residual"
                    ] * target_std,
                }
            )
        if readout.ddg_major_summary_adapter is not None:
            prediction_columns.update(
                {
                    "ddg_major_summary_y_true": (
                        auxiliary_true_values[part]
                    ),
                    "ddg_major_summary_logit": output[
                        "ddg_major_summary_logit"
                    ],
                    "ddg_major_summary_probability": 1.0 / (
                        1.0
                        + np.exp(
                            -output["ddg_major_summary_logit"]
                        )
                    ),
                    "ddg_major_summary_disagreement": output[
                        "ddg_major_summary_disagreement"
                    ],
                    "ddg_major_summary_residual": output[
                        "ddg_major_summary_residual"
                    ]
                    * target_std,
                }
            )
            view_names = (
                "fusion",
                "reactant",
                "solvent",
                "ligand",
                "metal",
            )
            for view_index, view_name in enumerate(view_names):
                prediction_columns[
                    f"ddg_major_summary_{view_name}_logit"
                ] = output["ddg_major_summary_view_logits"][
                    :,
                    view_index,
                ]
                prediction_columns[
                    f"ddg_major_summary_{view_name}_present"
                ] = output["ddg_major_summary_view_mask"][
                    :,
                    view_index,
                ]
        pd.DataFrame(prediction_columns).to_csv(
            out_dir / f"{part}_predictions.csv",
            index=False,
        )
        if args.export_fusion_features:
            np.savez_compressed(
                out_dir / f"{part}_fusion_features.npz",
                row_index=raw_bundles[part].row_index,
                y_true=true_values[part],
                fusion=output["fusion"].astype(np.float32, copy=False),
            )

    metrics_row = {
        "task": task.name,
        "task_type": task.task_type,
        "mapping_task": mapping_task,
        "effective_mapping_task": effective_mapping_task,
        "validation_mode": args.validation_mode,
        "group_val_column": args.group_val_column,
        "group_val_fraction": float(args.group_val_fraction),
        "group_val_strategy": args.group_val_strategy,
        "group_val_role": args.group_val_role,
        "group_val_fold_count": int(args.group_val_fold_count),
        "group_val_fold_index": int(args.group_val_fold_index),
        "model": (
            "unified_orra_orb_v2_finetune_single"
            if args.backbone_family == "orb"
            else "unified_orra_jmp_amc_single"
        ),
        "architecture": (
            "atom_self_attention+role_and_token_type_cross_attention+"
            "shared_set_self_attention+density_gate"
            if args.cross_attention_view == "roles_types"
            else "atom_self_attention+role_cross_attention+role_self_attention+density_gate"
        )
        + ("+grouped_mean_std_max" if args.grouped_statistics else "")
        + ("+cross_view_pair_attention" if args.cross_view_interactions else "")
        + (
            "+conformer_attention"
            if args.conformer_aggregation == "attention"
            else ""
        )
        + (
            "+reactant_conformer_consistency"
            if reactant_conformer_consistency_enabled
            else ""
        )
        + ("+directed_role_attention" if args.directed_role_attention else "")
        + (
            "+reaction_pair_transformer"
            if args.reaction_pair_transformer
            else ""
        )
        + (
            "+site_ligand_metal_attention"
            if args.site_ligand_metal_attention
            else ""
        )
        + (
            "+site_donor_relation_attention"
            if args.site_donor_relation_attention
            else ""
        )
        + (
            "+multi_level_orb_attention"
            if args.multi_level_orb_attention
            else ""
        )
        + (
            "+dual_backbone_feature_residual_fusion"
            if dual_backbone_feature_fusion_enabled
            else ""
        )
        + (
            "+dual_backbone_feature_role_gate"
            if args.dual_backbone_feature_role_gate
            else ""
        )
        + (
            "+cip_orientation_embedding"
            if cip_orientation_embedding_enabled
            else ""
        )
        + (
            "+byte_smiles_residual_transformer"
            if smiles_sequence_adapter_enabled
            else ""
        )
        + (
            "+sequence_auxiliary_warmup"
            if args.smiles_sequence_auxiliary
            else ""
        )
        + (
            "+aho_physical_consistency"
            if args.aho_consistency_auxiliary
            else ""
        )
        + (
            "+ddg_sign_magnitude"
            if args.ddg_sign_magnitude
            else ""
        )
        + (
            "+ddg_residual_calibrator"
            if args.ddg_residual_calibrator
            else ""
        )
        + (
            "+ddg_residual_nonlinear_adapter"
            if args.ddg_residual_calibrator_nonlinear_hidden_dim > 0
            else ""
        )
        + (
            "+ddg_major_role_summary"
            if args.ddg_major_summary
            else ""
        ),
        "site_ligand_metal_attention": args.site_ligand_metal_attention,
        "reaction_pair_transformer": (
            args.reaction_pair_transformer
        ),
        "reaction_pair_transformer_num_layers": (
            args.reaction_pair_transformer_num_layers
        ),
        "site_donor_relation_attention": (
            args.site_donor_relation_attention
        ),
        "site_donor_relation_num_layers": (
            args.site_donor_relation_num_layers
        ),
        "site_donor_relation_auxiliary_weight": (
            args.site_donor_relation_auxiliary_weight
        ),
        "multi_level_orb_attention": args.multi_level_orb_attention,
        "multi_level_orb_stack_indices": ",".join(
            str(index) for index in multi_level_orb_stack_indices
        ),
        "multi_level_orb_feature_levels": (
            len(multi_level_orb_stack_indices)
            if args.multi_level_orb_attention
            else 1
        ),
        "dual_backbone_feature_fusion": (
            dual_backbone_feature_fusion_enabled
        ),
        "dual_backbone_feature_fusion_hidden_dim": (
            args.dual_backbone_feature_fusion_hidden_dim
        ),
        "dual_backbone_feature_role_gate": (
            args.dual_backbone_feature_role_gate
        ),
        "cip_orientation_embedding": (
            cip_orientation_embedding_enabled
        ),
        "cip_orientation_id_schema": json.dumps(
            CIP_ORIENTATION_IDS,
            sort_keys=True,
        ),
        "cip_orientation_alignment": json.dumps(
            cip_orientation_alignment,
            sort_keys=True,
        ),
        "smiles_sequence_adapter": (
            smiles_sequence_adapter_enabled
        ),
        "smiles_sequence_vocab_size": (
            args.smiles_sequence_vocab_size
        ),
        "smiles_sequence_max_length": (
            args.smiles_sequence_max_length
        ),
        "smiles_sequence_hidden_dim": (
            args.smiles_sequence_hidden_dim
        ),
        "smiles_sequence_num_layers": (
            args.smiles_sequence_num_layers
        ),
        "smiles_sequence_num_heads": (
            args.smiles_sequence_num_heads
        ),
        "smiles_sequence_dropout": (
            args.smiles_sequence_dropout
        ),
        "smiles_sequence_residual_multiplier": (
            args.smiles_sequence_residual_multiplier
        ),
        "smiles_sequence_auxiliary": (
            args.smiles_sequence_auxiliary
        ),
        "smiles_sequence_auxiliary_weight": (
            args.smiles_sequence_auxiliary_weight
        ),
        "smiles_sequence_auxiliary_warmup_epochs": (
            args.smiles_sequence_auxiliary_warmup_epochs
        ),
        "smiles_sequence_auxiliary_post_warmup_lr": (
            args.smiles_sequence_auxiliary_post_warmup_lr
        ),
        "smiles_sequence_catalyst_h8_augmentation": (
            args.smiles_sequence_catalyst_h8_augmentation
        ),
        "smiles_sequence_catalyst_h8_target_mode": (
            args.smiles_sequence_catalyst_h8_target_mode
        ),
        "smiles_sequence_catalyst_h8_target_quantile": (
            args.smiles_sequence_catalyst_h8_target_quantile
        ),
        "smiles_sequence_catalyst_h8_target_mix": (
            args.smiles_sequence_catalyst_h8_target_mix
        ),
        "smiles_sequence_catalyst_h8_target_condition_columns": (
            ",".join(h8_target_condition_columns)
        ),
        "smiles_sequence_catalyst_h8_randomized_views": int(
            args.smiles_sequence_catalyst_h8_randomized_views
        ),
        "smiles_sequence_catalyst_h8_target_inventory": json.dumps(
            h8_target_inventory,
            sort_keys=True,
        ),
        "smiles_sequence_alignment": json.dumps(
            smiles_sequence_alignment,
            sort_keys=True,
        ),
        "secondary_atom_token_dir": (
            str(args.secondary_atom_token_dir)
            if args.secondary_atom_token_dir is not None
            else ""
        ),
        "primary_token_manifest_sha256": (
            primary_token_manifest_sha256
        ),
        "secondary_token_manifest_sha256": (
            secondary_token_manifest_sha256
        ),
        "aho_consistency_auxiliary": args.aho_consistency_auxiliary,
        "aho_consistency_mode": aho_consistency_mode,
        "aho_consistency_target_column": auxiliary_target_column,
        "aho_consistency_hidden_dim": args.aho_consistency_hidden_dim,
        "aho_consistency_weight": args.aho_consistency_weight,
        "aho_consistency_huber_delta": args.aho_consistency_huber_delta,
        "aho_consistency_target_mean": auxiliary_target_mean,
        "aho_consistency_target_std": auxiliary_target_std,
        "aho_consistency_auxiliary_zero_point": auxiliary_zero_point,
        "ddg_sign_magnitude": args.ddg_sign_magnitude,
        "ddg_sign_magnitude_hidden_dim": (
            args.ddg_sign_magnitude_hidden_dim
        ),
        "ddg_sign_magnitude_weight": args.ddg_sign_magnitude_weight,
        "ddg_sign_only_auxiliary_weight": (
            args.ddg_sign_only_auxiliary_weight
        ),
        "ddg_sign_magnitude_huber_delta": (
            args.ddg_sign_magnitude_huber_delta
        ),
        "ddg_sign_magnitude_target_zero_point": (
            ddg_sign_magnitude_zero_point
        ),
        "ddg_sign_magnitude_residual_multiplier": (
            args.ddg_sign_magnitude_residual_multiplier
        ),
        "ddg_residual_calibrator": args.ddg_residual_calibrator,
        "ddg_residual_calibrator_nonlinear_hidden_dim": (
            args.ddg_residual_calibrator_nonlinear_hidden_dim
        ),
        "ddg_major_summary": args.ddg_major_summary,
        "ddg_major_summary_classifier_hidden_dim": (
            args.ddg_major_summary_classifier_hidden_dim
        ),
        "ddg_major_summary_residual_hidden_dim": (
            args.ddg_major_summary_residual_hidden_dim
        ),
        "ddg_major_summary_weight": (
            args.ddg_major_summary_weight
        ),
        "ddg_major_summary_residual_multiplier": (
            args.ddg_major_summary_residual_multiplier
        ),
        "ddg_major_summary_classifier_warmup_epochs": (
            args.ddg_major_summary_classifier_warmup_epochs
        ),
        "ddg_major_summary_post_warmup_lr": (
            args.ddg_major_summary_post_warmup_lr
        ),
        "ddg_sign_bce_magnitude_strength": (
            args.ddg_sign_bce_magnitude_strength
        ),
        "ddg_sign_bce_magnitude_scale": (
            args.ddg_sign_bce_magnitude_scale
        ),
        "ddg_sign_bce_magnitude_scale_normalized": (
            ddg_sign_bce_magnitude_scale
        ),
        "ddg_output_sign_margin_weight": (
            args.ddg_output_sign_margin_weight
        ),
        "ddg_output_sign_margin_temperature": (
            args.ddg_output_sign_margin_temperature
        ),
        "ddg_output_sign_margin_dead_zone": (
            args.ddg_output_sign_margin_dead_zone
        ),
        "ddg_output_sign_margin_magnitude_scale": (
            args.ddg_output_sign_margin_magnitude_scale
        ),
        "backbone_family": args.backbone_family,
        "backbone_feature_source": args.backbone_feature_source,
        "training_backbone_forwards_per_batch": (
            1 if args.backbone_feature_source == "online" else 0
        ),
        "training_readout_forwards_per_batch": (
            2 if reactant_conformer_consistency_enabled else 1
        ),
        "inference_backbone_forwards_per_batch": (
            1 if args.backbone_feature_source == "online" else 0
        ),
        "inference_readout_forwards_per_batch": 1,
        "backbone_forwards_per_batch": (
            1 if args.backbone_feature_source == "online" else 0
        ),
        "orb_forwards_per_batch": (
            1 if args.backbone_feature_source == "online" else 0
        ),
        "single_checkpoint": True,
        "single_forward": True,
        "ensemble_size": 1,
        "backbone_trainable": bool(trainable_orb),
        "backbone_trainable_parameters": trainable_orb_parameters,
        "backbone_trainable_tensor_count": len(
            trainable_backbone_names
        ),
        "backbone_train_scope": args.orb_train_scope,
        "backbone_checkpoint": str(args.orb_checkpoint),
        "orb_trainable": bool(trainable_orb),
        "readout_trainable_parameters": trainable_readout_parameters,
        "readout_trainable_tensor_count": len(
            trainable_readout_names
        ),
        "readout_trainable_tensor_names": ",".join(
            trainable_readout_names
        ),
        "orb_trainable_parameters": trainable_orb_parameters,
        "orb_checkpoint": str(args.orb_checkpoint),
        "orb_module_mode": args.orb_module_mode,
        "readout_module_mode": args.readout_module_mode,
        "use_amp": use_amp,
        "conformer_feature_mode": args.conformer_feature_mode,
        "conformer_aggregation": args.conformer_aggregation,
        "conformer_attention_hidden_dim": args.conformer_attention_hidden_dim,
        "reactant_conformer_consistency_enabled": (
            reactant_conformer_consistency_enabled
        ),
        "reactant_conformer_view_probability": (
            args.reactant_conformer_view_probability
        ),
        "reactant_conformer_consistency_weight": (
            args.reactant_conformer_consistency_weight
        ),
        "reactant_conformer_consistency_huber_delta": (
            args.reactant_conformer_consistency_huber_delta
        ),
        "directed_role_attention": args.directed_role_attention,
        "directed_query_role_id": 0,
        "directed_context_role_ids": "2,8,9",
        "reaction_pair_transformer": (
            args.reaction_pair_transformer
        ),
        "reaction_pair_transformer_num_layers": (
            args.reaction_pair_transformer_num_layers
        ),
        "reaction_pair_role_ids": "0,2,8,9",
        "site_token_type_ids": "1,2,3",
        "ligand_token_type_ids": "6,7",
        "metal_token_type_ids": "9",
        "orb_train_scope": args.orb_train_scope,
        "readout_train_scope": args.readout_train_scope,
        "unfreeze_orb_stacks": args.unfreeze_orb_stacks,
        "readout_lr": args.lr,
        "orb_lr": args.orb_lr,
        "regression_group_mean_weight": float(
            args.regression_group_mean_weight
        ),
        "regression_group_mean_column": (
            args.regression_group_mean_column
        ),
        "training_role_feature_dropout_probability": float(
            args.training_role_feature_dropout_probability
        ),
        "training_role_feature_dropout_id": int(
            args.training_role_feature_dropout_id
        ),
        "orb_to_readout_lr_ratio": safe_learning_rate_ratio(
            args.orb_lr,
            args.lr,
        ),
        "density_train_mode": args.density_train_mode,
        "density_ratio_cap": float(args.density_ratio_cap),
        "unseen_density_floor": float(args.unseen_density_floor),
        "sample_prior_mode": args.sample_prior_mode,
        "prior_condition_columns": ",".join(prior_columns),
        "prior_quantile": float(args.prior_quantile),
        "ood_threshold": args.ood_threshold,
        "ood_scale_init": args.ood_scale_init,
        "ood_gate_mode": args.ood_gate_mode,
        "gate_residual_mode": args.gate_residual_mode,
        "ood_shift_scale": float(args.ood_shift_scale),
        "ood_base_shift": float(args.ood_base_shift),
        "residual_global_center_mix": float(args.residual_global_center_mix),
        "freeze_gate": args.freeze_gate,
        "label_smoothing": float(args.label_smoothing),
        "classification_focal_gamma": float(
            args.classification_focal_gamma
        ),
        "positive_class_weight_scale": float(args.positive_class_weight_scale),
        "classification_pairwise_ranking_weight": float(
            args.classification_pairwise_ranking_weight
        ),
        "classification_pairwise_ranking_temperature": float(
            args.classification_pairwise_ranking_temperature
        ),
        "adam_beta1": float(args.adam_beta1),
        "adam_beta2": float(args.adam_beta2),
        "adam_eps": float(args.adam_eps),
        "readout_gradient_clip_norm": float(
            args.readout_gradient_clip_norm
        ),
        "orb_gradient_clip_norm": float(args.orb_gradient_clip_norm),
        "checkpoint_selection_split": checkpoint_selection_split,
        "checkpoint_selection_mode": checkpoint_selection_mode,
        "epoch_ema_enabled": epoch_ema_enabled,
        "epoch_ema_decay": float(args.epoch_ema_decay),
        "epoch_ema_updates": epoch_ema_updates,
        "training_includes_validation": (
            final_retrain_train_plus_val
        ),
        "validation_is_training_data": (
            final_retrain_train_plus_val
        ),
        "observed_test_selection_acknowledged": bool(
            args.acknowledge_observed_test_selection
        ),
        "best_epoch": best_epoch,
        "n_train": effective_training_rows,
        "n_original_train": original_train_rows,
        "n_training_rows": effective_training_rows,
        "n_h8_augmented_training_rows": h8_augmented_training_rows,
        "n_h8_randomized_augmented_training_rows": (
            h8_randomized_augmented_training_rows
        ),
        "n_val": len(raw_bundles["val"].y),
        "n_test": len(frames["test"]),
        "test_evaluated": not args.skip_test_evaluation,
        "target_mean": target_mean,
        "target_std": target_std,
        **{f"validation_{key}": value for key, value in final_metrics["val"].items()},
        **{
            f"validation_auxiliary_{key}": value
            for key, value in final_auxiliary_metrics.get("val", {}).items()
        },
        **{
            f"validation_smiles_sequence_auxiliary_{key}": value
            for key, value in (
                final_smiles_sequence_auxiliary_metrics.get(
                    "val", {}
                ).items()
            )
        },
        **{
            f"validation_ddg_{key}": value
            for key, value in final_sign_magnitude_metrics.get(
                "val", {}
            ).items()
        },
        "init_readout_checkpoint": (
            str(args.init_readout_checkpoint)
            if args.init_readout_checkpoint is not None
            else ""
        ),
        "init_readout_mode": (
            args.init_readout_mode if init_info is not None else ""
        ),
        "resume_finetune_checkpoint": (
            str(args.resume_finetune_checkpoint)
            if resume_info is not None
            else ""
        ),
    }
    if "test" in final_metrics:
        metrics_row.update(final_metrics["test"])
        metrics_row.update(
            {
                f"test_smiles_sequence_auxiliary_{key}": value
                for key, value in (
                    final_smiles_sequence_auxiliary_metrics.get(
                        "test", {}
                    ).items()
                )
            }
        )
        metrics_row["test_mean_confidence"] = float(
            final_outputs["test"]["confidence"].mean()
        )
    pd.DataFrame([metrics_row]).to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    torch.save(
        {
            "readout_state_dict": readout.state_dict(),
            "orb_backbone_state_dict": orb_backbone.state_dict(),
            "orb_checkpoint": str(args.orb_checkpoint),
            "backbone_family": args.backbone_family,
            "backbone_feature_source": args.backbone_feature_source,
            "mapping_task": mapping_task,
            "effective_mapping_task": effective_mapping_task,
            "validation_mode": args.validation_mode,
            "group_val_column": args.group_val_column,
            "group_val_fraction": float(args.group_val_fraction),
            "group_val_strategy": args.group_val_strategy,
            "group_val_role": args.group_val_role,
            "group_val_fold_count": int(args.group_val_fold_count),
            "group_val_fold_index": int(args.group_val_fold_index),
            "orb_module_mode": args.orb_module_mode,
            "readout_module_mode": args.readout_module_mode,
            "use_amp": use_amp,
            "conformer_aggregation": args.conformer_aggregation,
            "reactant_conformer_consistency_enabled": (
                reactant_conformer_consistency_enabled
            ),
            "reactant_conformer_view_probability": (
                args.reactant_conformer_view_probability
            ),
            "reactant_conformer_consistency_weight": (
                args.reactant_conformer_consistency_weight
            ),
            "reactant_conformer_consistency_huber_delta": (
                args.reactant_conformer_consistency_huber_delta
            ),
            "directed_role_attention": args.directed_role_attention,
            "reaction_pair_transformer": (
                args.reaction_pair_transformer
            ),
            "reaction_pair_transformer_num_layers": (
                args.reaction_pair_transformer_num_layers
            ),
            "site_ligand_metal_attention": args.site_ligand_metal_attention,
            "site_donor_relation_attention": (
                args.site_donor_relation_attention
            ),
            "site_donor_relation_num_layers": (
                args.site_donor_relation_num_layers
            ),
            "site_donor_relation_auxiliary_weight": (
                args.site_donor_relation_auxiliary_weight
            ),
            "multi_level_orb_attention": args.multi_level_orb_attention,
            "multi_level_orb_stack_indices": (
                multi_level_orb_stack_indices
            ),
            "dual_backbone_feature_fusion": (
                dual_backbone_feature_fusion_enabled
            ),
            "dual_backbone_feature_fusion_hidden_dim": (
                args.dual_backbone_feature_fusion_hidden_dim
            ),
            "dual_backbone_feature_role_gate": (
                args.dual_backbone_feature_role_gate
            ),
            "cip_orientation_embedding": (
                cip_orientation_embedding_enabled
            ),
            "cip_orientation_id_schema": dict(
                CIP_ORIENTATION_IDS
            ),
            "cip_orientation_alignment": (
                cip_orientation_alignment
            ),
            "smiles_sequence_adapter": (
                smiles_sequence_adapter_enabled
            ),
            "smiles_sequence_vocab_size": (
                args.smiles_sequence_vocab_size
            ),
            "smiles_sequence_max_length": (
                args.smiles_sequence_max_length
            ),
            "smiles_sequence_hidden_dim": (
                args.smiles_sequence_hidden_dim
            ),
            "smiles_sequence_num_layers": (
                args.smiles_sequence_num_layers
            ),
            "smiles_sequence_num_heads": (
                args.smiles_sequence_num_heads
            ),
            "smiles_sequence_dropout": (
                args.smiles_sequence_dropout
            ),
            "smiles_sequence_residual_multiplier": (
                args.smiles_sequence_residual_multiplier
            ),
            "smiles_sequence_auxiliary": (
                args.smiles_sequence_auxiliary
            ),
            "smiles_sequence_auxiliary_weight": (
                args.smiles_sequence_auxiliary_weight
            ),
            "smiles_sequence_auxiliary_warmup_epochs": (
                args.smiles_sequence_auxiliary_warmup_epochs
            ),
            "smiles_sequence_auxiliary_post_warmup_lr": (
                args.smiles_sequence_auxiliary_post_warmup_lr
            ),
            "smiles_sequence_catalyst_h8_augmentation": (
                args.smiles_sequence_catalyst_h8_augmentation
            ),
            "smiles_sequence_catalyst_h8_target_mode": (
                args.smiles_sequence_catalyst_h8_target_mode
            ),
            "smiles_sequence_catalyst_h8_target_quantile": (
                args.smiles_sequence_catalyst_h8_target_quantile
            ),
            "smiles_sequence_catalyst_h8_target_mix": (
                args.smiles_sequence_catalyst_h8_target_mix
            ),
            "smiles_sequence_catalyst_h8_target_condition_columns": (
                h8_target_condition_columns
            ),
            "smiles_sequence_catalyst_h8_randomized_views": int(
                args.smiles_sequence_catalyst_h8_randomized_views
            ),
            "smiles_sequence_catalyst_h8_target_inventory": (
                h8_target_inventory
            ),
            "smiles_sequence_alignment": (
                smiles_sequence_alignment
            ),
            "secondary_atom_token_dir": (
                str(args.secondary_atom_token_dir)
                if args.secondary_atom_token_dir is not None
                else ""
            ),
            "primary_token_manifest_sha256": (
                primary_token_manifest_sha256
            ),
            "secondary_token_manifest_sha256": (
                secondary_token_manifest_sha256
            ),
            "dual_token_cache_alignment": (
                dual_token_cache_alignment
            ),
            "aho_consistency_auxiliary": args.aho_consistency_auxiliary,
            "aho_consistency_mode": aho_consistency_mode,
            "aho_consistency_target_column": auxiliary_target_column,
            "aho_consistency_hidden_dim": args.aho_consistency_hidden_dim,
            "aho_consistency_weight": args.aho_consistency_weight,
            "aho_consistency_huber_delta": (
                args.aho_consistency_huber_delta
            ),
            "aho_consistency_target_mean": auxiliary_target_mean,
            "aho_consistency_target_std": auxiliary_target_std,
            "aho_consistency_auxiliary_zero_point": auxiliary_zero_point,
            "ddg_sign_magnitude": args.ddg_sign_magnitude,
            "ddg_sign_magnitude_hidden_dim": (
                args.ddg_sign_magnitude_hidden_dim
            ),
            "ddg_sign_magnitude_weight": (
                args.ddg_sign_magnitude_weight
            ),
            "ddg_sign_only_auxiliary_weight": (
                args.ddg_sign_only_auxiliary_weight
            ),
            "ddg_sign_magnitude_huber_delta": (
                args.ddg_sign_magnitude_huber_delta
            ),
            "ddg_sign_magnitude_target_zero_point": (
                ddg_sign_magnitude_zero_point
            ),
            "ddg_sign_magnitude_residual_multiplier": (
                args.ddg_sign_magnitude_residual_multiplier
            ),
            "ddg_residual_calibrator": args.ddg_residual_calibrator,
            "ddg_residual_calibrator_nonlinear_hidden_dim": (
                args.ddg_residual_calibrator_nonlinear_hidden_dim
            ),
            "ddg_major_summary": args.ddg_major_summary,
            "ddg_major_summary_classifier_hidden_dim": (
                args.ddg_major_summary_classifier_hidden_dim
            ),
            "ddg_major_summary_residual_hidden_dim": (
                args.ddg_major_summary_residual_hidden_dim
            ),
            "ddg_major_summary_weight": (
                args.ddg_major_summary_weight
            ),
            "ddg_major_summary_residual_multiplier": (
                args.ddg_major_summary_residual_multiplier
            ),
            "ddg_major_summary_classifier_warmup_epochs": (
                args.ddg_major_summary_classifier_warmup_epochs
            ),
            "ddg_major_summary_post_warmup_lr": (
                args.ddg_major_summary_post_warmup_lr
            ),
            "ddg_sign_bce_magnitude_strength": (
                args.ddg_sign_bce_magnitude_strength
            ),
            "ddg_sign_bce_magnitude_scale": (
                args.ddg_sign_bce_magnitude_scale
            ),
            "ddg_sign_bce_magnitude_scale_normalized": (
                ddg_sign_bce_magnitude_scale
            ),
            "ddg_output_sign_margin_weight": (
                args.ddg_output_sign_margin_weight
            ),
            "ddg_output_sign_margin_temperature": (
                args.ddg_output_sign_margin_temperature
            ),
            "ddg_output_sign_margin_dead_zone": (
                args.ddg_output_sign_margin_dead_zone
            ),
            "ddg_output_sign_margin_magnitude_scale": (
                args.ddg_output_sign_margin_magnitude_scale
            ),
            "ddg_output_sign_margin_zero_point": (
                ddg_output_sign_margin_zero_point
            ),
            "training_backbone_forwards_per_batch": (
                1 if args.backbone_feature_source == "online" else 0
            ),
            "training_readout_forwards_per_batch": (
                2 if reactant_conformer_consistency_enabled else 1
            ),
            "inference_backbone_forwards_per_batch": (
                1 if args.backbone_feature_source == "online" else 0
            ),
            "inference_readout_forwards_per_batch": 1,
            "backbone_forwards_per_batch": (
                1 if args.backbone_feature_source == "online" else 0
            ),
            "orb_forwards_per_batch": (
                1 if args.backbone_feature_source == "online" else 0
            ),
            "orb_train_scope": args.orb_train_scope,
            "backbone_train_scope": args.orb_train_scope,
            "readout_train_scope": args.readout_train_scope,
            "readout_trainable_parameters": trainable_readout_parameters,
            "readout_trainable_tensor_count": len(
                trainable_readout_names
            ),
            "readout_trainable_tensor_names": (
                trainable_readout_names
            ),
            "orb_trainable_parameters": trainable_orb_parameters,
            "backbone_trainable_parameters": trainable_orb_parameters,
            "backbone_trainable_tensor_count": len(
                trainable_backbone_names
            ),
            "backbone_trainable_tensor_names": trainable_backbone_names,
            "unfreeze_orb_stacks": args.unfreeze_orb_stacks,
            "conformer_feature_mode": args.conformer_feature_mode,
            "target_mean": target_mean,
            "target_std": target_std,
            "sample_prior_mode": args.sample_prior_mode,
            "prior_condition_columns": prior_columns,
            "prior_quantile": float(args.prior_quantile),
            "checkpoint_selection_split": checkpoint_selection_split,
            "checkpoint_selection_mode": checkpoint_selection_mode,
            "epoch_ema_enabled": epoch_ema_enabled,
            "epoch_ema_decay": float(args.epoch_ema_decay),
            "epoch_ema_updates": epoch_ema_updates,
            "training_includes_validation": (
                final_retrain_train_plus_val
            ),
            "validation_is_training_data": (
                final_retrain_train_plus_val
            ),
            "n_original_train": original_train_rows,
            "n_training_rows": effective_training_rows,
            "n_h8_augmented_training_rows": (
                h8_augmented_training_rows
            ),
            "n_h8_randomized_augmented_training_rows": (
                h8_randomized_augmented_training_rows
            ),
            "observed_test_selection_acknowledged": bool(
                args.acknowledge_observed_test_selection
            ),
            "best_epoch": best_epoch,
            "task": task.name,
            "initialization": init_info,
            "resume": resume_info,
            "readout_model_config": {
                "orb_dim": orb_dim,
                "numeric_dim": 8,
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
                "regression_head_dropout": (
                    args.regression_head_dropout
                ),
                "use_token_type_cross_attention": args.cross_attention_view == "roles_types",
                "use_grouped_statistics": args.grouped_statistics,
                "use_cross_view_interactions": args.cross_view_interactions,
                "use_conformer_attention": (
                    args.conformer_aggregation == "attention"
                ),
                "conformer_attention_hidden_dim": (
                    args.conformer_attention_hidden_dim
                ),
                "use_directed_role_attention": args.directed_role_attention,
                "directed_query_role_id": 0,
                "directed_context_role_ids": (2, 8, 9),
                "use_reaction_pair_transformer": (
                    args.reaction_pair_transformer
                ),
                "reaction_pair_transformer_num_layers": (
                    args.reaction_pair_transformer_num_layers
                ),
                "reaction_pair_role_ids": (0, 2, 8, 9),
                "use_site_ligand_metal_attention": (
                    args.site_ligand_metal_attention
                ),
                "site_token_type_ids": (1, 2, 3),
                "ligand_token_type_ids": (6, 7),
                "metal_token_type_ids": (9,),
                "use_site_donor_relation_attention": (
                    args.site_donor_relation_attention
                ),
                "site_donor_relation_num_layers": (
                    args.site_donor_relation_num_layers
                ),
                "use_multi_level_orb_attention": (
                    args.multi_level_orb_attention
                ),
                "multi_level_orb_num_levels": len(
                    multi_level_orb_stack_indices
                ),
                "multi_level_orb_score_hidden_dim": (
                    args.multi_level_orb_score_hidden_dim
                ),
                "multi_level_orb_stack_indices": (
                    multi_level_orb_stack_indices
                ),
                "use_dual_backbone_feature_fusion": (
                    dual_backbone_feature_fusion_enabled
                ),
                "dual_backbone_feature_fusion_hidden_dim": (
                    args.dual_backbone_feature_fusion_hidden_dim
                ),
                "use_dual_backbone_feature_role_gate": (
                    args.dual_backbone_feature_role_gate
                ),
                "use_cip_orientation_embedding": (
                    cip_orientation_embedding_enabled
                ),
                "use_smiles_sequence_adapter": (
                    smiles_sequence_adapter_enabled
                ),
                "smiles_sequence_vocab_size": (
                    args.smiles_sequence_vocab_size
                ),
                "smiles_sequence_max_length": (
                    args.smiles_sequence_max_length
                ),
                "smiles_sequence_hidden_dim": (
                    args.smiles_sequence_hidden_dim
                ),
                "smiles_sequence_num_layers": (
                    args.smiles_sequence_num_layers
                ),
                "smiles_sequence_num_heads": (
                    args.smiles_sequence_num_heads
                ),
                "smiles_sequence_dropout": (
                    args.smiles_sequence_dropout
                ),
                "smiles_sequence_residual_multiplier": (
                    args.smiles_sequence_residual_multiplier
                ),
                "use_smiles_sequence_auxiliary_head": (
                    args.smiles_sequence_auxiliary
                ),
                "aho_consistency_mode": aho_consistency_mode,
                "aho_consistency_hidden_dim": (
                    args.aho_consistency_hidden_dim
                ),
                "aho_consistency_auxiliary_zero_point": (
                    auxiliary_zero_point
                ),
                "use_ddg_sign_magnitude": (
                    args.ddg_sign_magnitude
                ),
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
                "use_ddg_major_summary": (
                    args.ddg_major_summary
                ),
                "ddg_major_summary_classifier_hidden_dim": (
                    args.ddg_major_summary_classifier_hidden_dim
                ),
                "ddg_major_summary_residual_hidden_dim": (
                    args.ddg_major_summary_residual_hidden_dim
                ),
                "ddg_major_summary_role_ids": (0, 2, 8, 9),
                "ddg_major_summary_residual_multiplier": (
                    args.ddg_major_summary_residual_multiplier
                ),
                "ood_threshold": args.ood_threshold,
                "ood_scale_init": args.ood_scale_init,
                "exponential_ood_gate": args.ood_gate_mode == "exponential",
                "gate_residual_mode": args.gate_residual_mode,
                "ood_shift_scale": args.ood_shift_scale,
                "ood_base_shift": args.ood_base_shift,
                "residual_global_center_mix": args.residual_global_center_mix,
            },
        },
        out_dir / "model.pt",
    )
    joblib.dump(density_profile, out_dir / "density_profile.joblib")
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update(
        {
            "atom_token_dir": str(token_dir),
            "conformer_cache": str(conformer_cache),
            "mapping_task": mapping_task,
            "effective_mapping_task": effective_mapping_task,
            "validation_mode": args.validation_mode,
            "validation_part_names": part_names,
            "out_dir": str(out_dir),
        }
    )
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    print(pd.DataFrame([metrics_row]).to_string(index=False), flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
