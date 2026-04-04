# Closed_loop_Optimization

## Overview

This directory implements the most end-to-end workflow in CatAI: a **multi-round closed-loop optimization** pipeline coupling CatAI predictions with high-throughput experimental (HTE) feedback for **heterogeneous photocatalysis**.

In this paradigm, each experimental round selects a new batch of candidates based on CatAI's current model, measurements are returned from HTE, and the model is updated with new labels before the next round of candidate proposal. Four sequential conditional rounds (`conditional1` through `conditional4`) progressively refine the candidate pool.

This workflow led to the discovery of **Cu GAPC** — an earth-abundant copper-based geminal-atom photocatalyst that embeds molecular-level active-site precision within solid-state supports, achieving excellent yields across a wide variety of chlorine-containing substrates. The design principle — geminal metal coordination at crystallographic sites — combines the tunability of homogeneous catalysts with the stability and recyclability of heterogeneous materials.

## Workflow structure

The pipeline consists of two phases:

**Phase 1 — Top-level precompute (run once)**

1. Module selection: scores pretrained modules from `../catai_data/all_modules/` on the photocatalysis training set.
2. Performance evaluation: validates selected modules on photocatalysis-specific metrics.
3. Weight optimization: learns combination weights for the selected modules.

**Phase 2 — Sequential conditional rounds (four rounds)**

Each round `conditionalN` (N = 1, 2, 3, 4):
- **Stage 1**: train on current labeled pool (LMDB in `conditional_rounds/conditionalN/`).
- **Stage 2**: model-guided candidate recommendation.
- **Stage 3**: inference over candidate library; produce ranked predictions and embeddings.
- Experimental measurements come back from HTE and are added to the labeled pool for round N+1.

## Resource paths

Set by `run_common.sh`. Override with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `$CATAI_MODULE_HUB_PATH` | `../catai_data/all_modules` | Pretrained knowledge modules |
| `$CATAI_BASE_CHECKPOINT_PATH` | `../catai_data/all_modules/jmp-l.pt` | JMP-L base model checkpoint |
| `$CATAI_CONDITIONAL1_DATA_ROOT` | `conditional_rounds/conditional1` | Round-1 labeled pool |
| `$CATAI_CONDITIONAL2_DATA_ROOT` | `conditional_rounds/conditional2` | Round-2 labeled pool |
| `$CATAI_CONDITIONAL3_DATA_ROOT` | `conditional_rounds/conditional3` | Round-3 labeled pool |
| `$CATAI_CONDITIONAL4_DATA_ROOT` | `conditional_rounds/conditional4` | Round-4 labeled pool |

## One-click run

Run the top-level precompute chain and all four training rounds:

```bash
bash run_all.sh full_train
```

Run only the top-level precompute stages (module selection, evaluation, weight optimization):

```bash
bash run_all.sh precompute
```

Run the inference step for round 1 (provide checkpoint path, epoch tag, GPU, port):

```bash
bash run_all.sh infer1 /absolute/path/to/conditional1_ckpt.pt best_checkpoint 0 1301
```

Run the embedding export for round 3:

```bash
bash run_all.sh ebd3 /absolute/path/to/conditional3_ckpt.pt best_checkpoint 2 1303
```

## Step-by-step

```bash
source run_common.sh

# Phase 1: top-level precompute
bash scripts/01_module_select/catai_module_select.sh
python scripts/02_performance_evaluate/catai_performance_evaluate.py --dataset photocat
python scripts/03_weight_optimize/catai_weight_optimize.py --dataset photocat

# Phase 2: four conditional training rounds
bash run_conditional1.sh train
bash run_conditional2.sh train
bash run_conditional3.sh train
bash run_conditional4.sh train
```

## Main outputs

| Path | Contents |
|------|----------|
| `checkpoints/` | Top-level and per-round model checkpoints |
| `results_ebd/` | Graph embeddings for candidate ranking |
| `logs/` | Top-level training logs |
| `conditional_rounds/*/output_logs/` | Per-round training and inference logs |

## Note

Inference and embedding-export stages require the checkpoint path and epoch tag produced by the corresponding training round. These are not the base model in `../catai_data/all_modules/`, but the per-round checkpoints written to `conditional_rounds/conditionalN/checkpoints/`.
