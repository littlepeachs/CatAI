# Benchmark_performance

## Overview

This directory implements the standard CatAI performance benchmark pipeline. It evaluates CatAI's module-fusion strategy — the core mechanism of Composable Catalytic Intelligence — against baseline approaches across diverse catalytic prediction tasks.

CatAI operates through **module fusion**: a set of pretrained knowledge modules (stored in `../catai_data/all_modules/`) are scored on a small labeled split of the target task, a weighted combination is learned, and the result is fine-tuned on the full training set using the JMP-L backbone. This approach systematically transfers knowledge from source tasks to a target property without requiring full retraining.

On a comprehensive benchmark of six datasets covering prediction of adsorption, desorption, dissociation, and transfer energies, CatAI reduces the average mean absolute error by **7.8%** compared to single-model baselines.

The default reproducible entry runs the full pipeline on the **Alloy-GMAE** dataset, GPU 0, seed 0.

## Resource paths

Set by `run_common.sh`. Override with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `$CATAI_DATA_ROOT` | `../catai_data/catalyst_task` | Root of benchmark LMDB datasets |
| `$CATAI_MODULE_HUB_PATH` | `../catai_data/all_modules` | Directory containing pretrained knowledge modules |
| `$CATAI_BASE_CHECKPOINT_PATH` | `../catai_data/all_modules/jmp-l.pt` | JMP-L base model checkpoint |

## Four-stage pipeline

### Stage 1 — Module selection (`scripts/01_module_select/`)

Scores each pretrained knowledge module against a small labeled split of the target task. A greedy selection strategy identifies the most informative subset of modules. The output is a ranked list of module names written to `module_selection_outputs/`.

```bash
bash scripts/01_module_select/catai_module_select.sh
```

### Stage 2 — Performance evaluation (`scripts/02_performance_evaluate/`)

Evaluates the selected module set. Reports per-dataset MAE, enabling direct comparison between the module-selection result and baselines.

```bash
python scripts/02_performance_evaluate/performance-evaluate.py --dataset Alloy-GMAE
```

### Stage 3 — Weight optimization (`scripts/03_weight_optimize/`)

Optimizes the linear combination weights over the selected modules to minimize validation loss. Outputs a weight file used in the subsequent fine-tune step.

```bash
python scripts/03_weight_optimize/weight-optimize.py --dataset Alloy-GMAE
```

### Stage 4 — Finetune rerun (`scripts/04_finetune_rerun/`)

Loads the JMP-L backbone, applies the optimized module weights, and fine-tunes the combined model on the full training set. The best checkpoint is saved under `checkpoints/`.

```bash
bash scripts/04_finetune_rerun/catai_finetune_rerun.sh
```

## One-click run

From the `Benchmark_performance/` directory:

```bash
bash run_all.sh
```

Runs all four stages on Alloy-GMAE, seed 0, GPU 0.

## Step-by-step

```bash
source run_common.sh
bash scripts/01_module_select/catai_module_select.sh
python scripts/02_performance_evaluate/performance-evaluate.py --dataset Alloy-GMAE
python scripts/03_weight_optimize/weight-optimize.py --dataset Alloy-GMAE
bash scripts/04_finetune_rerun/catai_finetune_rerun.sh
```

## Override paths

```bash
export CATAI_PYTHON_BIN=$(which python)
export CATAI_DATA_ROOT=../catai_data/catalyst_task
export CATAI_MODULE_HUB_PATH=../catai_data/all_modules
export CATAI_BASE_CHECKPOINT_PATH=../catai_data/all_modules/jmp-l.pt
bash run_all.sh
```

## Main outputs

| Path | Contents |
|------|----------|
| `output_logs/` | Training logs for each stage |
| `checkpoints/` | Saved model checkpoints from the finetune stage |
| `results_ebd/` | Graph embeddings and evaluation results |
| `module_selection_outputs/` | Selected module lists and weight files |
