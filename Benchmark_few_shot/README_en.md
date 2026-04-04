# Benchmark_few_shot

## Overview

This directory implements the CatAI few-shot benchmark. It mirrors the four-stage performance pipeline but evaluates model behavior under **limited labeled data** — a regime critical for real-world catalyst discovery, where experimental measurements are expensive and data-sparse conditions are the norm.

CatAI's module-fusion mechanism is particularly well-suited to few-shot settings: instead of learning from scratch, it transfers knowledge from pretrained modules trained on diverse source tasks and combines them into a task-specific predictor with minimal target-domain data. In comprehensive few-shot evaluation across benchmark datasets, CatAI improves accuracy by over **30%** versus few-shot baselines.

The default reproducible entry uses the **Alloy-GMAE** dataset, GPU 0, seed 0.

## How it differs from Benchmark_performance

The pipeline stages are identical to `Benchmark_performance`. The key difference is in the dataset split used during training: few-shot splits contain substantially fewer labeled examples per task, stressing the module selection and weight optimization stages.

## Resource paths

Set by `run_common.sh`. Override with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `$CATAI_DATA_ROOT` | `../catai_data/catalyst_task` | Root of benchmark LMDB datasets |
| `$CATAI_MODULE_HUB_PATH` | `../catai_data/all_modules` | Directory containing pretrained knowledge modules |
| `$CATAI_BASE_CHECKPOINT_PATH` | `../catai_data/all_modules/jmp-l.pt` | JMP-L base model checkpoint |

## Four-stage pipeline

### Stage 1 — Module selection

Scores pretrained modules from `../catai_data/all_modules/` on the few-shot labeled split. Selects the most informative modules for the target task.

```bash
bash scripts/01_module_select/catai_module_select.sh
```

### Stage 2 — Performance evaluation

Evaluates the selected module combination under the few-shot regime and reports MAE.

```bash
python scripts/02_performance_evaluate/performance-evaluate.py --dataset Alloy-GMAE
```

### Stage 3 — Weight optimization

Learns the optimal combination weights over selected modules using the few-shot training set.

```bash
python scripts/03_weight_optimize/weight-optimize.py --dataset Alloy-GMAE
```

### Stage 4 — Finetune rerun

Fine-tunes the module-weighted JMP-L model on the few-shot training set. The best checkpoint is saved under `checkpoints/`.

```bash
bash scripts/04_finetune_rerun/catai_finetune_rerun.sh
```

## One-click run

```bash
bash run_all.sh
```

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
| `checkpoints/` | Saved model checkpoints |
| `results_ebd/` | Graph embeddings and evaluation results |
| `module_selection_outputs/` | Selected module lists and weight files |
