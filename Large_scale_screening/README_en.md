# Large_scale_screening

## Overview

This directory implements the full CatAI virtual screening pipeline applied to **industrial polyolefin (POE) polymerization**. Starting from a module selection step identical to the performance benchmark, it trains a task-specific screening model, deploys it over a large candidate library, ranks the predictions, and generates a removal list of low-performing candidates.

In this application, CatAI identified metallocene catalyst candidates exhibiting **300% activity enhancement** over commercial standards. The key design principle uncovered is a **saturated bridging architecture** that imports heterogeneous-style geometric control — precise spatial positioning of active-site ligands — into molecular catalyst frameworks. This represents a cross-paradigm insight: a design logic from solid-state heterogeneous catalysis transferred to homogeneous molecular systems through large-scale computational screening.

## Pipeline stages

### Stage 1 — Module selection

Scores pretrained modules from `../catai_data/all_modules/` against the POE training set. Identifies the module subset most predictive of polymerization-relevant properties.

```bash
bash scripts/01_module_select/catai_screening_module_select.sh
```

### Stage 2 — Performance evaluation

Evaluates the selected modules on POE-specific metrics.

```bash
python scripts/02_performance_evaluate/catai_screening_performance_evaluate.py --dataset POE_lmdb
```

### Stage 3 — Weight optimization

Optimizes combination weights for the selected modules on the POE dataset.

```bash
python scripts/03_weight_optimize/catai_screening_weight_optimize.py --dataset POE_lmdb
```

### Stage 4 — Screening model training

Fine-tunes the module-weighted model on the full POE training set to produce a screening model checkpoint.

```bash
bash scripts/04_screening_train/catai_screening_train.sh
```

### Stage 5 — Large-scale inference

Deploys the trained screening model over the full candidate library (`POE_infer/`). Requires the checkpoint from stage 4.

```bash
export CATAI_SCREENING_CHECKPOINT_PATH=/absolute/path/to/best_checkpoint.pt
bash scripts/05_inference/catai_screening_infer.sh
python scripts/05_inference/catai_screening_rank_poe.py
```

### Stage 6 — Visualization

Generates scatter plots and ranking visualizations of the screening results.

```bash
bash scripts/06_visualization/catai_screening_scatter.sh
```

## Resource paths

Set by `run_common.sh`. Override with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `$CATAI_MODULE_HUB_PATH` | `../catai_data/all_modules` | Pretrained knowledge modules |
| `$CATAI_BASE_CHECKPOINT_PATH` | `../catai_data/all_modules/jmp-l.pt` | JMP-L base model checkpoint |
| `$CATAI_DATA_ROOT` | local `POE_lmdb/` directory | POE training data (LMDB) |
| `$CATAI_POE_INFER_ROOT` | local `POE_infer/` directory | Large candidate library for inference |
| `$CATAI_SCREENING_CHECKPOINT_PATH` | trained by stage 4 | Screening model checkpoint for stages 5–6 |

## One-click run

Run the full training chain (stages 1–4):

```bash
bash run_all.sh train
```

Run inference, ranking, and visualization using an existing checkpoint:

```bash
bash run_all.sh infer /absolute/path/to/best_checkpoint.pt
```

Run the complete pipeline end-to-end:

```bash
bash run_all.sh full /absolute/path/to/best_checkpoint.pt
```

## Step-by-step

```bash
source run_common.sh
bash scripts/01_module_select/catai_screening_module_select.sh
python scripts/02_performance_evaluate/catai_screening_performance_evaluate.py --dataset POE_lmdb
python scripts/03_weight_optimize/catai_screening_weight_optimize.py --dataset POE_lmdb
bash scripts/04_screening_train/catai_screening_train.sh
export CATAI_SCREENING_CHECKPOINT_PATH=/absolute/path/to/best_checkpoint.pt
bash scripts/05_inference/catai_screening_infer.sh
python scripts/05_inference/catai_screening_rank_poe.py
bash scripts/06_visualization/catai_screening_scatter.sh
```

## Main outputs

| Path | Contents |
|------|----------|
| `output_logs/` | Training and inference logs |
| `checkpoints/` | Screening model checkpoints from stage 4 |
| `results/` | Per-candidate predictions and ranked lists |
| `remove.txt` | List of low-performing candidates to exclude |

## Note

Stages 5 and 6 require a screening model checkpoint explicitly. This checkpoint is produced by stage 4, **not** the read-only base model stored in `../catai_data/all_modules/`. Provide the stage-4 output path via `$CATAI_SCREENING_CHECKPOINT_PATH` before running inference.
