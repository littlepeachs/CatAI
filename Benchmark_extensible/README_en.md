# Benchmark_extensible

## Overview

This directory groups four extensibility and ablation-style experiments that probe the structural properties of CatAI's modular design:

1. **Random Select** — replaces the learned module selection with random sampling, serving as a lower bound.
2. **Random Select 2** — a second random-selection variant with different sampling parameters.
3. **Single** — uses a single pretrained module (no fusion), isolating the contribution of any one source task.
4. **Scaling** — varies the number of modules fused, tracing the accuracy-versus-module-count curve.

Together, these experiments demonstrate that CatAI's performance gains arise from principled module selection and combination, rather than from the base model or data volume alone. They also validate the core extensibility claim: **new knowledge modules can be added to `../catai_data/all_modules/` without retraining any existing module**, and immediately composed into new predictors.

All default benchmark entries use Alloy-GMAE, seed 0, GPU 0.

## Resource paths

| Variable | Default | Description |
|----------|---------|-------------|
| `$CATAI_EXTENSIBLE_DATA_ROOT` | `../catai_data/catalyst_task` | Root of benchmark LMDB datasets |
| `$CATAI_EXTENSIBLE_RANDOM_HUB_PATH` | `../catai_data/all_modules` | Module hub for random-selection experiments |
| `$CATAI_EXTENSIBLE_MODULE_HUB_PATH` | `../catai_data/all_modules` | Module hub for single / scaling experiments |
| `$CATAI_EXTENSIBLE_BASE_CHECKPOINT_PATH` | `../catai_data/all_modules/jmp-l.pt` | JMP-L base model checkpoint |

## One-click run

Run the full extensible suite:

```bash
bash run_all.sh all
```

Run a single-module baseline (module name, GPU, port):

```bash
bash run_all.sh single mp_eform 0 1290
```

Run one scaling configuration (number of modules, GPU, port):

```bash
bash run_all.sh scaling 10 0 1301
```

## Step-by-step

```bash
source run_common.sh

# Experiment 1: random module selection
bash scripts/01_random_select/catai_random_select.sh

# Experiment 2: random selection variant 2
bash scripts/02_random_select_2/catai_random_select_2.sh

# Experiment 3: single-module baseline
bash scripts/03_single/catai_single_module.sh mp_eform 0 1290

# Experiment 4: scaling (vary module count)
bash scripts/04_scaling/run_scaling_suite.sh 0 1301
```

## Environment check

For a quick check that the environment and data paths are correctly configured:

```bash
bash run_all.sh single mp_eform 0 1290
```

This is the lightest entry in the suite: it loads a single module from `../catai_data/all_modules/` and fine-tunes on Alloy-GMAE. If this runs to completion, the full suite is ready.

## Main outputs

| Path | Contents |
|------|----------|
| `output_logs/random_select/` | Random selection training logs |
| `output_logs/random_select_2/` | Random selection variant 2 logs |
| `output_logs/single/` | Single-module baseline logs |
| `output_logs/scaling/` | Scaling experiment logs |
| `checkpoints/` | Saved checkpoints per experiment |
