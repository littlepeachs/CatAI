# Benchmark_active

## Overview

This directory implements the CatAI active learning benchmark. It simulates the realistic scenario where a discovery campaign must propose informative candidate experiments **iteratively under an evolving, non-stationary data distribution** — the dominant deployment paradigm in experimental catalyst discovery.

Two complementary evaluation groups are provided:

1. **Baseline models** (`scripts/run_active_models.sh`): five established models serving as reference backends — `gemnet_oc`, `cgcnn`, `dpp`, `schnet`, `et`.
2. **CatAI active-learning backend** (`scripts/run_active_catai.sh`): the full CatAI module-fusion pipeline deployed in the active learning loop, current scripts targeting the `[C]` (carbon) and `[O]` (oxygen) adsorption tasks.

CatAI maintains consistent module-selection and fusion advantages as the labeled pool grows across rounds, increasing the overall discovery success rate by over **15%** compared to single-model active learning baselines.

## Data layout

This workflow uses a **different data layout** from the other benchmarks. The active learning loop expects:

```text
exp_data/
├── [C]/
│   ├── id_prop.csv        # ground-truth labels for all candidates
│   └── structures/        # CIF or ASE-compatible structure files
└── [O]/
    ├── id_prop.csv
    └── structures/
```

The `catalyst_task/al_C` and `catalyst_task/al_O` directories in `../catai_data/catalyst_task/` are **LMDB** datasets, not the `exp_data` layout required by `active_learning.py`. They cannot be used directly as `export_root`. You must prepare `exp_data/` with the structure above before running.

The default `export_root` is `exp_data/` inside this directory.

## Pretrained active learning checkpoints

CatAI-specific active learning checkpoints are stored in `al_ckpt/`:

- `al_ckpt/al_C.pt` — checkpoint for the `[C]` task
- `al_ckpt/al_O.pt` — checkpoint for the `[O]` task

## One-click run

After preparing `exp_data/`:

```bash
bash run_all.sh all
```

Run baseline models only:

```bash
bash run_all.sh baselines
```

Run CatAI only:

```bash
bash run_all.sh catai
```

## Custom export_root

```bash
export CATAI_ACTIVE_EXPORT_ROOT=/absolute/path/to/exp_data
bash run_all.sh all
```

## Step-by-step

```bash
source run_common.sh
bash scripts/run_active_models.sh    # baselines
bash scripts/run_active_catai.sh     # CatAI
```

## Main outputs

| Path | Contents |
|------|----------|
| `active_logs/` | Training and selection logs per round |
| `exp_data/[C]/al_runs/` | Per-round candidate selections and predictions for carbon task |
| `exp_data/[O]/al_runs/` | Per-round candidate selections and predictions for oxygen task |
| `catai_c.out` | Console output from the CatAI `[C]` run |
| `catai_o.out` | Console output from the CatAI `[O]` run |

## Note

If `exp_data/` is missing or does not contain both `[C]/id_prop.csv` and `[O]/id_prop.csv`, the one-click script stops immediately with a clear error.
