#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/run_common.sh"

mkdir -p "$ROOT/json/catai_module_ids"
mkdir -p "$ROOT/json/catai_module_num"
mkdir -p "$ROOT/json/catai_module_weights"

PYTHON_BIN="$CATAI_PYTHON_BIN"
target_dataset="Alloy-GMAE-k32"

# bash "$ROOT/scripts/01_module_select/catai_module_select.sh"
"$PYTHON_BIN" "$ROOT/scripts/02_performance_evaluate/performance-evaluate.py" --dataset "$target_dataset"
"$PYTHON_BIN" "$ROOT/scripts/03_weight_optimize/weight-optimize.py" --dataset "$target_dataset"
bash "$ROOT/scripts/04_finetune_rerun/catai_finetune_rerun.sh"