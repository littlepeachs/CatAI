#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/run_common.sh"

PYTHON_BIN="$CATAI_PYTHON_BIN"
mode=${1:-train}
screening_ckpt=${2:-$CATAI_SCREENING_CHECKPOINT_PATH}

case "$mode" in
    train)
        bash "$ROOT/scripts/01_module_select/catai_screening_module_select.sh"
        "$PYTHON_BIN" "$ROOT/scripts/02_performance_evaluate/catai_screening_performance_evaluate.py" --dataset POE_lmdb
        "$PYTHON_BIN" "$ROOT/scripts/03_weight_optimize/catai_screening_weight_optimize.py" --dataset POE_lmdb
        bash "$ROOT/scripts/04_screening_train/catai_screening_train.sh"
        ;;
    infer)
        export CATAI_SCREENING_CHECKPOINT_PATH="$screening_ckpt"
        bash "$ROOT/scripts/05_inference/catai_screening_infer.sh"
        "$PYTHON_BIN" "$ROOT/scripts/05_inference/catai_screening_rank_poe.py"
        # bash "$ROOT/scripts/06_visualization/catai_screening_scatter.sh"
        ;;
    full)
        export CATAI_SCREENING_CHECKPOINT_PATH="$screening_ckpt"
        bash "$ROOT/scripts/01_module_select/catai_screening_module_select.sh"
        "$PYTHON_BIN" "$ROOT/scripts/02_performance_evaluate/catai_screening_performance_evaluate.py" --dataset POE_lmdb
        "$PYTHON_BIN" "$ROOT/scripts/03_weight_optimize/catai_screening_weight_optimize.py" --dataset POE_lmdb
        bash "$ROOT/scripts/04_screening_train/catai_screening_train.sh"
        bash "$ROOT/scripts/05_inference/catai_screening_infer.sh"
        "$PYTHON_BIN" "$ROOT/scripts/05_inference/catai_screening_rank_poe.py"
        bash "$ROOT/scripts/06_visualization/catai_screening_scatter.sh"
        ;;
    *)
        echo "Usage: $0 [train|infer|full] [screening_checkpoint_path]"
        exit 1
        ;;
esac