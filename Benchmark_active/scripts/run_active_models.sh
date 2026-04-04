#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$BENCHMARK_ROOT/run_common.sh"
PYTHON_BIN="$CATAI_PYTHON_BIN"
EXPORT_ROOT="$CATAI_ACTIVE_EXPORT_ROOT"
LOG_DIR="$BENCHMARK_ROOT/active_logs"

mkdir -p "$LOG_DIR"

rm -rf "$EXPORT_ROOT"/[C]/al_runs/gemnet_oc "$EXPORT_ROOT"/[C]/al_runs/cgcnn "$EXPORT_ROOT"/[C]/al_runs/dpp "$EXPORT_ROOT"/[C]/al_runs/schnet "$EXPORT_ROOT"/[C]/al_runs/et
rm -rf "$EXPORT_ROOT"/[O]/al_runs/gemnet_oc "$EXPORT_ROOT"/[O]/al_runs/cgcnn "$EXPORT_ROOT"/[O]/al_runs/dpp "$EXPORT_ROOT"/[O]/al_runs/schnet "$EXPORT_ROOT"/[O]/al_runs/et

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" "$BENCHMARK_ROOT/active_learning.py" multi-trial --export-root "$EXPORT_ROOT" --model-type gemnet_oc --trials 40
CUDA_VISIBLE_DEVICES=1 "$PYTHON_BIN" "$BENCHMARK_ROOT/active_learning.py" multi-trial --export-root "$EXPORT_ROOT" --model-type cgcnn --trials 40
CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" "$BENCHMARK_ROOT/active_learning.py" multi-trial --export-root "$EXPORT_ROOT" --model-type dpp --trials 40
CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" "$BENCHMARK_ROOT/active_learning.py" multi-trial --export-root "$EXPORT_ROOT" --model-type schnet --trials 40
CUDA_VISIBLE_DEVICES=5 "$PYTHON_BIN" "$BENCHMARK_ROOT/active_learning.py" multi-trial --export-root "$EXPORT_ROOT" --model-type et --trials 40