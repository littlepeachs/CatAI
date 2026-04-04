#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$BENCHMARK_ROOT/run_common.sh"
PYTHON_BIN="$CATAI_PYTHON_BIN"
EXPORT_ROOT="$CATAI_ACTIVE_EXPORT_ROOT"

CUDA_VISIBLE_DEVICES=4 "$PYTHON_BIN" "$BENCHMARK_ROOT/active_learning.py" multi-trial --export-root "$EXPORT_ROOT" --model-type catai --trials 40 --task-name [C]
CUDA_VISIBLE_DEVICES=7 "$PYTHON_BIN" "$BENCHMARK_ROOT/active_learning.py" multi-trial --export-root "$EXPORT_ROOT" --model-type catai --trials 40 --task-name [O]