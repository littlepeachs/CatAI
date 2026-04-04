#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/run_common.sh"

PYTHON_BIN="$CATAI_PYTHON_BIN"
mode=${1:-full_train}

case "$mode" in
    precompute)
        bash "$ROOT/scripts/01_module_select/catai_module_select.sh"
        "$PYTHON_BIN" "$ROOT/scripts/02_performance_evaluate/catai_performance_evaluate.py" --dataset photocat
        "$PYTHON_BIN" "$ROOT/scripts/03_weight_optimize/catai_weight_optimize.py" --dataset photocat
        ;;
    full_train)
        bash "$ROOT/scripts/01_module_select/catai_module_select.sh"
        "$PYTHON_BIN" "$ROOT/scripts/02_performance_evaluate/catai_performance_evaluate.py" --dataset photocat
        "$PYTHON_BIN" "$ROOT/scripts/03_weight_optimize/catai_weight_optimize.py" --dataset photocat
        bash "$ROOT/run_conditional1.sh" train
        bash "$ROOT/run_conditional2.sh" train
        bash "$ROOT/run_conditional3.sh" train
        bash "$ROOT/run_conditional4.sh" train
        ;;
    infer1|infer2|infer3|infer4|ebd3|ebd4)
        if [[ $# -lt 3 ]]; then
            echo "Usage: $0 $mode <checkpoint_path> <epoch> [device] [port]"
            exit 1
        fi
        subcommand=${mode#infer}
        case "$mode" in
            infer1) bash "$ROOT/run_conditional1.sh" infer "$2" "$3" "${4:-$CATAI_CONDITIONAL1_DEVICE}" "${5:-$CATAI_CONDITIONAL1_PORT}" ;;
            infer2) bash "$ROOT/run_conditional2.sh" infer "$2" "$3" "${4:-$CATAI_CONDITIONAL2_DEVICE}" "${5:-$CATAI_CONDITIONAL2_PORT}" ;;
            infer3) bash "$ROOT/run_conditional3.sh" infer "$2" "$3" "${4:-$CATAI_CONDITIONAL3_DEVICE}" "${5:-$CATAI_CONDITIONAL3_PORT}" ;;
            infer4) bash "$ROOT/run_conditional4.sh" infer "$2" "$3" "${4:-$CATAI_CONDITIONAL4_DEVICE}" "${5:-$CATAI_CONDITIONAL4_PORT}" ;;
            ebd3) bash "$ROOT/run_conditional3.sh" ebd "$2" "$3" "${4:-$CATAI_CONDITIONAL3_DEVICE}" "${5:-$CATAI_CONDITIONAL3_PORT}" ;;
            ebd4) bash "$ROOT/run_conditional4.sh" ebd "$2" "$3" "${4:-$CATAI_CONDITIONAL4_DEVICE}" "${5:-$CATAI_CONDITIONAL4_PORT}" ;;
        esac
        ;;
    *)
        echo "Usage: $0 [precompute|full_train|infer1|infer2|infer3|infer4|ebd3|ebd4] ..."
        exit 1
        ;;
esac