#!/bin/bash
set -euo pipefail

RUN_COMMON_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CATAI_CLOSED_LOOP_ROOT=${CATAI_CLOSED_LOOP_ROOT:-$RUN_COMMON_DIR}

export CATAI_PYTHON_BIN=${CATAI_PYTHON_BIN:-python}
export CATAI_MODULE_HUB_PATH=${CATAI_MODULE_HUB_PATH:-$RUN_COMMON_DIR/../catai_data/all_modules}
export CATAI_BASE_CHECKPOINT_PATH=${CATAI_BASE_CHECKPOINT_PATH:-$RUN_COMMON_DIR/../catai_data/all_modules/jmp-l.pt}

export CATAI_CONDITIONAL1_DATA_ROOT=${CATAI_CONDITIONAL1_DATA_ROOT:-$CATAI_CLOSED_LOOP_ROOT/conditional_rounds/conditional1}
export CATAI_CONDITIONAL2_DATA_ROOT=${CATAI_CONDITIONAL2_DATA_ROOT:-$CATAI_CLOSED_LOOP_ROOT/conditional_rounds/conditional2}
export CATAI_CONDITIONAL3_DATA_ROOT=${CATAI_CONDITIONAL3_DATA_ROOT:-$CATAI_CLOSED_LOOP_ROOT/conditional_rounds/conditional3}
export CATAI_CONDITIONAL4_DATA_ROOT=${CATAI_CONDITIONAL4_DATA_ROOT:-$CATAI_CLOSED_LOOP_ROOT/conditional_rounds/conditional4}

export CATAI_CONDITIONAL1_DEVICE=${CATAI_CONDITIONAL1_DEVICE:-0}
export CATAI_CONDITIONAL2_DEVICE=${CATAI_CONDITIONAL2_DEVICE:-1}
export CATAI_CONDITIONAL3_DEVICE=${CATAI_CONDITIONAL3_DEVICE:-2}
export CATAI_CONDITIONAL4_DEVICE=${CATAI_CONDITIONAL4_DEVICE:-3}

export CATAI_CONDITIONAL1_PORT=${CATAI_CONDITIONAL1_PORT:-1301}
export CATAI_CONDITIONAL2_PORT=${CATAI_CONDITIONAL2_PORT:-1302}
export CATAI_CONDITIONAL3_PORT=${CATAI_CONDITIONAL3_PORT:-1303}
export CATAI_CONDITIONAL4_PORT=${CATAI_CONDITIONAL4_PORT:-1304}

print_conditional_usage() {
    local conditional_name="$1"
    local supports_ebd="$2"

    echo "Usage: $conditional_name <train|infer${supports_ebd:+|ebd}> [checkpoint_path] [epoch] [device] [port]"
    echo "  train                          运行 stage2_recommend.sh"
    echo "  infer <ckpt> <epoch> [d] [p]   运行 stage3_infer.sh"
    if [[ -n "$supports_ebd" ]]; then
        echo "  ebd <ckpt> <epoch> [d] [p]     运行 stage3_get_ebd.sh"
    fi
}