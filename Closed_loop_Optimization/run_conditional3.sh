#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/run_common.sh"

CONDITIONAL_ROOT="$SCRIPT_DIR/conditional_rounds/conditional3"
ACTION=${1:-}

case "$ACTION" in
    train)
        exec bash "$CONDITIONAL_ROOT/stage2_recommend.sh"
        ;;
    infer)
        if [[ $# -lt 3 ]]; then
            print_conditional_usage "run_conditional3.sh" "ebd"
            exit 1
        fi
        CKPT_PATH="$2"
        EPOCH_TAG="$3"
        DEVICE=${4:-$CATAI_CONDITIONAL3_DEVICE}
        PORT=${5:-$CATAI_CONDITIONAL3_PORT}
        exec bash "$CONDITIONAL_ROOT/stage3_infer.sh" "$DEVICE" "$PORT" "$CKPT_PATH" "$EPOCH_TAG"
        ;;
    ebd)
        if [[ $# -lt 3 ]]; then
            print_conditional_usage "run_conditional3.sh" "ebd"
            exit 1
        fi
        CKPT_PATH="$2"
        EPOCH_TAG="$3"
        DEVICE=${4:-$CATAI_CONDITIONAL3_DEVICE}
        PORT=${5:-$CATAI_CONDITIONAL3_PORT}
        exec bash "$CONDITIONAL_ROOT/stage3_get_ebd.sh" "$DEVICE" "$PORT" "$CKPT_PATH" "$EPOCH_TAG"
        ;;
    *)
        print_conditional_usage "run_conditional3.sh" "ebd"
        exit 1
        ;;
esac