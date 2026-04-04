#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
mode=${1:-all}
arg1=${2:-}
arg2=${3:-}

case "$mode" in
    all)
        bash "$ROOT/scripts/01_random_select/catai_random_select.sh"
        bash "$ROOT/scripts/02_random_select_2/catai_random_select_2.sh"
        bash "$ROOT/scripts/04_scaling/run_scaling_suite.sh" "${arg1:-0}" "${arg2:-1301}"
        ;;
    single)
        if [[ -z "$arg1" ]]; then
            echo "Usage: $0 single <module_name> [device] [port]"
            exit 1
        fi
        bash "$ROOT/scripts/03_single/catai_single_module.sh" "$arg1" "${arg2:-0}" "${4:-1290}"
        ;;
    scaling)
        if [[ -z "$arg1" ]]; then
            echo "Usage: $0 scaling <scale> [device] [port]"
            exit 1
        fi
        bash "$ROOT/scripts/04_scaling/catai_scaling_param.sh" "$arg1" "${arg2:-0}" "${4:-1301}"
        ;;
    random1)
        bash "$ROOT/scripts/01_random_select/catai_random_select.sh" "${arg1:-0}" "${arg2:-1301}"
        ;;
    random2)
        bash "$ROOT/scripts/02_random_select_2/catai_random_select_2.sh" "${arg1:-0}" "${arg2:-1300}"
        ;;
    *)
        echo "Usage: $0 [all|single|scaling|random1|random2] ..."
        exit 1
        ;;
esac