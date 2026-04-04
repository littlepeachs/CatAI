#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$ROOT/run_common.sh"

mode=${1:-all}

if [[ ! -d "$CATAI_ACTIVE_EXPORT_ROOT" ]]; then
    echo "Active benchmark expects export data at: $CATAI_ACTIVE_EXPORT_ROOT"
    echo "Prepare an exp_data directory first, then rerun this command."
    exit 1
fi

case "$mode" in
    baselines)
        bash "$ROOT/scripts/run_active_models.sh"
        ;;
    catai)
        bash "$ROOT/scripts/run_active_catai.sh"
        ;;
    all)
        bash "$ROOT/scripts/run_active_models.sh"
        bash "$ROOT/scripts/run_active_catai.sh"
        ;;
    *)
        echo "Usage: $0 [baselines|catai|all]"
        exit 1
        ;;
esac