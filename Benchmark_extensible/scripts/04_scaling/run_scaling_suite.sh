#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
device=${1:-0}
start_port=${2:-1301}

port=$start_port
for scale in 5 10 15 20 25; do
    bash "$SCRIPT_DIR/catai_scaling_param.sh" "$scale" "$device" "$port"
    port=$((port + 1))
done
