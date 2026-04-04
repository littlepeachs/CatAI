#!/bin/bash
set -euo pipefail

RUN_COMMON_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export CATAI_FEW_SHOT_ROOT=${CATAI_FEW_SHOT_ROOT:-$RUN_COMMON_DIR}
export CATAI_PYTHON_BIN=${CATAI_PYTHON_BIN:-python}
export CATAI_DATA_ROOT=${CATAI_DATA_ROOT:-$RUN_COMMON_DIR/../catai_data/catalyst_task}
export CATAI_MODULE_HUB_PATH=${CATAI_MODULE_HUB_PATH:-$RUN_COMMON_DIR/../catai_data/all_modules}
export CATAI_BASE_CHECKPOINT_PATH=${CATAI_BASE_CHECKPOINT_PATH:-$RUN_COMMON_DIR/../catai_data/all_modules/jmp-l.pt}