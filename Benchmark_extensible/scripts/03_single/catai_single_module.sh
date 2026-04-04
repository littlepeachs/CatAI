#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$BENCHMARK_ROOT/run_common.sh"

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <module_name> [device] [port]"
    exit 1
fi

MODULE_NAME="$1"
device=${2:-0}
port=${3:-1290}

PYTHON_BIN="$CATAI_EXTENSIBLE_PYTHON_BIN"
DATA_ROOT="$CATAI_EXTENSIBLE_DATA_ROOT"
MODULE_BASE_PATH="$CATAI_EXTENSIBLE_MODULE_HUB_PATH"
CKPT_NAME="$CATAI_EXTENSIBLE_SINGLE_CHECKPOINT_NAME"
CKPT_PATH="$MODULE_BASE_PATH/$MODULE_NAME/$CKPT_NAME.pt"

n_gpus=1
bsz=8
seeds=("0")

for data_name in {'Alloy-GMAE',}; do
for seed in ${seeds[@]}; do
    dict_mean=$($PYTHON_BIN -c "import json; data_name='$data_name'; print(json.load(open('$BENCHMARK_ROOT/json/stage2_mean.json'))[data_name])")
    dict_std=$($PYTHON_BIN -c "import json; data_name='$data_name'; print(json.load(open('$BENCHMARK_ROOT/json/stage2_std.json'))[data_name])")
    current_datetime=$(date '+%m-%d-%H-%M')

    WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$device $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$n_gpus --master_port=$port "$BENCHMARK_ROOT/main.py" \
        --num-gpus $n_gpus --distributed \
        --mode train \
        --task.data_name=$data_name \
        --config-yml "$BENCHMARK_ROOT/configs/stage2_base.yml" \
        --dataset.train.config.src="$DATA_ROOT/$data_name/$seed/train" \
        --dataset.val[0].config.src="$DATA_ROOT/$data_name/$seed/val" \
        --dataset.test[0].config.src="$DATA_ROOT/$data_name/$seed/test" \
        --dataset.normalization.y.mean=$dict_mean \
        --dataset.normalization.y.std=$dict_std \
        --identifier catai_single+$data_name+seed$seed+$MODULE_NAME \
        --task.base_checkpoint.src="$CKPT_PATH" \
        --optim.batch_size=$bsz \
        --optim.eval_batch_size=$bsz
done
done
