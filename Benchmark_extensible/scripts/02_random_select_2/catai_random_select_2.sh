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

PYTHON_BIN="$CATAI_EXTENSIBLE_PYTHON_BIN"
DATA_ROOT="$CATAI_EXTENSIBLE_DATA_ROOT"
MODULE_BASE_PATH="$CATAI_EXTENSIBLE_RANDOM_HUB_PATH"
BASE_CHECKPOINT="$CATAI_EXTENSIBLE_BASE_CHECKPOINT_PATH"

device=${1:-0}
port=${2:-1300}
n_gpus=1
bsz=4
early_stop_patience=10
epochs=60
splits=("0")

selected_module_names="'oqmd_stability','oqmd_band_gap','p_e_cond','n_e_cond','castelli_eform'"
weights="0.2,0.2,0.2,0.2,0.2"

for data_name in {'Alloy-GMAE',}; do
for split in ${splits[@]}; do
    dict_mean=$($PYTHON_BIN -c "import json; data_name='$data_name'; print(json.load(open('$BENCHMARK_ROOT/json/stage2_mean.json'))[data_name])")
    dict_std=$($PYTHON_BIN -c "import json; data_name='$data_name'; print(json.load(open('$BENCHMARK_ROOT/json/stage2_std.json'))[data_name])")
    current_datetime=$(date '+%m-%d-%H-%M')

    WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$device $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$n_gpus --master_port=$port "$BENCHMARK_ROOT/main.py" \
        --num-gpus $n_gpus --distributed \
        --mode train \
        --task.data_name=$data_name \
        --config-yml "$BENCHMARK_ROOT/configs/stage2_base.yml" \
        --dataset.train.config.src="$DATA_ROOT/$data_name/$split/train" \
        --dataset.val[0].config.src="$DATA_ROOT/$data_name/$split/val" \
        --dataset.test[0].config.src="$DATA_ROOT/$data_name/$split/test" \
        --task.expert_select=False \
        --task.load_experts=False \
        --task.load_full_module=True \
        --task.selected_modules="[$selected_module_names]" \
        --task.selected_module_weights="$weights" \
        --task.use_module_weights=True \
        --task.module_base_path="$MODULE_BASE_PATH" \
        --dataset.normalization.y.mean=$dict_mean \
        --dataset.normalization.y.std=$dict_std \
        --optim.batch_size=$bsz \
        --optim.eval_batch_size=$bsz \
        --optim.early_stop_patience=$early_stop_patience \
        --optim.max_epochs=$epochs \
        --model.edge_dropout=0.0 \
        --model.dropout=0.0 \
        --identifier catai+$data_name+split$split+random_select_2 \
        --timestamp-id catai+$data_name+split$split+random_select_2 \
        --task.base_checkpoint.src="$BASE_CHECKPOINT"
done
done
