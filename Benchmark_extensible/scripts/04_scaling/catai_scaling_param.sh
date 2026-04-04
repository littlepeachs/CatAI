#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <scale> [device] [port]"
    echo "  scale: 5 | 10 | 15 | 20 | 25"
    exit 1
fi

SCALE="$1"
if [[ "$SCALE" != "5" && "$SCALE" != "10" && "$SCALE" != "15" && "$SCALE" != "20" && "$SCALE" != "25" ]]; then
    echo "Error: scale must be one of 5, 10, 15, 20, 25"
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$BENCHMARK_ROOT/run_common.sh"

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

device=${2:-0}
port=${3:-1301}

PYTHON_BIN="$CATAI_EXTENSIBLE_PYTHON_BIN"
DATA_ROOT="$CATAI_EXTENSIBLE_DATA_ROOT"
MODULE_BASE_PATH="$CATAI_EXTENSIBLE_MODULE_HUB_PATH"
BASE_CHECKPOINT="$CATAI_EXTENSIBLE_BASE_CHECKPOINT_PATH"

all_modules=(
    "p_e_cond"
    "HL_Gap"
    "qmof_bg"
    "n_e_cond"
    "cepdb_pce"
    "qmof_e"
    "dos_fermi_level"
    "oqmd_delta_e"
    "oqmd_stability"
    "oqmd_band_gap"
    "surface_energy"
    "Ea_log_wo_cell"
    "gibbs_free_energy"
    "Electronic_E"
    "Metal_q"
    "castelli_eform"
    "jarvis_bandgap"
    "jarvis_eform"
    "jarvis_total_energy"
    "adsorption_energy"
    "mp_efermi"
    "n_th_cond"
    "mp_energy_above_hull"
    "mp_bandgap"
    "p_th_cond"
)
modules=("${all_modules[@]:0:$SCALE}")

n_gpus=1
bsz=4
early_stop_patience=15
epochs=60
splits=("0")

for data_name in {'Alloy-GMAE',}; do
for split in ${splits[@]}; do
    dict_mean=$($PYTHON_BIN -c "import json; data_name='$data_name'; print(json.load(open('$BENCHMARK_ROOT/json/stage2_mean.json'))[data_name])")
    dict_std=$($PYTHON_BIN -c "import json; data_name='$data_name'; print(json.load(open('$BENCHMARK_ROOT/json/stage2_std.json'))[data_name])")
    dict_select_modules=$($PYTHON_BIN -c "import json; data_name='$data_name'; mysplit='$split'; print(json.load(open('$BENCHMARK_ROOT/json/scaling_${SCALE}_module_ids/split_' + mysplit + '_' + data_name + '.json'))[data_name])")
    weights=$($PYTHON_BIN -c "import json; data_name='$data_name'; mysplit='$split'; data=json.load(open('$BENCHMARK_ROOT/json/scaling_${SCALE}_module_weights/split_' + mysplit + '_' + data_name + '.json')); print(','.join(map(str, data[data_name])))")
    IFS=',' read -r -a selected_modules_array <<< "$dict_select_modules"
    selected_module_names=""
    for selected_module in ${selected_modules_array[@]}; do
        selected_module_names+="'${modules[$selected_module]}',"
    done
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
        --identifier catai_scaling_${SCALE}+$data_name+split$split \
        --timestamp-id catai_scaling_${SCALE}+$data_name+split$split \
        --task.base_checkpoint.src="$BASE_CHECKPOINT"
done
done
