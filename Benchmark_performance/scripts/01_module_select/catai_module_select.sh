#!/bin/bash
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PROJECT_ROOT=$(cd "$BENCHMARK_ROOT/.." && pwd)
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH}"
source "$BENCHMARK_ROOT/run_common.sh"

PYTHON_BIN="$CATAI_PYTHON_BIN"

# GPU configs
device="0"
n_gpus=1

base_port=2049
few_shot_datapath="$CATAI_DATA_ROOT"
checkpoint_base_path="$CATAI_MODULE_HUB_PATH"
ckpt_name='best_checkpoint'

seeds=("0")

module_names=(
    "mp_eform"
    "mp_bandgap"
    "mp_energy_above_hull"
    "mp_efermi"
    "n_th_cond"
    "p_e_cond"
    "n_e_cond"
    "p_th_cond"
    "mp_gvrh"
    "mp_kvrh"
    "castelli_eform"
    "jarvis_gvrh"
    "jarvis_kvrh"
    "jarvis_bandgap"
    "jarvis_eform"
    "jarvis_total_energy"
    "adsorption_energy"
    "surface_energy"
    "Ea_log_wo_cell"
    "gibbs_free_energy"
    "Electronic_E"
    "Metal_q"
    "HL_Gap"
    "dos_fermi_level"
    "oqmd_delta_e"
    "oqmd_stability"
    "oqmd_band_gap"
    "qmof_bg"
    "qmof_e"
    "cepdb_pce"
)

port=$(($base_port))
for data_name in {'Alloy-GMAE',}; do
for module_name in ${module_names[@]}; do
for seed in ${seeds[@]}; do
        port=$((port+1))
        echo "Selecting device $device and port $port for seed $seed"
    dict_mean=$(python3 -c "import json; data_name='$data_name'; print(json.load(open('$BENCHMARK_ROOT/json/stage2_mean.json'))[data_name])")
    dict_std=$(python3 -c "import json; data_name='$data_name'; print(json.load(open('$BENCHMARK_ROOT/json/stage2_std.json'))[data_name])")
        echo "Running $data_name seed $seed"
        WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$device $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$n_gpus --master_port=$port "$BENCHMARK_ROOT/main.py" \
                --num-gpus $n_gpus --distributed \
                --mode train \
                --task.data_name="$data_name" \
                --task.split=$seed \
                --task.flag_batchsize=8 \
        --config-yml "$BENCHMARK_ROOT/configs/catai_module_select.yml" \
                --dataset.train.config.src="$few_shot_datapath/$data_name/$seed/train" \
                --dataset.val[0].config.src="$few_shot_datapath/$data_name/$seed/val" \
                --dataset.test[0].config.src="$few_shot_datapath/$data_name/$seed/test" \
                --task.load_module_name=$module_name \
                --dataset.normalization.y.mean=$dict_mean \
                --dataset.normalization.y.std=$dict_std \
        --identifier catai-module-select+$data_name+seed$seed \
                --task.base_checkpoint.src="${checkpoint_base_path}/${module_name}/$ckpt_name.pt"
done
done
done