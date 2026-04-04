#!/bin/bash
set -euo pipefail

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
PYTHON_BIN=${CATAI_PYTHON_BIN:-python}
FEW_SHOT_DATA_ROOT=${CATAI_CLOSED_LOOP_FEW_SHOT_DATA_ROOT:-$PROJECT_ROOT/../catai_data/catalyst_task}
MODULE_BASE_PATH=${CATAI_MODULE_HUB_PATH}
CKPT_NAME=${CATAI_MODULE_CHECKPOINT_NAME:-best_checkpoint}

mkdir -p "$PROJECT_ROOT/output_logs/select"

# GPU configs
devices="7"
n_gpus=1

port=1236

seeds=("0")

module_names=(
    "mp_eform"
    # "mp_bandgap"
    # "mp_energy_above_hull"
    # "mp_efermi"
    # "n_th_cond"
    # "p_e_cond"
    # "n_e_cond"
    # "p_th_cond"
    # "mp_gvrh"
    # "mp_kvrh"
    # "castelli_eform"
    # "jarvis_gvrh"
    # "jarvis_kvrh"
    # "jarvis_bandgap"
    # "jarvis_eform"
    # "jarvis_total_energy"
    # "adsorption_energy"
    # "surface_energy"
    # "Ea_log_wo_cell"
    # "gibbs_free_energy"
    # "Electronic_E"
    # "Metal_q"
    # "HL_Gap"
    # "dos_fermi_level"
    # "oqmd_delta_e"
    # "oqmd_stability"
    # "oqmd_band_gap"
    # "qmof_bg"
    # "qmof_e"
    # "cepdb_pce"
)

for data_name in {'photocat',}; do
for seed in ${seeds[@]}; do
# for data_name in {'jarvis_2d_gap_opt',}; do
# for seed in {'0',}; do
for module_name in ${module_names[@]}; do
    dict_mean=$($PYTHON_BIN -c "import json; from pathlib import Path; data_name='$data_name'; root=Path('$PROJECT_ROOT'); print(json.load(open(root / 'json' / 'stage2_mean.json'))[data_name])")
    dict_std=$($PYTHON_BIN -c "import json; from pathlib import Path; data_name='$data_name'; root=Path('$PROJECT_ROOT'); print(json.load(open(root / 'json' / 'stage2_std.json'))[data_name])")
        echo "Running $data_name seed $seed"
    WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$devices $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$n_gpus --master_port=$port "$PROJECT_ROOT/main.py" \
                --num-gpus $n_gpus --distributed \
                --mode train \
                --task.data_name="$data_name" \
                --task.split=$seed \
                --task.flag_batchsize=8 \
        --config-yml "$PROJECT_ROOT/configs/stage2_select_modules.yml" \
        --dataset.train.config.src="$FEW_SHOT_DATA_ROOT/$data_name/$seed/train" \
        --dataset.val[0].config.src="$FEW_SHOT_DATA_ROOT/$data_name/$seed/val" \
        --dataset.test[0].config.src="$FEW_SHOT_DATA_ROOT/$data_name/$seed/val" \
                --task.load_module_name=$module_name \
        --task.module_embeddings_save_path="$PROJECT_ROOT/module_selection_outputs/catai_embeddings" \
        --task.labels_save_path="$PROJECT_ROOT/module_selection_outputs/catai_labels" \
                --dataset.normalization.y.mean=$dict_mean \
                --dataset.normalization.y.std=$dict_std \
        --identifier catai-module-select+$data_name+seed$seed \
        --checkpoint_dir "$PROJECT_ROOT" \
        --task.base_checkpoint.src="${MODULE_BASE_PATH}/${module_name}/${CKPT_NAME}.pt" \
        > "$PROJECT_ROOT/output_logs/select/catai-module-select+$data_name+$module_name+seed$seed.log" 2>&1

done
done
done

# /data/oyyw/miniforge3/envs/catai/bin/python
# python catai_performance_evaluate.py  # knn
# python catai_weight_optimize.py  # module weight optimization