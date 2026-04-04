#!/bin/bash
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCREENING_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
source "$SCREENING_ROOT/run_common.sh"
PYTHON_BIN="$CATAI_PYTHON_BIN"
export PYTHONPATH="$CATAI_SCREENING_OCP_ROOT:${SCREENING_ROOT}:${PYTHONPATH:-}"

# GPU configs
devices="7"
n_gpus=1

base_port=1236
port=$base_port
poe_lmdb_root="$CATAI_DATA_ROOT"
checkpoint_base_path="$CATAI_MODULE_HUB_PATH"
ckpt_name='best_checkpoint'
split_label="0"
module_embeddings_dir="$SCREENING_ROOT/module_selection_outputs/catai_embeddings"
labels_dir="$SCREENING_ROOT/module_selection_outputs/catai_labels"

mkdir -p "$module_embeddings_dir"
mkdir -p "$labels_dir"

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

for data_name in {'POE_lmdb',}; do
for module_name in ${module_names[@]}; do
    port=$((port + 1))
    dict_mean=$(python3 -c "import json; data_name='$data_name'; print(json.load(open('$SCREENING_ROOT/json/stage2_mean.json'))[data_name])")
    dict_std=$(python3 -c "import json; data_name='$data_name'; print(json.load(open('$SCREENING_ROOT/json/stage2_std.json'))[data_name])")
    echo "Running $data_name split $split_label with module $module_name on port $port"
    WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$devices $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$n_gpus --master_port=$port "$SCREENING_ROOT/main.py" \
                --num-gpus $n_gpus --distributed \
                --mode train \
                --run-dir "$SCREENING_ROOT" \
                --task.data_name="$data_name" \
                --task.split=$split_label \
                --task.flag_batchsize=8 \
                --config-yml "$SCREENING_ROOT/configs/catai_screening_module_select.yml" \
                --dataset.train.config.src="$poe_lmdb_root/train" \
                --dataset.val[0].config.src="$poe_lmdb_root/val" \
                --dataset.test[0].config.src="$poe_lmdb_root/test" \
                --task.load_module_name=$module_name \
                --task.module_embeddings_save_path="$module_embeddings_dir" \
                --task.labels_save_path="$labels_dir" \
                --dataset.normalization.y.mean=$dict_mean \
                --dataset.normalization.y.std=$dict_std \
                --identifier catai-screening-module-select+$data_name+split$split_label \
                --task.base_checkpoint.src="${checkpoint_base_path}/${module_name}/$ckpt_name.pt"

done
done