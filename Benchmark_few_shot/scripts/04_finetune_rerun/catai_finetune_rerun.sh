#!/bin/bash
# system configs
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCHMARK_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
export PYTHONPATH="$BENCHMARK_ROOT:${PYTHONPATH}"
source "$BENCHMARK_ROOT/run_common.sh"

PYTHON_BIN="$CATAI_PYTHON_BIN"

mkdir -p "$BENCHMARK_ROOT/output_logs"

# GPU configs
device="0"
n_gpus=1
base_port=1298

# tasks configs
few_shot_datapath="$CATAI_DATA_ROOT"
wandb_mode=dryrun

# modules=('mp_eform' 'mp_bandgap' 'n_Seebeck' 'n_avg_eff_mass' 'n_th_cond' 'p_Seebeck' 'p_avg_eff_mass' 'p_e_cond' 'n_e_cond' 'p_th_cond' 'mp_gvrh' 'castelli_eform' 'jarvis_gvrh' 'jarvis_bandgap' 'jarvis_eform' 'jarvis_kvrh' 'mp_kvrh' 'jarvis_dielectric_opt')
modules=(
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

# datasets=('piezoelectric_tensor' 'expt_eform' 'mp_poly_total' 'mp_poisson_ratio' 'jarvis_2d_dielectric_opt' 'jarvis_2d_eform' 'jarvis_2d_exfoliation' 'jarvis_2d_gap_opt' 'mp_elastic_anisotropy' 'mp_eps_electronic' 'mp_eps_total' 'mp_phonons' 'mp_poly_electronic' 'expt_bandgap' 'jarvis_3d_eps_tbmbj' 'jarvis_3d_gap_tbmbj' 'mp_dielectric')
# ckpt_name='best_checkpoint'

bsz=4
K=32
early_stop_patience=20
epochs=60

splits=("0")

# for data_name in {'piezoelectric_tensor','expt_eform','mp_poly_total','mp_poisson_ratio','jarvis_2d_dielectric_opt','jarvis_2d_eform','jarvis_2d_exfoliation','jarvis_2d_gap_opt','mp_elastic_anisotropy','mp_eps_electronic','mp_eps_total','mp_phonons','mp_poly_electronic','expt_bandgap','jarvis_3d_eps_tbmbj','jarvis_3d_gap_tbmbj','mp_dielectric'}; do
for data_name in {'Alloy-GMAE-k32',}; do
    port=$base_port
    for split in ${splits[@]}; do
        echo $data_name $split
        # Read the data from json file
        dict_mean=$(python3 -c "import json; data_name='$data_name'; print(json.load(open('$BENCHMARK_ROOT/json/stage2_mean.json'))[data_name])")
        dict_std=$(python3 -c "import json; data_name='$data_name'; print(json.load(open('$BENCHMARK_ROOT/json/stage2_std.json'))[data_name])")
        dict_select_modules=$(python3 -c "import json; data_name='$data_name'; mysplit='$split'; print(json.load(open('$BENCHMARK_ROOT/json/catai_module_ids/split_' + mysplit + '_' + data_name + '.json'))[data_name])")
        dict_num_experts=$(python3 -c "import json; data_name='$data_name'; mysplit='$split'; print(json.load(open('$BENCHMARK_ROOT/json/catai_module_num/split_' + mysplit + '_' + data_name + '.json'))[data_name])")
        weights=$(python3 -c "import json; data_name='$data_name'; mysplit='$split'; data=json.load(open('$BENCHMARK_ROOT/json/catai_module_weights/split_' + mysplit + '_' + data_name + '.json')); print(','.join(map(str, data[data_name])))")

        # print the data
        echo "mean of $data_name is $dict_mean"
        echo "std of $data_name is $dict_std"
        echo "selected experts for $data_name is $dict_select_modules"
        echo "num of experts for $data_name is $dict_num_experts"
        echo "weights for selected experts is $weights"

        IFS=',' read -r -a selected_modules_array <<< "$dict_select_modules"

        selected_module_names=()

        for selected_module in ${selected_modules_array[@]}; do
            echo "selected_module: ${modules[$selected_module]}"
            selected_module_names+="'${modules[$selected_module]}',"
        done

        echo [$selected_module_names]
        current_datetime=$(date '+%m-%d-%H-%M')
        echo "当前时间: $current_datetime"
        # stage 2 with pre-trained modules
        current_device=$device
        echo "Running split $split on device $current_device with port $port"
        WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$current_device $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$n_gpus --master_port=$port "$BENCHMARK_ROOT/main.py" \
            --num-gpus $n_gpus --distributed \
            --mode train \
            --task.data_name=$data_name \
            --config-yml "$BENCHMARK_ROOT/configs/catai_base_few_shot.yml" \
            --dataset.train.config.src="$few_shot_datapath/$data_name/$split/train" \
            --dataset.val[0].config.src="$few_shot_datapath/$data_name/$split/val" \
            --dataset.test[0].config.src="$few_shot_datapath/$data_name/$split/test" \
            --task.expert_select=False \
            --task.load_experts=False \
            --task.load_full_module=True \
            --task.selected_modules="[$selected_module_names]" \
            --task.selected_module_weights=$weights \
            --task.use_module_weights=True \
            --task.module_base_path="$CATAI_MODULE_HUB_PATH" \
            --dataset.normalization.y.mean=$dict_mean \
            --dataset.normalization.y.std=$dict_std \
            --optim.batch_size=$bsz \
            --optim.eval_batch_size=$bsz \
            --optim.early_stop_patience=$early_stop_patience \
            --optim.max_epochs=$epochs \
            --model.edge_dropout=0.0 \
            --model.dropout=0.0 \
            --identifier catai+$data_name+split$split \
            --timestamp-id catai+$data_name+split$split \
            --task.base_checkpoint.src="$CATAI_BASE_CHECKPOINT_PATH"
        ((port++))
    done
    wait
done