#!/bin/bash
# system configs
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
device=1
n_gpus=1
port=1339
bsz=8

# tasks configs
poe_lmdb_root="$CATAI_DATA_ROOT"
ckpt_path="$CATAI_SCREENING_CHECKPOINT_PATH"
ckpt_name="surface_energy"

split_label="0"

for data_name in {'POE_lmdb',}; do # ,'FG-GMAE','OCD-GMAE','desorption-oc20','dissociation-oc20',"transfer-oc20"
echo "######################### ${data_name} + scatter #########################"

dict_mean=$(python3 -c "import json; data_name='$data_name'; print(json.load(open('$SCREENING_ROOT/json/stage2_mean.json'))[data_name])")
dict_std=$(python3 -c "import json; data_name='$data_name'; print(json.load(open('$SCREENING_ROOT/json/stage2_std.json'))[data_name])")

echo "mean of $data_name is $dict_mean"
echo "std of $data_name is $dict_std"
# 获取当前日期和时间，精确到分钟
current_datetime=$(date '+%m-%d-%H-%M')
echo "当前时间: $current_datetime"


WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$device $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$n_gpus --master_port=$port "$SCREENING_ROOT/main.py" \
        --num-gpus $n_gpus --distributed \
        --mode predict \
        --run-dir "$SCREENING_ROOT" \
        --task.data_name=$data_name \
        --config-yml "$SCREENING_ROOT/configs/catai_screening_base.yml" \
        --task.split=$split_label \
        --dataset.train.config.src="$poe_lmdb_root/train" \
        --dataset.val[0].config.src="$poe_lmdb_root/val" \
        --dataset.test[0].config.src="$poe_lmdb_root/test" \
        --dataset.normalization.y.mean=$dict_mean \
        --dataset.normalization.y.std=$dict_std \
        --identifier catai_screening_infer+${data_name}+${split_label}_scatter \
        --task.base_checkpoint.src="$ckpt_path" \
        --optim.batch_size=$bsz \
        --optim.eval_batch_size=$bsz
done


