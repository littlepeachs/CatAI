#!/bin/bash
set -euo pipefail

# system configs
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDITIONAL_ROOT=$(cd "$SCRIPT_DIR" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
CONDITIONAL_NAME=$(basename "$CONDITIONAL_ROOT")
active_round=${CONDITIONAL_NAME#conditional}
PYTHON_BIN=${CATAI_PYTHON_BIN:-python}
few_shot_datapath=${CATAI_CONDITIONAL4_DATA_ROOT:-$CONDITIONAL_ROOT}

mkdir -p "$CONDITIONAL_ROOT/output_logs"

# Accept three parameters from command line arguments
if [ $# -ne 4 ]; then
    echo "Error: Exactly 4 parameters are required."
    echo "Usage: $0 <parameter1> <parameter2> <parameter3> <parameter4>"
    exit 1
fi

device="$1"
port="$2"
ckpt_path="$3"
epoch="$4"

echo "Parameter 1: $device"
echo "Parameter 2: $port"
echo "Parameter 3: $ckpt_path"
echo "Parameter 4: $epoch"


# GPU configs
n_gpus=1
bsz=4


splits=("0")

for data_name in {'photocat',}; do # ,'FG-GMAE','OCD-GMAE','desorption-oc20','dissociation-oc20',"transfer-oc20"
for i in "${!splits[@]}"; do
split=${splits[$i]}
# port=${ports[$i]}
# device=${devices[$i]}
echo "######################### ${data_name} + ${splits} #########################"
# ckpt_path="/path/to/catai_active/conditional${active_round}/checkpoints/caiai+photocat+split0"

dict_mean=$($PYTHON_BIN -c "import json; from pathlib import Path; data_name='$data_name'; root=Path('$PROJECT_ROOT'); print(json.load(open(root / 'json' / 'stage2_mean.json'))[data_name])")
dict_std=$($PYTHON_BIN -c "import json; from pathlib import Path; data_name='$data_name'; root=Path('$PROJECT_ROOT'); print(json.load(open(root / 'json' / 'stage2_std.json'))[data_name])")

echo "mean of $data_name is $dict_mean"
echo "std of $data_name is $dict_std"
# 获取当前日期和时间，精确到分钟
current_datetime=$(date '+%m-%d-%H-%M')
echo "当前时间: $current_datetime"


WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=$device $PYTHON_BIN -m torch.distributed.run --nproc_per_node=$n_gpus --master_port=$port "$PROJECT_ROOT/main.py" \
        --num-gpus $n_gpus --distributed \
        --mode predict \
        --task.data_name=$data_name \
    --config-yml "$PROJECT_ROOT/configs/stage2_base.yml" \
        --dataset.train.config.src="$few_shot_datapath/train/$split/" \
        --dataset.val[0].config.src="$few_shot_datapath/val/$split/" \
    --dataset.test[0].config.src="$few_shot_datapath/infer/0" \
        --dataset.normalization.y.mean=${dict_mean[$data_name]} \
        --dataset.normalization.y.std=${dict_std[$data_name]} \
    --identifier catai_infer+${data_name}+conditional${active_round}+epoch${epoch} \
    --timestamp-id catai_infer+${data_name}+conditional${active_round}+epoch${epoch} \
        --task.base_checkpoint.src="$ckpt_path" \
        --optim.batch_size=$bsz \
        --optim.eval_batch_size=$bsz \
    --checkpoint_dir="$CONDITIONAL_ROOT" \
    > "$CONDITIONAL_ROOT/output_logs/catai_infer+${data_name}+conditional${active_round}+epoch${epoch}.log" 2>&1

done
wait
done


