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
PYTHON_BIN=${CATAI_PYTHON_BIN:-python}
CONDITIONAL_DATA_ROOT=${CATAI_CONDITIONAL1_DATA_ROOT:-$CONDITIONAL_ROOT}
SMOKE_TEST=${CATAI_SMOKE_TEST:-0}
active_round=1
SEED=${CATAI_SEED:-3407}

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

if [[ "$SMOKE_TEST" == "1" ]]; then
    RESULTS_DIR="$CONDITIONAL_ROOT/results/infer+photocat+conditional${active_round}+epoch${epoch}"
    RESULTS_DIR="$CONDITIONAL_ROOT/results/infer+photocat+conditional${active_round}+epoch${epoch}-seed${SEED}"
    mkdir -p "$RESULTS_DIR"
    CATAI_RESULTS_DIR="$RESULTS_DIR" CATAI_CONDITIONAL_ROOT="$CONDITIONAL_ROOT" "$PYTHON_BIN" - <<'PY'
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(os.environ["CATAI_CONDITIONAL_ROOT"])
results_dir = Path(os.environ["CATAI_RESULTS_DIR"])
matched = pd.read_csv(root / "matched_design_space.csv")
label = pd.read_csv(root / "label_data.csv")
merge_cols = ["base", "solvent", "nm", "cat"]
label_mean = float(label["YIELD"].mean()) if not label.empty else 0.0
known = label.groupby(merge_cols, dropna=False)["YIELD"].mean().reset_index()
df = matched.merge(known, on=merge_cols, how="left", suffixes=("", "_known"))

def stable_score(row):
    key = f"{row['base']}|{row['solvent']}|{row['nm']}|{row['cat']}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF

scores = []
uncertainties = []
for _, row in df.iterrows():
    base = row["YIELD_known"] if not pd.isna(row["YIELD_known"]) else label_mean
    noise = stable_score(row)
    scores.append(base + 8.0 * (noise - 0.5))
    uncertainties.append(0.05 + 0.95 * (1.0 - abs(noise - 0.5) * 2.0))

np.savez(
    results_dir / "ocp_test_predictions.npz",
    ids=df["ID"].to_numpy(),
    energy=np.asarray(scores, dtype=float),
    energy_std=np.asarray(uncertainties, dtype=float),
)
PY
    echo "Condition 1 smoke infer completed: $RESULTS_DIR/ocp_test_predictions.npz"
    exit 0
fi


# GPU configs
# devices=(1)
n_gpus=1
# ports=(1331)
bsz=${CATAI_CONDITIONAL1_BATCH_SIZE:-4}

# tasks configs
few_shot_datapath="$CONDITIONAL_DATA_ROOT"


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


