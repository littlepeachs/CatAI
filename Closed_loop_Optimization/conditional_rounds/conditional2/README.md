# conditional2 目录说明

`conditional2` 是闭环优化的第二轮，承接 `conditional1` 推荐出的候选，继续扩展高产率搜索空间。

## 本轮角色

- 输入：上一轮整理后的标签数据、当前轮条件空间、结构文件和模块权重。
- 目标：在更大搜索范围内继续训练和推理，得到更稳定的高产率候选。
- 输出：checkpoint、`ocp_test_predictions.npz`、`final_rec_yield.csv`、`final_rec_uncertainty.csv`。

## 关键脚本

- `create_design_space.py`：生成第二轮条件空间。
- `select_row.py`：按 `YIELD < 100` 过滤标签，生成 `filtered_label_data.csv`。
- `add_column.py`：把标签回填到 `design_space.csv`，生成 `matched_design_space.csv`。
- `create_lmdb.py`：写出 `train/0`、`val/0`、`infer/0` 三套 LMDB。
- `stage2_recommend.sh`：训练第二轮模型。
- `stage3_infer.sh`：在 `infer/0` 上推理。
- `results/rank_photocat.py`：输出高产率和高不确定性候选排行。
- `post_process.py`：检查推荐样本与既有标签的交集。

## 目录内建议流程

1. 准备本轮 `label_data.csv` 和 `design_space.csv`。
2. 运行 `select_row.py` 得到 `filtered_label_data.csv`。
3. 运行 `add_column.py` 得到 `matched_design_space.csv`。
4. 运行 `create_lmdb.py` 写出本轮训练、验证、推理数据。
5. 运行 `stage2_recommend.sh` 训练。
6. 运行 `stage3_infer.sh <device> <port> <ckpt> <epoch>` 推理。
7. 在 `results/` 中执行 `rank_photocat.py` 生成 top-k CSV。
8. 将 `final_rec_yield.csv` 和 `final_rec_uncertainty.csv` 作为进入 `conditional3` 的候选参考。

## 主要文件

- `label_data.csv`：第二轮已知实验标签。
- `filtered_label_data.csv`：过滤后的监督样本。
- `matched_design_space.csv`：回填标签后的条件空间。
- `final_rec_yield.csv`：高产率候选。
- `final_rec_uncertainty.csv`：高不确定性候选。
- `results/`：预测 NPZ、候选排行、可视化结果。

## 运行入口

```bash
bash run_conditional2.sh train
bash run_conditional2.sh infer <checkpoint_path> <epoch> [device] [port]
```

四轮完整闭环关系见仓库根目录 `Closed_loop_Optimization/README.md`。
