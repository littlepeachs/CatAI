# conditional1 目录说明

`conditional1` 是闭环优化的第一轮，作用是从初始标签集出发完成冷启动推荐，给后续 `conditional2` 提供第一批模型驱动候选。

## 本轮角色

- 输入：初始 `label_data.csv`、全条件空间 `design_space.csv`、结构文件与模块筛选结果。
- 目标：训练第一轮推荐模型，并在 `infer/0` 上给出高产率候选。
- 输出：checkpoint、`ocp_test_predictions.npz`、top-k 候选 CSV、`final_rec_yield_1.csv`。

## 关键脚本

- `create_design_space.py`：生成第一轮候选条件空间。
- `select_row.py`：按 `YIELD < 50` 过滤 `label_data.csv`，生成 `filtered_label_data.csv`。
- `add_column.py`：把已有标签回填到 `design_space.csv`，生成 `matched_design_space.csv`。
- `create_lmdb.py`：从 `matched_design_space.csv` 写出：
  - `train/0/data.lmdb`
  - `val/0/data.lmdb`
  - `infer/0/data.lmdb`
- `stage2_recommend.sh`：训练第一轮模型。
- `stage3_infer.sh`：对 `infer/0` 推理。
- `results/rank_photocat.py`：从预测结果中导出 top-k 候选。
- `post_process.py`：检查推荐候选是否与现有标签重合。

## 目录内建议流程

1. 运行 `create_design_space.py` 生成 `design_space.csv`。
2. 准备或更新 `label_data.csv`。
3. 运行 `select_row.py` 生成 `filtered_label_data.csv`。
4. 运行 `add_column.py` 生成 `matched_design_space.csv`。
5. 运行 `create_lmdb.py` 准备 `train/0`、`val/0`、`infer/0`。
6. 运行 `stage2_recommend.sh` 开始训练。
7. 训练完成后运行 `stage3_infer.sh <device> <port> <ckpt> <epoch>`。
8. 在 `results/` 中运行 `rank_photocat.py`，导出高产率候选。
9. 将 `final_rec_yield_1.csv` 或 top-k 候选带入下一轮 `conditional2`。

## 主要文件

- `label_data.csv`：第一轮已知实验标签。
- `all_label_data.csv`：用于做交集检查的全量标签集合。
- `filtered_label_data.csv`：经阈值过滤后保留的训练标签。
- `matched_design_space.csv`：回填标签后的条件空间。
- `final_rec_yield_1.csv`：第一轮主要推荐结果。
- `results/`：推理结果、排名 CSV 和分布图。

## 运行入口

顶层推荐通过仓库根目录调用：

```bash
bash run_conditional1.sh train
bash run_conditional1.sh infer <checkpoint_path> <epoch> [device] [port]
```

如果你需要看四轮如何连成闭环，优先参考仓库根目录的 `Closed_loop_Optimization/README.md`。
