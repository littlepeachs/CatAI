# conditional4 目录说明

`conditional4` 是闭环优化的最终轮次，负责整合第三轮推荐、参考产率补充和最终标签回灌，形成闭环收敛结果。

## 本轮角色

- 输入：第三轮输出的高产率 / 高不确定性候选及其参考产率补充信息。
- 目标：完成最终一轮训练与推理，并把可用参考结果追加回标签集。
- 输出：
  - `final_rec_yield.csv`
  - `final_rec_uncertainty.csv`
  - `final_rec_yield_with_refer.csv`
  - `final_rec_uncertainty_with_refer.csv`
  - 更新后的 `label_data.csv`

## 关键脚本

- `create_design_space.py`：生成第四轮条件空间。
- `select_row.py`：按 `YIELD < 100` 过滤标签。
- `add_column.py`：生成 `matched_design_space.csv`。
- `create_lmdb.py`：写出 `train/0`、`val/0`、`infer/0`。
- `create_all_lmdb.py`：写出全设计空间 `all/data.lmdb`。
- `stage2_recommend.sh`：训练最终一轮模型。
- `stage3_infer.sh`：对 `infer/0` 推理。
- `stage3_get_ebd.sh`：对 `all/0` 导出全量 embedding / 全量预测。
- `append_refer_to_label.py`：把 `final_rec_*_with_refer.csv` 中可用参考产率追加回 `label_data.csv`。
- `post_process.py`：检查最终推荐与历史标签重叠情况。

## 目录内建议流程

1. 准备本轮 `label_data.csv` 与 `design_space.csv`。
2. 运行 `select_row.py` 生成 `filtered_label_data.csv`。
3. 运行 `add_column.py` 生成 `matched_design_space.csv`。
4. 运行 `create_lmdb.py` 和 `create_all_lmdb.py`。
5. 运行 `stage2_recommend.sh` 训练。
6. 运行 `stage3_infer.sh <device> <port> <ckpt> <epoch>` 生成最终候选。
7. 如需全设计空间分析，运行 `stage3_get_ebd.sh <device> <port> <ckpt> <epoch>`。
8. 确认 `final_rec_*_with_refer.csv` 后，运行 `append_refer_to_label.py` 把参考产率追加回 `label_data.csv`。
9. 使用更新后的标签集作为闭环终态数据。

## 主要文件

- `final_rec_yield.csv`：最终高产率候选。
- `final_rec_uncertainty.csv`：最终高不确定性候选。
- `final_rec_yield_with_refer.csv`：带参考产率补充的高产率候选。
- `final_rec_uncertainty_with_refer.csv`：带参考产率补充的高不确定性候选。
- `label_data.csv`：最终会被 `append_refer_to_label.py` 更新。
- `results/`：最终推理与排序结果。

## 运行入口

```bash
bash run_conditional4.sh train
bash run_conditional4.sh infer <checkpoint_path> <epoch> [device] [port]
bash run_conditional4.sh ebd <checkpoint_path> <epoch> [device] [port]
```

四轮全局上下游关系见仓库根目录 `Closed_loop_Optimization/README.md`。
