# conditional3 目录说明

`conditional3` 是闭环优化的第三轮。从这一轮开始，流程不再只看高产率，也开始显式利用高不确定性候选和参考产率信息。

## 本轮角色

- 输入：前两轮累计标签、第三轮条件空间、参考实验信息 `refer.csv`。
- 目标：同时生成高产率候选和高不确定性候选，并在全设计空间上导出 embedding / 预测结果。
- 输出：
  - `final_rec_yield.csv`
  - `final_rec_uncertainty.csv`
  - `final_rec_yield_with_refer.csv`
  - `final_rec_uncertainty_with_refer.csv`
  - `all/data.lmdb`

## 关键脚本

- `create_design_space.py`：生成第三轮条件空间。
- `select_row.py`：按 `YIELD < 100` 过滤标签。
- `add_column.py`：生成 `matched_design_space.csv`。
- `create_lmdb.py`：写出 `train/0`、`val/0`、`infer/0`。
- `create_all_lmdb.py`：把整张 `matched_design_space.csv` 写成 `all/data.lmdb`，供全量 embedding / 预测使用。
- `stage2_recommend.sh`：训练第三轮模型。
- `stage3_infer.sh`：对 `infer/0` 推理。
- `stage3_get_ebd.sh`：对 `all/0` 推理，输出全设计空间 embedding / 预测结果。
- `fill_refer_yield.py`：把 `refer.csv` 中同条件组合的参考产率补进推荐结果。
- `post_process.py`：检查最终推荐样本与历史标签的交集。

## 目录内建议流程

1. 准备 `label_data.csv`、`design_space.csv`、`refer.csv`。
2. 运行 `select_row.py` 生成 `filtered_label_data.csv`。
3. 运行 `add_column.py` 生成 `matched_design_space.csv`。
4. 运行 `create_lmdb.py` 写出训练、验证、推理 LMDB。
5. 如需做全量 embedding，运行 `create_all_lmdb.py` 写出 `all/data.lmdb`。
6. 运行 `stage2_recommend.sh` 训练。
7. 运行 `stage3_infer.sh <device> <port> <ckpt> <epoch>` 生成本轮候选。
8. 运行 `fill_refer_yield.py` 生成 `final_rec_*_with_refer.csv`。
9. 如需对全设计空间评分，再运行 `stage3_get_ebd.sh <device> <port> <ckpt> <epoch>`。
10. 将 `final_rec_uncertainty_with_refer.csv` 或 `final_rec_yield_with_refer.csv` 带入 `conditional4`。

## 主要文件

- `refer.csv`：历史参考产率表。
- `final_rec_yield.csv`：高产率候选。
- `final_rec_uncertainty.csv`：高不确定性候选。
- `final_rec_yield_with_refer.csv`：补齐参考产率后的高产率候选。
- `final_rec_uncertainty_with_refer.csv`：补齐参考产率后的高不确定性候选。
- `all/data.lmdb`：全设计空间 LMDB。

## 运行入口

```bash
bash run_conditional3.sh train
bash run_conditional3.sh infer <checkpoint_path> <epoch> [device] [port]
bash run_conditional3.sh ebd <checkpoint_path> <epoch> [device] [port]
```

更高层的四轮衔接说明见仓库根目录 `Closed_loop_Optimization/README.md`。
