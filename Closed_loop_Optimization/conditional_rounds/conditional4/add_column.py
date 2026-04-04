import pandas as pd

# 读取两个 CSV 文件
design_space = pd.read_csv("design_space.csv")  # 无yield的设计空间
label_data = pd.read_csv("label_data.csv")  # 含yield的标签数据

# 定义匹配列（需确保两表中这些列的取值完全一致才能匹配）
match_columns = ["base", "solvent", "nm", "cat"]

# 筛选 label_data 中 YIELD > 50 的行
high_yield = label_data[label_data["YIELD"] < 100]

# 对 design_space 进行左连接，添加 temperature 和 YIELD 列
design_space = design_space.merge(
    high_yield[match_columns + ["YIELD"]],
    on=match_columns,
    how="left"
)

print(f"YIELD有值的行数: {len(design_space[design_space['YIELD'].notna()])}")

# 保存结果到新 CSV
design_space.to_csv("matched_design_space.csv", index=False)

print("匹配完成，结果保存至 matched_design_space.csv")