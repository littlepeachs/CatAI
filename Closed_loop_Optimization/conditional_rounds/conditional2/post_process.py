import pandas as pd

# 读取数据文件
all_label_df = pd.read_csv("all_label_data.csv")
final_recommend_df = pd.read_csv("final_rec_yield.csv")

# 对应字段名可能会有不同，做下标准化以方便匹配
def normalize_columns(df):
    df = df.rename(columns={
        "Base": "base",
        "Solvent": "solvent",
        "NM": "nm",
        "Cat": "cat",
        "Yield_Std": "YIELD_STD"
    })
    # nm 可能是字符串，有可能是int，全部转为str以避免类型带来的问题
    if "nm" in df.columns:
        df["nm"] = df["nm"].astype(str)
    return df

all_label_df = normalize_columns(all_label_df)
final_recommend_df = normalize_columns(final_recommend_df)

# 只需要四个字段
cols = ["base", "solvent", "nm", "cat"]
# 合并，并找交集
merged = pd.merge(
    all_label_df[cols],
    final_recommend_df[cols],
    how="inner",
    on=cols
)

# 输出匹配结果
if not merged.empty:
    print("匹配到的(base,solvent,nm,cat)如下：")
    print(merged)
    
    # 对于all_label_df中的对应的行进行打印
    matched_indices = all_label_df.index[all_label_df[cols].apply(tuple, axis=1).isin(merged[cols].apply(tuple, axis=1))]
    print("\nall_label_df中对应的行：")
    print(all_label_df.loc[matched_indices])
else:
    print("未找到匹配的行。")
