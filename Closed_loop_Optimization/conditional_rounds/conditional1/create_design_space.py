import itertools
import csv

# 定义各变量的取值列表
bases = [
    "BTMG",
    "Et3N",
    "4-Methylmorpholine",
    "MeONa",
    "NaOTMS",
    "tBuOK",
    "Cs2CO3",
    "K2CO3",
    "K3PO4",
    "LiOMe",
    "DIPEA"
]
solvents = ["DMF", "Toluene", "CH3CN", "1-4-dioxane", "DCE", "1-3-Xylene", 
            "1-2-3-4-tetrahydronaphthalene", "DMSO", "Anisole", "Cyclohexane"]
nms = [395, 425, 440,465]
catalysts = ["Ni-GACs", "Cu-GACs", "NiC","CuC","NiCN","CuN4","CuN3","Ni-SAC","Pd-CeO2"
 ]

# 生成笛卡尔积（所有组合）
combinations = itertools.product(bases, solvents, nms, catalysts)

# 写入 CSV 文件
with open("design_space.csv", "w", newline="") as csvfile:
    fieldnames = ["ID", "base", "solvent", "nm", "cat"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for idx, (base, solvent, nm, cat) in enumerate(combinations, 1):
        writer.writerow({
            "ID": f"exp-{idx}",
            "base": base,
            "solvent": solvent,
            "nm": nm,
            "cat": cat
        })

print("CSV 文件已生成：experimental_data.csv")