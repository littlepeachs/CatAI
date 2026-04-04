import pandas as pd
import numpy as np
import os
import lmdb
import pickle
import torch
from torch_geometric.data import Data
from pymatgen.core import Structure

# 读取sampled_indices.txt文件
with open('sampled_indices.txt', 'r') as f:
    sampled_indices = list(map(int, f.read().strip().split(',')))
# 读取df_all.csv文件
df_all = pd.read_csv('df_all.csv')

# 根据sampled_indices获取对应的数据行
selected_data = df_all.iloc[sampled_indices]
# 创建数据列表
all_data = []
for idx, row in selected_data.iterrows():
    try:
        # 读取CIF文件
        cif_path = row['Crystals']  # 假设csv中有cif_path列
        structure = Structure.from_file("./structures/" + cif_path)
        
        # 提取结构信息
        lattice_vectors = structure.lattice.matrix
        coordinates = structure.cart_coords
        atom_numbers = [site.specie.Z for site in structure]
        energy = row['y_relax']  # 假设csv中有energy列
        
        all_data.append({
            'ID': cif_path,
            'atom_numbers': np.array(atom_numbers),
            'coordinates': np.array(coordinates),
            'cell': np.array(lattice_vectors),
            'energy': float(energy)
        })
    except Exception as e:
        print(f"处理CIF文件 {cif_path} 时出错: {e}")
        continue

print(f"成功读取 {len(all_data)} 条数据")

# 设置基础目录
base_dir = "./Alloy_C_lmdb"
os.makedirs(base_dir, exist_ok=True)

# 创建并写入LMDB数据库（训练、验证、测试使用相同的数据集）
for split_name in ["train", "val", "test"]:
    split_dir = os.path.join(base_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # 创建data.lmdb文件路径
    output_path = os.path.join(split_dir, "data.lmdb")
    
    # 如果文件已存在，先删除
    if os.path.exists(output_path):
        os.remove(output_path)
    
    # 创建新的LMDB数据库
    split_env = lmdb.open(output_path, map_size=1099511627776, subdir=False)  # 1TB
    
    cnt = 0
    with split_env.begin(write=True) as txn:
        for data in all_data:
            pos = np.array(data['coordinates'], dtype=np.float32)
            atomic_numbers = np.array(data['atom_numbers'], dtype=np.float32)
            natoms = data['atom_numbers'].shape[0]
            cell = np.array(data['cell'], dtype=np.float32)

            tags = np.full((natoms,), 2, dtype=np.float32)
            fixed = torch.zeros(natoms).type(torch.float32)
            pos = torch.tensor(pos)
            atomic_numbers = torch.tensor(atomic_numbers)
            natoms = int(natoms)
            tags = torch.tensor(tags)
            y = torch.tensor(data['energy'])
            y_relaxed = torch.tensor(data['energy'])
            cell = torch.tensor(cell)
            graph_data = Data(pos=pos, 
                        atomic_numbers=atomic_numbers, 
                        natoms=natoms, 
                        tags=tags, 
                        fixed=fixed, 
                        y=data['energy'],
                        y_relaxed=data['energy'],
                        mat_index=data['ID'],
                        cell=cell.unsqueeze(0)
                    )
            
            serialized_data = pickle.dumps(graph_data)
            txn.put(str(cnt).encode(), serialized_data)
            cnt += 1
    
    split_env.close()
    print(f"已创建 {split_name}/data.lmdb，包含 {len(all_data)} 条数据")

print("LMDB数据库创建完成！训练、验证、测试集使用相同的数据。")
