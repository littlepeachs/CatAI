# ============================== Minimal CSV -> Batch ==============================
import os, csv, glob
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from ase.io import read as ase_read
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
import random
import pandas as pd

@dataclass
class CSVSpec:
    csv_path: str                  # CSV 文件
    poscar_root: str               # 结构文件所在目录（用 cat 列做文件名）
    cat_cols: Tuple[str, str, str] = ("base", "solvent", "nm")
    cat_filename_col: str = "cat"  # 用来定位结构的列
    cont_col: Optional[str] = None # 连续列名；None 时自动找以 "temperat" 开头的列
    target_col: Optional[str] = None  # 可选：监督目标，如 "YIELD"

class Vocab:
    def __init__(self): self.stoi={}; self.itos=[]
    def add(self, x:str)->int:
        if x not in self.stoi: self.stoi[x]=len(self.itos); self.itos.append(x)
        return self.stoi[x]
    def __len__(self): return len(self.itos)

def _resolve_poscar(root:str, name:str)->str:
    """支持 name, name.vasp/POSCAR, name/POSCAR 等多种命名。"""
    cand = [
        os.path.join(root, name),
        os.path.join(root, name + ".vasp"),
        os.path.join(root, name + ".POSCAR"),
        os.path.join(root, name, "POSCAR"),
        os.path.join(root, name, "poscar"),
        os.path.join(root, name, "CONTCAR"),
    ]
    cand += glob.glob(os.path.join(root, "**", name), recursive=True)
    cand += glob.glob(os.path.join(root, "**", name + ".*"), recursive=True)
    for p in cand:
        if os.path.isfile(p): return p
    raise FileNotFoundError(f"POSCAR not found for name='{name}' under '{root}'")
import lmdb
import pickle

def create_lmdb_from_csv(spec: CSVSpec,
                         train_lmdb_path: str = "train/0/data.lmdb",
                         infer_lmdb_path: str = "infer/0/data.lmdb",
                         valid_lmdb_path: str = "val/0/data.lmdb"):
    """创建三个 LMDB 单文件：有 yield 的进训练/验证集，无 yield 的进推理集（文件扩展名为 .lmdb）"""
    import os, csv, pickle, lmdb, torch
    from ase.io import read as ase_read
    
    ref_val_csv = pd.read_csv("split_csv/valid.csv")

    # 确保父目录存在（注意此时 lmdb_path 是“文件”，不是“目录”）
    for p in (train_lmdb_path, infer_lmdb_path, valid_lmdb_path):
        parent = os.path.dirname(p)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    # 读取 CSV
    
    with open(spec.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
        random.seed(42)  # 设置随机种子以确保可重复性
        random.shuffle(rows)  # 随机打乱 rows 列表

    # 构建类别词表
    vocabs: Dict[str, Vocab] = {c: Vocab() for c in spec.cat_cols}
    for r in rows:
        for c in spec.cat_cols:
            vocabs[c].add(str(r[c]).strip())
    # 打开 LMDB（单文件模式 subdir=False；文件名可自定义为 *.lmdb）
    map_size = 1099511627776  # 1 TB
    train_env = lmdb.open(train_lmdb_path, map_size=map_size, subdir=False)
    infer_env = lmdb.open(infer_lmdb_path, map_size=map_size, subdir=False)
    valid_env = lmdb.open(valid_lmdb_path, map_size=map_size, subdir=False)

    train_count = infer_count = valid_count = 0

    for i, r in enumerate(rows):
        # 结构
        poscar = _resolve_poscar(spec.poscar_root, str(r[spec.cat_filename_col]).strip())
        atoms = ase_read(poscar, format='vasp')
        Z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        R = torch.tensor(atoms.get_positions(), dtype=torch.float32)
        cell = torch.tensor(atoms.cell.array, dtype=torch.float32).view(3, 3)
        pbc = torch.tensor(atoms.pbc if atoms.pbc is not None else (True, True, True), dtype=torch.bool)

        # 条件（仅类别）
        cond_cat = {c: torch.tensor([vocabs[c].stoi[str(r[c]).strip()]], dtype=torch.long)
                    for c in spec.cat_cols}

        data_dict = {
            'atomic_numbers': Z.numpy(),
            'pos': R.numpy(),
            'cell': cell.numpy(),
            'pbc': pbc.numpy(),
            'natoms': Z.numel(),
            'cond_cat': {k: v.numpy() for k, v in cond_cat.items()},
            'struct_name': r[spec.cat_filename_col],
            'struct_path': poscar
        }

        has_yield = (spec.target_col and spec.target_col in r and str(r[spec.target_col]).strip() != "")
        if has_yield:
            data_dict['y'] = float(r[spec.target_col])
            data_dict['y_relaxed'] = float(r[spec.target_col])
            if valid_count < 8:
                match_dict = {'base':r['base'], 'solvent':r['solvent'], 'nm':r['nm'], 'cat':r['cat']}
                # 这四个字段匹配，则添加到测试集中
                is_valid = False
                for ref_row in ref_val_csv[['base', 'solvent', 'nm', 'cat']].to_dict('records'):
                    if all(match_dict[k] == str(ref_row[k]) for k in ['base', 'solvent', 'nm', 'cat']):
                        is_valid = True
                        break
                if is_valid:
                    env = valid_env; idx = valid_count; valid_count += 1
                else:
                    env = train_env; idx = train_count; train_count += 1
            else:
                env = train_env; idx = train_count; train_count += 1
        else:
            env = infer_env; idx = infer_count; infer_count += 1

        # Convert the data dictionary to a torch_geometric Data object
        tags = np.full((data_dict['natoms'],), 2, dtype=np.float32)
        data = Data(
            atomic_numbers=torch.tensor(data_dict['atomic_numbers'], dtype=torch.long),
            pos=torch.tensor(data_dict['pos'], dtype=torch.float32),
            cell=torch.tensor(data_dict['cell'], dtype=torch.float32).view(3, 3).unsqueeze(0),
            pbc=torch.tensor(data_dict['pbc'], dtype=torch.bool),
            natoms=data_dict['natoms'],
            tags=torch.tensor(tags),
            struct_name=data_dict['struct_name'],
            struct_path=data_dict['struct_path'],
            mat_index=r['ID'],
            fid =r['ID']
        )
        
        # Add conditional categorical features
        for k, v in data_dict['cond_cat'].items():
            setattr(data, f'cond_cat_{k}', torch.tensor(v, dtype=torch.long))
        setattr(data, 'cond_cat', data_dict['cond_cat'])
        # Add target values if available
        if 'y' in data_dict:
            match_dict = {'base':r['base'], 'solvent':r['solvent'], 'nm':r['nm'], 'cat':r['cat']}

            data.y = torch.tensor([data_dict['y']], dtype=torch.float32)
            data.y_relaxed = torch.tensor([data_dict['y_relaxed']], dtype=torch.float32)
        else:
            data.y = torch.tensor([0.0], dtype=torch.float32)
        data_dict = data

        with env.begin(write=True) as txn:
            txn.put(f"{idx}".encode(), pickle.dumps(data_dict))

    # 写入统计信息
    stats = {'vocabs': {k: {'stoi': v.stoi, 'itos': v.itos} for k, v in vocabs.items()}}
    with train_env.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(train_count))
        txn.put(b'length', pickle.dumps(train_count))
        txn.put(b'__stats__', pickle.dumps(stats))
    with infer_env.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(infer_count))
        txn.put(b'length', pickle.dumps(infer_count))
        txn.put(b'__stats__', pickle.dumps(stats))
    with valid_env.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(valid_count))
        txn.put(b'length', pickle.dumps(valid_count))
        txn.put(b'__stats__', pickle.dumps(stats))

    train_env.close(); infer_env.close(); valid_env.close()
    print(f"创建完成: 训练集 {train_count} 条, 验证集 {valid_count} 条, 推理集 {infer_count} 条")
    return train_count, valid_count, infer_count


if __name__ == "__main__":
    spec = CSVSpec(csv_path="matched_design_space.csv", poscar_root="./POSCAR", target_col="YIELD")
    create_lmdb_from_csv(spec)