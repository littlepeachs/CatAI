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

def create_lmdb_from_csv(spec: CSVSpec, all_lmdb_path: str = "all/data.lmdb"):
    """将整个设计空间写入单一 LMDB (all)。"""
    import os, csv, pickle, lmdb, torch
    from ase.io import read as ase_read

    parent = os.path.dirname(all_lmdb_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    with open(spec.csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
        random.seed(42)
        random.shuffle(rows)

    # 构建类别词表
    vocabs: Dict[str, Vocab] = {c: Vocab() for c in spec.cat_cols}
    
    for r in rows:
        for c in spec.cat_cols:
            vocabs[c].add(str(r[c]).strip())

    map_size = 1099511627776  # 1 TB
    all_env = lmdb.open(all_lmdb_path, map_size=map_size, subdir=False)

    total_count = 0

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

        # has_yield = (spec.target_col and spec.target_col in r and str(r[spec.target_col]).strip() != "")
        # if has_yield:
        #     data_dict['y'] = float(r[spec.target_col])
        #     data_dict['y_relaxed'] = float(r[spec.target_col])

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

        with all_env.begin(write=True) as txn:
            txn.put(f"{total_count}".encode(), pickle.dumps(data_dict))
        total_count += 1

    # 写入统计信息
    stats = {'vocabs': {k: {'stoi': v.stoi, 'itos': v.itos} for k, v in vocabs.items()}}
    with all_env.begin(write=True) as txn:
        txn.put(b'__len__', pickle.dumps(total_count))
        txn.put(b'length', pickle.dumps(total_count))
        txn.put(b'__stats__', pickle.dumps(stats))

    all_env.close()
    print(f"创建完成: all 集 {total_count} 条")
    return total_count


if __name__ == "__main__":
    spec = CSVSpec(csv_path="matched_design_space.csv", poscar_root="./POSCAR", target_col="YIELD")
    create_lmdb_from_csv(spec, all_lmdb_path="all/data.lmdb")