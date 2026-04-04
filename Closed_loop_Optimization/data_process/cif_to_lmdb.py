from torch import cdist

from torch import Tensor
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from pymatgen.core.structure import Structure
import os
from tqdm import tqdm
import numpy as np
import torch
import lmdb, pickle
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

"""
Please correct the following lines:
"""
folder_path = "./mofdb/mofdb_cif"
annotation_path = "./mofdb/H2_pre2.0.csv"
db_path = "./mofdb/mofdb_lmdb/mof_h2pre2_adsorption/"
csv_path = "./mofdb/mofdb_lmdb/mof_h2pre2_adsorption/"
ID_name = "id"
label_name = "adsorption"
cross_validation_para = False # 如果不是下游任务数据集，set to False
"""
Stop changing here.
"""

def convert_cif_to_dict(directory):
    print("Converting CIF to LMDB, dataset: ", db_path.split('/')[-2])
    # df = pd.read_csv(annotation_path, usecols=['ID', 'act'])
    # df = pd.read_csv(annotation_path, sep=';', usecols=['CSD_code', 'Electronic_E'])

    df = pd.read_csv(annotation_path, usecols=[ID_name, label_name], na_values=['na', 'NA', 'n/a', '--', '', '0', 'NaN'])
    cross_validation = cross_validation_para
    df = df.dropna()
    print('after dropna: ',df.shape[0])
    print(df.head())
    #exit(0)
    dataset = []
    y_arr = []
    # 读取 CIF 文件
    for file_name in tqdm(os.listdir(directory), desc="Processing", unit="file"):
        if os.path.isfile(os.path.join(folder_path, file_name)) and file_name.endswith(".cif"):
            # print("processing: ", file_name)
            file_path = os.path.join(directory, file_name)
            # ID = file_name.split("_")[0]    # for catalyst dataset and oct_Step1
            ID = file_name.split(".")[0]
            ID = file_name
            if df.loc[df[ID_name] == ID, label_name].notna().any():
                y = df.loc[df[ID_name] == ID, label_name].values[0]
            else:
                # 处理找不到匹配或 'gibbs_free_energy' 为空的情况
                # print(f"No valid 'act' value found for ID '{ID}'. Skipping.")
                continue

            log_abs_value = math.log(abs(y) + 1e-10)
            if y < 0:
                y = -log_abs_value
            elif y > 0:
                y = log_abs_value
            else:
                continue
            try:
                # y_arr.append(y)

                # 从 CIF 文件中读取结构数据
                structure = Structure.from_file(file_path)

                pos = np.array(structure.cart_coords, dtype=np.float32)

                lattice_matrix = [structure.lattice.matrix]
                cell = np.array(lattice_matrix, dtype=np.float32)

                atomic_numbers = np.array(structure.atomic_numbers, dtype=np.float32)
                # print('an: ',atomic_numbers)
                natoms = structure.num_sites
                if natoms < 4:
                    # print('The data natoms < 4, skipping: ', ID, natoms)
                    continue
                tags = np.full((structure.num_sites,), 2, dtype=np.float32)

                # fixed = np.zeros(structure.num_sites, dtype=np.double)
                fixed: Tensor = torch.zeros(natoms).type(torch.float32)
                # store the data into the lmdb
                y_arr.append(y)
                one_data = Data(pos=pos, cell=cell, atomic_numbers=atomic_numbers, natoms=natoms, tags=tags, fixed=fixed, y=y)
                dataset.append(one_data)
            except Exception as e:
                print("error:", e)
            # if len(y_arr) >= 100000:
            #     break

    print("check dataset's size",len(y_arr))

    if not cross_validation:
        train, temp = train_test_split(dataset, train_size=0.7)
        test, val = train_test_split(temp, train_size=0.5)
        store_in_lmdb(train, "train", db_path+'/')
        store_in_lmdb(test, "test", db_path+'/')
        store_in_lmdb(val, "val", db_path+'/')
        output_df = pd.DataFrame(train)
        output_df.to_csv(csv_path + "train.csv", index=False)
        output_df = pd.DataFrame(test)
        output_df.to_csv(csv_path + "test.csv", index=False)
        output_df = pd.DataFrame(val)
        output_df.to_csv(csv_path + "val.csv", index=False)
        create_metadata(db_path, cross_validation)
    else:
        for set in range(5):
            p = db_path + f"{set}/"
            val_set_len = len(y_arr)//5
            print("single valset size:", len(y_arr)//5)
            val = dataset[val_set_len*(set): val_set_len*(set+1)]
            # print("check valset: s&e", set, set+1)
            # print("check val size:", len(val))

            train = dataset[:val_set_len*(set)] + dataset[val_set_len*(set+1):]
            # print("check trainset:", f"s~ {set}, {set+1} ~e" )
            # print("check trainset size:", len(train))

            store_in_lmdb(train, "train", p)
            store_in_lmdb(val, "val", p)
            store_in_lmdb([], "test", p)

            # output_df = pd.DataFrame(train)
            # output_df.to_csv(csv_path+f"{set}_train.csv", index=False)
            # output_df = pd.DataFrame(val)
            # output_df.to_csv(csv_path+f"{set}_val.csv", index=False)
            create_metadata(p, cross_validation)

    dataset_attributes(y_arr)

def store_in_lmdb(ds, set_name, path):

    output_path = path + f"{set_name}/data.0000.lmdb"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    if os.path.exists(output_path):
        print(f"Removing existed {output_path}.")
        os.remove(output_path)  # 删除旧的 LMDB 文件

    env = lmdb.open(output_path, map_size=1099511627776, subdir=False)  # 1 TB
    cnt = 0
    # 打开一个写事务
    with env.begin(write=True) as txn:
        for data in ds:
            try:
                assert data.y is not None
            except:
                print(f"Skipping data {data}.")
                return
            # 将键值对存储到数据库中
            serialized_data = pickle.dumps(data)
            txn.put(str(cnt).encode(), serialized_data)
            cnt += 1

    # 关闭环境
    env.close()

def create_metadata(dir_path, cross_validation):
    if dir_path is None:
        # 这是一个功能函数，除了单独查看数据集时可以改下面目录，一般不会单独使用
        dir_path = "/path/to/catai_data/few_shot_lmdb/data_catalyst_229/0/"

    natoms_all_data = []
    # cnt = 0
    no_neighbors = 0
    sets = ["train", "test", "val"]
    y_arr = []
    error_na = 0
    for set in sets:
        cnt = 0
        natoms_arr = []
        #y_arr=[]
        lmdb_path = dir_path + f"{set}/data.0000.lmdb"
        if cross_validation and set == "test":
            natoms = np.array(natoms_arr, dtype=np.int32)
            # 将数组保存到 .npz 文件
            np.savez(dir_path + f'{set}/metadata.npz', natoms=natoms)
            continue
        # 打开 LMDB 环境
        src_env = lmdb.open(lmdb_path, readonly=True, subdir=False)
        txn = src_env.begin(buffers=True)
        for key, value in tqdm(txn.cursor(), desc=f"Reading {lmdb_path}", total=txn.stat()['entries']):
            data = pickle.loads(value)

            if hasattr(data, 'natoms'):
                natoms_arr.append(data.natoms)
                if data.natoms < 4:
                    print("data.natoms < 4: ", data.natoms)
                    error_na +=1
            else:
                print(f"No natoms. Skipping data {data}.")
                natoms_arr.append(0)
            if not has_neighbors(data, 12.0):
                no_neighbors += 1
            if False:
                print('pos: ', data.pos, type(data.pos))
                print('cell: ', data.cell, type(data.cell))
                print('atomic_numbers: ', data.atomic_numbers, type(data.atomic_numbers))
                print('natoms: ', data.natoms)                
                print('tags: ', data.tags, type(data.tags))
                print('fixed: ', data.fixed, type(data.fixed))
                #exit(0)


            if not hasattr(data, 'pos'): print("no pos")
            if not hasattr(data, 'cell'): print("no cell")
            if not hasattr(data, 'atomic_numbers'): print("no atom num")
            if not hasattr(data, 'natoms'): print("no natoms")
            if not hasattr(data,  'tags'): print("no tags")
            if not hasattr(data, 'fixed'): print('no fixed')
            if not hasattr(data, 'y'): print('no y')
            cnt += 1
            y_arr.append(data.y)


        src_env.close()
        print(f"Finish reading {cnt} data points from {lmdb_path}.")
        
        natoms_all_data += natoms_arr
        # 创建数组
        natoms = np.array(natoms_arr, dtype=np.int32)
        # 将数组保存到 .npz 文件
        np.savez(dir_path+f'{set}/metadata.npz', natoms=natoms)


    dataset_attributes(y_arr) 

    # print(f"no neighbors min dist max {max(dists_arr)}: ", dists_arr)
    # plt.hist(natoms_all_data, bins=5)
    # natoms_df = pd.DataFrame(natoms_all_data, columns=["num_of_atoms"])
    # sns.set_style("whitegrid")
    # sns.displot(natoms_df, bins=10, x="num_of_atoms",kde=True)
    # sns.displot(natoms_df, x="num_of_atoms", kde=True)
    # plt.savefig('guangfu_S1_natoms_hist.png')
    # y_df = pd.DataFrame(y_arr, columns=["y_value"])
    # sns.displot(y_df, bins=10, x="y_value",kde=True)
    # plt.savefig('guangfu_S1_y_hist.png')
    # df = pd.DataFrame(y_arr, columns=['act'])
    # df.to_csv("catalyst_dataset.csv", index=False)


dists_arr = []
def has_neighbors(data, cutoff=12.0):
    pos = torch.tensor(data.pos).float()
    dists = cdist(pos, pos)
    dists[dists == 0] = float('inf')
    if dists.min() >= cutoff:
        dists_arr.append(dists.min().item())
    return dists.min() < cutoff

def dataset_attributes(array):
    def log_transform(y):
        log_abs_value = math.log(abs(y) + 1e-10)
        if y < 0:
            return -log_abs_value
        elif y > 0:
            return log_abs_value
        else:
            return 0
    if array is None:
        df = pd.read_csv(annotation_path, sep=';', usecols=['Metal_q'])
        # print(df)
        df = df.dropna()
        df['Metal_q'] = df['Metal_q'].apply(log_transform)
        print(df)
        mean_value = df['Metal_q'].mean()
        std_deviation = df['Metal_q'].std()

    else:
        # 计算平均值
        mean_value = np.mean(array)
        # 计算标准差
        std_deviation = np.std(array)

    print("mean:", mean_value)
    print("std:", std_deviation)




convert_cif_to_dict(folder_path)

# dataset_attributes(None)
# create_metadata(None, False)
