import lmdb
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

def test_lmdb_contents(db_path: str, db_name: str, y_values_list=None):
    """测试LMDB数据库的内容与数量，并收集y字段的值用于分布图"""
    if not os.path.exists(db_path):
        print(f"{db_name} 数据库路径不存在: {db_path}")
        return
    
    env = lmdb.open(db_path+"/data.lmdb", subdir=False, readonly=True)
    y_values = []
    
    with env.begin() as txn:
        # 获取数据库长度
        db_len = pickle.loads(txn.get(b'__len__'))
        print(f"{db_name} 数据库数量: {db_len}")
        
        # 获取统计信息
        stats = pickle.loads(txn.get(b'__stats__'))
        print(f"{db_name} 数据库统计信息: {stats}")
        
        # 检查前几条数据的内容
        for i in range(min(25, db_len)):
            data = pickle.loads(txn.get(f'{i}'.encode()))
            print(data)
            print(f"\n{db_name} 数据库第 {i+1} 条数据:")
            print(f"  结构名称: {data.get('struct_name')}")
            print(f"  结构路径: {data.get('struct_path')}")
            print(f"  原子数量: {data.get('natoms')}")
            print(f"  条件类别: {data.get('cond_cat_base')}, {data.get('cond_cat_solvent')}, {data.get('cond_cat_nm')}")
            if 'y' in data:
                print(f"  目标值(yield): {data.get('y')}")
            if 'y_relaxed' in data:
                print(f"  目标值(relaxed): {data.get('y_relaxed')}")
        
        # 收集所有数据的y值
        if y_values_list is not None:
            for i in range(db_len):
                data = pickle.loads(txn.get(f'{i}'.encode()))
                if 'y' in data:
                    y_values.append(data.get('y').item())
            y_values_list.extend(y_values)
    
    env.close()
    return y_values

def plot_y_distribution(train_y_values, valid_y_values):
    """分别绘制训练集和验证集y字段的分布图"""
    # 绘制训练集y值分布
    if train_y_values:
        plt.figure(figsize=(10, 6))
        plt.hist(train_y_values, bins=30, alpha=0.7, color='blue', label='Train y values')
        plt.xlabel('y value')
        plt.ylabel('Frequency')
        plt.title('Distribution of y values in Train dataset')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('stat_figs/train_y.png')
    
    # 绘制验证集y值分布
    if valid_y_values:
        plt.figure(figsize=(10, 6))
        plt.hist(valid_y_values, bins=30, alpha=0.7, color='green', label='Valid y values')
        plt.xlabel('y value')
        plt.ylabel('Frequency')
        plt.title('Distribution of y values in Valid dataset')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('stat_figs/val_y.png')

if __name__ == "__main__":
    # 用于存储y值的列表
    train_y_values = []
    valid_y_values = []
    
    # 依次测试 train, valid, infer 数据库，并收集y值
    test_lmdb_contents("all", "all")
    import pdb; pdb.set_trace()
    test_lmdb_contents("val/0", "Valid", valid_y_values)
    # test_lmdb_contents("infer/0", "Infer")
    # import pdb; pdb.set_trace()
    # 绘制y值分布图
    # plot_y_distribution(train_y_values, valid_y_values)
