# %%
import h5py
import numpy as np

# 创建一个示例数据数组
data = np.array([[1, 2, 3], [4, 5, 6]])

# 指定多层级group的路径
group_path = 'group1/subgroup1'

# 使用'w'模式创建或覆盖HDF5文件
with h5py.File('example.h5', 'w') as file:
    # 根据group_path创建多层级group
    group = file.require_group(group_path)
    
    # 在指定的group中创建数据集dataset，并写入数据
    dataset = group.create_dataset('data', data=data)

# 验证数据是否写入
with h5py.File('example.h5', 'r') as file:
    # 读取多层级group中的dataset
    dataset = file['group1']['subgroup1']['data']
    print(dataset[:])

# %%
import pandas as pd

df = pd.read_hdf('example.h5', key='group1')
df.head()

