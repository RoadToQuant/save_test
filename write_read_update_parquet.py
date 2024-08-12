# %%
import pandas as pd 
import numpy as np 
import h5py
from joblib import Parallel, delayed
import os

from ttools.decorators import func_timer

# %%
# 创建模拟数据：5000只股票，10年数据
# obids = [str(i).rjust(6, "0") + ".XSHE" for i in range(1, 5001)]
obids = [str(i).rjust(6, "0") + ".XSHE" for i in range(1, 51)]
len(obids)
obids[:1]  # ['000001']

# freq = '1d'
freq = '1min'
# 日度数据
# dates = pd.date_range("2015-01-01", "2024-01-01", freq=freq)
# len(dates)  # 3288

# 分钟数据
dates = pd.date_range("2015-01-01", "2024-01-01", freq=freq)
len(dates)  # 
# 下面三列日期分别反映了日常更新、遗漏更新、补齐更新的状态
new_dates1 = pd.date_range('2023-12-01', '2024-01-10', freq=freq)  # 更新1：存在重复索引
new_dates2 = pd.date_range('2024-02-01', '2024-02-10', freq=freq)  # 更新2：跳跃索引
new_dates3 = pd.date_range('2024-01-01', '2024-02-15', freq=freq)  # 更新3：重复+新索引

columns = [
    'num_trades',
    'volume',
    'open',
    'high',
    'iopv',
    'prev_close',
    'limit_down',
    'limit_up',
    'total_turnover',
    'close',
    'low'
]

df_all = pd.DataFrame(
    np.random.randn(len(obids), len(dates), len(columns)).reshape(-1, len(columns)),
    index=pd.MultiIndex.from_product([obids, dates]).rename(['order_book_id', 'date']),
    columns=columns
)
df_all.shape  # (16440000, 11)

df_new1 = pd.DataFrame(
    np.random.randn(len(obids), len(new_dates1), len(columns)).reshape(-1, len(columns)),
    index=pd.MultiIndex.from_product([obids, new_dates1]).rename(['order_book_id', 'date']),
    columns=columns
)
df_new1.shape 

df_new2 = pd.DataFrame(
    np.random.randn(len(obids), len(new_dates2), len(columns)).reshape(-1, len(columns)),
    index=pd.MultiIndex.from_product([obids, new_dates2]).rename(['order_book_id', 'date']),
    columns=columns
)
df_new2.shape 

df_new3 = pd.DataFrame(
    np.random.randn(len(obids), len(new_dates3), len(columns)).reshape(-1, len(columns)),
    index=pd.MultiIndex.from_product([obids, new_dates3]).rename(['order_book_id', 'date']),
    columns=columns
)
df_new3.shape 


# %%
# 写入测试
def write_parquet(dataframe: pd.DataFrame, order_book_id, dirpath_parquet):
    _fp = os.path.join(dirpath_parquet, order_book_id + ".parquet")
    # print(dataframe.head(), _fp)
    dataframe.reset_index(0, drop=True).to_parquet(_fp)
    return order_book_id

@func_timer
def parallel_write_parquet(dataframe: pd.DataFrame):
    dirpath_parquet = f'./data/parquets_{freq}'
    _writed_obids = Parallel(n_jobs=4, verbose=10)(
        delayed(write_parquet)(
            _df, _obid, dirpath_parquet 
        ) for _obid, _df in dataframe.groupby(by='order_book_id')
    )
    return _writed_obids


# writed_obids = parallel_write_parquet(df_all)  # 1d, 5000只，n_jobs==4, 16s左右写入
# writed_obids = parallel_write_parquet(df_all)  # 1min, 50只，n_jobs==4, 119s左右写入

# %%
# 读取测试
def read_parquet(order_book_id, start_date, end_date, dirpath_parquet):
    _fp = os.path.join(dirpath_parquet, order_book_id + ".parquet")
    _df = pd.read_parquet(_fp).loc[start_date: end_date]
    _df['order_book_id'] = order_book_id
    return _df


@func_timer
def parallel_read_parquet(order_book_ids):
    # _obids = dataframe.index.get_level_values(0).unique()
    dirpath_parquet = f'./data/parquets_{freq}'
    st_dt, ed_dt = '2017-01-01', '2021-12-31'
    _list_dfs = Parallel(n_jobs=4, verbose=10)(
        delayed(read_parquet)(
            _obid, st_dt, ed_dt, dirpath_parquet
        ) for _obid in order_book_ids
    )
    return pd.concat(_list_dfs)


# df_all_sub = parallel_read_parquet(np.random.choice(obids, 3000))  # n_jobs==4，7s左右读取
# df_all_sub.head() # (5478000, 12)
# df_all_sub = parallel_read_parquet(np.random.choice(obids, 30))  # n_jobs==4，42s左右读取
# df_all_sub.head() # (5478000, 12)

# %%
# 读取并核查索引是否重复
def read_and_check_parquet(order_book_id, start_date, end_date, dirpath_parquet):
    _fp = os.path.join(dirpath_parquet, order_book_id + ".parquet")
    _df = pd.read_parquet(_fp).loc[start_date: end_date]
    _df['order_book_id'] = order_book_id
    ndates = _df.index.nunique()
    nshape = _df.shape[0]
    same_length = (ndates == nshape)
    return same_length
    # consist = ndates == len(pd.date_range(_df.index.min(), _df.index.max()))  # 仅测试环境针对连续日期
    # return consist and same_length


@func_timer
def parallel_read_and_check_parquet(order_book_ids):
    dirpath_parquet = f'./data/parquets_{freq}'
    st_dt, ed_dt = '2015-01-01', '2024-12-31'
    _list_uniques = Parallel(n_jobs=4, verbose=10)(
        delayed(read_and_check_parquet)(
            _obid, st_dt, ed_dt, dirpath_parquet
        ) for _obid in order_book_ids
    )
    return _list_uniques

list_uniques = parallel_read_and_check_parquet(obids)  # n_jobs==4，9s左右读取
np.min(list_uniques) # True

# %%
# 更新测试
def update_parquet(dataframe: pd.DataFrame, order_book_id, dirpath_parquet):
    _fp = os.path.join(dirpath_parquet, order_book_id + ".parquet")
    _df_local = pd.read_parquet(_fp)
    dataframe.reset_index(0, drop=True, inplace=True)
    _df_add = dataframe.loc[~dataframe.index.isin(_df_local.index)]
    _df_new = pd.concat([_df_local, _df_add]).sort_index()
    _df_new.to_parquet(_fp)
    return order_book_id


@func_timer
def parallel_update_parquet(dataframe):
    dirpath_parquet = f'./data/parquets_{freq}'
    _updated_obids = Parallel(n_jobs=4, verbose=10)(
        delayed(update_parquet)(
            _df, _obid, dirpath_parquet 
        ) for _obid, _df in dataframe.groupby(by='order_book_id')
    )
    return _updated_obids


# updated_obids1 = parallel_update_parquet(df_new1)  # n_jobs==4, 20s左右
# updated_obids2 = parallel_update_parquet(df_new2)  # n_jobs==4, 20s左右
# updated_obids2 = parallel_update_parquet(df_new3)  # n_jobs==4, 20s左右

# %%
fp_h5 = './data/temp.h5'
with h5py.File(fp_h5, 'r') as f:
    keys = list(f.keys())
    print(len(keys))

# %%

df = pd.read_hdf(fp_h5, key='000001.XSHE')
df.tail()
df.columns.tolist()
