"""
HDF文件保存测试
"""
# %%
import pandas as pd
import h5py
import time

fp = r'E:\Data\stocks\stocks_daily_bar\_all_stocks_post.parquet'
df = pd.read_parquet(fp)
df.head()

# %%
s = time.time()
fp_h5 = 'temp.h5'

obids = df.index.levels[0]
for obid in obids:
    key = obid
    df.loc[obid].to_hdf(fp_h5, key=key, mode='a', append=True, 
                        format='table', complevel=9, data_columns=True)
    print("save", obid)
e = time.time()

print("time consume", (e-s)/60)

# %%
# h5obj = h5py.File(fp_h5, 'r')
# h5obj.keys()
# h5obj.close()
df0 = pd.read_hdf(fp_h5, key='000001.XSHE', where='index >= "2024-01-01"')

df0.index.nunique(), df0.shape
# %%
from joblib import Parallel, delayed

results = Parallel(n_jobs=4, verbose=10)(
    delayed(pd.read_hdf)(
        fp_h5, key=key, where='index >= "2024-01-01"'
    ) for key in obids[:10]
)
results_full = [df.assign(order_book_id=obids[:10][i]).reset_index() for i, df in enumerate(results[:10])]
df_res = pd.concat(results_full, ignore_index=True)
df_res.head()
df.shape
# %%
len(results)
