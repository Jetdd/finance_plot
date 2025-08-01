'''
Author: Jet Deng
Date: 2025-07-29 16:23:06
LastEditTime: 2025-07-31 14:45:27
Description: 
'''
import polars as pl
import pandas as pd
from plot import FinancePlot


df_1 = (
    pl.read_ipc("D:/projects/data/bond/xbond/active/1Min.feather")
    .to_pandas()
    .set_index("source_time")
    .sort_index()
    .loc["20230701":"20230705"]
)
df_t = (
    pd.read_pickle("D:/projects/data/future/1m/dominant/adj/T.pkl")
    .reset_index()
    
    .set_index("datetime")
    .loc["20230701":"20230705"]
)


fp = FinancePlot(subplot=False, standardize=False, interactive=True)
# fp.plot_trend(df_t['close'].iloc[-100:], df_1['close'].iloc[-100:], labels=['T', 'bond'])
# fp.plot_distribution(df_t['close'].iloc[-100:], df_1['close'].iloc[-100:], labels=['T', 'bond'])
fp.plot_corr(df_t['close'], df_1['close'], labels=['T', 'bond'])