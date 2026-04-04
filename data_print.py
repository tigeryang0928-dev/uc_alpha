import pandas as pd


data = pd.read_parquet('backtest_2/data/turnover.parquet')

print(data.head())