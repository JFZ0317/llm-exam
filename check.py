import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 200)
data = pd.read_parquet("sft-data/preprocess/train_sft.parquet")
print(data.head())