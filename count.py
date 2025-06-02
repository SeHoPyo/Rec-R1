import pandas as pd
df = pd.read_parquet("data/amazon_c4/inst/dense/sports/full_train.parquet")
print(len(df))  # => 총 샘플 수

df = pd.read_parquet("data/amazon_c4/inst/dense/sports/full_val.parquet")
print(len(df))  # => 총 샘플 수

df = pd.read_parquet("data/amazon_c4/inst/dense/sports/full_test.parquet")
print(len(df))  # => 총 샘플 수