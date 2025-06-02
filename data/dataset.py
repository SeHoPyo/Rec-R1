from datasets import load_dataset
import pandas as pd
data_review = load_dataset('McAuley-Lab/Amazon-Reviews-2023', 'raw_review_All_Beauty')["full"].to_csv("data_review.csv")
df_review = pd.read_csv("data_review.csv")
data_meta = load_dataset('McAuley-Lab/Amazon-Reviews-2023', 'raw_meta_All_Beauty')["full"].to_csv("data_meta.csv")
df_meta = pd.read_csv("data_meta.csv")
data_c4 = load_dataset("McAuley-Lab/Amazon-C4")["test"].to_csv("data_c4.csv")
df_c4 = pd.read_csv("data_c4.csv")

print(df_review.head())
print(df_meta.head())

print(df_meta["parent_asin"].unique())