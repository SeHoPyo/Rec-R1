import pandas as pd

# df_combined = pd.read_csv("/home/s1/sehopyo/Rec-R1/data/data_combined.csv")

# df_combined = df_combined.head(5)

# df_combined.to_csv("data_combined_head.csv", index=False)


# data = pd.read_parquet("/home/s1/sehopyo/Rec-R1/data/amazon_c4/inst/subset_other/train.parquet")

# print(data["reward_model"][0])

meta = pd.read_csv("/home/s1/sehopyo/Rec-R1/data/data_meta.csv")
                   
print(meta[0]["author"])