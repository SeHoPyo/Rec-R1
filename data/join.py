import pandas as pd

df_review = pd.read_csv("data_review.csv")
df_meta = pd.read_csv("data_meta.csv")
df_c4 = pd.read_csv("data_c4.csv")

# Print data types before conversion
print("Before conversion:")
print("df_review['parent_asin'] dtype:", df_review['parent_asin'].dtype)
print("df_meta['parent_asin'] dtype:", df_meta['parent_asin'].dtype)

# Convert parent_asin to string type in both dataframes
df_review['parent_asin'] = df_review['parent_asin'].astype(str)
df_meta['parent_asin'] = df_meta['parent_asin'].astype(str)

# Print data types after conversion
print("After conversion:")
print("df_review['parent_asin'] dtype:", df_review['parent_asin'].dtype)
print("df_meta['parent_asin'] dtype:", df_meta['parent_asin'].dtype)

# Use merge instead of join
df_combined = pd.merge(df_review, df_meta, on="parent_asin", how="left")
df_combined = pd.merge(df_combined, df_c4, on="user_id", how="left")

df_combined.to_csv("data_combined.csv", index=False)

print(df_combined.head())
