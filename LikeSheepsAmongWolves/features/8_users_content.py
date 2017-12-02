import pandas as pd

users_attributes = pd.read_csv("../data/users_attributes.csv")
users_content = pd.read_csv("../data/users_content.csv")
users_content2 = pd.read_csv("../data/users_content2.csv")


df = pd.merge(users_attributes, users_content, on="user_id", how="inner")
df = pd.merge(df, users_content2, on="user_id", how="inner")


df.to_csv("../data/users_all.csv")
