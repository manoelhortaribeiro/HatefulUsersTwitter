import pandas as pd

users_all = pd.read_csv("../data/users_all.csv")
users_neighbor = pd.read_csv("../data/user_neighborhood.csv")


df = pd.merge(users_all, users_neighbor, on="user_id", how="inner")


df.to_csv("../data/users_all_neigh.csv")
