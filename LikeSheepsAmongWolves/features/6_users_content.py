from LikeSheepsAmongWolves.tmp.utils import cols_attr, cols_glove, cols_empath
import networkx as nx
import pandas as pd
import numpy as np

# # Gets mean and median between tweets
# tweets = pd.read_csv("../data/tweets.csv")
# tweets.sort_values(by=["user_id", "tweet_creation"], ascending=True, inplace=True)
# tweets["time_diff"] = tweets.groupby("user_id", sort=False).tweet_creation.diff()
# time_diff_series_mean = tweets.groupby("user_id", sort=False).time_diff.mean()
# time_diff_series_median = tweets.groupby("user_id", sort=False).time_diff.median()
# time_diff = time_diff_series_mean.to_frame()
# time_diff["time_diff_median"] = time_diff_series_median
# time_diff.to_csv("../data/time_diff.csv")
#
# users_attributes = pd.read_csv("../data/users_attributes.csv")
# users_content = pd.read_csv("../data/users_content.csv")
# users_content2 = pd.read_csv("../data/users_content2.csv")
# users_deleted = pd.read_csv("../data/deleted_account.csv")
# users_time = pd.read_csv("../data/time_diff.csv")
#
# df = pd.merge(users_attributes, users_content, on="user_id", how="inner")
# df = pd.merge(df, users_content2, on="user_id", how="inner")
# df = pd.merge(df, users_deleted, on="user_id", how="inner")
# df = pd.merge(df, users_time, on="user_id", how="inner")
#
# df.to_csv("../data/users_all.csv", index=False)
#
# users_date = pd.read_csv("../data/created_at.csv")
# created_at = pd.merge(users_attributes, users_date, on="user_id", how="inner")
#
# created_at = created_at[["user_id", "created_at", "hate", "hate_neigh", "normal_neigh"]]
# created_at.to_csv("../data/users_created_at.csv", index=False)
#
#
cols = cols_attr + cols_glove + cols_empath
num_cols = len(cols)

graph = nx.read_graphml("../data/users_hate.graphml")
df = pd.read_csv("../data/users_all.csv", index_col=0)

users = list()
for user_id in graph.nodes():
    if int(user_id) in df.index.values:
        tmp = []
        for neighbor in graph.neighbors(user_id):
            if int(neighbor) in df.index.values:
                tmp.append(list(df.loc[int(neighbor)][cols].values))
        users.append([user_id] + list(np.average(np.array(tmp), axis=0)))

df = pd.DataFrame.from_records(users, columns=["user_id"] + ["c_"+v for v in cols])
df.to_csv("../data/users_neighborhood.csv", index=False)

users_all = pd.read_csv("../data/users_all.csv")
users_neighbor = pd.read_csv("../data/users_neighborhood.csv")
df = pd.merge(users_all, users_neighbor, on="user_id", how="inner")
df.to_csv("../data/users_all_neigh.csv", index=False)
