import networkx as nx
import numpy as np
import pandas as pd

from tmp.utils import cols_attr, cols_glove, cols_empath

# Gets mean and median between tweets
tweets = pd.read_csv("../data/preprocessing/tweets.csv")
tweets.sort_values(by=["user_id", "tweet_creation"], ascending=True, inplace=True)
tweets["time_diff"] = tweets.groupby("user_id", sort=False).tweet_creation.diff()
time_diff_series_mean = tweets.groupby("user_id", sort=False).time_diff.mean()
time_diff_series_median = tweets.groupby("user_id", sort=False).time_diff.median()
time_diff = time_diff_series_mean.to_frame()
time_diff["time_diff_median"] = time_diff_series_median
time_diff.to_csv("../data/features/time_diff.csv")

users_attributes = pd.read_csv("../data/features/users_attributes.csv")
users_content = pd.read_csv("../data/features/users_content.csv")
users_content2 = pd.read_csv("../data/features/users_content2.csv")
users_time = pd.read_csv("../data/features/time_diff.csv")
users_deleted = pd.read_csv("../data/extra/deleted_account_before_guideline.csv")
users_deleted_after_guideline = pd.read_csv("../data/extra/deleted_account_after_guideline.csv")
users_date = pd.read_csv("../data/extra/created_at.csv")

df = pd.merge(left=users_attributes, right=users_content, on="user_id", how="left")
df = pd.merge(left=df, right=users_content2, on="user_id", how="left")
df = pd.merge(left=df, right=users_deleted, on="user_id", how="left")
df = pd.merge(left=df, right=users_deleted_after_guideline, on="user_id", how="left")
df = pd.merge(left=df, right=users_time, on="user_id", how="left")
df = pd.merge(left=df, right=users_date, on="user_id", how="left")

df.to_csv("../data/features/users_all.csv", index=False)

# df = pd.read_csv("../data/users_all.csv")

df1 = df.set_index("user_id", verify_integrity=True)

cols = cols_attr + cols_glove + cols_empath
num_cols = len(cols)
graph = nx.read_graphml("../data/features/users_hate.graphml")
users = list()
count = 0
for user_id in graph.nodes():
    count += 1
    if int(user_id) in df1.index.values:
        tmp = []
        for neighbor in graph.neighbors(user_id):
            if int(neighbor) in df1.index.values:
                tmp.append(list(df1.loc[int(neighbor)][cols].values))
        users.append([user_id] + list(np.average(np.array(tmp), axis=0)))


df2 = pd.DataFrame.from_records(users, columns=["user_id"] + ["c_" + v for v in cols])
df2.to_csv("../data/features/users_neighborhood.csv", index=False)

# df = pd.read_csv("../data/users_all.csv")
# df2 = pd.read_csv("../data/users_neighborhood.csv")

df3 = pd.merge(left=df, right=df2, on="user_id", how="left")
df3.to_csv("../data/features/users_all_neighborhood.csv", index=False)
