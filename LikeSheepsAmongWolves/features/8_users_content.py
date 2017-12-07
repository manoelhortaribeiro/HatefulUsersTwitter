from LikeSheepsAmongWolves.tmp.utils import cols_attr, cols_glove, cols_empath
import networkx as nx
import pandas as pd
import numpy as np

users_attributes = pd.read_csv("../data/users_attributes.csv")
users_content = pd.read_csv("../data/users_content.csv")
users_content2 = pd.read_csv("../data/users_content2.csv")
users_hate = pd.read_csv("../data/deleted_account.csv")

df = pd.merge(users_attributes, users_content, on="user_id", how="inner")
df = pd.merge(df, users_content2, on="user_id", how="inner")
df = pd.merge(df, users_hate, on="user_id", how="inner")

df.to_csv("../data/users_all.csv", index=False)

cols = cols_attr + cols_glove + cols_empath
num_cols = len(cols)

graph = nx.read_graphml("../data/users_infected_diffusion.graphml")
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
users_neighbor = pd.read_csv("../data/user_neighborhood.csv")
df = pd.merge(users_all, users_neighbor, on="user_id", how="inner")
print(list(df['Unnamed: 0']))
df = df.drop(['Unnamed: 0'], axis=1)
df.to_csv("../data/users_all_neigh.csv", index=False)
