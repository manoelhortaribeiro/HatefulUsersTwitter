from LikeSheepsAmongWolves.tmp.utils import cols_attr, cols_glove, cols_empath
import networkx as nx
import pandas as pd
import numpy as np

cols = cols_attr + cols_glove + cols_empath
num_cols = len(cols)

graph = nx.read_graphml("../data/users_infected_diffusion.graphml")
df = pd.read_csv("../data/users_all.csv", index_col=1)

df_annotated = df[df.hate != "other"]

users = list()
for user_id in graph.nodes():
    if int(user_id) in df.index.values:
        tmp = []
        for neighbor in graph.neighbors(user_id):
            if int(neighbor) in df.index.values:
                tmp.append(list(df.loc[int(neighbor)][cols].values))
        users.append([user_id] + list(np.average(np.array(tmp), axis=0)))

df = pd.DataFrame.from_records(users, columns=["user_id"] + ["c_"+v for v in cols])
df.to_csv("../data/users_neighborhood.csv")
