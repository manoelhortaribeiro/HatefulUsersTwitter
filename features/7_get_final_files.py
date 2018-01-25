import json

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph
from sklearn.model_selection import train_test_split
from tmp.utils import cols_attr, cols_glove

from sklearn.preprocessing import StandardScaler

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = makes new mapping  = = = = = = = = = = = = = = = = = =
# df = pd.read_csv("../data/users_all.csv", index_col=0)
#
# old_index = df.index
#
# df.index = np.array(range(len(df.index)))
# df.index.name = "user_id"
#
# new_index = df.index
#
# df.to_csv("../data/users_anon.csv")
#
# df = pd.read_csv("../data/users_all_neighborhood.csv", index_col=0)
#
# df.index = np.array(range(len(df.index)))
# df.index.name = "user_id"
#
# df.to_csv("../data/users_neighborhood_anon.csv")

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = cleans graph = = = = = = = = = = = = = = = = = = = = =
# graph = nx.read_graphml("../data/users_hate.graphml")
#
# for user_id in graph.nodes():
#
#     for att in graph_attributes:
#
#         if att in graph.node[user_id]:
#             del graph.node[user_id][att]
#
# df = pd.read_csv("../data/users_all.csv",  usecols=["user_id"])
#
# mapping = dict()
# for old_idx, new_idx in zip(df["user_id"].values, np.array(range(len(df["user_id"].values)))):
#     mapping[str(old_idx)] = int(new_idx)
#
# graph = nx.relabel_nodes(graph, mapping)
#
# nx.write_graphml(graph, "../data/users_clean.graphml")

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = graph sage input = = = = = = = = = = = = = = = = = = = = =

graph = nx.read_graphml("../data/users_clean.graphml")

nx.write_edgelist(graph, "../data/graph-input/users.edges", data=False)

df = pd.read_csv("../data/users_anon.csv")

print(len(cols_attr))
feats = np.nan_to_num(df[cols_glove].values)

ids = df["user_id"].values
hateful = df["hate"].values
suspended = df["is_63_2"].values

scaler = StandardScaler()
feats = scaler.fit_transform(feats)

f = open("../data/graph-input/users_hate_glove.content", "w")
for glove_feats, id_v, hate_v in zip(feats, ids, hateful):
    row = [id_v] + list(glove_feats) + [hate_v]
    row = list(map(str, row))
    f.write("\t".join(row))
    f.write("\n")
f.close()

f = open("../data/graph-input/users_suspended_glove.content", "w")
for glove_feats, id_v, sus_v in zip(feats, ids, suspended):
    row = [id_v] + list(glove_feats) + [sus_v]
    row = list(map(str, row))
    f.write("\t".join(row))
    f.write("\n")
f.close()

feats = np.nan_to_num(df[cols_attr + cols_glove].values)
for i in [0, 1, 2, 3, 4, 5, 6]:
    feats[:, i] = np.log(feats[:, i] + 1.0)

ids = df["user_id"].values
hateful = df["hate"].values
suspended = df["is_63_2"].values

scaler = StandardScaler()
feats = scaler.fit_transform(feats)

f = open("../data/graph-input/users_hate_all.content", "w")
for glove_feats, id_v, hate_v in zip(feats, ids, hateful):
    row = [id_v] + list(glove_feats) + [hate_v]
    row = list(map(str, row))
    f.write("\t".join(row))
    f.write("\n")
f.close()

f = open("../data/graph-input/users_suspended_all.content", "w")
for glove_feats, id_v, sus_v in zip(feats, ids, suspended):
    row = [id_v] + list(glove_feats) + [sus_v]
    row = list(map(str, row))
    f.write("\t".join(row))
    f.write("\n")
f.close()

