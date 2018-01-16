from LikeSheepsAmongWolves.tmp.utils import cols_attr, cols_glove, cols_empath, graph_attributes
import networkx as nx
import pandas as pd
import numpy as np
import gc

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = cleans graph = = = = = = = = = = = = = = = = = = = = = = = = =
# graph = nx.read_graphml("../data/users_hate.graphml")
#
# for user_id in graph.nodes():
#
#     for att in graph_attributes:
#
#         if att in graph.node[user_id]:
#             del graph.node[user_id][att]
#
# nx.write_graphml(graph, "../data/users_clean.graphml")
#
# del graph

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = get cols = = = = = = = = = = = = = = = = = = = = = = = = = = =
df = pd.read_csv("../data/users_all.csv", index_col=0)

cols = df.columns.values

del df
gc.collect()

graph = nx.read_graphml("../data/users_clean.graphml")

for col in cols:

    if col == "hashtags":
        continue

    df = pd.read_csv("../data/users_all.csv", usecols=["user_id", col])

    col_dict = dict()

    for i, v in zip(df["user_id"].values, df[col].values):
        if type(v) == np.float64:
            v = float(v)
        elif type(v) == np.int64:
            v = int(v)
        elif type(v) == np.bool:
            v = bool(v)
        elif type(v) == np.bool_:
            v = bool(v)

        col_dict[str(i)] = v

    nx.set_node_attributes(graph, values=col_dict, name=col)

nx.write_graphml(graph, "../data/users_all.graphml")

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = makes new mapping  = = = = = = = = = = = = = = = = = = = = = =
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

# # = = = = = = = = = = = = = = = = = = = = = = = = = = = = makes new mapping = = = = = = = = = = = = = = = = = = = = =
df = pd.read_csv("../data/users_all.csv",  usecols=["user_id"])
graph = nx.read_graphml("../data/users_all.graphml")

print(df)

mapping = dict()
for old_idx, new_idx in zip(df["user_id"].vales, np.array(range(len(df["user_id"].values)))):
    mapping[int(old_idx)] = new_idx

graph = nx.relabel_nodes(graph, mapping)

nx.write_graphml(graph, "../data/users_anon.graphml")
