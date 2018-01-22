import json

import networkx as nx
import numpy as np
import pandas as pd
from networkx.readwrite import json_graph

from tmp.utils import cols_attr, cols_glove

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

df = pd.read_csv("../data/users_anon.csv", usecols=["user_id", "hate"])

# Makes -class_map.json

class_map = dict()

for user_id, hate in zip(df["user_id"].values, df["hate"].values):
    if hate == "hateful":
        class_map[str(user_id)] = [1, 0]
    if hate == "normal":
        class_map[str(user_id)] = [0, 1]
    if hate == "other":
        class_map[str(user_id)] = [0, 0]

f = open("../data/graph-input/users_anon-class_map.json", "w")
json.dump(class_map, f)
f.close()

# Makes -G.json
np.random.seed(123)
hateful = df[df.hate == "hateful"][["user_id"]]
hateful = hateful.sample(frac=1)
hateful_splits = np.array_split(hateful, 10)
normal = df[df.hate == "normal"][["user_id"]]
normal = normal.sample(frac=1)
normal_splits = np.array_split(normal, 10)

for fold in range(5):

    train = pd.concat(hateful_splits[:fold] + hateful_splits[fold+1:] +
                      normal_splits[:fold] + normal_splits[fold+1:])
    test = pd.concat([hateful_splits[fold]] + [normal_splits[fold]])

    val_d, test_d = dict(), dict()

    graph = nx.read_graphml("../data/users_clean.graphml")

    for user_id in df["user_id"].values:
        val_d[str(user_id)] = False
        test_d[str(user_id)] = True

    for user_id in test["user_id"].values:
        val_d[str(user_id)] = True
        test_d[str(user_id)] = False

    for user_id in train["user_id"].values:
        val_d[str(user_id)] = False
        test_d[str(user_id)] = False

    nx.set_node_attributes(graph, values=val_d, name='val')
    nx.set_node_attributes(graph, values=test_d, name='test')
    nx.set_edge_attributes(graph, values=False, name='test_removed')
    nx.set_edge_attributes(graph, values=False, name='train_removed')

    graph = nx.DiGraph(graph)
    data = json_graph.node_link_data(graph)
    data["directed"] = True

    for i in range(len(data["nodes"])):
        data["nodes"][i]['id'] = int(data["nodes"][i]['id'])

    for i in range(len(data["links"])):
        data["links"][i]['target'] = int(data["links"][i]['target'])
        data["links"][i]['source'] = int(data["links"][i]['source'])

    f = open("../data/graph-input/users_anon{0}-G.json".format(fold), "w")
    json.dump(data, f)
    f.close()

# Makes -id_map.json

id_map = dict()

for node in graph:
    id_map[str(node)] = int(node)

f = open("../data/graph-input/users_anon-id_map.json", "w")
json.dump(id_map, f)
f.close()

# Makes -feats.npy

df = pd.read_csv("../data/users_anon.csv")

feats = df[cols_glove].values

# # Logistic gets thrown off by big counts, so log transform num comments and score
# for i in [0, 1, 2, 3, 4, 5]:
#     feats[:, 0] = np.log(feats[:, 0] + 1.0)

f = open("../data/graph-input/users_anon-feats.npy", "wb")
np.save(f, np.nan_to_num(feats))
f.close()
