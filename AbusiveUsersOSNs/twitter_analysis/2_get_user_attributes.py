import networkx as nx
import pandas as pd

nx_graph = nx.read_graphml("./users_hate.graphml")
betweenness = nx.betweenness_centrality(nx_graph, k=100, normalized=True)
print("asdasd")
eigenvector = nx.eigenvector_centrality(nx_graph)
print("asdasd")

in_degree = nx.in_degree_centrality(nx_graph)
print("asdasd")

out_degree = nx.out_degree_centrality(nx_graph)
print("asdasd")

# hate = nx.get_node_attributes(nx_graph, "hate")
# statuses_count = nx.get_node_attributes(nx_graph, "statuses_count")
# followers_count = nx.get_node_attributes(nx_graph, "followers_count")
# followees_count = nx.get_node_attributes(nx_graph, "followees_count")
# favorites_count = nx.get_node_attributes(nx_graph, "favorites_count")
# listed_count = nx.get_node_attributes(nx_graph, "listed_count")
# users = []
# for user_id in hate.keys():
#     try:
#         hateful = "other"
#         if hate[user_id] == 1:
#             hateful = "hateful"
#         elif hate[user_id] == 0:
#             hateful = "normal"
#
#         users.append((user_id,
#                       hateful,
#                       statuses_count[user_id],
#                       followers_count[user_id],
#                       followees_count[user_id],
#                       favorites_count[user_id],
#                       listed_count[user_id]))
#     except KeyError:
#         print(user_id)
#
#
# columns = ["user_id", "hate", "statuses_count", "followers_count", "followees_count", "favorites_count", "listed_count"]
# df = pd.DataFrame.from_records(users, columns=columns)
# df.to_csv("./users_attributes.csv", index=False)
