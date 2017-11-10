import networkx as nx
import pandas as pd

nx_graph = nx.read_graphml("../data/users_hate.graphml")


betweenness = nx.betweenness_centrality(nx_graph, k=2, normalized=True)
eigenvector = nx.eigenvector_centrality(nx_graph)
in_degree = nx.in_degree_centrality(nx_graph)
out_degree = nx.out_degree_centrality(nx_graph)
statuses_count = nx.get_node_attributes(nx_graph, "statuses_count")
followers_count = nx.get_node_attributes(nx_graph, "followers_count")
followees_count = nx.get_node_attributes(nx_graph, "followees_count")
favorites_count = nx.get_node_attributes(nx_graph, "favorites_count")
listed_count = nx.get_node_attributes(nx_graph, "listed_count")

hate = nx.get_node_attributes(nx_graph, "hate")
hate_n = nx.get_node_attributes(nx_graph, "hateful_neighbors")
normal_n = nx.get_node_attributes(nx_graph, "normal_neighbors")
users = []

for user_id in hate.keys():
    hateful = "other"
    if hate[user_id] == 1:
        hateful = "hateful"
    elif hate[user_id] == 0:
        hateful = "normal"

    hate_neigh = "False"
    if hate_n[user_id]:
        hateful_neigh = True
    else:
        hateful_neigh = False

    normal_neigh = "False"
    if normal_n[user_id]:
        normal_neigh = True
    else:
        normal_neigh = False

    users.append((user_id, hateful, hateful_neigh, normal_neigh,  # General Stuff
                  statuses_count[user_id], followers_count[user_id], followees_count[user_id],  # Numeric attributes
                  favorites_count[user_id], listed_count[user_id],
                  betweenness[user_id], eigenvector[user_id],  # Network Attributes
                  in_degree[user_id], out_degree[user_id]))


columns = ["user_id", "hate", "hate_neigh", "normal_neigh",
           "statuses_count", "followers_count", "followees_count", "favorites_count", "listed_count",
           "betweenness", "eigenvector", "in_degree", "out_degree"]
df = pd.DataFrame.from_records(users, columns=columns)
df.to_csv("../data/users_attributes.csv", index=False)
