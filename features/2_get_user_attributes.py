import networkx as nx
import pandas as pd

nx_graph = nx.read_graphml("../data/features/users_hate.graphml")

hate = nx.get_node_attributes(nx_graph, "hate")

hate_n = nx.get_node_attributes(nx_graph, "hateful_neighbors")
normal_n = nx.get_node_attributes(nx_graph, "normal_neighbors")
betweenness = nx.get_node_attributes(nx_graph, "betweenness")
eigenvector = nx.get_node_attributes(nx_graph, "eigenvector")
in_degree = nx.get_node_attributes(nx_graph, "in_degree")
out_degree = nx.get_node_attributes(nx_graph, "out_degree")
statuses_count = nx.get_node_attributes(nx_graph, "statuses_count")
followers_count = nx.get_node_attributes(nx_graph, "followers_count")
followees_count = nx.get_node_attributes(nx_graph, "followees_count")
favorites_count = nx.get_node_attributes(nx_graph, "favorites_count")
listed_count = nx.get_node_attributes(nx_graph, "listed_count")

users = []

for user_id in hate.keys():
    hateful = "other"

    if hate[user_id] == 1:
        hateful = "hateful"

    elif hate[user_id] == 0:
        hateful = "normal"

    users.append((user_id, hateful, hate_n[user_id], normal_n[user_id],  # General Stuff
                  statuses_count[user_id], followers_count[user_id], followees_count[user_id],
                  favorites_count[user_id], listed_count[user_id],  # Numeric attributes
                  betweenness[user_id], eigenvector[user_id],  # Network Attributes
                  in_degree[user_id], out_degree[user_id]))

columns = ["user_id", "hate", "hate_neigh", "normal_neigh", "statuses_count", "followers_count", "followees_count",
           "favorites_count", "listed_count", "betweenness", "eigenvector", "in_degree", "out_degree"]

df = pd.DataFrame.from_records(users, columns=columns)

df.to_csv("../data/features/users_attributes.csv", index=False)
