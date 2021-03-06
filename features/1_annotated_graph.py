import networkx as nx
import time
import csv

# Read annotated users

f = open("../data/extra/annotated_full.csv", "r")
csv_writer = csv.DictReader(f)

set_users = dict()
c = 0
tmp = set()
for line in csv_writer:
    if line["hate"] == '1':
        tmp.add(line["user_id"])
        set_users[line["user_id"]] = 1
    elif line["hate"] == "0":
        set_users[line["user_id"]] = 0
f.close()


# Set hate attributes

nx_graph = nx.read_graphml("../data/preprocessing/users_infected_diffusion.graphml")


nx.set_node_attributes(nx_graph, name="hate", values=-1)
nx.set_node_attributes(nx_graph, name="hate", values=set_users)
# Set hateful and normal neighbors attribute

nodes = nx_graph.nodes(data='hate')

hateful_neighbors = dict()
normal_neighbors = dict()

for i in nodes:
    if i[1] == 1:  # hateful node
        for j in nx_graph.neighbors(i[0]):
            hateful_neighbors[j] = True
    if i[1] == 0:
        for j in nx_graph.neighbors(i[0]):
            normal_neighbors[j] = True

nx.set_node_attributes(nx_graph, name="hateful_neighbors", values=False)
nx.set_node_attributes(nx_graph, name="hateful_neighbors", values=hateful_neighbors)
nx.set_node_attributes(nx_graph, name="normal_neighbors", values=False)
nx.set_node_attributes(nx_graph, name="normal_neighbors", values=normal_neighbors)


# Set node network-based attributes, such as betweenness and eigenvector

betweenness = nx.betweenness_centrality(nx_graph, k=16258, normalized=False)
eigenvector = nx.eigenvector_centrality(nx_graph)
in_degree = nx.in_degree_centrality(nx_graph)
out_degree = nx.out_degree_centrality(nx_graph)

nx.set_node_attributes(nx_graph, name="betweenness", values=betweenness)
nx.set_node_attributes(nx_graph, name="eigenvector", values=eigenvector)
nx.set_node_attributes(nx_graph, name="in_degree", values=in_degree)
nx.set_node_attributes(nx_graph, name="out_degree", values=out_degree)

nx.write_graphml(nx_graph, "../data/features/users_hate.graphml")
