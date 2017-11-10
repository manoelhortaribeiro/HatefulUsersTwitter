import networkx as nx
import csv

f = open("../data/annotated.csv", "r")
csv_writer = csv.DictReader(f)

set_users = dict()

for line in csv_writer:
    if line["hate"] == '1':
        set_users[line["user_id"]] = 1
    elif line["hate"] == "0":
        set_users[line["user_id"]] = 0


nx_graph = nx.read_graphml("../data/users_infected_diffusion1.graphml")
nx.set_node_attributes(nx_graph, name="hate", values=-1)
nx.set_node_attributes(nx_graph, name="hate", values=set_users)

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

nx.write_graphml(nx_graph, "../data/users_hate.graphml")
