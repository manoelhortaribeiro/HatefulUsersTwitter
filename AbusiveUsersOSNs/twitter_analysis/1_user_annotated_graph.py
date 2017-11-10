import networkx as nx
import csv

f = open("./data/annotated.csv", "r")
csv_writer = csv.DictReader(f)

set_users = dict()

for line in csv_writer:
    if line["hate"] == '1':
        set_users[line["user_id"]] = 1
    elif line["hate"] == "0":
        set_users[line["user_id"]] = 0


nx_graph = nx.read_graphml("./data/users_infected_diffusion1.graphml")
nx.set_node_attributes(nx_graph, name="hate", values=-1)
nx.set_node_attributes(nx_graph, name="hate", values=set_users)
nx.write_graphml(nx_graph, "./data/users_hate.graphml")

