import networkx as nx
import numpy as np
import csv

k = 1000

nx_graph = nx.read_graphml("./users_infected_diffusion1.graphml")
diffusion_slur = nx.get_node_attributes(nx_graph, name="diffusion_slur")
screen_names = nx.get_node_attributes(nx_graph, name="screen_name")

sum_vals = 0
for key in diffusion_slur:
    sum_vals += diffusion_slur[key] ** 2

items = list(diffusion_slur.items())
p = []
users = []
for key, item in items:
    p.append((item**2)/sum_vals)
    users.append(key)

users_to_annotate = np.random.choice(users, size=k, replace=False, p=p)

f = open("./users_to_annotate.csv", "w")
csv_writer = csv.writer(f)

csv_writer.writerow(["user_id", "screen_name", "diffusion_slur"])

for user in users_to_annotate:
    csv_writer.writerow([int(user), screen_names[user], diffusion_slur[user]])
