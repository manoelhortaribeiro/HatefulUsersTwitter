import networkx as nx
import numpy as np

initial_belief = 1
k = 2

np.random.seed(1)
graph = nx.read_graphml("../data/users_infected.graphml")

slur_nodes = list(nx.get_node_attributes(graph, "slur"))
other_nodes = list(set(graph.nodes()).difference(set(slur_nodes)))
node_list = slur_nodes + other_nodes

transition_matrix = nx.adjacency_matrix(graph, nodelist=node_list).asfptype()
n = transition_matrix.shape[0]

for i in range(n):
    total = transition_matrix[i, :].sum()
    if total != 0:
        transition_matrix[i, :] = transition_matrix[i, :] / total


beliefs = np.zeros(len(node_list))
beliefs[:len(slur_nodes)] = initial_belief

for _ in range(k):
    out = transition_matrix.dot(beliefs)
    beliefs = out


final_beliefs_dict = dict()
for node, belief in zip(node_list, beliefs):
    final_beliefs_dict[node] = float(belief)

nx.set_node_attributes(graph, name="diffusion_slur", values=final_beliefs_dict)
nx.write_graphml(graph, "../data/users_infected_diffusion.graphml".format(k))
