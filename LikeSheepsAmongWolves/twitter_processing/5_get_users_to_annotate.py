import networkx as nx
import numpy as np
import random
import csv

np.random.seed(1234)
random.seed(1234)

N = 6000

nx_graph = nx.read_graphml("../data/users_infected_diffusion1.graphml")

diffusion_slur = nx.get_node_attributes(nx_graph, name="diffusion_slur")

screen_names = nx.get_node_attributes(nx_graph, name="screen_name")

in_degree = nx_graph.in_degree()

strata1, strata2, strata3, strata4 = [], [], [], []

sum_vals = 0

for key in sorted(diffusion_slur):

    if diffusion_slur[key] < .25:
        strata1.append(int(key))

    if .50 > diffusion_slur[key] >= .25:
        strata2.append(int(key))

    if .75 > diffusion_slur[key] >= .50:
        strata3.append(int(key))

    if diffusion_slur[key] >= .75:
        strata4.append(int(key))

sample_strata4 = np.random.choice(strata4, size=min(int(N / 4), len(strata4)), replace=False)

sample_strata1 = np.random.choice(strata1, size=int(N / 4), replace=False)

sample_strata2 = np.random.choice(strata2, size=int(N / 4), replace=False)

sample_strata3 = np.random.choice(strata3, size=int(N / 4), replace=False)

f = open("../data/users_to_annotate.csv", "w")

csv_writer = csv.writer(f)

csv_writer.writerow(["user_id", "screen_name", "twitter", "diffusion_slur", "stratum"])

count = 0

sample = []

for strata in [sample_strata1, sample_strata2, sample_strata3, sample_strata4]:
    count += 1
    for key in strata:
        sample.append([int(key),
                       screen_names[str(key)],
                       "https://twitter.com/{0}".format(screen_names[str(key)]),
                       diffusion_slur[str(key)], count])

print(sample)
random.shuffle(sample)

for row in sample:
    csv_writer.writerow(row)
f.close()

# starts with 24075855,roopikarisam,https://twitter.com/roopikarisam,0.0,1
