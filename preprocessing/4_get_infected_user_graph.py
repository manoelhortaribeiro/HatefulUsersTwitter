import networkx as nx
import csv
import re

l = open("../data/extra/lexicon.txt", "r")
regexp = ""
for line in l.readlines():
    regexp += "({0})|".format(line.rstrip())
l.close()
regexp = regexp[:-1]
regexp = re.compile(regexp)

f = open("../data/preprocessing/tweets.csv", "r")
re.match(regexp, "")
csv_writer = csv.DictReader(f)

set_users = dict()

for line in csv_writer:
    text = regexp.search(line["tweet_text"])
    retweet = regexp.search(line["rt_text"])
    quote = regexp.search(line["qt_text"])
    if text is not None or retweet is not None or quote is not None:
        set_users[line["user_id"]] = True
f.close()


nx_graph = nx.read_graphml("../data/preprocessing/users.graphml")
nx_graph = nx_graph.reverse(copy=False)
nx.set_node_attributes(nx_graph, name="slur", values=set_users)
nx.write_graphml(nx_graph, "../data/preprocessing/users_infected.graphml")
