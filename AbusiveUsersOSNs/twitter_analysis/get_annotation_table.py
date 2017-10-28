import networkx as nx
import json
import csv
import re

l = open("./lexicon.txt", "r")
regexp = ""
for line in l.readlines():
    regexp += "({0})|".format(line.rstrip())
l.close()
regexp = regexp[:-1]
print(regexp)
regexp = re.compile(regexp)

print(regexp.search("asdas you are a ((fucking)) gsoy"))
f = open("./tweets.csv", "r")
re.match(regexp, "")
csv_writer = csv.DictReader(f)

print(csv_writer.fieldnames)

set_users = dict()

for line in csv_writer:
    text = regexp.search(line["tweet_text"])
    retweet = regexp.search(line["rt_text"])
    quote = regexp.search(line["qt_text"])
    if text is not None or retweet is not None or quote is not None:
        set_users[line["user_id"]] = True

print(len(set_users))

nx_graph = nx.read_graphml("./users.graphml")
nx.set_node_attributes(nx_graph, name="slur", values=set_users)
nx.write_graphml(nx_graph, "./users_infected.graphml")
