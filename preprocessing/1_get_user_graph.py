from py2neo import Graph
import networkx as nx
import json

nx_graph = nx.DiGraph()

f = open("../secrets/twitter_neo4jsecret.json", 'r')
config_neo4j = json.load(f)
f.close()

graph = Graph(config_neo4j["host"], password=config_neo4j["password"])

for node in graph.data("""MATCH (a:User) WHERE a.virtual="F" RETURN a as val"""):
    n = dict(node["val"])

    for key, item in n.items():
        if type(item) is str:
            n[key] = str(item.encode('utf-8'))

    nx_graph.add_node(n["id"], **n)

    nx_graph.add_edge(n["id"], n["id"])

for node in graph.data(
        """MATCH (a:User)-[:retweeted]->(b:User) WHERE a.virtual="F" AND b.virtual="F" RETURN a.id as a, b.id as b"""):
    nx_graph.add_edge(node['a'], node['b'])

nx.write_graphml(nx_graph, "../data/preprocessing/users.graphml")
