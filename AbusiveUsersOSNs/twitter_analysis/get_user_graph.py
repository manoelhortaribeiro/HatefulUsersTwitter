from py2neo import Graph
import json
import networkx as nx

nx_graph = nx.DiGraph()

f = open("./twitter_neo4jsecret.json", 'r')
config_neo4j = json.load(f)
f.close()
graph = Graph(config_neo4j["host"], password=config_neo4j["password"])

for node in graph.data("""MATCH (a:User) WHERE a.virtual="F" RETURN a as val LIMIT 1"""):
    n = dict(node["val"])
    nx_graph.add_node(n=n["id"], **dict(node["val"]))

for node in graph.data("""MATCH (a:User)-->(b:User) WHERE a.virtual="F" AND b.virtual="F" RETURN a.id as a, b.id as b"""):
    nx_graph.add_edge(node['a'], node['b'])

nx.write_graphml(nx_graph, "./users.graphml")
