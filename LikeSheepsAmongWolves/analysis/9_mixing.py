import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms.assortativity import attribute_mixing_dict

g = nx.read_graphml("../data/users_hate.graphml")


print(attribute_mixing_dict(g, "hate"))


df = pd.read_csv("../data/users_all.csv")

confusion = [[len(df[(df["hate"] == "hateful") & (df["is_50"])].index),
              len(df[(df["hate"] == "hateful") & (df["is_63"])].index),
              len(df[(df["hate"] == "hateful")].index)],
             [len(df[(df["hate"] == "normal") & (df["is_50"])].index),
              len(df[(df["hate"] == "normal") & (df["is_63"])].index),
              len(df[(df["hate"] == "normal") & (df["is_63"] == False) & (df["is_50"] == False)].index)],
             [len(df[(df["hate"] == "other") & (df["is_50"])].index),
              len(df[(df["hate"] == "other") & (df["is_63"])].index),
              len(df[(df["hate"] == "other") & (df["is_63"] == False) & (df["is_50"] == False)].index)]]

confusion = np.array(confusion)
print(confusion.shape)
v = np.sum(confusion, axis=1)
print(confusion)
