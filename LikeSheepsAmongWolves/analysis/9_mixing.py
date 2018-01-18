import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms.assortativity import attribute_mixing_dict

df = pd.read_csv("../data/users_anon.csv")

confusion = [len(df[(df["hate"] == "hateful") & (df["is_63_2"])].index),
             len(df[(df["hate"] == "normal") & (df["is_63_2"])].index),
             len(df[(df["hate"] == "other") & (df["is_63_2"])].index)]

print(confusion)

confusion_norm = [confusion[0]/len(df[df["hate"] == "hateful"]),
                  confusion[1] / len(df[df["hate"] == "normal"]),
                  confusion[2] / len(df[df["hate"] == "other"])
                  ]

print(confusion_norm)
