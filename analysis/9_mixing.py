import networkx as nx
import pandas as pd
import numpy as np
from networkx.algorithms.assortativity import attribute_mixing_dict

g = nx.read_graphml("../data/users_clean.graphml")
df = pd.read_csv("../data/users_anon.csv")

hate_dict = {}
susp_dict = {}
for idv, hate, susp in zip(df.user_id.values, df.hate.values, df.is_63_2.values):
    hate_dict[str(idv)] = hate
    susp_dict[str(idv)] = susp

nx.set_node_attributes(g, name="hate", values=hate_dict)
nx.set_node_attributes(g, name="susp", values=susp_dict)

mixing = attribute_mixing_dict(g, "hate")

print(mixing)
print(" hate  -> hate   ",
      mixing["hateful"]["hateful"] /
      (mixing["hateful"]["other"] + mixing["hateful"]["normal"] + mixing["hateful"]["hateful"])
      / (544 / 100386))
print(" hate  -> normal ", mixing["hateful"]["normal"] /
      (mixing["hateful"]["other"] + mixing["hateful"]["normal"] + mixing["hateful"]["hateful"])
      / (4427 / 100386))
print("normal -> normal ", mixing["normal"]["normal"] /
      (mixing["normal"]["other"] + mixing["normal"]["normal"] + mixing["normal"]["hateful"])
      / (4427 / 100386))
print("normal -> hate   ", mixing["normal"]["hateful"] /
      (mixing["normal"]["other"] + mixing["normal"]["normal"] + mixing["normal"]["hateful"])
      / (544 / 100386))

mixing = attribute_mixing_dict(g, "susp")
print(mixing)
print(" susp  -> susp   ", mixing[True][True] / (mixing[True][True] + mixing[True][False])
      ) #/(668 / 100386))
print(" susp  -> active ", mixing[True][False] / (mixing[True][True] + mixing[True][False])
      ) #/(99718/100386))
print("active -> active ", mixing[False][False] / (mixing[False][True] + mixing[False][False])
      ) #/(99718/100386))
print("active -> susp   ", mixing[False][True] / (mixing[False][True] + mixing[False][False])
      ) #/(668 / 100386))

del g

#
# df = pd.read_csv("../data/users_anon.csv")
#
#
# men = [df[df.hate == "hateful"],
#                df[df.hate == "normal"],
#                df[df.hate_neigh],
#                df[df.normal_neigh],
#                df[df.is_63_2 == True],
#                df[df.is_63_2 == False]]
#
# for i in men:
#     print(len(i.values))
#
# confusion = [len(df[(df["hate"] == "hateful") & (df["is_63"])].index),
#              len(df[(df["hate"] == "normal") & (df["is_63"])].index),
#              len(df[(df["hate"] == "other") & (df["is_63"])].index)]
#
# print(confusion)
#
# confusion_norm = [confusion[0]/len(df[df["hate"] == "hateful"]),
#                   confusion[1] / len(df[df["hate"] == "normal"]),
#                   confusion[2] / len(df[df["hate"] == "other"])
#                   ]
#
# print(confusion_norm)
