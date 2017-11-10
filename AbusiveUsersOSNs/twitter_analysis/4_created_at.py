import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.rc('font', family='serif')
sns.set(style="whitegrid", font="serif")

# This is in case created_at2.csv is not generated
# df1 = pd.read_csv("./created_at.csv")
# df2 = pd.read_csv("./users_attributes.csv")
# hateful = dict()
#
# for row in df2.iterrows():
#     if row[1][1] == "hateful":
#         hateful[row[1][0]] = "hateful"
#     if row[1][1] == "normal":
#         hateful[row[1][0]] = "normal"
#
# to_append = list()
# for row in df1.iterrows():
#     if row[1][0] in hateful:
#         to_append.append(hateful[row[1][0]])
#     else:
#         to_append.append("None")
#
# df1['hate'] = pd.Series(to_append, index=df1.index)
#
# df1.to_csv("created_at2.csv", index=False)

df1 = pd.read_csv("./created_at2.csv")
df1 = df1[df1.hate != "None"]

f, axs = plt.subplots(1, 1, figsize=(9, 1.8))
sns.violinplot(y="hate", x="created_at", ax=axs, data=df1, estimator=np.average, ci=95, vert=True)
axs.set_ylabel("")
axs.set_xlabel("")

x = df1.created_at.values
x_ticks = np.arange(min(x), max(x)+1, 3.154e+7)
axs.set_xticks(np.arange(min(x), max(x)+1, 3.154e+7))
f.canvas.draw()
axs.set_title("Creation Date of Users")

labels = [datetime.fromtimestamp(item).strftime('%Y-%m') for item in x_ticks]
axs.set_xticklabels(labels, rotation=30)
axs.set_yticklabels(["normal", "hateful"], rotation=20)
f.tight_layout()

f.savefig("created_at.pdf")
