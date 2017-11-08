import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sns.set(style="whitegrid", font="serif")

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
#
# axs = zip(*axs)
# attributes = ["statuses_count", "followers_count", "followees_count", "favorites_count", "listed_count"]
# titles = ["#statuses", "#followers", "#followees", "#favorites", "#listed"]
#
#
# for axis, attribute, title in zip(axs, attributes, titles):
#     sns.barplot(x="hate", y=attribute,  ax=axis[0], data=df, estimator=np.average, ci=95)
#     axis[0].set_xlabel("")
#     axis[0].set_ylabel("")
#     axis[0].set_title(title)
#
#     axis[0].set_xticks([])
#     sns.boxplot(x="hate", y=attribute, data=df, ax=axis[1],  showfliers=False)
#     axis[1].set_ylabel("")
#     axis[1].set_xlabel("")
#     axis[1].set_yscale("log")
#     axis[1].set_xticks([0 ,1], ["normal", "hateful"])
#
#
# f.tight_layout()
#
# f.savefig("attributes.pdf")
