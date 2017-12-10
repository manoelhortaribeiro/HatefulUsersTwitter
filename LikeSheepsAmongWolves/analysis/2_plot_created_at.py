import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#F8414A", "#5676A1", "#FD878D", "#385A89"]

df = pd.read_csv("../data/users_created_at.csv")

men = [df[df.hate == "hateful"], df[df.hate == "normal"], df[df.hate_neigh], df[df.normal_neigh]]

tmp = []

for category in men:
    tmp.append(category["created_at"].values)

f, axs = plt.subplots(1, 1, figsize=(4.4, 3.2))
sns.violinplot(ax=axs, data=tmp, palette=color_mine, orient="h")
axs.set_ylabel("")
axs.set_xlabel("")


x = df.created_at.values
x_ticks = np.arange(min(x), max(x)+1, 3.154e+7)
axs.set_xticks(np.arange(min(x), max(x)+1, 3.154e+7))
f.canvas.draw()
axs.set_title("Creation Date of Users")

labels = [datetime.fromtimestamp(item).strftime('%Y-%m') for item in x_ticks]
axs.set_xticklabels(labels, rotation=35)
axs.set_yticklabels(["", "", "", ""], rotation=20)
f.tight_layout()

f.savefig("../imgs/created_at.pdf")
