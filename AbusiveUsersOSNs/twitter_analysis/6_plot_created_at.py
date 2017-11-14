import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = {"hateful": "#F8414A", "normal": "#385A89"}

df1 = pd.read_csv("../data/created_at_hate.csv")
df1 = df1[df1.hate != "None"]

f, axs = plt.subplots(1, 1, figsize=(9, 1.8))
sns.violinplot(y="hate", x="created_at", ax=axs, data=df1, estimator=np.average, ci=95, vert=True, palette=color_mine)
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

f.savefig("../imgs/created_at.pdf")
