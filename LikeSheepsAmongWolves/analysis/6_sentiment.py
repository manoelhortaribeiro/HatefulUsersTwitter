from LikeSheepsAmongWolves.tmp.utils import formatter
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np

form = FuncFormatter(formatter)

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#F8414A", "#5676A1", "#FD878D", "#385A89"]

df = pd.read_csv("../data/users_all.csv")
df2 = pd.read_csv("../data/users_created_at.csv")

f, axzs = plt.subplots(1, 3, figsize=(5.4, 2))
axzs = [axzs]
boxprops = dict(linewidth=0.3)
whiskerprops = dict(linewidth=0.3)
capprops = dict(linewidth=0.3)
medianprops = dict(linewidth=1)

attributes_all = [["sentiment", "subjectivity", "baddies"]]
titles_all = [["sentiment", "subjectivity", "baddies"]]

rects = None
first = True
for axs, attributes, titles in zip(axzs, attributes_all, titles_all):

    for axis, attribute, title in zip(axs, attributes, titles):
        N = 4
        men = [df[df.hate == "hateful"],
               df[df.hate == "normal"],
               df[df.hate_neigh],
               df[df.normal_neigh],
               df[df["is_63"] == False],
               df[df["is_63"] == True]]
        tmp = []
        medians, medians_ci = [], []
        averages, averages_ci = [], []

        for category, color in zip(men, color_mine):
            tmp.append(category[attribute].values)

        sns.boxplot(data=tmp, palette=color_mine, showfliers=False, ax=axis, orient="v", width=0.8,
        boxprops = boxprops, whiskerprops = whiskerprops, capprops = capprops, medianprops = medianprops)

        _, n_h = stats.ttest_ind(tmp[0], tmp[1], equal_var=False)
        _, nn_nh = stats.ttest_ind(tmp[2], tmp[3], equal_var=False)
        _, s = stats.ttest_ind(tmp[3], tmp[4], equal_var=False)

        print(title)
        print(n_h)
        print(nn_nh)
        print(s)

        axis.yaxis.set_major_formatter(form)

        axis.set_xticks([])
        axis.set_title(title)
        axis.set_ylabel("")
        axis.set_xlabel("")
        axis.axvline(1.5, ls='dashed', linewidth=0.3, color="#C0C0C0")

        axzs[0][0].set_ylim(-.15, .4)
        axzs[0][1].set_ylim(.30, .70)
        axzs[0][2].set_ylim(-20, 100)

        f.tight_layout(rect=[0, 0, 1, 1])

        f.savefig("../imgs/sentiment.pdf")
