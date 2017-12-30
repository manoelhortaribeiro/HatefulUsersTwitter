from LikeSheepsAmongWolves.tmp.utils import formatter
from matplotlib.ticker import FuncFormatter
from seaborn.algorithms import bootstrap
import scipy.stats as stats
import matplotlib.pyplot as plt
from seaborn.utils import ci
import seaborn as sns
import pandas as pd
import numpy as np
import datetime


form = FuncFormatter(formatter)

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#F8414A", "#5676A1", "#FD878D", "#385A89"]

df = pd.read_csv("../data/users_all.csv")
df2 = pd.read_csv("../data/users_created_at.csv")


f, axzs = plt.subplots(1, 5, figsize=(10.8, 2))
axzs = [axzs]
boxprops = dict(linewidth=0.3)
whiskerprops = dict(linewidth=0.3)
capprops = dict(linewidth=0.3)
medianprops = dict(linewidth=1)

df["tweet_number"] = df["tweet number"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df2["created_at"] = -(df2["created_at"] - datetime.datetime(2017, 12, 29).timestamp())/86400

df = pd.merge(df, df2, on="user_id", how="inner")

df["statuses_count"] = df["statuses_count"] / df["created_at"]
df["followers_count"] = df["followers_count"] / df["created_at"]
df["followees_count"] = df["followees_count"] / df["created_at"]

attributes_all = [["statuses_count", "followers_count", "followees_count", "favorites_count", "time_diff"]]

titles_all = [["\#statuses/day", "\#followers/day", "\#followees/day", "\#favorites", "avg(interval)"]]

first = True
for axs, attributes, titles in zip(axzs, attributes_all, titles_all):

    for axis, attribute, title in zip(axs, attributes, titles):
        N = 4
        men = [df[df.hate_x == "hateful"],
               df[df.hate_x == "normal"],
               df[df.hate_neigh_x],
               df[df.normal_neigh_x]]
        tmp = []
        medians, medians_ci = [], []
        averages, averages_ci = [], []

        for category in men:
            boots = bootstrap(category[attribute], func=np.nanmean, n_boot=1000)
            ci_tmp = ci(boots)
            average = (ci_tmp[0] + ci_tmp[1]) / 2
            ci_average = (ci_tmp[1] - ci_tmp[0]) / 2
            averages.append(average)
            averages_ci.append(ci_average)
            boots = bootstrap(category[attribute], func=np.nanmedian, n_boot=1000)
            ci_tmp = ci(boots)
            median = (ci_tmp[0] + ci_tmp[1]) / 2
            ci_median = (ci_tmp[1] - ci_tmp[0]) / 2
            medians.append(median)
            medians_ci.append(ci_median)

            tmp.append(category[attribute].values)

        ind = np.array([0, 1, 2, 3])
        width = .6

        _, n_h = stats.ttest_ind(tmp[0], tmp[1], equal_var=False)
        _, nn_nh = stats.ttest_ind(tmp[1], tmp[2], equal_var=False)

        print(title)
        print(n_h)
        print(nn_nh)

        rects = axis.bar(ind, averages, width, yerr=averages_ci, color=color_mine,
                         ecolor="#212823", edgecolor=["#4D1A17"]*4, linewidth=.3)

        axis.yaxis.set_major_formatter(form)

        axis.set_xticks([])
        axis.set_title(title)
        axis.set_ylabel("")
        axis.set_xlabel("")
        axis.axvline(1.5, ls='dashed', linewidth=0.3, color="#C0C0C0")

f.legend((rects[0], rects[1], rects[2], rects[3]),
         ('Hateful User', 'Normal User', 'Hateful Neigh.', 'Normal Neigh.', 'Suspended', 'All'),
         loc='upper center',
         fancybox=True, shadow=True, ncol=6)
f.tight_layout(rect=[0, 0, 1, .95])

f.savefig("../imgs/attributes.pdf")
