from matplotlib.ticker import FuncFormatter
from seaborn.algorithms import bootstrap
import matplotlib.pyplot as plt
from seaborn.utils import ci
import seaborn as sns
import pandas as pd
import numpy as np
import math


def formatter(x, pos):
    if x == 0:
        return "0"
    if 0.01 < x < 10:
        return str(round(x,2))
    if 10 < x < 1000:
        return int(x)
    if x >= 1000:
        return "{0}K".format(int(x / 1000))
    else:
        return x


form = FuncFormatter(formatter)

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#F8414A", "#5676A1", "#FD878D", "#385A89", "#74C365", "#4A5D23"]

df = pd.read_csv("../data/users_all.csv")

df["tweet_number"] = df["tweet number"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["retweet_number"] = df["retweet number"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["number_urls"] = df["number urls"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["mentions"] = df["mentions"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["mentions"] = df["mentions"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["number hashtags"] = df["number hashtags"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["baddies"] = df["baddies"] / (df["tweet number"] + df["retweet number"] + df["quote number"])

f, axzs = plt.subplots(3, 6, figsize=(10.8, 6))
boxprops = dict(linewidth=0.3)
whiskerprops = dict(linewidth=0.3)
capprops = dict(linewidth=0.3)
medianprops = dict(linewidth=1)
attributes_all = [
    ["tweet_number", "retweet_number", "number_urls", "mentions", "number hashtags", "baddies"],
    ["statuses_count", "followers_count", "followees_count", "favorites_count", "average_int", "status length"],
    ["betweenness", "eigenvector", "in_degree", "out_degree", "sentiment", "subjectivity"]]

titles_all = [
    ["\%tweets", "\%retweets", "urls/tweet", "mentions/tweet", "hashtags/tweet", "profanity/tweet"],
    ["\#statuses", "\#followers", "\#followees", "\#favorites", "avg(interval)", "length"],
    ["betweenness", "eigenvector", "in degree", "out degree", "sentiment", "subjectivity"]]

for axs, attributes, titles in zip(axzs, attributes_all, titles_all):

    for axis, attribute, title in zip(axs, attributes, titles):
        N = 4
        men = [df[df.hate == "hateful"],
               df[df.hate == "normal"],
               df[df.hate_neigh],
               df[df.normal_neigh],
               df[df.is_63],
               df]
        tmp = []
        medians = []
        averages = []

        for category in men:
            tmp.append(category[attribute].values)
            medians.append(np.nanmedian(category[attribute].values))
            averages.append(np.nanmean(category[attribute].values))

        rects = sns.boxplot(data=tmp, palette=color_mine, showfliers=False, ax=axis, orient="v", width=0.8,
                            boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
                            medianprops=medianprops)

        axis.yaxis.set_major_formatter(form)

        axis.set_xticks([])
        axis.set_title(title)
        axis.set_ylabel("")
        axis.set_xlabel("")
        axis.axvline(1.5, ls='dashed', linewidth=0.3, color="#C0C0C0")
        axis.axvline(3.5, ls='dashed', linewidth=0.3, color="#C0C0C0")

ymin, _ = axzs[1][4].get_ylim()
axzs[1][4].set_ylim([ymin, 45000])
ymin, _ = axzs[2][1].get_ylim()
axzs[2][1].set_ylim([-0.00000001, 0.0000004])

# f.legend((rects[0], rects[1], rects[2], rects[3]),
#          ("Hateful Acc.", "Hateful Neigh.", "Normal Acc.", "Normal Neigh"),
#          loc='upper center',
#          fancybox=True, shadow=True, ncol=4)
f.tight_layout(rect=[0, 0, 1, .95])
f.savefig("../imgs/attributes.pdf")
