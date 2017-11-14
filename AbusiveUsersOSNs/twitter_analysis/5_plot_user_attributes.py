import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.algorithms import bootstrap
from matplotlib.ticker import FuncFormatter
from seaborn.utils import ci


def formatter(x, pos):
    if x == 0:
        return 0
    if x >= 100:
        return "{0}K".format(round(x/1000, 2))
    if x < 1:
        val = math.ceil(abs(np.log10(x)))
        number = int(x * 10 ** val)
        return "${" + str(number) + "}" + "*10^{-" + str(val) + "}$"
    else:
        return x

form = FuncFormatter(formatter)

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#F8414A", "#FD878D", "#385A89", "#5676A1", ]

df = pd.read_csv("../data/users_attributes.csv")
df = df[df.hate != "other"]


f, axzs = plt.subplots(2, 5, figsize=(10.8, 4))


attributes_all = [["statuses_count", "followers_count", "followees_count",
               "favorites_count", "betweenness"],
              ["median_int", "average_int", "eigenvector", "in_degree", "out_degree"]]

titles_all = [["\#statuses", "\#followers", "\#followees", "\#favorites", "betweenness"],
          ["avg(interval)", "median(interval)", "eigenvector", "in degree", "out degree"]]

for axs, attributes, titles in zip(axzs, attributes_all, titles_all):

    for axis, attribute, title in zip(axs, attributes, titles):
        N = 4
        men = [df[df.hate == "hateful"],
               df[df.hate_neigh],
               df[df.hate == "normal"],
               df[df.normal_neigh]]
        averages, averages_ci, medians, medians_ci = [], [], [], []
        for category in men:
            boots = bootstrap(category[attribute], func=np.nanmean, n_boot=10000)
            ci_tmp = ci(boots)
            average = (ci_tmp[0] + ci_tmp[1]) / 2
            ci_average = (ci_tmp[1] - ci_tmp[0]) / 2
            averages.append(average)
            averages_ci.append(ci_average)
            boots = bootstrap(category[attribute], func=np.nanmedian, n_boot=10000)
            ci_tmp = ci(boots)
            median = (ci_tmp[0] + ci_tmp[1]) / 2
            ci_median = (ci_tmp[1] - ci_tmp[0]) / 2
            medians.append(median)
            medians_ci.append(ci_median)

        ind = np.array([0, 1, 2, 3])
        width = .8
        rects = axis.bar(ind, medians, width, yerr=medians_ci, color=color_mine, ecolor="#46495C")
        axis.yaxis.set_major_formatter(form)
        axis.set_xticks([])
        axis.set_title(title)
        axis.set_ylabel("")
        axis.set_xlabel("")


f.legend( (rects[0], rects[1], rects[2], rects[3]),
          ("Hateful Users", "Hateful Neighborhood","Normal Users", "Normal Neighborhood"),
          loc='upper center',
          fancybox=True, shadow=True, ncol=4)
f.tight_layout(rect=[0, 0, 1, .95])


f.savefig("../imgs/attributes.pdf")
