from matplotlib.ticker import FuncFormatter
import scipy.stats as stats
from seaborn.algorithms import bootstrap
import matplotlib.pyplot as plt
from seaborn.utils import ci
import seaborn as sns
import pandas as pd
import numpy as np


def formatter(x, pos):
    if x == 0:
        return "0"
    if 0.01 < x < 10:
        return str(round(x, 2))
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

df["followers_followees"] = df["followers_count"] / (df["followees_count"])
df["number_urls"] = df["number urls"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["number hashtags"] = df["number hashtags"] / (df["tweet number"] + df["retweet number"] + df["quote number"])

f, axzs = plt.subplots(1, 3, figsize=(5.4, 2))
boxprops = dict(linewidth=0.3)
whiskerprops = dict(linewidth=0.3)
capprops = dict(linewidth=0.3)
medianprops = dict(linewidth=1)

attributes_all = [
    ["followers_followees", "number_urls", "number hashtags"]]

titles_all = [
    ["\#followers/followees", "\#URLs/tweet", "hashtags/tweet"]]

x_label_all = [
    ["(a)", "(b)", "(c)"]]

for axs, attributes, titles, x_labels in zip([axzs], attributes_all, titles_all, x_label_all):

    for axis, attribute, title, x_label in zip(axs, attributes, titles, x_labels):
        N = 4
        men = [df[df.hate == "hateful"],
               df[df.hate == "normal"],
               df[df.hate_neigh],
               df[df.normal_neigh],
               df[df.is_63],
               df[df.is_63 == False]]
        tmp = []
        medians, medians_ci = [], []
        averages, averages_ci = [], []

        for category in men:
            # boots = bootstrap(category[attribute], func=np.nanmean, n_boot=1000)
            # ci_tmp = ci(boots)
            # average = (ci_tmp[0] + ci_tmp[1]) / 2
            # ci_average = (ci_tmp[1] - ci_tmp[0]) / 2
            # averages.append(average)
            # averages_ci.append(ci_average)
            # boots = bootstrap(category[attribute], func=np.nanmedian, n_boot=1000)
            # ci_tmp = ci(boots)
            # median = (ci_tmp[0] + ci_tmp[1]) / 2
            # ci_median = (ci_tmp[1] - ci_tmp[0]) / 2
            # medians.append(median)
            # medians_ci.append(ci_median)
            w_inf = category[attribute].values
            non_inf = w_inf[w_inf < 1E308]
            tmp.append(non_inf)

        ind = np.array([0, 1, 2, 3, 4, 5])
        width = .6




        _, n_h = stats.ttest_ind(tmp[0], tmp[1], equal_var=False)
        _, nn_nh = stats.ttest_ind(tmp[1], tmp[2], equal_var=False)
        _, s_ns = stats.ttest_ind(tmp[3], tmp[4], equal_var=False)

        print(title)
        print(n_h)
        print(nn_nh)
        print(s_ns)


        rects = sns.boxplot(data=tmp, palette=color_mine, showfliers=False, ax=axis, orient="v", width=0.8,
                            boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, medianprops=medianprops)

        axis.yaxis.set_major_formatter(form)

        axis.set_xticks([])
        axis.set_title(title)
        axis.set_ylabel("")
        axis.set_xlabel(x_label)
        axis.axvline(1.5, ls='dashed', linewidth=0.3, color="#C0C0C0")
        axis.axvline(3.5, ls='dashed', linewidth=0.3, color="#C0C0C0")



# f.legend((rects[0], rects[1], rects[2], rects[3], rects[4], rects[5]),
#          ('Hateful User', 'Normal User', 'Hateful Neigh.', 'Normal Neigh.', 'Suspended', 'All'),
#          loc='upper center',
#          fancybox=True, shadow=True, ncol=6)
f.tight_layout(rect=[0, 0, 1, .95])

f.savefig("../imgs/spam.pdf")
