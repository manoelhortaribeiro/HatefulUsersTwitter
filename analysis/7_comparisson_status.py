import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import interp1d
import scipy.stats as stats

from tmp.utils import formatter



form = FuncFormatter(formatter)

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#71BC78", "#4F7942"]

df = pd.read_csv("../data/users_anon.csv")

f, axzs = plt.subplots(2, 3, figsize=(5.4, 3), sharey=True)
boxprops = dict(linewidth=0.3)
whiskerprops = dict(linewidth=0.3)
capprops = dict(linewidth=0.3)
medianprops = dict(linewidth=1)

attributes_all = [["statuses_count", "followers_count", "followees_count"],
                  ["betweenness", "eigenvector", "out_degree"]]

titles_all = [["\#statuses", "\#followers", "\#followees"],
              ["betweenness", "eigenvector", "out degree"]]

legend = ['Banned before 12/12 ', 'Banned after 12/12']

first = True
for axs, attributes, titles in zip(axzs, attributes_all, titles_all):

    for axis, attribute, title in zip(axs, attributes, titles):
        N = 4
        men = [df[(df.is_63_2 == False) & (df.is_63 == True)], df[(df.is_63_2 == True) & (df.is_63 == False)]]

        medians, medians_ci = [], []
        averages, averages_ci = [], []
        tmp = []

        for category, color, leg in zip(men, color_mine, legend):
            x = np.array(sorted(category[attribute].values))
            tmp.append(category[attribute].values)

            y = x.cumsum()
            y = y / y[-1]
            x10 = x[::2]
            y10 = y[::2]

            rect = axis.plot(x10, y10, color=color, label=leg, lw=1)

        _, n_h = stats.ttest_ind(tmp[0], tmp[1], equal_var=False, nan_policy='omit')
        print(stats.ks_2samp(tmp[0], tmp[1]))

        print(title)
        print(n_h)

        ind = np.array([0, 1])

        axis.xaxis.set_major_formatter(form)

        axis.set_yticks([0, 0.25, 0.5, .75, 1])
        axis.set_title(title)
        axis.set_xlabel("")

        axis.set_ylabel("")
        # axis.set_xscale("log")

        # axis.legend().set_visible(False)

        # if title in ["betweenness", "eigenvector", "out degree"]:
        #     axis.set_xlabel("\% Users")

f.legend(loc='upper center', fancybox=True, shadow=True, ncol=2)
f.tight_layout(rect=[0, 0, 1, .95])

f.savefig("../imgs/activity_2.pdf")
