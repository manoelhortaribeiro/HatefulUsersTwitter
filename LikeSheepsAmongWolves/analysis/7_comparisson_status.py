from LikeSheepsAmongWolves.tmp.utils import formatter
from matplotlib.ticker import FuncFormatter
from seaborn.algorithms import bootstrap
from scipy.interpolate import interp1d
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
color_mine = ["#71BC78", "#4F7942"]

df = pd.read_csv("../data/users_anon.csv")

f, axzs = plt.subplots(2, 3, figsize=(5.4, 3), sharex=True)
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
        men = [df[df.is_63], df[(df.is_63_2 == True) & (df.is_63 == False)]]

        medians, medians_ci = [], []
        averages, averages_ci = [], []
        rects = []

        for category, color, leg in zip(men, color_mine, legend):
            x = np.linspace(1, 100, num=len(category[attribute].values), endpoint=True)
            y = sorted(category[attribute].values, reverse=True)
            y = np.array(y).cumsum()
            fx = interp1d(x, y)
            x2 = np.linspace(1, 100, num=100, endpoint=True)
            y2 = np.array(fx(x2))

            rect = axis.plot(x2, y2, color=color, label=leg)
        ind = np.array([0, 1])

        axis.yaxis.set_major_formatter(form)

        axis.set_xticks([1, 50, 100])
        axis.set_title(title)
        axis.set_xlabel("")

        axis.set_ylabel("")

        if title in ["betweenness", "eigenvector", "out degree"]:
            axis.set_xlabel("\% Users")

f.legend(loc='upper center', fancybox=True, shadow=True, ncol=2)
f.tight_layout(rect=[0, 0, 1, .95])

f.savefig("../imgs/activity_2.pdf")
