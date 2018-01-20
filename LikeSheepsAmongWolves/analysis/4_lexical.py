from LikeSheepsAmongWolves.tmp.utils import formatter
from matplotlib.ticker import FuncFormatter
from seaborn.algorithms import bootstrap
import matplotlib.pyplot as plt
from seaborn.utils import ci
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np

form = FuncFormatter(formatter)

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#F8414A", "#5676A1", "#FD878D", "#385A89",  "#FFFACD", "#EFCC00"]

df = pd.read_csv("../data/users_anon.csv")

df["tweet_number"] = df["tweet number"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["retweet_number"] = df["retweet number"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["number_urls"] = df["number urls"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["mentions"] = df["mentions"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["mentions"] = df["mentions"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["number hashtags"] = df["number hashtags"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["baddies"] = df["baddies"] / (df["tweet number"] + df["retweet number"] + df["quote number"])

f, axzs = plt.subplots(3, 7, figsize=(10.8, 4))

attributes_all = [["sadness_empath", "fear_empath", "swearing_terms_empath", "independence_empath",
                   "positive_emotion_empath", "negative_emotion_empath", "government_empath", "love_empath"],
                   ["warmth_empath", "ridicule_empath", "masculine_empath", "feminine_empath",
                   "violence_empath", "suffering_empath", "dispute_empath", "anger_empath"],
                   ["envy_empath", "work_empath", "leader_empath", "politics_empath",
                   "terrorism_empath", "shame_empath", "confusion_empath", "hate_empath"]]

titles_all = [["Sadness", "Fear", "Swearing", "Independence", "Pos. Emotions", "Neg. Emotions", "Government", "Love"],
              ["Warmth", "Ridicule", "Masculine", "Feminine", "Violence", "Suffering", "Dispute", "Anger"],
              ["Envy", "Work", "Leader", "Politics", "Terrorism", "Shame", "Confusion", "Hate"]]

for axs, attributes, titles in zip(axzs, attributes_all, titles_all):

    for axis, attribute, title in zip(axs, attributes, titles):
        N = 4
        men = [df[df.hate == "hateful"],
               df[df.hate == "normal"],
               df[df.hate_neigh],
               df[df.normal_neigh],
               df[df.is_63_2 == True],
               df[df.is_63_2 == False]]
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

        ind = np.array([0, 1, 2, 3, 4, 5])
        _, n_h = stats.ttest_ind(tmp[0], tmp[1], equal_var=False, nan_policy='omit')
        _, nn_nh = stats.ttest_ind(tmp[2], tmp[3], equal_var=False, nan_policy='omit')
        _, s_ns = stats.ttest_ind(tmp[4], tmp[5], equal_var=False, nan_policy='omit')

        print(title)
        print(n_h)
        print(nn_nh)
        print(s_ns)

        rects = axis.bar(ind, averages, 0.6, yerr=averages_ci, color=color_mine,
                         ecolor="#212823", edgecolor=["#4D1A17"] * 6, linewidth=.3)

        axis.yaxis.set_major_formatter(form)

        axis.set_xticks([])
        axis.set_title(title)
        axis.set_ylabel("")
        axis.set_xlabel("")
        axis.axvline(1.5, ls='dashed', linewidth=0.3, color="#C0C0C0")
        axis.axvline(3.5, ls='dashed', linewidth=0.3, color="#C0C0C0")

f.legend((rects[0], rects[1], rects[2], rects[3], rects[4], rects[5]),
         ('Hateful User', 'Normal User', 'Hateful Neigh.', 'Normal Neigh.', 'Suspended', 'Non-Suspended'),
         loc='upper center',
         fancybox=True, shadow=True, ncol=6)
f.tight_layout(rect=[0, 0, 1, .95])
f.savefig("../imgs/lexical.pdf")
