import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#F8414A", "#FD878D", "#385A89", "#5676A1" ]
color_mine_rgba = [(248, 65, 74, 1), (253, 135, 141, 1), (56, 90, 137, 1), (86, 118, 161, 1)]
titles = ["\%tweets", "\%retweets", "\%quotes", "urls/tweet", "mentions/tweet", "hashtags/tweet", "profanity/tweet", "length"]
values = ["tweet_number", "retweet_number", "quote_number", "number_urls", "mentions", "number hashtags", "baddies", "status length"]
f, axis = plt.subplots(1, 8, figsize=(10.8, 2.5))

df = pd.read_csv("../data/users_all.csv")

df["tweet_number"] = df["tweet number"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["retweet_number"] = df["retweet number"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["quote_number"] = df["quote number"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["number_urls"] = df["number urls"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["mentions"] = df["mentions"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["mentions"] = df["mentions"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["number hashtags"] = df["number hashtags"] / (df["tweet number"] + df["retweet number"] + df["quote number"])
df["baddies"] = df["baddies"] / (df["tweet number"] + df["retweet number"] + df["quote number"])

men = [df[df.hate == "hateful"],
       df[df.hate_neigh],
       df[df.hate == "normal"],
       df[df.normal_neigh]]


for value, title, ax in zip(values, titles, axis):
    tmp = []
    for category in men:
        tmp.append(category[value].values)
    sns.boxplot(data=tmp, palette=color_mine, showfliers=False, ax=ax)
    ax.set_title(title)
    ax.set_xticklabels(["", "", "", ""])
    # sns.distplot(category["sentiment"], ax=axis[0], color=color, norm_hist=True, hist=False)
    # sns.distplot(category["subjectivity"], ax=axis[1], color=color, norm_hist=True, hist=False)

patches = []
for i, j in zip(color_mine, ["Hateful Users", "Hateful Neighborhood", "Normal Users", "Normal Neighborhood"]):
    patches.append(mpatches.Patch(color=i, label=j))

f.legend(handles=patches, labels=[''], loc='upper center', fancybox=True, shadow=True, ncol=4)
f.tight_layout()
f.savefig("../imgs/tweets_stats.pdf")
