import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')

sns.set(style="white", font="serif")

df = pd.read_csv("./users_attributes.csv")
df = df[df.hate != "other"]

f, axs = plt.subplots(1, 5, figsize=(9, 1.5))

attributes = ["statuses_count", "followers_count", "followees_count", "favorites_count", "listed_count"]
titles = ["#statuses", "#followers", "#followees", "#favorites", "#listed"]

for axis, attribute, title in zip(axs, attributes, titles):

    sns.barplot(x="hate", y=attribute, data=df, ax=axis, estimator=np.average, ci=95)
    axis.set_xticks([0 ,1], ["normal", "hateful"])
    axis.set_title(title)
    axis.set_ylabel("")
    axis.set_xlabel("")


f.tight_layout()

f.savefig("attributes.pdf")