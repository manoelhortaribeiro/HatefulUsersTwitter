import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#F8414A", "#FD878D", "#385A89", "#5676A1", ]

df = pd.read_csv("../data/users_all.csv")

f, axis = plt.subplots(1, 2, figsize=(10.8, 1.5))


men = [df[df.hate == "hateful"],
       df[df.hate_neigh],
       df[df.hate == "normal"],
       df[df.normal_neigh]]

titles = ["Hateful Users", "Hateful Neighborhood","Normal Users", "Normal Neighborhood"]

for category, title, color in zip(men, titles, color_mine):
    sns.distplot(category["sentiment"], ax=axis[0], color=color, norm_hist=True, hist=False)
    sns.distplot(category["subjectivity"], ax=axis[1], color=color, norm_hist=True, hist=False)

axis[0].set_title("Sentiment")
axis[1].set_title("Subjectivity")
axis[0].set_xlabel("")
axis[1].set_xlabel("")
axis[0].set_xlim([-0.1,0.35])
axis[1].set_xlim([0.3,0.7])
f.tight_layout()
f.savefig("../imgs/sentiment.pdf")
