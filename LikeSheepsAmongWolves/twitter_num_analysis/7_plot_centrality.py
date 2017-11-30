import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#F8414A", "#385A89"]
df = pd.read_csv("../data/users_all.csv")

f, axis = plt.subplots(1, 4, figsize=(10.8, 2.5))

men = [df[df.hate == "hateful"],
       df[df.hate == "normal"]]

titles = ["Hateful Users", "Normal Users"]

for category, title, color in zip(men, titles, color_mine):
    btw = np.array(sorted(category["betweenness"].values, reverse=False))
    # btw = btw.cumsum()
    eig = np.array(sorted(category["eigenvector"].values, reverse=False)) * (10 ** 10)
    # eig = eig.cumsum()
    ind = np.array(sorted(category["in_degree"].values, reverse=False))
    print(np.max(eig))
    # ind = ind.cumsum()
    # print(title, np.median(ind))

    oud = np.array(sorted(category["out_degree"].values, reverse=False))

    # print(max(oud), min(oud))
    # print(title, np.median(oud))

    # oud = oud.cumsum()
    print(title, np.median(oud))
    x = np.linspace(0, 100, len(btw))

    # axis[0].plot(btw, x, color=color, alpha=0.5)
    # axis[1].plot(eig, x, color=color,alpha=0.5)
    sns.distplot(btw, kde=True, color=color, ax=axis[0],  hist=False,
                 kde_kws={"bw": "silverman", "alpha": 0.5}, norm_hist=True)
    sns.distplot(eig, kde=True, color=color, ax=axis[1],  hist=False,
                 kde_kws={"bw": "silverman", "alpha": 0.5}, norm_hist=True)
    sns.distplot(ind, kde=True, color=color, ax=axis[2],  hist=False,
                 kde_kws={"bw": "silverman", "alpha": 0.5}, norm_hist=True)
    sns.distplot(oud, kde=True, color=color, ax=axis[3], hist=False,
                 kde_kws={"bw": 0.0003, "alpha": 0.5}, norm_hist=True)
    # axis[2].plot(x, ind, color=color)
    # axis[3].plot(x, oud, color=color)


    # axis[2].loglog(x, ind, color=color)
    # axis[3].loglog(x, oud, color=color)

axis[0].set_title("Betweenness (CDF)")
axis[1].set_title("Eigenvector (CDF)")
axis[2].set_title("In Degree (CDF)")
axis[3].set_title("Out Degree (CDF)")

axis[0].set_xlabel("")
axis[1].set_xlabel("")
# axis[0].set_xscale("symlog")
# axis[1].set_xscale("symlog")
axis[1].set_yscale("symlog")
axis[0].set_yscale("symlog")
# axis[1].set_ylim([0, 2405009])
#
# axis[2].set_xscale("symlog")
# axis[3].set_xscale("symlog")
axis[3].set_yscale("symlog")
axis[2].set_yscale("symlog")
# axis[2].set_ylim([99616, 835184539])
# axis[3].set_ylim([199232, 30183792])
f.tight_layout()
f.savefig("../imgs/centrality.pdf")
