import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.factorplot

plt.rc('font', family='serif')
# plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#F8414A", "#385A89"]
df = pd.read_csv("../data/users_all.csv")
df = df[df.hate != "other"]
f, axis = plt.subplots(1, 4, figsize=(10.8, 3))

sns.boxplot(x="hate", y="betweenness", data=df, ax=axis[0], showfliers=False)
sns.boxplot(x="hate", y="eigenvector", data=df, ax=axis[1], showfliers=False)
sns.boxplot(x="hate", y="in_degree", data=df, ax=axis[2], showfliers=False)
sns.boxplot(x="hate", y="out_degree", data=df, ax=axis[3], showfliers=False)

# men = [df[df.hate == "hateful"],
#        df[df.hate == "normal"]]
#
# titles = ["Hateful Users", "Normal Users"]
#
# for category, title, color in zip(men, titles, color_mine):
#     btw = np.array(sorted(category["betweenness"].values, reverse=False))
#
#     eig = np.array(sorted(category["eigenvector"].values, reverse=False))
#
#     ind = np.array(sorted(category["in_degree"].values, reverse=False))
#
#     oud = np.array(sorted(category["out_degree"].values, reverse=False))
#
#     # print(max(oud), min(oud))
#     # print(title, np.median(oud))
#
#     # oud = oud.cumsum()
#
#     axis[0].plot(x, btw, color=color, alpha=0.5)
#     axis[1].plot(x, eig, color=color, alpha=0.5)
#     axis[2].plot(x, ind, color=color, alpha=0.5)
#     axis[3].plot(x, oud, color=color, alpha=0.5)



    # axis[2].loglog(x, ind, color=color)
    # axis[3].loglog(x, oud, color=color)

axis[0].set_title("Betweenness")
axis[1].set_title("Eigenvector")
axis[2].set_title("In Degree")
axis[3].set_title("Out Degree")

axis[0].set_xlabel("")
axis[1].set_xlabel("")
# axis[0].set_xscale("symlog")
# axis[1].set_xscale("symlog")
axis[0].set_yscale("symlog")
# axis[1].set_yscale("symlog")
# axis[2].set_yscale("symlog")
# axis[3].set_yscale("symlog")

# axis[0].set_ylim([-1000, 60000])
#
# axis[2].set_xscale("symlog")
# axis[3].set_xscale("symlog")

# axis[3].set_ylim([199232, 30183792])
f.tight_layout()
f.savefig("../imgs/centrality.pdf")
