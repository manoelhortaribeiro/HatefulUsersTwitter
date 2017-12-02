import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")
color_mine = ["#F8414A", "#385A89"]

df = pd.read_csv("../data/users_all.csv")
df.fillna(0, inplace=True)

df = df[df.hate != "other"]

cols = ["{0}_glove".format(v) for v in range(1, 300)]

f, axis = plt.subplots(1, 2, figsize=(5.4, 2.5), sharex=True, sharey=True)
men = ["hateful", "normal"]

for ax, category, color in zip(axis, men, color_mine):

    X = np.array(df[cols].values).reshape(-1, len(cols))
    print(X.shape)
    scaling = MinMaxScaler().fit(X)

    X = scaling.transform(X)
    print(X)
    pca = PCA(n_components=2)
    tmp = pca.fit(X).transform(X).reshape(2, -1)
    df["pca1"] = tmp[0]
    df["pca2"] = tmp[1]
    men2 = df[df.hate == category]

    ax.scatter(men2["pca1"], men2["pca2"], c=color, alpha=0.05)

f.savefig("../imgs/glove.pdf")
