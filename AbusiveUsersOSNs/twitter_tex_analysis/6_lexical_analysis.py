import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font', family='serif')
plt.rc('text', usetex=True)
sns.set(style="whitegrid", font="serif")

df = pd.read_csv("../data/users_all.csv")
color_mine = ["#F8414A", "#FD878D", "#385A89", "#5676A1", ]

empaths = [["sadness_empath", "fear_empath", "swearing_terms_empath",
            "independence_empath", "positive_emotion_empath", "nervousness_empath"],
           ["deception_empath", "government_empath", "help_empath",
            "alcohol_empath", "ridicule_empath", "warmth_empath"]]

men = [df[df.hate == "hateful"],
       df[df.hate == "normal"]]

f, axises = plt.subplots(6, 2, figsize=(10.8, 6), sharex=True)
axises = zip(*axises)
for empath, axis in zip(empaths, axises):
    count = 0

    for col in empath:
        sns.distplot(men[0][col].values, color=color_mine[0], ax=axis[count],
                     norm_hist=True, kde_kws={"shade": True})
        sns.distplot(men[1][col].values, color=color_mine[2], ax=axis[count],
                     norm_hist=True, kde_kws={"shade": True})

        max_v = max(max(men[0][col].values,), max(men[1][col].values))
        axis[count].set_xlim([-0.001, 0.02])
        count += 1

f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
f.savefig("../imgs/lexical.pdf")
