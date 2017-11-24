import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud


df = pd.read_csv("../data/users_all.csv")


hts = [np.array(list(map(str, df[df.hate == "hateful"]["hashtags"].values))),
       np.array(list(map(str, df[df.hate == "normal"]["hashtags"].values)))]
dests = ['../imgs/wc_hate.pdf', '../imgs/wc_normal.pdf']


for i, j in zip(hts, dests):
    ts = " ".join(i)
    print(type(ts))
    wordcloud = WordCloud(max_font_size=120 ,width=1600, height=800,
                          background_color="white", collocations=False).generate(ts)

    plt.figure( figsize=(20,10), facecolor='w')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(j, facecolor='w', bbox_inches='tight')
