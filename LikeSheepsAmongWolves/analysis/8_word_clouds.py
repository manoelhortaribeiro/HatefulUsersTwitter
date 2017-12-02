import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import math

val = re.compile(r'[^A-Za-z0-9 #]+')


def non_alpha(string):
    if string is None or (type(string) == float and math.isnan(string)):
        return ""
    return val.sub("", string)


df = pd.read_csv("../data/users_all.csv")


hts = [np.array(list(map(non_alpha, df[df.hate == "hateful"]["hashtags"].values))),
       np.array(list(map(non_alpha, df[df.hate == "normal"]["hashtags"].values)))]
dests = ['../imgs/wc_hate.pdf', '../imgs/wc_normal.pdf']


for i, j in zip(hts, dests):
    ts = " ".join(i)
    wordcloud = WordCloud(max_font_size=120 ,width=1600, height=800, colormap="Dark2",
                          background_color="white", collocations=False).generate(ts)

    plt.figure( figsize=(20,10), facecolor='w')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.savefig(j, facecolor='w', bbox_inches='tight')
