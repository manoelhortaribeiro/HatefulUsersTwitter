import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from multiprocessing import Process
from empath import Empath
import pandas as pd
import numpy as np
import textblob
import spacy
import time
import csv
import re


stopWords = set(stopwords.words('english'))
nlp = spacy.load('en_core_web_lg')
prog = re.compile("(@[A-Za-z0-9]+)|([^0-9A-Za-z' \t])|(\w+:\/\/\S+)")
prog2 = re.compile(" +")
lexicon = Empath()
empath_cols = ["{0}_empath".format(v) for v in lexicon.cats.keys()]
glove_cols = ["{0}_glove".format(v) for v in range(300)]


def lemmatization(x, nlp):
    tweets = " ".join(list(x.values))
    letters_only = prog.sub(" ", tweets)
    lemmatized = []
    for token1 in nlp(letters_only):
        if token1.lemma_ != "-PRON-" and token1 not in stopWords:
            lemmatized.append(token1.lemma_)
        else:
            lemmatized.append(token1.text)
    final = prog2.sub(" ", " ".join(lemmatized))
    return final


def empath_analysis(x):
    val = lexicon.analyze(x, normalize=True)
    if val is None:
        return lexicon.analyze(x)
    else:
        return val


def processing(vals, columns, iterv):
    users = pd.DataFrame(vals)
    users = users[columns]

    print("{0}-------------".format(iterv))

    # PRE-PROCESSING

    users["any_text"] = users["tweet_text"] + users["rt_text"] + users["qt_text"]
    users_text = users.groupby(["user_id"])["any_text"].apply(lambda x: lemmatization(x, nlp)).reset_index()
    print("{0}-------------PRE-PROCESSING".format(iterv))

    # GLOVE ANALYSIS

    glove_arr = np.array(list(users_text["any_text"].apply(lambda x: list(nlp(x).vector)).values))
    df_glove = pd.DataFrame(glove_arr, columns=glove_cols, index=users_text.user_id.values)
    print("{0}-------------GLOVE".format(iterv))

    # SENTIMENT ANALYSIS

    sentiment_arr = np.array(list(users_text["any_text"].apply(lambda x: textblob.TextBlob(str(x)).sentiment).values))
    sentiment_cols = ["sentiment", "subjectivity"]
    df_sentiment = pd.DataFrame(sentiment_arr, columns=sentiment_cols, index=users_text.user_id.values)
    print("{0}-------------SENTIMENT".format(iterv))

    # EMPATH ANALYSIS

    lexicon_arr = np.array(list(users_text["any_text"].apply(lambda x: empath_analysis(x)).values))
    df_empath = pd.DataFrame.from_records(index=users_text.user_id.values, data=lexicon_arr)
    df_empath.columns = empath_cols
    print("{0}-------------EMPATH".format(iterv))

    # MERGE TO SINGLE

    df = pd.DataFrame(pd.concat([df_empath, df_sentiment, df_glove], axis=1))
    df.set_index("user_id", inplace=True)
    df.to_csv("../data/tmp/users_content_{0}.csv".format(iterv))
    print("-------------{0}".format(iterv))


f = open("../data/tweets.csv", "r")

cols = ["user_id", "screen_name", "tweet_id", "tweet_text", "tweet_creation", "tweet_fav", "tweet_rt", "rp_flag",
        "rp_status", "rp_user", "qt_flag", "qt_user_id", "qt_status_id", "qt_text", "qt_creation", "qt_fav",
        "qt_rt", "rt_flag", "rt_user_id", "rt_status_id", "rt_text", "rt_creation", "rt_fav", "rt_rt"]

csv_dict_reader = csv.DictReader(f)

acc_vals = []

iter_vals, count, count_max, last_u, v = 1, 0, 50000, None, []
for line in csv_dict_reader:
    if last_u is not None and last_u != line["user_id"]:
        # s = time.time()
        # processing(v, cols, iter_vals)
        # print(time.time() - s)
        acc_vals.append((v, cols, iter_vals))

        count, last_u, v = 0, None, []
        iter_vals += 1

    if len(acc_vals) == 2:
        s = time.time()
        processes = []
        for i in acc_vals:
            p = Process(target=processing, args=(i[0], i[1], i[2]))
            processes.append(p)
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        print(time.time() - s)
        acc_vals = []

    v.append(line)
    count += 1
    if count >= count_max:
        last_u = line["user_id"]

# s = time.time()
# processing(v, cols, iter_vals)
# print(time.time() - s)

s = time.time()
processes = []
for i in acc_vals:
    p = Process(target=processing, args=(i[0], i[1], i[2]))
    processes.append(p)
for p in processes:
    p.start()
for p in processes:
    p.join()
print(time.time() - s)
acc_vals = []
