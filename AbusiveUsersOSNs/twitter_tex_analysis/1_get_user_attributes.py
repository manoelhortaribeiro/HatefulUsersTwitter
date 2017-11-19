from empath import Empath
import pandas as pd
import numpy as np
import textblob
import spacy
import csv

nlp = spacy.load('en')


def processing(vals, columns, iterv):
    users = pd.DataFrame(vals)
    users = users[columns]

    print("{0}-------------".format(iterv))

    users["any_text"] = users["tweet_text"] + users["rt_text"] + users["qt_text"]

    # GLOVE ANALYSIS

    glove = users.groupby(["user_id"])["any_text"].apply(lambda x: glove_apply(x)).reset_index()

    glove_arr = np.array(list(glove.any_text.values))

    glove_cols = ["{0}_glove".format(v) for v in range(glove_arr.shape[1])]

    df_glove = pd.DataFrame(glove_arr, columns=glove_cols, index=glove.user_id.values)

    print("glove done")

    # SENTIMENT ANALYSIS

    sentiment = users.groupby(["user_id"])["any_text"].apply(lambda x: sentiment_apply(x)).reset_index()

    sentiment_vals = sentiment[sentiment["level_1"] == "sentiment"]

    subjectivity_vals = sentiment[sentiment["level_1"] == "subjectivity"]

    df_sentiment = pd.DataFrame({'sentiment': sentiment_vals.any_text.values,
                                 "subjectivity": subjectivity_vals.any_text.values},
                                index=sentiment_vals.user_id.values)

    print("sentiment done")

    # LEXICAL ANALYSIS

    full_text = users.groupby(["user_id"])["any_text"].apply(lambda x: x.sum())

    text_df = pd.DataFrame({'text': full_text.values}, index=full_text.index)

    lexicon = Empath()

    empath = text_df.text.apply(lambda x: lexicon.analyze(x, normalize=True))

    empath_cols = ["{0}_empath".format(v) for v in empath.head(1).values[0].keys()]

    df_empath = pd.DataFrame.from_records(index=empath.index, data=empath.values)

    df_empath.columns = empath_cols

    print("lexical analysis")

    df = pd.DataFrame(pd.concat([df_sentiment, df_empath, df_glove], axis=1))

    df.to_csv("../data/tmp/users_content_{0}.csv".format(iterv))


def sentiment_apply(x):
    len_x = len(x.values)
    sentiment_sum = 0
    subjectivity_sum = 0
    for value in x.values:
        st, sj = textblob.TextBlob(value).sentiment
        sentiment_sum += st
        subjectivity_sum += sj
    return {"sentiment": sentiment_sum / len_x, "subjectivity": subjectivity_sum / len_x}


def glove_apply(x):
    text = x.values.sum()
    return nlp(text).vector


f = open("../data/tweets.csv", "r")
cols = ["user_id", "screen_name", "tweet_id", "tweet_text", "tweet_creation", "tweet_fav", "tweet_rt", "rp_flag",
        "rp_status", "rp_user", "qt_flag", "qt_user_id", "qt_status_id", "qt_text", "qt_creation", "qt_fav",
        "qt_rt", "rt_flag", "rt_user_id", "rt_status_id", "rt_text", "rt_creation", "rt_fav", "rt_rt"]

csv_dict_reader = csv.DictReader(f)

iter_vals, count, count_max, last_u, v = 1, 0, 100000, None, []
for line in csv_dict_reader:
    if last_u is not None and last_u != line["user_id"]:
        count, last_u, v = 0, None, []
        iter_vals += 1

    v.append(line)
    count += 1
    if count >= count_max:
        last_u = line["user_id"]
processing(v, cols, iter_vals)
