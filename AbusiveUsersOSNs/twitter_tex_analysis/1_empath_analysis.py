from empath import Empath
import pandas as pd
import textblob


def sentiment_apply(x):
    len_x = len(x.values)
    sentiment_sum = 0
    subjectivity_sum = 0
    for value in x.values:
        st, sj = textblob.TextBlob(value).sentiment
        sentiment_sum += st
        subjectivity_sum += sj
    return {"sentiment": sentiment_sum / len_x, "subjectivity": subjectivity_sum / len_x}


users = pd.read_csv("../data/tweets_head.csv", keep_default_na=False)

users["any_text"] = users["tweet_text"] + users["rt_text"] + users["qt_text"]

# SENTIMENT ANALYSIS

sentiment = users.groupby(["user_id"])["any_text"].apply(lambda x: sentiment_apply(x)).reset_index()

sentiment_vals = sentiment[sentiment["level_1"] == "sentiment"]

subjectivity_vals = sentiment[sentiment["level_1"] == "subjectivity"]

df_sentiment = pd.DataFrame({'sentiment': sentiment_vals.any_text.values,
                             "subjectivity": subjectivity_vals.any_text.values},
                            index=sentiment_vals.user_id.values)

# LEXICAL ANALYSIS

full_text = users.groupby(["user_id"])["any_text"].apply(lambda x: x.sum())

text_df = pd.DataFrame({'text': full_text.values}, index=full_text.index)

lexicon = Empath()

empath = text_df.text.apply(lambda x: lexicon.analyze(x, normalize=True))

df_empath = pd.DataFrame.from_records(index=empath.index, data=empath.values)

df = pd.DataFrame(pd.concat([df_sentiment, df_empath], axis=1))

df.to_csv("../data/users_content.csv")


