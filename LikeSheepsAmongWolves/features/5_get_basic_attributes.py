from multiprocessing import Process
import pandas as pd
import numpy as np
import time
import csv
import re

hashtags = re.compile("#(\w+)")
regex_mentions = re.compile("@(\w+)")
urls = re.compile("http(s)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
regex_bad_words = re.compile("(" + "|".join(pd.read_csv("../data/bad_words.txt")["words"].values) + ")")


def mentions(tweets):
    ments = []
    for tweet in tweets.values:
        ments += regex_mentions.findall(tweet)

    return len(ments)

def bad_words(tweets):
    baddies = []
    for tweet in tweets.values:
        baddies += regex_bad_words.findall(tweet)

    return len(baddies)


def urls_all(tweets):
    urlss = []
    for tweet in tweets.values:
        urlss += urls.findall(tweet)

    return len(urlss)


def hashtags_all(tweets):
    hts = []
    for tweet in tweets.values:
        hts += hashtags.findall(tweet)

    return len(hts), " ".join(hts)


def get_values(tweets):
    c = 0
    for tweet in tweets.values:
        if tweet != "":
            c += 1

    return c


def tweet_size(tweets):
    c = 0
    for tweet in tweets.values:
        c += len(tweet)
    return c/len(tweets)


def processing(vals, columns, iterv):
    users = pd.DataFrame(vals)
    users = users[columns]

    print("{0}-------------".format(iterv))

    # HASHTAGS

    users["any_text"] = users["tweet_text"] + users["rt_text"] + users["qt_text"]
    users_hashtags = users.groupby(["user_id"])["any_text"].apply(lambda x: hashtags_all(x))
    hashtags_cols = np.array(list(users_hashtags.values))
    df_hashtags = pd.DataFrame(hashtags_cols, columns=["number hashtags", "hashtags"], index=users_hashtags.index)
    df_hashtags.index.names = ['user_id']

    # TWEETS NUMBER

    df_tweet_number = users.groupby(["user_id"])["tweet_text"].apply(lambda x: get_values(x)).reset_index()
    df_tweet_number.set_index("user_id", inplace=True)
    df_tweet_number.columns = ["tweet number"]

    df_retweet_number = users.groupby(["user_id"])["rt_text"].apply(lambda x: get_values(x)).reset_index()
    df_retweet_number.set_index("user_id", inplace=True)
    df_retweet_number.columns = ["retweet number"]

    df_quote_number = users.groupby(["user_id"])["qt_text"].apply(lambda x: get_values(x)).reset_index()
    df_quote_number.set_index("user_id", inplace=True)
    df_quote_number.columns = ["quote number"]

    df_tweet_length = users.groupby(["user_id"])["any_text"].apply(lambda x: tweet_size(x)).reset_index()
    df_tweet_length.set_index("user_id", inplace=True)
    df_tweet_length.columns = ["status length"]

    df_urls = users.groupby(["user_id"])["any_text"].apply(lambda x: urls_all(x)).reset_index()
    df_urls.set_index("user_id", inplace=True)
    df_urls.columns = ["number urls"]

    df_baddies = users.groupby(["user_id"])["any_text"].apply(lambda x: bad_words(x)).reset_index()
    df_baddies.set_index("user_id", inplace=True)
    df_baddies.columns = ["baddies"]

    df_mentions = users.groupby(["user_id"])["any_text"].apply(lambda x: mentions(x)).reset_index()
    df_mentions.set_index("user_id", inplace=True)
    df_mentions.columns = ["mentions"]

    df = pd.DataFrame(pd.concat([df_hashtags, df_tweet_number, df_retweet_number, df_quote_number,
                                 df_tweet_length, df_urls, df_baddies, df_mentions], axis=1))
    df.to_csv("../data/tmp2/users_content_{0}.csv".format(iterv))
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
