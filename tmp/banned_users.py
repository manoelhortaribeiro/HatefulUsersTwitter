import pandas as pd
import tweepy
import json
import csv


def get_tweepy(accs, val):
    auth = accs[val][1]
    oauth = tweepy.OAuthHandler(auth["consumer_key"], auth["consumer_secret"])
    oauth.set_access_token(auth["access_token"], auth["access_secret"])
    return tweepy.API(oauth)


f = open("../secrets/twitter_secrets.json", 'r')
tweepy_auth = json.load(f)
accounts = list(tweepy_auth.items())
f.close()

curr_account = 0
all_accs = 5
df = pd.read_csv("../data/users_attributes.csv")
ids = df.user_id.values
crawl = get_tweepy(accounts, curr_account)

deleted_accounts_50 = dict()
deleted_accounts_63 = dict()

count = 0
for user_id in ids:
    count += 1
    print(count, curr_account)

    while True:
        try:
            deleted_accounts_50[user_id] = False
            deleted_accounts_63[user_id] = False
            user = crawl.get_user(user_id)
            break

        except tweepy.TweepError as exc:
            if exc.api_code is None:
                curr_account += 1
                curr_account %= all_accs
                print(user_id)
                crawl = get_tweepy(accounts, curr_account)

            if exc.api_code == 50:
                deleted_accounts_50[user_id] = True
                break

            if exc.api_code == 63:
                deleted_accounts_63[user_id] = True
                break

# f = open("../data/deleted_account_before_guideline.csv", "w")
f = open("../data/deleted_account_after_guideline.csv", "w")

csv_writer = csv.writer(f)

csv_writer.writerow(["user_id", "is_50", "is_63"])

for key in deleted_accounts_50.keys():
    csv_writer.writerow([key, deleted_accounts_50[key], deleted_accounts_63[key]])

f.close()
