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

creation_time = dict()

count = 0
for user_id in ids:
    count += 1
    print(count, curr_account)

    while True:
        try:
            user = crawl.get_user(user_id)
            creation_time[user.id] = user.created_at
            break

        except tweepy.TweepError as exc:
            if exc.api_code is None:
                curr_account += 1
                curr_account %= all_accs
                print(curr_account)
                crawl = get_tweepy(accounts, curr_account)

            if exc.api_code == 50 or exc.api_code == 63:
                break

f = open("../data/created_at.csv", "w")
csv_writer = csv.writer(f)

csv_writer.writerow(["user_id", "created_at"])

for key in creation_time.keys():
    csv_writer.writerow([key, creation_time[key].timestamp()])

f.close()
