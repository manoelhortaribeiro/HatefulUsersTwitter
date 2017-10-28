from py2neo import Graph
import json
import csv
import re

f = open("./twitter_neo4jsecret.json", 'r')
config_neo4j = json.load(f)
f.close()
graph = Graph(config_neo4j["host"], password=config_neo4j["password"])

query = """MATCH (u:User)-[:tweeted]->(t:Tweet) RETURN u.id as id, u.screen_name as screen_name, t.content as content"""
df = graph.data(query)

f = open("./tweets.csv", "w")
csv_writer = csv.writer(f)

csv_writer.writerow(["user_id", "screen_name",
                     "tweet_id", "tweet_text", "tweet_creation", "tweet_fav", "tweet_rt"
                     "rp_flag", "rp_status", "rp_user",
                     "qt_flag", "qt_user_id", "qt_status_id", "qt_text", "qt_creation", "qt_fav", "qt_rt"
                     "rt_flag", "rt_user_id", "rt_status_id", "rt_text", "rt_creation", "rt_fav", "rt_rt" ])

for row in df:
    for tweet in row["content"]:
        new_tweet = []
        # print(tweet)
        len_tweet = len(tweet.split(";"))

        match = re.match("([0-9])+", tweet)
        start, end = match.span()
        new_tweet.append(tweet[start:end])
        # print(new_tweet)

        tweet = tweet[end+1:]
        match = re.match(".*?(?=;1[0-9]{9}\.0)", tweet)
        start, end = match.span()
        new_tweet.append(tweet[start:end])
        tweet = tweet[end+1:]
        # print(new_tweet)

        tmp = tweet.split(";")
        new_tweet += tmp[:9]
        tweet = ";".join(tmp[9:])
        # print(new_tweet)

        if new_tweet[-3] == 'True':
            match = re.match(".*?(?=;1[0-9]{9}\.0)", tweet)
            start, end = match.span()
            new_tweet.append(tweet[start:end])
            tweet = tweet[end+1:]
            tmp = tweet.split(";")
            new_tweet += tmp[:6]
            tweet = ";".join(tmp[6:])

        else:
            tmp = tweet.split(";")
            new_tweet += tmp[:7]
            tweet = ";".join(tmp[7:])
        # print(new_tweet)

        match = re.match(".*?(?=;1[0-9]{9}\.0)", tweet)
        if match:
            start, end = match.span()
            new_tweet.append(tweet[start:end])
            tweet = tweet[end+1:]
        tmp = tweet.split(";")
        new_tweet += tmp

        csv_writer.writerow([row["id"]] + [row["screen_name"]] + new_tweet)

f.close()
