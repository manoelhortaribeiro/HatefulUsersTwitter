from py2neo import Graph
import json
import csv
import re

f = open("../secrets/twitter_neo4jsecret.json", 'r')

config_neo4j = json.load(f)
f.close()
graph = Graph(config_neo4j["host"], password=config_neo4j["password"])

print(graph)

f = open("../data/preprocessing/tweets.csv", "w")
csv_writer = csv.writer(f)

csv_writer.writerow(["user_id", "screen_name",
                     "tweet_id", "tweet_text", "tweet_creation", "tweet_fav", "tweet_rt",
                     "rp_flag", "rp_status", "rp_user",
                     "qt_flag", "qt_status_id", "qt_user_id", "qt_text", "qt_creation", "qt_fav", "qt_rt",
                     "rt_flag", "rt_status_id", "rt_user_id", "rt_text", "rt_creation", "rt_fav", "rt_rt"])

q = """MATCH (u:User) where u.virtual="F" return count(u) as number"""
df = graph.data(q)
max_entries = df[0]["number"]
aux = (list(range(max_entries, 0, -10000)) + [0])[::-1]
ranges = zip(aux[:-1], aux[1:])

for lower, upper in ranges:
    print(lower, upper)

    query = """ MATCH (u:User)-[:tweeted]->(t:Tweet)
                WHERE u.number > {0} AND u.number < {1} 
                RETURN u.id as id, u.screen_name as screen_name, t.content as content """.format(lower, upper)

    df = graph.data(query)

    for row in df:
        for tweet in row["content"]:
            new_tweet = []
            len_tweet = len(tweet.split(";"))

            match = re.match("([0-9])+", tweet)
            start, end = match.span()
            new_tweet.append(tweet[start:end])

            tweet = tweet[end + 1:]
            match = re.match(".*?(?=;1[0-9]{9}\.0)", tweet)
            start, end = match.span()
            new_tweet.append(tweet[start:end])
            tweet = tweet[end + 1:]

            tmp = tweet.split(";")
            new_tweet += tmp[:9]
            tweet = ";".join(tmp[9:])

            if new_tweet[-3] == 'True':
                match = re.match(".*?(?=;1[0-9]{9}\.0)", tweet)
                start, end = match.span()
                new_tweet.append(tweet[start:end])
                tweet = tweet[end + 1:]
                tmp = tweet.split(";")
                new_tweet += tmp[:6]
                tweet = ";".join(tmp[6:])

            else:
                tmp = tweet.split(";")
                new_tweet += tmp[:7]
                tweet = ";".join(tmp[7:])

            match = re.match(".*?(?=;1[0-9]{9}\.0)", tweet)
            if match:
                start, end = match.span()
                new_tweet.append(tweet[start:end])
                tweet = tweet[end + 1:]
            tmp = tweet.split(";")
            new_tweet += tmp

            csv_writer.writerow([row["id"]] + [row["screen_name"]] + new_tweet)

f.close()
