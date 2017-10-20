import json
import tweepy
from py2neo import Graph
from AbusiveUsersOSNs.neo4j_ogm_schema import User, tweepy2neo4j_user, Tweet, tweepy2neo4j_tweet


class MHCrawler:

    def __init__(self, auth_tweepy, auth_neo4j):
        self.curr_acc = 0
        self.accounts = list(auth_tweepy.items())
        self.api = MHCrawler.auth_tweepy(self.accounts[self.curr_acc][1])
        self.graph = MHCrawler.auth_neo4j(auth_neo4j)

    @staticmethod
    def auth_tweepy(auth):
        oauth = tweepy.OAuthHandler(auth["consumer_key"], auth["consumer_secret"])
        oauth.set_access_token(auth["access_token"], auth["access_secret"])
        return tweepy.API(oauth)

    @staticmethod
    def auth_neo4j(auth):
        return Graph(auth["host"], password=auth["password"])

    def run(self, seed):
        user = self.api.get_user(seed)
        user_p = tweepy2neo4j_user(user)
        tl = self.api.user_timeline(include_rts=True,
                                    count=200,
                                    trim_user=False,
                                    exclude_replies=False,
                                    user_id=seed,
                                    tweet_mode='extended')
        tweets = []
        for tweet in tl:
            tweets.append(tweepy2neo4j_tweet(tweet))
        for tweet in tweets:
            print(tweet)
            user_p.tweeted_by_me.add(tweet)
            self.graph.create(tweet)
        self.graph.create(user_p)


if __name__ == "__main__":
    f = open("secrets.json", 'r')
    config_tweepy = json.load(f)
    f.close()

    f = open("neo4jsecret.json", 'r')
    config_neo4j = json.load(f)
    f.close()

    crawl = MHCrawler(config_tweepy, config_neo4j)
    crawl.run(850859913244852224)


