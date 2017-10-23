from AbusiveUsersOSNs.neo4j_ogm_schema import tweepy2neo4j_user, tweepy2neo4j_virtual_user, tweepy2neo4j_media
from AbusiveUsersOSNs.neo4j_ogm_schema import tweepy2neo4j_tweet, tweepy2neo4j_materialize_user
from py2neo import Graph, Relationship, Node, NodeSelector
from urllib.parse import urlparse
from datetime import datetime
import random
import tweepy
import json
import os


class MHCrawler:
    def __init__(self, auth_tweepy, auth_neo4j, n, seed, next_node, w=50):
        self.curr_acc = 0
        self.accounts = list(auth_tweepy.items())
        self.api = MHCrawler.auth_tweepy(self.accounts[self.curr_acc][1])
        self.graph = MHCrawler.auth_neo4j(auth_neo4j)
        self.node_selector = NodeSelector(self.graph)
        self.seed, self.n, self.w = seed, n, w
        self.next_node = Node("User", **next_node) if next_node is not None else next_node

        if self.graph.schema.get_uniqueness_constraints("User") != ["id"]:
            self.graph.schema.create_uniqueness_constraint("User", "id")

        if self.graph.schema.get_uniqueness_constraints("Tweet") != ["id"]:
            self.graph.schema.create_uniqueness_constraint("Tweet", "id")

        if self.graph.schema.get_uniqueness_constraints("Media") != ["url"]:
            self.graph.schema.create_uniqueness_constraint("Media", "url")

        if "n" not in self.graph.schema.get_indexes("User"):
            self.graph.schema.create_index("User", "n")

    @staticmethod
    def auth_tweepy(auth):
        oauth = tweepy.OAuthHandler(auth["consumer_key"], auth["consumer_secret"])
        oauth.set_access_token(auth["access_token"], auth["access_secret"])
        return tweepy.API(oauth)

    @staticmethod
    def auth_neo4j(auth):
        return Graph(auth["host"], password=auth["password"])

    @staticmethod
    def get_quoted_users(status):
        if hasattr(status, 'quoted_status'):
            return [status._json["quoted_status"]["user"]["id"]]
        return []

    @staticmethod
    def get_retweeted_users(status):
        if hasattr(status, "retweeted_status"):
            return [status._json["retweeted_status"]["user"]["id"]]
        return []

    def get_tweets(self, user_p):

        tl = self.api.user_timeline(include_rts=True, count=200, trim_user=False, exclude_replies=False,
                                    user_id=user_p["id"], tweet_mode='extended')

        tweets, urls, quoted, retweeted = [], [], [], []

        for tweet in tl:
            tweets.append("{0},{1},{2}".format(tweet.id, tweet.full_text, tweet.created_at.timestamp()))

            if 'urls' in tweet.entities:
                for url in tweet.entities['urls']:
                    if 'expanded_url' in url and url['expanded_url'] is not None:
                        if urlparse(url['expanded_url']).netloc != "twitter.com":
                            urls.append((url['expanded_url'], tweet.id))

            quoted += MHCrawler.get_quoted_users(tweet)
            retweeted += MHCrawler.get_retweeted_users(tweet)

        tweet_p = tweepy2neo4j_tweet(tweets)

        self.graph.create(tweet_p)
        self.graph.create(Relationship(user_p, "tweeted", tweet_p))

        for url in urls:
            url_p = tweepy2neo4j_media(url[0], url[1])
            self.graph.merge(url_p, "Media", "url")
            self.graph.create(Relationship(user_p, "shared", url_p))

        for node in set(retweeted).union(quoted):
            next_node = list(self.node_selector.select("User", id=node))

            if len(next_node) == 0:
                virtual_user = tweepy2neo4j_virtual_user(node)
            elif len(next_node) == 1:
                virtual_user = next_node[0]
            else:
                raise Exception("Multiple nodes with the same id")

            if node in retweeted:
                rel = Relationship(user_p, "retweeted", virtual_user)
                self.graph.merge(rel)
            if node in quoted:
                rel = Relationship(user_p, "quoted", virtual_user)
                self.graph.merge(rel)

    def push_node(self, node):
        user = self.api.get_user(node["id"])
        user_p = tweepy2neo4j_materialize_user(node, user, self.n)
        self.n += 1
        self.graph.push(user_p)
        self.get_tweets(user_p)
        return user_p

    def init_run(self):
        user = self.api.get_user(self.seed)
        user_p = tweepy2neo4j_user(user, self.n)
        self.n += 1
        self.graph.merge(user_p, "User", "id")
        self.get_tweets(user_p)
        self.next_node = user_p
        print("Started on {0}".format(user_p["screen_name"]))

    def walk(self, adj):
        self.next_node = random.choice(adj)
        print("Walk {0}".format(self.next_node["id"]))

    def jump(self):
        self.next_node = list(self.node_selector.select("User", n=random.randint(1, self.n - 1)))
        self.next_node = self.next_node[0]
        print("Jump {0}".format(self.next_node["screen_name"]))

    def get_adj(self):
        adj = []
        for rel in self.graph.match(start_node=self.next_node, rel_type="quoted", bidirectional=True):
            adj.append(rel.end_node())
        for rel in self.graph.match(start_node=self.next_node, rel_type="retweeted", bidirectional=True):
            adj.append(rel.end_node())
        return adj

    def run(self):

        if self.next_node is None:
            print("Starting...")
            print("-" * 100)
            self.init_run()

        while True:

            t1 = datetime.now()

            g = open("control.json", 'r')
            control_var = json.load(g)
            g.close()
            if not control_var["continue"]:
                break

            t2 = datetime.now()

            if self.next_node["virtual"] == "T":
                try:
                    self.next_node = self.push_node(self.next_node)
                    print("(Pushed {0}".format(self.next_node["screen_name"]), end=") ")
                except tweepy.TweepError as exception:
                    print(exception)
                    self.jump()
            else:
                print("(Existed {0}".format(self.next_node["screen_name"]), end=") ")

            adj = self.get_adj()

            threshold, rand = self.w / (self.w + len(adj)), random.random()

            self.jump() if rand < threshold else self.walk(adj)

            t3 = datetime.now()

            print("File management: {0} \t Iter: {1} \t n:{2}".format(t2 - t1, t3 - t2, self.n))
            print("-" * 100)

        tmp = dict()
        tmp["next"] = self.next_node
        tmp["continue"] = True
        tmp["seed"] = self.seed
        tmp["n"] = self.n
        g = open("control.json", 'w')
        json.dump(tmp, g)
        g.close()


if __name__ == "__main__":
    f = open("secrets.json", 'r')
    config_tweepy = json.load(f)
    f.close()

    f = open("neo4jsecret.json", 'r')
    config_neo4j = json.load(f)
    f.close()

    if os.path.exists("control.json"):
        f = open("control.json", 'r')
        control = json.load(f)
        f.close()
    else:
        control = dict()
        control["seed"] = 850859913244852224
        control["continue"] = True
        control["next"] = None
        control["n"] = 1

    crawl = MHCrawler(config_tweepy, config_neo4j, control["n"], control["seed"], control["next"])
    crawl.run()
