from py2neo import Graph, Relationship, NodeSelector
from crawler.neo4j_schema import *
from urllib.parse import urlparse
from datetime import datetime
import random
import tweepy
import json
import os


class MHCrawler:

    def __init__(self, auth_tweepy, auth_neo4j, n, seed, next_node, w=50):
        # sets current account on Twitter, as specified in the secrets file. Only uses one.
        self.curr_acc = 0
        self.accounts = list(auth_tweepy.items())
        self.api = MHCrawler.auth_tweepy(self.accounts[self.curr_acc][1])

        # authenticates neo4j
        self.graph = MHCrawler.auth_neo4j(auth_neo4j)

        # starts node selector from py2neo
        self.node_selector = NodeSelector(self.graph)

        # starts random walk parameters
        self.seed, self.n, self.w = seed, n, w
        self.next_node = list(self.node_selector.select("User", id=next_node["id"]))[0] \
            if next_node is not None else next_node
        self.previous_node = None

        # checks for uniqueness constraints/index keys in the database and creates them if they don't exist
        if self.graph.schema.get_uniqueness_constraints("User") != ["id"]:
            self.graph.schema.create_uniqueness_constraint("User", "id")
        if self.graph.schema.get_uniqueness_constraints("Tweet") != ["id"]:
            self.graph.schema.create_uniqueness_constraint("Tweet", "id")
        if self.graph.schema.get_uniqueness_constraints("Media") != ["url"]:
            self.graph.schema.create_uniqueness_constraint("Media", "url")
        if "number" not in self.graph.schema.get_indexes("User"):
            self.graph.schema.create_index("User", "number")

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # static methods to handle twitter_rwwj_control.json
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @staticmethod
    def control_break():
        """Basically the way to stop the crawler, just go into the control file and change the continue value to 0"""
        g = open("../secrets/twitter_rwwj_control.json", 'r')
        control_var = json.load(g)
        g.close()
        if not control_var["continue"]:
            return True
        else:
            return False

    @staticmethod
    def control_save(next_node, cont, seed, n):
        tmp = dict()
        tmp["next"], tmp["continue"], tmp["seed"], tmp["n"] = next_node, cont, seed, n
        g = open("../secrets/twitter_rwwj_control.json", 'w')
        json.dump(tmp, g)
        g.close()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # static methods authenticate twitter and neo4j
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    @staticmethod
    def auth_tweepy(auth):
        oauth = tweepy.OAuthHandler(auth["consumer_key"], auth["consumer_secret"])
        oauth.set_access_token(auth["access_token"], auth["access_secret"])
        return tweepy.API(oauth)

    @staticmethod
    def auth_neo4j(auth):
        return Graph(auth["host"], password=auth["password"])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # methods get info from twitter's payload
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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

    @staticmethod
    def get_urls(status):

        if 'urls' in status.entities:
            urls = []
            for url in status.entities['urls']:
                if 'expanded_url' in url and url['expanded_url'] is not None:
                    if urlparse(url['expanded_url']).netloc != "twitter.com":
                        tmp = urlparse(url["expanded_url"])
                        urls.append((tmp.netloc + tmp.path, tmp.netloc, tmp.path))
            return urls
        else:
            return []

    def get_tweets(self, user_p):

        tl = self.api.user_timeline(include_rts=True, count=200, trim_user=False, exclude_replies=False,
                                    user_id=user_p["id"], tweet_mode='extended')

        tweets, urls, quoted, retweeted = [], [], [], []

        for tweet in tl:
            tweets.append(tweepy2string_tweet(tweet))
            urls += MHCrawler.get_urls(tweet)
            quoted += MHCrawler.get_quoted_users(tweet)
            retweeted += MHCrawler.get_retweeted_users(tweet)

        tweet_p = tweepy2neo4j_tweet(tweets)
        self.graph.create(tweet_p)
        self.graph.create(Relationship(user_p, "tweeted", tweet_p))

        for url in urls:
            url_p = tweepy2neo4j_media(url)
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

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # pushes node into the graph
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    def push_node(self, node):
        user = self.api.get_user(node["id"])

        if "lang" not in user._json or user._json["lang"] != "en":
            raise IOError("User is not english speaking")

        user_p = tweepy2neo4j_materialize_user(node, user, self.n)
        self.n += 1
        self.graph.push(user_p)
        self.get_tweets(user_p)
        return user_p

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # random walk
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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
        print(list(self.node_selector.select("User", number=random.randint(1, self.n - 1))))
        self.next_node = list(self.node_selector.select("User", number=random.randint(1, self.n - 1)))[0]
        print("Jump {0}".format(self.next_node["screen_name"]))

    def get_adj(self):
        return list(set([rel.end_node() for rel in self.graph.match(start_node=self.next_node,
                                                                    rel_type="retweeted", bidirectional=True)]))

    def run(self):

        if self.next_node is None:
            print("Starting... \n " + "-" * 100)
            self.init_run()

        while True:

            # try:

            ts = datetime.now()

            if MHCrawler.control_break():
                break

            if self.next_node["virtual"] == "T":
                try:
                    self.next_node = self.push_node(self.next_node)
                    print("(Pushed {0}".format(self.next_node["screen_name"]), end=") ")
                except tweepy.TweepError as exception:
                    print(exception)
                    self.jump()
                except IOError as exception:
                    print(exception)
                    self.next_node = self.previous_node
                    continue
            else:
                print("(Existed {0}".format(self.next_node["screen_name"]), end=") ")

            adj = self.get_adj()
            print(adj)
            threshold, rand = self.w / (self.w + len(adj)), random.random()
            self.previous_node = self.next_node
            self.jump() if rand < threshold else self.walk(adj)

            print("Iter: {0} \t n:{1} \t tr:{2}".format(datetime.now() - ts, self.n, threshold))
            print("-" * 100)

            # except Exception as exception:
            #     print(exception)
            #     MHCrawler.control_save(self.next_node, True, self.seed, self.n)

        MHCrawler.control_save(self.next_node, True, self.seed, self.n)


if __name__ == "__main__":

    # Opens the twitter secrets as shown in the example _twitter_secrets.json;
    f = open("../secrets/twitter_secrets.json", 'r')
    config_tweepy = json.load(f)
    f.close()

    # Opens the neo4j secrets as shown in the example _twitter_secrets.json;
    f = open("../secrets/twitter_neo4jsecret.json", 'r')
    config_neo4j = json.load(f)
    f.close()

    # Opens a control json or sets custom seeds and starts control;
    if os.path.exists("twitter_rwwj_control.json"):
        f = open("../secrets/twitter_rwwj_control.json", 'r')
        control = json.load(f)
        f.close()
    else:
        control = dict()
        control["seed"], control["continue"], control["next"], control["n"] = 850859913244852224, True, None, 1

    crawl = MHCrawler(config_tweepy, config_neo4j, control["n"], control["seed"], control["next"])
    crawl.run()
