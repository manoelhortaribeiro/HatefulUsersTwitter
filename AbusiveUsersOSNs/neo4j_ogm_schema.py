from py2neo.ogm import RelatedFrom, RelatedTo, Property, GraphObject
import datetime
import json


class User(GraphObject):
    __primarykey__ = "id"

    id = Property()
    statuses_count = Property()
    followers_count = Property()
    followees_count = Property()
    listed_count = Property()
    favorites_count = Property()
    screen_name = Property()
    name = Property()
    description = Property()
    lang = Property()
    time_zone = Property()
    location = Property()
    profile_image_url = Property()
    default_profile = Property()
    default_profile_image = Property()
    geo_enabled = Property()
    verified = Property()
    created_at = Property()
    flag = Property()

    quoted = RelatedTo("User", "QUOTED")
    retweeted = RelatedTo("User", "RETWEETED")
    tweeted_by_me = RelatedTo("Tweet", "TWEETED")

    def __repr__(self):
            return "<id:{0}, screen_name:{1}, name:{2}, flwr:{3}, flwe:{4}, lang:{5}, desc:{6}, member_since:{7}>"\
                .format(self.id,
                        self.screen_name,
                        self.name,
                        self.followers_count,
                        self.followees_count,
                        self.lang,
                        self.description.replace('\n', ' '),
                        self.created_at)


class Tweet(GraphObject):
    __primarykey__ = "id"

    id = Property()
    content = Property()
    # tweeted_me = RelatedFrom("User", "TWEETED")

    def __repr__(self):
        content = json.loads(self.content)
        print(content)
        return "<id:{0}, lang:{1}, text:{2}, date:{3}>".format(content["id"],
                                                               content["lang"],
                                                               content["full_text"].replace('\n', ' '),
                                                               content["created_at"])


def tweepy2neo4j_user(user):
    user_neo4j = User()
    user_neo4j.id = user.id
    user_neo4j.statuses_count = user.statuses_count
    user_neo4j.followers_count = user.followers_count
    user_neo4j.followees_count = user.friends_count
    user_neo4j.favorites_count = user.favourites_count
    user_neo4j.listed_count = user.listed_count
    user_neo4j.screen_name = user.screen_name
    user_neo4j.name = user.name
    user_neo4j.description = user.description
    user_neo4j.location = user.location
    user_neo4j.profile_image_url = user.profile_image_url
    user_neo4j.time_zone = user.time_zone
    user_neo4j.lang = user.lang
    user_neo4j.default_profile = user.default_profile
    user_neo4j.default_profile_image = user.default_profile_image
    user_neo4j.geo_enabled = user.geo_enabled
    user_neo4j.verified = user.verified
    user_neo4j.created_at = user.created_at.timestamp()
    return user_neo4j


def tweepy2neo4j_tweet(tweet):
    tweet_neo4j = Tweet()
    tweet_neo4j.id = tweet._json['id']
    tweet_neo4j.content = json.dumps(tweet._json)
    return tweet_neo4j
