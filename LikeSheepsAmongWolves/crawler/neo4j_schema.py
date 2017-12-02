from functools import partial
from datetime import datetime
from py2neo import Node
import re

rem = partial(re.sub, "( |\n|\t)+", " ")


def tweepy2neo4j_virtual_user(user):
    user_neo4j = Node("User", id=user, virtual="T")
    return user_neo4j


def tweepy2neo4j_materialize_user(user_neo4j, user, n):
    user_neo4j["statuses_count"] = user.statuses_count
    user_neo4j["followers_count"] = user.followers_count
    user_neo4j["followees_count"] = user.friends_count
    user_neo4j["favorites_count"] = user.favourites_count
    user_neo4j["uname"] = user.name
    user_neo4j["listed_count"] = user.listed_count
    user_neo4j["screen_name"] = user.screen_name
    user_neo4j["time_zone"] = user.time_zone
    user_neo4j["description"] = user.description
    user_neo4j["location"] = user.location
    user_neo4j["profile_image_url"] = user.profile_image_url
    user_neo4j["lang"] = user.lang
    user_neo4j["default_profile"] = user.default_profile
    user_neo4j["geo_enabled"] = user.geo_enabled
    user_neo4j["default_profile_image"] = user.default_profile_image
    user_neo4j["created_at"] = user.created_at.timestamp()
    user_neo4j["verified"] = user.verified
    user_neo4j["virtual"] = "F"
    user_neo4j["number"] = n
    return user_neo4j


def tweepy2string_tweet(tweet):
    rt, qt, rp = "retweeted_status" in tweet._json and tweet._json["retweeted_status"] is not None, \
                 "quoted_status" in tweet._json and tweet._json["quoted_status"] is not None, \
                 "in_reply_to_screen_name" in tweet._json and tweet._json["in_reply_to_screen_name"] is not None
    return ";".join([str(tweet.id),
                     str("" if rt else rem(tweet.full_text)),
                     str(tweet.created_at.timestamp()),
                     str(tweet.favorite_count),
                     str(tweet.retweet_count),
                     str(rp), str("" if not rp else tweet.in_reply_to_status_id),
                     str("" if not rp else tweet.in_reply_to_user_id),
                     str(qt), str("" if not qt else tweet.quoted_status["id"]),
                     str("" if not qt else tweet.quoted_status["user"]["id"]),
                     str("" if not qt else rem(tweet.quoted_status["full_text"])),
                     str("" if not qt else datetime.strptime(tweet.quoted_status["created_at"],
                                                             "%a %b %d %H:%M:%S %z %Y").timestamp()),
                     str("" if not qt else tweet.quoted_status["favorite_count"]),
                     str("" if not qt else tweet.quoted_status["retweet_count"]),
                     str(rt), str("" if not rt else tweet.retweeted_status.id),
                     str("" if not rt else tweet.retweeted_status.user.id),
                     str("" if not rt else rem(tweet.retweeted_status.full_text)),
                     str("" if not rt else tweet.retweeted_status.created_at.timestamp()),
                     str("" if not rt else tweet.retweeted_status.favorite_count),
                     str("" if not rt else tweet.retweeted_status.retweet_count)])


def tweepy2neo4j_user(user, n):
    user_neo4j = Node("User",
                      id=user.id,
                      statuses_count=user.statuses_count,
                      followers_count=user.followers_count,
                      followees_count=user.friends_count,
                      favorites_count=user.favourites_count,
                      uname=user.name,
                      listed_count=user.listed_count,
                      screen_name=user.screen_name,
                      time_zone=user.time_zone,
                      description=rem(user.description),
                      location=user.location,
                      profile_image_url=user.profile_image_url,
                      lang=user.lang,
                      default_profile=user.default_profile,
                      geo_enabled=user.geo_enabled,
                      default_profile_image=user.default_profile_image,
                      verified=user.verified,
                      virtual="F",
                      created_at=user.created_at.timestamp(), number=n)
    return user_neo4j


def tweepy2neo4j_tweet(tweet):
    return Node("Tweet", content=tweet)


def tweepy2neo4j_media(media):
    return Node("Media", url=media[0], netloc=media[1], path=media[2])
