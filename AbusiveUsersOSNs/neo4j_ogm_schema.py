from py2neo import Node
import json


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
    user_neo4j["verified"] = user.verified
    user_neo4j["virtual"] = "F"
    user_neo4j["n"] = n
    return user_neo4j


def tweepy2neo4j_user(user, n):
    user_neo4j = Node("User", id=user.id, statuses_count=user.statuses_count, followers_count=user.followers_count,
                      followees_count=user.friends_count, favorites_count=user.favourites_count, uname=user.name,
                      listed_count=user.listed_count, screen_name=user.screen_name, time_zone=user.time_zone,
                      description=user.description, location=user.location, profile_image_url=user.profile_image_url,
                      lang=user.lang, default_profile=user.default_profile, geo_enabled=user.geo_enabled,
                      default_profile_image=user.default_profile_image, verified=user.verified, virtual="F",
                      created_at=user.created_at.timestamp(), n=n)
    return user_neo4j


def tweepy2neo4j_tweet(tweet):
    tweet_neo4j = Node("Tweet", content=tweet)
    return tweet_neo4j


def tweepy2neo4j_media(media, t_id):
    media_neo4j = Node("Media", url=media, id=t_id)
    return media_neo4j
