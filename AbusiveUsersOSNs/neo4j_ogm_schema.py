from py2neo.ogm import RelatedFrom, RelatedTo, Property, GraphObject
from py2neo import Graph


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


class Tweet(GraphObject):
    __primarykey__ = "id"

    id = Property()
    content = Property()
    tweeted_me = RelatedFrom("User", "TWEETED")
