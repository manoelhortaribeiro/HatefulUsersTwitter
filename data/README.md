## Users
user_id,hate,hate_neigh,normal_neigh,statuses_count,followers_count,followees_count,favorites_count,listed_count,betweenness,eigenvector,in_degree,out_degree

There is an anonymized and non-anonymized version, contains the following attributes (despite the ID):

    user_id                  - unique identifier from the user
    hate                     - ("hateful"|"normal"|"other"), other is non-annotated
    hate_neigh               - (True|False) - is it in the neighborhood of hateful?
    normal_neigh             - (True|False) - is it in the neighborhood of normal?
    statuses_count           - #Tweets
    follower_count           - #Followers
    followees_count          - #Followees
    favorites_count          - #Favorites
    listed_count             - #Listed
    betweenness              - Betwennes centrality, calculated in the RT graph
    eigenvector              - Eigenvector centrality,  "        "  "  "   "
    in_degree                - In Degree centrality,    "        "  "  "   "
    out_degree               - Out Degree centrality,   "        "  "  "   "
    *_empath                 - 100+ empath categories
    *_glove                  - 300 dim glove vector
    sentiment                - sentiment score,         
    subjectivity             - subjectivity score
    number hashtags          - #hashtags/tweet
    hashtags                 - actual hashtags
    tweet number             - %direct tweets
    retweet number           - %retweets
    quote number             - %quote tweets
    status length            - length of the statuses
    number urls              - number of urls per tweet in average
    baddies                  - number of bad words in average
    mentions                 - number of mentions in average
    is_50                    - was deleted up to 12/12/17
    is_63                    - was suspended up to 12/12/17
    is_50_2                  - was deleted up to 14/01/18
    is_63_2                  - was suspended up to 14/01/18
    time_diff                - average time diff between tweets
    time_diff_median         - median time diff between tweets

There is also a version with the average attributes for their neighborhood in the graph, which have additional attributes (the same), with a `c_` prefix.
    
    

## Tweets
\* Available upon request

19536788 tweets from 100,386 users. Up to 200 for each users.

Features:

    screen_name       - screename of the user on twitter
    tweet_id          - number with the identifier of a the tweet
    tweet_text        - if status is a tweet/quote, the text written
    tweet_creation    - date in unix time when the tweet was tweeted
    tweet_fav         - number of favorites
    tweet_rt          - number of retweets
    rp_flag           - flag that indicates if the tweet is a reply
    rp_status         - id of the replied status
    rp_user           - id of the replied user
    qt_flag           - flag that indicates if the tweet is a quote
    qt_user_id        - id of the quoted user
    qt_status_id      - id of the quoted status
    qt_text           - text of the quoted status
    qt_creation       - date of creation of the quoted status
    qt_fav            - number of favorites of the quoted status
    qt_rt             - number of retweets of the quoted status
    rt_flag           - flag that indicates if a tweet is a retweet
    rt_user_id        - id of the retweeted user
    rt_status_id      - id of the retweeted status
    rt_text           - text of the retweeted status
    rt_creation       - creation date of the retweeted status
    rt_fav            - number of favorites of the retweeted status
    rt_rt             - number of retweets of the retweeted status
    
## Graph

There is a networkx file with only the edges and IDs (anonymized) (user_clean.graphml) 

There is a networkx file with only the edges and IDs non_anon (user_clean_d.graphml) 

Also there are extra files for the GraphSage, on folders `hate` and `suspended`.