import pandas as pd

tweets = pd.read_csv("../data/tweets.csv")
tweets.sort_values(by=["user_id", "tweet_creation"], ascending=True, inplace=True)
tweets["time_diff"] = tweets.groupby("user_id", sort=False).tweet_creation.diff()
time_diff_series_mean = tweets.groupby("user_id", sort=False).time_diff.mean()
time_diff_series_median = tweets.groupby("user_id", sort=False).time_diff.median()
time_diff = time_diff_series_mean.to_frame()
time_diff["time_diff_median"] = time_diff_series_median
time_diff.to_csv("../data/time_diff.csv")
