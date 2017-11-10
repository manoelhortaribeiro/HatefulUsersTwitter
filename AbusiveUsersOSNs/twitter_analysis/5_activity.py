import pandas as pd

# This is in case tweets_selected.csv is not generated
# import csv
# df = pd.read_csv("./annotated.csv")
# dict_ann = dict()
#
# for user_id in df.user_id.values:
#     dict_ann[user_id] = True
#
# print(dict_ann)
# f1, f2 = open("./tweets.csv", "r"), open("./tweets_selected.csv", "w")
# csv_reader = csv.DictReader(f1)
# csv_writer = csv.DictWriter(f2, fieldnames=csv_reader.fieldnames)
# csv_writer.writeheader()
# for tweet in csv_reader:
#     if int(tweet["user_id"]) in dict_ann:
#         csv_writer.writerow(tweet)
#
# f1.close()
# f2.close()

df = pd.read_csv("tweets_selected.csv")
print(df)
