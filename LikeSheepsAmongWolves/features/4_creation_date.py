import pandas as pd

# This is in case created_at_hate.csv is not generated
df1 = pd.read_csv("./created_at.csv")
df2 = pd.read_csv("./users_attributes.csv")
hateful = dict()

for row in df2.iterrows():
    if row[1][1] == "hateful":
        hateful[row[1][0]] = "hateful"
    if row[1][1] == "normal":
        hateful[row[1][0]] = "normal"

to_append = list()
for row in df1.iterrows():
    if row[1][0] in hateful:
        to_append.append(hateful[row[1][0]])
    else:
        to_append.append("None")

df1['hate'] = pd.Series(to_append, index=df1.index)

df1.to_csv("created_at_hate.csv", index=False)
