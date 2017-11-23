from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import Normalizer, MinMaxScaler
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

df = pd.read_csv("../data/users_all.csv")
df.fillna(0, inplace=True)

df = df[df.hate != "other"]

cols_attr = ["statuses_count", "followers_count", "followees_count", "favorites_count", "listed_count", "median_int",
             "average_int", "betweenness", "eigenvector", "in_degree", "out_degree", "sentiment"]

cols_glove = ["{0}_glove".format(v) for v in range(384)]

cols_empath = [
    "traveling_empath", "fashion_empath", "sadness_empath", "fun_empath",
    "noise_empath", "phone_empath", "cold_empath", "driving_empath", "love_empath", "weather_empath",
    "magic_empath", "messaging_empath", "appearance_empath", "cooking_empath", "contentment_empath",
    "business_empath", "art_empath", "politics_empath", "pain_empath", "ocean_empath", "economics_empath",
    "lust_empath", "philosophy_empath", "communication_empath", "giving_empath", "kill_empath", "sleep_empath",
    "sound_empath", "leisure_empath", "body_empath", "law_empath", "pet_empath", "fear_empath", "affection_empath",
    "stealing_empath", "vacation_empath", "children_empath", "swearing_terms_empath", "independence_empath",
    "leader_empath", "toy_empath", "animal_empath", "monster_empath", "masculine_empath", "crime_empath",
    "health_empath", "valuable_empath", "legend_empath", "rage_empath", "writing_empath", "beauty_empath",
    "hipster_empath", "irritability_empath", "negotiate_empath", "work_empath", "meeting_empath", "worship_empath",
    "reading_empath", "neglect_empath", "youth_empath", "swimming_empath", "medical_emergency_empath",
    "white_collar_job_empath", "banking_empath", "zest_empath", "sports_empath", "social_media_empath",
    "morning_empath", "positive_emotion_empath", "breaking_empath", "hygiene_empath", "dance_empath",
    "aggression_empath", "computer_empath", "night_empath", "horror_empath", "military_empath", "optimism_empath",
    "exasperation_empath", "exercise_empath", "emotional_empath", "celebration_empath", "dispute_empath",
    "music_empath", "torment_empath", "nervousness_empath", "ridicule_empath", "warmth_empath", "royalty_empath",
    "speaking_empath", "prison_empath", "heroic_empath", "disgust_empath", "shape_and_size_empath",
    "movement_empath", "ancient_empath", "wedding_empath", "terrorism_empath", "envy_empath", "achievement_empath",
    "surprise_empath", "anticipation_empath", "real_estate_empath", "cheerfulness_empath", "furniture_empath",
    "domestic_work_empath", "play_empath", "deception_empath", "liquid_empath", "suffering_empath",
    "restaurant_empath", "competing_empath", "programming_empath", "negative_emotion_empath", "exotic_empath",
    "party_empath", "fabric_empath", "dominant_heirarchical_empath", "wealthy_empath", "timidity_empath",
    "childish_empath", "healing_empath", "listen_empath", "school_empath", "order_empath", "eating_empath",
    "water_empath", "war_empath", "joy_empath", "cleaning_empath", "government_empath", "clothing_empath",
    "help_empath", "vehicle_empath", "money_empath", "air_travel_empath", "science_empath", "beach_empath",
    "urban_empath", "sexual_empath", "tool_empath", "payment_empath", "superhero_empath", "ugliness_empath",
    "occupation_empath", "politeness_empath", "attractive_empath", "college_empath", "family_empath",
    "friends_empath", "anger_empath", "fire_empath", "weakness_empath", "strength_empath", "home_empath",
    "poor_empath", "gain_empath", "injury_empath", "office_empath", "divine_empath", "sailing_empath",
    "musical_empath", "dominant_personality_empath", "hearing_empath", "confusion_empath", "rural_empath",
    "weapon_empath", "internet_empath", "hate_empath", "technology_empath", "sympathy_empath", "fight_empath",
    "car_empath", "hiking_empath", "pride_empath", "disappointment_empath", "anonymity_empath", "shame_empath",
    "violence_empath", "trust_empath", "alcohol_empath", "smell_empath", "blue_collar_job_empath", "death_empath",
    "feminine_empath", "medieval_empath", "journalism_empath", "farming_empath", "plant_empath", "shopping_empath",
    "ship_empath", "religion_empath", "tourism_empath", "power_empath"]

y = np.array([1 if v == "hateful" else 0 for v in df["hate"].values])
pca = PCA(n_components=25)

X_attr = np.array(df[cols_attr].values).reshape(-1, len(cols_attr))
X_glove = np.array(df[cols_glove].values).reshape(-1, len(cols_glove))
X_empath = np.array(df[cols_empath].values).reshape(-1, len(cols_empath))

scaling = MinMaxScaler().fit(X_attr)
X_attr = scaling.transform(X_attr)

scaling = MinMaxScaler().fit(X_glove)
# X_glove = scaling.transform(X_glove)
# X_glove = pca.fit_transform(X_glove)

scaling = MinMaxScaler().fit(X_empath)
# X_empath = scaling.transform(X_empath)
# X_empath = pca.fit_transform(X_empath)

X = np.concatenate((X_attr, X_empath, X_glove), axis=1)
scaling = MinMaxScaler().fit(X)
X = scaling.transform(X)

accuracy = []
recall = []
f1 = []
# print(len(X))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
skf.get_n_splits(X_attr, y)

for train_index, test_index in skf.split(X, y):
    X_attr_train, X_attr_test = X_attr[train_index], X_attr[test_index]
    X_glove_train, X_glove_test = X_glove[train_index], X_glove[test_index]
    X_empath_train, X_empath_test = X_empath[train_index], X_empath[test_index]
    y_train, y_test = y[train_index], y[test_index]

    nb = GaussianNB()

    nb.fit(X_attr_train, y_train)
    y_attr = nb.predict_proba(X_attr_test)

    nb.fit(X_glove_train, y_train)
    y_glove = nb.predict_proba(X_glove_test)

    nb.fit(X_empath_train, y_train)
    y_empath = nb.predict_proba(X_empath_test)

    y_pred = y_attr
    final = (np.array(y_attr) + np.array(y_empath)) / 3
    final = np.argmax(final, axis=1)
    y_pred = final

    accuracy.append(accuracy_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred, labels=[1], pos_label=1))
    f1.append(f1_score(y_test, y_pred, pos_label=1))
    class_report = classification_report(y_test, y_pred)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    # print(recall)
    print(cnf_matrix)
    # print(class_report)

recall = np.array(recall)
print("Recall {0} +- {1}".format(recall.mean(), recall.std()))
f1 = np.array(f1)
print("F1-Score {0} +- {1}".format(f1.mean(), f1.std()))
accuracy = np.array(accuracy)
print("Accuracy {0} +- {1}".format(accuracy.mean(), accuracy.std()))
