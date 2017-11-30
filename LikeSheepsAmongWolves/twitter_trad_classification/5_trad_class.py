from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score, roc_curve, auc
from sklearn.preprocessing import Normalizer, MinMaxScaler, StandardScaler
import pandas as pd
import itertools
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

df = pd.read_csv("../data/users_all_neigh.csv")
df.fillna(0, inplace=True)

df = df[df.hate != "other"]

cols_attr = ["statuses_count", "followers_count", "followees_count", "favorites_count", "listed_count", "median_int",
             "average_int", "betweenness", "eigenvector", "in_degree", "out_degree", "sentiment",
             "number hashtags", "tweet number", "retweet number", "quote number", "status length",
             "number urls", "baddies",
             "mentions"]

cols_glove = ["{0}_glove".format(v) for v in range(300)]

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

cols = cols_attr + cols_glove
cols += ["c_" + v for v in cols]

X_all = np.array(df[cols].values).reshape(-1, len(cols))

pca = PCA(n_components=75)
scaling = StandardScaler().fit(X_all)
X_all = scaling.transform(X_all)
X_pca = X_all
X_pca = pca.fit_transform(X_all)
original_params = {'n_estimators': 400,
                   'max_leaf_nodes': 8,
                   'max_depth': None,
                   'learning_rate': 0.01,
                   'random_state': 2,
                   'min_samples_split': 3,
                   'subsample': 1}

accuracy, recall, f1, tprs, aucs = [], [], [], [], []

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
mean_fpr = np.linspace(0, 1, 100)


i = 1
for train_index, test_index in skf.split(X_all, y):
    X_all_train, X_all_test = X_pca[train_index], X_pca[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # nb = GradientBoostingClassifier(**original_params)

    nb = LinearSVC(penalty='l2', dual=True, C=0.5)

    weights = [30 if v == 1 else 1 for v in y_train]

    nb.fit(X_all_train, y_train, sample_weight=weights)
    y_all = nb.predict(X_all_test)

    y_pred = y_all
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    tprs.append(interp(mean_fpr, fpr, tpr))
    aucs.append(roc_auc)

    accuracy.append(accuracy_score(y_test, y_pred))
    recall.append(recall_score(y_test, y_pred, labels=[1], pos_label=1))
    f1.append(precision_score(y_test, y_pred, labels=[1], pos_label=1))
    cnf_matrix = confusion_matrix(y_test, y_pred)

    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    print(cnf_matrix)
    i += 1

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


recall = np.array(recall)
f1 = np.array(f1)
accuracy = np.array(accuracy)

print("Recall {0} +- {1}".format(recall.mean(), recall.std()))
print("F1-Score {0} +- {1}".format(f1.mean(), f1.std()))
print("Accuracy {0} +- {1}".format(accuracy.mean(), accuracy.std()))
