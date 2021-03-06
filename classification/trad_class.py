from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tmp.utils import cols_attr, cols_glove
import pandas as pd
import numpy as np


def eval_gb(flag_x1, flag_x2, flag_y):
    df = pd.read_csv("../data/users_all_neighborhood.csv")
    df.fillna(0, inplace=True)

    if flag_y == "hn":
        df = df[df.hate != "other"]
        y = np.array([1 if v == "hateful" else 0 for v in df["hate"].values])

    if flag_y == "sa":
        np.random.seed(321)
        df2 = df[df["is_63_2"] == True].sample(668, axis=0)
        df3 = df[df["is_63_2"] == False].sample(5405, axis=0)
        df = pd.concat([df2, df3])
        y = np.array([1 if v else 0 for v in df["is_63_2"].values])

    if flag_x1 == "save":
        x_id = np.array(df["user_id"].values)
        np.save("../data/graph-input/indexes_{0}.npy".format(flag_y), x_id)
        return

    # Get X values corresponding  only to some columns
    x_attr = np.array(df[cols_attr].values).reshape(-1, len(cols_attr))
    x_glove = np.array(df[cols_glove].values).reshape(-1, len(cols_glove))

    if flag_x2 == "all":
        x = np.concatenate((x_attr, x_glove), axis=1)
    if flag_x2 == "glove":
        x = x_glove

    scaling = StandardScaler()
    x = scaling.fit_transform(x)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=123)

    recall_test = []
    accuracy_test = []
    auc_test = []
    for train_index, test_index in skf.split(x, y):
        y_train, y_test = y[train_index], y[test_index]
        x_train, x_test = x[train_index], x[test_index]

        # Prepares weights for samples
        w_pos, w_neg = 10, 1

        weights = [w_pos if v == 1 else w_neg for v in y_train]

        if flag_x1 == "gradientboost":
            gb = GradientBoostingClassifier(max_depth=5, n_estimators=100, learning_rate=0.01)

        if flag_x1 == "adaboost":
            gb = AdaBoostClassifier(n_estimators=75, learning_rate=.01)

        gb.fit(x_train, y_train, sample_weight=weights)

        y_pred_train = gb.predict(x_train)
        y_pred_test = gb.predict(x_test)
        y_pred_proba_test = gb.predict_proba(x_test)[:, 1].flatten()

        y_true = y_test
        y_pred = y_pred_test

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba_test)
        auc_test.append(auc(fpr, tpr))
        accuracy_test.append(accuracy_score(y_true, y_pred))
        recall_test.append(f1_score(y_true, y_pred, pos_label=1))

    accuracy_test = np.array(accuracy_test)
    recall_test = np.array(recall_test)
    auc_test = np.array(auc_test)

    print("Accuracy   %0.4f +-  %0.4f" % (accuracy_test.mean(), accuracy_test.std()))
    print("F1-Score    %0.4f +-  %0.4f" % (recall_test.mean(), recall_test.std()))
    print("AUC    %0.4f +-  %0.4f" % (auc_test.mean(), auc_test.std()))


for flag_model in ["gradientboost", "adaboost"]:
    for flag_features in ["all", "glove"]:
        for flag_detect in ["hn", "sa"]:
            print()
            print(flag_model, flag_features, flag_detect)
            print("-" * 40)

            eval_gb(flag_model, flag_features, flag_detect)
