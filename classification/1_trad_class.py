import numpy as np
import pandas as pd
from scipy import interp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from tmp.utils import cols_attr, cols_glove, cols_empath, cols_attr_c, cols_glove_c, cols_empath_c


def scale_vals(x_attr, x_attr_c, x_glove, x_glove_c, x_empath, x_empath_c):
    scaling = StandardScaler()
    x_attr = scaling.fit_transform(x_attr)
    x_attr_c = scaling.fit_transform(x_attr_c)
    x_glove = scaling.fit_transform(x_glove)
    x_glove_c = scaling.fit_transform(x_glove_c)
    x_empath = scaling.fit_transform(x_empath)
    x_empath_c = scaling.fit_transform(x_empath_c)
    return x_attr, x_attr_c, x_glove, x_glove_c, x_empath, x_empath_c


def eval_gb(flag_x, flag_y):
    df = pd.read_csv("../data/users_neighborhood_anon.csv")
    df.fillna(0, inplace=True)

    if flag_y == "hateful_normal":
        df = df[df.hate != "other"]
        y = np.array([1 if v == "hateful" else 0 for v in df["hate"].values])
    if flag_y == "suspended_non_suspended":
        np.random.seed(321)
        df2 = df[df["is_63_2"] == True].sample(544, axis=0)
        df3 = df[df["is_63_2"] == False].sample(4427, axis=0)
        df = pd.concat([df2, df3])
        y = np.array([1 if v else 0 for v in df["is_63_2"].values])


    # Get X values correa sponding  only to some columns
    X_attr = np.array(df[cols_attr].values).reshape(-1, len(cols_attr))
    X_attr_c = np.array(df[cols_attr_c].values).reshape(-1, len(cols_attr_c))
    X_glove = np.array(df[cols_glove].values).reshape(-1, len(cols_glove))
    X_glove_c = np.array(df[cols_glove].values).reshape(-1, len(cols_glove_c))
    X_empath = np.array(df[cols_empath].values).reshape(-1, len(cols_empath))
    X_empath_c = np.array(df[cols_empath].values).reshape(-1, len(cols_empath_c))

    # Gets X and y with all the values
    X = np.concatenate((X_attr, X_empath, X_glove, X_attr_c, X_glove_c, X_empath_c), axis=1)

    # Set parameters for GBoost
    original_params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.01}

    # Prepares split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    # Prepares weights for samples
    w_pos, w_neg = 18, 1

    accuracy, recall, f1, tprs, aucs, i = [], [], [], [], [], 0
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_indaex in skf.split(X, y):
        print("|", end="")
        y_train, y_test = y[train_index], y[test_index]
        X_attr_train, X_attr_test = X_attr[train_index], X_attr[test_index]
        X_attr_c_train, X_attr_c_test = X_attr_c[train_index], X_attr_c[test_index]
        X_glove_train, X_glove_test = X_glove[train_index], X_glove[test_index]
        X_glove_c_train, X_glove_c_test = X_glove_c[train_index], X_glove_c[test_index]
        X_empath_train, X_empath_test = X_empath[train_index], X_empath[test_index]
        X_empath_c_train, X_empath_c_test = X_empath_c[train_index], X_empath_c[test_index]

        # Scale
        X_attr_train, X_attr_c_train, X_glove_train, X_glove_c_train, X_empath_train, X_empath_c_train = \
            scale_vals(X_attr_train, X_attr_c_train, X_glove_train, X_glove_c_train, X_empath_train, X_empath_c_train)
        X_attr_test, X_attr_c_test, X_glove_test, X_glove_c_test, X_empath_test, X_empath_c_test = \
            scale_vals(X_attr_test, X_attr_c_test, X_glove_test, X_glove_c_test, X_empath_test, X_empath_c_test)

        if flag_x == "all":
            X_all_train = np.concatenate((X_attr_train, X_attr_c_train, X_glove_train, X_glove_c_train), axis=1)
            X_all_test = np.concatenate((X_attr_test, X_attr_c_test, X_glove_test, X_glove_c_test), axis=1)
        if flag_x == "neigh":
            X_all_train = np.concatenate((X_attr_c_train, X_glove_c_train), axis=1)
            X_all_test = np.concatenate((X_attr_c_test, X_glove_c_test), axis=1)
        if flag_x == "user":
            X_all_train = np.concatenate((X_attr_train, X_glove_train), axis=1)
            X_all_test = np.concatenate((X_attr_test, X_glove_test), axis=1)

        weights = [w_pos if v == 1 else w_neg for v in y_train]

        nb = GradientBoostingClassifier(**original_params)
        nb.fit(X_all_train, y_train, sample_weight=weights)
        y_all = nb.predict(X_all_test)
        y_pred_train = nb.predict(X_all_train)

        y_pred = y_all
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_auc = auc(fpr, tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        aucs.append(roc_auc)

        # print("Test", accuracy_score(y_test, y_pred), "Train", accuracy_score(y_train, y_pred_train))
        # print("Test", recall_score(y_test, y_pred, labels=[1], pos_label=1),
        #       "Train", recall_score(y_train, y_pred_train, labels=[1], pos_label=1))

        accuracy.append(accuracy_score(y_test, y_pred))
        recall.append(recall_score(y_test, y_pred, labels=[1], pos_label=1))
        f1.append(precision_score(y_test, y_pred, labels=[1], pos_label=1))
        # cnf_matrix = confusion_matrix(y_test, y_pred)
        # print(cnf_matrix)

        i += 1
    print()

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    recall = np.array(recall)
    f1 = np.array(f1)
    accuracy = np.array(accuracy)

    print("Recall %0.4f   +- %0.4f" % (recall.mean(), recall.std()))
    print("AUC %0.4f      +- %0.4f" % (mean_auc, std_auc))
    print("Accuracy %0.4f +- %0.4f" % (accuracy.mean(), accuracy.std()))

eval_gb("all", "hateful_normal")
eval_gb("neigh", "hateful_normal")
eval_gb("user", "hateful_normal")
eval_gb("all", "suspended_non_suspended")
eval_gb("neigh", "suspended_non_suspended")
eval_gb("user", "suspended_non_suspended")


# |||||
# Recall 0.8401   +- 0.0420
# AUC 0.7939      +- 0.0173
# Accuracy 0.7578 +- 0.0344
# |||||
# Recall 0.8327   +- 0.0319
# AUC 0.7896      +- 0.0149
# Accuracy 0.7560 +- 0.0381
# |||||
# Recall 0.8622   +- 0.0420
# AUC 0.8020      +- 0.0202
# Accuracy 0.7550 +- 0.0129
# |||||
# Recall 0.8196   +- 0.0949
# AUC 0.7773      +- 0.0158
# Accuracy 0.7444 +- 0.0755
# |||||
# Recall 0.8694   +- 0.0272
# AUC 0.7885      +- 0.0155
# Accuracy 0.7254 +- 0.0300
# |||||
# Recall 0.7901   +- 0.1381
# AUC 0.7597      +- 0.0196
# Accuracy 0.7361 +- 0.1023