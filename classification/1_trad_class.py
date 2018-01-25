import numpy as np
import pandas as pd
from scipy import interp
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, accuracy_score, roc_curve, auc, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from tmp.utils import cols_attr, cols_glove, cols_empath, cols_attr_c, cols_glove_c, cols_empath_c


def performance(y_true, y_pred, name, write_flag=False, print_flag=False):
    fpr, tpr, _ = roc_curve(y_true, y_pred)

    output = "%s, " \
             "F1-Score     %0.4f\n" % \
             (name,
              f1_score(y_true, y_pred))

    if write_flag:
        f = open("./data/results_{0}.txt".format(name), "w")
        f.write(output)
        f.close()
    if print_flag:
        print(output, end="")


def eval_gb(flag_x1, flag_x2, flag_y):
    df = pd.read_csv("../data/users_neighborhood_anon.csv")
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
    x_attr_c = np.array(df[cols_attr_c].values).reshape(-1, len(cols_attr_c))
    x_glove = np.array(df[cols_glove].values).reshape(-1, len(cols_glove))
    x_glove_c = np.array(df[cols_glove].values).reshape(-1, len(cols_glove_c))

    if flag_x1 == "neigh" and flag_x2 == "all":
        x = np.concatenate((x_attr, x_glove, x_attr_c, x_glove_c), axis=1)
    if flag_x1 == "neigh" and flag_x2 == "user":
        x = np.concatenate((x_attr, x_attr_c), axis=1)
    if flag_x1 == "neigh" and flag_x2 == "glove":
        x = np.concatenate((x_glove, x_glove_c), axis=1)

    if flag_x1 == "user" and flag_x2 == "all":
        x = np.concatenate((x_attr, x_glove), axis=1)
    if flag_x1 == "user" and flag_x2 == "user":
        x = x_attr
    if flag_x1 == "user" and flag_x2 == "glove":
        x = x_glove

    scaling = StandardScaler()
    x = scaling.fit_transform(x)

    # Set parameters for GBoost
    original_params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.01}

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

        gb = AdaBoostClassifier(n_estimators=50, learning_rate=.01)#GradientBoostingClassifier(**original_params)
        gb.fit(x_train, y_train, sample_weight=weights)

        y_pred_train = gb.predict(x_train)
        y_pred_test = gb.predict(x_test)
        y_pred_proba_test = gb.predict_proba(x_test)[:, 1].flatten()


        y_true = y_test
        y_pred = y_pred_test

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba_test)
        auc_test.append(auc(fpr, tpr))
        accuracy_test.append(accuracy_score(y_true, y_pred))
        recall_test.append(recall_score(y_true, y_pred, pos_label=1))

        print(confusion_matrix(y_true, y_pred))
        print("Accuracy   %0.4f" % accuracy_test[-1])
        print("Recall   %0.4f" % recall_test[-1])
        print("AUC   %0.4f" % auc_test[-1])

        # print(f1_train[-1])
        # print(f1_test[-1])

    accuracy_test = np.array(accuracy_test)
    recall_test = np.array(recall_test)
    auc_test = np.array(auc_test)

    print("Accuracy   %0.4f +-  %0.4f" % (accuracy_test.mean(), accuracy_test.std()))
    print("Recall    %0.4f +-  %0.4f" % (recall_test.mean(), recall_test.std()))
    print("AUC    %0.4f +-  %0.4f" % (auc_test.mean(), auc_test.std()))


# eval_gb("save", "", "hn")
# eval_gb("save", "", "sa")
#
# exit()

for flag_topology in ["user"]:
    for flag_features in ["all", "glove"]:
        for flag_detect in ["hn", "sa"]:
            print(flag_topology, flag_features, flag_detect)
            print("-" * 40)

            eval_gb(flag_topology, flag_features, flag_detect)
