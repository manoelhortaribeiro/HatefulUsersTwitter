from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from LikeSheepsAmongWolves.tmp.utils import cols_attr, cols_glove, cols_empath, cols_attr_c, cols_glove_c, cols_empath_c
import matplotlib.pyplot as plt
from scipy import interp
import pandas as pd
import numpy as np


def scale_vals(x_attr, x_attr_c, x_glove, x_glove_c, x_empath, x_empath_c):
    scaling = StandardScaler()
    x_attr = scaling.fit_transform(x_attr)
    x_attr_c = scaling.fit_transform(x_attr_c)
    x_glove = scaling.fit_transform(x_glove)
    x_glove_c = scaling.fit_transform(x_glove_c)
    x_empath = scaling.fit_transform(x_empath)
    x_empath_c = scaling.fit_transform(x_empath_c)
    return x_attr, x_attr_c, x_glove, x_glove_c, x_empath, x_empath_c


df = pd.read_csv("../data/features/users_all_neighborhood.csv")
df.fillna(0, inplace=True)

df = df[df.hate != "other"]

# Get X values corresponding only to some columns
X_attr = np.array(df[cols_attr].values).reshape(-1, len(cols_attr))
X_attr_c = np.array(df[cols_attr_c].values).reshape(-1, len(cols_attr_c))
X_glove = np.array(df[cols_glove].values).reshape(-1, len(cols_glove))
X_glove_c = np.array(df[cols_glove].values).reshape(-1, len(cols_glove_c))
X_empath = np.array(df[cols_empath].values).reshape(-1, len(cols_empath))
X_empath_c = np.array(df[cols_empath].values).reshape(-1, len(cols_empath_c))

# Gets X and y with all the values
X = np.concatenate((X_attr, X_empath, X_glove, X_attr_c, X_glove_c, X_empath_c), axis=1)
y = np.array([1 if v == "hateful" else 0 for v in df["hate"].values])

# Set parameters for GBoost
original_params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.01}


# Prepares split
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

# Prepares weights for samples
w_pos, w_neg = 18, 1

accuracy, recall, f1, tprs, aucs, i = [], [], [], [], [], 0
mean_fpr = np.linspace(0, 1, 100)

for train_index, test_index in skf.split(X, y):
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

    print("ALL")
    X_all_train = np.concatenate((X_attr_train, X_attr_c_train, X_glove_train, X_glove_c_train), axis=1)
    X_all_test = np.concatenate((X_attr_test, X_attr_c_test, X_glove_test, X_glove_c_test), axis=1)

    # print("NEIGH")
    # X_all_train = np.concatenate((X_attr_c_train, X_glove_c_train), axis=1)
    # X_all_test = np.concatenate((X_attr_c_test, X_glove_c_test), axis=1)

    # print("USER")
    # X_all_train = np.concatenate((X_attr_train, X_glove_train), axis=1)
    # X_all_test = np.concatenate((X_attr_test, X_glove_test), axis=1)

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

    print("Test", accuracy_score(y_test, y_pred), "Train", accuracy_score(y_train, y_pred_train))
    print("Test", recall_score(y_test, y_pred, labels=[1], pos_label=1),
          "Train", recall_score(y_train, y_pred_train, labels=[1], pos_label=1))

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






# Dimensionality Reduction


# X_attr = scaling.fit_transform(X_attr)
# # X_all = pca.fit_transform(X_all)
#
# scaling = StandardScaler().fit(X_attr)
# # X_attr = scaling.transform(X_attr)
#
# # pca = PCA(n_components=50)
# scaling = StandardScaler().fit(X_glove)
# X_glove = scaling.transform(X_glove)
# # X_glove = pca.fit_transform(X_glove)
#
# scaling = StandardScaler().fit(X_empath)
# X_empath = scaling.transform(X_empath)
# # X_empath = pca.fit_transform(X_empath)
#
#
# X_all = np.array(df[cols].values).reshape(-1, len(cols))
#
# pca = PCA(n_components=75)
# scaling = StandardScaler().fit(X_all)
# X_all = scaling.transform(X_all)
# X_pca = X_all
# X_pca = pca.fit_transform(X_all)

# accuracy, recall, f1, tprs, aucs = [], [], [], [], []
#
# mean_fpr = np.linspace(0, 1, 100)
#
#
# i = 1
# for train_index, test_index in skf.split(X_all, y):
#     X_all_train, X_all_test = X_pca[train_index], X_pca[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     nb = GradientBoostingClassifier(**original_params)
#
#     weights = [30 if v == 1 else 1 for v in y_train]
# #
#     nb.fit(X_all_train, y_train, sample_weight=weights)
#     y_all = nb.predict(X_all_test)
#
#     y_pred = y_all
#     fpr, tpr, _ = roc_curve(y_test, y_pred)
#     roc_auc = auc(fpr, tpr)
#     tprs.append(interp(mean_fpr, fpr, tpr))
#     aucs.append(roc_auc)
#
#     accuracy.append(accuracy_score(y_test, y_pred))
#     recall.append(recall_score(y_test, y_pred, labels=[1], pos_label=1))
#     f1.append(precision_score(y_test, y_pred, labels=[1], pos_label=1))
#     cnf_matrix = confusion_matrix(y_test, y_pred)
#
#     plt.plot(fpr, tpr, lw=1, alpha=0.3,
#              label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#     print(cnf_matrix)
#     i += 1
#
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# plt.plot(mean_fpr, mean_tpr, color='b',
#          label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#          lw=2, alpha=.8)
#
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                  label=r'$\pm$ 1 std. dev.')
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.legend(loc="lower right")
# plt.show()
#
#
# recall = np.array(recall)
# f1 = np.array(f1)
# accuracy = np.array(accuracy)
#
# print("Recall {0} +- {1}".format(recall.mean(), recall.std()))
# print("F1-Score {0} +- {1}".format(f1.mean(), f1.std()))
# print("Accuracy {0} +- {1}".format(accuracy.mean(), accuracy.std()))
