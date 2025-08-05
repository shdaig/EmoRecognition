import numpy as np
import pandas as pd

from drowsiness.emo import dfeats_utils as dfu

from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

blendshapes_df = pd.read_csv("stage_1_feature_extraction/stage_1_results/dfeats_diff.csv")
face_blendshapes_names = dfu.get_features_names()
y = blendshapes_df["label"].to_numpy()
blendshapes_df = blendshapes_df.drop(["filename", "label"], axis=1)
x = blendshapes_df.to_numpy()

label, cnt = np.unique(y, return_counts=True)
print(f"neutral: {label[0]} - {cnt[0]} [{round(cnt[0] / np.sum(cnt) * 100, 2)} %]")
print(f"smile: {label[1]} - {cnt[1]} [{round(cnt[1] / np.sum(cnt) * 100, 2)} %]")
print(f"disgust: {label[2]} - {cnt[2]} [{round(cnt[2] / np.sum(cnt) * 100, 2)} %]")

logreg_3classes_score = []
mlp_3classes_score = []

logreg_disgust_score = []
mlp_disgust_score = []

logreg_smile_score = []
mlp_smile_score = []

logreg_neutral_score = []
mlp_neutral_score = []

logreg_all_vs_rest_score = []
mlp_all_vs_rest_score = []

confusion_matrix_3classes_lr = np.zeros((3, 3))
confusion_matrix_3classes_mlp = np.zeros((3, 3))

confusion_matrix_ovr_lr = np.zeros((3, 3))
confusion_matrix_ovr_mlp = np.zeros((3, 3))

n_split = 4

skf = StratifiedKFold(n_splits=n_split)
for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    x_train = x[train_index]
    x_test = x[test_index]
    y_train = y[train_index]
    y_test = y[test_index]

    logreg = LogisticRegression(random_state=0, class_weight='balanced', C=1.0).fit(x_train, y_train)
    y_pred = logreg.predict(x_test)
    logreg_3classes_score.append(balanced_accuracy_score(y_test, y_pred))
    confusion_matrix_3classes_lr += confusion_matrix(y_test, y_pred)
    mlpc = MLPClassifier(random_state=0, max_iter=300, hidden_layer_sizes=(200, 200)).fit(x_train, y_train)
    # mlpc = RandomForestClassifier(random_state=0).fit(x_train, y_train)
    y_pred = mlpc.predict(x_test)
    mlp_3classes_score.append(balanced_accuracy_score(y_test, y_pred))
    confusion_matrix_3classes_mlp += confusion_matrix(y_test, y_pred)

    # disgust vs rest
    y_add = y.copy()
    y_add[y == 2] = 0
    y_add[y == 1] = 1
    y_add[y == 0] = 1
    y_train = y_add[train_index]
    y_test = y_add[test_index]
    logreg_disgust = LogisticRegression(random_state=0, class_weight='balanced', C=1.0).fit(x_train, y_train)
    y_pred = logreg_disgust.predict(x_test)
    y_pred_proba_disgust_lr = logreg_disgust.predict_proba(x_test)
    logreg_disgust_score.append(balanced_accuracy_score(y_test, y_pred))
    mlpc_disgust = MLPClassifier(random_state=0, max_iter=300, hidden_layer_sizes=(200, 200)).fit(x_train, y_train)
    # mlpc_disgust = RandomForestClassifier(random_state=0).fit(x_train, y_train)
    y_pred = mlpc_disgust.predict(x_test)
    mlp_disgust_score.append(balanced_accuracy_score(y_test, y_pred))
    y_pred_proba_disgust_mlp = mlpc_disgust.predict_proba(x_test)

    # smile vs rest
    y_add = y.copy()
    y_add[y == 2] = 1
    y_add[y == 1] = 0
    y_add[y == 0] = 1
    y_train = y_add[train_index]
    y_test = y_add[test_index]
    logreg_smile = LogisticRegression(random_state=0, class_weight='balanced', C=1.0).fit(x_train, y_train)
    y_pred = logreg_smile.predict(x_test)
    y_pred_proba_smile_lr = logreg_smile.predict_proba(x_test)
    logreg_smile_score.append(balanced_accuracy_score(y_test, y_pred))
    mlpc_smile = MLPClassifier(random_state=0, max_iter=300, hidden_layer_sizes=(200, 200)).fit(x_train, y_train)
    # mlpc_smile = RandomForestClassifier(random_state=0).fit(x_train, y_train)
    y_pred = mlpc_smile.predict(x_test)
    mlp_smile_score.append(balanced_accuracy_score(y_test, y_pred))
    y_pred_proba_smile_mlp = mlpc_smile.predict_proba(x_test)

    # neutral vs rest
    y_add = y.copy()
    y_add[y == 2] = 1
    y_add[y == 1] = 1
    y_add[y == 0] = 0
    y_train = y_add[train_index]
    y_test = y_add[test_index]
    logreg_neutral = LogisticRegression(random_state=0, class_weight='balanced', C=1.0).fit(x_train, y_train)
    y_pred = logreg_neutral.predict(x_test)
    y_pred_proba_neutral_lr = logreg_neutral.predict_proba(x_test)
    logreg_neutral_score.append(balanced_accuracy_score(y_test, y_pred))
    mlpc_neutral = MLPClassifier(random_state=0, max_iter=300, hidden_layer_sizes=(200, 200)).fit(x_train, y_train)
    # mlpc_neutral = RandomForestClassifier(random_state=0).fit(x_train, y_train)
    y_pred = mlpc_neutral.predict(x_test)
    mlp_neutral_score.append(balanced_accuracy_score(y_test, y_pred))
    y_pred_proba_neutral_mlp = mlpc_neutral.predict_proba(x_test)

    y_all_vs_rest_pred_lr = []
    y_all_vs_rest_pred_mlp = []
    for j in range(len(test_index)):
        classes_probas = [y_pred_proba_neutral_lr[j][0], y_pred_proba_smile_lr[j][0], y_pred_proba_disgust_lr[j][0]]
        class_idx = np.argmax(classes_probas)
        y_all_vs_rest_pred_lr.append(class_idx)

        classes_probas = [y_pred_proba_neutral_mlp[j][0], y_pred_proba_smile_mlp[j][0], y_pred_proba_disgust_mlp[j][0]]
        class_idx = np.argmax(classes_probas)
        y_all_vs_rest_pred_mlp.append(class_idx)

    y_test = y[test_index]
    logreg_all_vs_rest_score.append(balanced_accuracy_score(y_test, y_all_vs_rest_pred_lr))
    confusion_matrix_ovr_lr += confusion_matrix(y_test, y_all_vs_rest_pred_lr)
    mlp_all_vs_rest_score.append(balanced_accuracy_score(y_test, y_all_vs_rest_pred_mlp))
    confusion_matrix_ovr_mlp += confusion_matrix(y_test, y_all_vs_rest_pred_mlp)

confusion_matrix_3classes_lr /= n_split
confusion_matrix_3classes_mlp /= n_split
confusion_matrix_ovr_lr /= n_split
confusion_matrix_ovr_mlp /= n_split

print()
print("\t3 classes")
print("\t\tlogreg: ", round(np.mean(logreg_3classes_score), 4))
print(np.round((confusion_matrix_3classes_lr.T / np.sum(confusion_matrix_3classes_lr, axis=1)).T, 2))
print("\t\tmlp: ", round(np.mean(mlp_3classes_score), 4))
print(np.round((confusion_matrix_3classes_mlp.T / np.sum(confusion_matrix_3classes_mlp, axis=1)).T, 2))
print()
print("\tdisgust vs rest")
print("\t\tlogreg: ", round(np.mean(logreg_disgust_score), 4))
print("\t\tmlp: ", round(np.mean(mlp_disgust_score), 4))
print()
print("\tsmile vs rest")
print("\t\tlogreg: ", round(np.mean(logreg_smile_score), 4))
print("\t\tmlp: ", round(np.mean(mlp_smile_score), 4))
print()
print("\tneutral vs rest")
print("\t\tlogreg: ", round(np.mean(logreg_neutral_score), 4))
print("\t\tmlp: ", round(np.mean(mlp_neutral_score), 4))
print()
print("\tone vs rest - 3 classes - max proba")
print("\t\tlogreg: ", round(np.mean(logreg_all_vs_rest_score), 4))
print(np.round((confusion_matrix_ovr_lr.T / np.sum(confusion_matrix_ovr_lr, axis=1)).T, 2))
print("\t\tmlp: ", round(np.mean(mlp_all_vs_rest_score), 4))
print(np.round((confusion_matrix_ovr_mlp.T / np.sum(confusion_matrix_ovr_mlp, axis=1)).T, 2))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
forest = RandomForestClassifier(random_state=0).fit(x_train, y_train)
y_pred = forest.predict(x_test)
print("forest: ", balanced_accuracy_score(y_test, y_pred))
feature_importances = forest.feature_importances_.tolist()
temp = reversed(sorted(feature_importances))
res = []
for ele in temp:
    res.append(feature_importances.index(ele))
k = 1
for idx in res:
    print(f"[{k}] {face_blendshapes_names[idx]} - {feature_importances[idx]}")
    k += 1
