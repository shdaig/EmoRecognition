import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

import os
import glob
import warnings
warnings.filterwarnings('ignore')


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


face_blendshapes_names = ['_neutral',
                          'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
                          'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
                          'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
                          'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
                          'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
                          'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
                          'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight',
                          'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft',
                          'mouthLowerDownLeft', 'mouthLowerDownRight',
                          'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
                          'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
                          'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
                          'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']

x = []
y_raw = []

labels_paths = glob.glob("/home/neuron/mnt/a/A/arhiv/emoscroll/*/face_labels.csv")
labels_paths.remove("/home/neuron/mnt/a/A/arhiv/emoscroll/20240411_am54/face_labels.csv")

# Извлекаем векторы blendshape'ов в X в соответствии с временем меток классов
for labels_path in labels_paths:
    labels_splited = labels_path.split('/')
    blendshapes_path = os.path.join("blendshapes", labels_splited[-2], "blendshapes.csv")

    try:
        face_labels_df = pd.read_csv(labels_path, header=None, dtype=str)
        blendshapes_df = pd.read_csv(blendshapes_path)
        timestamp_seconds = []

        for index, row in face_labels_df.iterrows():
            timest = row[0]
            timest_seconds = int(timest.split('.')[0]) * 60 + int(timest.split('.')[1])
            timestamp_seconds.append(timest_seconds)
            timestamp_seconds.append(timest_seconds + 0.5)
            label = int(row[1])
            y_raw.append(label)
            y_raw.append(label)
        blendshapes_timestamps = blendshapes_df["time(s.)"].to_numpy()
        blendshapes_features = blendshapes_df.drop("time(s.)", axis=1)
        for timest in timestamp_seconds:
            bs_idx = find_nearest_idx(blendshapes_timestamps, timest)
            x.append(blendshapes_features.iloc[bs_idx].to_numpy())
    except Exception:
        pass

x = np.array(x)
y_raw = np.array(y_raw)

# Приравниваем эмоции большой и малой степени.
y_raw[y_raw == -2] = -1
y_raw[y_raw == 2] = 1

le = LabelEncoder().fit(y_raw)
y = le.transform(y_raw)

param_grid = {'C': [1e-3, 1e-2, 1.0, 10, 100]}

for grd in ParameterGrid(param_grid):
    logreg_score = []
    mlp_score = []
    xgb_score = []

    skf = StratifiedKFold(n_splits=4)
    for i, (train_index, test_index) in enumerate(skf.split(x, y)):
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        logreg = LogisticRegression(random_state=0, class_weight='balanced', C=grd['C']).fit(x_train, y_train)
        y_pred = logreg.predict(x_test)
        logreg_score.append(balanced_accuracy_score(y_test, y_pred))

    print()
    print("logreg: ", np.mean(logreg_score))
