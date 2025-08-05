import numpy as np
import pandas as pd

from drowsiness.emo import dfeats_utils as dfu

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

face_blendshapes_names = dfu.get_features_names()

blendshapes_df = pd.read_csv("../stage_1_feature_extraction/stage_1_results/dfeats_diff.csv")

y = blendshapes_df["label"].to_numpy()
blendshapes_df = blendshapes_df.drop(["filename", "label"], axis=1)

label, cnt = np.unique(y, return_counts=True)
print(f"neutral: {label[0]} - {cnt[0]} [{round(cnt[0] / np.sum(cnt) * 100, 2)} %]")
print(f"smile: {label[1]} - {cnt[1]} [{round(cnt[1] / np.sum(cnt) * 100, 2)} %]")
print(f"disgust: {label[2]} - {cnt[2]} [{round(cnt[2] / np.sum(cnt) * 100, 2)} %]")

x = blendshapes_df.to_numpy()

for i in range(x.shape[1]):
    x_feature = blendshapes_df.to_numpy()[:, i]

    plt.figure(figsize=(10, 6))
    n, b, p = plt.hist(x_feature, bins=30, alpha=0.0, color='w')
    plt.hist(x_feature[y == 0], bins=b, alpha=0.5, label='neutral')
    plt.hist(x_feature[y == 1], bins=b, alpha=0.5, label='smile')
    plt.hist(x_feature[y == 2], bins=b, alpha=0.5, label='disgust')

    plt.xlabel('value')
    plt.ylabel('freq')
    plt.title(f'{face_blendshapes_names[i]}')
    plt.legend()

    plt.show()
