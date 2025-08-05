import glob

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data_file_path = "../blendshapes/20230731_by2/blendshapes.csv"

df = pd.read_csv(data_file_path)
df = df.drop(["time(s.)"], axis=1)

dict_df = df.to_dict(orient='list')
feats_names = list(dict_df.keys())
print(feats_names)

feat_data = []
for key in dict_df:
    feat_data.append(dict_df[key])
feat_data = np.array(feat_data)
print(feat_data.shape)

mean_data = np.mean(feat_data, axis=1)
std_data = np.std(feat_data, axis=1)
print(mean_data.shape)
print(std_data.shape)

x_axis = list(range(52))

plt.figure(figsize=(17, 9))
plt.bar(x_axis, mean_data)
plt.errorbar(x_axis, mean_data, yerr=std_data, fmt="o", color="r")
plt.xticks(x_axis, feats_names, rotation=90)
plt.grid(True)
plt.tight_layout()
plt.show()
