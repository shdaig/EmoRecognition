import os
import glob
import drowsiness.eeg.eeg as eeg
from drowsiness.statesutils.preprocessing.label_encoder import StatesLabelEncoder
import drowsiness.qualplot.qualplot as qp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mne
import warnings
warnings.filterwarnings("ignore")

subjs_dates = {
    "subj_name": ["20240101"]
}


for subj in subjs_dates:
    for date in subjs_dates[subj]:
        print(f"{subj}_{date}")
        fname_fif = glob.glob(f"/home/neuron/mnt/a/A/12proj_sorted/{subj}/{date}/*.raw.fif.gz")[0]

        raw = mne.io.read_raw_fif(fname_fif, verbose=False)
        sle = StatesLabelEncoder()
        q_discrete = sle.get_hard_sleep_state(raw, 3, 2)
        sfreq = raw.info["sfreq"]
        # q_discrete[np.where(q_discrete == -1)[0]] = -2
        # state_idx = np.where(np.diff(q_discrete) == -1)[0]
        x_state = np.arange(len(q_discrete)) / sfreq
        q_discrete *= 30

        plot_data = qp.qual_plot_data(fname_fif, force=True)
        ax = qp.plot_qual(*plot_data, plot_IPE=False)
        df_angles = pd.read_csv(f"../angles_dataset_10_frames/{subj}_{date}.csv")
        df_angles = df_angles.fillna(-30.)
        k = 1
        x = df_angles["timestamp"].to_numpy()[::k] / 1000
        target_name = "pitch"
        # target_name = "roll"
        # target_name = "yaw"
        target = df_angles[target_name].to_numpy()[::k]

        window = 180
        shift = 30

        end_time = x[-1]

        segment_end_time = window

        mean_pitch = []
        while segment_end_time <= end_time:
            mean_pitch.append(np.mean(target[np.all([x >= segment_end_time - window, x <= segment_end_time], axis=0)]))
            segment_end_time += shift
        mean_pitch = np.array(mean_pitch)
        print(mean_pitch.shape[0])
        x_counts = list(range(window, window + mean_pitch.shape[0] * shift, shift))

        # work_start_idx = np.argwhere(q_discrete == 30)[0][0]
        # work_start_time = x_state[work_start_idx]
        # start_x = np.argwhere(x >= work_start_time)[0][0]
        # target_mean = np.mean(target[start_x:start_x + (1800 // k)])
        # print(f"{start_x}:{start_x + (1800 // k)} - {target_mean}")
        # print()
        # target -= target_mean
        # target[target > 30.] = 30.
        # target[target < -30.] = -30.

        # print(roll)
        # ax.plot(x, roll, label="roll")
        # ax.plot(x, yaw, label="yaw")
        ax.plot(x, target, label=f"{target_name} raw")
        x_line = np.arange(len(q_discrete)) / sfreq
        line = np.zeros((len(q_discrete),))
        ax.plot(x_counts, mean_pitch, label="mean_pitch")
        plt.legend()
        plt.show()
        # plt.savefig(f"if_mean_{target_name}_nanfilled/{subj}_{date}.png")
        # plt.close()
