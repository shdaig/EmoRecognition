import os.path
import shutil

import mne
import numpy as np
import drowsiness.eeg.eeg as eeg
import drowsiness.qualplot.qualplot as qp
import matplotlib.pyplot as plt
import scipy.signal as signal

import warnings
warnings.filterwarnings("ignore")


def get_labels(fname_fif, angles_data_fname, window=5, debug_blinks_plots=False):
    raw = mne.io.read_raw_fif(fname_fif, verbose=False)
    sfreq = int(raw.info['sfreq'])

    plot_data = qp.qual_plot_data(fname_fif, force=True)
    lags, lag_times, lags2, lag_times2, first_mark_time, react_range, q = plot_data
    reactions_idxs = (np.array(lag_times) - np.array(lags) + first_mark_time) * sfreq
    errors_idxs = (lag_times2 - lags2 + first_mark_time) * sfreq
    reactions_idxs = reactions_idxs.astype(np.integer)
    errors_idxs = errors_idxs.astype(np.integer)

    events_dict = {}
    for stim_idx in reactions_idxs:
        events_dict[stim_idx] = 'reaction'
    for stim_idx in errors_idxs:
        events_dict[stim_idx] = 'error'
    events_dict = dict(sorted(events_dict.items()))



    save_dir = "head_video_plots_around_stim"
    if debug_blinks_plots:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        for stim_idx in reactions_idxs:
            segment = filtered_f_fir[stim_idx - 20 * sfreq: stim_idx + 20 * sfreq]
            plt.plot(segment)
            ymin = np.min(segment) * 1.1
            ymax = np.max(segment) * 1.1
            plt.vlines([(20 - window) * sfreq], ymin, ymax, colors='g', linestyles="dashed")
            plt.vlines([20 * sfreq], ymin, ymax, colors='g')
            plt.vlines([(20 + window) * sfreq], ymin, ymax, colors='g', linestyles="dashed")
            plt.savefig(f"{save_dir}/{stim_idx}_reaction.png")
            plt.close()
            print(f"{save_dir}/{stim_idx}_reaction.png")

        for stim_idx in errors_idxs:
            segment = filtered_f_fir[stim_idx - 20 * sfreq: stim_idx + 20 * sfreq]
            plt.plot(segment)
            ymin = np.min(segment) * 1.1
            ymax = np.max(segment) * 1.1
            plt.vlines([(20 - window) * sfreq], ymin, ymax, colors='r', linestyles="dashed")
            plt.vlines([20 * sfreq], ymin, ymax, colors='r')
            plt.vlines([(20 + window) * sfreq], ymin, ymax, colors='r', linestyles="dashed")
            plt.savefig(f"{save_dir}/{stim_idx}_error.png")
            plt.close()
            print(f"{save_dir}/{stim_idx}_error.png")

    return labels