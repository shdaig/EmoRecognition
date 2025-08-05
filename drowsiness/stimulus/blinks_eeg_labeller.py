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


def _firwin_bandpass_filter(data, ntaps, lowcut, highcut, signal_freq, window='hamming'):
    taps = signal.firwin(ntaps, [lowcut, highcut], fs=signal_freq, pass_zero=False, window=window, scale=False)
    y = signal.lfilter(taps, 1.0, data)
    return y


def _findpeaks(data, spacing=1, limit=None):
    len = data.size
    x = np.zeros(len + 2 * spacing)
    x[:spacing] = data[0] - 1.e-6
    x[-spacing:] = data[-1] - 1.e-6
    x[spacing:spacing + len] = data
    peak_candidate = np.zeros(len)
    peak_candidate[:] = True
    for s in range(spacing):
        start = spacing - s - 1
        h_b = x[start: start + len]  # before
        start = spacing
        h_c = x[start: start + len]  # central
        start = spacing + s + 1
        h_a = x[start: start + len]  # after
        peak_candidate = np.logical_and(peak_candidate, np.logical_and(h_c > h_b, h_c > h_a))

    ind = np.argwhere(peak_candidate)
    ind = ind.reshape(ind.size)
    if limit is not None:
        ind = ind[data[ind] > limit]
    return ind


def get_labels(fname, window=5, debug_labels_plot=False, debug_blinks_plots=False):
    raw = mne.io.read_raw_fif(fname, verbose=False)
    raw.load_data(verbose=False)
    raw = raw.set_eeg_reference(ref_channels='average', verbose=False)
    sfreq = int(raw.info['sfreq'])

    _, channel_names, data = eeg.fetch_channels(raw)

    if "Fp1" in channel_names:
        ch1, ch2 = data[channel_names == "Fp1"][0], data[channel_names == "Fp2"][0]
    else:
        ch1, ch2 = data[channel_names == "F3"][0], data[channel_names == "F4"][0]

    signal_argmax = np.argmax(np.vstack((np.abs(ch1), np.abs(ch2))), axis=0, keepdims=True)
    ch_max = np.take_along_axis(np.vstack((ch1, ch2)), signal_argmax, axis=0)

    ch_max = -ch_max
    ntaps = sfreq * 10
    if sfreq == 250:
        ntaps *= 2
    # Фильтрация
    filtered_f_fir = _firwin_bandpass_filter(ch_max, ntaps=ntaps,
                                             lowcut=0.1,
                                             highcut=4,
                                             signal_freq=sfreq)
    filtered_f_fir = np.concatenate((filtered_f_fir[:, ntaps // 2:], filtered_f_fir[:, :ntaps // 2]), axis=1)
    filtered_f_fir = filtered_f_fir.reshape((filtered_f_fir.shape[1],))

    events, events_id = mne.events_from_annotations(raw, verbose=0)
    gz_idxs = np.sort(events[events[:, 2] == events_id['GZ']][:, 0])

    # Расчет порогов для детектироваия открытия и закрытия глаз
    data_filt = filtered_f_fir[gz_idxs[0]:gz_idxs[0] + 30 * 4 * sfreq]
    Q1 = np.percentile(data_filt, 25)
    Q3 = np.percentile(data_filt, 75)
    IQR = Q3 - Q1
    threshold_gz = Q3 + 1.5 * IQR
    filtered_f_fir_go = -filtered_f_fir
    data_filt = filtered_f_fir_go[gz_idxs[0]:gz_idxs[0] + 30 * 4 * sfreq]
    Q1 = np.percentile(data_filt, 25)
    Q3 = np.percentile(data_filt, 75)
    IQR = Q3 - Q1
    threshold_go = Q3 + 1.5 * IQR

    filtered_f_fir = -filtered_f_fir
    threshold_gz = -threshold_gz

    plot_data = qp.qual_plot_data(fname, force=True)
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
    event_positions = list(events_dict.keys())

    state_dict = {}
    for event_position in events_dict:
        segment = filtered_f_fir[event_position - window * sfreq: event_position + window * sfreq]
        go_segment_idxs = _findpeaks(data=segment,
                                     limit=threshold_go,
                                     spacing=int(sfreq * 0.33))
        gz_segment_idxs = _findpeaks(data=-segment,
                                     limit=-threshold_gz,
                                     spacing=int(sfreq * 0.33))
        if len(go_segment_idxs) + len(gz_segment_idxs) != 0:
            if events_dict[event_position] == 'reaction':
                state_dict[event_position] = 2
            else:
                state_dict[event_position] = 1
        else:
            if events_dict[event_position] == 'reaction':
                state_dict[event_position] = 1
            else:
                state_dict[event_position] = 0
    state_dict[filtered_f_fir.shape[0]] = state_dict[event_positions[-1]]

    labels = []
    current_position = 0
    last_state = 2
    for event_position in state_dict:
        change_point = current_position + (event_position - current_position) // 2
        while current_position < change_point:
            labels.append(last_state)
            current_position += 1
        last_state = state_dict[event_position]
        while current_position < event_position:
            labels.append(last_state)
            current_position += 1
    labels = np.array(labels)
    x_labels = np.array(list(range(len(labels)))) / sfreq

    if debug_labels_plot:
        ax = qp.plot_qual(*plot_data, plot_IPE=False)
        ax.plot(x_labels, labels)
        plt.savefig("blinks_eeg_labels.png")
        plt.close()

    save_dir = "blinks_eeg_plots_around_stim"
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
            plt.hlines([threshold_gz], 0.0, segment.shape[0], colors='b')
            plt.hlines([threshold_go], 0.0, segment.shape[0], colors='b')

            go_segment_idxs = _findpeaks(data=segment[(20 - window) * sfreq: (20 + window) * sfreq],
                                         limit=threshold_go,
                                         spacing=int(sfreq * 0.33))
            gz_segment_idxs = _findpeaks(data=-segment[(20 - window) * sfreq: (20 + window) * sfreq],
                                         limit=-threshold_gz,
                                         spacing=int(sfreq * 0.33))
            go_segment_idxs_plt = go_segment_idxs + (20 - window) * sfreq
            gz_segment_idxs_plt = gz_segment_idxs + (20 - window) * sfreq
            plt.scatter(go_segment_idxs_plt, segment[go_segment_idxs_plt])
            plt.scatter(gz_segment_idxs_plt, segment[gz_segment_idxs_plt])

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
            plt.hlines([threshold_gz], 0.0, segment.shape[0], colors='b')
            plt.hlines([threshold_go], 0.0, segment.shape[0], colors='b')

            go_segment_idxs = _findpeaks(data=segment[(20 - window) * sfreq: (20 + window) * sfreq],
                                         limit=threshold_go,
                                         spacing=int(sfreq * 0.33))
            gz_segment_idxs = _findpeaks(data=-segment[(20 - window) * sfreq: (20 + window) * sfreq],
                                         limit=-threshold_gz,
                                         spacing=int(sfreq * 0.33))
            go_segment_idxs_plt = go_segment_idxs + (20 - window) * sfreq
            gz_segment_idxs_plt = gz_segment_idxs + (20 - window) * sfreq
            plt.scatter(go_segment_idxs_plt, segment[go_segment_idxs_plt])
            plt.scatter(gz_segment_idxs_plt, segment[gz_segment_idxs_plt])

            plt.savefig(f"{save_dir}/{stim_idx}_error.png")
            plt.close()
            print(f"{save_dir}/{stim_idx}_error.png")

    return labels
