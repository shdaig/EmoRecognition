import mne
import numpy as np
import scipy.signal as signal

import warnings
warnings.filterwarnings("ignore")


def _firwin_bandpass_filter(data, ntaps, lowcut, highcut, signal_freq, window='hamming'):
    taps = signal.firwin(ntaps, [lowcut, highcut], fs=signal_freq, pass_zero=False, window=window, scale=False)
    y = signal.lfilter(taps, 1.0, data)
    return y


def _fetch_channels(raw: mne.io.Raw): # -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param raw: Raw data from fif file
    :return: Time samples, channel names, channel data
    """
    return raw.times, np.array(raw.ch_names), raw.get_data()


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


# Расчет метрик связанных с морганиями

def _fetch_eyes_events(raw, epoch_size, step_seconds):
    features = {}
    
    sfreq = int(raw.info["sfreq"])
    _, channel_names, data = _fetch_channels(raw)
    f_avg = None
    if "Fp1" in channel_names:
        fp1, fp2 = data[channel_names == "Fp1"][0], data[channel_names == "Fp2"][0]
        f_avg = np.expand_dims((fp1 + fp2) / 2, axis=0)
    else:
        f3, f4 = data[channel_names == "F3"][0], data[channel_names == "F4"][0]
        f_avg = np.expand_dims(f3, axis=0)
    f_avg = -f_avg
    ntaps = sfreq * 10
    if sfreq == 250:
        ntaps *= 2
    # Фильтрация
    filtered_f_fir = _firwin_bandpass_filter(f_avg, ntaps=ntaps,
                                            lowcut=0.1,
                                            highcut=4,
                                            signal_freq=sfreq)
    filtered_f_fir = np.concatenate((filtered_f_fir[:, ntaps // 2:], filtered_f_fir[:, :ntaps // 2]), axis=1)
    filtered_f_fir = filtered_f_fir.reshape((filtered_f_fir.shape[1],))

    events, events_id = mne.events_from_annotations(raw, verbose=0)
    try:
        gz_idxs = np.sort(events[events[:, 2] == events_id['GZ']][:, 0])
    except Exception:
        gz_idxs = [sfreq * 30]
    
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
    
    # Нарезка сигнада на эпохи
    epoch_samples = list(range(sfreq * epoch_size, filtered_f_fir.shape[0], sfreq * step_seconds))[1:]
    
    # Частоты появления пиков ГО и ГЗ, средняя ширина ГЗ в эпохе
    freqs_gz = []
    freqs_go = []
    mean_gz_amp = []
    max_gz_amp = []
    mean_go_amp = []
    max_go_amp = []
    mean_gz_width_filtered = []
    mean_gz_width_unfiltered = []
    median_gz_width_filtered = []
    median_gz_width_unfiltered = []
    max_gz_width_filtered = []
    max_gz_width_unfiltered = []
    
    print(f"threshold_gz: {threshold_gz}")
    print(f"threshold_go: {threshold_go}")
    
    unfiltered_f_avg = f_avg.reshape((f_avg.shape[1],))
    for epoch_sample in epoch_samples:
        detected_gz_indices = _findpeaks(data=filtered_f_fir[epoch_sample - sfreq * epoch_size:epoch_sample],
                                         limit=threshold_gz,
                                         spacing=int(sfreq * 0.33))
        detected_go_indices = _findpeaks(data=filtered_f_fir_go[epoch_sample - sfreq * epoch_size:epoch_sample],
                                     limit=threshold_go,
                                     spacing=int(sfreq * 0.33))
        freqs_gz.append(len(detected_gz_indices))
        freqs_go.append(len(detected_go_indices))
        
        # высота пика
        if len(detected_gz_indices) != 0:
            mean_gz_amp.append(np.mean(filtered_f_fir[epoch_sample - sfreq * epoch_size:epoch_sample][detected_gz_indices]))
            max_gz_amp.append(np.max(filtered_f_fir[epoch_sample - sfreq * epoch_size:epoch_sample][detected_gz_indices]))
        else:
            mean_gz_amp.append(0.)
            max_gz_amp.append(0.)
            
        if len(detected_go_indices) != 0:
            mean_go_amp.append(np.mean(filtered_f_fir_go[epoch_sample - sfreq * epoch_size:epoch_sample][detected_go_indices]))
            max_go_amp.append(np.max(filtered_f_fir_go[epoch_sample - sfreq * epoch_size:epoch_sample][detected_go_indices]))
        else:
            mean_go_amp.append(0.)
            max_go_amp.append(0.)
        
        # ширина пика в фильтрованом сигнале
        filtered_segment = filtered_f_fir[epoch_sample - sfreq * epoch_size:epoch_sample]
        gz_event_len = 0
        gz_events = []
        flag_gz_event_start = False
        for j in range(len(filtered_segment)):
            if filtered_segment[j] >= threshold_gz:
                if flag_gz_event_start:
                    gz_event_len += 1
                else:
                    flag_gz_event_start = True
                    gz_event_len += 1
            else:
                if flag_gz_event_start:
                    flag_gz_event_start = False
                    gz_events.append(gz_event_len)
                    gz_event_len = 0
        if len(gz_events) != 0:
            mean_gz_width_filtered.append(np.mean(gz_events))
            median_gz_width_filtered.append(np.median(gz_events))
            max_gz_width_filtered.append(np.max(gz_events))
        else:
            mean_gz_width_filtered.append(int(sfreq * 0.6))
            median_gz_width_filtered.append(int(sfreq * 0.6))
            max_gz_width_filtered.append(int(sfreq * 0.6))
        
        unfiltered_segment = unfiltered_f_avg[epoch_sample - sfreq * epoch_size:epoch_sample]
        gz_event_len = 0
        gz_events = []
        for idx in detected_gz_indices:
            i = idx
            while i < len(unfiltered_segment) and unfiltered_segment[i] >= threshold_gz:
                gz_event_len += 1
                i += 1
            i = idx - 1
            while  i >= 0 and unfiltered_segment[i] >= threshold_gz:
                gz_event_len += 1
                i -= 1
            gz_events.append(gz_event_len)
            gz_event_len = 0
        if len(gz_events) != 0:
            mean_gz_width_unfiltered.append(np.mean(gz_events))
            median_gz_width_unfiltered.append(np.median(gz_events))
            max_gz_width_unfiltered.append(np.max(gz_events))
        else:
            mean_gz_width_unfiltered.append(int(sfreq * 0.6))
            median_gz_width_unfiltered.append(int(sfreq * 0.6))
            max_gz_width_unfiltered.append(int(sfreq * 0.6))
            
    features["freqs_gz"] = np.array(freqs_gz)
    features["freqs_go"] = np.array(freqs_go)
    features["mean_gz_amp"] = np.array(mean_gz_amp)
    features["max_gz_amp"] = np.array(max_gz_amp)
    features["mean_go_amp"] = np.array(mean_go_amp)
    features["max_go_amp"] = np.array(max_go_amp)
    features["mean_gz_width_filtered"] = np.array(mean_gz_width_filtered)
    features["mean_gz_width_unfiltered"] = np.array(mean_gz_width_unfiltered)
    features["median_gz_width_filtered"] = np.array(median_gz_width_filtered)
    features["median_gz_width_unfiltered"] = np.array(median_gz_width_unfiltered)
    features["max_gz_width_filtered"] = np.array(max_gz_width_filtered)
    features["max_gz_width_unfiltered"] = np.array(max_gz_width_unfiltered)
    
    # Среднее расстояние между пиками ГЗ в эпохе
    # Максимальное растоение между пиками ГЗ в эпохе
    mean_intervals_gz = []
    median_intervals_gz = []
    max_intervals_gz = []
    detected_gz_indices_full_signal = _findpeaks(data=filtered_f_fir,
                                                 limit=threshold_gz,
                                                 spacing=int(sfreq * 0.33))
    intervals_gz = detected_gz_indices_full_signal[1:] - detected_gz_indices_full_signal[:-1]
    indexs_intervals = detected_gz_indices_full_signal[1:]
    for epoch_sample in epoch_samples:
        epoch_start = epoch_sample - sfreq * epoch_size
        epoch_end = epoch_sample

        mask = (indexs_intervals >= epoch_start) & (indexs_intervals <= epoch_end)
        intervals_gz_epoch = intervals_gz[np.where(mask)]
        if len(intervals_gz_epoch): mean_intervals_gz.append(np.mean(intervals_gz_epoch))
        else: mean_intervals_gz.append(sfreq * epoch_size)   
        
        if len(intervals_gz_epoch): median_intervals_gz.append(np.median(intervals_gz_epoch))
        else: median_intervals_gz.append(sfreq * epoch_size)
        
        if len(intervals_gz_epoch): max_intervals_gz.append(np.max(intervals_gz_epoch))
        else: max_intervals_gz.append(sfreq * epoch_size)
    features["mean_intervals_gz"] = np.array(mean_intervals_gz)
    features["median_intervals_gz"] = np.array(median_intervals_gz)
    features["max_intervals_gz"] = np.array(max_intervals_gz)
    
    # Среднее расстояние между пиками ГО в эпохе
    # Максимальное растоение между пиками ГО в эпохе
    mean_intervals_go = []
    median_intervals_go = []
    max_intervals_go = []
    detected_go_indices_full_signal = _findpeaks(data=filtered_f_fir_go,
                                                 limit=threshold_go,
                                                 spacing=int(sfreq * 0.33))
    intervals_go = detected_go_indices_full_signal[1:] - detected_go_indices_full_signal[:-1]
    indexs_intervals = detected_go_indices_full_signal[1:]
    for epoch_sample in epoch_samples:
        epoch_start = epoch_sample - sfreq * epoch_size
        epoch_end = epoch_sample

        mask = (indexs_intervals >= epoch_start) & (indexs_intervals <= epoch_end)
        intervals_go_epoch = intervals_go[np.where(mask)]
        if len(intervals_go_epoch): mean_intervals_go.append(np.mean(intervals_go_epoch))
        else: mean_intervals_go.append(sfreq * epoch_size)   
        
        if len(intervals_go_epoch): median_intervals_go.append(np.median(intervals_go_epoch))
        else: median_intervals_go.append(sfreq * epoch_size)
        
        if len(intervals_go_epoch): max_intervals_go.append(np.max(intervals_go_epoch))
        else: max_intervals_go.append(sfreq * epoch_size)
    features["mean_intervals_go"] = np.array(mean_intervals_go)
    features["median_intervals_go"] = np.array(median_intervals_go)
    features["max_intervals_go"] = np.array(max_intervals_go)
    
    
    # Врем удержания закрытых глаз
    mean_gzgo_intervals = []
    median_gzgo_intervals = []
    max_gzgo_intervals = []
    detected_go_indices_full_signal = _findpeaks(data=filtered_f_fir_go,
                                                 limit=threshold_go,
                                                 spacing=int(sfreq * 0.33))
    is_sleep_states = np.zeros(filtered_f_fir_go.shape)
    events_dict = {}
    for go_idx in detected_go_indices_full_signal:
        events_dict[go_idx] = 'go'
    for gz_idx in detected_gz_indices_full_signal:
        events_dict[gz_idx] = 'gz'
    events_dict[0] = 'start'
    events_dict = dict(sorted(events_dict.items()))
    event_positions = list(events_dict.keys())
    for j in range(len(event_positions) - 1):
        if (event_positions[j+1] - event_positions[j]) >= 15 * sfreq:
            is_sleep_states[event_positions[j]:event_positions[j+1]] = 1
        if events_dict[event_positions[j+1]] == 'go' and events_dict[event_positions[j]] == 'gz':
            is_sleep_states[event_positions[j]:event_positions[j+1]] = 1
    for epoch_sample in epoch_samples:
        epoch_start = epoch_sample - sfreq * epoch_size
        epoch_end = epoch_sample
        mean_gzgo_intervals.append(np.sum(is_sleep_states[epoch_start:epoch_end]) / sfreq)
    features["mean_gzgo_intervals"] = np.array(mean_gzgo_intervals)
    
    x_counts = list(range(epoch_size, filtered_f_fir.shape[0] // sfreq, step_seconds))
    # if len(x_counts) < len(features[list(features.keys())[0]]):
    #     for feature_array in features:
    #         features[feature_array] = features[feature_array][:len(x_counts)]
    # else:
    #     x_counts = x_counts[:len(features[list(features.keys())[0]])]
    
    return x_counts, features


def blink_features(fname: str, window: int, shift: int) -> dict:
    raw = mne.io.read_raw_fif(fname, preload=True, verbose=False)
    raw = raw.pick('eeg', verbose=False)
    raw = raw.set_eeg_reference(ref_channels='average', verbose=False)
    
    x_counts, features = _fetch_eyes_events(raw, window, shift)
    for key in features.keys():
        features[key] = np.append(features[key], features[key][-1])
    return features


def blink_features_counts(fname: str, window: int, shift: int): # -> tuple[list, dict]:
    raw = mne.io.read_raw_fif(fname, preload=True, verbose=False)
    raw = raw.pick('eeg', verbose=False)
    raw = raw.set_eeg_reference(ref_channels='average', verbose=False)
    
    x_counts, features = _fetch_eyes_events(raw, window, shift)
    for key in features.keys():
        features[key] = np.append(features[key], features[key][-1])
    return x_counts, features