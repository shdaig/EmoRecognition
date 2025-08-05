import mne
import numpy as np
import scipy.signal as signal

import warnings

warnings.filterwarnings("ignore")


def _firwin_bandpass_filter(data, ntaps, lowcut, highcut, signal_freq, window='hamming'):
    taps = signal.firwin(ntaps, [lowcut, highcut], fs=signal_freq, pass_zero=False, window=window, scale=False)
    y = signal.lfilter(taps, 1.0, data)
    return y


def _fetch_channels(raw: mne.io.Raw):
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


# Расчет меток с процентом времени закрытых в эпохе
def _fetch_eyes_events(raw, epoch_size, step_seconds):
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
    epoch_samples = list(range(sfreq * epoch_size, filtered_f_fir.shape[0], sfreq * step_seconds))

    # Процент времени закрытых глаз в эпохе
    ec_epoch_percent = []

    detected_gz_indices_full_signal = _findpeaks(data=filtered_f_fir,
                                                 limit=threshold_gz,
                                                 spacing=int(sfreq * 0.33))
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
        if (event_positions[j + 1] - event_positions[j]) >= 15 * sfreq:
            is_sleep_states[event_positions[j]:event_positions[j + 1]] = 1
        if events_dict[event_positions[j + 1]] == 'go' and events_dict[event_positions[j]] == 'gz':
            is_sleep_states[event_positions[j]:event_positions[j + 1]] = 1

    for epoch_sample in epoch_samples:
        epoch_start = epoch_sample - sfreq * epoch_size
        epoch_end = epoch_sample
        ec_epoch_percent.append(np.sum(is_sleep_states[epoch_start:epoch_end]) / len(is_sleep_states[epoch_start:epoch_end]))
    ec_epoch_percent = np.array(ec_epoch_percent)

    x_counts = np.array(epoch_samples) // sfreq

    return x_counts, ec_epoch_percent


def get_labels(fname: str, window: int, shift: int):
    """
    :param fname: Путь к файлу .raw.fif.gz, содержащий запись ЭЭГ
    :param window: Окно анализа времени закрытых глаз в секундах
    :param shift: Шаг смещение окна в секундах
    :return: Отсчеты границ эпох анализа в секундах, метки [0..1]
    """
    raw = mne.io.read_raw_fif(fname, preload=True, verbose=False)
    raw = raw.pick('eeg', verbose=False)
    raw = raw.set_eeg_reference(ref_channels='average', verbose=False)

    x_counts, ec_epoch_percent = _fetch_eyes_events(raw, window, shift)

    return x_counts, ec_epoch_percent