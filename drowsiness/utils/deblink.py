import mne
from scipy.signal import butter, lfilter
import numpy as np
mne.set_log_level(False)


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y

def deblink(raw, state, data):
    # в конце вернет только каналы ЭЭГ, независимо от изначального набора каналов
    raw = raw.pick('eeg')
    channel_names, channel_data = np.array(raw.ch_names), raw.get_data()
    if data == 'tag' and state != 'М':
        channel_names = np.array([el.split('-')[0].split()[1] for el in channel_names]).flatten()
    if any((not ('Fp1' in channel_names), not ('Fp2' in channel_names))):
        print('Deblinking failed, absent Fp channels')
        return raw
    fp1, fp2 = channel_data[channel_names == "Fp1"], channel_data[channel_names == "Fp2"]
    # print(fp1.shape)
    # print(fp2.shape)
    # print(channel_names)
    fp_avg = np.clip((fp1 + fp2) / 2, -0.0002, 0.0002)
    fp_avg = bandpass_filter(fp_avg, lowcut=0.1, highcut=30.0, signal_freq=raw.info['sfreq'], filter_order=4)#.flatten()[None]
    # print(fp_avg.shape)
    
    full_raw = mne.io.RawArray(np.vstack((raw.get_data(), fp_avg)), mne.create_info(list(channel_names)+['EOG'], raw.info['sfreq'], ch_types=raw.get_channel_types()+['eog']))
    full_raw.set_montage(mne.channels.make_standard_montage("standard_1020"))
    full_raw.set_eeg_reference("average")
    eog_epochs = mne.preprocessing.create_eog_epochs(full_raw)
    eog_evoked = eog_epochs.average("all")
    model_evoked = mne.preprocessing.EOGRegression(picks="eeg", picks_artifact="eog").fit(eog_evoked)
    return model_evoked.apply(full_raw).pick('eeg')