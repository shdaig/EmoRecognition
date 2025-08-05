from drowsiness.utils.features_x import provide_specs, feature_set
from drowsiness.utils.utilss import qual_plot_data, plot_qual
import os
from glob import glob
from drowsiness.eeg.vlad_load_data import *
from drowsiness.utils.feature_extract import ECGFeature
from drowsiness.features_extraction.blink_features_extraction import blink_features
import statsmodels.api as sm
import math
import matplotlib.pyplot as plt

def replace_outliers(data, window_size=5, threshold=1):
    rolling_df = pd.Series(data).rolling(window=window_size, center=True)

    q1 = rolling_df.quantile(0.25)
    q3 = rolling_df.quantile(0.75)
    
    iqr = q3 - q1
    
    outliers = (data - rolling_df.median()).abs() > threshold * iqr
    
    clean_data = data.copy()
    clean_data[outliers] = rolling_df.median()[outliers]
    
    return clean_data

def plot_feats(feat, label_feat, errors, reactions, std_reac, seconds, r2_threshold, plot_r2_bar, plot_r2_exp, behavioral_lbls):
    errors = np.array(errors)[~np.isnan(feat).reshape(-1)]
    reactions = np.array(reactions)[~np.isnan(feat).reshape(-1)]
    std_reac = np.array(std_reac)[~np.isnan(feat).reshape(-1)]
    seconds = np.array(seconds)[~np.isnan(feat).reshape(-1)]
    feat = feat[~np.isnan(feat).reshape(-1)]

    errors = errors[~np.isinf(feat).reshape(-1)]
    reactions = reactions[~np.isinf(feat).reshape(-1)]
    std_reac = std_reac[~np.isinf(feat).reshape(-1)]
    feat = feat[~np.isinf(feat).reshape(-1)]

    def plot_feat(feat1, feat2, label_feat1, label_feat2, model):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 3), gridspec_kw={'width_ratios': [3, 1]})
        ax1.set_title(f'{label_feat1} vs {label_feat2}', fontsize=15)
        ln1 = ax1.plot(seconds, feat1, label=label_feat1)
        ax1.set_xlabel('Time, sec')
        ax1.set_ylabel(label_feat1)
        ax1 = ax1.twinx()
        ln2 = ax1.plot(seconds, feat2, c='g', label=label_feat2)
        ax1.set_ylabel(label_feat2)

        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs)

        ax2.scatter(feat1, feat2, s=5)
        ax2.plot([min(feat1), max(feat1)], [model.predict([1, min(feat_df[:, 1])])[0], model.predict([1, max(feat_df[:, 1])])[0]], c='r')
        ax2.set_title(f'coef: {round(model.params[1], 2)}, p-value: {round(model.pvalues[1], 3)}, R2: {round(model.rsquared, 3)}')
        ax2.set_xlabel(label_feat1)
        ax2.set_ylabel(label_feat2)
        plt.show()

    feat_df = sm.add_constant(feat)
    model_err = sm.OLS(errors, feat_df).fit()
    model_reac = sm.OLS(reactions, feat_df).fit()
    model_std = sm.OLS(std_reac, feat_df).fit()

    if not plot_r2_bar and not plot_r2_exp:
        if model_err.rsquared >= r2_threshold and 'num errors' in behavioral_lbls:
            plot_feat(feat, errors, label_feat, 'num errors', model_err)
        if model_reac.rsquared >= r2_threshold and 'ave reactions' in behavioral_lbls:
            plot_feat(feat, reactions, label_feat, 'ave reactions', model_reac)
        if model_std.rsquared >= r2_threshold and 'std reactions' in behavioral_lbls:
            plot_feat(feat, std_reac, label_feat, 'std reactions', model_std)

    return model_err.rsquared, model_reac.rsquared, model_std.rsquared

def get_paths(required_subjs):
    global root
    root = r'/home/neuron/mnt/a/A/12proj_sorted'
    fnames = []

    for subj, dates in required_subjs.items():
        if len(dates) == 0:
            dates = list(map(int, os.listdir(os.path.join(root, subj))))
        for date in dates:
            fifs = glob(os.path.join(root, subj, str(date), '*.raw.fif.gz'))
            if len(fifs) != 0: fnames.append(fifs[0])
            else:
                try:
                    fifs = glob(os.path.join(root, subj, str(date), '1', '*.raw.fif.gz'))
                    fnames.append(fifs[0])
                except Exception: None
                try:
                    fifs = glob(os.path.join(root, subj, str(date), '2', '*.raw.fif.gz'))
                    fnames.append(fifs[0])
                except Exception: None
                try:
                    fifs = glob(os.path.join(root, subj, str(date), '3', '*.raw.fif.gz'))
                    fnames.append(fifs[0])
                except Exception: None
    return fnames

def r2_bar_plot(r_squared, feat_names):
    max_r2 = max(max(inner_list) for sublist in r_squared for inner_list in sublist)
    min_r2 = min(min(inner_list) for sublist in r_squared for inner_list in sublist)
    if len(feat_names) <= 4: num_cols = len(feat_names)
    else: num_cols = 4
    num_rows = math.ceil(len(feat_names)/4)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 3*num_rows))
    rows, cols = np.unravel_index(range(num_cols*num_rows), (num_rows, num_cols))
    for idx, title in enumerate(feat_names):
        if num_rows > 1: axs_idx = (rows[idx], cols[idx])
        else: axs_idx = idx
        if len(feat_names) != 1:
            # axs[axs_idx].hist([r_squared[idx][0], r_squared[idx][1], r_squared[idx][2]], range=(min_r2, max_r2), label = ['errors', 'reaction', 'std_reaction'])
            axs[axs_idx].hist([r_squared[idx][0]], range=(min_r2, max_r2), label = ['errors', 'reaction', 'std_reaction'])
            axs[axs_idx].set_title(title)
            axs[axs_idx].set_xlabel('R-squared')
            axs[axs_idx].set_xlim(-0.01, max_r2+0.01)
            axs[axs_idx].legend()
        else:
            # axs.hist([r_squared[idx][0], r_squared[idx][1], r_squared[idx][2]], range=(min_r2, max_r2), label = ['errors', 'reaction', 'std_reaction'])
            axs.hist([r_squared[idx][0]], range=(min_r2, max_r2), label = ['errors', 'reaction', 'std_reaction'])
            axs.set_title(title)
            axs.set_xlabel('R-squared')
            axs.set_xlim(-0.01, max_r2+0.01)
            axs.legend()
    plt.subplots_adjust(hspace=0.4)
    plt.show()

def r2_experements_plot(r_squared, experiment_lbls, metrics_lbls, behavioral_lbls):
    x = np.arange(len(experiment_lbls))
    width = 0.25
    
    fig, ax = plt.subplots(nrows=r_squared.shape[0], layout='constrained', figsize=(17, 2 * r_squared.shape[0]))
    
    for i in range(r_squared.shape[0]):
        multiplier = 0
        for j in range(len(behavioral_lbls)):
            offset = width * multiplier
            rects = ax[i].bar(x + offset, r_squared[i, j], width, label=behavioral_lbls[j])
            multiplier += 1
        ax[i].set_ylabel(metrics_lbls[i])
        ax[i].grid(True)
        if (i+1) % 3 == 0: ax[i].set_xticks(x + width, experiment_lbls, rotation=90)
        elif i+1 == r_squared.shape[0]: ax[i].set_xticks(x + width, experiment_lbls, rotation=90)
        else: ax[i].set_xticks([])
        ax[i].legend(fontsize="8", loc ="upper right")
        ax[i].set_ylim(0, 1)
    plt.show()

def eeg_features(fname, window, shift):
    try:
        specs, epoch_sec = provide_specs(fname, window=window, shift=shift, shortfeats=True)
        eeg_feats_df = feature_set(specs, add_new=True)

        epoch_sec = epoch_sec[:, -1] # np.mean(epoch_sec, axis=1)
        feat1 = (specs['O1']['alpha']/specs['O1']['beta2'] + specs['O2']['alpha']/specs['O2']['beta2'])/2
        feat2O1 = (specs['O1']['delta'] + specs['O1']['theta'])/(specs['O1']['alpha']+specs['O1']['beta'])
        feat2O2 = (specs['O2']['delta'] + specs['O2']['theta'])/(specs['O2']['alpha']+specs['O2']['beta'])
        feat2 = (feat2O1 + feat2O2)/2
        feat3 = (specs['O1']['alpha_dom'] + specs['O2']['alpha_dom'])/2
        eeg_feats_df['α/β'] = feat1
        eeg_feats_df['(δ+θ)/(α+β)'] = feat2
        eeg_feats_df['mod. α freq.'] = feat3
    except BaseException as e:
        print(str(e))
    return eeg_feats_df

def plot_correlate(required_subjs: dict, feat_names: dict, behavioral_lbls: list = ['num errors', 'ave reactions', 'std reactions'],
                   r2_threshold: float = 0, window: int = 180, shift: int = 30, plot_r2_bar: bool = False, plot_r2_exp: bool = False):
    fnames = get_paths(required_subjs)
    r_squared = np.zeros((sum([len(i) for i in feat_names.values()]), 3, len(fnames)))
    for idx_fname, fname in enumerate(fnames):
        try:
            if not plot_r2_bar and not plot_r2_exp: print(f'\n{os.path.dirname(fname)[len(root)+1:]}')
            ecg_channel = True
            
            if 'EEG' in feat_names.keys():
                eeg_feats_df = eeg_features(fname, window, shift)

            if 'Blink' in feat_names.keys():
                blink_feats_dict = blink_features(fname, window, shift)
            
            name = [os.path.dirname(fname)[len(root)+1:].split('/')[0]]
            date = [os.path.dirname(fname)[len(root)+1:].split('/')[1]]
            channel_names = ['ECG']
            label_names = ['all']
            info_fif = load_12proj_data(root, name, date, channel_names, label_names)

            plot_data, sfreq = qual_plot_data(fname, force=True, window=window, shift=shift) #
            sfreq = int(sfreq) 

            errors_list, reactions_list, std_reac_list = [], [], []
            if 'ECG' in info_fif[name[0]][date[0]][1]:
                raw_ecg = info_fif[name[0]][date[0]][0][0]
                errors = info_fif[name[0]][date[0]][0][1]
                reactions = info_fif[name[0]][date[0]][0][2]
            else:
                if not plot_r2_bar and not plot_r2_exp: print('Отсутствует ЭКГ канал')
                ecg_channel = False
                errors = info_fif[name[0]][date[0]][0][0]
                reactions = info_fif[name[0]][date[0]][0][1]

            seconds = []
            sliced_ecg = None
            for idx in range(0, len(errors), shift*sfreq):
                if idx + window*sfreq <= len(errors)-1:
                    if ecg_channel:
                        sliced = raw_ecg[idx:idx + window*sfreq]
                        if sliced_ecg is None: sliced_ecg = np.array([sliced])
                        else: sliced_ecg = np.dstack((sliced_ecg, [sliced]))
                    
                    seconds.append(idx/sfreq + window/2)
                    errors_list.append(sum(errors[idx:idx + window*sfreq]))
                    reactions_list.append(np.mean(reactions[idx:idx + window*sfreq]))
                    std_reac_list.append(np.std(reactions[idx:idx + window*sfreq]))

            feat_idx = 0
            if ecg_channel and 'ECG' in feat_names.keys():
                sliced_ecg = np.transpose(sliced_ecg, (2, 0, 1))
                ecg_feats = ECGFeature(name=feat_names['ECG'], sample_rate=sfreq).transform(sliced_ecg)

                for feat_idx_ecg in range(len(feat_names['ECG'])):
                    a, b, c = plot_feats(ecg_feats[:, feat_idx_ecg], feat_names['ECG'][feat_idx_ecg],
                                        errors_list, reactions_list, std_reac_list, seconds, r2_threshold, plot_r2_bar, plot_r2_exp, behavioral_lbls)
                    for i, r2 in enumerate([a, b, c]):
                        r_squared[feat_idx_ecg, i, idx_fname] = r2
                feat_idx = feat_idx_ecg+1

            if 'EEG' in feat_names.keys():
                for feat_idx_eeg, eeg_feat_name in enumerate(feat_names['EEG']):
                    feat_idx_eeg += feat_idx
                    feat = eeg_feats_df[eeg_feat_name].to_numpy()
                    if len(feat) > len(errors_list): feat = feat[:len(errors_list)]
                    elif len(feat) < len(errors_list):
                        errors_list = errors_list[:len(feat)]
                        reactions_list = reactions_list[:len(feat)]
                        std_reac_list = std_reac_list[:len(feat)]
                        seconds = seconds[:len(feat)]
                    try:
                        a, b, c = plot_feats(feat, eeg_feat_name, errors_list,
                                            reactions_list, std_reac_list, seconds, r2_threshold, plot_r2_bar, plot_r2_exp, behavioral_lbls)
                        for i, r2 in enumerate([a, b, c]):
                            r_squared[feat_idx_eeg, i, idx_fname] = r2
                    except Exception: None
                feat_idx = feat_idx_eeg+1

            if 'Blink' in feat_names.keys():
                for feat_idx_blink, blink_feat_name in enumerate(feat_names['Blink']):
                    feat_idx_blink += feat_idx
                    feat = blink_feats_dict[blink_feat_name]
                    if len(feat) > len(errors_list): feat = feat[:len(errors_list)]
                    elif len(feat) < len(errors_list):
                        errors_list = errors_list[:len(feat)]
                        reactions_list = reactions_list[:len(feat)]
                        std_reac_list = std_reac_list[:len(feat)]
                        seconds = seconds[:len(feat)]
                    try:
                        a, b, c = plot_feats(feat, blink_feat_name, errors_list,
                                            reactions_list, std_reac_list, seconds, r2_threshold, plot_r2_bar, plot_r2_exp, behavioral_lbls)
                        for i, r2 in enumerate([a, b, c]):
                            r_squared[feat_idx_blink, i, idx_fname] = r2
                    except Exception: None
        except Exception: None
    
    if plot_r2_bar:
        all_feat_names = []
        for i in feat_names.values(): all_feat_names.extend(i)
        r2_bar_plot(r_squared, all_feat_names)
    
    
    if plot_r2_exp:
        experiment_lbls = []
        for subj in fnames:
            root_elems = subj.split('/')
            if root_elems[-2] in ['1', '2', '3']:
                experiment_lbls.append(f"{root_elems[-4]}_{root_elems[-3]}_{root_elems[-2]}")
            else:
                experiment_lbls.append(f"{root_elems[-3]}_{root_elems[-2]}")
        
        all_feat_names = []
        for i in feat_names.values(): all_feat_names.extend(i) 
        r2_experements_plot(r_squared, experiment_lbls, all_feat_names, ['err', 'reac', 'std'])
