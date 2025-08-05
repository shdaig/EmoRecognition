import numpy as np
import mne
from drowsiness.utils.scp_sig_custom import welch, coherence #
import scipy.signal as ss
from scipy.stats import zscore
from itertools import combinations, product
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.simplefilter('ignore')
mne.set_log_level(False)


def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order, mode='highpass'):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq if mode=='bandpass' else None
    b, a = ss.butter(filter_order, low if mode=='highpass' else (low, high), btype=mode)
    y = ss.filtfilt(b, a, data)
    return y

def overlap_epochs(data, ep_len_sec, ep_shift_sec, sfreq):
    ep_len_point, ep_shift_point = int(sfreq * ep_len_sec), int(sfreq * ep_shift_sec)
    start = 0
    eps = []
    times = []
    while start < (len(data) - ep_len_point):
        eps.append(data[start:start+ep_len_point].T[None, :, :])
        times.append((start/sfreq, (start+ep_len_point)/sfreq))
        start += ep_shift_point
    return np.vstack(eps), np.array(times)

def corr_feat(data):
    ms = [np.triu((np.corrcoef(stim)), 1) for stim in data]
    return np.array([m[m != 0] for m in ms])

extractor_dom = lambda x: x[1][np.argmax(x[0], axis=-1).flatten()].reshape((-1, 1))
extractor_spec = lambda x: np.nansum(x[0], axis=-1).reshape((-1, 1)) # 

BANDS = {v: k for k, v in [[[1, 4], 'delta'],
                             [[4, 8], 'theta'],
                             [[8, 13], 'alpha'],
                             [[13, 30], 'beta'],
                             [[20, 30], 'beta2'],
                          [[30, 49], 'gamma1'],
                           [[1, 70], 'full']
                          ]}
THR_DICT = {'F3': 150,
             'F4': 150,
             'O1': 150,
             'O2': 150,
             'Fp1': 150,
             'Fp2': 150,
             'C3': 90,
             'C4': 90,
             'P3': 85,
             'P4': 85,
             'T3': 100,
             'T4': 100,
             'Pz': 120,
             'Cz': 120,
             'Aux': np.inf}
LEFT = ['Fp1', 'F3', 'O1', 'C3', 'P3', 'T3']
RIGHT = ['Fp2', 'F4', 'O2', 'C4', 'P4', 'T4']
def coh_combs(ch_names):
    if any(['Ch' in chn for chn in ch_names]): 
        return list(combinations(list(range(n_ch)), 2))
    left = LEFT
    right = RIGHT
    fore = ['Fp1', 'Fp2', 'F3', 'F4', 'T3', 'T4', 'C3', 'C4']
    hind = ['T3', 'T4', 'P3', 'P4', 'O1', 'O2']
    
    symm = list(zip(left, right))
    f_h = list(product(['F3', 'F4'], ['P3', 'P4', 'O1', 'O2']))
    
    combs = symm + f_h
    for lst in (left, right, fore, hind):
        combs.extend(combinations(lst, 2))
    combs = list(set(combs))
    
    res = []
    for ch1, ch2 in combs:
        if all((ch1 in ch_names, ch2 in ch_names)): res.append((ch1, ch2))
    # res = []
    # for el in [[(el, ch_names[j]) for j in range(i+1, len(ch_names))] for i, el in enumerate(ch_names[:-1])]:
    #     res.extend(el)
    return res

def provide_specs(fname='', data_dict=False, window=30, shift=30, shortfeats=False):
    """
    Если хочешь использовать уже нарезанные эпохи, задаешь fname='', а в data_dict
    передаешь словарь с ключами "epochs" (значение - 3D массив), "sfreq" (значение -
    частота дискретизации и опционально "ch_names" (значения - iterable с именами
    каналов, чтобы были понятные ключи в возвращаемом словаре res)
    """
    
    if len(fname) > 0:
        raw = mne.io.read_raw_fif(fname).pick(['O1', 'O2', 'F3', 'F4']) #'eeg' 
        raw.load_data()
        raw = raw.filter(.1, None, method='iir')
        sfreq = raw.info['sfreq']
        eeg = raw.get_data()
        ch_names = raw.ch_names
        epochs, epoch_sec = overlap_epochs(eeg.T, window, shift, sfreq)
    elif data_dict:
        epochs = data_dict['epochs'] # 3D array
        sfreq = data_dict['sfreq']
        ch_names = data_dict.get('ch_names', [f'Ch{i}' for i in range(epochs.shape[1])])
        epoch_sec = []
        epochs = np.apply_along_axis(bandpass_filter, -1, epochs, .1, None, sfreq, 4)
    else:
        print('No data to process')
        return {}, []
    
    mantiss = np.floor(np.log10(np.percentile(epochs[0], 95))) #eeg[:, :int(sfreq*60)]
    if mantiss < 0: epochs *= 1e6
    
    res = {ch: {} for ch in ch_names}
    
    # PSD
    thresholds = [THR_DICT.get(ch, np.inf) for ch in ch_names]
    epochs_bckup = np.array(epochs)
    freq, spec = welch(epochs, fs=sfreq, axis=-1, nperseg=1024, nfft=1024, average='median', thresholds=thresholds)
    epochs = np.array(epochs_bckup)
    
    for bn, band in BANDS.items():
        slc = (freq >= band[0]) & (freq <= band[1])
        for ch, chn in enumerate(ch_names):
            seq = extractor_spec((spec[:, ch, slc], freq[slc]))
            res[chn][bn] = seq
            
    # Alpha mod. freq.
    bn = 'alpha_dom'
    band = BANDS['alpha']
    slc = (freq >= band[0]) & (freq <= band[1])
    for ch, chn in enumerate(ch_names):
        seq = extractor_dom((spec[:, ch, slc], freq[slc]))
        res[chn][bn] = seq
        
    # COHS        
    cohs = {}
    chcombs = coh_combs(ch_names)
    for ch1idx, ch2idx in chcombs:
        ch1idx, ch2idx = ch_names.index(ch1idx), ch_names.index(ch2idx)
        pair = '_'.join((str(ch1idx), str(ch2idx)))
        pairname = f'{ch_names[ch1idx]}~{ch_names[ch2idx]}'
        for bn, (b1, b2) in BANDS.items():
            if bn in ('full', 'beta2'): continue
            if not (pair in cohs):
                # print(epochs[:, ch1idx][:, None].shape)
                cohs[pair] = coherence(epochs[:, ch1idx][:, None], epochs[:, ch2idx][:, None], fs=sfreq, nperseg=min(epochs.shape[-1]/2, 1024), nfft=1024, axis=-1, thresholds=thresholds) #, average='median'
                # print(np.min(cohs[pair][1]), np.max(cohs[pair][1]))
            freq, cxy = cohs[pair]
            freq_mask = (freq >= b1) & (freq <= b2)
            res.setdefault(pairname, {}) 
            res[pairname][bn]= np.mean(cxy[:, :, freq_mask], axis=-1)    
    
    epochs = np.array(epochs_bckup)
    
    if shortfeats:
        shortres = []
        for epoch in epochs: # ch x time*
            epochs_ = overlap_epochs(epoch.T, 2, 1, sfreq)[0]# np.vstack([epoch_[None, :, i*part_len: (i+1)*part_len] for i in range(n_partsvar)])
            freq, spec = welch(epochs_, fs=sfreq, axis=-1, nperseg=256, nfft=512, noverlap=160, average='median', thresholds=[THR_DICT.get(ch, np.inf) for ch in ch_names])
            for ch, chn in enumerate(ch_names):
                res[chn].setdefault('shortepochsPSD', [])
                res[chn]['shortepochsPSD'].append(spec[:, ch])
        res['shortfreq'] = freq 
    epochs = np.array(epochs_bckup)
    res['sync'] = corr_feat(epochs).mean(axis=1)
    
    epochs = np.apply_along_axis(bandpass_filter, -1, epochs, 4, 13, sfreq, 4, mode='bandpass')
    res['sync(theta-alpha)'] = corr_feat(epochs).mean(axis=1)
    return res, epoch_sec

def func_from_short_specs(speclist, freq, func=np.max, band=(8, 13), rel=False):
    res = []
    fslc = (freq >= band[0]) & (freq <= band[1]) 
    for specs2d in speclist:
        bandpsd = specs2d[:, fslc].sum(axis=-1)
        bandpsd = bandpsd/specs2d.sum(axis=-1) if rel else bandpsd
        res.append(func(bandpsd))
    return np.array(res)

def feature_set(specs, add_new=False):
    names = ['rel. α (O)', 'α (O) / α (F)', 'δ/β (O)', '(δ+θ)/(α+β) (O)', 'α/β (O)', 'δ/β (F)', 'mod. α freq. (O)', 'rel. α (F)', 'β/δ (O)', '(α+β)/(δ+θ) (O)', '(δ+θ)/(α+β) (F)', 'β/α (O)', 'β/α (F)', '(δ+θ)/(β+γ) (O)', '(δ+θ)/(β+γ) (F)', '(β+γ)/(δ+θ) (O)', '(δ+θ)/γ (O)', '(α+β)/(δ+θ) (F)']
    feat1 = (specs['O1']['alpha']/specs['O1']['full'] + specs['O2']['alpha']/specs['O2']['full'])/2
    feat2 = (specs['O1']['alpha'] + specs['O2']['alpha'])/(specs['F3']['alpha'] + specs['F4']['alpha'])
    feat3 = (specs['O1']['delta']/specs['O1']['beta'] + specs['O2']['delta']/specs['O2']['beta'])/2
    
    feat4O1 = (specs['O1']['delta'] + specs['O1']['theta'])/(specs['O1']['alpha']+specs['O1']['beta'])
    feat4O2 = (specs['O2']['delta'] + specs['O2']['theta'])/(specs['O2']['alpha']+specs['O2']['beta'])
    feat4 = (feat4O1 + feat4O2)/2
    
    feat5 = (specs['O1']['alpha']/specs['O1']['beta'] + specs['O2']['alpha']/specs['O2']['beta'])/2
    feat6 = (specs['F3']['delta']/specs['F3']['beta'] + specs['F4']['delta']/specs['F4']['beta'])/2
    feat7 = (specs['O1']['alpha_dom'] + specs['O2']['alpha_dom'])/2
    feat8 = (specs['F3']['alpha']/specs['F3']['full'] + specs['F4']['alpha']/specs['F4']['full'])/2
    
    feat9 = 1/feat3
    feat10 = 1/feat4
    
    feat11O1 = (specs['F3']['delta'] + specs['F3']['theta'])/(specs['F3']['alpha']+specs['F3']['beta'])
    feat11O2 = (specs['F4']['delta'] + specs['F4']['theta'])/(specs['F4']['alpha']+specs['F4']['beta'])
    feat11 = (feat11O1 + feat11O2)/2
    
    feat12 = 1/feat5
    feat13 = 2/(specs['F3']['alpha']/specs['F3']['beta'] + specs['F4']['alpha']/specs['F4']['beta'])
    
    feat14O1 = (specs['O1']['delta'] + specs['O1']['theta'])/(specs['O1']['gamma1']+specs['O1']['beta'])
    feat14O2 = (specs['O2']['delta'] + specs['O2']['theta'])/(specs['O2']['gamma1']+specs['O2']['beta'])
    feat14 = (feat14O1 + feat14O2)/2
    
    feat15O1 = (specs['F3']['delta'] + specs['F3']['theta'])/(specs['F3']['gamma1']+specs['F3']['beta'])
    feat15O2 = (specs['F4']['delta'] + specs['F4']['theta'])/(specs['F4']['gamma1']+specs['F4']['beta'])
    feat15 = (feat15O1 + feat15O2)/2 
    
    feat16 = 1/feat14
    
    feat17O1 = (specs['O1']['delta'] + specs['O1']['theta'])/specs['O1']['gamma1']
    feat17O2 = (specs['O2']['delta'] + specs['O2']['theta'])/specs['O2']['gamma1']
    feat17 = (feat17O1 + feat17O2)/2
    
    feat18 = 1/feat11
    featlist = [feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8, feat9, feat10, feat11, feat12,feat13, feat14, feat15, feat16, feat17, feat18]
    if add_new:
        names += ['α (O)', '1 / α (O)', 'α (F) / α (O)', 'rel. α (O) / rel. α (F)', 'rel. α (F) / rel. α (O)', 'α (O) / θ (F)', 'θ (F) / α (O)', 'rel. α (O) / rel. θ (F)', 'rel. θ (F) / rel. α (O)', 'θ (O) / θ (F)', 'θ (F) / θ (O)', 'rel. θ (O) / rel. θ (F)', 'rel. θ (F) / rel. θ (O)', 'θ (O) * α (F) / (α (O) * θ (F))', 'α (O) * θ (F) / (θ (O) * α (F))', 'rel. θ (O) * rel. α (F) / (rel. α (O) * rel. θ (F))', 'rel. α (O) * rel. θ (F) / (rel. θ (O) * rel. α (F))', 
                  
                  'max. 2s PSD θ (F)', 'max. 2s PSD θ (O)', 'max. 2s PSD α (O)', 
                  'max. 2s rel. PSD θ (F)', 'max. 2s rel. PSD θ (O)', 'max. 2s rel. PSD α (O)', '(δ+θ)/(α+β) (O1-O2)',
                 ]
        
        featlist.append((specs['O1']['alpha'] + specs['O2']['alpha'])/2)
        featlist.append(1/featlist[-1])
        
        featlist.append(1/featlist[1])
        featlist.append(featlist[0]/featlist[7])
        featlist.append(1/featlist[-1])
        
        thetaF = (specs['F3']['theta'] + specs['F4']['theta'])/2
        relthetaF = (specs['F3']['theta']/specs['F3']['full'] + specs['F4']['theta']/specs['F4']['full'])/2
        thetaO = (specs['O1']['theta'] + specs['O2']['theta'])/2
        relthetaO = (specs['O1']['theta']/specs['O1']['full'] + specs['O2']['theta']/specs['O2']['full'])/2
        
        featlist.append(featlist[-5]/thetaF)
        featlist.append(1/featlist[-1])
        featlist.append(featlist[0]/relthetaF)
        featlist.append(1/featlist[-1])
        
        featlist.append(thetaO/thetaF)
        featlist.append(1/featlist[-1])
        featlist.append(relthetaO/relthetaF)
        featlist.append(1/featlist[-1])
        
        alphaF = (specs['F3']['alpha'] + specs['F4']['alpha'])/2
        relalphaF = (specs['F3']['alpha']/specs['F3']['full'] + specs['F4']['alpha']/specs['F4']['full'])/2
        
        featlist.append(featlist[-4]*alphaF/featlist[-13])
        featlist.append(1/featlist[-1])
        featlist.append(relthetaO*relalphaF/featlist[0]/relthetaF)
        featlist.append(1/featlist[-1])
        ### res['O1']['shortepochsPSD']
        freq = specs['shortfreq']        
        maxthF3 = func_from_short_specs(specs['F3']['shortepochsPSD'], freq, func=np.nanmax, band=(4, 8))
        maxthF4 = func_from_short_specs(specs['F4']['shortepochsPSD'], freq, func=np.nanmax, band=(4, 8))
        maxthF = np.mean(np.hstack((maxthF3[:, None], maxthF4[:, None])), axis=-1)
        featlist.append(maxthF)
        
        maxthO1 = func_from_short_specs(specs['O1']['shortepochsPSD'], freq, func=np.nanmax, band=(4, 8))
        maxthO2 = func_from_short_specs(specs['O2']['shortepochsPSD'], freq, func=np.nanmax, band=(4, 8))
        maxthO = np.mean(np.hstack((maxthO1[:, None], maxthO2[:, None])), axis=-1)
        featlist.append(maxthO)
        
        maxalO1 = func_from_short_specs(specs['O1']['shortepochsPSD'], freq, func=np.nanmax, band=(8, 13))
        maxalO2 = func_from_short_specs(specs['O2']['shortepochsPSD'], freq, func=np.nanmax, band=(8, 13))
        maxalO = np.mean(np.hstack((maxalO1[:, None], maxalO2[:, None])), axis=-1)
        featlist.append(maxalO)
        
        maxthF3 = func_from_short_specs(specs['F3']['shortepochsPSD'], freq, func=np.nanmax, band=(4, 8), rel=True)
        maxthF4 = func_from_short_specs(specs['F4']['shortepochsPSD'], freq, func=np.nanmax, band=(4, 8), rel=True)
        maxthF = np.mean(np.hstack((maxthF3[:, None], maxthF4[:, None])), axis=-1)
        featlist.append(maxthF)
        
        maxthO1 = func_from_short_specs(specs['O1']['shortepochsPSD'], freq, func=np.nanmax, band=(4, 8), rel=True)
        maxthO2 = func_from_short_specs(specs['O2']['shortepochsPSD'], freq, func=np.nanmax, band=(4, 8), rel=True)
        maxthO = np.mean(np.hstack((maxthO1[:, None], maxthO2[:, None])), axis=-1)
        featlist.append(maxthO)
        
        maxalO1 = func_from_short_specs(specs['O1']['shortepochsPSD'], freq, func=np.nanmax, band=(8, 13), rel=True)
        maxalO2 = func_from_short_specs(specs['O2']['shortepochsPSD'], freq, func=np.nanmax, band=(8, 13), rel=True)
        maxalO = np.mean(np.hstack((maxalO1[:, None], maxalO2[:, None])), axis=-1)
        featlist.append(maxalO)
        
        featlist.append(np.abs(feat4O1 - feat4O2))
        
        pairs = ['F3~O1', 'O1~O2']
        pairbands = [['alpha', 'gamma1'], ['alpha', 'beta', 'gamma1']]
        replace_dct = {'delta': 'δ', 'theta': 'θ', 'alpha': 'α', 'beta': 'β', 'gamma1': 'γ'}
        for pair, pbs in zip(pairs, pairbands):
            for pb in pbs:
                names.append(pair + f' @ {replace_dct[pb]}')
                featlist.append(specs[pair][pb])
        names.append('sync')
        featlist.append(specs['sync'])
        names.append('sync(theta-alpha)')
        featlist.append(specs['sync(theta-alpha)'])
    return pd.DataFrame(np.hstack([np.array(el).reshape((-1, 1)) for el in featlist]), columns=names)

def quality_windows(df, ws):
    times = np.array(df.index)
    reacs, errs, lags = df.values.T
    res = []
    full = np.mean(df['lags'].where(df['reaction'] == 1).dropna().values[:20])
    for l, r in ws:
        slc = np.argwhere((times >= l) & (times < r))
        if slc.size > 0:
            n_reacs, n_errs = np.sum(reacs[slc]), np.sum(errs[slc])
            rts = reacs[slc] * lags[slc]
            rt = np.mean(rts[rts != 0])
            rt_var = np.std(rts[rts != 0])
            res.append([n_reacs/(n_errs + n_reacs), rt, rt_var])
            
            q = min(1, (n_reacs/max((n_reacs+n_errs), 1e-6))*(1-np.abs(1-full/rt)/2))
            res[-1].append(q)
        else: res.append((-1, -1, -1, -1))
    return pd.DataFrame(np.array(res).reshape((-1, 4)), columns=('Err. frac.', 'Av. reac.', 'Av. reac. var.', 'New IPE'))

def spec_plot(sig, sf, ax1ylims=(1, 30), percents=(0, 100), tstart=0, **kwargs):
    """
    :param sig: 1D array or iterable, одномерный сигнал
    :param sf: numeric, частота дискретизации сигнала
    :param ax1ylims: iterable of 2 numeric, нижняя и верхняя границы по оси Y (частоты)
    :param percents: iterable of 2 numeric, нижняя и верхняя границы цветовой схемы в персентилях
    :param tstart: numeric, время начала сегмента в эксперименте для корректных подписей к оси Х
    :param kwargs: именованные аргументы для вычисления спектрограммы (scipy.signal.stft)
    """
    plt.figure(figsize=(25, 5))
    ax = plt.subplot(111)
    f, t, Zxx = ss.stft(sig, fs=sf, axis=-1, **kwargs)
    if ax1ylims is None:
        ax1ylims = (0, self.sf//2)
    power = 10*np.log(np.abs(Zxx[(f <= ax1ylims[1])&(f >= ax1ylims[0])])) 
    # power = np.abs(Zxx[(f <= ax1ylims[1])&(f >= ax1ylims[0])])
    f = f[(f <= ax1ylims[1])&(f >= ax1ylims[0])]
    vmin, vmax = np.percentile(power, percents[0]), np.percentile(power, percents[1])
    ax.pcolormesh(t+tstart, f, power,  cmap='jet', vmin=vmin, vmax=vmax, rasterized=False) #shading='gouraud',
    ax.axvline((t[-1]+t[0])/2 + tstart, c='k')
    ax.set_xlabel('Time, s')
    ax.set_ylabel('Frequency, Hz')
    plt.show()

def mm(x):
    return (x - np.min(x))/np.max(x)

def topo_parts(epoch_, sfreq, n_parts=4, bands=[(4, 8), (8, 14)], ch_names=['F3', 'F4', 'O1', 'O2']):
    init_len = epoch_.shape[-1]
    part_len = int(epoch_.shape[1]/n_parts)
    epochs = np.vstack([epoch_[None, :, i*part_len: (i+1)*part_len] for i in range(n_parts)])
    
    evks = np.zeros((len(epochs), len(bands), len(ch_names)))
    for e, epoch in enumerate(epochs):
        freq, spec = welch(epoch[None], fs=sfreq, axis=-1, nperseg=1024, nfft=1024, average='median', thresholds=[THR_DICT.get(ch, np.inf) for ch in ch_names])
        spec = spec[0]
        for b, (b1, b2) in enumerate(bands):    
            evks[e, b] = spec[:, (freq >= b1) & (freq < b2)].sum(axis=-1)
            
    fig, axes = plt.subplots(len(bands), n_parts, figsize=(5*n_parts, 5*len(bands)))
    for e, epoch in enumerate(epochs):
        for b, (b1, b2) in enumerate(bands): 
            evk = mne.EvokedArray(evks[e, b][:, None], mne.create_info(ch_names, 1, 'eeg'))
            evk.set_montage('standard_1020')
            axes[b, e].set_title(f'{round(init_len/sfreq/n_parts)*e} s, {b1}-{b2} Hz')
            mne.viz.plot_topomap(evk.get_data().flatten(), evk.info, ch_type='eeg', sensors=True, names=ch_names, mask=None, mask_params=None, contours=6, outlines='head', sphere=None, image_interp='cubic', extrapolate='auto', border='mean', res=64, size=1, cmap=mpl.colormaps['coolwarm'], vlim=(np.min(evks[:, b]), np.max(evks[:, b])), cnorm=None, axes=axes[b, e], show=False, onselect=None)
    plt.tight_layout()
    plt.show()

def parts_1d(epoch_, sfreq, n_parts1d=15, n_partsvar=30, ch_names=['F3', 'F4', 'O1', 'O2']):
    init_len = epoch_.shape[-1]
    part_len = int(epoch_.shape[1]/n_parts1d)
    epochs = overlap_epochs(epoch_.T, part_len/sfreq, part_len/sfreq/3, sfreq)[0]# np.vstack([epoch_[None, :, i*part_len: (i+1)*part_len] for i in range(n_partsvar)])#np.vstack([epoch_[None, :, i*part_len: (i+1)*part_len] for i in range(n_parts1d)])
    freq, spec = welch(epochs, fs=sfreq, axis=-1, nperseg=1024, nfft=1024, average='median', thresholds=[THR_DICT.get(ch, np.inf) for ch in ch_names])
    
    relalpha = spec[:, :, (freq >= 8) & (freq <= 14)].sum(axis=-1)#/spec.sum(axis=-1)
    relalphaF = relalpha[:, ch_names.index('F3')] + relalpha[:, ch_names.index('F4')]
    relalphaO = relalpha[:, ch_names.index('O1')] + relalpha[:, ch_names.index('O2')]
    
    reltheta = spec[:, :, (freq >= 4) & (freq <= 8)].sum(axis=-1)#/spec.sum(axis=-1)
    relthetaF = reltheta[:, ch_names.index('F3')] + relalpha[:, ch_names.index('F4')]
    
    relthetaO = reltheta[:, ch_names.index('O1')] + relalpha[:, ch_names.index('O2')]
    
    relbeta = spec[:, :, (freq >= 14) & (freq <= 30)].sum(axis=-1)#/spec.sum(axis=-1)
    relbetaF = relbeta[:, ch_names.index('F3')] + relalpha[:, ch_names.index('F4')]
    
    reldelta = spec[:, :, (freq >= 1) & (freq <= 4)].sum(axis=-1)#/spec.sum(axis=-1)
    reldeltaF = reldelta[:, ch_names.index('F3')] + relalpha[:, ch_names.index('F4')]
    
    feat1 = relalphaF + relalphaO -relthetaF
    
    plt.figure(figsize=(25, 5))
    x = (init_len/sfreq/n_parts1d)*np.arange(len(feat1))
    ax = plt.subplot(111)
    # ax.plot(x, feat1, label='rel. α (F) + rel. α (O) - rel. θ (f)')
    ax.plot(x, mm(relalphaO), label='α (O)')
    ax.plot(x, mm(relalphaF), label='α (F)')
    ax.plot(x, mm(relthetaF), label='θ (F)')
    ax.legend() #loc='upper left'
    plt.show()
    
    plt.figure(figsize=(25, 5))
    x = (init_len/sfreq/n_parts1d)*np.arange(len(feat1))
    ax = plt.subplot(111)
    ax.plot(x, mm(relthetaO/relalphaO), label='θ / α (O)')
    ax.legend(loc='center left')
    ax = ax.twinx()
    ax.plot(x, mm(relthetaF/relalphaF), label='θ / α (F)', c='orange')
    ax.legend(loc='center right')
    ax.legend() #loc='upper left'
    plt.show()
    
#     # F slpashes
#     ax = plt.subplot(111) #ax.twinx()
#     ax.plot(x, zscore(relthetaF) + zscore(relbetaF) + zscore(reldeltaF), label='rel. θ (f) + rel. δ (F) + rel. β (F)', c='orange')
#     ax.legend() #loc='upper right'
    
#     plt.show()
    
    
#     part_len = int(epoch_.shape[1]/n_partsvar)
#     epochs = overlap_epochs(epoch_.T, part_len/sfreq, part_len/sfreq/4, sfreq)[0]# np.vstack([epoch_[None, :, i*part_len: (i+1)*part_len] for i in range(n_partsvar)])
#     freq, spec = welch(epochs, fs=sfreq, axis=-1, nperseg=1000, nfft=1024, noverlap=750, average='median', thresholds=[THR_DICT.get(ch, np.inf) for ch in ch_names])
#     thdbF = spec[..., ((freq >= 1) & (freq <= 8)) | ((freq >= 14) & (freq <= 30))].sum(axis=-1)[:, [ch_names.index('F3'), ch_names.index('F4')]].sum(axis=-1)
#     var = [np.var(thdbF[i:i+5]) for i in range(0, len(thdbF)-5)]
#     plt.figure(figsize=(25, 5))
#     plt.plot(var, label=' var(θ+δ+β) @ F')
#     plt.legend()
#     plt.show()
    