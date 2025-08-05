from sklearn.base import BaseEstimator, TransformerMixin
import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.stats import skew
from scipy.stats import kurtosis
import nolds
import antropy as ant
# import hrvanalysis.extract_features as hrv
import math

import matplotlib.pyplot as plt

class ECGFeature(BaseEstimator, TransformerMixin):
    def __init__(self, name, sample_rate):
        self.name = name
        self.sample_rate = sample_rate
        #self.params = params

    def replace_outliers(self, data, window_size=700, threshold=0.7):
        rolling_df = pd.Series(data).rolling(window=window_size, center=True)

        q1 = rolling_df.quantile(0.25)
        q3 = rolling_df.quantile(0.75)
        
        iqr = q3 - q1
        
        outliers = (data - rolling_df.median()).abs() > threshold * iqr
        
        clean_data = data.copy()
        clean_data[outliers] = rolling_df.median()[outliers]
        
        return clean_data
    
    def smooth(self, array, window_size=2, w_pred=2/3, w_next=1/3):
        smoothed_arr = array
        for i in range(window_size, len(array)):
            smoothed_arr[i] = (array[i - window_size] * w_pred + array[i] * w_next)
        return smoothed_arr[1:]
    
    def zero_crossing(self, array):
        return ant.num_zerocross(array - np.mean(array))
    # def r_offset_loc(self, peaks, slicedChannel):
    #     _, offsets = nk.ecg_delineate(slicedChannel, rpeaks=peaks, sampling_rate=self.sample_rate)
    #     offsets = offsets['ECG_R_Offsets']
    #     rLoc = []
    #     offsetLoc = []              
    #     for idx, val in enumerate(offsets):
    #         if not np.isnan(val):
    #             offsetLoc.append(val)
    #             rLoc.append(peaks[idx])
    #     rAmpl = []
    #     for idx, val in enumerate(slicedChannel[rLoc]): rAmpl.append(np.abs(val - slicedChannel[offsetLoc[idx]]))
    #     return rAmpl
    
    # def rq(self, slicedChannel, peaks):
    #     _, offsets = nk.ecg_delineate(slicedChannel, rpeaks=peaks, sampling_rate=self.sample_rate)
    #     Q = offsets['ECG_Q_Peaks']
    #     rLocQ = []
    #     QLoc = []              
    #     for idx, val in enumerate(Q):
    #         if not np.isnan(val):
    #             QLoc.append(val)
    #             rLocQ.append(peaks[idx])
    #     RQ = []
    #     for idx, val in enumerate(slicedChannel[rLocQ]): RQ.append(np.abs(val / slicedChannel[QLoc[idx]]))
    #     return np.mean(RQ)
    
    # def rs(self, slicedChannel, peaks):
    #     _, offsets = nk.ecg_delineate(slicedChannel, rpeaks=peaks, sampling_rate=self.sample_rate)
    #     S = offsets['ECG_S_Peaks']
    #     rLocS = []
    #     SLoc = []              
    #     for idx, val in enumerate(S):
    #         if not np.isnan(val):
    #             SLoc.append(val)
    #             rLocS.append(peaks[idx])
    #     RS = []
    #     for idx, val in enumerate(slicedChannel[rLocS]): RS.append(np.abs(val / slicedChannel[SLoc[idx]]))
    #     return np.mean(RS)
    
    def transform(self, X):
        if type(self.name) is not list: self.name = [self.name]
        if 'all' in self.name: self.name = ['rr', 'var', 'baevsky', 'kurtosis', 'skewness', 'hurst',
                                            'DFA', 'zero_cross', 'min', 'max', 'range', 'VLF', 'LF', 'HF',
                                            'LF/HF', 'total_power', '8Mean_energy', 'SSE', '1th_deriv', '2nd_deriv',
                                            'sdsd', 'median_nni', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20',
                                            'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr',
                                            'triangular_index', 'tinn', 'sd1', 'sd2', 'ratio_sd2_sd1', 'csi', 'cvi',
                                            'Modified_csi', 'raw_LF', 'raw_HF', 'raw_LF/HF']
        if 'raw_freq' in self.name: self.name = ['raw_freq']
        for slicedChannel in X[:, 0]:
            if np.any(np.isnan(slicedChannel)):
                nan_mask = np.isnan(slicedChannel)
                slicedChannel[nan_mask] = np.mean(slicedChannel[~nan_mask])
            
            feats = None
            try:
                if '8Mean_energy' in self.name or 'SSE' in self.name:
                    subband_powers = np.zeros(8)
                    rfft = np.fft.rfft(slicedChannel)
                    freq = np.fft.rfftfreq(len(slicedChannel), 1/self.sample_rate)
                    for idx, i in enumerate(np.arange(0, 10, 1.25)):
                        subband_powers[idx] = np.mean(np.abs(rfft[np.where((freq >= i) & (freq <= i+1.25))]))

                if 'raw_LF' in self.name or 'raw_HF' in self.name:
                    rfft = np.fft.rfft(slicedChannel)
                    freq = np.fft.rfftfreq(len(slicedChannel), 1/self.sample_rate)
                    raw_LF = np.mean(np.abs(rfft[np.where((freq >= 0.04) & (freq <= 0.15))]))
                    raw_HF = np.mean(np.abs(rfft[np.where((freq >= 0.15) & (freq <= 4))]))
                
                peaks = nk.ecg_findpeaks(slicedChannel, sampling_rate=self.sample_rate)['ECG_R_Peaks']
                rrInterval = np.diff(peaks)

                # plt.figure(figsize=(20, 3))
                # plt.plot(slicedChannel)
                # plt.scatter(peaks, slicedChannel[peaks], c='r')
                # plt.show()

                cleanedRr = self.smooth(self.replace_outliers(rrInterval) / self.sample_rate)
                cleanedRr = self.smooth(self.replace_outliers(cleanedRr))

                rr_intervals_ms = cleanedRr*1000
                if 'LF' in self.name or 'HF' in self.name or 'LF/HF' in self.name or 'VLF' in self.name or 'total_power' in self.name:
                    freq_features = hrv.get_frequency_domain_features(rr_intervals_ms)

                time_features = hrv.get_time_domain_features(rr_intervals_ms)

                if 'tinn' in self.name or 'triangular_index' in self.name:
                    geom_features = hrv.get_geometrical_features(rr_intervals_ms)

                if 'sd1' in self.name or 'sd2' in self.name or 'ratio_sd2_sd1' in self.name:
                    poincare_features = hrv.get_poincare_plot_features(rr_intervals_ms)

                if 'csi' in self.name or 'cvi' in self.name or 'Modified_csi' in self.name:
                    csi_cvi_features = hrv.get_csi_cvi_features(rr_intervals_ms)

                for name in self.name:
                    if name == '8Mean_energy':
                        if feats is None: feats = np.array(subband_powers)
                        else: feats = np.concatenate((feats, subband_powers))          
                    else:
                        # if name == 'heart_rate':
                        #     values = np.mean(self.heart_rate(peaks))
                        if name == 'rr':
                            values = np.mean(cleanedRr)
                        # elif name == 'r_ampl':
                        #     values = np.mean(self.r_offset_loc(peaks, slicedChannel))
                        # elif name == 'var_heart_rate':
                        #     values = np.var(self.heart_rate(peaks))
                        elif name == 'var':
                            values = np.var(cleanedRr)
                        # elif name == 'var_r_ampl':
                        #     values = np.var(self.r_offset_loc(peaks, slicedChannel))
                        elif name == 'baevsky':
                            histog, binEdges = np.histogram(cleanedRr, bins=10)
                            regVal = binEdges[np.argmax(histog)]
                            values = ((np.max(histog) / len(cleanedRr)) * 100) / (2 * regVal * (np.max(cleanedRr) - np.min(cleanedRr)))
                        elif name == 'kurtosis':
                            values = kurtosis(cleanedRr)
                        elif name == 'skewness':
                            values = skew(cleanedRr)
                        elif name == 'hurst':
                            values = nolds.hurst_rs(cleanedRr)
                        elif name == 'DFA':
                            values = nolds.dfa(cleanedRr)
                        elif name == 'zero_cross':
                            values = self.zero_crossing(cleanedRr)
                        elif name == 'min':
                            values = np.min(cleanedRr)
                        elif name == 'max':
                            values = np.max(cleanedRr)
                        elif name == 'range':
                            values = np.max(cleanedRr) - np.min(cleanedRr)
                        elif name == 'raw_freq':
                            return 1/cleanedRr, peaks[2:]
                        elif name == 'VLF':
                            values = freq_features['vlf']
                        elif name == 'LF':
                            values = freq_features['lf']
                        elif name == 'HF':
                            values = freq_features['hf']
                        elif name == 'LF/HF':
                            values = freq_features['lf_hf_ratio']
                        elif name == 'total_power':
                            values = freq_features['total_power']
                        elif name == 'SSE':
                            normalized_sp = np.zeros(8)
                            for idx, i in enumerate(subband_powers):
                                normalized_sp[idx] = i / sum(subband_powers)
                            values = -sum([i*math.log2(i) for i in normalized_sp])
                        elif name == '1th_deriv':
                            values = np.mean(np.diff(cleanedRr))
                        elif name == '2nd_deriv':
                            values = np.mean(np.diff(np.diff(cleanedRr)))
                        elif name == 'sdsd':
                            values = time_features['sdsd']
                        elif name == 'median_nni':
                            values = time_features['median_nni']
                        elif name == 'nni_50':
                            values = time_features['nni_50']
                        elif name == 'pnni_50':
                            values = time_features['pnni_50']
                        elif name == 'nni_20':
                            values = time_features['nni_20']
                        elif name == 'pnni_20':
                            values = time_features['pnni_20']
                        elif name == 'range_nni':
                            values = time_features['range_nni']
                        elif name == 'cvsd':
                            values = time_features['cvsd']
                        elif name == 'cvnni':
                            values = time_features['cvnni']
                        elif name == 'mean_hr':
                            values = time_features['mean_hr']
                        elif name == 'max_hr':
                            values = time_features['max_hr']
                        elif name == 'min_hr':
                            values = time_features['min_hr']
                        elif name == 'std_hr':
                            values = time_features['std_hr']
                        elif name == 'triangular_index':
                            values = geom_features['triangular_index']
                        elif name == 'tinn':
                            values = geom_features['tinn']
                        elif name == 'sd1':
                            values = poincare_features['sd1']
                        elif name == 'sd2':
                            values = poincare_features['sd2']
                        elif name == 'ratio_sd2_sd1':
                            values = poincare_features['ratio_sd2_sd1']
                        elif name == 'csi':
                            values = csi_cvi_features['csi']
                        elif name == 'cvi':
                            values = csi_cvi_features['cvi']
                        elif name == 'Modified_csi':
                            values = csi_cvi_features['Modified_csi']
                        elif name == 'raw_LF':
                            values = raw_LF
                        elif name == 'raw_HF':
                            values = raw_HF
                        elif name == 'raw_LF/HF':
                            values = raw_LF/raw_HF
                        # elif name == 'r/q':
                        #     values = self.rq(slicedChannel, peaks)
                        # elif name == 'r/s':
                        #     values = self.rs(slicedChannel, peaks)

                        if feats is None: feats = np.array([values])
                        else: feats = np.append(feats, values)
            except Exception: feats = np.ones(len(self.name))*-1

            #else: feats = np.zeros(len(self.name))
            
            try: all_feats = np.vstack((all_feats, feats))
            except NameError: all_feats = feats
        
        #all_feats[np.isnan(all_feats)] = -1
        return all_feats
    
    def fit(self, X, y=None):
        return self
