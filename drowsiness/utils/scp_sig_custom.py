import numpy as np
from scipy.signal._arraytools import odd_ext, const_ext, zero_ext, even_ext
import warnings
# from scipy._lib.six import string_types
from scipy.signal.windows import get_window
from scipy import fftpack
from scipy.signal import signaltools

string_types = str

CUR_THR = np.inf
def csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
        detrend='constant', return_onesided=True, scaling='density',
        axis=-1, average='mean', thresholds=False):
    global CUR_THR
    # freqs, _, Pxy = _spectral_helper(x, y, fs, window, nperseg, noverlap, nfft,
    #                                  detrend, return_onesided, scaling, axis,
    #                                  mode='psd')
    Pxys = []
    for ch in range(x.shape[axis-1]):
        CUR_THR = thresholds[ch]
        freqs, _, Pxy = _spectral_helper(x[:, ch][:, None], y[:, ch][:, None], fs, window, nperseg, noverlap, nfft,
                                     detrend, return_onesided, scaling, axis,
                                     mode='psd')
        Pxys.append(Pxy)
    Pxy = np.hstack(Pxys)

    if len(Pxy.shape) >= 2 and Pxy.size > 0:
        if Pxy.shape[-1] > 1:
            if average == 'median':
                Pxy = np.nanmedian(Pxy, axis=-1) / _median_bias(Pxy.shape[-1])
            elif average == 'mean':
                Pxy = np.nanmean(Pxy, axis=-1)
            else:
                raise ValueError('average must be "median" or "mean", got %s'
                                 % (average,))
        else:
            Pxy = np.reshape(Pxy, Pxy.shape[:-1])

    return freqs, Pxy


def welch(x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None,
          detrend='constant', return_onesided=True, scaling='density',
          axis=-1, average='mean', thresholds=False):

    freqs, Pxx = csd(x, x, fs=fs, window=window, nperseg=nperseg,
                     noverlap=noverlap, nfft=nfft, detrend=detrend,
                     return_onesided=return_onesided, scaling=scaling,
                     axis=axis, average=average, thresholds=thresholds)

    return freqs, Pxx.real


def _spectral_helper(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
                     nfft=None, detrend='constant', return_onesided=True,
                     scaling='density', axis=-1, mode='psd', boundary=None,
                     padded=False):

    if mode not in ['psd', 'stft']:
        raise ValueError("Unknown value for mode %s, must be one of: "
                         "{'psd', 'stft'}" % mode)

    boundary_funcs = {'even': even_ext,
                      'odd': odd_ext,
                      'constant': const_ext,
                      'zeros': zero_ext,
                      None: None}

    if boundary not in boundary_funcs:
        raise ValueError("Unknown boundary option '{0}', must be one of: {1}"
                         .format(boundary, list(boundary_funcs.keys())))

    # If x and y are the same object we can save ourselves some computation.
    same_data = y is x

    if not same_data and mode != 'psd':
        raise ValueError("x and y must be equal if mode is 'stft'")

    axis = int(axis)

    # Ensure we have np.arrays, get outdtype
    x = np.asarray(x)
    if not same_data:
        y = np.asarray(y)
        outdtype = np.result_type(x, y, np.complex64)
    else:
        outdtype = np.result_type(x, np.complex64)

    if not same_data:
        # Check if we can broadcast the outer axes together
        xouter = list(x.shape)
        youter = list(y.shape)
        xouter.pop(axis)
        youter.pop(axis)
        try:
            outershape = np.broadcast(np.empty(xouter), np.empty(youter)).shape
        except ValueError:
            raise ValueError('x and y cannot be broadcast together.')

    if same_data:
        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape), np.empty(x.shape)
    else:
        if x.size == 0 or y.size == 0:
            outshape = outershape + (min([x.shape[axis], y.shape[axis]]),)
            emptyout = np.rollaxis(np.empty(outshape), -1, axis)
            return emptyout, emptyout, emptyout

    if x.ndim > 1:
        if axis != -1:
            x = np.rollaxis(x, axis, len(x.shape))
            if not same_data and y.ndim > 1:
                y = np.rollaxis(y, axis, len(y.shape))

    # Check if x and y are the same length, zero-pad if necessary
    if not same_data:
        if x.shape[-1] != y.shape[-1]:
            if x.shape[-1] < y.shape[-1]:
                pad_shape = list(x.shape)
                pad_shape[-1] = y.shape[-1] - x.shape[-1]
                x = np.concatenate((x, np.zeros(pad_shape)), -1)
            else:
                pad_shape = list(y.shape)
                pad_shape[-1] = x.shape[-1] - y.shape[-1]
                y = np.concatenate((y, np.zeros(pad_shape)), -1)

    if nperseg is not None:  # if specified by user
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    # parse window; if array like, then set nperseg = win.shape
    win, nperseg = _triage_segments(window, nperseg, input_length=x.shape[-1])

    if nfft is None:
        nfft = nperseg
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    # Padding occurs after boundary extension, so that the extended signal ends
    # in zeros, instead of introducing an impulse at the end.
    # I.e. if x = [..., 3, 2]
    # extend then pad -> [..., 3, 2, 2, 3, 0, 0, 0]
    # pad then extend -> [..., 3, 2, 0, 0, 0, 2, 3]

    if boundary is not None:
        ext_func = boundary_funcs[boundary]
        x = ext_func(x, nperseg//2, axis=-1)
        if not same_data:
            y = ext_func(y, nperseg//2, axis=-1)

    if padded:
        # Pad to integer number of windowed segments
        # I.e make x.shape[-1] = nperseg + (nseg-1)*nstep, with integer nseg
        nadd = (-(x.shape[-1]-nperseg) % nstep) % nperseg
        zeros_shape = list(x.shape[:-1]) + [nadd]
        x = np.concatenate((x, np.zeros(zeros_shape)), axis=-1)
        if not same_data:
            zeros_shape = list(y.shape[:-1]) + [nadd]
            y = np.concatenate((y, np.zeros(zeros_shape)), axis=-1)

    # Handle detrending and window functions
    if not detrend:
        def detrend_func(d):
            return d
    elif not hasattr(detrend, '__call__'):
        def detrend_func(d):
            return signaltools.detrend(d, type=detrend, axis=-1)
    elif axis != -1:
        # Wrap this function so that it receives a shape that it could
        # reasonably expect to receive.
        def detrend_func(d):
            d = np.rollaxis(d, -1, axis)
            d = detrend(d)
            return np.rollaxis(d, axis, len(d.shape))
    else:
        detrend_func = detrend

    if np.result_type(win, np.complex64) != outdtype:
        win = win.astype(outdtype)

    if scaling == 'density':
        scale = 1.0 / (fs * (win*win).sum())
    elif scaling == 'spectrum':
        scale = 1.0 / win.sum()**2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)

    if mode == 'stft':
        scale = np.sqrt(scale)

    if return_onesided:
        if np.iscomplexobj(x):
            sides = 'twosided'
            warnings.warn('Input data is complex, switching to '
                          'return_onesided=False')
        else:
            sides = 'onesided'
            if not same_data:
                if np.iscomplexobj(y):
                    sides = 'twosided'
                    warnings.warn('Input data is complex, switching to '
                                  'return_onesided=False')
    else:
        sides = 'twosided'

    if sides == 'twosided':
        freqs = fftpack.fftfreq(nfft, 1/fs)
    elif sides == 'onesided':
        freqs = np.fft.rfftfreq(nfft, 1/fs)

    # Perform the windowed FFTs
    result = _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides)

    if not same_data:
        # All the same operations on the y data
        result_y = _fft_helper(y, win, detrend_func, nperseg, noverlap, nfft,
                               sides)
        result = np.conjugate(result) * result_y
    elif mode == 'psd':
        result = np.conjugate(result) * result

    result *= scale
    if sides == 'onesided' and mode == 'psd':
        if nfft % 2:
            result[..., 1:] *= 2
        else:
            # Last point is unpaired Nyquist freq point, don't double
            result[..., 1:-1] *= 2

    time = np.arange(nperseg/2, x.shape[-1] - nperseg/2 + 1,
                     nperseg - noverlap)/float(fs)
    if boundary is not None:
        time -= (nperseg/2) / fs

    result = result.astype(outdtype)

    # All imaginary parts are zero anyways
    if same_data and mode != 'stft':
        result = result.real

    # Output is going to have new last axis for time/window index, so a
    # negative axis index shifts down one
    if axis < 0:
        axis -= 1

    # Roll frequency axis back to axis where the data came from
    result = np.rollaxis(result, -1, axis)

    return freqs, time, result

def _fft_helper(x, win, detrend_func, nperseg, noverlap, nfft, sides):

    # Created strided array of data segments
    if nperseg == 1 and noverlap == 0:
        result = x[..., np.newaxis]
    else:
        # https://stackoverflow.com/a/5568169
        step = nperseg - noverlap
        shape = x.shape[:-1]+((x.shape[-1]-noverlap)//step, nperseg)
        strides = x.strides[:-1]+(step*x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape,
                                                 strides=strides)
        mask = np.any(np.abs(result) > CUR_THR, axis=-1)
        result[mask] = np.nan
        # print(result.shape, mask.shape, np.where(np.isnan(result))[0].size, CUR_THR, np.nanmax(result))
        # print(mask[121])
        

    # Detrend each data segment individually
    result = detrend_func(result)

    # Apply window by multiplication
    result = win * result

    # Perform the fft. Acts on last axis by default. Zero-pads automatically
    if sides == 'twosided':
        func = fftpack.fft
    else:
        result = result.real
        func = np.fft.rfft
    result = func(result, n=nfft)

    return result

def _triage_segments(window, nperseg, input_length):

    # parse window; if array like, then set nperseg = win.shape
    if isinstance(window, string_types) or isinstance(window, tuple):
        # if nperseg not specified
        if nperseg is None:
            nperseg = 256  # then change to default
        if nperseg > input_length:
            warnings.warn('nperseg = {0:d} is greater than input length '
                          ' = {1:d}, using nperseg = {1:d}'
                          .format(nperseg, input_length))
            nperseg = input_length
        win = get_window(window, nperseg)
    else:
        win = np.asarray(window)
        if len(win.shape) != 1:
            raise ValueError('window must be 1-D')
        if input_length < win.shape[-1]:
            raise ValueError('window is longer than input signal')
        if nperseg is None:
            nperseg = win.shape[0]
        elif nperseg is not None:
            if nperseg != win.shape[0]:
                raise ValueError("value specified for nperseg is different"
                                 " from length of window")
    return win, nperseg

def _median_bias(n):
    ii_2 = 2 * np.arange(1., (n-1) // 2 + 1)
    return 1 + np.sum(1. / (ii_2 + 1) - 1. / ii_2)

def coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None,
              nfft=None, detrend='constant', axis=-1, average='mean', thresholds=False):
   
    freqs, Pxx = welch(x, fs=fs, window=window, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft, detrend=detrend,
                       axis=axis, average=average, thresholds=thresholds)
    _, Pyy = welch(y, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
                   nfft=nfft, detrend=detrend, axis=axis, average=average, thresholds=thresholds)
    _, Pxy = csd(x, y, fs=fs, window=window, nperseg=nperseg,
                 noverlap=noverlap, nfft=nfft, detrend=detrend, axis=axis, average=average, thresholds=thresholds)
    
    Cxy = np.abs(Pxy)**2 / Pxx / Pyy

    return freqs, Cxy