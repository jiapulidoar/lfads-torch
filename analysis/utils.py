import numpy as np
from scipy.stats import norm
from scipy.signal import lfilter

def filter_gauss(spike_tensor, SD):
    """
    spike_tensor: (trials, time, neurons)
    SD: standard deviation in ms (same units as time bins)
    
    returns:
        smoothed firing rate (same shape), in spikes/ms
    """

    # --- Gaussian kernel ---
    SDrounded = int(2 * round(SD / 2))
    gausswidth = int(8 * SDrounded)

    x = np.arange(1, gausswidth + 1)
    F = norm.pdf(x, loc=gausswidth / 2, scale=SD)
    F = F / np.sum(F)

    shift = len(F) // 2

    n_trials, T, n_neurons = spike_tensor.shape

    # --- padding with mean (like MATLAB) ---
    start_mean = np.mean(spike_tensor[:, :SDrounded, :], axis=1, keepdims=True)
    end_mean   = np.mean(spike_tensor[:, -SDrounded:, :], axis=1, keepdims=True)

    pad_start = np.repeat(start_mean, shift, axis=1)
    pad_end   = np.repeat(end_mean, shift, axis=1)

    prefilt = np.concatenate([pad_start, spike_tensor, pad_end], axis=1)

    # --- filtering (causal, along time axis=1) ---
    postfilt = lfilter(F, 1, prefilt, axis=1)

    # --- trim (same as MATLAB) ---
    out = postfilt[:, 2*shift : 2*shift + T, :]

    return out

def bin_spikes_counts(data, bin_size=20):
    """
    data: (trials, time, neurons)
    bin_size: in timesteps (e.g., 20 if 1 ms bins)
    """
    T = data.shape[1]
    T_trim = (T // bin_size) * bin_size
    
    data_trim = data[:, :T_trim, :]
    
    binned = data_trim.reshape(
        data.shape[0],
        T_trim // bin_size,
        bin_size,
        data.shape[2]
    ).sum(axis=2)  #key: SUM, not mean
    
    return binned