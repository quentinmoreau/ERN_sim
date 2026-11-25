"""Morlet-wavelet time-frequency power and summary measures."""

import numpy as np
from scipy.signal import fftconvolve


def morlet_wavelet(freq, n_cycles, sfreq, max_duration=0.5):
    """Complex Morlet wavelet (unnormalised) with half-length capped at ``max_duration`` s."""
    duration = min(3 * n_cycles / freq, max_duration)
    t = np.arange(-duration, duration, 1 / sfreq)
    sigma_t = n_cycles / (2 * np.pi * freq)
    return np.exp(2j * np.pi * freq * t) * np.exp(-t**2 / (2 * sigma_t**2))


def compute_tfr(trials, freqs, sfreq):
    """Trial-averaged Morlet power.

    Parameters
    ----------
    trials : array, shape (n_trials, n_times) or (n_times,)
        Pass the grand average (1-D) to get *evoked* power, or single trials to
        get *total* power (the mean of single-trial power).
    freqs : array
        Frequencies in Hz. Wavelets use ``n_cycles = freqs / 2``.
    sfreq : float
        Sampling rate in Hz.

    Returns
    -------
    tfr : array, shape (n_freqs, n_times)
    """
    trials = np.atleast_2d(trials)
    tfr = np.empty((len(freqs), trials.shape[1]))
    for i, freq in enumerate(freqs):
        wavelet = morlet_wavelet(freq, freq / 2.0, sfreq)
        conv = fftconvolve(trials, wavelet[None, :], mode='same', axes=1)
        tfr[i] = (np.abs(conv) ** 2).mean(axis=0)
    return tfr


def baseline_correct(tfr, times, bl_window=(-0.5, 0)):
    """Percent change relative to the mean power in ``bl_window`` (seconds)."""
    bl_mask = (times >= bl_window[0]) & (times <= bl_window[1])
    bl = tfr[:, bl_mask].mean(axis=1, keepdims=True)
    return (tfr - bl) / bl * 100


def band_power(tfr, freqs, times, freq_range=(4, 8), time_window=(0, 0.5)):
    """Mean of ``tfr`` within a frequency band and time window (default: theta, 0-500 ms)."""
    f_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    t_mask = (times >= time_window[0]) & (times <= time_window[1])
    return tfr[f_mask][:, t_mask].mean()
