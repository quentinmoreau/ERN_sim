"""Simulated response-locked ERN + Pe single trials."""

import numpy as np

# Default ERN / Pe shape (Gaussian components; times in s, amplitudes in µV)
ERN = dict(peak_time=0.15, amplitude=-8.0, width=0.05)
PE = dict(peak_time=0.30, amplitude=10.0, width=0.07)


def gaussian(times, peak_time, amplitude, width):
    return amplitude * np.exp(-0.5 * ((times - peak_time) / width) ** 2)


def pink_noise(n_samples, rng):
    """Unit-variance 1/f (pink) noise."""
    spectrum = np.fft.rfft(rng.standard_normal(n_samples))
    f = np.fft.rfftfreq(n_samples)
    f[0] = 1
    noise = np.fft.irfft(spectrum / np.sqrt(f), n=n_samples)
    return noise / noise.std()


def simulate_trials(times, n_trials, rng, jitter_sd=0.035, noise_amplitude=3.0,
                    amplitude_jitter_sd=0.2, ern=ERN, pe=PE):
    """Simulate jittered ERN + Pe trials in pink noise.

    Each trial gets one Gaussian latency shift (s.d. ``jitter_sd`` seconds)
    shared by ERN and Pe, independent log-normal amplitude scaling of each
    component, and additive pink noise scaled by ``noise_amplitude``. There is
    no oscillatory activity in the simulated data.

    Returns
    -------
    trials : array, shape (n_trials, n_times)
    true_jitters : array, shape (n_trials,)
        Latency shift of each trial in seconds.
    """
    trials = np.zeros((n_trials, len(times)))
    true_jitters = np.zeros(n_trials)
    for i in range(n_trials):
        jitter = max(rng.normal(0, jitter_sd), -ern['peak_time'])
        true_jitters[i] = jitter
        ern_amp = ern['amplitude'] * rng.lognormal(0, amplitude_jitter_sd)
        pe_amp = pe['amplitude'] * rng.lognormal(0, amplitude_jitter_sd)
        trials[i] = (gaussian(times, ern['peak_time'] + jitter, ern_amp, ern['width'])
                     + gaussian(times, pe['peak_time'] + jitter, pe_amp, pe['width']))
        if noise_amplitude:
            trials[i] += noise_amplitude * pink_noise(len(times), rng)
    return trials, true_jitters
