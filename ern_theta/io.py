"""Loading single-channel epochs from the preprocessed MNE files."""

import warnings

from .config import CHANNEL


def load_single_channel(epochs_path, codes, channel=CHANNEL):
    """Return ``(data, times, sfreq)`` for the epochs matching ``codes``.

    ``data`` has shape ``(n_trials, n_times)`` and is in µV.
    """
    import mne

    with warnings.catch_warnings():
        # The BIDS-style '*_desc-epochs_eeg.fif' names are fine; MNE just prefers '-epo.fif'.
        warnings.filterwarnings('ignore', message='This filename .* does not conform to MNE naming')
        epochs = mne.read_epochs(epochs_path, preload=True, verbose=False)[codes]
    data = epochs.copy().pick([channel]).get_data()[:, 0, :] * 1e6  # V -> µV
    return data, epochs.times, epochs.info['sfreq']
