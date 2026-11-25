"""Single-trial alignment and ERP regression.

All functions take ``trials`` shaped ``(n_trials, n_times)``.
"""

import numpy as np


def align_trials(trials, lags):
    """Shift each trial by its Woody lag so the evoked response is aligned.

    A positive lag shifts the trial to the left (earlier); samples shifted in
    at the edge are zero-padded.
    """
    aligned = np.zeros_like(trials)
    for i, lag in enumerate(lags):
        if lag > 0:
            aligned[i, :-lag] = trials[i, lag:]
        elif lag < 0:
            aligned[i, -lag:] = trials[i, :lag]
        else:
            aligned[i] = trials[i]
    return aligned


def reinstate_lags(trials, lags):
    """Undo :func:`align_trials`: put each trial back at its original latency.

    Used after regressing the aligned ERP out of aligned trials, so that any
    remaining (genuine) activity returns to its true time relative to the event.
    """
    unaligned = np.zeros_like(trials)
    for i, lag in enumerate(lags):
        if lag > 0:
            unaligned[i, lag:] = trials[i, :-lag]
        elif lag < 0:
            unaligned[i, :lag] = trials[i, -lag:]
        else:
            unaligned[i] = trials[i]
    return unaligned


def regress_erp(trials, erp):
    """Regress the average ERP out of every trial and return the residuals.

    For each trial, fits ``trial = b0 + b1 * erp + e`` by ordinary least
    squares and keeps ``e``.
    """
    erp_c = erp - erp.mean()
    trials_c = trials - trials.mean(axis=1, keepdims=True)
    slopes = trials_c @ erp_c / (erp_c @ erp_c)
    intercepts = trials.mean(axis=1) - slopes * erp.mean()
    return trials - (intercepts[:, None] + slopes[:, None] * erp[None, :])
