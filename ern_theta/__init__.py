"""Helpers for 'Not all theta is real': removing illusory induced oscillations
with Woody filtering and single-trial ERP regression."""

from .preprocessing import align_trials, regress_erp, reinstate_lags
from .tfr import band_power, baseline_correct, compute_tfr
from .woody import woody, woody_stochastic
