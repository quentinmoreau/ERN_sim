"""
Woody filter for temporal alignment of single trials with latency jitter.

Python translation of the MATLAB implementation used in
Bonaiuto et al. (2021), "Laminar dynamics of high amplitude beta bursts in
human motor cortex". Original method: Woody (1967), Med. & Biol. Eng. 5, 539-554.

Lag convention: ``lag > 0`` means the event occurred *later* than in the
template, so the trial must be shifted ``lag`` samples to the left to align it
(see :func:`ern_theta.preprocessing.align_trials`).

Author: Quentin Moreau (2026-01-19)
"""

import numpy as np
from scipy.signal import correlate
from typing import Tuple, Optional


def woody(x: np.ndarray, 
          tol: float = 0.1,
          max_it: int = 100,
          est_mthd: str = 'woody',
          xcorr_mthd: str = 'unbiased',
          thornton_sub: int = 3,
          max_shift: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weighted average using Woody filter for signals with temporal jitter.
    
    Parameters
    ----------
    x : np.ndarray
        Signal measurements. Each COLUMN represents an independent measure 
        of the signal (or trial). Shape: (n_timepoints, n_trials)
    tol : float, optional
        Tolerance parameter to stop iteration (default is 0.1)
    max_it : int, optional
        Maximum number of iterations (default is 100)
    est_mthd : str, optional
        Estimation method to use:
        - 'woody': classical approach (default)
        - 'thornton': implements Thornton approach for different noise sources
    xcorr_mthd : str, optional
        Method for cross-correlation normalization:
        - 'biased': scales by 1/M
        - 'unbiased': scales by 1/(M-abs(lags)) (default)
    thornton_sub : int, optional
        Number of sub-averages to use in Thornton procedure (default is 3)
    max_shift : int, optional
        Maximum shift (in samples) to search for alignment. If None, searches
        entire cross-correlation range. Recommended to set this to avoid 
        spurious alignments at low SNR (default is None)
    
    Returns
    -------
    out : np.ndarray
        Final averaged waveform (time aligned). Shape: (n_timepoints,)
    est_lags : np.ndarray
        Estimated lags for each measurement. Shape: (n_trials,) or (n_subaverages,)
    
    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> t = np.linspace(-1, 1, 1001)
    >>> s = np.exp(-t**2 / (2 * 0.05**2))                  # transient "ERP"
    >>> true_lags = rng.integers(-40, 41, size=100)
    >>> S = np.stack([np.roll(s, k) for k in true_lags], axis=1)
    >>> S += 0.1 * rng.standard_normal(S.shape)
    >>> aligned_avg, lags = woody(S, max_shift=100)
    >>> np.corrcoef(lags, true_lags)[0, 1] > 0.99
    True
    """
    
    if est_mthd == 'woody':
        out, est_lags = woody_core(x, tol, max_it, xcorr_mthd, max_shift)
        
    elif est_mthd == 'thornton':
        # Implement Thornton 2008 procedure
        N, M = x.shape
        K = M // thornton_sub
        
        # Call woody several times implementing sub-averages
        for k in range(1, K + 1):
            sub = thornton_sub * k
            ind = np.round(np.linspace(0, M, sub + 1)).astype(int)
            
            if (len(ind) - 2) > (M / 2):
                # Number of sub-averages is equal to or just less than
                # half the number of trials, move to final stage
                out, est_lags = woody_core(x, tol, max_it, xcorr_mthd, max_shift)
                break
            
            # Get woody average from sub-averages
            y = gen_subave(x, ind)
            y_old = y.copy()
            err = 1
            
            while err > 0:
                _, est_lags = woody_core(y, tol, max_it, xcorr_mthd, max_shift)
                x = shift_data(x, est_lags, ind, N, M)
                y = gen_subave(x, ind)
                err = np.sum(np.abs(y - y_old))
                y_old = y.copy()
        
        # Final average
        out, est_lags = woody_core(x, tol, max_it, xcorr_mthd, max_shift)
    
    else:
        raise ValueError(f"Invalid est_mthd: '{est_mthd}'. Valid options are: 'woody', 'thornton'")
    
    return out, est_lags


def woody_core(x: np.ndarray, 
               tol: float,
               max_it: int,
               xcorr_mthd: str,
               max_shift: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core Woody filtering algorithm.
    
    Parameters
    ----------
    x : np.ndarray
        Signal matrix (n_timepoints, n_trials)
    tol : float
        Convergence tolerance
    max_it : int
        Maximum iterations
    xcorr_mthd : str
        Cross-correlation method ('biased' or 'unbiased')
    max_shift : int, optional
        Maximum shift to search for alignment (default is None = unlimited)
    
    Returns
    -------
    out : np.ndarray
        Aligned average waveform
    lag_data : np.ndarray
        Estimated lag for each trial
    """
    N, M = x.shape
    mx = np.mean(x, axis=1)  # Initial template
    p = 0.0
    conv = True
    run = 0
    
    sig_x = np.sqrt(np.sum(x**2, axis=0))
    
    X = correlate(mx, mx, mode='full')
    ref = len(X) // 2
    
    lag_data = np.zeros(M, dtype=int)
    
    while conv and (run < max_it):
        z = np.zeros(N)
        w = np.ones(N)
        
        for i in range(M):
            y = x[:, i]
            
            xy = correlate(mx, y, mode='full')
            
            if xcorr_mthd == 'unbiased':
                lags = np.arange(-N+1, N)
                norm = N - np.abs(lags)
                xy = xy / norm
            elif xcorr_mthd == 'biased':
                xy = xy / N
            
            if max_shift is not None:
                search_start = max(0, ref - max_shift)
                search_end = min(len(xy), ref + max_shift + 1)
                search_range = xy[search_start:search_end]
                ind = np.argmax(search_range) + search_start
            else:
                ind = np.argmax(xy)
            
            lag = ref - ind
            
            if lag > 0:
                num = w[lag:] - 1
                z[:N-lag] = (z[:N-lag] * num + y[lag:]) / w[lag:]
                w[lag:] += 1
            elif lag < 0:
                lag_abs = abs(lag)
                num = w[lag_abs:] - 1
                z[lag_abs:] = (z[lag_abs:] * num + y[:N-lag_abs]) / w[lag_abs:]
                w[lag_abs:] += 1
            else:
                z = z * (w - 1) / w + y / w
                w += 1
            
            lag_data[i] = lag
        
        old_mx = mx.copy()
        mx = z.copy()
        
        p_old = p
        if np.sqrt(np.dot(mx, mx)) > 0 and np.all(sig_x > 0):
            p_vec = np.dot(mx, x) / (np.sqrt(np.dot(mx, mx)) * sig_x)
            p = np.sum(p_vec) / M
        else:
            p = 0
        
        err = abs(p - p_old)
        if err < tol:
            conv = False
        
        run += 1
    
    return mx, lag_data


def shift_data(x: np.ndarray, 
               est_lags: np.ndarray,
               ind: np.ndarray,
               N: int,
               M: int) -> np.ndarray:
    """
    Shift individual trials within each sub-average.
    
    Parameters
    ----------
    x : np.ndarray
        Signal matrix (n_timepoints, n_trials)
    est_lags : np.ndarray
        Estimated lags for each sub-average
    ind : np.ndarray
        Indices defining sub-averages
    N : int
        Number of timepoints
    M : int
        Number of trials
    
    Returns
    -------
    x : np.ndarray
        Shifted signal matrix
    """
    x = x.copy()
    K = len(est_lags)
    
    for k in range(K):
        lag = est_lags[k]
        
        if lag != 0:
            if k != K - 1:
                sel_ind = np.arange(ind[k], ind[k+1])
            else:
                sel_ind = np.arange(ind[k], M)
            
            pad = len(sel_ind)
            
            if lag > 0:
                x[:, sel_ind] = np.vstack([
                    np.zeros((lag, pad)),
                    x[:N-lag, sel_ind]
                ])
            elif lag < 0:
                lag_abs = abs(lag)
                x[:, sel_ind] = np.vstack([
                    x[lag_abs:, sel_ind],
                    np.zeros((lag_abs, pad))
                ])
    
    return x


def gen_subave(x: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """
    Generate sub-averages from trials.
    
    Parameters
    ----------
    x : np.ndarray
        Signal matrix (n_timepoints, n_trials)
    ind : np.ndarray
        Indices defining sub-average boundaries
    
    Returns
    -------
    y : np.ndarray
        Sub-averaged signals (n_timepoints, n_subaverages)
    """
    N, M = x.shape
    T = len(ind) - 1
    y = np.zeros((N, T))
    
    for i in range(T - 1):
        y[:, i] = np.mean(x[:, ind[i]:ind[i+1]], axis=1)
    
    y[:, T-1] = np.mean(x[:, ind[T-1]:], axis=1)
    
    return y


def woody_stochastic(x: np.ndarray,
                     n_runs: int = 1000,
                     max_lag_init: int = 50,
                     tol: float = 0.1,
                     max_it: int = 100,
                     est_mthd: str = 'woody',
                     xcorr_mthd: str = 'unbiased',
                     max_shift: int = 50,
                     verbose: bool = False,
                     rng: Optional[np.random.Generator] = None) -> Tuple:
    """
    Stochastic Woody filter with random initialization.
    
    Runs Woody filter multiple times with different random initial lag offsets
    to avoid local minima. Returns the solution with highest correlation coefficient.
    
    Parameters
    ----------
    x : np.ndarray
        Signal matrix (n_timepoints, n_trials)
    n_runs : int
        Number of random initializations to try (default=1000)
    max_lag_init : int
        Maximum random lag offset for initialization (default=50 samples)
    tol : float
        Convergence tolerance (default=0.1)
    max_it : int
        Maximum iterations per run (default=100)
    est_mthd : str
        Estimation method ('woody' or 'thornton')
    xcorr_mthd : str
        Cross-correlation method ('biased' or 'unbiased')
    max_shift : int
        Maximum shift to search for alignment (default=50)
    verbose : bool
        Print progress (default=False)
    rng : np.random.Generator, optional
        Random number generator used for the initial lag offsets. Pass a seeded
        generator (``np.random.default_rng(seed)``) for reproducible results.
    
    Returns
    -------
    best_avg : np.ndarray
        Best aligned average waveform
    best_lags : np.ndarray
        Best lag estimates for each trial
    best_corr : float
        Correlation coefficient of best solution
    """
    N, M = x.shape
    rng = np.random.default_rng() if rng is None else rng
    
    best_corr = -np.inf
    best_avg = None
    best_lags = None
    
    sig_x = np.sqrt(np.sum(x**2, axis=0))
    
    for run in range(n_runs):
        init_lags = rng.integers(-max_lag_init, max_lag_init + 1, size=M)
        
        x_shifted = np.zeros_like(x)
        for i in range(M):
            lag = init_lags[i]
            if lag > 0:
                x_shifted[:N-lag, i] = x[lag:, i]
            elif lag < 0:
                x_shifted[-lag:, i] = x[:N+lag, i]
            else:
                x_shifted[:, i] = x[:, i]
        
        try:
            aligned_avg, detected_lags = woody(
                x_shifted,
                tol=tol,
                max_it=max_it,
                est_mthd=est_mthd,
                xcorr_mthd=xcorr_mthd,
                max_shift=max_shift
            )
            
            if np.sqrt(np.dot(aligned_avg, aligned_avg)) > 0 and np.all(sig_x > 0):
                x_aligned = np.zeros_like(x)
                for i in range(M):
                    total_lag = init_lags[i] + detected_lags[i]
                    if total_lag > 0 and total_lag < N:
                        x_aligned[:N-total_lag, i] = x[total_lag:, i]
                    elif total_lag < 0 and -total_lag < N:
                        x_aligned[-total_lag:, i] = x[:N+total_lag, i]
                    else:
                        x_aligned[:, i] = x[:, i]
                
                p_vec = np.dot(aligned_avg, x_aligned) / (np.sqrt(np.dot(aligned_avg, aligned_avg)) * sig_x)
                p = np.sum(p_vec) / M
            else:
                p = -np.inf
            
            total_lags = init_lags + detected_lags
            
            if p > best_corr:
                best_corr = p
                best_avg = aligned_avg.copy()
                best_lags = total_lags.copy()
                
                if verbose and run % 100 == 0:
                    print(f"Run {run}/{n_runs}: New best corr = {best_corr:.4f}")
                
        except (ValueError, IndexError) as e:
            # A random initialisation can occasionally push trials far enough
            # that the alignment fails; skip that run and keep searching.
            if verbose:
                print(f"Run {run} failed: {e}")
            continue
    
    if best_lags is None:
        raise RuntimeError("woody_stochastic: all runs failed to produce an alignment.")
    
    return best_avg, best_lags, best_corr