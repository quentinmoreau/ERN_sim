"""
Woody filter for temporal alignment of signals with jitter.

Translation from MATLAB implementation used in:
Bonaiuto et al. (2021) "Laminar dynamics of high amplitude beta bursts in human motor cortex"

Author: Translated to Python
Date: 2026-01-19
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
    >>> # Create synthetic data with jitter
    >>> t = np.linspace(0, 1, 1001)
    >>> x_base = np.sin(2*np.pi*t) + np.sin(4*np.pi*t) + np.sin(8*np.pi*t)
    >>> y_env = np.exp(0.01 * np.concatenate([-np.arange(500, 0, -1), [0], -np.arange(1, 501)]))
    >>> s = x_base * y_env
    >>> 
    >>> # Add random temporal jitter
    >>> n_trials = 100
    >>> S = np.zeros((1001, n_trials))
    >>> for i in range(n_trials):
    >>>     tau = int((np.random.rand() - 0.5) * 160)
    >>>     if tau < 0:
    >>>         S[:, i] = np.concatenate([s[-tau:], np.zeros(-tau-1)])
    >>>     else:
    >>>         S[:, i] = np.concatenate([np.zeros(tau), s[:1001-tau]])
    >>>     S[:, i] += np.random.randn(1001) * 0.1
    >>> 
    >>> # Apply Woody filter with max_shift constraint
    >>> aligned_avg, lags = woody(S, max_shift=100)
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
    p = 0.0  # Initialize as scalar
    conv = True
    run = 0
    
    # Pre-compute signal norms
    sig_x = np.sqrt(np.sum(x**2, axis=0))
    
    # Reference point for lag calculation
    X = correlate(mx, mx, mode='full')
    ref = len(X) // 2
    
    # Store lags for each trial
    lag_data = np.zeros(M, dtype=int)
    
    while conv and (run < max_it):
        z = np.zeros(N)
        w = np.ones(N)
        
        for i in range(M):
            y = x[:, i]
            
            # Cross-correlate trial with template
            xy = correlate(mx, y, mode='full')
            
            # Apply normalization
            if xcorr_mthd == 'unbiased':
                lags = np.arange(-N+1, N)
                norm = N - np.abs(lags)
                xy = xy / norm
            elif xcorr_mthd == 'biased':
                xy = xy / N
            
            # Find lag that maximizes correlation
            if max_shift is not None:
                # Constrain search to ±max_shift samples around center
                search_start = max(0, ref - max_shift)
                search_end = min(len(xy), ref + max_shift + 1)
                search_range = xy[search_start:search_end]
                ind = np.argmax(search_range) + search_start
            else:
                # Search entire range
                ind = np.argmax(xy)
            
            lag = ref - ind
            
            # Update weighted average with shifted trial
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
        
        # Update template
        old_mx = mx.copy()
        mx = z.copy()
        
        # Compute correlation coefficient for convergence check
        p_old = p
        if np.sqrt(np.dot(mx, mx)) > 0 and np.all(sig_x > 0):
            p_vec = np.dot(mx, x) / (np.sqrt(np.dot(mx, mx)) * sig_x)
            p = np.sum(p_vec) / M
        else:
            p = 0
        
        # Check convergence
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
    
    # Generate sub-averages
    for i in range(T - 1):
        y[:, i] = np.mean(x[:, ind[i]:ind[i+1]], axis=1)
    
    # Last sub-average
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
                     return_all: bool = False,
                     verbose: bool = True) -> Tuple:
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
    return_all : bool
        If True, return all runs for analysis (default=False)
    verbose : bool
        Print progress (default=True)
    
    Returns
    -------
    best_avg : np.ndarray
        Best aligned average waveform
    best_lags : np.ndarray
        Best lag estimates for each trial
    best_corr : float
        Correlation coefficient of best solution (p value from convergence)
    all_results : list (optional)
        List of dicts with all runs if return_all=True
    """
    N, M = x.shape
    
    best_corr = -np.inf
    best_avg = None
    best_lags = None
    
    if return_all:
        all_results = []
    
    # Pre-compute signal norms for efficiency
    sig_x = np.sqrt(np.sum(x**2, axis=0))
    
    for run in range(n_runs):
        # Generate random initial lags for each trial
        init_lags = np.random.randint(-max_lag_init, max_lag_init + 1, size=M)
        
        # Apply initial random shifts
        x_shifted = np.zeros_like(x)
        for i in range(M):
            lag = init_lags[i]
            if lag > 0:
                x_shifted[:N-lag, i] = x[lag:, i]
            elif lag < 0:
                x_shifted[-lag:, i] = x[:N+lag, i]
            else:
                x_shifted[:, i] = x[:, i]
        
        # Run Woody filter on shifted data
        try:
            aligned_avg, detected_lags = woody(
                x_shifted,
                tol=tol,
                max_it=max_it,
                est_mthd=est_mthd,
                xcorr_mthd=xcorr_mthd,
                max_shift=max_shift
            )
            
            # Compute final correlation coefficient (p value)
            if np.sqrt(np.dot(aligned_avg, aligned_avg)) > 0 and np.all(sig_x > 0):
                # Reconstruct aligned trials to compute correlation
                x_aligned = np.zeros_like(x)
                for i in range(M):
                    total_lag = init_lags[i] + detected_lags[i]
                    if total_lag > 0 and total_lag < N:
                        x_aligned[:N-total_lag, i] = x[total_lag:, i]
                    elif total_lag < 0 and -total_lag < N:
                        x_aligned[-total_lag:, i] = x[:N+total_lag, i]
                    else:
                        x_aligned[:, i] = x[:, i]
                
                # Compute correlation coefficient
                p_vec = np.dot(aligned_avg, x_aligned) / (np.sqrt(np.dot(aligned_avg, aligned_avg)) * sig_x)
                p = np.sum(p_vec) / M
            else:
                p = -np.inf
            
            # Combine initial and detected lags
            total_lags = init_lags + detected_lags
            
            # Store if best
            if p > best_corr:
                best_corr = p
                best_avg = aligned_avg.copy()
                best_lags = total_lags.copy()
                
                if verbose and run % 100 == 0:
                    print(f"Run {run}/{n_runs}: New best corr = {best_corr:.4f}")
            
            if return_all:
                all_results.append({
                    'run': run,
                    'init_lags': init_lags.copy(),
                    'detected_lags': detected_lags.copy(),
                    'total_lags': total_lags.copy(),
                    'aligned_avg': aligned_avg.copy(),
                    'corr': p
                })
                
        except Exception as e:
            if verbose:
                print(f"Run {run} failed: {e}")
            continue
    
    if verbose:
        print(f"\nBest correlation: {best_corr:.4f}")
        print(f"Lag range: [{best_lags.min()}, {best_lags.max()}] samples")
        print(f"Lag std: {best_lags.std():.2f} samples")
    
    if return_all:
        return best_avg, best_lags, best_corr, all_results
    else:
        return best_avg, best_lags, best_corr


# Example usage and testing
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("Testing Woody filter implementation...")
    
    # Create synthetic signal with jitter
    t = np.linspace(0, 1, 1001)
    x_base = np.sin(2*np.pi*t) + np.sin(4*np.pi*t) + np.sin(8*np.pi*t)
    y_env = np.exp(0.01 * np.concatenate([-np.arange(500, 0, -1), [0], -np.arange(1, 501)]))
    s = x_base * y_env
    
    # Generate trials with random temporal jitter
    np.random.seed(42)
    n_trials = 100
    S = np.zeros((1001, n_trials))
    true_lags = []
    
    for i in range(n_trials):
        tau = int((np.random.rand() - 0.5) * 160)
        true_lags.append(tau)
        
        if tau < 0:
            S[:, i] = np.concatenate([s[-tau:], np.zeros(-tau)])
        elif tau > 0:
            S[:, i] = np.concatenate([np.zeros(tau), s[:1001-tau]])
        else:
            S[:, i] = s.copy()
        
        # Add noise
        if i < 50:
            S[:, i] += np.random.randn(1001) * 0.01
        else:
            S[:, i] += np.random.randn(1001) * 0.1
    
    # Apply Woody filter with max_shift
    print("Applying Woody filter with max_shift=100...")
    wood_avg, est_lags = woody(S, tol=0.1, max_it=100, est_mthd='woody', 
                                xcorr_mthd='unbiased', max_shift=100)
    
    # Apply Thornton method
    print("Applying Thornton method...")
    thor_avg, thor_lags = woody(S, tol=0.1, max_it=100, est_mthd='thornton', 
                                 xcorr_mthd='biased', max_shift=100)
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Original signal and jittered trials
    axes[0].plot(s, 'b', linewidth=2, label='Original signal')
    axes[0].plot(S[:, ::5], 'r', alpha=0.3)  # Plot every 5th trial
    axes[0].set_title('Original Signal and Jittered Trials')
    axes[0].legend()
    axes[0].grid(True)
    
    # Compare averaging methods
    axes[1].plot(s, 'b', linewidth=2, label='Original signal')
    axes[1].plot(np.mean(S, axis=1), 'r', label='Simple average')
    axes[1].plot(wood_avg, 'g', label='Woody average')
    axes[1].plot(thor_avg, 'k', label='Thornton average')
    axes[1].set_title('Comparison of Averaging Methods')
    axes[1].legend()
    axes[1].grid(True)
    
    # Estimated vs true lags
    axes[2].scatter(range(n_trials), true_lags, alpha=0.5, label='True lags')
    axes[2].scatter(range(n_trials), est_lags, alpha=0.5, label='Estimated lags')
    axes[2].set_xlabel('Trial number')
    axes[2].set_ylabel('Lag (samples)')
    axes[2].set_title('True vs Estimated Lags')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/claude/woody_filter_test.png', dpi=150, bbox_inches='tight')
    print("Test plot saved to: /home/claude/woody_filter_test.png")
    
    # Compute correlation with original signal
    corr_simple = np.corrcoef(s, np.mean(S, axis=1))[0, 1]
    corr_woody = np.corrcoef(s, wood_avg)[0, 1]
    corr_thornton = np.corrcoef(s, thor_avg)[0, 1]
    
    print(f"\nCorrelation with original signal:")
    print(f"  Simple average:    {corr_simple:.4f}")
    print(f"  Woody average:     {corr_woody:.4f}")
    print(f"  Thornton average:  {corr_thornton:.4f}")
    
    # Compute lag estimation accuracy
    lag_error = np.abs(np.array(true_lags) - est_lags)
    print(f"\nLag estimation accuracy:")
    print(f"  Mean absolute error: {np.mean(lag_error):.2f} samples")
    print(f"  Std absolute error:  {np.std(lag_error):.2f} samples")
    
    print("\nWoody filter test completed successfully!")