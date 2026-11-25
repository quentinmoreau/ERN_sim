"""Figures shared by the simulation notebooks."""

import matplotlib.pyplot as plt
import numpy as np


def plot_trials(times, trials, title, vlim=15, trace_alpha=0.3):
    """Single-trial traces with grand average (top) and trial-by-time image (bottom)."""
    ms = times * 1000
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.03], hspace=0.15, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[1, 1])

    ax1.plot(ms, trials.T, 'gray', alpha=trace_alpha, linewidth=0.5)
    ax1.plot(ms, trials.mean(axis=0), 'k', linewidth=2)
    ax1.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(title)
    ax1.tick_params(labelbottom=False)

    im = ax2.imshow(trials, aspect='auto', origin='lower',
                    extent=[ms[0], ms[-1], 1, len(trials)],
                    cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax2.axvline(0, color='white', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Trial #')
    fig.colorbar(im, cax=cax, label='Amplitude (µV)')
    return fig


def plot_tf_decomposition(times, freqs, trials, tfr_total, tfr_evoked, prefix='',
                          power_vmax=140000, induced_vmax=30000):
    """Grand average ± SD, then total, evoked and induced (total - evoked) power."""
    ms = times * 1000
    extent = [ms[0], ms[-1], freqs[0], freqs[-1]]
    avg, sd = trials.mean(axis=0), trials.std(axis=0)

    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(4, 2, width_ratios=[1, 0.025], hspace=0.3, wspace=0.05)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[i, 0], sharex=axes[0]) for i in range(1, 4)]
    caxes = [fig.add_subplot(gs[i, 1]) for i in range(1, 4)]
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[0].plot(ms, avg, 'k', linewidth=2)
    axes[0].fill_between(ms, avg - sd, avg + sd, color='gray', alpha=0.3)
    axes[0].axvline(0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].set_title(f'{prefix}Grand average (n={len(trials)})')

    panels = [(tfr_total, 'Total power', 'viridis', power_vmax, 'white'),
              (tfr_evoked, 'Evoked power', 'viridis', power_vmax, 'white'),
              (tfr_total - tfr_evoked, 'Induced power (total - evoked)', 'RdBu_r',
               induced_vmax, 'black')]
    for ax, cax, (tfr, label, cmap, vmax, line) in zip(axes[1:], caxes, panels):
        im = ax.imshow(tfr, aspect='auto', origin='lower', extent=extent,
                       cmap=cmap, vmin=0, vmax=vmax)
        ax.axvline(0, color=line, linestyle='--', linewidth=2)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(f'{prefix}{label}')
        fig.colorbar(im, cax=cax, label='Power (µV²)')
    axes[-1].set_xlabel('Time (ms)')
    return fig


def plot_residual_tfr(times, freqs, residuals, tfr, title, vmax=30000):
    """Residual grand average ± SD (top) and mean single-trial power of the residuals."""
    ms = times * 1000
    avg, sd = residuals.mean(axis=0), residuals.std(axis=0)
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.025], hspace=0.3, wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cax = fig.add_subplot(gs[1, 1])
    ax1.tick_params(labelbottom=False)
    ax1.plot(ms, avg, 'k', linewidth=2)
    ax1.fill_between(ms, avg - sd, avg + sd, color='gray', alpha=0.3)
    ax1.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.set_ylabel('Amplitude (µV)')
    ax1.set_title(f'{title}: residuals')

    im = ax2.imshow(tfr, aspect='auto', origin='lower',
                    extent=[ms[0], ms[-1], freqs[0], freqs[-1]],
                    cmap='RdBu_r', vmin=0, vmax=vmax)
    ax2.axvline(0, color='black', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title(f'{title}: residual power')
    fig.colorbar(im, cax=cax, label='Power (µV²)')
    return fig


def plot_lag_recovery(true_lags, detected_lags, title=''):
    """Compare Woody-estimated lags with the simulated ground truth (both in samples)."""
    error = detected_lags - true_lags
    lim = np.abs(np.concatenate([true_lags, detected_lags])).max()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].scatter(true_lags, detected_lags, alpha=0.5)
    axes[0, 0].plot([-lim, lim], [-lim, lim], 'r--', label='Perfect recovery')
    axes[0, 0].set_xlabel('True lag (samples)')
    axes[0, 0].set_ylabel('Detected lag (samples)')
    axes[0, 0].set_title(f'{title}True vs detected')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(error, bins=20, edgecolor='black')
    axes[0, 1].axvline(0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Error (samples)')
    axes[0, 1].set_title(f'Error: mean={error.mean():.2f}, SD={error.std():.2f}')

    axes[1, 0].hist(true_lags, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Lag (samples)')
    axes[1, 0].set_title('True lags')

    axes[1, 1].hist(detected_lags, bins=20, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Lag (samples)')
    axes[1, 1].set_title('Detected lags')
    plt.tight_layout()
    print(f'Correlation true vs detected lags: {np.corrcoef(true_lags, detected_lags)[0, 1]:.3f}')
    return fig
