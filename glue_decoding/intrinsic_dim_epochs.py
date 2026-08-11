#!/usr/bin/env python3
"""
intrinsic_dim_epochs.py

Computes intrinsic dimensionality (participation ratio) for two stim-locked
epochs per ROI per band, then plots a bar/dot summary across subjects.

Epochs (stim-locked only):
    stim  : [0.0,  0.2] s  (stimulus presentation)
    delay : [0.2,  1.7] s  (delay / maintenance period)

For each (subject, band, ROI, epoch):
    1. Slice the epoch window from the stim-locked trial data.
    2. Average across the time axis  → (n_trials, n_sources).
    3. Z-score across sources, compute covariance, apply participation ratio.

This gives one scalar PR per (subject, band, ROI, epoch).  Cross-subject
mean ± SEM is then plotted as a grouped bar chart:
    rows = bands (theta / alpha / beta / lowgamma / highgamma)
    cols = ROIs  (visual / parietal / frontal)
    within each panel: two bars -- Stim (epoch 1) and Delay (epoch 2)

Parallelism:
    Subjects are loaded in parallel (one worker per subject) via joblib,
    the same way plot_timeseries.py works.  Each worker only reads the
    small per-ROI .npz caches, so IO is minimal.

Usage:
    python intrinsic_dim_epochs.py [--voxRes 8mm]
                                   [--subjects 1 2 ...]
                                   [--rois visual parietal frontal]
                                   [--bands theta alpha beta lowgamma highgamma]
                                   [--outdir <path>]
                                   [--n_jobs 8]
                                   [--var_threshold 0.90]
"""

import os

os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from joblib import Parallel, delayed

from constants import AMP_ONLY_BANDS, SUBJECT_LIST, ROI_NAMES, get_bids_root
from io_g04 import load_g04_band

# -- Epoch definitions (stim-locked) -----------------------------------------

EPOCHS = {
    'stim':  (0.0,  0.2),   # stimulus presentation
    'delay': (0.2,  1.7),   # delay / maintenance
}
EPOCH_ORDER  = ['stim', 'delay']
EPOCH_LABELS = {'stim': 'Stim\n(0-0.2 s)', 'delay': 'Delay\n(0.2-1.7 s)'}

# -- Visual design ------------------------------------------------------------

_BG        = '#0d0d0d'
_FG        = '#e0e0e0'
_GRID      = '#1e1e1e'

EPOCH_COLOURS = {
    'stim':  '#7EB8F7',   # sky blue
    'delay': '#F4A261',   # warm orange
}

AMP_BAND_ORDER = ['theta', 'alpha', 'beta', 'lowgamma', 'highgamma']
BAND_LABELS = {
    'theta':     'Theta\n(4-8 Hz)',
    'alpha':     'Alpha\n(8-12 Hz)',
    'beta':      'Beta\n(13-30 Hz)',
    'lowgamma':  'Low gamma\n(30-80 Hz)',
    'highgamma': 'High gamma\n(80-150 Hz)',
}

# -- Dimensionality estimator -------------------------------------------------

def participation_ratio(X):
    """PR from z-scored (n_samples, n_sources) matrix."""
    C   = np.cov(X, rowvar=False)
    lam = np.maximum(np.linalg.eigvalsh(C), 0.0)
    s1  = lam.sum()
    if s1 < 1e-30:
        return 1.0
    return float(s1**2 / (lam**2).sum())


def epoch_pr(data, time_vector, t_start, t_end, var_threshold=0.90):
    """
    Compute one PR scalar for the epoch [t_start, t_end].

    data : (n_trials, n_times, n_sources)
    Returns (pr, n_pcs) or (None, None) if the epoch window is empty.

    Workflow: average over epoch timepoints -> (n_trials, n_sources),
    z-score across sources, covariance, eigendecomposition.
    """
    mask = (time_vector >= t_start) & (time_vector <= t_end)
    if not mask.any():
        return None, None

    # Average over epoch timepoints -> (n_trials, n_sources)
    X = data[:, mask, :].mean(axis=1)

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-10] = 1.0
    X_z = (X - mu) / sd

    pr = participation_ratio(X_z)

    # n_pcs for >= var_threshold
    C   = np.cov(X_z, rowvar=False)
    lam = np.sort(np.maximum(np.linalg.eigvalsh(C), 0.0))[::-1]
    s   = lam.sum()
    if s < 1e-30:
        npcs = 1
    else:
        cum  = np.cumsum(lam) / s
        hits = np.where(cum >= var_threshold)[0]
        npcs = int(hits[0] + 1) if hits.size > 0 else len(lam)

    return pr, npcs


# -- Per-subject loader (runs in parallel) ------------------------------------

def load_subject_epoch_pr(subjID, voxRes, bids_root, rois_all, bands,
                           var_threshold=0.90):
    """
    For one subject, compute PR for each (band, roi, epoch).

    Returns dict: result[band][roi][epoch] = (pr, npcs) or (None, None).
    """
    lockType = 'stim'   # epoch analysis is stim-locked only
    result = {}

    for band in bands:
        result[band] = {}
        for roi in rois_all:
            result[band][roi] = {}
            try:
                g04 = load_g04_band(subjID, lockType, band, voxRes, bids_root,
                                     want_phase=False, roi=roi)
            except (FileNotFoundError, ValueError) as e:
                print(f'  sub-{subjID:02d} {band} roi={roi}: {e}', flush=True)
                for ep in EPOCH_ORDER:
                    result[band][roi][ep] = (None, None)
                continue

            amp = g04['amp']            # (n_trials, n_times, n_sources)
            tv  = g04['time_vector']    # (n_times,)

            for ep_name, (t0, t1) in EPOCHS.items():
                pr, npcs = epoch_pr(amp, tv, t0, t1, var_threshold)
                result[band][roi][ep_name] = (pr, npcs)

    print(f'  sub-{subjID:02d} done', flush=True)
    return result


# -- Cross-subject aggregation ------------------------------------------------

def aggregate_epoch(all_results, band, roi, epoch, metric='pr'):
    """Returns (mean, sem, n_subj) or (None, None, 0)."""
    idx  = 0 if metric == 'pr' else 1
    vals = []
    for r in all_results:
        if r is None:
            continue
        v = r.get(band, {}).get(roi, {}).get(epoch, (None, None))[idx]
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None, 0
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std() / np.sqrt(len(arr))), len(arr)


# -- Plotting -----------------------------------------------------------------

def _apply_black_style(fig, axes_flat):
    fig.patch.set_facecolor(_BG)
    for ax in axes_flat:
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_FG, which='both', labelsize=11)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.grid(True, axis='y', color=_GRID, linewidth=0.5,
                linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)


def plot_epoch_figure(all_results, rois_all, bands, voxRes, outdir,
                      metric='pr', metric_label='Participation Ratio',
                      var_threshold=0.90):
    """
    Bar/dot plot: rows = bands, cols = ROIs.
    Within each panel: two grouped bars (stim vs delay) with individual
    subject dots jittered on top.
    """
    n_rows = len(bands)
    n_cols = len(rois_all)
    fig_w  = max(4.0 * n_cols, 10)
    fig_h  = max(3.0 * n_rows, 8)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              sharex=False, sharey=False,
                              squeeze=False)
    _apply_black_style(fig, axes.flatten())

    bar_w   = 0.35
    offsets = {'stim': -bar_w / 2 - 0.02, 'delay': bar_w / 2 + 0.02}
    x_pos   = np.array([0.0])

    # First pass: per-row (band) y-limits for consistent scale across ROI cols
    row_ylim = {}
    for band in bands:
        row_max = 0.0
        for roi in rois_all:
            for ep in EPOCH_ORDER:
                mean, sem, n = aggregate_epoch(all_results, band, roi, ep, metric)
                if mean is not None:
                    row_max = max(row_max, mean + sem)
        row_ylim[band] = (0.0, row_max * 1.20) if row_max > 0 else None

    rng = np.random.default_rng(42)

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois_all):
            ax = axes[r_idx, c_idx]

            if row_ylim[band] is not None:
                ax.set_ylim(*row_ylim[band])

            has_data = False
            for ep in EPOCH_ORDER:
                mean, sem, n_subj = aggregate_epoch(
                    all_results, band, roi, ep, metric)
                if mean is None:
                    continue
                has_data = True

                colour = EPOCH_COLOURS[ep]
                xc     = x_pos + offsets[ep]

                # Bar
                ax.bar(xc, mean, bar_w,
                       color=colour, alpha=0.75, zorder=3,
                       label=EPOCH_LABELS[ep])
                # Error bar
                ax.errorbar(xc, mean, yerr=sem,
                            fmt='none', color=_FG,
                            linewidth=1.5, capsize=5, capthick=1.5, zorder=4)

                # Individual-subject dots
                idx_m  = 0 if metric == 'pr' else 1
                s_vals = [
                    r.get(band, {}).get(roi, {}).get(ep, (None, None))[idx_m]
                    for r in all_results if r is not None
                ]
                s_vals = [v for v in s_vals if v is not None]
                jitter = rng.uniform(-0.07, 0.07, len(s_vals))
                ax.scatter(xc + jitter, s_vals,
                           color=colour, s=22, alpha=0.55,
                           linewidths=0, zorder=5)

            if not has_data:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center',
                        color='#555555', fontsize=9)
                continue

            ax.set_xticks([])
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))

            if r_idx == 0:
                roi_lbl = roi.capitalize() if roi != 'whole' else 'Whole brain'
                n_max   = max(
                    aggregate_epoch(all_results, band, roi, ep, metric)[2]
                    for ep in EPOCH_ORDER
                )
                ax.set_title(f'{roi_lbl}  (n={n_max})',
                             fontsize=14, fontweight='bold', pad=6)

            if c_idx == 0:
                ax.set_ylabel(metric_label, fontsize=11)
                ax.annotate(BAND_LABELS.get(band, band),
                             xy=(-0.36, 0.5), xycoords='axes fraction',
                             fontsize=12, color=_FG,
                             ha='right', va='center',
                             rotation=90, fontweight='bold')

            # Legend in top-right panel only
            if r_idx == 0 and c_idx == n_cols - 1:
                handles = [
                    plt.Rectangle((0, 0), 1, 1,
                                  color=EPOCH_COLOURS[ep], alpha=0.75)
                    for ep in EPOCH_ORDER
                ]
                labels = [EPOCH_LABELS[ep] for ep in EPOCH_ORDER]
                leg = ax.legend(handles, labels, fontsize=10,
                                loc='upper right', framealpha=0.2,
                                edgecolor='#444444', labelcolor=_FG)
                leg.get_frame().set_facecolor('#1a1a1a')

    metric_str = 'participation_ratio' if metric == 'pr' else 'n_pcs'
    fig.suptitle(
        f'Intrinsic Dimensionality by Epoch  |  Stim-locked  |  {voxRes}  |  '
        f'metric: {metric_str}',
        color=_FG, fontsize=17, fontweight='bold', y=1.01
    )
    fig.tight_layout(rect=[0.07, 0, 1, 1])

    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir,
                          f'intrinsic_dim_epochs_{metric_str}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


# -- Main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Epoch-level intrinsic dimensionality: stim vs delay.')
    parser.add_argument('--voxRes',        default='8mm')
    parser.add_argument('--subjects',      nargs='+', type=int,
                        default=SUBJECT_LIST)
    parser.add_argument('--rois',          nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--bands',         nargs='+', default=AMP_BAND_ORDER,
                        help='Bands to analyse (default: theta alpha beta '
                             'lowgamma highgamma).')
    parser.add_argument('--outdir',        default=None)
    parser.add_argument('--n_jobs',        type=int, default=None,
                        help='Parallel workers (subjects). '
                             'Default: len(--subjects).')
    parser.add_argument('--var_threshold', type=float, default=0.90)
    args = parser.parse_args()

    if args.n_jobs is None:
        args.n_jobs = len(args.subjects)
    n_jobs = max(1, args.n_jobs)

    bids_root = get_bids_root()
    outdir    = args.outdir or os.path.join(
        bids_root, 'derivatives', 'glueDecoding', 'intrinsicDim', 'epochs')

    rois_all = list(args.rois)

    print(f'intrinsic_dim_epochs | voxRes={args.voxRes} | '
          f'subjects={args.subjects} | rois={rois_all} | '
          f'bands={args.bands} | n_jobs={n_jobs}')
    print(f'Epochs: {EPOCHS}')

    all_results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=5)(
        delayed(load_subject_epoch_pr)(
            subjID, args.voxRes, bids_root, rois_all,
            args.bands, args.var_threshold
        )
        for subjID in args.subjects
    )

    # Participation ratio plot
    plot_epoch_figure(all_results, rois_all, args.bands, args.voxRes, outdir,
                      metric='pr', metric_label='Participation Ratio',
                      var_threshold=args.var_threshold)

    # n_pcs plot
    pct = int(args.var_threshold * 100)
    plot_epoch_figure(all_results, rois_all, args.bands, args.voxRes, outdir,
                      metric='npcs', metric_label=f'# PCs (>={pct}% var)',
                      var_threshold=args.var_threshold)

    print('\nDone.')


if __name__ == '__main__':
    main()
