#!/usr/bin/env python3
"""
aggregate_glue_capacity.py

Cross-subject aggregation + plotting for manifold_capacity.py's per-subject
CSVs (glue_capacity_sub-XX_{lockType}_{voxRes}.csv, produced by
run_glue_capacity.sh). For each metric x epoch, builds a bands x rois grid
of grouped bar panels (Real vs Shuffle, mean +/- SEM across subjects, with
individual-subject dots), the same visual convention as
intrinsic_dim_epochs.py's plot_epoch_figure.

Metrics plotted (see glue's ManifoldAnalysisResults / glue_analysis.py):
    capacity, dimension, radius, utility, center_alignment, axis_alignment

Two separate figures per metric (stim vs delay), rather than epoch as a
within-panel grouping, so the primary Real-vs-Shuffle comparison isn't
crowded by a second grouping dimension.

Usage:
    python aggregate_glue_capacity.py [--voxRes 8mm] [--lockType stim]
                                       [--subjects 1 2 ...]
                                       [--bands theta alpha beta lowgamma highgamma]
                                       [--rois visual parietal frontal]
                                       [--epochs stim delay]
                                       [--indir <bids_root>/derivatives/glueDecoding/capacity]
                                       [--outdir <indir>/figures]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from constants import SUBJECT_LIST, ROI_NAMES, get_bids_root

# -- Visual design (mirrors intrinsic_dim_epochs.py / plot_timeseries.py) -----

_BG        = '#000000'
_FG        = '#e0e0e0'
_GRID      = '#1c1c1c'

ROI_COLOURS = {
    'visual':   '#FFC629',
    'parietal': '#A78BFA',
    'frontal':  '#34D399',
    'whole':    '#E76F51',
}

BAND_LABELS = {
    'theta':     'Theta\n(4-8 Hz)',
    'alpha':     'Alpha\n(8-12 Hz)',
    'beta':      'Beta\n(13-30 Hz)',
    'lowgamma':  'Low gamma\n(30-80 Hz)',
    'highgamma': 'High gamma\n(80-150 Hz)',
}

EPOCH_LABELS = {'stim': 'Stim (0-0.2 s)', 'delay': 'Delay (0.2-1.7 s)'}

METRICS = ['capacity', 'dimension', 'radius', 'utility',
           'center_alignment', 'axis_alignment']
METRIC_LABELS = {
    'capacity':          'Capacity',
    'dimension':         'Dimension',
    'radius':            'Radius',
    'utility':           'Utility',
    'center_alignment':  'Center alignment',
    'axis_alignment':    'Axis alignment',
}

STATE_ORDER  = [False, True]   # shuffle column: False=Real, True=Shuffle
STATE_LABELS = {False: 'Real', True: 'Shuffle'}


def state_shades(roi_name):
    """(real_colour, shuffle_colour): full-saturation ROI hue for Real,
    lighter/desaturated variant for Shuffle -- same recipe as
    intrinsic_dim_epochs.py's epoch_shades, just relabeled."""
    base = ROI_COLOURS.get(roi_name, '#ffffff')
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(base))
    real_colour    = base
    shuffle_colour = mcolors.hsv_to_rgb((h, s * 0.4, min(1.0, v * 1.15 + 0.1)))
    return real_colour, shuffle_colour


# -- Loading -------------------------------------------------------------------

def load_all_subjects(subjects, lockType, voxRes, indir):
    """Loads + concatenates every subject's glue_capacity CSV that exists.
    Returns one long DataFrame with subjID/band/roi/epoch/shuffle/seed as
    plain columns (index reset), or an empty DataFrame if none are found."""
    dfs = []
    for subjID in subjects:
        fpath = os.path.join(indir, f'glue_capacity_sub-{subjID:02d}_{lockType}_{voxRes}.csv')
        if not os.path.exists(fpath):
            print(f'  missing: {fpath}')
            continue
        df = pd.read_csv(fpath)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# -- Aggregation -----------------------------------------------------------------

def aggregate_cell(df, band, roi, epoch, shuffle_val, metric):
    """Returns (mean, sem, subject_vals) across subjects for one
    (band, roi, epoch, shuffle) cell, or (None, None, []) if no rows match."""
    sel = df[(df['band'] == band) & (df['roi'] == roi) &
             (df['epoch'] == epoch) & (df['shuffle'] == shuffle_val)]
    vals = sel[metric].dropna().to_numpy()
    if vals.size == 0:
        return None, None, []
    mean = float(vals.mean())
    sem  = float(vals.std() / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, sem, vals.tolist()


# -- Plotting --------------------------------------------------------------------

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
        ax.grid(True, axis='y', color=_GRID, linewidth=0.5, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)


def plot_metric_figure(df, metric, epoch, bands, rois, voxRes, outdir):
    """Bar/dot plot for one metric x epoch: rows = bands, cols = rois.
    Within each panel: Real vs Shuffle grouped bars, mean +/- SEM across
    subjects, individual-subject dots jittered on top."""
    n_rows, n_cols = len(bands), len(rois)
    fig_w = max(4.0 * n_cols, 10)
    fig_h = max(3.0 * n_rows, 8)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              sharex=False, sharey=False, squeeze=False)
    _apply_black_style(fig, axes.flatten())

    bar_w   = 0.35
    offsets = {False: -bar_w / 2 - 0.02, True: bar_w / 2 + 0.02}
    x_pos   = np.array([0.0])
    rng     = np.random.default_rng(42)

    # Per-row (band) y-limits, shared across ROI cols within that row.
    row_ylim = {}
    for band in bands:
        vmin, vmax = np.inf, -np.inf
        for roi in rois:
            for sh in STATE_ORDER:
                mean, sem, _ = aggregate_cell(df, band, roi, epoch, sh, metric)
                if mean is not None:
                    vmin = min(vmin, mean - sem)
                    vmax = max(vmax, mean + sem)
        if np.isfinite(vmin) and np.isfinite(vmax):
            pad = max(1e-6, (vmax - vmin) * 0.20)
            row_ylim[band] = (min(0.0, vmin - pad), vmax + pad)
        else:
            row_ylim[band] = None

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois):
            ax = axes[r_idx, c_idx]
            if row_ylim[band] is not None:
                ax.set_ylim(*row_ylim[band])

            real_colour, shuffle_colour = state_shades(roi)
            state_colour = {False: real_colour, True: shuffle_colour}

            has_data = False
            for sh in STATE_ORDER:
                mean, sem, subj_vals = aggregate_cell(df, band, roi, epoch, sh, metric)
                if mean is None:
                    continue
                has_data = True

                colour = state_colour[sh]
                xc     = x_pos + offsets[sh]

                ax.bar(xc, mean, bar_w, color=colour, alpha=0.75, zorder=3,
                       label=STATE_LABELS[sh])
                ax.errorbar(xc, mean, yerr=sem, fmt='none', color=_FG,
                            linewidth=1.5, capsize=5, capthick=1.5, zorder=4)

                jitter = rng.uniform(-0.07, 0.07, len(subj_vals))
                ax.scatter(xc + jitter, subj_vals, color=colour, s=22,
                           alpha=0.55, linewidths=0, zorder=5)

            if not has_data:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', color='#555555', fontsize=9)
                continue

            ax.set_xticks([])
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))

            if r_idx == 0:
                roi_lbl = roi.capitalize() if roi != 'whole' else 'Whole brain'
                n_max = 0
                for sh in STATE_ORDER:
                    _, _, subj_vals = aggregate_cell(df, band, roi, epoch, sh, metric)
                    n_max = max(n_max, len(subj_vals))
                ax.set_title(f'{roi_lbl}  (n={n_max})', fontsize=14, fontweight='bold', pad=6)

            if c_idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
                ax.annotate(BAND_LABELS.get(band, band),
                            xy=(-0.36, 0.5), xycoords='axes fraction',
                            fontsize=12, color=_FG, ha='right', va='center',
                            rotation=90, fontweight='bold')

            if r_idx == 0 and c_idx == n_cols - 1:
                handles = [plt.Rectangle((0, 0), 1, 1, color=state_colour[sh], alpha=0.75)
                           for sh in STATE_ORDER]
                labels  = [STATE_LABELS[sh] for sh in STATE_ORDER]
                leg = ax.legend(handles, labels, fontsize=10, loc='upper right',
                                 framealpha=0.2, edgecolor='#444444', labelcolor=_FG)
                leg.get_frame().set_facecolor('#1a1a1a')

    fig.suptitle(
        f'Manifold Capacity -- {METRIC_LABELS.get(metric, metric)}  |  '
        f'{EPOCH_LABELS.get(epoch, epoch)}  |  {voxRes}',
        color=_FG, fontsize=17, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0.07, 0, 1, 1])

    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f'glue_capacity_{metric}_{epoch}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


# -- Main ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate + plot glue manifold-capacity results across subjects.')
    parser.add_argument('--voxRes',   default='8mm')
    parser.add_argument('--lockType', default='stim', choices=['stim', 'resp'])
    parser.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    parser.add_argument('--bands',    nargs='+',
                         default=['theta', 'alpha', 'beta', 'lowgamma', 'highgamma'])
    parser.add_argument('--rois',     nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--epochs',   nargs='+', default=['stim', 'delay'])
    parser.add_argument('--metrics',  nargs='+', default=METRICS)
    parser.add_argument('--indir',    default=None,
                         help='Directory containing per-subject glue_capacity CSVs. '
                              'Default: <bids_root>/derivatives/glueDecoding/capacity')
    parser.add_argument('--outdir',   default=None,
                         help='Directory for figures. Default: <indir>/figures')
    args = parser.parse_args()

    bids_root = get_bids_root()
    indir  = args.indir or os.path.join(bids_root, 'derivatives', 'glueDecoding', 'capacity')
    outdir = args.outdir or os.path.join(indir, 'figures')

    print(f'aggregate_glue_capacity | voxRes={args.voxRes} | lockType={args.lockType} | '
          f'subjects={args.subjects} | bands={args.bands} | rois={args.rois} | '
          f'epochs={args.epochs} | metrics={args.metrics}')
    print(f'Loading from: {indir}')

    df = load_all_subjects(args.subjects, args.lockType, args.voxRes, indir)
    if df.empty:
        print('No per-subject CSVs found -- nothing to plot.')
        return

    n_subj_loaded = df['subjID'].nunique()
    print(f'Loaded {n_subj_loaded}/{len(args.subjects)} subjects, {len(df)} rows total.')

    for metric in args.metrics:
        if metric not in df.columns:
            print(f'  SKIP metric {metric!r}: not a column in the loaded results.')
            continue
        for epoch in args.epochs:
            plot_metric_figure(df, metric, epoch, args.bands, args.rois, args.voxRes, outdir)

    print('\nDone.')


if __name__ == '__main__':
    main()
