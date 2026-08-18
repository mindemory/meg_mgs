#!/usr/bin/env python3
"""
plot_intrinsic_dim_pooled_epochs.py

Bar-plot summary of intrinsic_dim_pooled_epochs.py's output: for each of the
two pooled per-location measures (participation ratio, normalized total
variation), one figure of rows=bands x cols=ROIs, four bars per panel (the
task epochs fixation/stimulus/early_delay/late_delay).

Each cell in the source CSV is already a single POOLED-across-subjects
estimate per (band, roi, epoch, location) -- there is no subject axis left
to average over. The bar height/error bar here is instead the mean +/- SEM
across the 10 stimulus-location manifolds, which is the direct analogue of
the original "average across subjects with error bars" request now that
subjects have been pooled rather than kept separate: the unit being
averaged (and whose spread the error bar reflects) is locations instead of
subjects.

Usage:
    python plot_intrinsic_dim_pooled_epochs.py [--voxRes 8mm]
        [--bands theta alpha beta lowgamma highgamma]
        [--rois visual parietal frontal] [--outdir <path>]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from constants import AMP_ONLY_BANDS, get_bids_root
from visual_geometry_epochs_cell import EPOCH_ORDER
from intrinsic_dim_pooled_epochs import output_csv_path

_BG        = '#000000'
_FG        = '#e0e0e0'
_GRID      = '#1c1c1c'

ROI_COLOURS = {
    'visual':   '#FFC629',
    'parietal': '#A78BFA',
    'frontal':  '#34D399',
}

BAND_LABELS = {'theta': 'Theta (4-8 Hz)', 'alpha': 'Alpha (8-12 Hz)',
               'beta': 'Beta (13-30 Hz)', 'lowgamma': 'Low gamma (30-80 Hz)',
               'highgamma': 'High gamma (80-150 Hz)'}
EPOCH_LABELS = {'fixation': 'Fixation', 'stimulus': 'Stimulus',
                 'early_delay': 'Early delay', 'late_delay': 'Late delay'}

METRICS = {
    'pr':  dict(col='pr',  label='Participation ratio (pooled, per location)'),
    'ntv': dict(col='ntv', label='Normalized total variation (pooled, per location)'),
}


def epoch_shades(roi_name, n=4):
    """One shade per epoch, from the ROI's own colour (dark->light)."""
    base = ROI_COLOURS.get(roi_name, '#ffffff')
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(base))
    shades = []
    for i in range(n):
        frac = i / max(n - 1, 1)
        shades.append(mcolors.hsv_to_rgb((h, s * (1.0 - 0.55 * frac),
                                           min(1.0, v * (0.75 + 0.35 * frac)))))
    return shades


def aggregate(df, band, roi, epoch, col):
    """Mean +/- SEM across the 10 locations. Returns (mean, sem, n_locs)."""
    vals = df[(df.band == band) & (df.roi == roi) & (df.epoch == epoch)][col].values
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None, None, 0
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, sem, vals.size


def style_ax(ax):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, which='both', labelsize=11)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, axis='y', color=_GRID, linewidth=0.5, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)


def plot_metric_figure(df, metric, bands, rois, voxRes, outdir):
    spec = METRICS[metric]
    n_r, n_c = len(bands), len(rois)
    fig, axes = plt.subplots(n_r, n_c, figsize=(3.4 * n_c + 1.6, 2.6 * n_r + 1.0),
                              facecolor=_BG, squeeze=False)

    for r, band in enumerate(bands):
        for c, roi in enumerate(rois):
            ax = axes[r][c]
            style_ax(ax)
            shades = epoch_shades(roi)
            means, sems, ns = [], [], []
            for ep in EPOCH_ORDER:
                m, se, n = aggregate(df, band, roi, ep, spec['col'])
                means.append(m if m is not None else np.nan)
                sems.append(se if se is not None else 0.0)
                ns.append(n)

            x = np.arange(len(EPOCH_ORDER))
            ax.bar(x, means, yerr=sems, color=shades, edgecolor=_FG,
                   linewidth=0.6, capsize=4,
                   error_kw=dict(ecolor=_FG, elinewidth=1.2, capthick=1.2))
            ax.set_xticks(x)
            ax.set_xticklabels([EPOCH_LABELS[e] for e in EPOCH_ORDER],
                                rotation=30, ha='right', fontsize=8.5)

            if r == 0:
                ax.set_title(roi.capitalize(), fontsize=12, color=ROI_COLOURS.get(roi, _FG),
                             fontweight='bold', pad=6)
            if c == 0:
                ax.text(-0.42, 0.5, BAND_LABELS.get(band, band), transform=ax.transAxes,
                        fontsize=11, color=_FG, ha='right', va='center',
                        rotation=90, fontweight='bold')

            n_str = '/'.join(str(n) for n in ns)
            ax.text(0.98, 0.98, f'n_loc={n_str}', transform=ax.transAxes,
                    fontsize=6.5, color='#888888', ha='right', va='top')

    metric_str = metric
    fig.suptitle(
        f'{spec["label"]} -- pooled across subjects, {voxRes}',
        color=_FG, fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0.06, 0, 1, 1])

    fp = os.path.join(outdir, f'group_task-mgs_intrinsicDimPooledEpochs_{metric_str}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--bands', nargs='+', default=list(AMP_ONLY_BANDS))
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    bids_root = get_bids_root()
    in_csv = output_csv_path(bids_root, args.voxRes)
    if not os.path.exists(in_csv):
        print(f'Not found: {in_csv} -- run intrinsic_dim_pooled_epochs.py first.')
        sys.exit(1)
    df = pd.read_csv(in_csv)

    outdir = args.outdir or os.path.dirname(in_csv)
    os.makedirs(outdir, exist_ok=True)

    for metric in METRICS:
        plot_metric_figure(df, metric, list(args.bands), list(args.rois), args.voxRes, outdir)


if __name__ == '__main__':
    main()
