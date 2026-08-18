#!/usr/bin/env python3
"""
plot_intrinsic_dim_pooled_epochs.py

Bar-plot summary of intrinsic_dim_pooled_epochs.py's output: for each of the
three pooled per-location measures (participation ratio, normalized total
variation, and the between-subject variance-share diagnostic), one figure
of rows=bands x cols=ROIs, four bars per panel (the task epochs
fixation/stimulus/early_delay/late_delay).

Each cell in the source CSV is already a single POOLED-across-subjects
estimate per (band, roi, epoch, location) -- there is no subject axis left
to average over. The bar height/error bar here is instead the mean +/- SEM
across the 10 stimulus-location manifolds, which is the direct analogue of
the original "average across subjects with error bars" request now that
subjects have been pooled rather than kept separate: the unit being
averaged (and whose spread the error bar reflects) is locations instead of
subjects.

between_subj_share is the diagnostic that says how much to trust the
pooled PR/NTV numbers for a given cell: it's the fraction of each
location's pooled sum-of-squares explained by between-subject mean
differences rather than genuine within-subject trial-to-trial variability
(see intrinsic_dim_pooled_epochs.py's docstring). Low -> pooling is close
to "more trials of the same manifold" and PR/NTV are trustworthy. High ->
the pooled estimate is substantially reflecting which subjects were
pooled, not single-manifold geometry.

One figure triple (PR, NTV, between_subj_share) is produced PER CONDITION
found in the CSV ('ampOnly', 'ampPhase'). Bands default to theta/alpha/beta
for both conditions; panels are further limited to whichever bands/ROIs
actually have rows for that condition, so there are no empty panels.

Usage:
    python plot_intrinsic_dim_pooled_epochs.py [--voxRes 8mm]
        [--bands theta alpha beta] [--rois visual parietal frontal]
        [--conditions ampOnly ampPhase] [--outdir <path>]
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

from constants import get_bids_root
from visual_geometry_epochs_cell import EPOCH_ORDER
from intrinsic_dim_pooled_epochs import output_csv_path

_BG        = '#000000'
_FG        = '#e0e0e0'
_GRID      = '#1c1c1c'

# Font sizes -- same scale as plot_circular_tgm.py / plot_visual_geometry_*.py,
# so every figure in this project reads consistently: big, bold, legible.
FS_SUPTITLE   = 18
FS_PANEL_TTL  = 14
FS_AXIS_LABEL = 13
FS_ROW_LABEL  = 14
FS_TICK       = 10

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
    'pr':     dict(col='pr', label='Participation ratio'),
    'ntv':    dict(col='ntv', label='Normalized total variation'),
    'bshare': dict(col='between_subj_share', label='Between-subject variance share'),
}

COND_LABELS = {'ampOnly': 'Amplitude', 'ampPhase': 'Amplitude + Phase'}


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


def aggregate(df, band, roi, condition, epoch, col):
    """Mean +/- SEM across the 10 locations. Returns (mean, sem, n_locs)."""
    vals = df[(df.band == band) & (df.roi == roi) & (df.condition == condition) &
              (df.epoch == epoch)][col].values
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None, None, 0
    mean = float(np.mean(vals))
    sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
    return mean, sem, vals.size


def style_ax(ax):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, which='both', labelsize=FS_TICK)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    ax.grid(True, axis='y', color=_GRID, linewidth=0.5, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)


def plot_metric_figure(df, metric, condition, bands, rois, voxRes, outdir):
    spec = METRICS[metric]
    n_r, n_c = len(bands), len(rois)
    # Wider/taller than the original 3.4x2.6 per panel: the bigger fonts below
    # need the room, and the rotated epoch tick labels want horizontal space.
    fig, axes = plt.subplots(n_r, n_c, figsize=(4.0 * n_c + 1.8, 3.0 * n_r + 1.2),
                              facecolor=_BG, squeeze=False)

    for r, band in enumerate(bands):
        for c, roi in enumerate(rois):
            ax = axes[r][c]
            style_ax(ax)
            shades = epoch_shades(roi)
            means, sems, ns = [], [], []
            for ep in EPOCH_ORDER:
                m, se, n = aggregate(df, band, roi, condition, ep, spec['col'])
                means.append(m if m is not None else np.nan)
                sems.append(se if se is not None else 0.0)
                ns.append(n)

            x = np.arange(len(EPOCH_ORDER))
            ax.bar(x, means, yerr=sems, color=shades, edgecolor=_FG,
                   linewidth=0.6, capsize=4,
                   error_kw=dict(ecolor=_FG, elinewidth=1.2, capthick=1.2))
            if metric == 'bshare':
                ax.set_ylim(0, 1)
                ax.axhline(0.5, color='#888888', linewidth=0.8, linestyle=':', zorder=1)
            ax.set_xticks(x)
            ax.set_xticklabels([EPOCH_LABELS[e] for e in EPOCH_ORDER],
                                rotation=30, ha='right', fontsize=FS_TICK)

            if r == 0:
                ax.set_title(roi.capitalize(), fontsize=FS_PANEL_TTL,
                             color=ROI_COLOURS.get(roi, _FG), fontweight='bold', pad=8)
            if c == 0:
                # No y-axis label: the suptitle already names the metric (one
                # metric per figure), and spelling it out again per row collided
                # with the band label for the longer names.
                ax.text(-0.22, 0.5, BAND_LABELS.get(band, band), transform=ax.transAxes,
                        fontsize=FS_ROW_LABEL, color=_FG, ha='right', va='center',
                        rotation=90, fontweight='bold')

            n_str = '/'.join(str(n) for n in ns)
            ax.text(0.98, 0.98, f'n_loc={n_str}', transform=ax.transAxes,
                    fontsize=8, color='#888888', ha='right', va='top')

    fig.suptitle(f'{spec["label"]}  |  {COND_LABELS.get(condition, condition)}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0.05, 0, 1, 1])

    fp = os.path.join(
        outdir, f'group_task-mgs_intrinsicDimPooledEpochs_{metric}_{condition}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'],
                     help='Default theta/alpha/beta for both ampOnly and ampPhase; '
                          'pass lowgamma/highgamma explicitly to include them (they '
                          'exist in the CSV for ampOnly only -- no saved phase).')
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
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

    present_conditions = set(df.condition.unique())
    for condition in args.conditions:
        if condition not in present_conditions:
            print(f'Skipping {condition}: no rows in {in_csv}')
            continue
        cdf = df[df.condition == condition]
        # Restrict to bands/ROIs actually present for this condition (ampPhase
        # is typically theta/alpha/beta x visual only) so panels aren't empty.
        bands = [b for b in args.bands if b in set(cdf.band.unique())]
        rois = [r for r in args.rois if r in set(cdf.roi.unique())]
        if not bands or not rois:
            print(f'Skipping {condition}: no matching bands/rois in {in_csv}')
            continue
        for metric in METRICS:
            plot_metric_figure(cdf, metric, condition, bands, rois, args.voxRes, outdir)


if __name__ == '__main__':
    main()
