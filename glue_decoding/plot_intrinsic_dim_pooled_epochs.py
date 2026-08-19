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

# NTV = sqrt(Tr(Sigma)) / ||mu|| is a RATIO of two things that answer different
# questions, so it is plotted alongside its own numerator and denominator. A
# single NTV value cannot distinguish "manifolds shrank" from "centers moved
# apart" -- both lower it -- and which of the two an area/band uses to support
# decoding is exactly the question NTV was suggested for.
METRICS = {
    'pr':     dict(col='pr', label='Participation ratio'),
    'ntv':    dict(col='ntv', label='Normalized total variation'),
    'size':   dict(col='sqrt_trace', label='Manifold size  $\\sqrt{Tr(\\Sigma)}$'),
    'center': dict(col='mu_norm', label='Center displacement  $\\|\\mu\\|$'),
    'bshare': dict(col='between_subj_share', label='Between-subject variance share'),
}

# Metrics whose numerator/denominator decomposition the mechanism scatter uses.
MECH_SIZE, MECH_CENTER = 'sqrt_trace', 'mu_norm'

REF_EPOCH_DEFAULT = 'fixation'


def add_derived(df):
    """sqrt(Tr(Sigma)) -- NTV's numerator, saved as trace_sigma."""
    df = df.copy()
    df['sqrt_trace'] = np.sqrt(df['trace_sigma'].astype(float).clip(lower=0))
    return df


def relativize(df, ref_epoch, cols):
    """
    Express each value as a ratio to the SAME cell's ref_epoch value, where a
    cell is (band, roi, condition, LOCATION) -- normalising per location rather
    than after averaging, so location-specific scale divides out and the spread
    across locations then reflects consistency of the ratio itself.

    This is not only for interpretability. Tr(Sigma) sums variance over every
    feature, so sqrt(Tr(Sigma)) scales roughly as sqrt(n_features), and the ROIs
    differ a lot in feature count (visual ~597, parietal ~501, frontal ~179 ->
    24.4 / 22.4 / 13.4). Raw NTV and raw manifold size are therefore NOT
    comparable across areas. Taking a ratio within a cell cancels that factor,
    since it is the same feature count in numerator and denominator, which is
    what makes the cross-area comparison licensable at all.
    """
    key = ['band', 'roi', 'condition', 'location']
    ref = (df[df.epoch == ref_epoch][key + cols]
           .rename(columns={c: f'{c}__ref' for c in cols}))
    out = df.merge(ref, on=key, how='left')
    for c in cols:
        denom = out[f'{c}__ref'].astype(float).replace(0.0, np.nan)
        out[c] = out[c].astype(float) / denom
    return out.drop(columns=[f'{c}__ref' for c in cols])

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


def plot_metric_figure(df, metric, condition, bands, rois, voxRes, outdir,
                        ref_epoch=None):
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
            means, sems = [], []
            for ep in EPOCH_ORDER:
                m, se, _ = aggregate(df, band, roi, condition, ep, spec['col'])
                means.append(m if m is not None else np.nan)
                sems.append(se if se is not None else 0.0)

            x = np.arange(len(EPOCH_ORDER))
            ax.bar(x, means, yerr=sems, color=shades, edgecolor=_FG,
                   linewidth=0.6, capsize=4,
                   error_kw=dict(ecolor=_FG, elinewidth=1.2, capthick=1.2))
            if metric == 'bshare' and ref_epoch is None:
                # Reference line kept, hard 0-1 limit dropped: real values sit
                # far below 0.5 and the fixed range flattened them.
                ax.axhline(0.5, color='#888888', linewidth=0.8, linestyle=':', zorder=1)
            if ref_epoch is not None:
                # Baseline is 1.0 by construction; the line is the read-off.
                ax.axhline(1.0, color='#888888', linewidth=1.0, linestyle=':', zorder=1)
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

    rel = '' if ref_epoch is None else f'  (relative to {EPOCH_LABELS.get(ref_epoch, ref_epoch).lower()})'
    fig.suptitle(f'{spec["label"]}{rel}  |  {COND_LABELS.get(condition, condition)}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0.05, 0, 1, 1])

    tag = metric if ref_epoch is None else f'{metric}-rel'
    fp = os.path.join(
        outdir, f'group_task-mgs_intrinsicDimPooledEpochs_{tag}_{condition}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')


BAND_MARKERS = {'theta': 'o', 'alpha': 's', 'beta': '^',
                'lowgamma': 'D', 'highgamma': 'v'}


def mechanism_points(df, condition, band, roi, epoch, ref_epoch):
    """
    (x, y) = mean log2 ratio of center displacement and manifold size against
    ref_epoch, averaged over locations.

    Logs, because NTV is a ratio: log2(NTV_ratio) = y - x exactly, so lines of
    constant NTV change are 45-degree diagonals and the vertical distance from
    y = x IS the NTV improvement. Averaging the LOGS (a geometric mean of the
    ratios) is the right summary for ratio data -- the arithmetic mean of ratios
    would be dominated by locations whose baseline happened to be small.
    """
    sel = df[(df.band == band) & (df.roi == roi) &
             (df.condition == condition) & (df.epoch == epoch)]
    ref = df[(df.band == band) & (df.roi == roi) &
             (df.condition == condition) & (df.epoch == ref_epoch)]
    if sel.empty or ref.empty:
        return None
    m = sel.merge(ref[['location', MECH_SIZE, MECH_CENTER]], on='location',
                  suffixes=('', '_ref'))
    with np.errstate(divide='ignore', invalid='ignore'):
        y = np.log2(m[MECH_SIZE].values / m[f'{MECH_SIZE}_ref'].values)
        x = np.log2(m[MECH_CENTER].values / m[f'{MECH_CENTER}_ref'].values)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return None
    return float(np.mean(x[ok])), float(np.mean(y[ok])), int(ok.sum())


def plot_mechanism_scatter(df, condition, bands, rois, voxRes, outdir,
                            ref_epoch=REF_EPOCH_DEFAULT):
    """
    HOW each area/band changes its geometry, not just how much.

    x = log2 change in center displacement ||mu||  (centers pushed apart ->)
    y = log2 change in manifold size sqrt(Tr(Sigma))  (manifolds grew ^)

    Because log2(NTV ratio) = y - x, the y = x diagonal is "NTV unchanged" and
    everything BELOW it has improved NTV. That splits the two routes a cell can
    take to better separability, which a single NTV number cannot:
      right of centre  -> centers moved apart
      below centre     -> manifolds shrank
      lower-right      -> both
    """
    eps = [e for e in EPOCH_ORDER if e != ref_epoch]
    fig, axes = plt.subplots(1, len(eps), figsize=(5.0 * len(eps) + 1.5, 5.4),
                              facecolor=_BG, squeeze=False)
    any_pt = False
    for c, ep in enumerate(eps):
        ax = axes[0][c]
        style_ax(ax)
        ax.grid(True, axis='both', color=_GRID, linewidth=0.5, linestyle='--', alpha=0.6)
        for roi in rois:
            for band in bands:
                pt = mechanism_points(df, condition, band, roi, ep, ref_epoch)
                if pt is None:
                    continue
                x, y, _ = pt
                any_pt = True
                ax.scatter(x, y, s=170, color=ROI_COLOURS.get(roi, '#ffffff'),
                           marker=BAND_MARKERS.get(band, 'o'), edgecolors='k',
                           linewidths=0.8, zorder=4)
        # NOT a scale restriction to remove: x and y must share one symmetric
        # range or the 45-degree iso-NTV diagonal below stops being 45 degrees,
        # and "distance below the diagonal = NTV improvement" -- the whole point
        # of this panel -- silently breaks. The limit still tracks the data.
        lim = max(0.25, *(abs(v) for v in ax.get_xlim() + ax.get_ylim()))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.axhline(0, color='#666666', lw=1.0, zorder=1)
        ax.axvline(0, color='#666666', lw=1.0, zorder=1)
        # y = x is the iso-NTV line; below it NTV improved.
        ax.plot([-lim, lim], [-lim, lim], color='#4EA1F3', lw=1.2, ls='--', zorder=2)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(EPOCH_LABELS.get(ep, ep), fontsize=FS_PANEL_TTL,
                     color=_FG, fontweight='bold', pad=8)
        ax.set_xlabel(r'log$_2$ change in $\|\mu\|$   (centers apart $\rightarrow$)',
                      fontsize=FS_AXIS_LABEL, fontweight='bold')
        if c == 0:
            ax.set_ylabel(r'log$_2$ change in $\sqrt{Tr(\Sigma)}$   ($\uparrow$ bigger)',
                          fontsize=FS_AXIS_LABEL, fontweight='bold')
    if not any_pt:
        plt.close(fig); print(f'  (no mechanism points for {condition})'); return

    handles = [plt.Line2D([0], [0], marker='o', ls='', markersize=11,
                          markerfacecolor=ROI_COLOURS.get(r, '#fff'),
                          markeredgecolor='k', label=r.capitalize()) for r in rois]
    handles += [plt.Line2D([0], [0], marker=BAND_MARKERS.get(b, 'o'), ls='',
                           markersize=11, markerfacecolor='#cccccc',
                           markeredgecolor='k', label=BAND_LABELS.get(b, b))
                for b in bands]
    leg = fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.99, 0.5),
                     fontsize=11, framealpha=0.25, edgecolor='#444444', labelcolor=_FG)
    leg.get_frame().set_facecolor('#1a1a1a')

    fig.suptitle(f'How geometry changes vs {EPOCH_LABELS.get(ref_epoch, ref_epoch).lower()}'
                 f'  |  {COND_LABELS.get(condition, condition)}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1.02)
    fig.text(0.5, 0.965,
             'dashed = NTV unchanged; below it NTV improved  |  '
             'right = centers apart, down = manifolds shrank',
             ha='center', va='top', color='#aaaaaa', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fp = os.path.join(
        outdir, f'group_task-mgs_intrinsicDimPooledEpochs_mechanism_{condition}_{voxRes}.png')
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
    ap.add_argument('--relative_to', default=REF_EPOCH_DEFAULT,
                     choices=list(EPOCH_ORDER) + ['none'],
                     help="Also emit every metric as a ratio to this epoch's value "
                          "within the same band/roi/condition/location (default "
                          "fixation; 'none' to skip). This is what makes the metrics "
                          "comparable ACROSS AREAS: sqrt(Tr(Sigma)) scales as "
                          "sqrt(n_features) and the ROIs differ 3x in feature count, "
                          "which the ratio cancels.")
    ap.add_argument('--no_mechanism', action='store_true',
                     help='Skip the size-vs-center mechanism scatter.')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    bids_root = get_bids_root()
    in_csv = output_csv_path(bids_root, args.voxRes)
    if not os.path.exists(in_csv):
        print(f'Not found: {in_csv} -- run intrinsic_dim_pooled_epochs.py first.')
        sys.exit(1)
    df = add_derived(pd.read_csv(in_csv))
    ref_epoch = None if args.relative_to == 'none' else args.relative_to

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

        if ref_epoch is not None:
            cols = sorted({m['col'] for m in METRICS.values()})
            rdf = relativize(cdf, ref_epoch, cols)
            for metric in METRICS:
                plot_metric_figure(rdf, metric, condition, bands, rois, args.voxRes,
                                    outdir, ref_epoch=ref_epoch)
            if not args.no_mechanism:
                # Built from the RAW frame: mechanism_points does its own
                # per-location ratio in log space, so handing it already-
                # relativized values would divide by the baseline twice.
                plot_mechanism_scatter(cdf, condition, bands, rois, args.voxRes,
                                        outdir, ref_epoch=ref_epoch)


if __name__ == '__main__':
    main()
