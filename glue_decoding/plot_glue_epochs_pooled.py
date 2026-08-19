#!/usr/bin/env python3
"""
plot_glue_epochs_pooled.py

Figures for manifold_capacity_epochs.py's POOLED glue capacity CSV
(derivatives/glueDecoding/glueEpochsPooled/). Nothing else reads that file --
aggregate_glue_capacity.py targets the older sliding-window `glueFits` output
and its columns do not match.

WHAT THE ERROR BAR IS. These fits are pooled across subjects, so there is no
subject axis left to average over: the replication unit is the BOOTSTRAP DRAW
(independent random subsamples of the pooled points, drawn because the
separability ceiling forces any single fit to use only a fraction of the
pooled trials). So the error bar is a within-cell resampling spread, NOT a
between-subject SEM, and it says how stable the fit is -- not how consistent
subjects are. It will therefore look far tighter than the CCGP/geometry
figures, and the two are not comparable as evidence.

WHAT TO READ AGAINST. glue is run with shuffle=True as well, so each cell has
a label-shuffled counterpart. Capacity/radius/dimension have no meaningful
absolute scale here -- pooled capacity is expected to be LOWER than
per-subject capacity because between-subject variability sits inside the
manifolds -- so REAL vs SHUFFLE is the comparison, exactly as the module
docstring of manifold_capacity_epochs.py says. Both are plotted.

between_subj_share is carried on every row and shown in the panel corner: it
is the fraction of each manifold's spread that is between-subject rather than
within-subject trial variability. It is the check on whether pooling was
legitimate at all. Low values mean the pooled manifolds behave like "more
trials of the same manifold" and the capacity numbers can be read as a shared
code; high values would mean they largely report which subjects were pooled.

Usage:
    python plot_glue_epochs_pooled.py [--voxRes 8mm]
        [--bands theta alpha beta] [--conditions ampOnly ampPhase]
        [--rois visual parietal frontal] [--scheme 10]
        [--csv <path>] [--figdir <path>]
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

from constants import get_bids_root
from visual_geometry_epochs_cell import EPOCH_ORDER

_BG, _FG, _GRID = '#000000', '#e0e0e0', '#1c1c1c'
FS_SUPTITLE, FS_PANEL_TTL, FS_AXIS_LABEL, FS_ROW_LABEL, FS_TICK = 18, 14, 13, 14, 10

ROI_COLOURS = {'visual': '#FFC629', 'parietal': '#A78BFA', 'frontal': '#34D399'}
BAND_LABELS = {'theta': 'Theta (4-8 Hz)', 'alpha': 'Alpha (8-12 Hz)',
               'beta': 'Beta (13-30 Hz)', 'lowgamma': 'Low gamma (30-80 Hz)',
               'highgamma': 'High gamma (80-150 Hz)'}
COND_LABELS = {'ampOnly': 'Amplitude', 'ampPhase': 'Amplitude + Phase'}
EPOCH_LABELS = {'fixation': 'Fixation', 'stimulus': 'Stimulus',
                 'early_delay': 'Early delay', 'late_delay': 'Late delay'}

# Metric columns glue may emit. Only those actually present are plotted, since
# the exact set depends on the glue version rather than on anything here.
CANDIDATE_METRICS = {
    'capacity':  'Manifold capacity',
    'radius':    'Manifold radius',
    'dimension': 'Manifold dimension',
    'center_correlation': 'Center correlation',
}

REAL_C, SHUF_C = '#FFC629', '#888888'


def style_ax(ax):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, which='both', labelsize=FS_TICK)
    ax.xaxis.label.set_color(_FG); ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, lw=0.5, ls='--', alpha=0.6)
    ax.set_axisbelow(True)


def mean_sem(v):
    """Across BOOTSTRAP DRAWS -- see module docstring on what this is not."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    return (float(v.mean()),
            float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
            int(v.size))


def figure_metric(df, metric, label, condition, bands, rois, voxRes, figdir,
                  has_shuffle, sharey=False):
    cdf = df[df.condition == condition]
    bands = [b for b in bands if b in set(cdf.band.unique())]
    rois = [r for r in rois if r in set(cdf.roi.unique())]
    if not bands or not rois:
        print(f'  no {condition} data for {metric}'); return
    n_r, n_c = len(bands), len(rois)
    # Per-panel autoscale by DEFAULT. A shared axis is the honest choice only
    # when panels are comparable, and for capacity they are not: the points per
    # manifold differ across ROIs (the separability ceiling scales with feature
    # count), so a shared axis both invites a comparison that is confounded AND
    # flattens the epoch-to-epoch structure inside each panel behind the
    # between-ROI offset. --sharey restores it when that is what is wanted.
    fig, axes = plt.subplots(n_r, n_c, figsize=(4.4 * n_c + 2.0, 3.2 * n_r + 1.4),
                              facecolor=_BG, squeeze=False, sharey=sharey)
    x = np.arange(len(EPOCH_ORDER))
    for r, band in enumerate(bands):
        for c, roi in enumerate(rois):
            ax = axes[r][c]; style_ax(ax)
            sub = cdf[(cdf.band == band) & (cdf.roi == roi)]
            for is_shuf, col, lab in ((False, REAL_C, 'Real'),
                                       (True, SHUF_C, 'Shuffled')):
                if is_shuf and not has_shuffle:
                    continue
                s2 = sub[sub.shuffle == is_shuf] if has_shuffle else sub
                m, e = [], []
                for ep in EPOCH_ORDER:
                    mm, ee, _ = mean_sem(s2[s2.epoch == ep][metric].values)
                    m.append(mm); e.append(ee)
                ax.errorbar(x, m, yerr=e, color=col, lw=2.0, marker='o', ms=6,
                            capsize=4, zorder=4 if not is_shuf else 3,
                            ls='-' if not is_shuf else '--',
                            label=lab if (r == 0 and c == 0) else None)
            # Pooling-validity check AND points-per-manifold, per cell rather
            # than buried in the CSV. The point count matters for reading
            # CAPACITY ACROSS ROIs: the separability ceiling scales with the
            # feature count, so bigger ROIs get more points per manifold, and
            # more points make manifolds harder to separate -- i.e. capacity
            # falls. A cross-ROI capacity difference at unequal point counts is
            # therefore partly an artefact of ROI size, not of the code. Shown
            # so that comparison is never made by accident.
            bs = sub['between_subj_share'].dropna()
            note = []
            if len(bs):
                note.append(f'btwn-subj {bs.mean():.2f}')
            if 'points_per_manifold' in sub.columns and len(sub):
                pm = sub['points_per_manifold'].dropna()
                if len(pm):
                    note.append(f'{int(pm.median())} pts/man')
            if note:
                # Top-left: the shuffled line sits low and flat, so a
                # bottom-right annotation lands on top of it.
                ax.text(0.03, 0.97, '  '.join(note), transform=ax.transAxes,
                        ha='left', va='top', fontsize=8.5, color='#888888')
            ax.set_xticks(x)
            ax.set_xticklabels([EPOCH_LABELS.get(e, e) for e in EPOCH_ORDER],
                                rotation=30, ha='right', fontsize=FS_TICK)
            if r == 0:
                ax.set_title(roi.capitalize(), fontsize=FS_PANEL_TTL,
                             color=ROI_COLOURS.get(roi, _FG), fontweight='bold', pad=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=FS_AXIS_LABEL, fontweight='bold')
                ax.text(-0.30, 0.5, BAND_LABELS.get(band, band),
                        transform=ax.transAxes, fontsize=FS_ROW_LABEL, color=_FG,
                        ha='right', va='center', rotation=90, fontweight='bold')
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        leg = fig.legend(h, l, loc='center left', bbox_to_anchor=(0.99, 0.5),
                         fontsize=11, framealpha=0.25, edgecolor='#444444',
                         labelcolor=_FG)
        leg.get_frame().set_facecolor('#1a1a1a')
    fig.suptitle(f'{label}  |  {COND_LABELS.get(condition, condition)}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1.005)
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'glue_epochs_pooled_{metric}_{condition}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')


def figure_delta(df, metric, label, condition, bands, rois, voxRes, figdir,
                 sharey=False):
    """
    REAL minus SHUFFLED per epoch -- the quantity that actually bears on "did
    the code change over the delay".

    Shuffling permutes location labels WITHIN an epoch, so it keeps that
    epoch's overall variance and SNR and destroys only the location structure.
    Any drift the shuffled fits show across epochs is therefore drift that has
    nothing to do with location coding (changing signal amplitude, changing
    trial counts), and subtracting it removes exactly that. A real effect
    survives; an epoch-wise amplitude artefact cancels.

    CAVEAT ON THE ERROR BAR, which is the whole reason this figure exists: it
    is still propagated from the BOOTSTRAP spread, and bootstrap draws are
    subsamples of the SAME pooled trials from the SAME 21 subjects. It measures
    how stable the fit is, NOT how much the effect would vary in a new sample
    of subjects, and with thousands of pooled trials it is tiny by
    construction. Treat it as numerical precision, not as inferential
    uncertainty -- a difference many "SEM" wide is not thereby significant.
    """
    cdf = df[df.condition == condition]
    bands = [b for b in bands if b in set(cdf.band.unique())]
    rois = [r for r in rois if r in set(cdf.roi.unique())]
    if not bands or not rois:
        return
    n_r, n_c = len(bands), len(rois)
    fig, axes = plt.subplots(n_r, n_c, figsize=(4.4 * n_c + 2.0, 3.2 * n_r + 1.4),
                              facecolor=_BG, squeeze=False, sharey=sharey)
    x = np.arange(len(EPOCH_ORDER))
    for r, band in enumerate(bands):
        for c, roi in enumerate(rois):
            ax = axes[r][c]; style_ax(ax)
            sub = cdf[(cdf.band == band) & (cdf.roi == roi)]
            m, e = [], []
            for ep in EPOCH_ORDER:
                s2 = sub[sub.epoch == ep]
                rm, re_, _ = mean_sem(s2[s2.shuffle == False][metric].values)
                sm, se_, _ = mean_sem(s2[s2.shuffle == True][metric].values)
                m.append(rm - sm)
                e.append(float(np.hypot(re_, se_)))
            ax.errorbar(x, m, yerr=e, color=REAL_C, lw=2.0, marker='o', ms=6,
                        capsize=4, zorder=4)
            ax.axhline(0.0, color='#888888', lw=1.2, ls=':', zorder=2)
            ax.set_xticks(x)
            ax.set_xticklabels([EPOCH_LABELS.get(ep, ep) for ep in EPOCH_ORDER],
                                rotation=30, ha='right', fontsize=FS_TICK)
            if r == 0:
                ax.set_title(roi.capitalize(), fontsize=FS_PANEL_TTL,
                             color=ROI_COLOURS.get(roi, _FG), fontweight='bold', pad=8)
            if c == 0:
                ax.set_ylabel(f'{label}\nreal $-$ shuffled', fontsize=FS_AXIS_LABEL,
                              fontweight='bold')
                ax.text(-0.30, 0.5, BAND_LABELS.get(band, band),
                        transform=ax.transAxes, fontsize=FS_ROW_LABEL, color=_FG,
                        ha='right', va='center', rotation=90, fontweight='bold')
    fig.suptitle(f'{label}: real $-$ shuffled  |  {COND_LABELS.get(condition, condition)}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1.005)
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'glue_epochs_pooled_{metric}_delta_{condition}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')


def print_summary(df, metrics, has_shuffle):
    print('\n' + '=' * 92)
    print('POOLED GLUE CAPACITY -- mean +/- SEM across BOOTSTRAP DRAWS (not subjects)')
    print('  real vs shuffled is the comparison; pooled absolute values are not')
    print('  comparable to per-subject fits (between-subject variance sits inside')
    print('  the manifolds). between-subj = fraction of manifold spread that is')
    print('  between-subject rather than within-subject trial variability.')
    print('=' * 92)
    hdr = (f"{'band':6s} {'cond':9s} {'roi':9s} {'epoch':12s} "
           + ' '.join(f'{m[:9]:>21s}' for m in metrics) + f" {'btwn':>6s}")
    print(hdr); print('-' * len(hdr))
    for (band, cond, roi), g in df.groupby(['band', 'condition', 'roi']):
        for ep in EPOCH_ORDER:
            s = g[g.epoch == ep]
            if s.empty:
                continue
            cells = []
            for m in metrics:
                real = s[s.shuffle == False][m] if has_shuffle else s[m]
                shuf = s[s.shuffle == True][m] if has_shuffle else pd.Series(dtype=float)
                rm, re_, _ = mean_sem(real.values)
                sm, _, _ = mean_sem(shuf.values) if len(shuf) else (np.nan, 0, 0)
                cells.append(f'{rm:.4f}+/-{re_:.4f}|{sm:.4f}'
                             if np.isfinite(sm) else f'{rm:.4f}+/-{re_:.4f}')
            bs = s['between_subj_share'].dropna()
            print(f'{band:6s} {cond:9s} {roi:9s} {ep:12s} '
                  + ' '.join(f'{c:>21s}' for c in cells)
                  + f' {(bs.mean() if len(bs) else np.nan):6.3f}')


def main():
    ap = argparse.ArgumentParser(
        description='Figures for the pooled glue capacity CSV (real vs shuffled).')
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--sharey', action='store_true',
                     help='Force one y-scale across all panels. Off by default: '
                          'capacity is not comparable across ROIs at unequal '
                          'points-per-manifold, and sharing the axis hides the '
                          'within-panel epoch structure behind that offset.')
    ap.add_argument('--scheme', type=int, default=None,
                     help='Category scheme to plot if the CSV has more than one '
                          '(default: all present).')
    ap.add_argument('--csv', default=None,
                     help='Default: the path manifold_capacity_epochs.py writes to.')
    ap.add_argument('--figdir', default=None,
                     help='Default: derivatives/glueDecoding/glueEpochsPooled/figures')
    args = ap.parse_args()

    bids_root = get_bids_root()
    base = os.path.join(bids_root, 'derivatives', 'glueDecoding', 'glueEpochsPooled')
    # Imported lazily: manifold_capacity_epochs pulls in the glue env's importer
    # path, and this plotter must stay runnable without it.
    csv = args.csv or os.path.join(
        base, f'group_task-mgs_glueEpochsPooled_stim_{args.voxRes}.csv')
    figdir = args.figdir or os.path.join(base, 'figures')
    print(f'  reading : {csv}\n  figdir  : {figdir}')
    if not os.path.exists(csv):
        print('Not found -- run manifold_capacity_epochs.py first.')
        sys.exit(1)

    df = pd.read_csv(csv)
    print(f'Loaded {len(df)} rows | columns: {sorted(df.columns)}')

    has_shuffle = 'shuffle' in df.columns
    if has_shuffle:
        df['shuffle'] = df['shuffle'].astype(bool)
    else:
        print('  NOTE no "shuffle" column -- plotting real fits only, with no '
              'null to read them against.')
    if args.scheme is not None and 'scheme' in df.columns:
        df = df[df.scheme == args.scheme]
    if 'between_subj_share' not in df.columns:
        df['between_subj_share'] = np.nan

    metrics = [m for m in CANDIDATE_METRICS if m in df.columns]
    if not metrics:
        print(f'No known metric columns found. Present: {sorted(df.columns)}')
        sys.exit(1)
    print(f'Plotting metrics: {metrics}')

    for cond in args.conditions:
        if cond not in set(df.condition.unique()):
            continue
        for m in metrics:
            figure_metric(df, m, CANDIDATE_METRICS[m], cond, args.bands,
                          args.rois, args.voxRes, figdir, has_shuffle,
                          sharey=args.sharey)
            if has_shuffle:
                figure_delta(df, m, CANDIDATE_METRICS[m], cond, args.bands,
                             args.rois, args.voxRes, figdir, sharey=args.sharey)
    print_summary(df, metrics, has_shuffle)


if __name__ == '__main__':
    main()
