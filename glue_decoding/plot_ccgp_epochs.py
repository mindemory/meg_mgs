#!/usr/bin/env python3
"""
plot_ccgp_epochs.py

Cross-subject aggregation + figures for ccgp_epochs_cell.py: mean +/- SEM
ACROSS SUBJECTS of CCGP and shattering dimensionality, per band x ROI x epoch.

The subject axis is the point of this analysis. Unlike the RDM/MDS geometry --
which only exists as a group average, with inter-subject RDM correlation
~0.02-0.07 -- CCGP and SD are computed independently per subject, so here the
error bar is a genuine across-subject SEM and the group claim rests on
consistency between subjects rather than on averaging their noise together.

TWO FIGURES per condition:
  ..._ccgp_...png  three lines per panel (horizontal / vertical / axis) over
      the four epochs, with each dichotomy's own shuffled null as a shaded
      band. CCGP chance is NOT 0.5 -- correlated conditions push it up -- so
      every value is read against its own null, never against 0.5.
  ..._sd_...png    shattering dimensionality over the epochs, against the
      noiseless ring reference (4/35 of balanced dichotomies separable for 8
      points in a plane).

READING THEM -- the ring makes a JOINT prediction, not three separate ones:
'axis' (near-horizontal vs near-vertical) is not a linear coordinate on the
ring plane, so a genuine planar code puts it AT CHANCE by construction. The
signature is horizontal and vertical above their nulls WHILE axis sits on its
null, together with low SD. High CCGP on all three would not be a ring, and
high SD would mean structure beyond the plane.

Usage:
    python plot_ccgp_epochs.py [--voxRes 8mm] [--bands theta alpha beta]
        [--conditions ampOnly ampPhase] [--rois visual parietal frontal]
        [--subjects 1 2 ...] [--outdir <path>] [--figdir <path>] [--csvdir <path>]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from constants import SUBJECT_LIST, get_bids_root
from visual_geometry_epochs_cell import EPOCH_ORDER
from ccgp_epochs_cell import output_path, DICHOTOMIES, GEOMETRIC_RING_SD

_BG, _FG, _GRID = '#000000', '#e0e0e0', '#1c1c1c'
FS_SUPTITLE, FS_PANEL_TTL, FS_AXIS_LABEL, FS_ROW_LABEL, FS_TICK = 18, 14, 13, 14, 10

ROI_COLOURS = {'visual': '#FFC629', 'parietal': '#A78BFA', 'frontal': '#34D399'}
DICH_COLOURS = {'horizontal': '#FFC629', 'vertical': '#4EA1F3', 'axis': '#FF6B6B'}
DICH_LABELS = {'horizontal': 'Left / right', 'vertical': 'Top / bottom',
                'axis': 'Near-H / near-V  (control)'}
BAND_LABELS = {'theta': 'Theta (4-8 Hz)', 'alpha': 'Alpha (8-12 Hz)',
               'beta': 'Beta (13-30 Hz)'}
COND_LABELS = {'ampOnly': 'Amplitude', 'ampPhase': 'Amplitude + Phase'}
EPOCH_LABELS = {'fixation': 'Fixation', 'stimulus': 'Stimulus',
                 'early_delay': 'Early delay', 'late_delay': 'Late delay'}

# Null-based significant fraction, NOT frac>0.7: the 0.7 cut was calibrated on
# high-SNR synthetic data and saturates at 0.00 for real MEG, where a genuine
# effect only lifts individual dichotomies to ~0.55. sd_frac_significant scores
# each dichotomy against its own shuffled null, so it adapts to the noise level.
SD_KEY = 'sd_frac_significant'


def load_all(subjects, bids_root, voxRes, bands, conditions, rois, outdir):
    """Long-format table, one row per (subject, band, condition, roi, epoch)."""
    rows = []
    for s in subjects:
        for band in bands:
            for cond in conditions:
                for roi in rois:
                    fp = output_path(bids_root, s, band, cond, roi, voxRes, outdir)
                    if not os.path.exists(fp):
                        continue
                    with np.load(fp, allow_pickle=True) as z:
                        for ep in EPOCH_ORDER:
                            if f'{ep}__pca_dim' not in z.files:
                                continue
                            r = dict(subjID=s, band=band, condition=cond, roi=roi,
                                     epoch=ep, pca_dim=int(z[f'{ep}__pca_dim']),
                                     n_trials_used=int(z[f'{ep}__n_trials_used']),
                                     sd=float(z[f'{ep}__{SD_KEY}']),
                                     sd_mean_acc=float(z[f'{ep}__sd_mean_acc']),
                                     sd_frac_above_07=float(z[f'{ep}__sd_frac_above_0.7']))
                            for d in DICHOTOMIES:
                                cc = float(z[f'{ep}__ccgp_{d}'])
                                nu = float(z[f'{ep}__ccgp_{d}_null_mean'])
                                r[f'ccgp_{d}'] = cc
                                r[f'null_{d}'] = nu
                                r[f'p_{d}'] = float(z[f'{ep}__ccgp_{d}_p'])
                                # CCGP MINUS THAT SUBJECT'S OWN NULL is the
                                # quantity to average. Chance is not 0.5 and
                                # varies with the condition correlations, which
                                # differ per subject/cell, so averaging raw CCGP
                                # mixes signal with per-subject chance level.
                                r[f'delta_{d}'] = cc - nu
                            rows.append(r)
    return pd.DataFrame(rows)


def mean_sem(df, col):
    """(mean, sem, n) across SUBJECTS -- the unit the error bar represents."""
    v = df[col].values
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    return (float(v.mean()),
            float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
            int(v.size))


def style_ax(ax):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, which='both', labelsize=FS_TICK)
    ax.xaxis.label.set_color(_FG); ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, lw=0.5, ls='--', alpha=0.6)
    ax.set_axisbelow(True)


def figure_ccgp(df, condition, bands, rois, voxRes, figdir):
    cdf = df[df.condition == condition]
    bands = [b for b in bands if b in set(cdf.band.unique())]
    rois = [r for r in rois if r in set(cdf.roi.unique())]
    if not bands or not rois:
        print(f'  no data for {condition}'); return
    n_r, n_c = len(bands), len(rois)
    fig, axes = plt.subplots(n_r, n_c, figsize=(4.4 * n_c + 2.0, 3.2 * n_r + 1.6),
                              facecolor=_BG, squeeze=False)
    x = np.arange(len(EPOCH_ORDER))
    for r, band in enumerate(bands):
        for c, roi in enumerate(rois):
            ax = axes[r][c]; style_ax(ax)
            sub = cdf[(cdf.band == band) & (cdf.roi == roi)]
            nn = []
            for d in DICHOTOMIES:
                m, e = [], []
                for ep in EPOCH_ORDER:
                    ss = sub[sub.epoch == ep]
                    mm, ee, n = mean_sem(ss, f'delta_{d}')
                    m.append(mm); e.append(ee); nn.append(n)
                ax.errorbar(x, m, yerr=e, color=DICH_COLOURS[d], lw=2.0, marker='o',
                            ms=6, capsize=4, zorder=4,
                            label=DICH_LABELS[d] if (r == 0 and c == 0) else None)
            # 0 = that subject's own shuffled null. Everything is expressed
            # relative to it, so this line is chance.
            ax.axhline(0.0, color='#888888', lw=1.2, ls=':', zorder=2)
            ax.set_xticks(x)
            ax.set_xticklabels([EPOCH_LABELS[e] for e in EPOCH_ORDER],
                                rotation=30, ha='right', fontsize=FS_TICK)
            if r == 0:
                ax.set_title(f'{roi.capitalize()} (n={max(nn) if nn else 0})',
                             fontsize=FS_PANEL_TTL,
                             color=ROI_COLOURS.get(roi, _FG), fontweight='bold', pad=8)
            if c == 0:
                ax.set_ylabel('CCGP $-$ null', fontsize=FS_AXIS_LABEL, fontweight='bold')
                ax.text(-0.30, 0.5, BAND_LABELS.get(band, band), transform=ax.transAxes,
                        fontsize=FS_ROW_LABEL, color=_FG, ha='right', va='center',
                        rotation=90, fontweight='bold')
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        leg = fig.legend(h, l, loc='center left', bbox_to_anchor=(0.99, 0.5),
                         fontsize=11, framealpha=0.25, edgecolor='#444444',
                         labelcolor=_FG)
        leg.get_frame().set_facecolor('#1a1a1a')
    fig.suptitle(f'CCGP  |  {COND_LABELS.get(condition, condition)}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1.005)
    fig.text(0.5, 0.972,
             'mean +/- SEM across subjects of (CCGP - that subject\'s own shuffled '
             'null)  |  0 = chance  |  ring predicts L/R and T/B above 0 WITH the '
             'control at 0',
             ha='center', va='top', color='#aaaaaa', fontsize=10.5)
    fig.tight_layout(rect=[0.02, 0, 1, 0.955])
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'ccgp_epochs_ccgp_{condition}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG); plt.close(fig)
    print(f'Saved: {fp}')


def figure_sd(df, condition, bands, rois, voxRes, figdir):
    cdf = df[df.condition == condition]
    bands = [b for b in bands if b in set(cdf.band.unique())]
    rois = [r for r in rois if r in set(cdf.roi.unique())]
    if not bands or not rois:
        return
    fig, axes = plt.subplots(1, len(bands), figsize=(4.6 * len(bands) + 2.0, 4.2),
                              facecolor=_BG, squeeze=False)
    x = np.arange(len(EPOCH_ORDER))
    for c, band in enumerate(bands):
        ax = axes[0][c]; style_ax(ax)
        for roi in rois:
            sub = cdf[(cdf.band == band) & (cdf.roi == roi)]
            m, e = [], []
            for ep in EPOCH_ORDER:
                mm, ee, _ = mean_sem(sub[sub.epoch == ep], 'sd')
                m.append(mm); e.append(ee)
            ax.errorbar(x, m, yerr=e, color=ROI_COLOURS.get(roi, '#fff'), lw=2.0,
                        marker='o', ms=6, capsize=4, zorder=4,
                        label=roi.capitalize() if c == 0 else None)
        # Noiseless geometric bound, NOT an achievable target: finite trials and
        # noise let near-separable dichotomies clear threshold, so real data sits
        # above this even for a perfect ring.
        # Autoscale: real values sit near 0.05-0.15, so a hardcoded 0-1 axis
        # renders every band as a flat line at the bottom.
        ax.axhline(GEOMETRIC_RING_SD, color='#4EA1F3', lw=1.4, ls='--', zorder=2)
        ax.axhline(0.01, color='#888888', lw=1.0, ls=':', zorder=2)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(0.0, lo), max(hi, GEOMETRIC_RING_SD * 1.35))
        ax.set_xticks(x)
        ax.set_xticklabels([EPOCH_LABELS[e] for e in EPOCH_ORDER],
                            rotation=30, ha='right', fontsize=FS_TICK)
        ax.set_title(BAND_LABELS.get(band, band), fontsize=FS_PANEL_TTL,
                     color=_FG, fontweight='bold', pad=8)
        if c == 0:
            ax.set_ylabel('Shattering dimensionality\n(frac. dichotomies sig. vs null)',
                          fontsize=FS_AXIS_LABEL, fontweight='bold')
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        leg = fig.legend(h, l, loc='center left', bbox_to_anchor=(0.99, 0.5),
                         fontsize=11, framealpha=0.25, edgecolor='#444444',
                         labelcolor=_FG)
        leg.get_frame().set_facecolor('#1a1a1a')
    fig.suptitle(f'Shattering dimensionality  |  {COND_LABELS.get(condition, condition)}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1.02)
    fig.text(0.5, 0.945,
             f'mean +/- SEM across subjects, fraction of the 35 dichotomies '
             f'significant vs their own shuffled null  |  dashed = planar-ring '
             f'bound ({GEOMETRIC_RING_SD:.3f} = 4/35)  |  dotted = nominal '
             f'per-dichotomy threshold (0.01)',
             ha='center', va='top', color='#aaaaaa', fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'ccgp_epochs_sd_{condition}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG); plt.close(fig)
    print(f'Saved: {fp}')


def print_summary(df):
    if df.empty:
        print('No data.'); return
    print('\n' + '=' * 78)
    print('CCGP / SD -- mean +/- SEM ACROSS SUBJECTS')
    print('  values are CCGP MINUS that subject\'s own null (0 = chance), with a')
    print('  one-sample t-test across subjects; p is that t-test, not the shuffle p.')
    print('  the ring signature is L/R and T/B above null WITH the control at null.')
    print('=' * 78)
    hdr = (f"{'band':6s} {'cond':9s} {'roi':9s} {'epoch':12s} {'n':>3s} "
           + ' '.join(f'{d[:4]:>13s}' for d in DICHOTOMIES) + f" {'SD':>12s}")
    print(hdr); print('-' * len(hdr))
    for (band, cond, roi), g in df.groupby(['band', 'condition', 'roi']):
        for ep in EPOCH_ORDER:
            s = g[g.epoch == ep]
            if s.empty:
                continue
            cells = []
            for d in DICHOTOMIES:
                v = s[f'delta_{d}'].values
                v = v[np.isfinite(v)]
                m, e, n = mean_sem(s, f'delta_{d}')
                pv = stats.ttest_1samp(v, 0).pvalue if v.size > 1 else np.nan
                cells.append(f'{m:+.3f}p{pv:.3f}')
            sm, se, n = mean_sem(s, 'sd')
            print(f'{band:6s} {cond:9s} {roi:9s} {ep:12s} {n:3d} '
                  + ' '.join(f'{c:>13s}' for c in cells) + f' {sm:.2f}+/-{se:.2f}')


def main():
    ap = argparse.ArgumentParser(description='Aggregate + plot CCGP / SD across subjects.')
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--figdir', default=None)
    ap.add_argument('--csvdir', default=None)
    args = ap.parse_args()

    bids_root = get_bids_root()
    base = os.path.join(bids_root, 'derivatives', 'glueDecoding', 'ccgpEpochs')
    outdir = args.outdir or os.path.join(base, 'data')
    figdir = args.figdir or os.path.join(base, 'figures')
    csvdir = args.csvdir or os.path.join(base, 'tables')
    print(f'  outdir (reading) = {outdir}\n  figdir (writing) = {figdir}\n'
          f'  csvdir (writing) = {csvdir}')

    df = load_all(args.subjects, bids_root, args.voxRes, args.bands,
                  args.conditions, args.rois, outdir)
    if df.empty:
        print('Nothing loaded -- run ccgp_epochs_cell.py first.')
        sys.exit(1)
    print(f'Loaded {len(df)} rows from {df.subjID.nunique()} subjects.')

    os.makedirs(csvdir, exist_ok=True)
    fp = os.path.join(csvdir, f'ccgp_epochs_{args.voxRes}.csv')
    df.to_csv(fp, index=False)
    print(f'Saved: {fp}')

    for cond in args.conditions:
        if cond not in set(df.condition.unique()):
            continue
        figure_ccgp(df, cond, args.bands, args.rois, args.voxRes, figdir)
        figure_sd(df, cond, args.bands, args.rois, args.voxRes, figdir)
    print_summary(df)


if __name__ == '__main__':
    main()
