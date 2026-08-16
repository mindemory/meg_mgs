#!/usr/bin/env python3
"""
plot_circular_tgm.py

Cross-subject aggregation + figures for circular_tgm_cell.py.

Per condition (ampOnly / ampPhase) it writes two figures, each a
bands x ROIs grid:
  1. circular_tgm_<condition>_<voxRes>.png -- the TGM heatmaps themselves
     (group mean decoding error, train time on y, test time on x).
  2. circular_tgm_diagonal_<condition>_<voxRes>.png -- the matrix DIAGONAL as
     a timeseries, which is exactly the same quantity decoding_ts_cell.py /
     plot_decoding_ts.py produce, so the two analyses can be checked against
     each other directly.

READING THE HEATMAPS
Colour is decoding error in degrees with the scale CENTRED ON CHANCE (90 deg,
the expected |circular error| with no location information), and the map is
oriented so WARM = LOWER error = better decoding. Lower is better throughout;
the colourbar is labelled accordingly so the direction is never ambiguous.
  - A narrow warm band ON the diagonal only => a code that is present but
    changes over time (each moment decodable only by a model trained at that
    same moment).
  - A broad warm square spanning many train/test pairs => a STATIONARY code
    (one pattern persisting), which is the signature usually of interest for
    a delay-period maintenance account.
  - Warm off-diagonal blocks linking two separated epochs => reactivation of
    an earlier pattern.
Note off-diagonal cells can also sit ABOVE chance (cool): applying a model to
a period whose geometry has rotated yields systematically wrong angles, not
merely uninformative ones.

SIGNIFICANCE
2-D cluster-based permutation (Maris & Oostenveld 2007) against chance, by
sign-flipping subjects -- the natural extension of the 1-D sign-flip test
plot_linear_decoding_categories.py already uses, with candidate clusters
formed from 2-D connected components of the supra-threshold t-map (so a
contiguous train x test region is treated as one unit, which is exactly the
structure a TGM has). Significant clusters are outlined, not colour-masked,
so the underlying error values stay readable. This corrects for the ~2.9k
train/test cells being tested, which an uncorrected per-cell test would not.

Usage:
    python plot_circular_tgm.py [--voxRes 8mm] [--bands theta alpha beta]
                                 [--rois visual parietal frontal]
                                 [--conditions ampOnly ampPhase]
                                 [--subjects 1 2 ...] [--metric signed|unsigned]
                                 [--n_perm 1000] [--outdir <path>] [--figdir <path>]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats, ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from constants import SUBJECT_LIST, ROI_NAMES, get_bids_root
from circular_tgm_cell import output_path, CHANCE_ERROR_DEG

_BG, _FG, _GRID = '#000000', '#e0e0e0', '#1c1c1c'
_FLAG = '#888888'

BAND_LABELS = {'theta': 'Theta (4-8 Hz)', 'alpha': 'Alpha (8-12 Hz)',
               'beta': 'Beta (13-30 Hz)'}
COND_LABELS = {'ampOnly': 'Amplitude', 'ampPhase': 'Amplitude + Phase'}
ROI_COLOURS = {'visual': '#FFC629', 'parietal': '#A78BFA', 'frontal': '#34D399'}
EVENT_TIMES = [(0.0, 'Stim'), (0.2, 'Delay')]

METRIC_KEY = {'signed': 'err_signed_circmean_abs', 'unsigned': 'err_unsigned_mean'}


def _style_ax(ax):
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color('#333333')
    ax.tick_params(colors=_FG, labelsize=7)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)


def load_all(subjects, bids_root, voxRes, bands, rois, conditions, metric, outdir):
    """data[(band, roi, condition)] = (err_stack (n_subj,T,T), tv, n_subj)."""
    key = METRIC_KEY[metric]
    data = {}
    for band in bands:
        for roi in rois:
            for cond in conditions:
                mats, tv = [], None
                for s in subjects:
                    fp = output_path(bids_root, s, band, roi, cond, voxRes, outdir)
                    if not os.path.exists(fp):
                        continue
                    with np.load(fp, allow_pickle=True) as npz:
                        mats.append(np.asarray(npz[key], dtype=float))
                        if tv is None:
                            tv = np.asarray(npz['eval_time_vector'], dtype=float)
                data[(band, roi, cond)] = (np.stack(mats) if mats else None, tv, len(mats))
    return data


def cluster_permutation_2d(err_stack, chance=CHANCE_ERROR_DEG, n_perm=1000,
                            cluster_alpha=0.05, alpha=0.05, seed=0):
    """
    2-D sign-flip cluster permutation of err(t_train, t_test) against chance.

    err_stack: (n_subj, T, T). Returns a (T, T) bool mask of cells belonging
    to a cluster significant at `alpha`. Candidate clusters are 2-D connected
    components (8-neighbour) of |t| > t_crit; the cluster statistic is
    sum(|t|) within a component; the null is the max cluster statistic over
    sign-flipped relabellings, which is what provides family-wise control
    across all T*T cells.
    """
    if err_stack is None or err_stack.shape[0] < 2:
        return None
    n_subj = err_stack.shape[0]
    diff = err_stack - chance
    t_crit = stats.t.ppf(1 - cluster_alpha / 2, df=n_subj - 1)
    struct = np.ones((3, 3), dtype=bool)   # 8-connectivity

    def tmap(d):
        m = d.mean(axis=0)
        s = d.std(axis=0, ddof=1)
        s = np.where(s < 1e-12, 1e-12, s)
        return m / (s / np.sqrt(n_subj))

    def clusters(t):
        lab, n = ndimage.label(np.abs(t) > t_crit, structure=struct)
        if n == 0:
            return [], lab
        stats_ = ndimage.sum(np.abs(t), lab, index=np.arange(1, n + 1))
        return list(np.atleast_1d(stats_)), lab

    t_obs = tmap(diff)
    obs_stats, obs_lab = clusters(t_obs)
    if not obs_stats:
        return np.zeros(t_obs.shape, dtype=bool)

    rng = np.random.default_rng(seed)
    null = np.zeros(n_perm)
    for p in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_subj)
        st, _ = clusters(tmap(diff * signs[:, None, None]))
        null[p] = max(st) if st else 0.0

    sig = np.zeros(t_obs.shape, dtype=bool)
    for ci, cstat in enumerate(obs_stats, start=1):
        if (np.sum(null >= cstat) + 1) / (n_perm + 1) < alpha:
            sig |= (obs_lab == ci)
    return sig


def _time_ticks(ax, tv, axis='x'):
    idx = [i for i, t in enumerate(tv) if abs(t - round(t * 2) / 2) < 1e-9 and
           abs((round(t * 2) / 2) % 0.5) < 1e-9]
    idx = idx[::max(1, len(idx) // 6)]
    locs, labs = idx, [f'{tv[i]:.1f}' for i in idx]
    if axis == 'x':
        ax.set_xticks(locs); ax.set_xticklabels(labs, fontsize=6.5)
    else:
        ax.set_yticks(locs); ax.set_yticklabels(labs, fontsize=6.5)


def figure_tgm(data, bands, rois, cond, voxRes, figdir, n_perm, alpha, metric):
    n_r, n_c = len(bands), len(rois)
    fig_h = 3.4 * n_r + 1.4
    fig = plt.figure(figsize=(3.5 * n_c + 1.3, fig_h), facecolor=_BG)
    # Title block needs a fixed ~1.05 in of headroom regardless of how many band
    # rows there are -- a fixed fractional top collides with the column titles
    # whenever the figure is short (e.g. a single-band run).
    gs = gridspec.GridSpec(n_r, n_c, figure=fig, hspace=0.30, wspace=0.28,
                            left=0.09, right=0.90,
                            top=1 - 1.05 / fig_h, bottom=0.55 / fig_h)
    im = None
    for r, band in enumerate(bands):
        for c, roi in enumerate(rois):
            ax = fig.add_subplot(gs[r, c])
            stack, tv, n_subj = data.get((band, roi, cond), (None, None, 0))
            if stack is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color=_FG, fontsize=9)
                _style_ax(ax)
                continue
            m = stack.mean(axis=0)
            span = max(2.0, np.nanmax(np.abs(m - CHANCE_ERROR_DEG)))
            # 'RdBu': low (better than chance) -> red/warm, high -> blue.
            im = ax.imshow(m, origin='lower', cmap='RdBu', aspect='equal',
                           vmin=CHANCE_ERROR_DEG - span, vmax=CHANCE_ERROR_DEG + span,
                           interpolation='nearest')
            sig = cluster_permutation_2d(stack, n_perm=n_perm, alpha=alpha)
            if sig is not None and sig.any():
                ax.contour(sig.astype(float), levels=[0.5], colors='#000000',
                           linewidths=1.4)
                ax.contour(sig.astype(float), levels=[0.5], colors='#ffffff',
                           linewidths=0.7)
            for t_ev, _lab in EVENT_TIMES:
                i_ev = int(np.argmin(np.abs(tv - t_ev)))
                ax.axhline(i_ev, color=_FLAG, lw=0.6, ls=':')
                ax.axvline(i_ev, color=_FLAG, lw=0.6, ls=':')
            ax.plot([0, len(tv) - 1], [0, len(tv) - 1], color='#666666', lw=0.6, ls='--')
            _time_ticks(ax, tv, 'x'); _time_ticks(ax, tv, 'y')
            if r == 0:
                ax.set_title(f'{roi.capitalize()}  (n={n_subj})', fontsize=9,
                             color=_FG, fontweight='bold', pad=4)
            if r == n_r - 1:
                ax.set_xlabel('Test time (s)', fontsize=8)
            if c == 0:
                ax.set_ylabel('Train time (s)', fontsize=8)
                ax.annotate(BAND_LABELS.get(band, band), xy=(-0.34, 0.5),
                            xycoords='axes fraction', fontsize=9.5, color=_FG,
                            ha='right', va='center', rotation=90, fontweight='bold')
            _style_ax(ax)

    if im is not None:
        cax = fig.add_axes([0.915, 0.25, 0.013, 0.45])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('circular error (deg)  |  WARM = lower error = better',
                     color=_FG, fontsize=8)
        cb.ax.tick_params(colors=_FG, labelsize=7)
        cb.outline.set_edgecolor('#333333')

    fig.suptitle(f'Circular TGM (LOO ridge, sin/cos)  |  {COND_LABELS.get(cond, cond)}  |  '
                 f'{voxRes}  |  ERP removed  |  metric={metric}\n'
                 f'chance = {CHANCE_ERROR_DEG:.0f} deg; outlined = 2-D cluster permutation '
                 f'vs chance, p<{alpha}; dashed = diagonal',
                 color=_FG, fontsize=10.5, fontweight='bold', y=1 - 0.10 / fig_h)
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'circular_tgm_{cond}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')
    return fp


def figure_diagonal(data, bands, rois, cond, voxRes, figdir, n_perm, alpha, metric):
    """
    The TGM diagonal == the standard decoding-over-time curve, so this figure
    is directly comparable to plot_decoding_ts.py's output for the same cells.
    """
    n_r, n_c = len(bands), len(rois)
    fig_h = 2.3 * n_r + 1.3
    fig = plt.figure(figsize=(3.6 * n_c + 0.8, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_r, n_c, figure=fig, hspace=0.40, wspace=0.28,
                            left=0.09, right=0.98,
                            top=1 - 1.00 / fig_h, bottom=0.60 / fig_h)
    for r, band in enumerate(bands):
        for c, roi in enumerate(rois):
            ax = fig.add_subplot(gs[r, c])
            stack, tv, n_subj = data.get((band, roi, cond), (None, None, 0))
            if stack is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color=_FG, fontsize=9)
                _style_ax(ax)
                continue
            diag = np.stack([np.diag(s) for s in stack])          # (n_subj, T)
            mean, sem = diag.mean(0), diag.std(0) / np.sqrt(diag.shape[0])
            col = ROI_COLOURS.get(roi, '#ffffff')
            ax.axhline(CHANCE_ERROR_DEG, color='#555555', lw=0.8, ls=':')
            ax.fill_between(tv, mean - sem, mean + sem, color=col, alpha=0.28)
            ax.plot(tv, mean, color=col, lw=1.5)
            # 1-D sign-flip cluster test on the diagonal
            sig = cluster_permutation_2d(diag[:, :, None], n_perm=n_perm, alpha=alpha)
            if sig is not None and sig.any():
                y = (mean - sem).min()
                ax.plot(tv[sig[:, 0]], np.full(sig[:, 0].sum(), y), '.',
                        color='#ffffff', ms=3)
            for t_ev, lab in EVENT_TIMES:
                ax.axvline(t_ev, color=_FLAG, lw=0.7, ls='--')
            ax.set_xlim(tv[0], tv[-1])
            ax.invert_yaxis()          # lower error upward = better decoding upward
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.grid(True, color=_GRID, lw=0.4)
            if r == 0:
                ax.set_title(f'{roi.capitalize()}  (n={n_subj})', fontsize=9,
                             color=_FG, fontweight='bold')
            if r == n_r - 1:
                ax.set_xlabel('Time (s)', fontsize=8)
            if c == 0:
                ax.set_ylabel('error (deg)\n(inverted)', fontsize=7.5, color=_FG)
                ax.annotate(BAND_LABELS.get(band, band), xy=(-0.26, 0.5),
                            xycoords='axes fraction', fontsize=9, color=_FG,
                            ha='right', va='center', rotation=90, fontweight='bold')
            _style_ax(ax)

    fig.suptitle(f'Circular TGM DIAGONAL (= decoding over time)  |  '
                 f'{COND_LABELS.get(cond, cond)}  |  {voxRes}  |  ERP removed\n'
                 f'y-axis INVERTED so upward = better; dots = cluster permutation '
                 f'vs {CHANCE_ERROR_DEG:.0f} deg chance, p<{alpha}',
                 color=_FG, fontsize=10.5, fontweight='bold', y=1 - 0.10 / fig_h)
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'circular_tgm_diagonal_{cond}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')
    return fp


def print_summary(data, bands, rois, conditions, metric):
    hdr = (f"{'band':6s} {'roi':9s} {'cond':9s} {'n':>3s} {'diag_min':>9s} "
           f"{'@t':>7s} {'tgm_min':>8s} {'offdiag':>8s}")
    print('\n' + '=' * len(hdr))
    print(f'SUMMARY (metric={metric}; chance={CHANCE_ERROR_DEG:.0f} deg, LOWER is better)')
    print('  diag_min = best on-diagonal error, @t = when; tgm_min = best cell anywhere;')
    print('  offdiag  = mean error over strictly off-diagonal cells (generalization)')
    print('=' * len(hdr))
    print(hdr); print('-' * len(hdr))
    for cond in conditions:
        for band in bands:
            for roi in rois:
                stack, tv, n = data.get((band, roi, cond), (None, None, 0))
                if stack is None:
                    print(f'{band:6s} {roi:9s} {cond:9s} {0:3d} ' + '-' * 34)
                    continue
                m = stack.mean(0)
                d = np.diag(m)
                off = m[~np.eye(m.shape[0], dtype=bool)]
                print(f'{band:6s} {roi:9s} {cond:9s} {n:3d} {d.min():9.1f} '
                      f'{tv[int(np.argmin(d))]:+7.2f} {m.min():8.1f} {off.mean():8.1f}')


def main():
    ap = argparse.ArgumentParser(description='Aggregate + plot circular TGM across subjects.')
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--rois', nargs='+', default=list(ROI_NAMES))
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    ap.add_argument('--metric', default='signed', choices=['signed', 'unsigned'])
    ap.add_argument('--n_perm', type=int, default=1000)
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--figdir', required=True)
    args = ap.parse_args()

    bids_root = get_bids_root()
    print(f'Loading | {args.voxRes} | bands={args.bands} | rois={args.rois} | '
          f'conditions={args.conditions} | metric={args.metric}')
    data = load_all(args.subjects, bids_root, args.voxRes, args.bands, args.rois,
                    args.conditions, args.metric, args.outdir)
    tot = sum(v[2] for v in data.values())
    print(f'Loaded {tot} subject-cells.')
    if tot == 0:
        print('Nothing to plot -- run circular_tgm_cell.py first.')
        return

    for cond in args.conditions:
        figure_tgm(data, args.bands, args.rois, cond, args.voxRes, args.figdir,
                   args.n_perm, args.alpha, args.metric)
        figure_diagonal(data, args.bands, args.rois, cond, args.voxRes, args.figdir,
                        args.n_perm, args.alpha, args.metric)
    print_summary(data, args.bands, args.rois, args.conditions, args.metric)
    print('\nDone.')


if __name__ == '__main__':
    main()
