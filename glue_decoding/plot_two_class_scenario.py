#!/usr/bin/env python3
"""
plot_two_class_scenario.py

Focused P=2 (left/right) case study plot: visual ROI only, theta/alpha/beta
only, ampOnly -- overlays the P=2 ridge-LOO classifier accuracy timecourse
WITH ERP removal against WITHOUT ERP removal, one panel per band, so the
two are directly comparable before deciding how to scope the (separate,
not-yet-built) GLUE moving-window step -- see chat history / module
docstring of run_two_class_scenario.sh.

Reuses plot_linear_decoding_categories.py's load_all_subjects,
cluster_permutation_test, and design constants directly (no duplicated
logic) -- this script only adds the two-curve overlay layout, since the
existing make_timeseries_figure draws one curve per panel, not two.

One figure, 3 rows (theta/alpha/beta) x 1 col (visual only): each panel
shows both curves (ERP removed = solid/brighter, ERP kept = dashed/dimmer),
mean +/- SEM across subjects, theoretical chance line (0.5 for P=2), and
TWO independent cluster-permutation significance dot rows (one per curve,
offset vertically, colour-matched to its curve) -- see
plot_linear_decoding_categories.py's cluster_permutation_test for the method.

Usage:
    python plot_two_class_scenario.py [--voxRes 8mm]
                                       [--bands theta alpha beta]
                                       [--roi visual] [--subjects 1 2 ...]
                                       [--erp_removed_dir <path>] [--erp_kept_dir <path>]
                                       [--n_perm 1000] [--cluster_alpha 0.05] [--alpha 0.05]
                                       [--figdir <path>]
"""

import os, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

from constants import SUBJECT_LIST, get_bids_root
from plot_linear_decoding_categories import (
    load_all_subjects, aggregate_lindecode, cluster_permutation_test,
    _style_ax, _BG, _FG, _GRID, _FLAG_LINE, _FLAG_TXT, _CHANCE,
    ROI_COLOURS, BAND_LABELS, EVENT_FLAGS,
)

SCHEME = 2          # left/right -- this script is specifically the P=2 case study
CONDITION = 'ampOnly'

_SIG_REMOVED = '#ffffff'   # ERP-removed significance dots
_SIG_KEPT    = '#999999'   # ERP-kept significance dots (dimmer, matches its dashed/dimmer curve)


def _dim_colour(colour, factor=0.5):
    """Lighter/desaturated variant of `colour`, for the ERP-kept curve --
    same recipe as plot_linear_decoding_categories.py's state_shades."""
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(colour))
    return mcolors.hsv_to_rgb((h, s * factor, min(1.0, v * 1.1 + 0.1)))


def plot_band_panel(ax, roi, band,
                     tv_removed, mean_removed, sem_removed, sig_removed,
                     tv_kept, mean_kept, sem_kept, sig_kept,
                     chance_level, is_bottom_row, show_title, show_flag_labels=True):
    col_removed = ROI_COLOURS[roi]
    col_kept    = _dim_colour(col_removed)

    ax.axhline(chance_level, color=_CHANCE, lw=0.8, ls=':', zorder=1)

    lo_candidates, hi_candidates = [chance_level], [chance_level]

    if tv_removed is not None:
        ax.fill_between(tv_removed, mean_removed - sem_removed, mean_removed + sem_removed,
                         alpha=0.30, color=col_removed, zorder=3)
        ax.plot(tv_removed, mean_removed, color=col_removed, lw=1.6, zorder=4, label='ERP removed')
        lo_candidates.append(np.nanmin(mean_removed - sem_removed))
        hi_candidates.append(np.nanmax(mean_removed + sem_removed))

    if tv_kept is not None:
        ax.fill_between(tv_kept, mean_kept - sem_kept, mean_kept + sem_kept,
                         alpha=0.22, color=col_kept, zorder=2)
        ax.plot(tv_kept, mean_kept, color=col_kept, lw=1.4, ls='--', zorder=3, label='ERP kept')
        lo_candidates.append(np.nanmin(mean_kept - sem_kept))
        hi_candidates.append(np.nanmax(mean_kept + sem_kept))

    lo, hi = min(lo_candidates), max(hi_candidates)
    pad = max(1e-6, (hi - lo) * 0.15)
    lo, hi = lo - pad, hi + pad

    if sig_removed is not None and sig_removed.any() and tv_removed is not None:
        y_sig = lo + 0.10 * (hi - lo)
        ax.plot(tv_removed[sig_removed], np.full(sig_removed.sum(), y_sig),
                '.', color=_SIG_REMOVED, ms=3.0, zorder=6)
    if sig_kept is not None and sig_kept.any() and tv_kept is not None:
        y_sig = lo + 0.05 * (hi - lo)
        ax.plot(tv_kept[sig_kept], np.full(sig_kept.sum(), y_sig),
                '.', color=_SIG_KEPT, ms=3.0, zorder=6)

    tv_any = tv_removed if tv_removed is not None else tv_kept
    t0, t1 = float(tv_any[0]), float(tv_any[-1])
    ax.set_xlim(t0, t1)
    ax.set_ylim(lo, hi)

    y_lo, y_hi = ax.get_ylim()
    for (t_flag, label, y_frac) in EVENT_FLAGS:
        ax.axvline(t_flag, color=_FLAG_LINE, lw=0.8, ls='--', zorder=4)
        if show_flag_labels:
            ax.text(t_flag, y_lo + y_frac * (y_hi - y_lo), label,
                    rotation=90, va='top', ha='right', fontsize=6.5, color=_FLAG_TXT, zorder=5)

    ax.set_ylabel('LOO accuracy', fontsize=8, color=_FG)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    if not is_bottom_row:
        ax.set_xticklabels([])

    ax.grid(True, color=_GRID, lw=0.4, zorder=0)
    _style_ax(ax)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    if show_title:
        leg = ax.legend(fontsize=8, loc='upper right', framealpha=0.2,
                         edgecolor='#444444', labelcolor=_FG)
        leg.get_frame().set_facecolor('#1a1a1a')


def make_figure(bands, roi, all_data_removed, all_data_kept, voxRes, outdir_fig,
                 n_perm=1000, cluster_alpha=0.05, alpha=0.05):
    n_bands = len(bands)
    fig_w, row_h = 6.5, 2.0
    fig_h = row_h * n_bands + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_bands, 1, figure=fig, hspace=0.5,
                            left=0.12, right=0.97,
                            top=1 - 0.6 / fig_h, bottom=0.35 / fig_h)

    for r_idx, band in enumerate(bands):
        ax = fig.add_subplot(gs[r_idx, 0])
        is_bottom_row = (r_idx == n_bands - 1)
        show_title    = (r_idx == 0)

        res_removed = aggregate_lindecode(all_data_removed, band, roi, CONDITION, SCHEME)
        res_kept    = aggregate_lindecode(all_data_kept,    band, roi, CONDITION, SCHEME)

        tv_r, acc_r, mean_r, sem_r, chance_r = res_removed[:5]
        tv_k, acc_k, mean_k, sem_k, chance_k = res_kept[:5]

        if tv_r is None and tv_k is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, color=_FG, fontsize=9)
            _style_ax(ax)
            continue

        chance_level = chance_r if chance_r is not None else chance_k

        sig_r = cluster_permutation_test(acc_r, chance_level, n_perm=n_perm,
                                          cluster_alpha=cluster_alpha, alpha=alpha) \
            if tv_r is not None and acc_r.shape[0] >= 2 else None
        sig_k = cluster_permutation_test(acc_k, chance_level, n_perm=n_perm,
                                          cluster_alpha=cluster_alpha, alpha=alpha) \
            if tv_k is not None and acc_k.shape[0] >= 2 else None

        plot_band_panel(ax, roi, band,
                         tv_r, mean_r, sem_r, sig_r,
                         tv_k, mean_k, sem_k, sig_k,
                         chance_level, is_bottom_row, show_title, show_flag_labels=show_title)

        ax.annotate(BAND_LABELS.get(band, band).replace('\n', '  '),
                    xy=(-0.14, 0.5), xycoords='axes fraction',
                    fontsize=9, color=_FG, ha='right', va='center',
                    rotation=90, fontweight='bold')

    fig.suptitle(f'P=2 (Left/Right) Ridge-LOO Classifier  |  {roi.capitalize()}  |  '
                 f'Amplitude only  |  {voxRes}  |  dots = cluster permutation vs chance, p<{alpha}',
                 color=_FG, fontsize=11, fontweight='bold', y=1.0)
    fig.text(0.5, 0.01, 'Time (s)', ha='center', va='bottom', color=_FG, fontsize=9)

    os.makedirs(outdir_fig, exist_ok=True)
    fpath = os.path.join(outdir_fig, f'two_class_scenario_{roi}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


def main():
    parser = argparse.ArgumentParser(
        description='Plot the P=2 (left/right) ridge-LOO classifier timecourse, '
                     'ERP removed vs. ERP kept, overlaid.')
    parser.add_argument('--voxRes', default='8mm')
    parser.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    parser.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    parser.add_argument('--roi', default='visual')
    parser.add_argument('--erp_removed_dir', required=True,
                         help='Directory containing the ERP-removed per-subject .npz files.')
    parser.add_argument('--erp_kept_dir', required=True,
                         help='Directory containing the ERP-kept per-subject .npz files.')
    parser.add_argument('--n_perm', type=int, default=1000)
    parser.add_argument('--cluster_alpha', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--figdir', default=None,
                         help='Directory to save the figure (default: erp_removed_dir\'s parent).')
    args = parser.parse_args()

    bids_root = get_bids_root()
    figdir = args.figdir or os.path.dirname(args.erp_removed_dir.rstrip('/'))

    print(f'Loading ERP-removed data from: {args.erp_removed_dir}')
    all_data_removed = load_all_subjects(
        args.subjects, bids_root, args.voxRes, args.bands, [args.roi],
        [CONDITION], [SCHEME], outdir=args.erp_removed_dir)

    print(f'Loading ERP-kept data from: {args.erp_kept_dir}')
    all_data_kept = load_all_subjects(
        args.subjects, bids_root, args.voxRes, args.bands, [args.roi],
        [CONDITION], [SCHEME], outdir=args.erp_kept_dir)

    n_removed = sum(1 for d in all_data_removed if d)
    n_kept    = sum(1 for d in all_data_kept if d)
    print(f'Loaded ERP-removed: {n_removed}/{len(args.subjects)} subjects | '
          f'ERP-kept: {n_kept}/{len(args.subjects)} subjects.')

    make_figure(args.bands, args.roi, all_data_removed, all_data_kept, args.voxRes, figdir,
                n_perm=args.n_perm, cluster_alpha=args.cluster_alpha, alpha=args.alpha)

    print('\nDone.')


if __name__ == '__main__':
    main()
