#!/usr/bin/env python3
"""
plot_two_class_scenario.py

Focused P=2 (left/right) case study, visual ROI only, theta/alpha/beta only,
ampOnly. One figure, 3 rows (theta/alpha/beta) x 2 cols:
    col 0: P=2 ridge-LOO classifier accuracy timecourse (ERP removed vs kept)
    col 1: ipsi- vs contra-visual amplitude timecourse (ERP removed vs kept)
-- see chat history / module docstrings of run_two_class_scenario.sh,
linear_decoding_categories_cell.py, and ipsi_contra_cell.py for the full
motivation and each computation's details.

IMPORTANT caveat verified analytically (see chat history): for col 0's
classifier, remove_erp has ZERO effect on the plotted accuracy. Every
evaluated timepoint is independently z-scored across trials inside
ridge_ovr_timeseries (mu = trial-mean, sd = trial-std, both computed AFTER
ERP removal or not); subtracting each cell's grand trial-mean before that
z-scoring step is subtracting a value that is IDENTICAL across every trial
at that timepoint (a pure constant shift), which the very next z-scoring
step removes again as `mu` regardless of whether it was pre-subtracted --
and it doesn't change the trial-to-trial variance either (Var(X - c) =
Var(X) for constant c), so `sd` is unaffected too. X_z is therefore
bit-identical whether or not remove_erp was applied, for this classifier
specifically. This is NOT true for col 1 (ipsi/contra amplitude), which has
no such per-timepoint re-centering step -- removing the common
stimulus-locked ERP there genuinely changes the plotted curves, since it's
plotting a trial-averaged signal, not a re-standardized-per-timepoint one.
Both ERP variants are still plotted for col 0 as a sanity check on this
exact-equivalence claim (any visible difference would indicate a bug), but
if compute time ever matters, the ERP-kept classifier runs are fully
redundant -- unlike the ipsi/contra column's ERP-kept runs, which are not.

Reuses plot_linear_decoding_categories.py's load_all_subjects,
cluster_permutation_test, and design constants directly (col 0), and
ipsi_contra_cell.py's saved per-subject .npz layout directly (col 1) -- no
duplicated computation logic.

Usage:
    python plot_two_class_scenario.py [--voxRes 8mm]
                                       [--bands theta alpha beta]
                                       [--roi visual] [--subjects 1 2 ...]
                                       [--erp_removed_dir <path>] [--erp_kept_dir <path>]
                                       [--ipsi_contra_removed_dir <path>]
                                       [--ipsi_contra_kept_dir <path>]
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
from ipsi_contra_cell import output_path as ipsi_contra_output_path

SCHEME = 2          # left/right -- this script is specifically the P=2 case study
CONDITION = 'ampOnly'

_SIG_REMOVED = '#ffffff'   # ERP-removed significance dots
_SIG_KEPT    = '#999999'   # ERP-kept significance dots (dimmer, matches its dashed/dimmer curve)


def _dim_colour(colour, factor=0.5):
    """Lighter/desaturated variant of `colour` -- same recipe as
    plot_linear_decoding_categories.py's state_shades."""
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(colour))
    return mcolors.hsv_to_rgb((h, s * factor, min(1.0, v * 1.1 + 0.1)))


# ── Column 0: classifier accuracy (ERP removed vs kept) ─────────────────────

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
        leg = ax.legend(fontsize=7.5, loc='upper right', framealpha=0.2,
                         edgecolor='#444444', labelcolor=_FG)
        leg.get_frame().set_facecolor('#1a1a1a')
        ax.set_title('P=2 ridge-LOO classifier accuracy', fontsize=9, color=_FG, pad=4)


# ── Column 1: ipsi vs contra amplitude (ERP removed vs kept) ────────────────

def load_ipsi_contra_all_subjects(subjects, bids_root, voxRes, bands, outdir):
    """Returns list (one dict per subject, aligned with `subjects`) of
    {band: {'tv':, 'ipsi':, 'contra':}} -- None entries for missing cells."""
    all_data = []
    for subjID in subjects:
        subj_data = {}
        for band in bands:
            fpath = ipsi_contra_output_path(bids_root, subjID, band, voxRes, outdir)
            if not os.path.exists(fpath):
                subj_data[band] = None
                continue
            with np.load(fpath) as npz:
                subj_data[band] = dict(tv=npz['time_vector'], ipsi=npz['ipsi_curve'],
                                        contra=npz['contra_curve'])
        all_data.append(subj_data)
    return all_data


def aggregate_ipsi_contra(all_data, band):
    """Returns (tv, mean_ipsi, sem_ipsi, mean_contra, sem_contra) or all-None."""
    ipsi_curves, contra_curves, tv = [], [], None
    for subj_data in all_data:
        cell = subj_data.get(band) if subj_data else None
        if cell is None:
            continue
        ipsi_curves.append(cell['ipsi'])
        contra_curves.append(cell['contra'])
        if tv is None:
            tv = cell['tv']
    if not ipsi_curves:
        return None, None, None, None, None
    ipsi_mat, contra_mat = np.stack(ipsi_curves), np.stack(contra_curves)
    n = ipsi_mat.shape[0]
    return (tv, ipsi_mat.mean(axis=0), ipsi_mat.std(axis=0) / np.sqrt(n),
            contra_mat.mean(axis=0), contra_mat.std(axis=0) / np.sqrt(n))


def plot_ipsi_contra_panel(ax, roi, band,
                            tv_r, ipsi_r, ipsi_r_sem, contra_r, contra_r_sem,
                            tv_k, ipsi_k, ipsi_k_sem, contra_k, contra_k_sem,
                            is_bottom_row, show_title, show_flag_labels=True):
    ipsi_colour   = ROI_COLOURS[roi]
    contra_colour = _dim_colour(ipsi_colour)

    ax.axhline(0.0, color=_CHANCE, lw=0.8, ls=':', zorder=1)
    lo_candidates, hi_candidates = [0.0], [0.0]

    def _draw(tv, mean_c, sem_c, colour, ls, label):
        if tv is None:
            return
        ax.fill_between(tv, mean_c - sem_c, mean_c + sem_c, alpha=0.20, color=colour, zorder=2)
        ax.plot(tv, mean_c, color=colour, lw=1.5, ls=ls, zorder=3, label=label)
        lo_candidates.append(np.nanmin(mean_c - sem_c))
        hi_candidates.append(np.nanmax(mean_c + sem_c))

    _draw(tv_r, ipsi_r,   ipsi_r_sem,   ipsi_colour,   '-',  'Ipsi (ERP removed)')
    _draw(tv_r, contra_r, contra_r_sem, contra_colour, '-',  'Contra (ERP removed)')
    _draw(tv_k, ipsi_k,   ipsi_k_sem,   ipsi_colour,   '--', 'Ipsi (ERP kept)')
    _draw(tv_k, contra_k, contra_k_sem, contra_colour, '--', 'Contra (ERP kept)')

    tv_any = tv_r if tv_r is not None else tv_k
    if tv_any is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color=_FG, fontsize=9)
        _style_ax(ax)
        return

    lo, hi = min(lo_candidates), max(hi_candidates)
    pad = max(1e-6, (hi - lo) * 0.15)
    lo, hi = lo - pad, hi + pad
    t0, t1 = float(tv_any[0]), float(tv_any[-1])
    ax.set_xlim(t0, t1)
    ax.set_ylim(lo, hi)

    y_lo, y_hi = ax.get_ylim()
    for (t_flag, label, y_frac) in EVENT_FLAGS:
        ax.axvline(t_flag, color=_FLAG_LINE, lw=0.8, ls='--', zorder=4)
        if show_flag_labels:
            ax.text(t_flag, y_lo + y_frac * (y_hi - y_lo), label,
                    rotation=90, va='top', ha='right', fontsize=6.5, color=_FLAG_TXT, zorder=5)

    ax.set_ylabel('Baselined amplitude (z)', fontsize=8, color=_FG)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    if not is_bottom_row:
        ax.set_xticklabels([])

    ax.grid(True, color=_GRID, lw=0.4, zorder=0)
    _style_ax(ax)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    if show_title:
        leg = ax.legend(fontsize=6.5, loc='upper right', framealpha=0.2,
                         edgecolor='#444444', labelcolor=_FG, ncol=1)
        leg.get_frame().set_facecolor('#1a1a1a')
        ax.set_title('Ipsi vs contra visual amplitude', fontsize=9, color=_FG, pad=4)


# ── Figure assembly ──────────────────────────────────────────────────────────

def make_figure(bands, roi, all_data_removed, all_data_kept,
                 ipsi_contra_removed, ipsi_contra_kept, voxRes, outdir_fig,
                 n_perm=1000, cluster_alpha=0.05, alpha=0.05):
    n_bands = len(bands)
    fig_w, row_h = 12.0, 2.2
    fig_h = row_h * n_bands + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_bands, 2, figure=fig, hspace=0.5, wspace=0.28,
                            left=0.08, right=0.98,
                            top=1 - 0.6 / fig_h, bottom=0.35 / fig_h)

    for r_idx, band in enumerate(bands):
        is_bottom_row = (r_idx == n_bands - 1)
        show_title    = (r_idx == 0)

        # -- col 0: classifier accuracy --
        ax0 = fig.add_subplot(gs[r_idx, 0])
        res_removed = aggregate_lindecode(all_data_removed, band, roi, CONDITION, SCHEME)
        res_kept    = aggregate_lindecode(all_data_kept,    band, roi, CONDITION, SCHEME)
        tv_r, acc_r, mean_r, sem_r, chance_r = res_removed[:5]
        tv_k, acc_k, mean_k, sem_k, chance_k = res_kept[:5]

        if tv_r is None and tv_k is None:
            ax0.text(0.5, 0.5, 'No data', ha='center', va='center',
                     transform=ax0.transAxes, color=_FG, fontsize=9)
            _style_ax(ax0)
        else:
            chance_level = chance_r if chance_r is not None else chance_k
            sig_r = cluster_permutation_test(acc_r, chance_level, n_perm=n_perm,
                                              cluster_alpha=cluster_alpha, alpha=alpha) \
                if tv_r is not None and acc_r.shape[0] >= 2 else None
            sig_k = cluster_permutation_test(acc_k, chance_level, n_perm=n_perm,
                                              cluster_alpha=cluster_alpha, alpha=alpha) \
                if tv_k is not None and acc_k.shape[0] >= 2 else None
            plot_band_panel(ax0, roi, band,
                             tv_r, mean_r, sem_r, sig_r,
                             tv_k, mean_k, sem_k, sig_k,
                             chance_level, is_bottom_row, show_title, show_flag_labels=show_title)

        ax0.annotate(BAND_LABELS.get(band, band).replace('\n', '  '),
                     xy=(-0.16, 0.5), xycoords='axes fraction',
                     fontsize=9, color=_FG, ha='right', va='center',
                     rotation=90, fontweight='bold')

        # -- col 1: ipsi vs contra amplitude --
        ax1 = fig.add_subplot(gs[r_idx, 1])
        tv_ir, mean_ipsi_r, sem_ipsi_r, mean_contra_r, sem_contra_r = \
            aggregate_ipsi_contra(ipsi_contra_removed, band)
        tv_ik, mean_ipsi_k, sem_ipsi_k, mean_contra_k, sem_contra_k = \
            aggregate_ipsi_contra(ipsi_contra_kept, band)
        plot_ipsi_contra_panel(ax1, roi, band,
                                tv_ir, mean_ipsi_r, sem_ipsi_r, mean_contra_r, sem_contra_r,
                                tv_ik, mean_ipsi_k, sem_ipsi_k, mean_contra_k, sem_contra_k,
                                is_bottom_row, show_title, show_flag_labels=show_title)

    fig.suptitle(f'P=2 (Left/Right) Case Study  |  {roi.capitalize()}  |  Amplitude only  |  {voxRes}\n'
                 f'left: LOO accuracy, dots = cluster permutation vs chance p<{alpha}  |  '
                 f'right: ipsi/contra baselined amplitude, mean +/- SEM across subjects',
                 color=_FG, fontsize=10.5, fontweight='bold', y=1.0)
    fig.text(0.5, 0.01, 'Time (s)', ha='center', va='bottom', color=_FG, fontsize=9)

    os.makedirs(outdir_fig, exist_ok=True)
    fpath = os.path.join(outdir_fig, f'two_class_scenario_{roi}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


def main():
    parser = argparse.ArgumentParser(
        description='Plot the P=2 (left/right) case study: ridge-LOO classifier accuracy '
                     'and ipsi/contra visual amplitude, both ERP removed vs. ERP kept.')
    parser.add_argument('--voxRes', default='8mm')
    parser.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    parser.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    parser.add_argument('--roi', default='visual')
    parser.add_argument('--erp_removed_dir', required=True,
                         help='Directory with the classifier ERP-removed per-subject .npz files.')
    parser.add_argument('--erp_kept_dir', required=True,
                         help='Directory with the classifier ERP-kept per-subject .npz files.')
    parser.add_argument('--ipsi_contra_removed_dir', required=True,
                         help='Directory with ipsi_contra_cell.py ERP-removed .npz files.')
    parser.add_argument('--ipsi_contra_kept_dir', required=True,
                         help='Directory with ipsi_contra_cell.py ERP-kept .npz files.')
    parser.add_argument('--n_perm', type=int, default=1000)
    parser.add_argument('--cluster_alpha', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--figdir', required=True,
                         help='Directory to save the figure.')
    args = parser.parse_args()

    bids_root = get_bids_root()

    print(f'Loading classifier ERP-removed data from: {args.erp_removed_dir}')
    all_data_removed = load_all_subjects(
        args.subjects, bids_root, args.voxRes, args.bands, [args.roi],
        [CONDITION], [SCHEME], outdir=args.erp_removed_dir)

    print(f'Loading classifier ERP-kept data from: {args.erp_kept_dir}')
    all_data_kept = load_all_subjects(
        args.subjects, bids_root, args.voxRes, args.bands, [args.roi],
        [CONDITION], [SCHEME], outdir=args.erp_kept_dir)

    print(f'Loading ipsi/contra ERP-removed data from: {args.ipsi_contra_removed_dir}')
    ipsi_contra_removed = load_ipsi_contra_all_subjects(
        args.subjects, bids_root, args.voxRes, args.bands, args.ipsi_contra_removed_dir)

    print(f'Loading ipsi/contra ERP-kept data from: {args.ipsi_contra_kept_dir}')
    ipsi_contra_kept = load_ipsi_contra_all_subjects(
        args.subjects, bids_root, args.voxRes, args.bands, args.ipsi_contra_kept_dir)

    n_removed = sum(1 for d in all_data_removed if d)
    n_kept    = sum(1 for d in all_data_kept if d)
    n_ic_removed = sum(1 for d in ipsi_contra_removed if any(v is not None for v in d.values()))
    n_ic_kept    = sum(1 for d in ipsi_contra_kept if any(v is not None for v in d.values()))
    print(f'Loaded classifier ERP-removed: {n_removed}/{len(args.subjects)} | '
          f'ERP-kept: {n_kept}/{len(args.subjects)} | '
          f'ipsi/contra ERP-removed: {n_ic_removed}/{len(args.subjects)} | '
          f'ipsi/contra ERP-kept: {n_ic_kept}/{len(args.subjects)}')

    make_figure(args.bands, args.roi, all_data_removed, all_data_kept,
                ipsi_contra_removed, ipsi_contra_kept, args.voxRes, args.figdir,
                n_perm=args.n_perm, cluster_alpha=args.cluster_alpha, alpha=args.alpha)

    print('\nDone.')


if __name__ == '__main__':
    main()
