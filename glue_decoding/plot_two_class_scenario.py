#!/usr/bin/env python3
"""
plot_two_class_scenario.py

Focused P=2 (left/right) case study, visual ROI only, theta/alpha/beta only,
ampOnly. One figure, 3 rows (theta/alpha/beta) x 4 cols:
    col 0: ipsi/contra visual amplitude, ERP kept
    col 1: ipsi/contra visual amplitude, ERP removed
    col 2: P=2 ridge-LOO classifier accuracy, ERP kept
    col 3: P=2 ridge-LOO classifier accuracy, ERP removed
-- ERP kept/removed as separate panels (not overlaid) so each is readable on
its own. The classifier pair (cols 2-3) SHARE one y-axis scale (harmless --
see caveat below, they're bit-identical). The amplitude pair (cols 0-1)
deliberately do NOT share a y-axis: the common per-hemisphere-ROI response
(and any raw left/right ROI scale offset) that ERP removal strips out is
typically much larger than the residual ipsi/contra difference, so a shared
axis flattens the ERP-removed panel to an indistinguishable near-zero line.
Each amplitude panel is auto-scaled to its own data and annotates its exact
y-range (top-left) so the two panels' relative scale is still legible.
See chat history / module docstrings of run_two_class_scenario.sh,
linear_decoding_categories_cell.py, and ipsi_contra_cell.py for the full
motivation and each computation's details.

IMPORTANT caveat verified analytically AND numerically (see chat history):
for the classifier columns, remove_erp has ZERO effect on the plotted
accuracy. Every evaluated timepoint is independently z-scored across trials
inside ridge_ovr_timeseries (mu = trial-mean, sd = trial-std, both computed
AFTER ERP removal or not); subtracting each cell's grand trial-mean before
that z-scoring step is subtracting a value that is IDENTICAL across every
trial at that timepoint (a pure constant shift), which the very next
z-scoring step removes again as `mu` regardless of whether it was
pre-subtracted -- and it doesn't change the trial-to-trial variance either
(Var(X - c) = Var(X) for constant c), so `sd` is unaffected too. X_z is
therefore bit-identical whether or not remove_erp was applied, for this
classifier specifically (confirmed via np.allclose on real runs). This is
NOT true for the amplitude columns, which have no such per-timepoint
re-centering step -- removing the common stimulus-locked ERP there genuinely
changes the plotted curves, since it's plotting a trial-averaged signal, not
a re-standardized-per-timepoint one. Both ERP variants are still plotted for
the classifier as a sanity check on this exact-equivalence claim (any
visible difference there would indicate a bug), but if compute time ever
matters, the ERP-kept classifier runs are fully redundant -- unlike the
amplitude column's ERP-kept runs, which are not.

Reuses plot_linear_decoding_categories.py's load_all_subjects,
cluster_permutation_test, and design constants directly (classifier
columns), and ipsi_contra_cell.py's saved per-subject .npz layout directly
(amplitude columns) -- no duplicated computation logic.

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

_SIG_COLOUR = '#ffffff'


def _dim_colour(colour, factor=0.5):
    """Lighter/desaturated variant of `colour` -- same recipe as
    plot_linear_decoding_categories.py's state_shades."""
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(colour))
    return mcolors.hsv_to_rgb((h, s * factor, min(1.0, v * 1.1 + 0.1)))


def _finish_axes(ax, t0, t1, lo, hi, is_bottom_row, show_flag_labels):
    ax.set_xlim(t0, t1)
    ax.set_ylim(lo, hi)
    y_lo, y_hi = ax.get_ylim()
    for (t_flag, label, y_frac) in EVENT_FLAGS:
        ax.axvline(t_flag, color=_FLAG_LINE, lw=0.8, ls='--', zorder=4)
        if show_flag_labels:
            ax.text(t_flag, y_lo + y_frac * (y_hi - y_lo), label,
                    rotation=90, va='top', ha='right', fontsize=6.5, color=_FLAG_TXT, zorder=5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    if not is_bottom_row:
        ax.set_xticklabels([])
    ax.grid(True, color=_GRID, lw=0.4, zorder=0)
    _style_ax(ax)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


# ── Classifier accuracy panel (one ERP state) ────────────────────────────────

def classifier_row_ylim(res_kept, res_removed, chance_level):
    lo_candidates, hi_candidates = [chance_level], [chance_level]
    for res in (res_kept, res_removed):
        tv, acc, mean_c, sem_c, _ = res[:5]
        if tv is None:
            continue
        lo_candidates.append(np.nanmin(mean_c - sem_c))
        hi_candidates.append(np.nanmax(mean_c + sem_c))
    lo, hi = min(lo_candidates), max(hi_candidates)
    pad = max(1e-6, (hi - lo) * 0.15)
    return lo - pad, hi + pad


def plot_classifier_panel(ax, roi, band, tv, mean_c, sem_c, sig, chance_level,
                           ylim, is_bottom_row, show_title, show_flag_labels, erp_label):
    colour = ROI_COLOURS[roi]
    ax.axhline(chance_level, color=_CHANCE, lw=0.8, ls=':', zorder=1)

    if tv is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color=_FG, fontsize=9)
        _style_ax(ax)
        return

    ax.fill_between(tv, mean_c - sem_c, mean_c + sem_c, alpha=0.28, color=colour, zorder=3)
    ax.plot(tv, mean_c, color=colour, lw=1.6, zorder=4)

    lo, hi = ylim
    if sig is not None and sig.any():
        y_sig = lo + 0.06 * (hi - lo)
        ax.plot(tv[sig], np.full(sig.sum(), y_sig), '.', color=_SIG_COLOUR, ms=3.0, zorder=6)

    t0, t1 = float(tv[0]), float(tv[-1])
    _finish_axes(ax, t0, t1, lo, hi, is_bottom_row, show_flag_labels)
    ax.set_ylabel('LOO accuracy', fontsize=8, color=_FG)

    if show_title:
        ax.set_title(f'LOO ridge accuracy -- ERP {erp_label}', fontsize=9, color=_FG, pad=4)


# ── Ipsi/contra amplitude panel (one ERP state) ──────────────────────────────

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


def plot_amp_panel(ax, roi, band, tv, mean_ipsi, sem_ipsi, mean_contra, sem_contra,
                    is_bottom_row, show_title, show_flag_labels, erp_label):
    """
    NOTE: deliberately auto-scales its OWN y-range instead of sharing one
    with its ERP-kept/removed partner (unlike the classifier panels). The
    common per-hemisphere-ROI response removed by ERP subtraction here is
    typically much larger than the residual ipsi/contra difference, so a
    shared axis flattens the ERP-removed panel to an indistinguishable
    near-zero line -- the annotated y-range (top-left) is what lets you
    still judge the two panels' relative scale despite the different axes.
    """
    ipsi_colour   = ROI_COLOURS[roi]
    contra_colour = _dim_colour(ipsi_colour)

    ax.axhline(0.0, color=_CHANCE, lw=0.8, ls=':', zorder=1)

    if tv is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color=_FG, fontsize=9)
        _style_ax(ax)
        return

    ax.fill_between(tv, mean_ipsi - sem_ipsi, mean_ipsi + sem_ipsi,
                     alpha=0.25, color=ipsi_colour, zorder=3)
    ax.plot(tv, mean_ipsi, color=ipsi_colour, lw=1.6, zorder=4, label='Ipsi')
    ax.fill_between(tv, mean_contra - sem_contra, mean_contra + sem_contra,
                     alpha=0.20, color=contra_colour, zorder=2)
    ax.plot(tv, mean_contra, color=contra_colour, lw=1.4, ls='--', zorder=3, label='Contra')

    lo = min(np.nanmin(mean_ipsi - sem_ipsi), np.nanmin(mean_contra - sem_contra))
    hi = max(np.nanmax(mean_ipsi + sem_ipsi), np.nanmax(mean_contra + sem_contra))
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    pad = max(1e-6, (hi - lo) * 0.15)
    lo, hi = lo - pad, hi + pad

    t0, t1 = float(tv[0]), float(tv[-1])
    _finish_axes(ax, t0, t1, lo, hi, is_bottom_row, show_flag_labels)
    ax.set_ylabel('Baselined amplitude (z)', fontsize=8, color=_FG)
    ax.text(0.02, 0.96, f'range: [{lo+pad:.3f}, {hi-pad:.3f}]', transform=ax.transAxes,
            fontsize=6.5, color=_FLAG_TXT, ha='left', va='top', zorder=7)

    if show_title:
        leg = ax.legend(fontsize=7, loc='upper right', framealpha=0.2,
                         edgecolor='#444444', labelcolor=_FG)
        leg.get_frame().set_facecolor('#1a1a1a')
        ax.set_title(f'Ipsi vs contra amplitude -- ERP {erp_label}', fontsize=9, color=_FG, pad=4)


# ── Figure assembly ──────────────────────────────────────────────────────────

def make_figure(bands, roi, all_data_removed, all_data_kept,
                 ipsi_contra_removed, ipsi_contra_kept, voxRes, outdir_fig,
                 n_perm=1000, cluster_alpha=0.05, alpha=0.05):
    n_bands = len(bands)
    fig_w, row_h = 15.5, 2.2
    fig_h = row_h * n_bands + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_bands, 4, figure=fig, hspace=0.5, wspace=0.32,
                            left=0.06, right=0.99,
                            top=1 - 0.6 / fig_h, bottom=0.35 / fig_h)

    for r_idx, band in enumerate(bands):
        is_bottom_row = (r_idx == n_bands - 1)
        show_title    = (r_idx == 0)

        # -- amplitude: kept vs removed, EACH auto-scales its own y-axis
        # (see plot_amp_panel docstring for why these are NOT shared) --
        res_amp_kept    = aggregate_ipsi_contra(ipsi_contra_kept, band)
        res_amp_removed = aggregate_ipsi_contra(ipsi_contra_removed, band)

        ax0 = fig.add_subplot(gs[r_idx, 0])
        plot_amp_panel(ax0, roi, band, *res_amp_kept,
                        is_bottom_row, show_title, show_title, 'kept')
        ax0.annotate(BAND_LABELS.get(band, band).replace('\n', '  '),
                     xy=(-0.20, 0.5), xycoords='axes fraction',
                     fontsize=9, color=_FG, ha='right', va='center',
                     rotation=90, fontweight='bold')

        ax1 = fig.add_subplot(gs[r_idx, 1])
        plot_amp_panel(ax1, roi, band, *res_amp_removed,
                        is_bottom_row, show_title, False, 'removed')

        # -- classifier: kept vs removed, shared y-lim --
        res_clf_kept    = aggregate_lindecode(all_data_kept,    band, roi, CONDITION, SCHEME)
        res_clf_removed = aggregate_lindecode(all_data_removed, band, roi, CONDITION, SCHEME)
        tv_k, acc_k, mean_k, sem_k, chance_k = res_clf_kept[:5]
        tv_r, acc_r, mean_r, sem_r, chance_r = res_clf_removed[:5]
        chance_level = chance_k if chance_k is not None else chance_r

        if chance_level is None:
            clf_ylim = (0.0, 1.0)
            sig_k = sig_r = None
        else:
            clf_ylim = classifier_row_ylim(res_clf_kept, res_clf_removed, chance_level)
            sig_k = cluster_permutation_test(acc_k, chance_level, n_perm=n_perm,
                                              cluster_alpha=cluster_alpha, alpha=alpha) \
                if tv_k is not None and acc_k.shape[0] >= 2 else None
            sig_r = cluster_permutation_test(acc_r, chance_level, n_perm=n_perm,
                                              cluster_alpha=cluster_alpha, alpha=alpha) \
                if tv_r is not None and acc_r.shape[0] >= 2 else None

        ax2 = fig.add_subplot(gs[r_idx, 2])
        plot_classifier_panel(ax2, roi, band, tv_k, mean_k, sem_k, sig_k, chance_level,
                               clf_ylim, is_bottom_row, show_title, show_title, 'kept')

        ax3 = fig.add_subplot(gs[r_idx, 3])
        plot_classifier_panel(ax3, roi, band, tv_r, mean_r, sem_r, sig_r, chance_level,
                               clf_ylim, is_bottom_row, show_title, False, 'removed')

    fig.suptitle(f'P=2 (Left/Right) Case Study  |  {roi.capitalize()}  |  Amplitude only  |  {voxRes}\n'
                 f'cols 1-2: ipsi/contra baselined amplitude, mean +/- SEM, each auto-scaled '
                 f'(see annotated range -- kept vs removed axes differ on purpose)  |  '
                 f'cols 3-4: LOO ridge accuracy (shared y-axis), dots = cluster permutation vs chance p<{alpha}',
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
        description='Plot the P=2 (left/right) case study: ipsi/contra visual amplitude and '
                     'ridge-LOO classifier accuracy, each ERP kept vs. ERP removed.')
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
