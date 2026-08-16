#!/usr/bin/env python3
"""
plot_two_class_scenario.py

Focused visual-ROI case study, theta/alpha/beta only, across three decoders
x three feature conditions x two ERP states -- see chat history / module
docstring of run_two_class_scenario.sh for the full motivation.

Produces THREE separate figures, each rows=bands(3) x cols=conditions(3) x
ERP-state(2)=6, columns ordered [ampOnly-kept, ampOnly-removed,
ampPhase-kept, ampPhase-removed, phaseOnly-kept, phaseOnly-removed] so each
condition's kept/removed pair sits side by side:
    1. two_class_scenario_leftright_<roi>_<voxRes>.png
       LOO ridge binary classifier, left vs right (scheme=2)
    2. two_class_scenario_topbottom_<roi>_<voxRes>.png
       LOO ridge binary classifier, top vs bottom (scheme=3)
    3. two_class_scenario_circular_<roi>_<voxRes>.png
       LOO ridge circular regression (sin/cos targets), all 10 locations

Each row (band) shares one y-axis across all 6 of its panels -- same
decoder/chance-level throughout a row, so conditions and ERP states are
directly comparable by eye.

IMPORTANT caveat verified analytically AND numerically (see chat history):
BOTH decoders here z-score every feature across trials at each
independently-evaluated timepoint before fitting -- see
linear_decoding_categories_cell.py's ridge_ovr_timeseries (mu/sd computed
from X_t.mean(axis=0)/X_t.std(axis=0) at each t) and decoding_ts_cell.py's
ridge_loocv_timeseries (identical pattern, X_t.mean(axis=0)/X_t.std(axis=0)
inside the per-timepoint loop). That per-timepoint z-scoring subtracts out
any trial-invariant constant -- exactly what ERP removal subtracts (E[t,f]
is by definition the same value added to every trial at time t, feature f)
-- so X_z is bit-identical whether or not remove_erp ran, for BOTH decoders,
regardless of feature condition. remove_erp is therefore a mathematically
forced no-op for every panel in figures 1-3 above. This was directly
confirmed via np.allclose on saved real accuracy arrays earlier this
session (max diff ~1e-7, float rounding only) -- it is not a bug, and both
ERP states are still run/plotted purely as a live sanity check on this
claim (any visible difference between a kept/removed pair would indicate
something has actually changed in the pipeline).

Reuses linear_decoding_categories_cell.output_path + plot_linear_decoding_
categories.py's aggregate_lindecode/cluster_permutation_test (classifiers),
and decoding_ts_cell.output_path + plot_decoding_ts.py's aggregate_timeseries
(circular regression, reimplemented as a thin direct .npz loader here rather
than plot_decoding_ts.load_all_subjects, which also attaches behavioral
data via an extra G03 load per subject that this figure doesn't need).

Usage:
    python plot_two_class_scenario.py [--voxRes 8mm]
                                       [--bands theta alpha beta]
                                       [--roi visual]
                                       [--conditions ampOnly ampPhase phaseOnly]
                                       [--subjects 1 2 ...]
                                       [--classifier_removed_dir <path>] [--classifier_kept_dir <path>]
                                       [--circular_removed_dir <path>] [--circular_kept_dir <path>]
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

from constants import SUBJECT_LIST, get_bids_root
from plot_linear_decoding_categories import (
    load_all_subjects as load_classifier_all_subjects,
    aggregate_lindecode, cluster_permutation_test,
    _style_ax, _BG, _FG, _GRID, _FLAG_LINE, _FLAG_TXT, _CHANCE,
    ROI_COLOURS, BAND_LABELS, EVENT_FLAGS,
)
from decoding_ts_cell import output_path as circular_output_path
from plot_decoding_ts import aggregate_timeseries as aggregate_circular

CONDITION_LABELS = {'ampOnly': 'Amp', 'ampPhase': 'Amp+Phase', 'phaseOnly': 'Phase'}
ERP_LABELS = {True: 'removed', False: 'kept'}
_SIG_COLOUR = '#ffffff'
CIRCULAR_CHANCE = 90.0   # deg -- expected mean circular error under chance


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


def plot_scalar_panel(ax, colour, tv, mean_c, sem_c, sig, chance_level, ylim,
                       is_bottom_row, show_title, show_flag_labels, title_str, ylabel):
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
    ax.set_ylabel(ylabel, fontsize=7.5, color=_FG)

    if show_title:
        ax.set_title(title_str, fontsize=8.5, color=_FG, pad=4)


def row_ylim(results, chance_level):
    """results: list of (tv, mean, sem) tuples (any subset may be (None,None,None))."""
    lo_candidates, hi_candidates = [chance_level], [chance_level]
    for tv, mean_c, sem_c in results:
        if tv is None:
            continue
        lo_candidates.append(np.nanmin(mean_c - sem_c))
        hi_candidates.append(np.nanmax(mean_c + sem_c))
    lo, hi = min(lo_candidates), max(hi_candidates)
    pad = max(1e-6, (hi - lo) * 0.15)
    return lo - pad, hi + pad


# ── Circular regression: minimal direct loader (no behavior attach) ─────────

def load_circular_all_subjects(subjects, bids_root, voxRes, bands, roi, conditions, outdir):
    """Returns list (one dict per subject) of {(band, roi, condition): npz dict or absent}
    -- same key convention as plot_decoding_ts.load_all_subjects/aggregate_timeseries
    expect, built via a direct .npz read (skips that loader's per-subject
    behavioral-alignment G03 load, which this figure doesn't need)."""
    all_data = []
    for subjID in subjects:
        d = {}
        for band in bands:
            for condition in conditions:
                fp = circular_output_path(bids_root, subjID, band, roi, condition, voxRes, outdir)
                if os.path.exists(fp):
                    d[(band, roi, condition)] = dict(np.load(fp, allow_pickle=True))
        all_data.append(d if d else None)
    return all_data


# ── Figure assembly (shared by all three figures) ────────────────────────────

def make_figure(bands, roi, conditions, cell_getter, chance_level, ylabel, fname_tag,
                 voxRes, outdir_fig, colour, title_prefix, n_perm, cluster_alpha, alpha,
                 lower_is_better=False):
    """
    cell_getter(band, condition, remove_erp) -> (tv, acc_or_err_matrix, mean, sem)
    (acc_or_err_matrix: (n_subj, T), used for the cluster-permutation test).
    """
    n_bands = len(bands)
    col_specs = [(cond, remove_erp) for cond in conditions for remove_erp in (False, True)]
    n_cols = len(col_specs)

    fig_w, row_h = 3.6 * n_cols, 2.2
    fig_h = row_h * n_bands + 0.9
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_bands, n_cols, figure=fig, hspace=0.5, wspace=0.35,
                            left=0.045, right=0.99,
                            top=1 - 0.6 / fig_h, bottom=0.35 / fig_h)

    for r_idx, band in enumerate(bands):
        is_bottom_row = (r_idx == n_bands - 1)
        show_title    = (r_idx == 0)

        cells = [cell_getter(band, cond, remove_erp) for (cond, remove_erp) in col_specs]
        ylim = row_ylim([(tv, mean_c, sem_c) for (tv, _, mean_c, sem_c) in cells], chance_level)

        for c_idx, ((cond, remove_erp), (tv, mat, mean_c, sem_c)) in enumerate(zip(col_specs, cells)):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            sig = None
            if tv is not None and mat is not None and mat.shape[0] >= 2:
                sig = cluster_permutation_test(mat, chance_level, n_perm=n_perm,
                                                cluster_alpha=cluster_alpha, alpha=alpha)
            title_str = f'{CONDITION_LABELS.get(cond, cond)} -- ERP {ERP_LABELS[remove_erp]}'
            plot_scalar_panel(ax, colour, tv, mean_c, sem_c, sig, chance_level, ylim,
                               is_bottom_row, show_title, show_title, title_str, ylabel)
            if c_idx == 0:
                ax.annotate(BAND_LABELS.get(band, band).replace('\n', '  '),
                            xy=(-0.28, 0.5), xycoords='axes fraction',
                            fontsize=9, color=_FG, ha='right', va='center',
                            rotation=90, fontweight='bold')

    better = 'lower = better' if lower_is_better else 'higher = better'
    fig.suptitle(f'{title_prefix}  |  {roi.capitalize()}  |  {voxRes}\n'
                 f'each row (band) shares one y-axis across all 6 panels ({better})  |  '
                 f'dots = cluster permutation vs chance p<{alpha}',
                 color=_FG, fontsize=10.5, fontweight='bold', y=1.0)
    fig.text(0.5, 0.01, 'Time (s)', ha='center', va='bottom', color=_FG, fontsize=9)

    os.makedirs(outdir_fig, exist_ok=True)
    fpath = os.path.join(outdir_fig, f'two_class_scenario_{fname_tag}_{roi}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


def main():
    parser = argparse.ArgumentParser(
        description='Plot the visual-ROI case study: left/right classifier, top/bottom '
                     'classifier, and circular regression, each ERP kept vs. ERP removed, '
                     'across ampOnly/ampPhase/phaseOnly.')
    parser.add_argument('--voxRes', default='8mm')
    parser.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    parser.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    parser.add_argument('--roi', default='visual')
    parser.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase', 'phaseOnly'])
    parser.add_argument('--classifier_removed_dir', required=True)
    parser.add_argument('--classifier_kept_dir', required=True)
    parser.add_argument('--circular_removed_dir', required=True)
    parser.add_argument('--circular_kept_dir', required=True)
    parser.add_argument('--n_perm', type=int, default=1000)
    parser.add_argument('--cluster_alpha', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--figdir', required=True)
    args = parser.parse_args()

    bids_root = get_bids_root()

    print(f'Loading classifier ERP-removed data from: {args.classifier_removed_dir}')
    clf_removed = load_classifier_all_subjects(
        args.subjects, bids_root, args.voxRes, args.bands, [args.roi],
        args.conditions, [2, 3], outdir=args.classifier_removed_dir)
    print(f'Loading classifier ERP-kept data from: {args.classifier_kept_dir}')
    clf_kept = load_classifier_all_subjects(
        args.subjects, bids_root, args.voxRes, args.bands, [args.roi],
        args.conditions, [2, 3], outdir=args.classifier_kept_dir)

    print(f'Loading circular ERP-removed data from: {args.circular_removed_dir}')
    circ_removed = load_circular_all_subjects(
        args.subjects, bids_root, args.voxRes, args.bands, args.roi,
        args.conditions, args.circular_removed_dir)
    print(f'Loading circular ERP-kept data from: {args.circular_kept_dir}')
    circ_kept = load_circular_all_subjects(
        args.subjects, bids_root, args.voxRes, args.bands, args.roi,
        args.conditions, args.circular_kept_dir)

    def clf_getter(scheme):
        def _get(band, cond, remove_erp):
            data = clf_removed if remove_erp else clf_kept
            tv, acc_mat, mean_acc, sem_acc, chance = aggregate_lindecode(
                data, band, args.roi, cond, scheme)[:5]
            return tv, acc_mat, mean_acc, sem_acc
        return _get

    def circ_getter(band, cond, remove_erp):
        data = circ_removed if remove_erp else circ_kept
        tv, err_mat, mean_err, sem_err = aggregate_circular(data, band, args.roi, cond)
        return tv, err_mat, mean_err, sem_err

    make_figure(args.bands, args.roi, args.conditions, clf_getter(2), 0.5,
                'LOO accuracy', 'leftright', args.voxRes, args.figdir, ROI_COLOURS[args.roi],
                'LOO Ridge Classifier -- Left vs Right (scheme=2)',
                args.n_perm, args.cluster_alpha, args.alpha, lower_is_better=False)

    make_figure(args.bands, args.roi, args.conditions, clf_getter(3), 0.5,
                'LOO accuracy', 'topbottom', args.voxRes, args.figdir, ROI_COLOURS[args.roi],
                'LOO Ridge Classifier -- Top vs Bottom (scheme=3)',
                args.n_perm, args.cluster_alpha, args.alpha, lower_is_better=False)

    make_figure(args.bands, args.roi, args.conditions, circ_getter, CIRCULAR_CHANCE,
                'Circular error (deg)', 'circular', args.voxRes, args.figdir, ROI_COLOURS[args.roi],
                'LOO Ridge Circular Regression -- 10 locations (sin/cos)',
                args.n_perm, args.cluster_alpha, args.alpha, lower_is_better=True)

    print('\nDone.')


if __name__ == '__main__':
    main()
