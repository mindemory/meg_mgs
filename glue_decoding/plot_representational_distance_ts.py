#!/usr/bin/env python3
"""
plot_representational_distance_ts.py

Aggregates representational_distance_ts_cell.py's per-subject .npz files and
plots the same-location-vs-different-location NORMALIZED distance gap
(between-within)/(between+within), bounded [-1, 1], over time, bands x rois
grid (mirrors plot_decoding_ts.py's make_timeseries_figure layout/styling
exactly, so the two are visually and temporally comparable).

Positive gap = same-category trials sit closer together than
different-category trials (real spatial structure). Zero = no structure.

Real line: mean +/- SEM of the per-subject gap across subjects.
Horizontal line at 0: theoretical no-structure reference.
Significance: two-sided Wilcoxon signed-rank test of the per-subject
gap(t) against 0 across subjects, FDR (Benjamini-Hochberg) corrected across
this PANEL's own timepoints (i.e. per band/roi/scheme/condition -- not
corrected across panels), marked as a dot row under each panel where
significant. This is the group-level test that actually matters here --
NOT the per-subject label-permutation null saved in each .npz (null_mean/
null_std/p_value), which reflects single-subject permutation spread, a
different and not directly comparable quantity to cross-subject SEM/CI (see
chat history) -- those fields are loaded/computed by
representational_distance_ts_cell.py but intentionally not plotted here.

Usage:
    python plot_representational_distance_ts.py [--voxRes 8mm]
                                                  [--bands theta alpha beta lowgamma highgamma]
                                                  [--rois visual parietal frontal]
                                                  [--conditions ampOnly]
                                                  [--subjects 1 2 ...]
                                                  [--alpha 0.05]
                                                  [--outdir <path>] [--figdir <path>]
"""

import os, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from constants import SUBJECT_LIST, ROI_NAMES, CATEGORY_SCHEMES, get_bids_root
from representational_distance_ts_cell import output_path

# NOTE: category count uses len(groups), NOT the scheme key `s` -- they
# diverge as of scheme 3 (top_bottom, 2 categories, not 3; see constants.py).
SCHEME_LABELS = {s: f"{CATEGORY_SCHEMES[s]['name']} ({len(CATEGORY_SCHEMES[s]['groups'])} categories)"
                  for s in CATEGORY_SCHEMES}

# ── Design constants (mirrors plot_decoding_ts.py) ─────────────────────────────

_BG        = '#000000'
_FG        = '#e0e0e0'
_GRID      = '#1c1c1c'
_FLAG_LINE = '#777777'
_FLAG_TXT  = '#cccccc'
_ZERO      = '#444444'     # horizontal no-structure line
_SIG       = '#ffffff'     # significance marker dots

ROI_COLOURS = {
    'visual':   '#FFC629',
    'parietal': '#A78BFA',
    'frontal':  '#34D399',
}

BAND_LABELS = {
    'theta':     'Theta\n(4-8 Hz)',
    'alpha':     'Alpha\n(8-12 Hz)',
    'beta':      'Beta\n(13-30 Hz)',
    'lowgamma':  'Low gamma\n(30-80 Hz)',
    'highgamma': 'High gamma\n(80-150 Hz)',
}

EVENT_FLAGS = [
    (0.0,  'Stim',        0.93),
    (0.2,  'Delay Onset', 0.55),
]

# ── I/O ─────────────────────────────────────────────────────────────────────

def load_all_subjects(subjects, bids_root, voxRes, bands, rois, conditions, schemes, outdir=None):
    """Returns list (one entry per subject) of dicts: data[(band, roi, condition, scheme)] = npz dict or None."""
    all_data = []
    for subjID in subjects:
        d = {}
        for band in bands:
            for roi in rois:
                for condition in conditions:
                    for scheme in schemes:
                        fp = output_path(bids_root, subjID, band, roi, condition, voxRes, scheme, outdir)
                        if os.path.exists(fp):
                            d[(band, roi, condition, scheme)] = dict(np.load(fp, allow_pickle=True))
        all_data.append(d if d else None)
    return all_data


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_repdist(all_data, band, roi, condition, scheme):
    """
    Returns (tv, gap_matrix, mean_gap, sem_gap) or (None, None, None, None).

    gap_matrix: (n_subj, T) raw per-subject gap(t), returned (not just its
    mean/SEM) so compute_significance can run the actual across-subject test
    on it rather than a lossy summary.
    mean_gap/sem_gap: arithmetic mean/SEM across subjects -- standard
    cross-subject inference on the statistic itself.
    """
    gap_list = []
    tv = None
    for d in all_data:
        if d is None:
            continue
        k = (band, roi, condition, scheme)
        if k not in d:
            continue
        npz = d[k]
        gap_list.append(npz['gap'])
        if tv is None:
            tv = npz['time_vector']

    if not gap_list:
        return None, None, None, None

    gap = np.stack(gap_list)   # (n_subj, T)
    n   = gap.shape[0]

    return tv, gap, gap.mean(0), gap.std(0) / np.sqrt(n)


def _fdr_bh(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR: returns a bool significance mask, same shape as pvals."""
    p = np.asarray(pvals)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    sig_sorted = np.zeros(n, dtype=bool)
    if below.any():
        sig_sorted[:np.max(np.where(below)) + 1] = True
    sig = np.zeros(n, dtype=bool)
    sig[order] = sig_sorted
    return sig


def compute_significance(gap_matrix, alpha=0.05):
    """
    Two-sided Wilcoxon signed-rank test of gap(t) against 0 across subjects
    (gap_matrix: (n_subj, T)), one test per timepoint, FDR (Benjamini-
    Hochberg) corrected across this array's own T timepoints. Returns a
    (T,) bool significance mask. A timepoint with < 2 subjects, all-zero
    differences, or a scipy ValueError (e.g. all-identical values) is
    treated as p=1 (not significant) rather than raising.
    """
    n_subj, n_times = gap_matrix.shape
    pvals = np.ones(n_times)
    for t in range(n_times):
        vals = gap_matrix[:, t]
        vals = vals[~np.isnan(vals)]
        if vals.size < 2 or np.allclose(vals, 0):
            continue
        try:
            _, p = stats.wilcoxon(vals)
            pvals[t] = p
        except ValueError:
            pass
    return _fdr_bh(pvals, alpha=alpha)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _style_ax(ax, spine_col='#333333'):
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color(spine_col)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)


def plot_repdist_panel(ax, tv, mean_gap, sem_gap, sig_mask,
                        roi, is_bottom_row, is_left_col, show_title, title_str,
                        show_flag_labels=True):
    col = ROI_COLOURS[roi]
    t0, t1 = float(tv[0]), float(tv[-1])

    ax.axhline(0, color=_ZERO, lw=0.8, ls=':', zorder=1)

    # Real gap: mean +/- SEM across subjects
    ax.fill_between(tv, mean_gap - sem_gap, mean_gap + sem_gap,
                     alpha=0.30, color=col, zorder=3)
    ax.plot(tv, mean_gap, color=col, lw=1.5, zorder=3)

    lo = min(np.nanmin(mean_gap - sem_gap), 0)
    hi = max(np.nanmax(mean_gap + sem_gap), 0)
    pad = max(1e-6, (hi - lo) * 0.15)
    lo, hi = lo - pad, hi + pad

    # Significance dots (Wilcoxon vs 0 across subjects, FDR-corrected --
    # see compute_significance) along a fixed row near the panel bottom,
    # rather than shading the line itself, so significant/non-significant
    # stretches stay legible even when they're brief or scattered.
    if sig_mask is not None and sig_mask.any():
        y_sig = lo + 0.06 * (hi - lo)
        ax.plot(tv[sig_mask], np.full(sig_mask.sum(), y_sig),
                '.', color=_SIG, ms=3.0, zorder=6)

    ax.set_xlim(t0, t1)
    ax.set_ylim(lo, hi)

    y_lo, y_hi = ax.get_ylim()
    for (t_flag, label, y_frac) in EVENT_FLAGS:
        ax.axvline(t_flag, color=_FLAG_LINE, lw=0.8, ls='--', zorder=4)
        if show_flag_labels:
            ax.text(t_flag, y_lo + y_frac * (y_hi - y_lo), label,
                    rotation=90, va='top', ha='right',
                    fontsize=6.5, color=_FLAG_TXT, zorder=5)

    if is_left_col:
        ax.set_ylabel('Discriminability index\n(between-within)/(between+within)',
                       fontsize=7, color=_FG)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    if not is_bottom_row:
        ax.set_xticklabels([])

    ax.grid(True, color=_GRID, lw=0.4, zorder=0)
    _style_ax(ax)

    if show_title:
        ax.set_title(title_str, color=_FG, fontsize=10, fontweight='bold', pad=4)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))


def _band_row_grid(n_bands, n_rois, row_h_in, fig_w_per_roi=4.2,
                    hspace=0.45, title_margin_in=0.65, bottom_margin_in=0.15):
    fig_w = fig_w_per_roi * n_rois
    fig_h = row_h_in * n_bands + title_margin_in + bottom_margin_in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(
        n_bands, n_rois, figure=fig, hspace=hspace, wspace=0.30,
        left=0.13, right=0.97,
        top=1 - title_margin_in / fig_h, bottom=bottom_margin_in / fig_h,
    )
    return fig, gs


def make_timeseries_figure(all_data, bands, rois, condition, scheme, voxRes, outdir_fig, alpha=0.05):
    n_bands, n_rois = len(bands), len(rois)
    fig, gs = _band_row_grid(n_bands, n_rois, row_h_in=1.3, bottom_margin_in=0.40)

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois):
            ax_ts = fig.add_subplot(gs[r_idx, c_idx])

            tv, gap_matrix, mean_gap, sem_gap = aggregate_repdist(
                all_data, band, roi, condition, scheme)

            is_left_col   = (c_idx == 0)
            is_bottom_row = (r_idx == n_bands - 1)
            show_title    = (r_idx == 0)
            title_str     = roi.capitalize()

            if tv is not None:
                sig_mask = compute_significance(gap_matrix, alpha=alpha) \
                    if gap_matrix.shape[0] >= 2 else None
                plot_repdist_panel(
                    ax_ts, tv, mean_gap, sem_gap, sig_mask,
                    roi, is_bottom_row, is_left_col, show_title, title_str,
                    show_flag_labels=show_title)
            else:
                ax_ts.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax_ts.transAxes, color=_FG, fontsize=9)
                _style_ax(ax_ts)

            if c_idx == 0:
                ax_ts.annotate(BAND_LABELS.get(band, band),
                               xy=(-0.42, 0.5), xycoords='axes fraction',
                               fontsize=9, color=_FG, ha='right', va='center',
                               rotation=90, fontweight='bold')

    cond_label   = 'Amplitude only' if condition == 'ampOnly' else 'Amplitude + Phase'
    scheme_label = SCHEME_LABELS.get(scheme, str(scheme))
    fig.suptitle(f'Representational Distance Discriminability  |  {cond_label}  |  '
                 f'{scheme_label}  |  {voxRes}  |  dots = Wilcoxon vs 0, FDR q<{alpha}',
                 color=_FG, fontsize=13, fontweight='bold', y=0.97)
    fig.text(0.5, 0.005, 'Time (s)', ha='center', va='bottom', color=_FG, fontsize=9)

    os.makedirs(outdir_fig, exist_ok=True)
    fpath = os.path.join(outdir_fig, f'repdist_ts_{condition}_scheme{scheme}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and plot representational-distance timeseries across subjects.')
    parser.add_argument('--voxRes',     default='8mm')
    parser.add_argument('--subjects',   nargs='+', type=int, default=SUBJECT_LIST)
    parser.add_argument('--bands',      nargs='+',
                        default=['theta', 'alpha', 'beta', 'lowgamma', 'highgamma'])
    parser.add_argument('--rois',       nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions', nargs='+', default=['ampOnly'])
    parser.add_argument('--schemes',    nargs='+', type=int, default=[2, 4, 6, 10],  # NOT sorted(CATEGORY_SCHEMES) -- scheme 3 (top_bottom) is opt-in only (two_class_scenario), not part of this pipeline's standard sweep
                        choices=sorted(CATEGORY_SCHEMES))
    parser.add_argument('--alpha',      type=float, default=0.05,
                        help='FDR (Benjamini-Hochberg) significance threshold for the '
                             'per-timepoint Wilcoxon-vs-0 test (default 0.05).')
    parser.add_argument('--outdir',     default=None,
                        help='Directory containing the per-subject .npz files.')
    parser.add_argument('--figdir',     default=None,
                        help='Directory to save figures (default: same as outdir '
                             'or BIDS derivatives/glueDecoding/repDistTS).')
    args = parser.parse_args()

    bids_root = get_bids_root()
    figdir = args.figdir or (args.outdir if args.outdir else
                              os.path.join(bids_root, 'derivatives', 'glueDecoding', 'repDistTS'))

    print(f'Loading data | {args.voxRes} | subjects={args.subjects} | '
          f'bands={args.bands} | rois={args.rois} | conditions={args.conditions} | '
          f'schemes={args.schemes}')

    all_data = load_all_subjects(
        args.subjects, bids_root, args.voxRes,
        args.bands, args.rois, args.conditions, args.schemes, args.outdir)

    n_loaded = sum(1 for d in all_data if d)
    print(f'Loaded data for {n_loaded}/{len(args.subjects)} subjects.')

    for condition in args.conditions:
        for scheme in args.schemes:
            print(f'\n-- Plotting condition: {condition} | scheme: {scheme} --')
            make_timeseries_figure(all_data, args.bands, args.rois, condition, scheme,
                                    args.voxRes, figdir, alpha=args.alpha)

    print('\nAll figures saved.')


if __name__ == '__main__':
    main()
