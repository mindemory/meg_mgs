#!/usr/bin/env python3
"""
plot_representational_distance_ts.py

Aggregates representational_distance_ts_cell.py's per-subject .npz files and
plots the same-location-vs-different-location distance gap over time,
bands x rois grid (mirrors plot_decoding_ts.py's make_timeseries_figure
layout/styling exactly, so the two are visually and temporally comparable).

Gap = mean(between-location distance) - mean(within-location distance) at
each timepoint. Positive = same-location trials sit closer together than
different-location trials (real spatial structure). Zero = no structure,
matching the null's expected center.

Real line: mean +/- SEM of the per-subject gap across subjects.
Reference band: mean(null_mean) +/- mean(null_std) across subjects --
the cross-subject average of each subject's own label-permutation null
(NOT a cross-subject SEM of the null center -- see aggregate_repdist
docstring for why that distinction matters).
Horizontal line at 0: theoretical no-structure reference.

Usage:
    python plot_representational_distance_ts.py [--voxRes 8mm]
                                                  [--bands theta alpha beta lowgamma highgamma]
                                                  [--rois visual parietal frontal]
                                                  [--conditions ampOnly]
                                                  [--subjects 1 2 ...]
                                                  [--outdir <path>] [--figdir <path>]
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

from constants import SUBJECT_LIST, ROI_NAMES, get_bids_root
from representational_distance_ts_cell import output_path

# ── Design constants (mirrors plot_decoding_ts.py) ─────────────────────────────

_BG        = '#000000'
_FG        = '#e0e0e0'
_GRID      = '#1c1c1c'
_FLAG_LINE = '#777777'
_FLAG_TXT  = '#cccccc'
_ZERO      = '#444444'     # horizontal no-structure line

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

def load_all_subjects(subjects, bids_root, voxRes, bands, rois, conditions, outdir=None):
    """Returns list (one entry per subject) of dicts: data[(band, roi, condition)] = npz dict or None."""
    all_data = []
    for subjID in subjects:
        d = {}
        for band in bands:
            for roi in rois:
                for condition in conditions:
                    fp = output_path(bids_root, subjID, band, roi, condition, voxRes, outdir)
                    if os.path.exists(fp):
                        d[(band, roi, condition)] = dict(np.load(fp, allow_pickle=True))
        all_data.append(d if d else None)
    return all_data


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_repdist(all_data, band, roi, condition):
    """
    Returns (tv, mean_gap, sem_gap, mean_null, mean_null_std) or Nones.

    mean_gap/sem_gap: arithmetic mean/SEM across subjects of each subject's
    real gap(t) -- standard cross-subject inference on the statistic itself.

    mean_null/mean_null_std: arithmetic mean across subjects of each
    subject's OWN null_mean(t)/null_std(t) (each subject's label-permutation
    null, computed independently per subject in representational_distance_ts_cell.py).
    This is deliberately NOT a cross-subject SEM of the null center -- it's
    a representative "typical single-subject null" reference band, the same
    role plot_decoding_ts.py's shuffle line plays, not a formal group-level
    null itself. Read significance from the real gap's SEM/CI relative to
    zero (or from the per-subject p_value field), not from this band alone.
    """
    gap_list, null_mean_list, null_std_list = [], [], []
    tv = None
    for d in all_data:
        if d is None:
            continue
        k = (band, roi, condition)
        if k not in d:
            continue
        npz = d[k]
        gap_list.append(npz['gap'])
        null_mean_list.append(npz['null_mean'])
        null_std_list.append(npz['null_std'])
        if tv is None:
            tv = npz['time_vector']

    if not gap_list:
        return None, None, None, None, None

    gap  = np.stack(gap_list)         # (n_subj, T)
    nmu  = np.stack(null_mean_list)   # (n_subj, T)
    nsd  = np.stack(null_std_list)    # (n_subj, T)
    n    = gap.shape[0]

    return (tv,
            gap.mean(0), gap.std(0) / np.sqrt(n),
            nmu.mean(0), nsd.mean(0))


# ── Plotting ──────────────────────────────────────────────────────────────────

def _style_ax(ax, spine_col='#333333'):
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color(spine_col)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)


def plot_repdist_panel(ax, tv, mean_gap, sem_gap, mean_null, mean_null_std,
                        roi, is_bottom_row, is_left_col, show_title, title_str,
                        show_flag_labels=True):
    col = ROI_COLOURS[roi]
    t0, t1 = float(tv[0]), float(tv[-1])

    ax.axhline(0, color=_ZERO, lw=0.8, ls=':', zorder=1)

    # Null reference band (see aggregate_repdist docstring)
    ax.fill_between(tv, mean_null - mean_null_std, mean_null + mean_null_std,
                     alpha=0.15, color=col, zorder=2)
    ax.plot(tv, mean_null, color=col, lw=0.9, ls='--', alpha=0.55, zorder=2)

    # Real gap
    ax.fill_between(tv, mean_gap - sem_gap, mean_gap + sem_gap,
                     alpha=0.30, color=col, zorder=3)
    ax.plot(tv, mean_gap, color=col, lw=1.5, zorder=3)

    lo = min(np.nanmin(mean_gap - sem_gap), np.nanmin(mean_null - mean_null_std), 0)
    hi = max(np.nanmax(mean_gap + sem_gap), np.nanmax(mean_null + mean_null_std), 0)
    pad = max(1e-6, (hi - lo) * 0.15)
    lo, hi = lo - pad, hi + pad

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
        ax.set_ylabel('Between - Within\ndistance', fontsize=8, color=_FG)
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


def make_timeseries_figure(all_data, bands, rois, condition, voxRes, outdir_fig):
    n_bands, n_rois = len(bands), len(rois)
    fig, gs = _band_row_grid(n_bands, n_rois, row_h_in=1.3, bottom_margin_in=0.40)

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois):
            ax_ts = fig.add_subplot(gs[r_idx, c_idx])

            tv, mean_gap, sem_gap, mean_null, mean_null_std = aggregate_repdist(
                all_data, band, roi, condition)

            is_left_col   = (c_idx == 0)
            is_bottom_row = (r_idx == n_bands - 1)
            show_title    = (r_idx == 0)
            title_str     = roi.capitalize()

            if tv is not None:
                plot_repdist_panel(
                    ax_ts, tv, mean_gap, sem_gap, mean_null, mean_null_std,
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

    cond_label = 'Amplitude only' if condition == 'ampOnly' else 'Amplitude + Phase'
    fig.suptitle(f'Representational Distance (Between - Within)  |  {cond_label}  |  {voxRes}',
                 color=_FG, fontsize=14, fontweight='bold', y=0.97)
    fig.text(0.5, 0.005, 'Time (s)', ha='center', va='bottom', color=_FG, fontsize=9)

    os.makedirs(outdir_fig, exist_ok=True)
    fpath = os.path.join(outdir_fig, f'repdist_ts_{condition}_{voxRes}.png')
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
          f'bands={args.bands} | rois={args.rois} | conditions={args.conditions}')

    all_data = load_all_subjects(
        args.subjects, bids_root, args.voxRes,
        args.bands, args.rois, args.conditions, args.outdir)

    n_loaded = sum(1 for d in all_data if d)
    print(f'Loaded data for {n_loaded}/{len(args.subjects)} subjects.')

    for condition in args.conditions:
        print(f'\n-- Plotting condition: {condition} --')
        make_timeseries_figure(all_data, args.bands, args.rois, condition, args.voxRes, figdir)

    print('\nAll figures saved.')


if __name__ == '__main__':
    main()
