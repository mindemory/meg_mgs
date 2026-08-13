#!/usr/bin/env python3
"""
plot_linear_decoding_categories.py

Aggregates linear_decoding_categories_cell.py's per-subject .npz files and
plots LOO (or k-fold) SVM one-vs-rest classification accuracy over time,
bands x rois grid, one figure per (condition, scheme) -- mirrors
plot_decoding_ts.py / plot_representational_distance_ts.py's layout/styling
so all three are visually and temporally comparable.

Real line: mean +/- SEM of the per-subject accuracy across subjects.
Chance line: 1/n_categories (theoretical), always shown.
Empirical null band (only if the cells were run with --n_shuffle > 0):
mean(shuffle_acc_mean) +/- mean(shuffle_acc_std) across subjects -- same
"typical single-subject null" caveat as plot_representational_distance_ts.py
(NOT a formal group-level null; read the real accuracy's SEM/CI relative to
the theoretical chance line for group-level inference instead).

Usage:
    python plot_linear_decoding_categories.py [--voxRes 8mm]
                                               [--bands theta alpha beta lowgamma highgamma]
                                               [--rois visual parietal frontal]
                                               [--conditions ampOnly]
                                               [--schemes 2 4 6 10]
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

from constants import SUBJECT_LIST, ROI_NAMES, CATEGORY_SCHEMES, get_bids_root
from linear_decoding_categories_cell import output_path

SCHEME_LABELS = {s: f"{CATEGORY_SCHEMES[s]['name']} ({s} categories)" for s in CATEGORY_SCHEMES}

# ── Design constants (mirrors plot_decoding_ts.py / plot_representational_distance_ts.py) ──

_BG        = '#000000'
_FG        = '#e0e0e0'
_GRID      = '#1c1c1c'
_FLAG_LINE = '#777777'
_FLAG_TXT  = '#cccccc'
_CHANCE    = '#444444'

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

def aggregate_lindecode(all_data, band, roi, condition, scheme):
    """
    Returns (tv, mean_acc, sem_acc, chance_level, mean_shuf, mean_shuf_std,
    has_shuffle) or (None,)*7.

    mean_acc/sem_acc: arithmetic mean/SEM across subjects of each subject's
    real LOO/k-fold accuracy(t).
    chance_level: 1/n_categories (identical across subjects for the same
    scheme, just read from the first available cell).
    mean_shuf/mean_shuf_std: same "typical single-subject null" caveat as
    plot_representational_distance_ts.py's aggregate_repdist -- only
    meaningful if cells were run with --n_shuffle > 0 (all-NaN otherwise,
    has_shuffle=False).
    """
    acc_list, shuf_mean_list, shuf_std_list = [], [], []
    tv = None
    chance_level = None
    for d in all_data:
        if d is None:
            continue
        k = (band, roi, condition, scheme)
        if k not in d:
            continue
        npz = d[k]
        acc_list.append(npz['accuracy'])
        shuf_mean_list.append(npz['shuffle_acc_mean'])
        shuf_std_list.append(npz['shuffle_acc_std'])
        if tv is None:
            tv = npz['eval_time_vector']
            chance_level = float(npz['chance_level'][0])

    if not acc_list:
        return None, None, None, None, None, None, False

    acc  = np.stack(acc_list)          # (n_subj, T)
    smu  = np.stack(shuf_mean_list)    # (n_subj, T)
    ssd  = np.stack(shuf_std_list)     # (n_subj, T)
    n    = acc.shape[0]

    has_shuffle = not np.all(np.isnan(smu))
    mean_shuf     = np.nanmean(smu, axis=0) if has_shuffle else None
    mean_shuf_std = np.nanmean(ssd, axis=0) if has_shuffle else None

    return (tv, acc.mean(0), acc.std(0) / np.sqrt(n), chance_level,
            mean_shuf, mean_shuf_std, has_shuffle)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _style_ax(ax, spine_col='#333333'):
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color(spine_col)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)


def plot_lindecode_panel(ax, tv, mean_acc, sem_acc, chance_level, mean_shuf, mean_shuf_std,
                          roi, is_bottom_row, is_left_col, show_title, title_str,
                          show_flag_labels=True):
    col = ROI_COLOURS[roi]
    t0, t1 = float(tv[0]), float(tv[-1])

    ax.axhline(chance_level, color=_CHANCE, lw=0.8, ls=':', zorder=1)

    if mean_shuf is not None:
        ax.fill_between(tv, mean_shuf - mean_shuf_std, mean_shuf + mean_shuf_std,
                         alpha=0.15, color=col, zorder=2)
        ax.plot(tv, mean_shuf, color=col, lw=0.9, ls='--', alpha=0.55, zorder=2)

    ax.fill_between(tv, mean_acc - sem_acc, mean_acc + sem_acc,
                     alpha=0.30, color=col, zorder=3)
    ax.plot(tv, mean_acc, color=col, lw=1.5, zorder=3)

    lo_candidates = [np.nanmin(mean_acc - sem_acc), chance_level]
    hi_candidates = [np.nanmax(mean_acc + sem_acc), chance_level]
    if mean_shuf is not None:
        lo_candidates.append(np.nanmin(mean_shuf - mean_shuf_std))
        hi_candidates.append(np.nanmax(mean_shuf + mean_shuf_std))
    lo, hi = min(lo_candidates), max(hi_candidates)
    pad = max(1e-6, (hi - lo) * 0.15)
    lo, hi = max(0.0, lo - pad), min(1.0, hi + pad)

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
        ax.set_ylabel('LOO accuracy', fontsize=8, color=_FG)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    if not is_bottom_row:
        ax.set_xticklabels([])

    ax.grid(True, color=_GRID, lw=0.4, zorder=0)
    _style_ax(ax)

    if show_title:
        ax.set_title(title_str, color=_FG, fontsize=10, fontweight='bold', pad=4)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


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


def make_timeseries_figure(all_data, bands, rois, condition, scheme, voxRes, outdir_fig):
    n_bands, n_rois = len(bands), len(rois)
    fig, gs = _band_row_grid(n_bands, n_rois, row_h_in=1.3, bottom_margin_in=0.40)

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois):
            ax_ts = fig.add_subplot(gs[r_idx, c_idx])

            (tv, mean_acc, sem_acc, chance_level,
             mean_shuf, mean_shuf_std, has_shuffle) = aggregate_lindecode(
                all_data, band, roi, condition, scheme)

            is_left_col   = (c_idx == 0)
            is_bottom_row = (r_idx == n_bands - 1)
            show_title    = (r_idx == 0)
            title_str     = roi.capitalize()

            if tv is not None:
                plot_lindecode_panel(
                    ax_ts, tv, mean_acc, sem_acc, chance_level, mean_shuf, mean_shuf_std,
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
    fig.suptitle(f'Linear (SVM, OvR, LOO) Decoding Accuracy  |  {cond_label}  |  '
                 f'{scheme_label}  |  {voxRes}',
                 color=_FG, fontsize=14, fontweight='bold', y=0.97)
    fig.text(0.5, 0.005, 'Time (s)', ha='center', va='bottom', color=_FG, fontsize=9)

    os.makedirs(outdir_fig, exist_ok=True)
    fpath = os.path.join(outdir_fig, f'lindecode_cat_{condition}_scheme{scheme}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and plot linear decoding-by-category accuracy timeseries across subjects.')
    parser.add_argument('--voxRes',     default='8mm')
    parser.add_argument('--subjects',   nargs='+', type=int, default=SUBJECT_LIST)
    parser.add_argument('--bands',      nargs='+',
                        default=['theta', 'alpha', 'beta', 'lowgamma', 'highgamma'])
    parser.add_argument('--rois',       nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions', nargs='+', default=['ampOnly'])
    parser.add_argument('--schemes',    nargs='+', type=int, default=sorted(CATEGORY_SCHEMES),
                        choices=sorted(CATEGORY_SCHEMES))
    parser.add_argument('--outdir',     default=None,
                        help='Directory containing the per-subject .npz files.')
    parser.add_argument('--figdir',     default=None,
                        help='Directory to save figures (default: same as outdir '
                             'or BIDS derivatives/glueDecoding/linDecodeCat).')
    args = parser.parse_args()

    bids_root = get_bids_root()
    figdir = args.figdir or (args.outdir if args.outdir else
                              os.path.join(bids_root, 'derivatives', 'glueDecoding', 'linDecodeCat'))

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
                                    args.voxRes, figdir)

    print('\nAll figures saved.')


if __name__ == '__main__':
    main()
