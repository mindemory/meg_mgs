#!/usr/bin/env python3
"""
plot_timeseries.py

Plots mean +/- SEM timeseries of source-space MEG activity averaged across
subjects, for stim-locked and resp-locked epochs, for all frequency bands
(unfiltered broadband + theta/alpha/beta/lowgamma/highgamma amplitude) and
all ROIs (visual, parietal, frontal, whole-brain).

Data sources:
  - Unfiltered  : G03 raw broadband voltage,   mean across sources in ROI
  - Band amps   : G04 Hilbert amplitude,        mean across sources in ROI

Time windows & event flags (hard-coded per lock type):
  stim-locked : -1.0 to +1.7 s   | Stim at 0 s, Delay Onset at +0.2 s
  resp-locked : -4.5 to -0.5 s   | Delay Onset at -4 s, R Onset at -2.5 s,
                                    Feedback at -2 s

Parallelism:
  Subjects are processed in parallel using joblib (processes, not threads,
  to avoid GIL contention in numpy operations). Default n_jobs = min(21, 8).
  The bottleneck is HDF5 IO + memory reduction for whole-brain G03/G04 data;
  parallelising across subjects gives roughly linear speedup up to IO saturation.

Usage:
    python plot_timeseries.py [--voxRes 8mm] [--lockTypes stim resp]
                              [--rois visual parietal frontal]
                              [--subjects 1 2 ...]
                              [--outdir <path>]
                              [--n_jobs 8]
"""

import os
import sys
import argparse
from pathlib import Path
from multiprocessing import cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from joblib import Parallel, delayed

from atlas import load_atlas_masks, roi_local_indices
from constants import AMP_ONLY_BANDS, SUBJECT_LIST, ROI_NAMES, get_bids_root
from io_g03 import load_g03_unfiltered
from io_g04 import load_g04_band

# ── Time windows and event flags per lock type ────────────────────────────────
TIME_WINDOWS = {
    'stim': (-1.0, 1.7),
    'resp': (-4.5, -0.5),
}

# Each flag: (time_s, label, label_y_frac)
# label_y_frac controls vertical placement of the text label (0=bottom, 1=top)
EVENT_FLAGS = {
    'stim': [
        (0.0,  'Stim',        0.93),
        (0.2,  'Delay\nOnset', 0.93),
    ],
    'resp': [
        (-4.0, 'Delay\nOnset', 0.93),
        (-2.5, 'R Onset',     0.93),
        (-2.0, 'Feedback',    0.78),
    ],
}

# ── Visual design ─────────────────────────────────────────────────────────────
_BG   = '#0d0d0d'
_FG   = '#e0e0e0'
_GRID = '#1e1e1e'
_FLAG_LINE = '#888888'
_FLAG_TXT  = '#cccccc'

ROI_COLOURS = {
    'visual':    '#7EB8F7',
    'parietal':  '#F4A261',
    'frontal':   '#A8DADC',
    'whole':     '#E76F51',
}

BAND_ORDER = ['unfiltered', 'theta', 'alpha', 'beta', 'lowgamma', 'highgamma']
BAND_LABELS = {
    'unfiltered': 'Unfiltered\n(broadband)',
    'theta':      'Theta\n(4-8 Hz)',
    'alpha':      'Alpha\n(8-12 Hz)',
    'beta':       'Beta\n(13-30 Hz)',
    'lowgamma':   'Low gamma\n(30-80 Hz)',
    'highgamma':  'High gamma\n(80-150 Hz)',
}


# ── Data loading (one worker per subject, runs in parallel) ───────────────────

def _filter_inside_pos(inside_pos, data, atlas_masks):
    """Drop source columns whose 1-based inside_pos exceeds atlas grid."""
    n_grid     = len(next(iter(atlas_masks.values())))
    valid_mask = (inside_pos >= 1) & (inside_pos <= n_grid)
    if valid_mask.all():
        return inside_pos, data
    valid_cols = np.where(valid_mask)[0]
    return inside_pos[valid_cols], data[:, :, valid_cols]


def _crop_to_window(tv, curve, t_min, t_max):
    """Crop (time_vector, curve) to [t_min, t_max]."""
    mask = (tv >= t_min) & (tv <= t_max)
    return tv[mask], curve[mask]


def load_subject_timeseries(subjID, lockType, voxRes, bids_root,
                             atlas_masks, rois_all, t_min, t_max):
    """
    Load and reduce one subject's data to per-ROI mean timeseries curves,
    cropped to [t_min, t_max].

    Returns dict:
        result[band][roi]           = (n_crop_times,) float
        result['time_vectors'][band] = (n_crop_times,) float
    or None if G03 is missing.

    Called in parallel via joblib -- must be importable at top level.
    """
    try:
        g03 = load_g03_unfiltered(subjID, lockType, voxRes, bids_root)
    except FileNotFoundError as e:
        print(f'  sub-{subjID:02d}: G03 missing: {e}', flush=True)
        return None

    inside_pos, g03_data = _filter_inside_pos(
        g03['inside_pos'], g03['data'], atlas_masks)

    # ROI index maps
    roi_idx = {}
    for roi in rois_all:
        if roi == 'whole':
            roi_idx[roi] = np.arange(g03_data.shape[2])
        else:
            roi_idx[roi] = roi_local_indices(atlas_masks, inside_pos, roi)

    result = {'time_vectors': {}}

    # ── Unfiltered (G03 broadband voltage) ────────────────────────────────────
    result['unfiltered'] = {}
    tv_full = g03['time_vector']
    tv_crop_mask = (tv_full >= t_min) & (tv_full <= t_max)
    tv_crop = tv_full[tv_crop_mask]
    result['time_vectors']['unfiltered'] = tv_crop

    for roi in rois_all:
        idx = roi_idx[roi]
        if idx.size == 0:
            result['unfiltered'][roi] = None
            continue
        # mean across trials -> (n_times, n_roi), then mean across sources
        curve = g03_data[:, :, idx].mean(axis=0).mean(axis=1)   # (n_times,)
        result['unfiltered'][roi] = curve[tv_crop_mask]

    # Free the large broadband array immediately
    del g03_data, g03

    # ── G04 band amplitudes ───────────────────────────────────────────────────
    for band in AMP_ONLY_BANDS:
        result[band] = {}
        try:
            g04 = load_g04_band(subjID, lockType, band, voxRes, bids_root,
                                 want_phase=False)
        except (FileNotFoundError, ValueError) as e:
            print(f'  sub-{subjID:02d} {band}: {e}', flush=True)
            result['time_vectors'][band] = None
            for roi in rois_all:
                result[band][roi] = None
            continue

        amp        = g04['amp']
        tv_g04     = g04['time_vector']
        tv_g04_mask = (tv_g04 >= t_min) & (tv_g04 <= t_max)
        result['time_vectors'][band] = tv_g04[tv_g04_mask]

        for roi in rois_all:
            idx = roi_idx[roi]
            if idx.size == 0:
                result[band][roi] = None
                continue
            curve = amp[:, :, idx].mean(axis=0).mean(axis=1)
            result[band][roi] = curve[tv_g04_mask]

        del amp, g04

    print(f'  sub-{subjID:02d} done', flush=True)
    return result


# ── Cross-subject aggregation ─────────────────────────────────────────────────

def aggregate(all_results, band, roi):
    """Returns (time_vector, mean, sem) or (None, None, None)."""
    curves = []
    tv     = None
    for r in all_results:
        if r is None:
            continue
        curve = r.get(band, {}).get(roi)
        if curve is None:
            continue
        curves.append(curve)
        if tv is None:
            tv = r['time_vectors'].get(band)
    if not curves or tv is None:
        return None, None, None
    stacked = np.stack(curves, axis=0)
    return tv, stacked.mean(axis=0), stacked.std(axis=0) / np.sqrt(len(curves))


# ── Plotting ──────────────────────────────────────────────────────────────────

def _apply_black_style(fig, axes_flat):
    fig.patch.set_facecolor(_BG)
    for ax in axes_flat:
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_FG, which='both', labelsize=7)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.grid(True, color=_GRID, linewidth=0.5, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)


def _draw_event_flags(ax, flags, y_lim):
    """Draw vertical flag lines and rotated text labels."""
    for t_flag, label, y_frac in flags:
        ax.axvline(t_flag, color=_FLAG_LINE, linewidth=0.9,
                   linestyle=':', alpha=0.85, zorder=3)
        y_pos = y_lim[0] + y_frac * (y_lim[1] - y_lim[0])
        ax.text(t_flag, y_pos, label,
                color=_FLAG_TXT, fontsize=5.5, ha='left', va='top',
                rotation=90, rotation_mode='anchor',
                fontweight='bold', zorder=4,
                transform=ax.get_xaxis_transform() if False else ax.transData)


def plot_timeseries_figure(all_results, rois_all, lockType, voxRes, outdir):
    """
    One figure per lockType:
        rows = bands (6)
        cols = ROIs  (4)
    """
    bands  = BAND_ORDER
    n_rows = len(bands)
    n_cols = len(rois_all)
    t_min, t_max = TIME_WINDOWS.get(lockType, (-1.0, 2.0))
    flags        = EVENT_FLAGS.get(lockType, [])

    fig_w = max(4.5 * n_cols, 14)
    fig_h = max(2.8 * n_rows, 12)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              sharex=False, sharey=False,
                              squeeze=False)
    _apply_black_style(fig, axes.flatten())

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois_all):
            ax     = axes[r_idx, c_idx]
            colour = ROI_COLOURS.get(roi, '#ffffff')

            tv, mean_curve, sem_curve = aggregate(all_results, band, roi)

            if tv is None:
                ax.set_xlim(t_min, t_max)
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', color='#555555', fontsize=9)
                continue

            ax.fill_between(tv,
                             mean_curve - sem_curve,
                             mean_curve + sem_curve,
                             color=colour, alpha=0.25)
            ax.plot(tv, mean_curve, color=colour, linewidth=1.6)

            ax.set_xlim(t_min, t_max)
            ax.axhline(0, color='#333333', linewidth=0.4, alpha=0.5)

            # Draw event flags
            y_lim = ax.get_ylim()
            _draw_event_flags(ax, flags, y_lim)

            # Count valid subjects
            n_subj = sum(
                1 for r in all_results
                if r is not None and r.get(band, {}).get(roi) is not None
            )

            if r_idx == 0:
                roi_lbl = roi.capitalize() if roi != 'whole' else 'Whole brain'
                ax.set_title(f'{roi_lbl}  (n={n_subj})',
                             fontsize=10, fontweight='bold', pad=5)

            if r_idx == n_rows - 1:
                ax.set_xlabel('Time (s)', fontsize=8)

            if c_idx == 0:
                ylabel = ('Voltage (a.u.)' if band == 'unfiltered'
                          else 'Amplitude (a.u.)')
                ax.set_ylabel(ylabel, fontsize=7)

            if c_idx == 0:
                ax.annotate(BAND_LABELS.get(band, band),
                             xy=(-0.34, 0.5), xycoords='axes fraction',
                             fontsize=7.5, color=_FG,
                             ha='right', va='center',
                             rotation=90, fontweight='bold')

            # Tick spacing: 0.5 s major, 0.1 s minor
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))

    lock_label = 'Stimulus-locked' if lockType == 'stim' else 'Response-locked'
    win_str    = f'{t_min:+.1f} to {t_max:+.1f} s'
    fig.suptitle(
        f'Mean +/- SEM Source Activity  |  {lock_label}  ({win_str})  |  {voxRes}',
        color=_FG, fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout(rect=[0.07, 0, 1, 1])

    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f'timeseries_{lockType}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot mean +/- SEM timeseries for all bands and ROIs.')
    parser.add_argument('--voxRes',    default='8mm')
    parser.add_argument('--lockTypes', nargs='+', default=['stim', 'resp'])
    parser.add_argument('--rois',      nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--subjects',  nargs='+', type=int,
                        default=SUBJECT_LIST)
    parser.add_argument('--outdir',    default=None)
    parser.add_argument('--n_jobs',    type=int,
                        default=min(len(SUBJECT_LIST), cpu_count() - 1, 8),
                        help='Parallel workers (subjects). Default: min(21, ncpu-1, 8)')
    args = parser.parse_args()

    bids_root = get_bids_root()
    outdir    = args.outdir or os.path.join(
        bids_root, 'derivatives', 'glueDecoding', 'timeseries')

    rois_all = list(args.rois)
    if 'whole' not in rois_all:
        rois_all.append('whole')

    n_jobs = max(1, args.n_jobs)
    print(f'plot_timeseries | voxRes={args.voxRes} | '
          f'subjects={args.subjects} | rois={rois_all} | n_jobs={n_jobs}')

    atlas_masks = load_atlas_masks(args.voxRes, bids_root)

    for lockType in args.lockTypes:
        t_min, t_max = TIME_WINDOWS.get(lockType, (-1.0, 2.0))
        print(f'\n=== lockType: {lockType}  window: [{t_min}, {t_max}] s ===')

        all_results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=5)(
            delayed(load_subject_timeseries)(
                subjID, lockType, args.voxRes, bids_root,
                atlas_masks, rois_all, t_min, t_max
            )
            for subjID in args.subjects
        )

        plot_timeseries_figure(all_results, rois_all, lockType,
                                args.voxRes, outdir)

    print('\nDone.')


if __name__ == '__main__':
    main()
