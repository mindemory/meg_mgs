#!/usr/bin/env python3
"""
plot_timeseries.py

Plots mean ± SEM timeseries of source-space MEG activity averaged across
subjects, for stim-locked and resp-locked epochs, for all frequency bands
(unfiltered broadband + theta/alpha/beta/lowgamma/highgamma amplitude) and
all ROIs (visual, parietal, frontal, whole-brain).

Data sources:
  - Unfiltered  : G03 raw broadband voltage,   mean across sources in ROI
  - Band amps   : G04 Hilbert amplitude,        mean across sources in ROI

For each (lockType, band, ROI):
  1. Per subject: mean across trials, then mean across sources in ROI
     -> (n_times,) curve per subject
  2. Across subjects: mean ± SEM

Output figures (saved with black background):
  <outdir>/timeseries_<lockType>_<voxRes>.png
    rows = bands (unfiltered, theta, alpha, beta, lowgamma, highgamma)
    cols = ROIs  (visual, parietal, frontal, whole-brain)

Usage:
    python plot_timeseries.py [--voxRes 8mm] [--lockTypes stim resp]
                              [--rois visual parietal frontal]
                              [--subjects 1 2 ...]
                              [--outdir <path>]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from atlas import load_atlas_masks, roi_local_indices
from constants import AMP_ONLY_BANDS, SUBJECT_LIST, ROI_NAMES, get_bids_root
from io_g03 import load_g03_unfiltered
from io_g04 import load_g04_band

# ── Visual design ─────────────────────────────────────────────────────────────
_BG   = '#0d0d0d'
_FG   = '#e0e0e0'
_GRID = '#1e1e1e'

ROI_COLOURS = {
    'visual':    '#7EB8F7',
    'parietal':  '#F4A261',
    'frontal':   '#A8DADC',
    'whole':     '#E76F51',
}

BAND_ORDER = ['unfiltered', 'theta', 'alpha', 'beta', 'lowgamma', 'highgamma']
BAND_LABELS = {
    'unfiltered': 'Unfiltered (broadband)',
    'theta':      'Theta (4–8 Hz)',
    'alpha':      'Alpha (8–12 Hz)',
    'beta':       'Beta (13–30 Hz)',
    'lowgamma':   'Low \u03b3 (30–80 Hz)',
    'highgamma':  'High \u03b3 (80–150 Hz)',
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _filter_inside_pos(inside_pos, data, atlas_masks):
    """Drop source columns whose 1-based inside_pos exceeds atlas grid."""
    n_grid     = len(next(iter(atlas_masks.values())))
    valid_mask = (inside_pos >= 1) & (inside_pos <= n_grid)
    if valid_mask.all():
        return inside_pos, data
    valid_cols = np.where(valid_mask)[0]
    return inside_pos[valid_cols], data[:, :, valid_cols]


def load_subject_timeseries(subjID, lockType, voxRes, bids_root,
                             atlas_masks, rois_all):
    """
    Returns dict:
        result[band][roi] = (n_times,) mean-across-trials-and-sources
        result['time_vectors'][band] = (n_times,) float
    or None if G03 is missing.
    """
    try:
        g03 = load_g03_unfiltered(subjID, lockType, voxRes, bids_root)
    except FileNotFoundError as e:
        print(f'  sub-{subjID:02d}: G03 missing: {e}')
        return None

    inside_pos, g03_data = _filter_inside_pos(
        g03['inside_pos'], g03['data'], atlas_masks)

    # Build ROI index maps once per subject
    roi_idx = {}
    for roi in rois_all:
        if roi == 'whole':
            roi_idx[roi] = np.arange(g03_data.shape[2])
        else:
            roi_idx[roi] = roi_local_indices(atlas_masks, inside_pos, roi)

    result = {'time_vectors': {}}

    # ── Unfiltered (G03 raw voltage) ──────────────────────────────────────────
    result['unfiltered'] = {}
    result['time_vectors']['unfiltered'] = g03['time_vector']
    for roi in rois_all:
        idx = roi_idx[roi]
        if idx.size == 0:
            result['unfiltered'][roi] = None
            continue
        # Mean across trials -> (n_times, n_roi), then mean across sources
        curve = g03_data[:, :, idx].mean(axis=0).mean(axis=1)   # (n_times,)
        result['unfiltered'][roi] = curve

    # ── G04 band amplitudes ───────────────────────────────────────────────────
    for band in AMP_ONLY_BANDS:
        result[band] = {}
        try:
            g04 = load_g04_band(subjID, lockType, band, voxRes, bids_root,
                                 want_phase=False)
        except (FileNotFoundError, ValueError) as e:
            print(f'  sub-{subjID:02d} {band}: {e}')
            result['time_vectors'][band] = None
            for roi in rois_all:
                result[band][roi] = None
            continue

        amp = g04['amp']   # (n_trials, n_times, n_sources)
        # Apply same inside_pos filter if needed
        n_src_g03 = g03_data.shape[2]
        if amp.shape[2] > n_src_g03:
            amp = amp[:, :, :n_src_g03]   # same valid_cols as g03_data

        result['time_vectors'][band] = g04['time_vector']
        for roi in rois_all:
            idx = roi_idx[roi]
            if idx.size == 0:
                result[band][roi] = None
                continue
            curve = amp[:, :, idx].mean(axis=0).mean(axis=1)   # (n_times,)
            result[band][roi] = curve

    n_trials = g03_data.shape[0]
    print(f'  sub-{subjID:02d} done ({n_trials} trials)')
    return result


# ── Cross-subject aggregation ─────────────────────────────────────────────────

def aggregate(all_results, band, roi):
    """
    Returns (time_vector, mean, sem) or (None, None, None).
    """
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


def plot_timeseries_figure(all_results, rois_all, lockType, voxRes, outdir):
    """
    One figure per lockType:
        rows = bands
        cols = ROIs
    """
    bands  = BAND_ORDER
    n_rows = len(bands)
    n_cols = len(rois_all)

    fig_w = max(4.5 * n_cols, 12)
    fig_h = max(2.8 * n_rows, 10)

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
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', color='#555555', fontsize=9)
                continue

            ax.fill_between(tv,
                             mean_curve - sem_curve,
                             mean_curve + sem_curve,
                             color=colour, alpha=0.25)
            ax.plot(tv, mean_curve, color=colour, linewidth=1.6)

            # Epoch-onset marker
            ax.axvline(0, color='#666666', linewidth=0.8,
                        linestyle='--', alpha=0.9)
            ax.axhline(0, color='#333333', linewidth=0.5, alpha=0.6)

            # Count valid subjects
            n_subj = sum(
                1 for r in all_results
                if r is not None and r.get(band, {}).get(roi) is not None
            )

            # Column headers (first row only)
            if r_idx == 0:
                roi_lbl = roi.capitalize() if roi != 'whole' else 'Whole brain'
                ax.set_title(f'{roi_lbl}  (n={n_subj})',
                             fontsize=10, fontweight='bold', pad=5)

            # X-axis label (last row only)
            if r_idx == n_rows - 1:
                ax.set_xlabel('Time (s)', fontsize=8)

            # Y-axis label (first col only)
            if c_idx == 0:
                is_unfiltered = (band == 'unfiltered')
                ax.set_ylabel('Amplitude (a.u.)' if not is_unfiltered
                              else 'Voltage (a.u.)', fontsize=7)

            # Band label annotated on left of first column
            if c_idx == 0:
                ax.annotate(BAND_LABELS.get(band, band),
                             xy=(-0.30, 0.5), xycoords='axes fraction',
                             fontsize=8, color=_FG,
                             ha='right', va='center',
                             rotation=90, fontweight='bold')

            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))

    lock_label = 'Stimulus-locked' if lockType == 'stim' else 'Response-locked'
    fig.suptitle(
        f'Mean ± SEM Source Activity  |  {lock_label}  |  {voxRes}',
        color=_FG, fontsize=12, fontweight='bold', y=1.01
    )
    fig.tight_layout(rect=[0.06, 0, 1, 1])

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
        description='Plot mean timeseries for all bands and ROIs.')
    parser.add_argument('--voxRes',   default='8mm')
    parser.add_argument('--lockTypes', nargs='+', default=['stim', 'resp'])
    parser.add_argument('--rois',      nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--subjects',  nargs='+', type=int,
                        default=SUBJECT_LIST)
    parser.add_argument('--outdir',    default=None)
    args = parser.parse_args()

    bids_root = get_bids_root()
    outdir    = args.outdir or os.path.join(
        bids_root, 'derivatives', 'glueDecoding', 'timeseries')

    rois_all = list(args.rois)
    if 'whole' not in rois_all:
        rois_all.append('whole')

    print(f'plot_timeseries | voxRes={args.voxRes} | '
          f'subjects={args.subjects} | rois={rois_all}')

    atlas_masks = load_atlas_masks(args.voxRes, bids_root)

    for lockType in args.lockTypes:
        print(f'\n=== lockType: {lockType} ===')

        all_results = []
        for subjID in args.subjects:
            print(f'sub-{subjID:02d} ...')
            r = load_subject_timeseries(subjID, lockType, args.voxRes,
                                         bids_root, atlas_masks, rois_all)
            all_results.append(r)

        plot_timeseries_figure(all_results, rois_all, lockType,
                                args.voxRes, outdir)

    print('\nDone.')


if __name__ == '__main__':
    main()
