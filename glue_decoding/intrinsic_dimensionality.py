#!/usr/bin/env python3
"""
intrinsic_dimensionality.py

Estimates the intrinsic dimensionality of MEG source-space data over time,
for different frequency bands (broadband, theta, alpha, beta, lowgamma,
highgamma) and ROIs (visual, parietal, frontal, whole-brain).

Intrinsic dimensionality is estimated via the "participation ratio" (PR),
also called the effective dimensionality:

    PR = (sum_i lambda_i)^2 / sum_i lambda_i^2

where lambda_i are the eigenvalues of the trial-covariance matrix at each
timepoint. PR = 1 means a single PC dominates; PR = n_sources means all
components contribute equally. This is a smooth, threshold-free estimator
that reflects the spread of the explained variance spectrum.

Alternatively, the number of PCs needed to explain >= VAR_THRESHOLD of
variance is also stored (int-valued, noisier but interpretable).

Aggregates per-subject PR curves are averaged across subjects and shown
as mean +/- SEM, one panel per ROI. Panels are arranged as:
    rows = frequency bands
    cols = ROIs (visual, parietal, frontal, whole-brain)

All figures are saved with a black background.

Usage:
    python intrinsic_dimensionality.py [--voxRes 8mm]
                                       [--lockTypes stim resp]
                                       [--rois visual parietal frontal]
                                       [--outdir <bids_root>/derivatives/glueDecoding/intrinsicDim]
                                       [--var_threshold 0.90]
"""

import glob
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

from constants import (AMP_ONLY_BANDS, SUBJECT_LIST, ROI_NAMES, get_bids_root)

# --- Colour palette (black-background friendly) ----------------------------
# One vivid colour per ROI, plus whole-brain
ROI_COLOURS = {
    'visual':    '#7EB8F7',   # sky blue
    'parietal':  '#F4A261',   # warm orange
    'frontal':   '#A8DADC',   # teal
    'whole':     '#E76F51',   # coral-red
}

# Band display order and pretty labels
BAND_ORDER = ['broadband', 'theta', 'alpha', 'beta', 'lowgamma', 'highgamma']
BAND_LABELS = {
    'broadband': 'Broadband',
    'theta':     'Theta (4-8 Hz)',
    'alpha':     'Alpha (8-12 Hz)',
    'beta':      'Beta (13-30 Hz)',
    'lowgamma':  'Low gamma (30-80 Hz)',
    'highgamma': 'High gamma (80-150 Hz)',
}

VAR_THRESHOLD_DEFAULT = 0.90  # fraction of variance for n_pcs metric


# --- .npz file discovery and loading ----------------------------------------

def _npz_path(bids_root, subjID, band, lockType, voxRes, flat_outdir=None):
    """Match the path convention used by intrinsic_dim_cell.py."""
    subName = f'sub-{subjID:02d}'
    fname   = f'{subName}_task-mgs_intrinsicDim_{band}_{lockType}_{voxRes}.npz'
    if flat_outdir:
        return os.path.join(flat_outdir, fname)
    return os.path.join(bids_root, 'derivatives', subName,
                         'sourceRecon', 'intrinsicDim', fname)


def load_subject_npz(bids_root, subjID, band, lockType, voxRes,
                     rois_all, flat_outdir=None):
    """
    Load one subject's saved .npz cell output.
    Returns a dict mirroring compute_subject_dim's per-band result:
        {roi: {'pr': ..., 'npcs': ..., 'time_vector': ...}}
    or None if the file doesn't exist.
    """
    fpath = _npz_path(bids_root, subjID, band, lockType, voxRes, flat_outdir)
    if not os.path.exists(fpath):
        return None
    d = np.load(fpath)
    result = {}
    for roi in rois_all:
        pr_key = f'pr_{roi}'
        if pr_key not in d:
            result[roi] = None
            continue
        result[roi] = {
            'pr':          d[f'pr_{roi}'],
            'npcs':        d[f'npcs_{roi}'],
            'time_vector': d[f'time_vector_{roi}'],
        }
    return result




# --- Cross-subject aggregation (reads from loaded npz dicts) ----------------

def aggregate_subjects(all_subject_results, band, roi, metric='pr'):
    """
    all_subject_results : list where each element is either:
      - a dict  {band: {roi: {'pr', 'npcs', 'time_vector'}}}  (all_wrapped from main)
      - or None if that subject had no data at all.

    Returns (time_vector, mean_curve, sem_curve) or (None, None, None).
    """
    curves      = []
    time_vector = None
    for subj_result in all_subject_results:
        if subj_result is None:
            continue
        band_dict = subj_result.get(band)  # {roi: {...}} or None
        if band_dict is None:
            continue
        entry = band_dict.get(roi)          # {'pr', 'npcs', 'time_vector'} or None
        if entry is None:
            continue
        curves.append(entry[metric])
        if time_vector is None:
            time_vector = entry['time_vector']

    if not curves:
        return None, None, None

    stacked = np.stack(curves, axis=0)    # (n_subjects, n_times)
    mean    = stacked.mean(axis=0)
    sem     = stacked.std(axis=0) / np.sqrt(stacked.shape[0])
    return time_vector, mean, sem




# --- Plotting ---------------------------------------------------------------

def _apply_black_style(fig, axes_flat):
    """Apply black background and matching text/spine colours to a figure."""
    BG   = '#0d0d0d'
    FG   = '#e0e0e0'
    GRID = '#2a2a2a'
    fig.patch.set_facecolor(BG)
    for ax in axes_flat:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, which='both', labelsize=7)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)


def plot_dim_figure(all_subject_results, rois_all, metric, metric_label,
                    lockType, voxRes, outdir, bands=None):
    """
    One figure: rows = bands, cols = ROIs.
    Saves with black background.

    metric      : 'pr' or 'npcs'
    metric_label: axis label string
    """
    if bands is None:
        bands = BAND_ORDER

    n_rows = len(bands)
    n_cols = len(rois_all)
    fig_w  = max(4.5 * n_cols, 12)
    fig_h  = max(3.0 * n_rows, 8)

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              sharex=False, sharey=False,
                              squeeze=False)

    axes_flat = axes.flatten()
    _apply_black_style(fig, axes_flat)

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois_all):
            ax = axes[r_idx, c_idx]
            colour = ROI_COLOURS.get(roi, '#ffffff')

            tv, mean_curve, sem_curve = aggregate_subjects(
                all_subject_results, band, roi, metric)

            if tv is None:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', color='#666666', fontsize=9)
                continue

            ax.fill_between(tv,
                             mean_curve - sem_curve,
                             mean_curve + sem_curve,
                             color=colour, alpha=0.25)
            ax.plot(tv, mean_curve, color=colour, linewidth=1.8)

            # Epoch zero line
            ax.axvline(0, color='#555555', linewidth=1.0,
                        linestyle='--', alpha=0.8)

            # Count subjects with valid data for this (band, roi)
            n_subj = sum(
                1 for s in all_subject_results
                if s is not None
                and s.get(band) is not None
                and s.get(band, {}).get(roi) is not None
            )

            if r_idx == 0:
                roi_lbl = roi.capitalize() if roi != 'whole' else 'Whole brain'
                ax.set_title(f'{roi_lbl}  (n={n_subj})',
                             fontsize=10, fontweight='bold', pad=6)
            if c_idx == 0:
                ax.set_ylabel(metric_label, fontsize=8)
            if r_idx == n_rows - 1:
                ax.set_xlabel('Time (s)', fontsize=8)

            # Band label on left edge of first column
            if c_idx == 0:
                band_txt = BAND_LABELS.get(band, band)
                ax.annotate(band_txt, xy=(-0.32, 0.5),
                             xycoords='axes fraction',
                             fontsize=8, color='#e0e0e0',
                             ha='right', va='center', rotation=90,
                             fontweight='bold')

            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    metric_str = 'participation_ratio' if metric == 'pr' else 'n_pcs'
    fig.suptitle(
        f'Intrinsic Dimensionality over Time  |  '
        f'{lockType}-locked  |  {voxRes}  |  metric: {metric_str}',
        color='#e0e0e0', fontsize=11, fontweight='bold', y=1.01
    )

    fig.tight_layout(rect=[0.05, 0, 1, 1])

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir,
                          f'intrinsic_dim_{metric_str}_{lockType}_{voxRes}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {fname}')
    return fname


def plot_overview_figure(all_subject_results, rois_all, lockType, voxRes,
                          outdir, bands=None):
    """
    Overview plot: one panel per ROI, all bands overlaid with distinct colours,
    showing participation ratio mean +/- SEM. Saved with black background.
    """
    if bands is None:
        bands = BAND_ORDER

    # Distinct colours per band (perceptually even, vivid on black)
    band_palette = {
        'broadband': '#FFFFFF',
        'theta':     '#7EB8F7',
        'alpha':     '#A8DADC',
        'beta':      '#F4A261',
        'lowgamma':  '#E9C46A',
        'highgamma': '#E76F51',
    }

    n_cols = len(rois_all)
    fig_w  = max(5 * n_cols, 12)
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, 4.5), squeeze=False)
    axes_flat = axes.flatten()
    _apply_black_style(fig, axes_flat)

    for c_idx, roi in enumerate(rois_all):
        ax = axes[0, c_idx]
        for band in bands:
            colour = band_palette.get(band, '#ffffff')
            tv, mean_curve, sem_curve = aggregate_subjects(
                all_subject_results, band, roi, 'pr')
            if tv is None:
                continue
            lbl = BAND_LABELS.get(band, band)
            ax.fill_between(tv,
                             mean_curve - sem_curve,
                             mean_curve + sem_curve,
                             color=colour, alpha=0.15)
            ax.plot(tv, mean_curve, color=colour, linewidth=1.6, label=lbl)

        ax.axvline(0, color='#555555', linewidth=1.0,
                    linestyle='--', alpha=0.8)
        roi_lbl = roi.capitalize() if roi != 'whole' else 'Whole brain'
        ax.set_title(roi_lbl, fontsize=11, fontweight='bold', pad=6)
        ax.set_xlabel('Time (s)', fontsize=9)
        if c_idx == 0:
            ax.set_ylabel('Participation Ratio', fontsize=9)
        if c_idx == n_cols - 1:
            leg = ax.legend(fontsize=7, loc='upper right',
                             framealpha=0.2, edgecolor='#444444',
                             labelcolor='#e0e0e0')
            leg.get_frame().set_facecolor('#1a1a1a')

        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    fig.suptitle(
        f'Intrinsic Dimensionality (Participation Ratio) -- All Bands  |  '
        f'{lockType}-locked  |  {voxRes}',
        color='#e0e0e0', fontsize=11, fontweight='bold', y=1.01
    )
    fig.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir,
                          f'intrinsic_dim_overview_{lockType}_{voxRes}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {fname}')
    return fname


# --- Main (plot-only: reads saved .npz cell outputs) -----------------------

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and plot intrinsic-dimensionality results '
                    'from pre-computed .npz cell files.')
    parser.add_argument('--voxRes',        default='8mm')
    parser.add_argument('--lockTypes',     nargs='+', default=['stim', 'resp'])
    parser.add_argument('--rois',          nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--subjects',      nargs='+', type=int,
                        default=SUBJECT_LIST)
    parser.add_argument('--outdir',        default=None,
                        help='Directory where .npz files AND plots are saved. '
                             'If not set, plots go to '
                             '<bids_root>/derivatives/glueDecoding/intrinsicDim/plots '
                             'and .npz files are expected under each subject.')
    parser.add_argument('--var_threshold', type=float, default=VAR_THRESHOLD_DEFAULT)
    parser.add_argument('--bands',         nargs='+', default=BAND_ORDER)
    args = parser.parse_args()

    bids_root = get_bids_root()

    # If --outdir was given, treat it as both the .npz source AND plot dest.
    # Otherwise: .npz files live per-subject; plots go to a central dir.
    flat_outdir = args.outdir  # may be None
    plot_dir = args.outdir or os.path.join(
        bids_root, 'derivatives', 'glueDecoding', 'intrinsicDim', 'plots')

    rois_all = list(args.rois)
    if 'whole' not in rois_all:
        rois_all.append('whole')

    print(f'intrinsic_dimensionality (plot) | voxRes={args.voxRes} | '
          f'subjects={args.subjects} | rois={rois_all} | bands={args.bands}')

    for lockType in args.lockTypes:
        print(f'\n=== lockType: {lockType} ===')

        # Load per-subject, per-band .npz results
        # Structure: all_results[band] = list of per-subject dicts (one per subject)
        all_results = {}   # band -> [subj0_roi_dict, subj1_roi_dict, ...]
        for band in args.bands:
            band_results = []
            for subjID in args.subjects:
                rd = load_subject_npz(bids_root, subjID, band, lockType,
                                      args.voxRes, rois_all, flat_outdir)
                if rd is None:
                    print(f'  MISSING: sub-{subjID:02d} {band} {lockType}')
                band_results.append(rd)
            all_results[band] = band_results

        # aggregate_subjects expects all_subject_results[i] to be a per-band
        # dict (as produced by compute_subject_dim).  Wrap the npz dicts into
        # the same shape so the plotting functions need no changes.
        # Shape: all_wrapped[i] = {band: {roi: {'pr', 'npcs', 'time_vector'}}}
        all_wrapped = [{} for _ in args.subjects]
        for band in args.bands:
            for i, rd in enumerate(all_results[band]):
                all_wrapped[i][band] = rd  # rd is {roi: {...}} or None per roi

        # 1) Full grid: participation ratio
        plot_dim_figure(all_wrapped, rois_all,
                         metric='pr',
                         metric_label='Participation Ratio',
                         lockType=lockType, voxRes=args.voxRes,
                         outdir=plot_dir, bands=args.bands)

        # 2) Full grid: n_pcs
        plot_dim_figure(all_wrapped, rois_all,
                         metric='npcs',
                         metric_label=f'# PCs (>={int(args.var_threshold*100)}% var)',
                         lockType=lockType, voxRes=args.voxRes,
                         outdir=plot_dir, bands=args.bands)

        # 3) Overview: all bands per ROI
        plot_overview_figure(all_wrapped, rois_all,
                              lockType=lockType, voxRes=args.voxRes,
                              outdir=plot_dir, bands=args.bands)

    print('\nDone.')


if __name__ == '__main__':
    main()

