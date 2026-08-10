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
from constants import (AMP_ONLY_BANDS, SUBJECT_LIST, ROI_NAMES,
                       get_bids_root)
from io_g03 import load_g03_unfiltered
from io_g04 import load_g04_band

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


# --- Core dimensionality estimators ----------------------------------------

def participation_ratio(X_t):
    """
    Compute the participation ratio (effective dimensionality) for a single
    timepoint.

    X_t : (n_trials, n_sources)  -- z-scored recommended but not required.
    Returns scalar float.
    """
    # Trial covariance (n_sources x n_sources), unbiased
    C = np.cov(X_t, rowvar=False)           # (n_sources, n_sources)
    # Eigenvalues only (faster than full SVD; eigvalsh for symmetric matrices)
    lam = np.linalg.eigvalsh(C)
    lam = np.maximum(lam, 0.0)             # numerical safety
    s1 = lam.sum()
    if s1 < 1e-30:
        return 1.0
    s2 = (lam ** 2).sum()
    return float(s1 ** 2 / s2)


def n_pcs_for_var(X_t, var_threshold=0.90):
    """
    Number of principal components needed to explain >= var_threshold of
    the total variance at one timepoint.

    X_t : (n_trials, n_sources)
    Returns int.
    """
    C = np.cov(X_t, rowvar=False)
    lam = np.sort(np.maximum(np.linalg.eigvalsh(C), 0.0))[::-1]
    s = lam.sum()
    if s < 1e-30:
        return 1
    cum = np.cumsum(lam) / s
    hits = np.where(cum >= var_threshold)[0]
    return int(hits[0] + 1) if hits.size > 0 else len(lam)


def dim_over_time(data, var_threshold=0.90):
    """
    Compute participation ratio and n_pcs over every timepoint.

    data : (n_trials, n_times, n_sources)
    Returns:
        pr   : (n_times,) float  -- participation ratio
        npcs : (n_times,) int    -- n PCs for >= var_threshold variance
    """
    n_trials, n_times, n_sources = data.shape
    pr   = np.zeros(n_times)
    npcs = np.zeros(n_times, dtype=int)
    for t in range(n_times):
        X_t = data[:, t, :]                 # (n_trials, n_sources)
        # z-score across sources (centre trials; each source unit-variance)
        mu = X_t.mean(axis=0, keepdims=True)
        sd = X_t.std(axis=0, keepdims=True)
        sd[sd < 1e-10] = 1.0
        X_t_z = (X_t - mu) / sd
        pr[t]   = participation_ratio(X_t_z)
        npcs[t] = n_pcs_for_var(X_t_z, var_threshold)
    return pr, npcs


# --- Per-subject, per-band loaders -----------------------------------------

def compute_subject_dim(subjID, lockType, voxRes, bids_root, atlas_masks,
                        rois_all, var_threshold):
    """
    For one subject, compute PR and n_pcs over time for every (band, ROI)
    combination.

    Returns a nested dict:
        result[band][roi_name] = {'pr': (n_times,), 'npcs': (n_times,),
                                   'time_vector': (n_times,)}
    or None entries where data files are missing.
    """
    # Load G03 (needed for inside_pos in ALL conditions)
    try:
        g03 = load_g03_unfiltered(subjID, lockType, voxRes, bids_root)
    except FileNotFoundError as e:
        print(f'  sub-{subjID:02d} G03 missing: {e}')
        return None

    inside_pos = g03['inside_pos']

    # Build ROI index maps once per subject
    roi_indices = {}
    for roi in rois_all:
        if roi == 'whole':
            roi_indices[roi] = np.arange(g03['data'].shape[2])
        else:
            roi_indices[roi] = roi_local_indices(atlas_masks, inside_pos, roi)

    result = {}

    # Broadband (G03 data)
    result['broadband'] = {}
    for roi in rois_all:
        idx = roi_indices[roi]
        if idx.size == 0:
            print(f'  sub-{subjID:02d} broadband {roi}: empty ROI, skipping')
            result['broadband'][roi] = None
            continue
        data_roi = g03['data'][:, :, idx]          # (n_trials, n_times, n_roi)
        pr, npcs = dim_over_time(data_roi, var_threshold)
        result['broadband'][roi] = {
            'pr':          pr,
            'npcs':        npcs,
            'time_vector': g03['time_vector'],
        }
    print(f'  sub-{subjID:02d} broadband done '
          f'({g03["data"].shape[0]} trials, {g03["data"].shape[1]} times)')

    # G04 bands (amplitude only; phase not used for dimensionality)
    for band in AMP_ONLY_BANDS:
        result[band] = {}
        try:
            g04 = load_g04_band(subjID, lockType, band, voxRes, bids_root,
                                 want_phase=False)
        except (FileNotFoundError, ValueError) as e:
            print(f'  sub-{subjID:02d} {band}: {e}')
            for roi in rois_all:
                result[band][roi] = None
            continue

        # G04 columns same as G03 columns - reuse same inside_pos
        amp = g04['amp']   # (n_trials, n_times, n_sources)
        for roi in rois_all:
            idx = roi_indices[roi]
            if idx.size == 0:
                result[band][roi] = None
                continue
            data_roi = amp[:, :, idx]
            pr, npcs = dim_over_time(data_roi, var_threshold)
            result[band][roi] = {
                'pr':          pr,
                'npcs':        npcs,
                'time_vector': g04['time_vector'],
            }
        print(f'  sub-{subjID:02d} {band} done '
              f'({amp.shape[0]} trials, {amp.shape[1]} times)')

    return result


# --- Cross-subject aggregation ----------------------------------------------

def aggregate_subjects(all_subject_results, band, roi, metric='pr'):
    """
    Stack per-subject curves for (band, roi) and return mean +/- SEM.
    Returns (time_vector, mean_curve, sem_curve) or (None, None, None)
    if no valid subjects.
    """
    curves = []
    time_vector = None
    for subj_result in all_subject_results:
        if subj_result is None:
            continue
        entry = subj_result.get(band, {}).get(roi)
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

            # Count subjects
            n_subj = sum(
                1 for s in all_subject_results
                if s is not None and s.get(band, {}).get(roi) is not None
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


# --- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Intrinsic dimensionality of MEG source data over time.')
    parser.add_argument('--voxRes',        default='8mm')
    parser.add_argument('--lockTypes',     nargs='+', default=['stim', 'resp'])
    parser.add_argument('--rois',          nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--subjects',      nargs='+', type=int,
                        default=SUBJECT_LIST)
    parser.add_argument('--outdir',        default=None)
    parser.add_argument('--var_threshold', type=float,
                        default=VAR_THRESHOLD_DEFAULT,
                        help='Variance threshold for n_pcs metric '
                             '(default: 0.90)')
    parser.add_argument('--bands',         nargs='+', default=BAND_ORDER,
                        help='Which bands to process (default: all)')
    args = parser.parse_args()

    bids_root = get_bids_root()
    outdir = args.outdir or os.path.join(
        bids_root, 'derivatives', 'glueDecoding', 'intrinsicDim')

    # Always include whole-brain in ROI list
    rois_all = list(args.rois)
    if 'whole' not in rois_all:
        rois_all.append('whole')

    print(f'intrinsic_dimensionality | voxRes={args.voxRes} | '
          f'subjects={args.subjects} | rois={rois_all} | '
          f'bands={args.bands} | var_threshold={args.var_threshold}')

    for lockType in args.lockTypes:
        print(f'\n=== lockType: {lockType} ===')

        # Load atlas masks (shared across subjects)
        atlas_masks = load_atlas_masks(args.voxRes, bids_root)

        # Per-subject computation
        all_subject_results = []
        for subjID in args.subjects:
            print(f'\nsub-{subjID:02d} ...')
            subj_result = compute_subject_dim(
                subjID, lockType, args.voxRes, bids_root,
                atlas_masks, rois_all, args.var_threshold)
            all_subject_results.append(subj_result)

        # Plotting
        # 1) Full grid: rows=bands, cols=ROIs, metric=participation ratio
        plot_dim_figure(all_subject_results, rois_all,
                         metric='pr',
                         metric_label='Participation Ratio',
                         lockType=lockType, voxRes=args.voxRes,
                         outdir=outdir, bands=args.bands)

        # 2) Full grid: rows=bands, cols=ROIs, metric=n_pcs
        plot_dim_figure(all_subject_results, rois_all,
                         metric='npcs',
                         metric_label=f'# PCs (>={int(args.var_threshold*100)}% var)',
                         lockType=lockType, voxRes=args.voxRes,
                         outdir=outdir, bands=args.bands)

        # 3) Overview: one panel per ROI, all bands overlaid
        plot_overview_figure(all_subject_results, rois_all,
                              lockType=lockType, voxRes=args.voxRes,
                              outdir=outdir, bands=args.bands)

    print('\nDone.')


if __name__ == '__main__':
    main()
