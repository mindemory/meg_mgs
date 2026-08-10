#!/usr/bin/env python3
"""
aggregate_glue_decoding.py

Cross-subject aggregation + plotting for glue_decoding's per-subject TGM
pickles (produced by run_glue_cell.py). Strictly a post-hoc step: it never
touches raw features or pools trials across subjects -- it loads each
subject's already-computed (n_trials, n_train_t, n_test_t) predicted-angle
TGM, converts it to a per-subject (n_train_t, n_test_t) mean absolute
angular-error surface, then averages THAT across subjects.

Usage:
    python aggregate_glue_decoding.py [--voxRes 8mm] [--lockTypes stim resp]
                                       [--outdir <bids_root>/derivatives/glueDecoding/aggregated]
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from constants import AMP_ONLY_BANDS, AMP_PHASE_BANDS, ANGLE_MAPPING, ROI_NAMES, get_bids_root

# The valid (condition, band) grid glue_decoding produces (see run_glue_cell.py).
CONDITION_BANDS = (
    [('unfiltered', 'broadband')]
    + [('ampOnly', b) for b in AMP_ONLY_BANDS]
    + [('ampPhase', b) for b in AMP_PHASE_BANDS]
)

# Sequential single-hue (blue, light->dark) ramp -- dataviz skill's default
# sequential palette (references/palette.md), used here for a magnitude
# metric (mean absolute angular error): lightest = near 0 (best decoding),
# darkest = 180 (worst/anti-correlated). Fixed vmin=0/vmax=180 across every
# panel in the comparison grid so color is directly comparable across
# ROI/band/condition/lockType.
_SEQ_BLUE_HEXES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
ERROR_VMIN, ERROR_VMAX = 0.0, 180.0
CHANCE_LEVEL = 90.0  # circular-uniform expected mean abs error for 10 equally-spaced targets


def sequential_cmap():
    return LinearSegmentedColormap.from_list('glue_seq_blue', _SEQ_BLUE_HEXES)


def discover_subject_files(bids_root, condition, band, lockType, voxRes):
    pattern = os.path.join(
        bids_root, 'derivatives', 'sub-*', 'sourceRecon', 'decodingGlue',
        f'sub-*_task-mgs_glueTGM_{condition}_{band}_{lockType}_{voxRes}.pkl')
    return sorted(glob.glob(pattern))


def per_subject_mean_abs_err(output, roi_name):
    """
    output: one subject's unpickled dict from run_glue_cell.py.
    Returns (n_train_t, n_test_t) mean absolute angular error across trials,
    folded into [0, 180] (0=perfect, 90=chance, 180=anti-correlated).
    Arithmetic mean is appropriate here -- these are unsigned magnitudes, not
    directions, so no circular-mean wraparound issue (unlike raw predicted
    angle, which does need circular treatment).
    """
    pred = output[f'pred_angles_deg_{roi_name}']            # (n_trials, n_train_t, n_test_t)
    target_labels = output['target_labels']
    target_deg = np.array([ANGLE_MAPPING[int(t)] for t in target_labels])[:, None, None]
    diff = pred - target_deg
    abs_err = 180.0 - np.abs(np.abs(diff) - 180.0)           # fold circular diff into [0, 180]
    return abs_err.mean(axis=0)


def aggregate_one_combo(bids_root, condition, band, lockType, voxRes, rois):
    """Returns {roi_name: {'mean_abs_err': (n_train_t,n_test_t), 'n_subjects': int,
    'time_vector': (n_train_t,)}} or None if no subject files were found."""
    files = discover_subject_files(bids_root, condition, band, lockType, voxRes)
    if not files:
        return None

    per_roi_stacks = {roi: [] for roi in rois}
    time_vector = None
    for fpath in files:
        with open(fpath, 'rb') as fh:
            output = pickle.load(fh)
        if time_vector is None:
            time_vector = output['time_vector']
        for roi in rois:
            per_roi_stacks[roi].append(per_subject_mean_abs_err(output, roi))

    result = {}
    for roi in rois:
        stacked = np.stack(per_roi_stacks[roi], axis=0)      # (n_subjects, n_train_t, n_test_t)
        result[roi] = {
            'mean_abs_err': stacked.mean(axis=0),
            'n_subjects': stacked.shape[0],
            'time_vector': time_vector,
        }
    return result


def save_aggregated(outdir, condition, band, roi, lockType, voxRes, agg):
    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f'{condition}_{band}_{roi}_{lockType}_{voxRes}.npz')
    np.savez(fpath, mean_abs_err=agg['mean_abs_err'], n_subjects=agg['n_subjects'],
              time_vector=agg['time_vector'])
    return fpath


def plot_combo(plot_dir, condition, band, lockType, voxRes, per_roi_result, rois):
    """One figure per (condition, band, lockType): ROI panels side by side,
    shared colorbar, fixed [0,180] scale, dashed chance-level (90) contour."""
    os.makedirs(plot_dir, exist_ok=True)
    cmap = sequential_cmap()

    fig, axes = plt.subplots(1, len(rois), figsize=(4.2 * len(rois), 4), squeeze=False)
    axes = axes[0]
    im = None
    for ax, roi in zip(axes, rois):
        agg = per_roi_result[roi]
        tv = agg['time_vector']
        im = ax.imshow(agg['mean_abs_err'], origin='lower', aspect='auto', cmap=cmap,
                        vmin=ERROR_VMIN, vmax=ERROR_VMAX,
                        extent=[tv[0], tv[-1], tv[0], tv[-1]])
        ax.contour(tv, tv, agg['mean_abs_err'], levels=[CHANCE_LEVEL],
                   colors='white', linewidths=1, linestyles='dashed')
        ax.set_title(f'{roi} (n={agg["n_subjects"]})')
        ax.set_xlabel('Test time (s)')
        ax.plot([tv[0], tv[-1]], [tv[0], tv[-1]], color='white', linewidth=0.5, alpha=0.5)
    axes[0].set_ylabel('Train time (s)')

    fig.suptitle(f'{condition} / {band} / {lockType} / {voxRes} -- mean abs. angular error')
    cbar = fig.colorbar(im, ax=list(axes), shrink=0.85, label='Mean abs. angular error (deg)')
    cbar.ax.axhline(CHANCE_LEVEL, color='white', linewidth=1, linestyle='dashed')

    out_fpath = os.path.join(plot_dir, f'{condition}_{band}_{lockType}_{voxRes}.png')
    fig.savefig(out_fpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_fpath


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxRes', default='8mm')
    parser.add_argument('--lockTypes', nargs='+', default=['stim', 'resp'])
    parser.add_argument('--rois', nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--outdir', default=None)
    args = parser.parse_args()

    bids_root = get_bids_root()
    outdir = args.outdir or os.path.join(bids_root, 'derivatives', 'glueDecoding', 'aggregated')
    plot_dir = os.path.join(outdir, 'plots')

    for lockType in args.lockTypes:
        for condition, band in CONDITION_BANDS:
            per_roi_result = aggregate_one_combo(bids_root, condition, band, lockType,
                                                  args.voxRes, args.rois)
            if per_roi_result is None:
                print(f'SKIP {condition}/{band}/{lockType}: no subject files found')
                continue

            for roi in args.rois:
                fpath = save_aggregated(outdir, condition, band, roi, lockType, args.voxRes,
                                         per_roi_result[roi])
                print(f'Saved: {fpath} (n_subjects={per_roi_result[roi]["n_subjects"]})')

            plot_fpath = plot_combo(plot_dir, condition, band, lockType, args.voxRes,
                                     per_roi_result, args.rois)
            print(f'Plotted: {plot_fpath}')


if __name__ == '__main__':
    main()
