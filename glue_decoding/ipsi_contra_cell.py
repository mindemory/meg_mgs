#!/usr/bin/env python3
"""
ipsi_contra_cell.py

Per-subject "ipsi vs contra visual" amplitude timecourse -- the raw-signal
companion to linear_decoding_categories_cell.py's P=2 (left/right)
classifier, both driving two_class_scenario's figure (see
run_two_class_scenario.sh / plot_two_class_scenario.py).

Uses the per-HEMISPHERE visual ROI caches (roi='visual_left' /
'visual_right', see atlas.py's MASK_KEYS and constants.HEMI_ROI_NAMES) --
these must be precomputed first:
    python precompute_roi_splits.py --rois visual_left visual_right [--subjects ...]
(one-time; independent of everything else here, since it's just an
additional ROI mask slice of the same G03/G04 files precompute_roi_splits.py
already knows how to cache).

Definitions (matches user's stated convention -- NOT the "contralateral
visual field drives the contralateral hemisphere more strongly" retinotopy
convention, just whichever the ROI/target hemifield LABELS happen to match):
    ipsi curve   = trials where (roi=visual_left  & target in left  hemifield)
                 concatenated with (roi=visual_right & target in right hemifield)
    contra curve = trials where (roi=visual_left  & target in right hemifield)
                 concatenated with (roi=visual_right & target in left  hemifield)
Target hemifield membership uses CATEGORY_SCHEMES[2] (left_right scheme --
see constants.py) via category_labels_for_scheme.

ERP removal (--no_erp_removal to disable, default ON): per (band, roi) cell,
subtract that ROI's own grand trial-average (over ALL trials for this roi/
band, BOTH target hemifields, at native time resolution) from every trial --
same convention as linear_decoding_categories_cell.py / decoding_ts_cell.py
-- computed BEFORE the ipsi/contra hemifield split.

Cross-subject scale: each subject's (band) curve is baseline z-scored (using
BASELINE_WINDOWS['stim'] = [-1.0, 0.0] s, same window as plot_timeseries.py)
against a SINGLE pooled mean/std computed from ALL trials/both hemisphere
ROIs in that band -- i.e. ipsi and contra share one scale factor per subject/
band, so their absolute DIFFERENCE is preserved exactly while normalizing
away cross-subject amplitude-scale differences. That mean/std comes from
RAW single-trial values in the baseline window (not the temporal std of the
already trial-averaged curve, which can be near-zero for a smooth trace and
blow the z-score up to absurd magnitudes -- see run_cell's comment).

Usage:
    python ipsi_contra_cell.py <subjID> [--bands theta alpha beta]
                                          [--voxRes 8mm] [--no_erp_removal]
                                          [--outdir <path>] [--force]
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from constants import AMP_ONLY_BANDS, get_bids_root, category_labels_for_scheme
from io_g04 import load_g04_band

LOCK_TYPE = 'stim'
T_MIN, T_MAX = -1.0, 1.7
BASELINE_WINDOW = (-1.0, 0.0)
ROI_LEFT, ROI_RIGHT = 'visual_left', 'visual_right'


def output_path(bids_root, subjID, band, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'ipsiContraVisual')
    os.makedirs(base, exist_ok=True)
    fname = f'{subName}_task-mgs_ipsiContraVisual_{band}_{voxRes}.npz'
    return os.path.join(base, fname)


def _reduce_sources(amp):
    """(n_trials, n_times, n_sources) -> (n_trials, n_times), mean across sources."""
    return amp.mean(axis=2)


def run_cell(subjID, bands, voxRes, bids_root, remove_erp=True, outdir=None, force=False):
    for band in bands:
        out_path = output_path(bids_root, subjID, band, voxRes, outdir)
        if not force and os.path.exists(out_path):
            print(f'SKIP (exists): {out_path}')
            continue

        try:
            g_left  = load_g04_band(subjID, LOCK_TYPE, band, voxRes, bids_root,
                                     want_phase=False, roi=ROI_LEFT)
            g_right = load_g04_band(subjID, LOCK_TYPE, band, voxRes, bids_root,
                                     want_phase=False, roi=ROI_RIGHT)
        except (FileNotFoundError, ValueError) as e:
            print(f'  SKIP sub-{subjID:02d} {band}: {e}')
            continue
        except Exception:
            print(f'  FAILED sub-{subjID:02d} {band} loading (skipping):', flush=True)
            traceback.print_exc()
            continue

        tv = g_left['time_vector']
        t_mask = (tv >= T_MIN) & (tv <= T_MAX)
        tv_crop = tv[t_mask]

        amp_left  = g_left['amp']    # (n_trials_L, n_times, n_sources_L)
        amp_right = g_right['amp']   # (n_trials_R, n_times, n_sources_R)
        labels_left  = g_left['target_labels'].astype(np.int64)
        labels_right = g_right['target_labels'].astype(np.int64)

        if remove_erp:
            amp_left  = amp_left  - amp_left.mean(axis=0, keepdims=True)
            amp_right = amp_right - amp_right.mean(axis=0, keepdims=True)

        curve_left_all  = _reduce_sources(amp_left)    # (n_trials_L, n_times)
        curve_right_all = _reduce_sources(amp_right)   # (n_trials_R, n_times)

        group_left,  keep_left  = category_labels_for_scheme(labels_left, 2)
        group_right, keep_right = category_labels_for_scheme(labels_right, 2)
        curve_left_kept  = curve_left_all[keep_left]
        curve_right_kept = curve_right_all[keep_right]

        ipsi_trials = np.concatenate([
            curve_left_kept[group_left == 'left'],
            curve_right_kept[group_right == 'right'],
        ], axis=0)
        contra_trials = np.concatenate([
            curve_left_kept[group_left == 'right'],
            curve_right_kept[group_right == 'left'],
        ], axis=0)

        if ipsi_trials.shape[0] == 0 or contra_trials.shape[0] == 0:
            print(f'  SKIP sub-{subjID:02d} {band}: empty ipsi/contra split '
                  f'(ipsi n={ipsi_trials.shape[0]}, contra n={contra_trials.shape[0]})')
            continue

        ipsi_curve   = ipsi_trials.mean(axis=0)[t_mask]
        contra_curve = contra_trials.mean(axis=0)[t_mask]

        # Single pooled baseline scale (see module docstring) -- both hemisphere
        # ROIs, both hemifields, i.e. every trial loaded for this (subject, band).
        # b_mean/b_std are computed from RAW per-trial values in the baseline
        # window (trials x baseline-timepoints), NOT from the already
        # trial-averaged curve's std across time -- the latter (what
        # plot_timeseries.py's _baseline_zscore does) measures how much the
        # MEAN trace wiggles within the baseline window, which for a smooth
        # Hilbert-amplitude trace can be near-zero even though single-trial
        # amplitude varies a lot -- dividing by that near-zero temporal std
        # is what produced the absurd (~1e5-1e6) z-scores seen in practice.
        # True trial-to-trial variance is a far more robust denominator.
        b_mask = (tv_crop >= BASELINE_WINDOW[0]) & (tv_crop <= BASELINE_WINDOW[1])
        if b_mask.any():
            pooled_all = np.concatenate([curve_left_all, curve_right_all], axis=0)
            baseline_vals = pooled_all[:, t_mask][:, b_mask]   # (n_trials_pooled, n_baseline_times)
            b_mean = baseline_vals.mean()
            b_std  = baseline_vals.std()
            if b_std < 1e-12:
                ipsi_curve   = ipsi_curve - b_mean
                contra_curve = contra_curve - b_mean
            else:
                ipsi_curve   = (ipsi_curve - b_mean) / b_std
                contra_curve = (contra_curve - b_mean) / b_std

        print(f'sub-{subjID:02d} | {band} | remove_erp={remove_erp} | '
              f'ipsi n={ipsi_trials.shape[0]} | contra n={contra_trials.shape[0]}', flush=True)

        np.savez_compressed(
            out_path,
            time_vector = tv_crop.astype(np.float32),
            ipsi_curve   = ipsi_curve.astype(np.float32),
            contra_curve = contra_curve.astype(np.float32),
            n_ipsi   = np.array([ipsi_trials.shape[0]]),
            n_contra = np.array([contra_trials.shape[0]]),
            subjID     = np.array([subjID]),
            band       = np.array([band]),
            voxRes     = np.array([voxRes]),
            remove_erp = np.array([remove_erp]),
        )
        print(f'  Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Per-subject ipsi-vs-contra visual amplitude timecourse '
                     '(visual_left/visual_right ROI x left/right target hemifield).')
    parser.add_argument('subjID', type=int)
    parser.add_argument('--bands',  nargs='+', default=list(AMP_ONLY_BANDS))
    parser.add_argument('--voxRes', default='8mm')
    parser.add_argument('--no_erp_removal', action='store_false', dest='remove_erp',
                         help='Skip ERP (grand trial-average, per hemisphere ROI) subtraction '
                              '(default: ERP IS subtracted).')
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    bids_root = get_bids_root()
    print(f'ipsi_contra_cell | sub-{args.subjID:02d} | bands={args.bands} | '
          f'{args.voxRes} | remove_erp={args.remove_erp} | force={args.force}', flush=True)

    t0 = time.time()
    run_cell(args.subjID, list(args.bands), args.voxRes, bids_root,
              remove_erp=args.remove_erp, outdir=args.outdir, force=args.force)
    print(f'Done | sub-{args.subjID:02d} | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
