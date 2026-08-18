#!/usr/bin/env python3
"""
intrinsic_dim_by_location_epochs_cell.py

Per-LOCATION participation ratio (intrinsic dimensionality) over the four
task epochs, per subject/band/ROI -- the per-manifold counterpart of
intrinsic_dim_epochs.py, which computed one PR per (band, roi, epoch) from
the WHOLE trial cloud (all 10 locations pooled). This computes one PR per
LOCATION (10 per cell), matching Will Slatton's suggested GLUE sanity check:
compare glue's "dimension" (one number per manifold-set fit) against the
participation ratio of each individual stimulus-location manifold.

    PR(Sigma) = Tr(Sigma)^2 / Tr(Sigma^2)

Epochs match visual_geometry_epochs_cell.py exactly (fixation / stimulus /
early_delay / late_delay), so this lines up cell-for-cell with the RDM/MDS
geometry work and, once built, with the pooled glue capacity fits.

WHY THIS IS NOT A RAW COVARIANCE ON THE SOURCE SPACE (the naive reading of
"compute its covariance matrix Sigma" from the sanity-check note): n_features
(597 for visual) vastly exceeds trials-per-location-per-epoch (~10-40), and a
sample covariance in that regime is severely rank-deficient -- PR is then
bounded by ~n_trials-1 regardless of the TRUE dimensionality. Measured
directly on synthetic data with a true intrinsic dimensionality of 5: raw PR
gave 2.76 at n=10 trials and only reached 4.64 at n=100, i.e. it never
recovers the true value and is systematically biased by trial count alone --
exactly the conditioning failure the RDM/MDS geometry work in this repo
already solved. The fix is reused unchanged: PCA-project to a shared basis
(fit once per band/roi/epoch cell on ALL locations pooled, so every location's
PR lives in the same subspace and is directly comparable) via
visual_geometry_cell.pca_project, then Ledoit-Wolf shrink that location's own
residual covariance via visual_geometry_cell._ledoit_wolf_cov before computing
PR. Verified this recovers 5.29/6.86/5.69/5.37 at n=10/20/40/100 against the
same true rank-5 structure that raw PR could not recover at any n tested.

Preprocessing matches the rest of the repo: features z-scored across trials at
each timepoint before epoch averaging (visual_geometry_cell.zscore_per_timepoint
-- its mean-subtraction half performs ERP removal, its SD half stops loud
sources from dominating the PCA/covariance).

MIN_TRIALS_PER_LOC (from visual_geometry_cell, currently 2) gates which
locations are usable per cell; below that PR is left NaN for that location
rather than computed on a degenerate covariance.

Output per (band, roi): pr_by_location (n_epochs, 10) and its across-location
mean (n_epochs,) -- the summary scalar the bar-plot / GLUE comparison uses.

Usage:
    python intrinsic_dim_by_location_epochs_cell.py <subjID>
        [--bands theta alpha beta lowgamma highgamma]
        [--rois visual parietal frontal] [--voxRes 8mm]
        [--max_pca_dim 50] [--seed 0] [--outdir <path>] [--force]
"""

import os

os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import time
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from constants import AMP_ONLY_BANDS, get_bids_root
from features import build_features
from io_g04 import load_g04_band
from visual_geometry_cell import (
    LOCATIONS, MAX_PCA_DIM, MIN_TRIALS_PER_LOC, epoch_average,
    zscore_per_timepoint, pca_project, _ledoit_wolf_cov,
)
from visual_geometry_epochs_cell import EPOCHS, EPOCH_ORDER

LOCK_TYPE = 'stim'
CONDITION = 'ampOnly'   # matches intrinsic_dim_epochs.py (amplitude only)


def output_path(bids_root, subjID, band, roi, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'intrinsicDimByLocation')
    os.makedirs(base, exist_ok=True)
    return os.path.join(
        base, f'{subName}_task-mgs_intrinsicDimByLoc_{band}_{roi}_{voxRes}.npz')


def participation_ratio(Sigma):
    lam = np.maximum(np.linalg.eigvalsh(Sigma), 0.0)
    s1 = lam.sum()
    return float(s1 ** 2 / (lam ** 2).sum()) if s1 > 1e-30 else np.nan


def per_location_pr(Xe, y, max_pca_dim, min_trials=MIN_TRIALS_PER_LOC):
    """
    Xe: (n_trials, n_features) epoch-averaged, already z-scored per timepoint
    upstream. y: (n_trials,) location labels.

    Returns (pr_by_loc (10,), pca_dim, n_per_loc (10,)).
    """
    Xp, k, _ = pca_project(Xe, max_pca_dim)
    pr = np.full(len(LOCATIONS), np.nan)
    n_per_loc = np.zeros(len(LOCATIONS), int)
    for li, loc in enumerate(LOCATIONS):
        idx = np.where(y == loc)[0]
        n_per_loc[li] = idx.size
        if idx.size < min_trials:
            continue
        resid = Xp[idx] - Xp[idx].mean(axis=0)
        Sigma, _ = _ledoit_wolf_cov(resid)
        pr[li] = participation_ratio(Sigma)
    return pr, k, n_per_loc


def run_cell(subjID, bands, rois, voxRes, bids_root, max_pca_dim=MAX_PCA_DIM,
             seed=0, outdir=None, force=False):
    for band in bands:
        for roi in rois:
            out_path = output_path(bids_root, subjID, band, roi, voxRes, outdir)
            if not force and os.path.exists(out_path):
                print(f'SKIP (exists): {out_path}', flush=True)
                continue

            t_start = time.time()
            try:
                g04 = load_g04_band(subjID, LOCK_TYPE, band, voxRes, bids_root,
                                     want_phase=False, roi=roi)
                amp = g04['amp']
                tv = g04['time_vector']
                y = g04['target_labels'].astype(int)
                del g04

                X = build_features(CONDITION, amp, None)
                del amp
                X = zscore_per_timepoint(X)

                n_ep = len(EPOCH_ORDER)
                pr_by_loc = np.full((n_ep, len(LOCATIONS)), np.nan)
                pca_dims = np.zeros(n_ep, int)
                n_per_loc = np.zeros((n_ep, len(LOCATIONS)), int)

                for i, ep in enumerate(EPOCH_ORDER):
                    lo, hi = EPOCHS[ep]
                    Xe, _ = epoch_average(X, tv, lo, hi, hi_inclusive=False)
                    pr_by_loc[i], pca_dims[i], n_per_loc[i] = per_location_pr(
                        Xe, y, max_pca_dim)

                pr_mean = np.nanmean(pr_by_loc, axis=1)

                np.savez_compressed(
                    out_path,
                    pr_by_location = pr_by_loc.astype(np.float64),
                    pr_mean        = pr_mean.astype(np.float64),
                    epochs         = np.array(EPOCH_ORDER),
                    epoch_bounds   = np.array([EPOCHS[e] for e in EPOCH_ORDER]),
                    locations      = np.array(LOCATIONS),
                    n_per_location = n_per_loc,
                    pca_dim        = pca_dims,
                    n_trials       = np.array([X.shape[0]]),
                    n_features     = np.array([X.shape[2]]),
                    subjID = np.array([subjID]), band = np.array([band]),
                    roi = np.array([roi]), voxRes = np.array([voxRes]),
                    condition = np.array([CONDITION]), seed = np.array([seed]),
                )
                print(f'sub-{subjID:02d} | {band} | {roi}: N={X.shape[0]} '
                      f'F={X.shape[2]} | ' +
                      '  '.join(f'{e}={m:.2f}(k={k})'
                                for e, m, k in zip(EPOCH_ORDER, pr_mean, pca_dims)) +
                      f' | {time.time() - t_start:.1f}s', flush=True)
                del X
            except (FileNotFoundError, ValueError) as e:
                print(f'  SKIP sub-{subjID:02d} {band}/{roi}: {e}', flush=True)
            except Exception:
                print(f'  FAILED sub-{subjID:02d} {band}/{roi}:', flush=True)
                traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(
        description='Per-location participation ratio over the four task epochs.')
    ap.add_argument('subjID', type=int)
    ap.add_argument('--bands', nargs='+', default=list(AMP_ONLY_BANDS))
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--max_pca_dim', type=int, default=MAX_PCA_DIM)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    bids_root = get_bids_root()
    print(f'intrinsic_dim_by_location_epochs_cell | sub-{args.subjID:02d} | '
          f'bands={args.bands} | rois={args.rois} | {args.voxRes} | '
          f'force={args.force}', flush=True)

    t0 = time.time()
    run_cell(args.subjID, list(args.bands), list(args.rois), args.voxRes, bids_root,
             max_pca_dim=args.max_pca_dim, seed=args.seed,
             outdir=args.outdir, force=args.force)
    print(f'Done | sub-{args.subjID:02d} | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
