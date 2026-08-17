#!/usr/bin/env python3
"""
visual_geometry_epochs_cell.py

EPOCH-based crossnobis RDMs for MDS geometry -- the four-epoch counterpart of
visual_geometry_ts_cell.py's sliding window. One RDM per
(band, condition, roi, epoch), plus a label-shuffle null.

EPOCHS (half-open [lo, hi), so adjacent epochs never share a sample):
    fixation    -1.0 .. 0.0 s
    stimulus     0.0 .. 0.2 s
    early_delay  0.2 .. 0.8 s
    late_delay   1.0 .. 1.6 s
Note the deliberate 0.8-1.0 s gap: early and late delay are separated rather
than contiguous, so any difference between them reflects two clearly distinct
periods rather than a boundary drawn through the middle of one.

CAVEAT worth carrying into interpretation -- the epochs are NOT equal length
(1.0 / 0.2 / 0.6 / 0.6 s). The stimulus epoch averages over ~5x fewer
timepoints than fixation, so its RDM is intrinsically noisier, and a lower
ring-ness there is partly a sample-size effect rather than purely a geometric
one. The per-epoch label-shuffle null absorbs this (each epoch is compared to
a null computed with that epoch's own trial/timepoint count), which is exactly
why the null is computed per cell rather than assumed.

BANDS / CONDITIONS: theta/alpha/beta carry saved phase and so run both
ampOnly and ampPhase; lowgamma/highgamma have no phase
(constants.AMP_PHASE_BANDS) and are skipped for ampPhase automatically, giving
amplitude-only cells for them.

Everything else matches visual_geometry_ts_cell.py and is reused from the
already-validated helpers: per-timepoint z-scoring before epoch averaging (its
mean-subtraction half IS the ERP removal), PCA conditioning to
min(n_trials-1, max_pca_dim, n_features), Ledoit-Wolf shrunk whitening applied
only when residual dof >= 2k, and crossnobis averaged over n_splits random
2-fold partitions. The PCA basis is held fixed across the shuffles (it is
unsupervised) while the noise covariance is recomputed per shuffle (it is
label-defined).

Usage:
    python visual_geometry_epochs_cell.py <subjID>
        [--bands theta alpha beta lowgamma highgamma]
        [--conditions ampOnly ampPhase] [--rois visual parietal frontal]
        [--voxRes 8mm] [--n_splits 10] [--n_null 100]
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

from constants import AMP_PHASE_BANDS, ANGLE_MAPPING, get_bids_root
from features import build_features
from io_g04 import load_g04_band
from visual_geometry_cell import (
    LOCATIONS, MAX_PCA_DIM, epoch_average, zscore_per_timepoint,
)
from visual_geometry_ts_cell import run_timepoint

LOCK_TYPE = 'stim'
DEFAULT_N_SPLITS = 10
DEFAULT_N_NULL = 100

# (lo, hi) half-open. See module docstring for the 0.8-1.0 s gap.
EPOCHS = {
    'fixation':    (-1.0, 0.0),
    'stimulus':    ( 0.0, 0.2),
    'early_delay': ( 0.2, 0.8),
    'late_delay':  ( 1.0, 1.6),
}
EPOCH_ORDER = ('fixation', 'stimulus', 'early_delay', 'late_delay')


def output_path(bids_root, subjID, band, condition, roi, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'visualGeometryEpochs')
    os.makedirs(base, exist_ok=True)
    return os.path.join(
        base, f'{subName}_task-mgs_visGeomEp_{condition}_{band}_{roi}_{voxRes}.npz')


def run_cell(subjID, bands, conditions, rois, voxRes, bids_root,
             n_splits=DEFAULT_N_SPLITS, n_null=DEFAULT_N_NULL,
             max_pca_dim=MAX_PCA_DIM, seed=0, outdir=None, force=False):
    for band in bands:
        for condition in conditions:
            want_phase = (condition == 'ampPhase')
            if want_phase and band not in AMP_PHASE_BANDS:
                print(f'SKIP {condition}/{band}: no saved phase '
                      f'(AMP_PHASE_BANDS={AMP_PHASE_BANDS})', flush=True)
                continue

            for roi in rois:
                out_path = output_path(bids_root, subjID, band, condition, roi,
                                        voxRes, outdir)
                if not force and os.path.exists(out_path):
                    print(f'SKIP (exists): {out_path}', flush=True)
                    continue

                t_start = time.time()
                try:
                    g04 = load_g04_band(subjID, LOCK_TYPE, band, voxRes, bids_root,
                                         want_phase=want_phase, roi=roi)
                    amp   = g04['amp']
                    phase = g04['phase'] if want_phase else None
                    tv    = g04['time_vector']
                    y     = g04['target_labels'].astype(int)

                    X = build_features(condition, amp, phase)
                    del amp, phase, g04
                    X = zscore_per_timepoint(X)

                    n_ep = len(EPOCH_ORDER)
                    rdms = np.full((n_ep, len(LOCATIONS), len(LOCATIONS)), np.nan)
                    rdms_null = (np.full((n_ep, n_null, len(LOCATIONS), len(LOCATIONS)),
                                          np.nan) if n_null > 0 else None)
                    pca_dims = np.zeros(n_ep, int)
                    whit = np.zeros(n_ep, bool)
                    n_win = np.zeros(n_ep, int)

                    for i, ep in enumerate(EPOCH_ORDER):
                        lo, hi = EPOCHS[ep]
                        Xe, n_times = epoch_average(X, tv, lo, hi, hi_inclusive=False)
                        n_win[i] = n_times
                        rdm, rdm_null, meta = run_timepoint(
                            Xe, y, max_pca_dim, n_splits, n_null, seed)
                        rdms[i] = rdm
                        if n_null > 0:
                            rdms_null[i] = rdm_null
                        pca_dims[i], whit[i] = meta['pca_dim'], meta['whitened']

                    save_kw = dict(
                        rdm                 = rdms.astype(np.float32),
                        epochs              = np.array(EPOCH_ORDER),
                        epoch_bounds        = np.array([EPOCHS[e] for e in EPOCH_ORDER]),
                        n_window_times      = n_win,
                        locations           = np.array(LOCATIONS),
                        location_angles_deg = np.array([ANGLE_MAPPING[l] for l in LOCATIONS],
                                                        dtype=float),
                        pca_dim             = pca_dims,
                        whitened            = whit,
                        target_labels       = y.astype(np.int32),
                        n_trials            = np.array([X.shape[0]]),
                        n_features          = np.array([X.shape[2]]),
                        n_splits            = np.array([n_splits]),
                        n_null              = np.array([n_null]),
                        subjID = np.array([subjID]), band = np.array([band]),
                        condition = np.array([condition]), roi = np.array([roi]),
                        voxRes = np.array([voxRes]), seed = np.array([seed]),
                    )
                    if n_null > 0:
                        save_kw['rdm_null'] = rdms_null.astype(np.float32)
                    np.savez_compressed(out_path, **save_kw)

                    print(f'sub-{subjID:02d} | {band} | {condition} | {roi}: '
                          f'N={X.shape[0]} F={X.shape[2]} | '
                          f'epoch timepoints={list(n_win)} | k={int(np.median(pca_dims))} | '
                          f'whitened={whit.mean()*100:.0f}% | n_null={n_null} | '
                          f'{time.time() - t_start:.1f}s', flush=True)
                    del X
                except (FileNotFoundError, ValueError) as e:
                    print(f'  SKIP sub-{subjID:02d} {band}/{condition}/{roi}: {e}', flush=True)
                except Exception:
                    print(f'  FAILED sub-{subjID:02d} {band}/{condition}/{roi}:', flush=True)
                    traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(
        description='Epoch-based crossnobis RDMs (+ label-shuffle null) for MDS geometry.')
    ap.add_argument('subjID', type=int)
    ap.add_argument('--bands', nargs='+',
                     default=['theta', 'alpha', 'beta', 'lowgamma', 'highgamma'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'],
                     choices=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--n_splits', type=int, default=DEFAULT_N_SPLITS)
    ap.add_argument('--n_null', type=int, default=DEFAULT_N_NULL)
    ap.add_argument('--max_pca_dim', type=int, default=MAX_PCA_DIM)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    bids_root = get_bids_root()
    print(f'visual_geometry_epochs_cell | sub-{args.subjID:02d} | bands={args.bands} | '
          f'conditions={args.conditions} | rois={args.rois} | {args.voxRes} | '
          f'epochs={ {e: EPOCHS[e] for e in EPOCH_ORDER} } | n_splits={args.n_splits} | '
          f'n_null={args.n_null} | force={args.force}', flush=True)

    t0 = time.time()
    run_cell(args.subjID, list(args.bands), list(args.conditions), list(args.rois),
             args.voxRes, bids_root, n_splits=args.n_splits, n_null=args.n_null,
             max_pca_dim=args.max_pca_dim, seed=args.seed,
             outdir=args.outdir, force=args.force)
    print(f'Done | sub-{args.subjID:02d} | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
