#!/usr/bin/env python3
"""
visual_geometry_ts_cell.py

TIME-RESOLVED version of visual_geometry_cell.py: a cross-validated RDM at
every timepoint on a sliding window (no temporal generalization -- each
timepoint is independent, like the decoding timecourses), so the MDS geometry
of the location code can be tracked over time.

Scope vs the earlier epoch-based script:
  KEPT    : the crossnobis RDM (SIGNAL), PCA conditioning, Ledoit-Wolf
            whitening -- all reused directly from visual_geometry_cell.py,
            already validated (unbiasedness, shrinkage vs sklearn, etc).
  DROPPED : radius / participation-ratio-of-noise / alignment angle.
  ADDED   : a per-timepoint label-shuffle NULL, because every MDS spectral
            metric computed downstream has a strongly non-zero chance level
            (see plot_visual_geometry_ts.py) and is therefore uninterpretable
            without one.

Windows: features are averaged over a +/- win_ms window (default 50 ms, i.e.
a 100 ms window) before each RDM, then evaluated every time_stride_ms. The
wider window is the point -- more timepoints averaged per RDM means a less
noisy crossnobis estimate at each step.

Per timepoint, per subject, in that subject's own source space:
  1. PCA-project to k = min(n_trials-1, max_pca_dim, n_features).
  2. Pool within-location residuals -> Ledoit-Wolf shrunk noise covariance
     (whitening applied only if residual dof >= 2k, else unwhitened; flagged).
  3. Crossnobis RDM over the 10 target locations, averaged over n_splits
     random 2-fold partitions.
  4. n_null label-shuffled repeats of steps 2-3. The PCA basis is deliberately
     NOT recomputed under shuffle -- it is unsupervised (never sees the
     labels) so it is legitimately held fixed -- but the noise covariance IS
     recomputed, since "within-location" is itself defined by the labels.

Preprocessing matches the rest of the repo: features z-scored across trials
at each timepoint before windowing (its mean-subtraction half performs ERP
removal; the SD half stops loud sources from dominating the PCA the geometry
lives in -- see visual_geometry_cell.zscore_per_timepoint).

Output per (band, feature_rep, roi): rdm (T, 10, 10) and rdm_null
(T, n_null, 10, 10), both tiny, plus the evaluated time vector. All MDS
spectral analysis happens in plot_visual_geometry_ts.py from these RDMs.

Usage:
    python visual_geometry_ts_cell.py <subjID> [--bands theta alpha beta]
                                       [--feature_reps ampOnly ampPhase]
                                       [--rois visual] [--voxRes 8mm]
                                       [--win_ms 50] [--time_stride_ms 50]
                                       [--n_splits 10] [--n_null 50]
                                       [--max_pca_dim 50] [--seed 0]
                                       [--outdir <path>] [--force]
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
from decoding_ts_cell import DEFAULT_WIN_MS, moving_window_mean
from features import build_features
from io_g04 import load_g04_band
from visual_geometry_cell import (
    LOCATIONS, MIN_TRIALS_PER_LOC, WHITEN_DOF_FACTOR, MAX_PCA_DIM,
    pca_project, _ledoit_wolf_cov, crossvalidated_rdm, zscore_per_timepoint,
)

LOCK_TYPE = 'stim'
DEFAULT_TIME_STRIDE_MS = 50.0
DEFAULT_N_SPLITS = 10
DEFAULT_N_NULL = 50


def output_path(bids_root, subjID, band, feature_rep, roi, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'visualGeometryTS')
    os.makedirs(base, exist_ok=True)
    return os.path.join(
        base, f'{subName}_task-mgs_visGeomTS_{feature_rep}_{band}_{roi}_{voxRes}.npz')


def _noise_cov_inv(Xp, y, k):
    """
    Pooled within-location noise covariance (Ledoit-Wolf shrunk) -> its
    inverse, or (None, nan, dof) when the residual dof does not support
    whitening (caller then falls back to unwhitened cross-validated
    Euclidean, exactly as visual_geometry_cell.compute_cell does).
    """
    resid, n_used = [], 0
    for loc in LOCATIONS:
        idx = np.where(y == loc)[0]
        if idx.size < MIN_TRIALS_PER_LOC:
            continue
        resid.append(Xp[idx] - Xp[idx].mean(axis=0))
        n_used += 1
    if not resid:
        return None, np.nan, 0
    R = np.concatenate(resid, axis=0)
    dof = max(0, R.shape[0] - n_used)
    if dof < WHITEN_DOF_FACTOR * k or R.shape[0] <= 1:
        return None, np.nan, dof
    Sigma, shrink = _ledoit_wolf_cov(R)
    return np.linalg.pinv(Sigma), float(shrink), dof


def run_timepoint(Xe, y, max_pca_dim, n_splits, n_null, seed):
    """
    One timepoint -> (rdm, rdm_null (n_null,10,10), meta dict).

    The PCA basis is fit once and reused for the real and every shuffled
    RDM (unsupervised, so holding it fixed under label shuffling is
    legitimate and keeps the null matched to the real analysis); the noise
    covariance is recomputed per shuffle because it is label-defined.
    """
    Xp, k, explained = pca_project(Xe, max_pca_dim)
    Sinv, shrink, dof = _noise_cov_inv(Xp, y, k)
    rdm, _ = crossvalidated_rdm(Xp, y, n_splits=n_splits, Sigma_inv=Sinv, seed=seed)

    rdm_null = np.full((n_null, len(LOCATIONS), len(LOCATIONS)), np.nan)
    if n_null > 0:
        rng = np.random.default_rng(seed + 12345)
        for j in range(n_null):
            y_perm = rng.permutation(y)
            Sinv_p, _, _ = _noise_cov_inv(Xp, y_perm, k)
            rdm_null[j], _ = crossvalidated_rdm(Xp, y_perm, n_splits=n_splits,
                                                 Sigma_inv=Sinv_p, seed=seed + j)
    meta = dict(pca_dim=k, pca_explained_var=explained,
                whitened=Sinv is not None, lw_shrinkage=shrink, whiten_dof=dof)
    return rdm, rdm_null, meta


def run_cell(subjID, bands, feature_reps, rois, voxRes, bids_root,
             win_ms=DEFAULT_WIN_MS, time_stride_ms=DEFAULT_TIME_STRIDE_MS,
             n_splits=DEFAULT_N_SPLITS, n_null=DEFAULT_N_NULL,
             max_pca_dim=MAX_PCA_DIM, seed=0, outdir=None, force=False):
    for band in bands:
        need_phase = any(fr == 'ampPhase' for fr in feature_reps)
        loaded = None

        for feature_rep in feature_reps:
            for roi in rois:
                out_path = output_path(bids_root, subjID, band, feature_rep, roi,
                                        voxRes, outdir)
                if not force and os.path.exists(out_path):
                    print(f'SKIP (exists): {out_path}', flush=True)
                    continue

                t_start = time.time()
                try:
                    if loaded is None or loaded[0] != roi:
                        g04 = load_g04_band(subjID, LOCK_TYPE, band, voxRes, bids_root,
                                             want_phase=need_phase, roi=roi)
                        loaded = (roi, g04)
                    g04 = loaded[1]

                    amp     = g04['amp']
                    phase   = g04['phase'] if need_phase else None
                    tv      = g04['time_vector']
                    fsample = float(g04['actualRate'])
                    y       = g04['target_labels'].astype(int)

                    X = build_features(feature_rep, amp, phase)
                    X = zscore_per_timepoint(X)
                    X = moving_window_mean(X, fsample, win_ms)

                    stride = max(1, int(round(time_stride_ms * 1e-3 * fsample)))
                    eval_idx = np.arange(0, X.shape[1], stride)
                    tv_eval = tv[eval_idx]
                    n_eval = eval_idx.size

                    rdms = np.full((n_eval, len(LOCATIONS), len(LOCATIONS)), np.nan)
                    rdms_null = np.full((n_eval, n_null, len(LOCATIONS), len(LOCATIONS)),
                                         np.nan) if n_null > 0 else None
                    pca_dims, whit = np.zeros(n_eval, int), np.zeros(n_eval, bool)

                    for i, t in enumerate(eval_idx):
                        rdm, rdm_null, meta = run_timepoint(
                            X[:, t, :], y, max_pca_dim, n_splits, n_null, seed)
                        rdms[i] = rdm
                        if n_null > 0:
                            rdms_null[i] = rdm_null
                        pca_dims[i], whit[i] = meta['pca_dim'], meta['whitened']

                    save_kw = dict(
                        rdm                 = rdms.astype(np.float32),
                        eval_time_vector    = tv_eval.astype(np.float32),
                        locations           = np.array(LOCATIONS),
                        location_angles_deg = np.array([ANGLE_MAPPING[l] for l in LOCATIONS],
                                                        dtype=float),
                        pca_dim             = pca_dims,
                        whitened            = whit,
                        target_labels       = y.astype(np.int32),
                        n_trials            = np.array([X.shape[0]]),
                        n_features          = np.array([X.shape[2]]),
                        win_ms              = np.array([win_ms]),
                        time_stride_ms      = np.array([time_stride_ms]),
                        n_splits            = np.array([n_splits]),
                        n_null              = np.array([n_null]),
                        subjID  = np.array([subjID]),  band = np.array([band]),
                        feature_rep = np.array([feature_rep]), roi = np.array([roi]),
                        voxRes  = np.array([voxRes]),  seed = np.array([seed]),
                        fsample = np.array([fsample]),
                    )
                    if n_null > 0:
                        save_kw['rdm_null'] = rdms_null.astype(np.float32)
                    np.savez_compressed(out_path, **save_kw)

                    print(f'sub-{subjID:02d} | {band} | {feature_rep} | {roi}: '
                          f'N={X.shape[0]} F={X.shape[2]} T={n_eval} | '
                          f'win=+-{win_ms:.0f}ms stride={time_stride_ms:.0f}ms | '
                          f'k={int(np.median(pca_dims))} | whitened={whit.mean()*100:.0f}% | '
                          f'n_null={n_null} | {time.time() - t_start:.1f}s', flush=True)
                    del X
                except (FileNotFoundError, ValueError) as e:
                    print(f'  SKIP sub-{subjID:02d} {band}/{feature_rep}/{roi}: {e}', flush=True)
                except Exception:
                    print(f'  FAILED sub-{subjID:02d} {band}/{feature_rep}/{roi}:', flush=True)
                    traceback.print_exc()
        loaded = None


def main():
    ap = argparse.ArgumentParser(
        description='Time-resolved crossnobis RDM (+ label-shuffle null) for MDS geometry.')
    ap.add_argument('subjID', type=int)
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--feature_reps', nargs='+', default=['ampOnly', 'ampPhase'],
                     choices=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual'])
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--win_ms', type=float, default=DEFAULT_WIN_MS,
                     help='ONE-SIDED half-width; default 50 => a 100 ms window.')
    ap.add_argument('--time_stride_ms', type=float, default=DEFAULT_TIME_STRIDE_MS)
    ap.add_argument('--n_splits', type=int, default=DEFAULT_N_SPLITS)
    ap.add_argument('--n_null', type=int, default=DEFAULT_N_NULL,
                     help='Label-shuffle repeats per timepoint (0 disables; the MDS '
                          'metrics have strongly non-zero chance levels, so keep this on).')
    ap.add_argument('--max_pca_dim', type=int, default=MAX_PCA_DIM)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    bids_root = get_bids_root()
    print(f'visual_geometry_ts_cell | sub-{args.subjID:02d} | bands={args.bands} | '
          f'feature_reps={args.feature_reps} | rois={args.rois} | {args.voxRes} | '
          f'win_ms={args.win_ms} (=> {2*args.win_ms:.0f} ms window) | '
          f'stride={args.time_stride_ms} | n_splits={args.n_splits} | '
          f'n_null={args.n_null} | force={args.force}', flush=True)

    t0 = time.time()
    run_cell(args.subjID, list(args.bands), list(args.feature_reps), list(args.rois),
             args.voxRes, bids_root, win_ms=args.win_ms,
             time_stride_ms=args.time_stride_ms, n_splits=args.n_splits,
             n_null=args.n_null, max_pca_dim=args.max_pca_dim, seed=args.seed,
             outdir=args.outdir, force=args.force)
    print(f'Done | sub-{args.subjID:02d} | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
