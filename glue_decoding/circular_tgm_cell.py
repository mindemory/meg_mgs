#!/usr/bin/env python3
"""
circular_tgm_cell.py

Temporal Generalization Matrix (TGM) for CIRCULAR (sin/cos) target-location
decoding, one subject, via closed-form LOO ridge -- the TGM generalization of
decoding_ts_cell.py's ridge_loocv_timeseries (same targets, same estimator
family, same LOO semantics), extended so a model trained at time t1 is tested
at every time t2.

WHY RIDGE AND NOT SVR
svr_tgm.py does the same analysis with two RBF-SVRs refit inside the LOO loop:
that is n_trials x n_train_t separate pairs of SVR fits (~300 x 54 = 16k fits
per cell, each on ~300 x 600 data), which is what makes the SVR TGM
hours-per-cell. Ridge admits an exact closed form for the entire LOO TGM with
ONE SVD per TRAIN timepoint and no refitting at all -- identical LOO
semantics, ~4 orders of magnitude less work. Derivation:

  At train time t1 (features X1, N x F, z-scored), let
      A = X1' X1 + alpha I,   P = A^-1,   Q = P X1'   (F x N)
      g = Q y                      (full-data ridge weights)
      h = diag(X1 Q)               (leverages)
      r = X1 g - y                 (full-data residuals)
  Dropping trial i (Sherman-Morrison on A - x1_i x1_i'):
      beta_-i = g + p_i * r_i / (1 - h_i),      p_i = Q[:, i]
  so the LOO prediction at ANY test time t2 (features X2) is
      pred(t2)_i = (X2 g)_i + (x2_i . p_i) * c_i,     c_i = r_i / (1 - h_i)
                 = (X2 g)_i + diag(X2 Q)_i * c_i.

  Only the DIAGONAL of X2 Q is needed, so each (t1, t2) pair costs O(N F),
  not O(N^2 F) -- that is what makes the full LOO TGM cheap. Setting t2 = t1
  reduces this algebraically to the standard LOO identity
  yhat_i^(-i) = y_i + r_i/(1-h_i) that decoding_ts_cell.py already uses, so
  the TGM diagonal must reproduce that script's timeseries exactly -- checked
  numerically in validation rather than assumed.

Everything is computed once for BOTH targets (sin and cos) since Q, h, and
the leverages depend only on X1, never on the labels; the two targets differ
only in g and c. Angles come back via arctan2(pred_sin, pred_cos).

FEATURE SCALING (a real choice, not a detail)
Each timepoint is z-scored across trials using ITS OWN mean/SD (--test_scaling
own, the default), matching every other decoder in this repo. svr_tgm.py
instead standardizes test timepoints with the TRAIN timepoint's mean/SD
(--test_scaling train reproduces that). The default isolates PATTERN
generalization: t1-vs-t2 amplitude differences are normalized away, so the
matrix answers "does the spatial pattern transfer", not "does the pattern
transfer AND have comparable amplitude". Note that for circular decoding the
distinction is weaker than it looks -- a uniform shrinkage of pred_sin and
pred_cos leaves arctan2 unchanged -- so the two options differ only through
non-uniform per-feature rescaling.

ERP is removed (spec: ERP-removed only, no ERP-kept arm). Note that
per-timepoint z-scoring already subtracts the across-trial mean at each
timepoint, which IS the ERP, so the explicit subtraction is mathematically
redundant here; it is kept because it is free, and because it keeps this
script's preprocessing line-for-line comparable with decoding_ts_cell.py.

TIME RESOLUTION
A TGM is quadratic in evaluated timepoints, so it is evaluated on a strided
grid (--time_stride_ms, default 50 ms, the same stride
linear_decoding_categories_cell.py uses) after +/- win_ms moving-window
averaging. At ~100 Hz storage rate over -1.0..1.7 s that is ~54 timepoints ->
~2.9k (train, test) cells, versus ~1.8M if run at native resolution.

Outputs per (band, condition, roi): the (T, T) cross-subject-ready error
summaries, NOT the (N, T, T) trial-level predictions (which would be ~2 GB
across the grid); pass --save_trials if trial-level output is needed.

Usage:
    python circular_tgm_cell.py <subjID> [--bands theta alpha beta]
                                 [--conditions ampOnly ampPhase]
                                 [--rois visual parietal frontal]
                                 [--voxRes 8mm] [--win_ms 50]
                                 [--time_stride_ms 50] [--alpha 1.0]
                                 [--test_scaling own|train]
                                 [--outdir <path>] [--force] [--save_trials]
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
from scipy.stats import circmean

from constants import AMP_PHASE_BANDS, ROI_NAMES, ANGLE_MAPPING, get_bids_root
from decoding_ts_cell import DEFAULT_WIN_MS, circular_dist, moving_window_mean
from features import build_features
from io_g04 import load_g04_band

LOCK_TYPE = 'stim'
DEFAULT_TIME_STRIDE_MS = 50.0
RIDGE_ALPHA = 1.0          # matches decoding_ts_cell.py's RIDGE_ALPHA
CHANCE_ERROR_DEG = 90.0    # expected |circular error| with no information


def output_path(bids_root, subjID, band, roi, condition, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'circularTGM')
    os.makedirs(base, exist_ok=True)
    return os.path.join(
        base, f'{subName}_task-mgs_circTGM_{condition}_{band}_{roi}_{voxRes}.npz')


def zscore_per_timepoint(X):
    """(N, T, F) -> z-scored across TRIALS independently at each timepoint."""
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-10, np.asarray(1.0, dtype=sd.dtype), sd)
    return (X - mu) / sd


def ridge_loo_tgm(X, angles_deg, alpha=RIDGE_ALPHA, test_scaling='own',
                   raw_X=None):
    """
    Closed-form LOO ridge TGM for sin/cos circular targets -- see module
    docstring for the derivation.

    X          : (N, T, F) features, ALREADY z-scored per timepoint when
                 test_scaling='own'.
    raw_X      : (N, T, F) pre-z-scoring features, required when
                 test_scaling='train' (each train timepoint then supplies the
                 mean/SD used to standardize every test timepoint).
    angles_deg : (N,) true target angles.

    Returns pred_angles (N, T, T) float32, indexed [trial, train_t, test_t].
    """
    n_trials, n_times, n_feat = X.shape
    ang = np.radians(angles_deg)
    Y = np.stack([np.sin(ang), np.cos(ang)], axis=1)          # (N, 2)

    pred_angles = np.empty((n_trials, n_times, n_times), dtype=np.float32)

    for t1 in range(n_times):
        if test_scaling == 'train':
            mu = raw_X[:, t1, :].mean(axis=0, keepdims=True)
            sd = raw_X[:, t1, :].std(axis=0, keepdims=True)
            sd = np.where(sd < 1e-10, np.asarray(1.0, dtype=sd.dtype), sd)
            X1 = (raw_X[:, t1, :] - mu) / sd
        else:
            X1 = X[:, t1, :]
        X1 = np.asarray(X1, dtype=np.float64)

        # One SVD per TRAIN timepoint serves every test timepoint and both targets.
        U, S, Vt = np.linalg.svd(X1, full_matrices=False)      # U (N,r) S (r,) Vt (r,F)
        S2 = S ** 2
        d = S2 / (S2 + alpha)
        h = np.clip((U ** 2) @ d, 0.0, 1.0 - 1e-9)             # diag(X1 Q)
        lev = 1.0 - h

        UtY = U.T @ Y                                          # (r, 2)
        Yhat_full = U @ (d[:, None] * UtY)                     # (N, 2)
        C = (Yhat_full - Y) / lev[:, None]                     # (N, 2)

        s_over = S / (S2 + alpha)                              # (r,)
        G = Vt.T @ (s_over[:, None] * UtY)                     # (F, 2) full-data weights
        # Q = V diag(s_over) U'  (F, N); QT keeps the fast elementwise form below.
        QT = (U * s_over[None, :]) @ Vt                        # (N, F)  == Q.T

        for t2 in range(n_times):
            if test_scaling == 'train':
                X2 = (raw_X[:, t2, :] - mu) / sd
            else:
                X2 = X[:, t2, :]
            X2 = np.asarray(X2, dtype=np.float64)

            diag_XQ = np.einsum('if,if->i', X2, QT)            # diag(X2 Q), O(N F)
            pred = X2 @ G + diag_XQ[:, None] * C               # (N, 2)
            pred_angles[:, t1, t2] = np.degrees(
                np.mod(np.arctan2(pred[:, 0], pred[:, 1]), 2 * np.pi)).astype(np.float32)

    return pred_angles


def summarize(pred_angles, angles_deg):
    """
    (N, T, T) predictions -> the two (T, T) per-subject error summaries used
    downstream, matching plot_decoding_ts.py's conventions exactly:

      err_signed_circmean_abs : |circular mean across trials of the SIGNED
          error| -- the statistic plot_decoding_ts.py's timeseries figure uses
          (ported there from megScripts/plotDecodBehav.py). Same-magnitude
          opposite-direction trial errors partially cancel before abs().
      err_unsigned_mean : mean across trials of the per-trial UNSIGNED
          circular distance -- the older, more conservative metric.

    Both are ~90 deg under chance.
    """
    signed = ((pred_angles - angles_deg[:, None, None] + 180.0) % 360.0) - 180.0
    n_times = pred_angles.shape[1]
    signed_circmean_abs = np.empty((n_times, n_times))
    for t1 in range(n_times):
        signed_circmean_abs[t1] = np.abs(
            circmean(signed[:, t1, :], high=180, low=-180, axis=0))
    unsigned_mean = circular_dist(pred_angles, angles_deg[:, None, None]).mean(axis=0)
    return signed_circmean_abs, unsigned_mean


def run_cell(subjID, bands, conditions, rois, voxRes, bids_root,
             win_ms=DEFAULT_WIN_MS, time_stride_ms=DEFAULT_TIME_STRIDE_MS,
             alpha=RIDGE_ALPHA, test_scaling='own', outdir=None, force=False,
             save_trials=False):
    for band in bands:
        for condition in conditions:
            want_phase = condition in ('ampPhase', 'phaseOnly')
            if want_phase and band not in AMP_PHASE_BANDS:
                print(f'SKIP {condition}/{band}: phase unavailable '
                      f'(AMP_PHASE_BANDS={AMP_PHASE_BANDS})', flush=True)
                continue

            for roi in rois:
                out_path = output_path(bids_root, subjID, band, roi, condition,
                                        voxRes, outdir)
                if not force and os.path.exists(out_path):
                    print(f'SKIP (exists): {out_path}', flush=True)
                    continue

                t_start = time.time()
                try:
                    g04 = load_g04_band(subjID, LOCK_TYPE, band, voxRes, bids_root,
                                         want_phase=want_phase, roi=roi)
                    amp     = g04['amp']
                    phase   = g04['phase'] if want_phase else None
                    tv      = g04['time_vector']
                    fsample = float(g04['actualRate'])
                    labels  = g04['target_labels'].astype(int)
                    angles_deg = np.array([float(ANGLE_MAPPING[int(l)]) for l in labels])

                    X = build_features(condition, amp, phase)
                    del amp, phase, g04

                    # ERP removal (redundant under per-timepoint z-scoring --
                    # see module docstring -- but kept for comparability).
                    X = X - X.mean(axis=0, keepdims=True)
                    X = moving_window_mean(X, fsample, win_ms)

                    # Subsample the time grid BEFORE the TGM: it is quadratic in
                    # timepoints, and this also frees the large native-resolution
                    # array immediately (memory matters with 21 concurrent jobs).
                    stride = max(1, int(round(time_stride_ms * 1e-3 * fsample)))
                    eval_idx = np.arange(0, X.shape[1], stride)
                    X = np.ascontiguousarray(X[:, eval_idx, :])
                    tv_eval = tv[eval_idx]

                    raw_X = X if test_scaling == 'train' else None
                    Xz = X if test_scaling == 'train' else zscore_per_timepoint(X)

                    n_trials, n_times, n_feat = Xz.shape
                    print(f'sub-{subjID:02d} | {band} | {condition} | {roi}: '
                          f'N={n_trials} T={n_times} F={n_feat} | fsample={fsample:.0f}Hz | '
                          f'win=+-{win_ms:.0f}ms stride={time_stride_ms:.0f}ms | '
                          f'alpha={alpha} | scaling={test_scaling}', flush=True)

                    pred_angles = ridge_loo_tgm(Xz, angles_deg, alpha=alpha,
                                                 test_scaling=test_scaling, raw_X=raw_X)
                    signed_abs, unsigned = summarize(pred_angles, angles_deg)

                    save_kw = dict(
                        eval_time_vector        = tv_eval.astype(np.float32),
                        err_signed_circmean_abs = signed_abs.astype(np.float32),
                        err_unsigned_mean       = unsigned.astype(np.float32),
                        true_angles             = angles_deg.astype(np.float32),
                        target_labels           = labels.astype(np.int32),
                        chance_deg              = np.array([CHANCE_ERROR_DEG]),
                        subjID    = np.array([subjID]),   band   = np.array([band]),
                        condition = np.array([condition]), roi   = np.array([roi]),
                        voxRes    = np.array([voxRes]),
                        win_ms    = np.array([win_ms]),
                        time_stride_ms = np.array([time_stride_ms]),
                        alpha     = np.array([alpha]),
                        test_scaling = np.array([test_scaling]),
                        fsample   = np.array([fsample]),
                        n_trials  = np.array([n_trials]),
                        n_features = np.array([n_feat]),
                    )
                    if save_trials:
                        save_kw['pred_angles'] = pred_angles.astype(np.float32)
                    np.savez_compressed(out_path, **save_kw)

                    diag = np.diag(signed_abs)
                    print(f'  Saved: {out_path}  ({time.time() - t_start:.1f}s) | '
                          f'best diag err={diag.min():.1f}deg @ '
                          f't={tv_eval[int(np.argmin(diag))]:+.2f}s | '
                          f'best off-diag err={signed_abs.min():.1f}deg', flush=True)
                    del X, Xz, pred_angles
                except (FileNotFoundError, ValueError) as e:
                    print(f'  SKIP sub-{subjID:02d} {band}/{condition}/{roi}: {e}', flush=True)
                except Exception:
                    print(f'  FAILED sub-{subjID:02d} {band}/{condition}/{roi}:', flush=True)
                    traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Circular (sin/cos) temporal generalization matrix via closed-form '
                     'LOO ridge, one subject.')
    parser.add_argument('subjID', type=int)
    parser.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    parser.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    parser.add_argument('--rois', nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--voxRes', default='8mm')
    parser.add_argument('--win_ms', type=float, default=DEFAULT_WIN_MS)
    parser.add_argument('--time_stride_ms', type=float, default=DEFAULT_TIME_STRIDE_MS,
                         help='TGM is quadratic in evaluated timepoints (default 50 ms).')
    parser.add_argument('--alpha', type=float, default=RIDGE_ALPHA)
    parser.add_argument('--test_scaling', default='own', choices=['own', 'train'],
                         help="'own' (default): each timepoint z-scored by its own "
                              "across-trial stats, as elsewhere in this repo. 'train': "
                              "test timepoints standardized with the TRAIN timepoint's "
                              "stats, reproducing svr_tgm.py.")
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--save_trials', action='store_true',
                         help='Also save the (N, T, T) trial-level predictions '
                              '(large -- off by default).')
    args = parser.parse_args()

    bids_root = get_bids_root()
    print(f'circular_tgm_cell | sub-{args.subjID:02d} | bands={args.bands} | '
          f'conditions={args.conditions} | rois={args.rois} | {args.voxRes} | '
          f'win_ms={args.win_ms} | stride={args.time_stride_ms} | alpha={args.alpha} | '
          f'scaling={args.test_scaling} | force={args.force}', flush=True)

    t0 = time.time()
    run_cell(args.subjID, list(args.bands), list(args.conditions), list(args.rois),
             args.voxRes, bids_root, win_ms=args.win_ms,
             time_stride_ms=args.time_stride_ms, alpha=args.alpha,
             test_scaling=args.test_scaling, outdir=args.outdir, force=args.force,
             save_trials=args.save_trials)
    print(f'Done | sub-{args.subjID:02d} | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
