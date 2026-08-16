#!/usr/bin/env python3
"""
decoding_ts_cell.py

Linear decoding-over-time for one subject, across one or more bands.

For each (band x condition x ROI):
  - Loads per-ROI .npz cache via load_g04_band (no whole-grid HDF5 touch).
  - Builds features via features.build_features (shared with run_glue_cell.py):
      ampOnly  -> amp                                 shape: (N, T, S)
      ampPhase -> [amp*cos(phase), amp*sin(phase)]    shape: (N, T, 2S)
    NOT [amp, cos(phase), sin(phase)] (3S) -- amplitude is fully recoverable
    from (amp*cos, amp*sin) as sqrt(real**2+imag**2), so including it
    separately would double-count it and inflate ampPhase's feature count
    3x vs ampOnly's 1x, confounding "phase helps" with "more features
    helps" under LOO (see features.py's docstring).
  - Subtracts the ERP (trial-averaged response, per band/roi/condition cell,
    computed at native time resolution BEFORE windowing) from every single
    trial (default ON, --no_erp_removal to disable). This is the grand
    average across ALL trials in the cell (not per target location -- that
    would remove location-specific structure, defeating the point), meant
    to strip the common stimulus-locked response shared regardless of which
    location was shown, so the decoder sees trial-to-trial deviations from
    that common response rather than being dominated by it.
  - Applies ±win_ms ms temporal averaging window.
  - Analytical closed-form Ridge LOO decoding (sin + cos targets, arctan2
    for angle) -- a linear regularized regression, same family as a linear
    SVR, but with an exact O(1)-per-shuffle LOO shortcut (see
    ridge_loocv_timeseries), which is what makes n_shuffle=100 affordable.
  - Shuffle baseline: n_shuffle label-permuted runs, reusing the same
    per-timepoint A_inv/hat-matrix; per-permutation circular error is
    computed BEFORE averaging (averaging sin/cos across permutations first
    would collapse toward a near-arbitrary angle instead of the expected
    ~90 deg chance level). Two shuffle statistics are saved per permutation:
    the original per-trial unsigned circular_dist (shuffle_errors, used by
    plot_decoding_ts.py's quartile figure) and a signed-circmean-across-trials
    version (shuffle_signed_circmean, the method-matched null for
    plot_decoding_ts.py's timeseries figure, which ports its real-error
    statistic from megScripts/plotDecodBehav.py's convention).
  - Saves one .npz per (subject, band, roi, condition).

Complexity: O(bands x T x N^2 x F + bands x T x n_shuffle x N x F)
  T=270, N=150, F=50, n_shuffle=100 => ~seconds per cell
  vs RBF TGM: O(T^2 x N^2 x S) => hours per cell  (~10000x faster)

Stim-locked only (by design decision: stim epoch 0-0.2s, delay 0.2-1.7s).
"""

import os

os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.stats import circmean

from align import g04_orig_row_index
from constants import AMP_ONLY_BANDS, AMP_PHASE_BANDS, ANGLE_MAPPING, ROI_NAMES, get_bids_root
from features import build_features
from io_g03 import load_g03_unfiltered
from io_g04 import load_g04_band

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_WIN_MS    = 50.0    # one-sided: ±50 ms = 100 ms total window
DEFAULT_N_SHUFFLE = 100     # label permutations for shuffle baseline (cheap: see complexity note above)
RIDGE_ALPHA       = 1.0     # Ridge regularization (features are z-scored per timepoint)

# ── Utility functions ─────────────────────────────────────────────────────────

def circular_dist(pred_deg, true_deg):
    """
    Circular angular distance in degrees, result always in [0, 180].
    Supports arbitrary broadcastable shapes.
    """
    diff = np.abs(pred_deg - true_deg) % 360.0
    return np.minimum(diff, 360.0 - diff)


def moving_window_mean(X, fsample, win_ms):
    """
    X: (n_trials, n_times, n_feat)
    At each timepoint t, returns mean over [t - half_win, t + half_win]
    samples (clamped to array bounds).  Returns copy unchanged if win_ms == 0.
    """
    if win_ms <= 0:
        return X.copy()
    half = max(1, round(win_ms * 1e-3 * fsample))
    n_trials, n_times, n_feat = X.shape
    X_out = np.empty_like(X)
    for t in range(n_times):
        t0 = max(0, t - half)
        t1 = min(n_times, t + half + 1)
        X_out[:, t, :] = X[:, t0:t1, :].mean(axis=1)
    return X_out


def angles_deg_from_labels(target_labels):
    """Integer trial labels -> degrees via ANGLE_MAPPING."""
    return np.array([float(ANGLE_MAPPING[int(l)]) for l in target_labels])


# ── Core decoder ──────────────────────────────────────────────────────────────

def ridge_loocv_timeseries(X, true_angles_deg, alpha=RIDGE_ALPHA,
                            n_shuffle=DEFAULT_N_SHUFFLE, seed=0):
    """
    Analytical LOO Ridge decoding of sin/cos angle targets, at every
    timepoint independently (diagonal decoding, not a T x T grid).

    X              : (N, T, F) feature array (windowed, NOT yet z-scored)
    true_angles_deg: (N,) target angles in degrees
    alpha          : Ridge regularisation parameter
    n_shuffle      : label-permuted shuffle runs

    Efficiency trick: A_inv = (X_z^T X_z + alpha*I)^{-1} and the hat-matrix
    diagonal H_diag are computed ONCE per timepoint and reused for all
    n_shuffle permutations. Each shuffle then only needs two O(N*F) matmuls
    -- this is what makes n_shuffle=100 affordable (no equivalent shortcut
    exists for an iterative solver like LinearSVR).

    Shuffle baseline: for each permutation k, LOO-decode the PERMUTED
    labels, compute that permutation's circular error against the REAL
    labels, THEN average errors across permutations -- this is the null
    distribution of errors expected when brain activity has no information
    about target location. (Averaging sin/cos predictions across
    permutations BEFORE computing one angle/error is wrong: it collapses
    toward a near-zero resultant vector and arctan2 returns a near-arbitrary
    angle, not the mean null error.)

    Returns
    -------
    pred_angles    : (N, T) float32  predicted angles in degrees [0, 360)
    errors         : (N, T) float32  circular error in degrees  [0, 180]
    shuffle_errors : (N, T) float32  mean circular error over n_shuffle
                                      permutations
    shuffle_signed_circmean : (n_shuffle, T) float32  per-permutation signed
                                      circular mean of (perm_pred - true)
                                      across trials, in [-180, 180]. This is
                                      the method-matched null for
                                      plot_decoding_ts.py's
                                      abs(circmean(signed_error)) real-error
                                      statistic (ported from
                                      megScripts/plotDecodBehav.py) --
                                      shuffle_errors above is NOT
                                      method-matched to that statistic (it's
                                      unsigned circular_dist averaged over
                                      permutations at the per-trial level,
                                      the null for the OLD per-trial
                                      unsigned-distance real-error metric,
                                      still used by the quartile figure).
    """
    n_trials, n_times, n_feat = X.shape

    angles_rad = np.radians(true_angles_deg)
    sin_y = np.sin(angles_rad)   # (N,)
    cos_y = np.cos(angles_rad)   # (N,)

    pred_sin = np.zeros((n_trials, n_times), dtype=np.float32)
    pred_cos = np.zeros((n_trials, n_times), dtype=np.float32)
    shuffle_errors = np.zeros((n_trials, n_times), dtype=np.float32)
    shuffle_signed_circmean = np.zeros((n_shuffle, n_times), dtype=np.float32)

    rng   = np.random.default_rng(seed)
    perms = [rng.permutation(n_trials) for _ in range(n_shuffle)]

    I_F = np.eye(n_feat)

    for t in range(n_times):
        # ── Z-score features at this timepoint ──────────────────────────────
        X_t = X[:, t, :]                          # (N, F)
        mu  = X_t.mean(axis=0)
        sd  = X_t.std(axis=0)
        sd[sd < 1e-10] = 1.0
        X_z = (X_t - mu) / sd                    # (N, F)

        # ── Precompute A_inv and intermediates ───────────────────────────────
        # A = X_z^T X_z + alpha*I  (F, F)
        A     = X_z.T @ X_z
        A_inv = np.linalg.solve(A + alpha * I_F, I_F)   # (F, F)

        # P = A_inv @ X_z^T  (F, N) -- projection matrix, reused for all shuffles
        P = A_inv @ X_z.T                         # (F, N)

        # Hat matrix diagonal: H_ii = x_i @ A_inv @ x_i
        #   = sum over j of (X_z @ A_inv) * X_z  (row-wise dot product)
        Q      = X_z @ A_inv                      # (N, F)
        H_diag = np.sum(Q * X_z, axis=1)          # (N,)
        H_diag = np.clip(H_diag, 0.0, 1.0 - 1e-7)
        lev    = 1.0 - H_diag                     # (N,) denominator for LOO

        def _loo(y):
            """LOO predictions given response vector y (N,).
            Returns ŷ_{-i} = y_i - (y_i - ŷ_i) / (1 - H_ii)
            """
            beta = P @ y              # (F,) Ridge coefficients
            yhat = X_z @ beta         # (N,) fitted values
            e    = y - yhat           # (N,) residuals
            return y - e / lev        # (N,) LOO predictions

        pred_sin[:, t] = _loo(sin_y).astype(np.float32)
        pred_cos[:, t] = _loo(cos_y).astype(np.float32)

        # ── Shuffle baseline: per-permutation error, then average ──────────
        # Reuses A_inv / P / H_diag (they depend only on X_z, not on labels),
        # so each shuffle is just two cheap O(N*F) matmuls via _loo().
        if n_shuffle > 0:
            perm_errors = np.empty((n_shuffle, n_trials), dtype=np.float64)
            for k, perm in enumerate(perms):
                p_sin = _loo(sin_y[perm])
                p_cos = _loo(cos_y[perm])
                perm_angles = np.degrees(np.mod(np.arctan2(p_sin, p_cos), 2 * np.pi))
                perm_errors[k] = circular_dist(perm_angles, true_angles_deg)
                # Method-matched null for the signed-circmean real-error stat
                # (see docstring) -- signed error per trial, THEN circular
                # mean across trials for this one permutation/timepoint.
                signed_err_k = ((perm_angles - true_angles_deg + 180) % 360) - 180
                shuffle_signed_circmean[k, t] = circmean(signed_err_k, high=180, low=-180)
            shuffle_errors[:, t] = perm_errors.mean(axis=0).astype(np.float32)

    # ── Convert real predictions to angles / errors ──────────────────────────
    pred_angles = np.degrees(
        np.mod(np.arctan2(pred_sin.astype(np.float64),
                           pred_cos.astype(np.float64)), 2 * np.pi)
    ).astype(np.float32)
    errors = circular_dist(pred_angles, true_angles_deg[:, None]).astype(np.float32)

    return pred_angles, errors, shuffle_errors, shuffle_signed_circmean


# ── Output path ───────────────────────────────────────────────────────────────

def output_path(bids_root, subjID, band, roi, condition, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base    = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'decodingTS')
    try:
        os.makedirs(base, exist_ok=True)
    except OSError as e:
        fallback = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'derivatives', subName,
            'sourceRecon', 'decodingTS'))
        print(f'  WARNING: could not create {base!r} ({e}) -- falling back to '
              f'{fallback!r}. plot_decoding_ts.py must be pointed at this same '
              f'--outdir or it will not find these files.', flush=True)
        base = fallback
        os.makedirs(base, exist_ok=True)
    return os.path.join(
        base,
        f'{subName}_task-mgs_decodingTS_{condition}_{band}_{roi}_{voxRes}.npz')


# ── Cell computation ──────────────────────────────────────────────────────────

def _get_trial_idx(subjID, lockType, voxRes, bids_root, rois, n_trials):
    """
    Original sourcedataCombined row indices for this subject/lockType's G04
    rows, recomputed from G03's trialinfo (see align.g04_orig_row_index).
    Every band/ROI G04 file for this subject/lockType shares the same row
    order (io_g04.py's documented target-grouping convention), so this only
    needs to be computed once per subject, from any one ROI's small G03
    cache (not the whole-grid file).

    Returns (n_trials,) int64, or np.arange(n_trials) with a warning if G03
    metadata is unavailable or the row count doesn't match (e.g. mismatched
    cache generation) -- trial_idx degrades gracefully rather than blocking
    the whole run.
    """
    try:
        g03_meta = load_g03_unfiltered(subjID, lockType, voxRes, bids_root, roi=rois[0])
        trial_idx = g04_orig_row_index(g03_meta['trialinfo_col2'])
    except (FileNotFoundError, OSError) as e:
        print(f'  WARNING: sub-{subjID:02d}: could not compute trial_idx from G03 '
              f'({e}) -- falling back to np.arange(n_trials).', flush=True)
        return np.arange(n_trials, dtype=np.int64)
    if trial_idx.shape[0] != n_trials:
        print(f'  WARNING: sub-{subjID:02d}: trial_idx length {trial_idx.shape[0]} != '
              f'n_trials {n_trials} -- falling back to np.arange(n_trials).', flush=True)
        return np.arange(n_trials, dtype=np.int64)
    return trial_idx


def run_cell(subjID, bands, voxRes, bids_root, rois, conditions,
             win_ms, n_shuffle, alpha, remove_erp=True, outdir=None, force=False):

    lockType = 'stim'   # stim-locked only
    trial_idx_cache = {}   # lazily computed once per band (n_trials can differ across bands)

    for band in bands:
        for condition in conditions:
            want_phase = condition in ('ampPhase', 'phaseOnly')

            # Guard: phase data only available for theta/alpha/beta
            if want_phase and band not in AMP_PHASE_BANDS:
                print(f'SKIP {condition}/{band}: phase not available '
                      f'(AMP_PHASE_BANDS={AMP_PHASE_BANDS})')
                continue

            for roi in rois:
                out_path = output_path(bids_root, subjID, band, roi,
                                        condition, voxRes, outdir)
                if os.path.exists(out_path) and not force:
                    print(f'SKIP (exists): {out_path}')
                    continue

                print(f'\nsub-{subjID:02d} | {band} | {condition} | {roi}', flush=True)

                # Isolate this one (band, condition, roi) cell -- any failure
                # here (missing/corrupt cache, bad angle label, degenerate
                # covariance, ...) should skip just this cell, not crash the
                # rest of this subject's run.
                try:
                    g04 = load_g04_band(subjID, lockType, band, voxRes, bids_root,
                                         want_phase=want_phase, roi=roi)

                    amp           = g04['amp']            # (N, T, S)
                    phase         = g04['phase'] if want_phase else None   # (N, T, S) radians
                    tv            = g04['time_vector']    # (T,)
                    fsample       = float(g04['actualRate'])
                    target_labels = g04['target_labels']  # (N,)
                    n_trials      = amp.shape[0]

                    true_angles_deg = angles_deg_from_labels(target_labels)

                    if band not in trial_idx_cache:
                        trial_idx_cache[band] = _get_trial_idx(
                            subjID, lockType, voxRes, bids_root, rois, n_trials)
                    trial_idx = trial_idx_cache[band]
                    if trial_idx.shape[0] != n_trials:
                        # This roi/condition's trial count differs from the one
                        # trial_idx was computed from for this band -- fall back
                        # locally rather than misaligning rows.
                        trial_idx = np.arange(n_trials, dtype=np.int64)

                    # Feature matrix: (N, T, F) -- shared with run_glue_cell.py,
                    # so ampPhase is [amp*cos(phase), amp*sin(phase)] (2S), not
                    # [amp, cos(phase), sin(phase)] (3S) -- see module docstring.
                    X = build_features(condition, amp, phase)

                    # ERP removal: subtract the grand trial-average (native
                    # time resolution, BEFORE windowing) from every trial --
                    # see module docstring.
                    if remove_erp:
                        erp = X.mean(axis=0, keepdims=True)   # (1, T, F)
                        X = X - erp

                    # Temporal averaging window
                    X_win = moving_window_mean(X, fsample, win_ms)
                    del X

                    n_times, n_feat = X_win.shape[1], X_win.shape[2]
                    print(f'  shape=({n_trials},{n_times},{n_feat}) | '
                          f'fsample={fsample:.0f}Hz | win=±{win_ms:.0f}ms | '
                          f'n_shuffle={n_shuffle} | alpha={alpha} | '
                          f'remove_erp={remove_erp}', flush=True)

                    pred_angles, errors, shuffle_errors, shuffle_signed_circmean = \
                        ridge_loocv_timeseries(X_win, true_angles_deg,
                                                alpha=alpha, n_shuffle=n_shuffle)

                    np.savez_compressed(
                        out_path,
                        pred_angles    = pred_angles,                        # (N, T) deg
                        errors         = errors,                             # (N, T) deg [0,180]
                        shuffle_errors = shuffle_errors,                     # (N, T) deg
                        shuffle_signed_circmean = shuffle_signed_circmean,   # (n_shuffle, T) deg [-180,180]
                        true_angles    = true_angles_deg.astype(np.float32), # (N,) deg
                        target_labels  = target_labels.astype(np.int32),     # (N,)
                        trial_idx      = trial_idx.astype(np.int64),         # (N,) orig row idx
                        time_vector    = tv.astype(np.float32),              # (T,)
                        subjID         = np.array([subjID]),
                        band           = np.array([band]),
                        condition      = np.array([condition]),
                        roi            = np.array([roi]),
                        win_ms         = np.array([win_ms]),
                        n_shuffle      = np.array([n_shuffle]),
                        alpha          = np.array([alpha]),
                        remove_erp     = np.array([remove_erp]),
                        fsample        = np.array([fsample]),
                    )
                    print(f'  Saved: {out_path}')
                except (FileNotFoundError, ValueError) as e:
                    print(f'  SKIP: {e}')
                except Exception:
                    print(f'  FAILED sub-{subjID:02d} {band}/{condition}/{roi} '
                          f'(skipping this cell only):', flush=True)
                    traceback.print_exc()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Linear decoding-over-time cell: one subject, one or more bands.')
    parser.add_argument('subjID', type=int)
    parser.add_argument('--bands',       nargs='+', default=list(AMP_ONLY_BANDS),
                        help=f'Frequency bands (default: all of {AMP_ONLY_BANDS}). '
                             f'ampPhase is silently skipped for bands outside '
                             f'AMP_PHASE_BANDS={AMP_PHASE_BANDS}.')
    parser.add_argument('--voxRes',      default='8mm')
    parser.add_argument('--rois',        nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions',  nargs='+', default=['ampOnly', 'ampPhase'],
                        help='Feature conditions (ampOnly / ampPhase).')
    parser.add_argument('--win_ms',      type=float, default=DEFAULT_WIN_MS,
                        help=f'One-sided temporal window half-width in ms '
                             f'(default {DEFAULT_WIN_MS} ms -> '
                             f'+-{DEFAULT_WIN_MS:.0f} ms = '
                             f'{2*DEFAULT_WIN_MS:.0f} ms total).')
    parser.add_argument('--n_shuffle',   type=int, default=DEFAULT_N_SHUFFLE,
                        help=f'Label-permuted shuffle runs (default {DEFAULT_N_SHUFFLE}, '
                             f'cheap thanks to the closed-form Ridge LOO shortcut).')
    parser.add_argument('--alpha',       type=float, default=RIDGE_ALPHA,
                        help=f'Ridge regularisation alpha (default {RIDGE_ALPHA}).')
    parser.add_argument('--no_erp_removal', action='store_false', dest='remove_erp',
                        help='Skip ERP (grand trial-average) subtraction before decoding '
                             '(default: ERP IS subtracted -- see module docstring).')
    parser.add_argument('--outdir',      default=None)
    parser.add_argument('--force',       action='store_true',
                        help='Overwrite existing .npz outputs instead of skipping them '
                             '(default: skip cells whose output file already exists).')
    args = parser.parse_args()

    bids_root = get_bids_root()
    print(f'decoding_ts_cell | sub-{args.subjID:02d} | bands={args.bands} | '
          f'{args.voxRes} | conditions={args.conditions} | rois={args.rois} | '
          f'win_ms={args.win_ms} | n_shuffle={args.n_shuffle} | alpha={args.alpha} | '
          f'remove_erp={args.remove_erp} | force={args.force}',
          flush=True)

    run_cell(args.subjID, list(args.bands), args.voxRes, bids_root,
             list(args.rois), list(args.conditions),
             args.win_ms, args.n_shuffle, args.alpha,
             remove_erp=args.remove_erp, outdir=args.outdir, force=args.force)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
