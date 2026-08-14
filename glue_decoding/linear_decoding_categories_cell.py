#!/usr/bin/env python3
"""
linear_decoding_categories_cell.py

Standard linear one-vs-rest classification, at several category
granularities, over the FULL stim-locked timeline -- same moving-window
convention as decoding_ts_cell.py (imports moving_window_mean directly), so
this is time-aligned/comparable to both the decoding-error timeseries and
representational_distance_ts_cell.py's distance-gap timeseries.

Motivation (collaborator's suggestion, see chat history): the decoders
actually used elsewhere in this repo (svr_tgm.py, decoding_ts_cell.py) do
continuous circular regression (sin/cos of angle), never a discrete
classification -- so it's genuinely unclear whether responses to nearby
target locations are linearly separable AS DISCRETE CATEGORIES, which is
what GLUE's manifold_capacity.py ONE_VERSUS_REST assumes. This runs a
standard linear-separability tool at four granularities
(constants.CATEGORY_SCHEMES):
    2  categories : left vs right hemifield
    4  categories : quadrants (excludes the 2 axis locations)
    6  categories : quadrants + the 2 axis locations as singletons
    10 categories : every raw location (current manifold_capacity.py scope)
so we can see at what granularity linear separability actually appears,
rather than assuming 10-way is the right unit of analysis.

CLASSIFIER: closed-form ridge one-vs-rest, NOT SVM (an earlier version used
sklearn LinearSVC with exhaustive per-fold refits). Both are linear
classifiers testing the identical question -- squared-error vs. hinge loss
is the only difference -- but ridge is also what svr_tgm.py/
decoding_ts_cell.py already use, so this is more methodologically
consistent with the rest of the repo AND dramatically cheaper:
  - Dummy-code each category as a +-1 column of Y (n_trials, n_categories).
  - The leave-one-out prediction for every trial falls out of a SINGLE fit
    algebraically via the hat matrix H = X(X^T X + alpha*I)^-1 X^T:
    yhat_i_loo = (yhat_i - h_ii*y_i) / (1 - h_ii) -- no refitting per fold.
    This is exactly decoding_ts_cell.py's "O(1)-per-shuffle LOO shortcut"
    generalized from 2 output columns (sin, cos) to n_categories columns
    (dummy-coded classes) and from regression to argmax classification.
  - H depends only on X, not on the labels -- so a label-permutation null
    reuses the SAME factorization (just a couple matmuls per shuffle),
    making n_shuffle cheap too, unlike SVM where every shuffle needed a
    full refit sweep.

ALPHA IS SELECTED ADAPTIVELY PER TIMEPOINT, NOT FIXED (important -- see
chat history for the full diagnosis): a fixed alpha=1.0 (decoding_ts_cell.py's
value, calibrated for its own 2-column regression) is catastrophically
under-regularized here. We're in the F > N regime essentially always (up to
597 features, as few as 20-150 trials after balancing) -- with weak
regularization, leverage h_ii approaches 1 for every point, the exact LOO
formula's (1-h_ii) denominator blows up, and the result is numerically
degenerate "100% accuracy" on PURE NOISE, reproducibly (verified: same
result across seeds, and even with independently reshuffled labels) -- not
a sign of real decodability. Fix: fit via a thin SVD of X (X=U S V^T, cheap
since F>N), which lets every candidate alpha in ALPHA_GRID reuse the same
decomposition (leverage h(alpha)=(U**2)@(S**2/(S**2+alpha)), predictions
via U/S only) -- so scanning ~30 alpha values costs barely more than one
fit, and is selected via each candidate's own LOO MSE (generalized
cross-validation on the regression targets Y, NOT on classification
accuracy itself, to avoid circularity). Verified: pure noise correctly
selects heavy regularization and returns to chance accuracy; injected
signal selects light regularization and recovers it. Alpha is selected
ONCE per timepoint using the REAL labels, then reused for that timepoint's
accuracy AND every shuffle (cheap, and conservative -- reusing a
real-data-tuned alpha on shuffled labels can only inflate the null
slightly, never deflate it, so if anything this understates significance
rather than overstating it).

Every scheme is balanced via constants.balance_categories --
points_per_category defaults to None (auto: each subject's own smallest
category count for that scheme, NOT a fixed value shared across schemes --
matches representational_distance_ts_cell.py's fix; pass an int to force a
fixed cap instead).

Output: one .npz per (subject, band, roi, condition, scheme) at
derivatives/sub-XX/sourceRecon/linDecodeCat/sub-XX_task-mgs_linDecodeCat_
{condition}_{band}_{roi}_{voxRes}_scheme{scheme}.npz

Does NOT require the `glue` package -- pure numpy, runs in this repo's
normal Python environment.

Usage:
    python linear_decoding_categories_cell.py <subjID>
        [--bands theta alpha beta lowgamma highgamma]
        [--voxRes 8mm] [--rois visual parietal frontal]
        [--conditions ampOnly] [--schemes 2 4 6 10]
        [--points_per_category N] [--win_ms 50] [--time_stride_ms 50]
        [--n_shuffle 100]
        [--seed 0] [--outdir <path>] [--force]
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

from constants import (AMP_ONLY_BANDS, AMP_PHASE_BANDS, ROI_NAMES, get_bids_root,
                        CATEGORY_SCHEMES, category_labels_for_scheme, balance_categories)
from decoding_ts_cell import DEFAULT_WIN_MS, moving_window_mean
from features import build_features
from io_g04 import load_g04_band

DEFAULT_TIME_STRIDE_MS = 50.0
# Cheap now (hat-matrix reuse -- see module docstring), so default matches
# decoding_ts_cell.py's DEFAULT_N_SHUFFLE instead of the SVM-era 0.
DEFAULT_N_SHUFFLE = 100
# None = auto: balance each (subject, scheme) to that subject's own smallest
# category count for that scheme -- see module docstring / constants.balance_categories.
DEFAULT_POINTS_PER_CATEGORY = None
# Candidate ridge regularization strengths, log-spaced -- see module
# docstring for why a single fixed alpha is unsafe here. 30 points from
# essentially unregularized (1) to heavily shrunk (1e6) gave noise accuracy
# correctly back near chance in validation (chance-level runs pinned the
# upper end of this range, see chat history), while still recovering strong
# injected signal at a much lighter alpha -- reused across every scheme.
ALPHA_GRID = np.logspace(0, 6, 30)

# ── Output path ─────────────────────────────────────────────────────────────

def output_path(bids_root, subjID, band, roi, condition, voxRes, scheme, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'linDecodeCat')
    os.makedirs(base, exist_ok=True)
    fname = f'{subName}_task-mgs_linDecodeCat_{condition}_{band}_{roi}_{voxRes}_scheme{scheme}.npz'
    return os.path.join(base, fname)


# ── Core closed-form ridge one-vs-rest LOO, adaptive alpha via SVD ──────────

MAX_LEVERAGE = 0.9   # see _ridge_ovr_svd_fit docstring


def _ridge_ovr_svd_fit(X, Y, alphas=ALPHA_GRID, max_leverage=MAX_LEVERAGE):
    """
    X: (n_trials, n_feat) z-scored features (n_feat > n_trials, our regime).
    Y: (n_trials, n_categories) +-1 dummy-coded labels.

    One thin SVD of X serves every candidate alpha cheaply (leverage and
    predictions are both simple functions of U, S -- see module docstring).
    Selects alpha via each SAFE candidate's own LOO mean squared error on Y
    (generalized cross-validation on the regression targets, NOT
    classification accuracy, to avoid circularity).

    "Safe" = max leverage <= max_leverage, REQUIRED before the LOO-MSE
    comparison even runs -- not just a tie-breaker. The LOO-MSE objective
    itself becomes unreliable, not just the predictions, once leverage
    approaches 1: at low alpha the near-interpolating fit's in-sample
    residual (Y-Yhat) shrinks just as fast as the (1-h) denominator it's
    divided by, so the "LOO MSE" can look spuriously tiny in exactly the
    regime where the LOO formula is numerically degenerate. This escaped
    detection for large n_categories (K=10) in validation -- MSE averaged
    over 10 columns correctly rose sharply at low alpha -- but NOT for
    small K (K=2 reproducibly selected the MOST degenerate alpha in the
    grid, giving 100% "accuracy" on pure noise every time, because with
    only 2 -- anti-correlated, effectively 1 independent -- output columns
    the same degenerate-MSE artifact isn't reliably outvoted by safer
    alphas). Excluding unsafe alphas from consideration entirely, rather
    than trusting GCV-MSE to always find its own way out, fixes this for
    every n_categories. See chat history for the full diagnosis.

    Returns (U, d, h) for the SELECTED alpha -- U:(n_trials,n_trials),
    d:(n_trials,) = S^2/(S^2+alpha), h:(n_trials,) leverage -- everything
    needed to get LOO predictions for this OR any relabeled Y via
    _ridge_ovr_loo_from_svd, without recomputing the SVD or re-searching alpha.
    """
    U, S, _ = np.linalg.svd(X, full_matrices=False)   # U:(N,N) S:(N,)
    S2 = S ** 2
    UtY = U.T @ Y   # (N, n_categories)

    best_mse, best_d, best_h = np.inf, None, None
    for alpha in alphas:
        d = S2 / (S2 + alpha)
        h = np.clip((U ** 2) @ d, 0.0, 1.0 - 1e-8)
        if h.max() > max_leverage:
            continue   # numerically unsafe regime -- excluded, not just deprioritized
        Yhat = U @ (d[:, None] * UtY)
        mse = np.mean(((Y - Yhat) / (1.0 - h)[:, None]) ** 2)
        if mse < best_mse:
            best_mse, best_d, best_h = mse, d, h

    if best_d is None:
        # No candidate alpha was safe (pathologically small N) -- fall back
        # to the grid's largest alpha (maximal shrinkage, always safe since
        # d->0 uniformly as alpha->inf regardless of X).
        alpha = alphas[-1]
        best_d = S2 / (S2 + alpha)
        best_h = np.clip((U ** 2) @ best_d, 0.0, 1.0 - 1e-8)

    return U, best_d, best_h


def _ridge_ovr_loo_from_svd(U, d, h, Y):
    """LOO predictions for (possibly label-permuted) Y, reusing a
    precomputed (U, d, h) from _ridge_ovr_svd_fit -- no refitting, no
    re-searching alpha. This is what makes shuffle nulls cheap."""
    Yhat = U @ (d[:, None] * (U.T @ Y))
    return (Yhat - h[:, None] * Y) / (1.0 - h[:, None])


def _accuracy_from_loo(Yhat_loo, class_index):
    pred = Yhat_loo.argmax(axis=1)
    return float((pred == class_index).mean())


# ── Core computation ─────────────────────────────────────────────────────────

def ridge_ovr_timeseries(X_win, labels, tv, fsample, time_stride_ms=DEFAULT_TIME_STRIDE_MS,
                          n_shuffle=DEFAULT_N_SHUFFLE, seed=0):
    """
    X_win: (n_trials, n_times, n_feat) already moving-window-averaged.
    labels: (n_trials,) category labels (any hashable type, e.g. strings
    from category_labels_for_scheme).

    Returns dict: eval_time_vector, accuracy, shuffle_acc_mean, shuffle_acc_std,
    p_value (all length n_eval, shuffle_*/p_value all-NaN if n_shuffle == 0),
    chance_level (scalar), n_categories (scalar).
    """
    n_trials, n_times, n_feat = X_win.shape
    categories = np.unique(labels)
    n_categories = categories.shape[0]
    chance_level = 1.0 / n_categories
    class_index = np.searchsorted(categories, labels)          # (n_trials,) int in [0, n_categories)

    Y = -np.ones((n_trials, n_categories), dtype=np.float64)
    Y[np.arange(n_trials), class_index] = 1.0

    stride = max(1, round(time_stride_ms * 1e-3 * fsample))
    eval_idx = np.arange(0, n_times, stride)
    n_eval = eval_idx.shape[0]

    accuracy         = np.zeros(n_eval, dtype=np.float64)
    shuffle_acc_mean = np.full(n_eval, np.nan, dtype=np.float64)
    shuffle_acc_std  = np.full(n_eval, np.nan, dtype=np.float64)
    p_value          = np.full(n_eval, np.nan, dtype=np.float64)

    rng = np.random.default_rng(seed)

    for i, t in enumerate(eval_idx):
        X_t = X_win[:, t, :]
        mu = X_t.mean(axis=0, keepdims=True)
        sd = X_t.std(axis=0, keepdims=True)
        sd[sd < 1e-10] = 1.0
        X_z = (X_t - mu) / sd

        U, d, h = _ridge_ovr_svd_fit(X_z, Y)
        Yhat_loo = _ridge_ovr_loo_from_svd(U, d, h, Y)
        accuracy[i] = _accuracy_from_loo(Yhat_loo, class_index)

        if n_shuffle > 0:
            shuf_accs = np.empty(n_shuffle, dtype=np.float64)
            for k in range(n_shuffle):
                perm = rng.permutation(n_trials)
                # Row i of Y[perm] is the ORIGINAL Y[perm[i]] -- i.e. trial i is
                # now paired with the label trial perm[i] actually had, so the
                # correct comparison target for row i is class_index[perm[i]] =
                # class_index[perm][i], NOT the unpermuted class_index.
                Yhat_loo_perm = _ridge_ovr_loo_from_svd(U, d, h, Y[perm])
                shuf_accs[k] = _accuracy_from_loo(Yhat_loo_perm, class_index[perm])
            shuffle_acc_mean[i] = shuf_accs.mean()
            shuffle_acc_std[i]  = shuf_accs.std()
            p_value[i] = (np.sum(shuf_accs >= accuracy[i]) + 1) / (n_shuffle + 1)

    return dict(eval_time_vector=tv[eval_idx], accuracy=accuracy,
                shuffle_acc_mean=shuffle_acc_mean, shuffle_acc_std=shuffle_acc_std,
                p_value=p_value, chance_level=chance_level, n_categories=n_categories)


# ── Per-subject cell runner ─────────────────────────────────────────────────

def run_cell(subjID, bands, voxRes, bids_root, rois, conditions, schemes,
             points_per_category, win_ms, time_stride_ms, n_shuffle,
             seed, outdir=None, force=False):
    lockType = 'stim'   # stim-locked only, matches decoding_ts_cell.py

    for band in bands:
        for condition in conditions:
            want_phase = (condition == 'ampPhase')
            if want_phase and band not in AMP_PHASE_BANDS:
                print(f'SKIP {condition}/{band}: phase not available '
                      f'(AMP_PHASE_BANDS={AMP_PHASE_BANDS})')
                continue

            for roi in rois:
                out_paths = {scheme: output_path(bids_root, subjID, band, roi, condition,
                                                  voxRes, scheme, outdir) for scheme in schemes}
                pending_schemes = [s for s in schemes if force or not os.path.exists(out_paths[s])]
                for s in schemes:
                    if s not in pending_schemes:
                        print(f'SKIP (exists): {out_paths[s]}')
                if not pending_schemes:
                    continue

                print(f'\nsub-{subjID:02d} | {band} | {condition} | {roi} | '
                      f'schemes={pending_schemes}', flush=True)
                try:
                    g04 = load_g04_band(subjID, lockType, band, voxRes, bids_root,
                                         want_phase=want_phase, roi=roi)
                    amp     = g04['amp']
                    phase   = g04['phase'] if want_phase else None
                    tv      = g04['time_vector']
                    fsample = float(g04['actualRate'])
                    raw_target_labels = g04['target_labels'].astype(np.int64)

                    X = build_features(condition, amp, phase)
                    X_win = moving_window_mean(X, fsample, win_ms)
                    del X
                except (FileNotFoundError, ValueError) as e:
                    print(f'  SKIP: {e}')
                    continue
                except Exception:
                    print(f'  FAILED sub-{subjID:02d} {band}/{condition}/{roi} loading '
                          f'(skipping this cell entirely):', flush=True)
                    traceback.print_exc()
                    continue

                for scheme in pending_schemes:
                    out_path = out_paths[scheme]
                    t_start = time.time()
                    try:
                        group_labels, keep_mask = category_labels_for_scheme(
                            raw_target_labels, scheme)

                        if points_per_category is None:
                            # Auto: this subject's own smallest category count
                            # for this scheme -- see module docstring.
                            _, counts = np.unique(group_labels, return_counts=True)
                            ppc_used = int(counts.min())
                        else:
                            ppc_used = points_per_category
                        balance_mask = balance_categories(group_labels, ppc_used, seed=seed)

                        X_win_scheme  = X_win[keep_mask][balance_mask]
                        labels_scheme = group_labels[balance_mask]

                        n_trials, n_times, n_feat = X_win_scheme.shape
                        n_categories = len(CATEGORY_SCHEMES[scheme]['groups'])
                        print(f'  scheme={scheme}: shape=({n_trials},{n_times},{n_feat}) | '
                              f'fsample={fsample:.0f}Hz | win=+-{win_ms:.0f}ms | '
                              f'time_stride={time_stride_ms:.0f}ms | '
                              f'n_shuffle={n_shuffle} | {n_categories} categories x '
                              f'{ppc_used} pts (points_per_category='
                              f'{"auto" if points_per_category is None else points_per_category})',
                              flush=True)

                        result = ridge_ovr_timeseries(
                            X_win_scheme, labels_scheme, tv, fsample,
                            time_stride_ms=time_stride_ms,
                            n_shuffle=n_shuffle, seed=seed)

                        np.savez_compressed(
                            out_path,
                            eval_time_vector = result['eval_time_vector'].astype(np.float32),
                            accuracy         = result['accuracy'].astype(np.float32),
                            shuffle_acc_mean = result['shuffle_acc_mean'].astype(np.float32),
                            shuffle_acc_std  = result['shuffle_acc_std'].astype(np.float32),
                            p_value          = result['p_value'].astype(np.float32),
                            chance_level     = np.array([result['chance_level']]),
                            n_categories     = np.array([result['n_categories']]),
                            group_labels     = labels_scheme,
                            subjID       = np.array([subjID]),
                            band         = np.array([band]),
                            condition    = np.array([condition]),
                            roi          = np.array([roi]),
                            scheme       = np.array([scheme]),
                            points_per_category = np.array([ppc_used]),   # actual value used
                            win_ms       = np.array([win_ms]),
                            time_stride_ms = np.array([time_stride_ms]),
                            n_shuffle    = np.array([n_shuffle]),
                            seed         = np.array([seed]),
                            fsample      = np.array([fsample]),
                        )
                        dt = time.time() - t_start
                        print(f'  Saved: {out_path}  ({dt:.1f}s)')
                    except ValueError as e:
                        print(f'  SKIP scheme={scheme}: {e}')
                    except Exception:
                        print(f'  FAILED sub-{subjID:02d} {band}/{condition}/{roi}/scheme{scheme} '
                              f'(skipping this scheme only):', flush=True)
                        traceback.print_exc()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Linear (closed-form ridge, one-vs-rest, LOO) decoding-over-time at '
                     'several category granularities: one subject, one or more bands.')
    parser.add_argument('subjID', type=int)
    parser.add_argument('--bands',      nargs='+', default=list(AMP_ONLY_BANDS))
    parser.add_argument('--voxRes',     default='8mm')
    parser.add_argument('--rois',       nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions', nargs='+', default=['ampOnly'])
    parser.add_argument('--schemes',    nargs='+', type=int, default=sorted(CATEGORY_SCHEMES),
                         choices=sorted(CATEGORY_SCHEMES))
    parser.add_argument('--points_per_category', type=int, default=DEFAULT_POINTS_PER_CATEGORY,
                         help='Fixed points-per-category, applied identically across every scheme. '
                              'Default: None (auto -- balance each scheme to that subject\'s own '
                              'smallest category count for THAT scheme).')
    parser.add_argument('--win_ms',     type=float, default=DEFAULT_WIN_MS,
                         help=f'One-sided moving-window half-width in ms '
                              f'(default {DEFAULT_WIN_MS} -> {2*DEFAULT_WIN_MS:.0f}ms total window).')
    parser.add_argument('--time_stride_ms', type=float, default=DEFAULT_TIME_STRIDE_MS,
                         help=f'Evaluate every this-many ms instead of every native sample '
                              f'(default {DEFAULT_TIME_STRIDE_MS}ms).')
    parser.add_argument('--n_shuffle', type=int, default=DEFAULT_N_SHUFFLE,
                         help=f'Label-permutation null repeats (default {DEFAULT_N_SHUFFLE} -- '
                              f'cheap here since the hat matrix is reused across shuffles, see '
                              f'module docstring; pass 0 to skip and compare against the '
                              f'theoretical chance_level=1/n_categories instead).')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--force', action='store_true',
                         help='Overwrite existing .npz outputs instead of skipping them.')
    args = parser.parse_args()

    bids_root = get_bids_root()
    print(f'linear_decoding_categories_cell | sub-{args.subjID:02d} | bands={args.bands} | '
          f'{args.voxRes} | conditions={args.conditions} | rois={args.rois} | '
          f'schemes={args.schemes} | points_per_category={args.points_per_category} | '
          f'win_ms={args.win_ms} | time_stride_ms={args.time_stride_ms} | '
          f'n_shuffle={args.n_shuffle} | seed={args.seed} | force={args.force}', flush=True)

    t0 = time.time()
    run_cell(args.subjID, list(args.bands), args.voxRes, bids_root,
              list(args.rois), list(args.conditions), list(args.schemes),
              args.points_per_category, args.win_ms, args.time_stride_ms,
              args.n_shuffle, args.seed,
              outdir=args.outdir, force=args.force)

    print(f'Done | sub-{args.subjID:02d} | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
