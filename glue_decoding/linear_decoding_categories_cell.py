#!/usr/bin/env python3
"""
linear_decoding_categories_cell.py

Standard linear (one-vs-rest, via sklearn LinearSVC's native multi_class
handling) classification with leave-one-out cross-validation, at several
category granularities, over the FULL stim-locked timeline -- same
moving-window convention as decoding_ts_cell.py (imports moving_window_mean
directly), so this is time-aligned/comparable to both the decoding-error
timeseries and representational_distance_ts_cell.py's distance-gap timeseries.

Motivation (collaborator's suggestion, see chat history): the decoders
actually used elsewhere in this repo (svr_tgm.py, decoding_ts_cell.py) do
continuous circular regression (sin/cos of angle), never a discrete
classification -- so it's genuinely unclear whether responses to nearby
target locations are linearly separable AS DISCRETE CATEGORIES, which is
what GLUE's manifold_capacity.py ONE_VERSUS_REST assumes. This runs the
standard, well-understood tool for that exact question (SVM + LOO CV) at
four granularities (constants.CATEGORY_SCHEMES):
    2  categories : left vs right hemifield
    4  categories : quadrants (excludes the 2 axis locations)
    6  categories : quadrants + the 2 axis locations as singletons
    10 categories : every raw location (current manifold_capacity.py scope)
so we can see at what granularity linear separability actually appears,
rather than assuming 10-way is the right unit of analysis.

Every scheme is balanced to the SAME --points_per_category via
constants.balance_categories, so accuracy differences across schemes
reflect genuine granularity-dependent separability, not different
per-category sample sizes (same rationale as representational_distance_ts_cell.py).

COST NOTE (see chat history for the full benchmark): exhaustive LOO refits
one LinearSVC per left-out trial, per timepoint, per scheme -- for all 4
schemes combined at points_per_category=10, that's ~1.75s of fitting PER
EVALUATED TIMEPOINT (dominated by the 10-category scheme's 100 folds).
Two cost controls keep this tractable:
  --time_stride_ms (default 50): evaluate every ~50ms instead of every
    native ~5ms sample (the +-win_ms moving window already smooths
    adjacent samples heavily, so little is lost) -- cuts evaluated
    timepoints by ~10x.
  --n_shuffle (default 0): an empirical label-permutation null multiplies
    total cost by (1+n_shuffle) -- even n_shuffle=5 pushes a full
    subject x band x roi sweep to several hours. Default is 0 (no empirical
    null; compare real accuracy against the THEORETICAL chance level
    1/n_categories instead). Pass --n_shuffle explicitly (small values
    first) once the real-accuracy result looks worth the extra cost.
  --cv kfold is also available as a much cheaper (~20-30x) alternative to
    --cv loo (the default) if LOO proves too slow even with the above.

Output: one .npz per (subject, band, roi, condition, scheme) at
derivatives/sub-XX/sourceRecon/linDecodeCat/sub-XX_task-mgs_linDecodeCat_
{condition}_{band}_{roi}_{voxRes}_scheme{scheme}.npz

Does NOT require the `glue` package -- runs in this repo's normal Python
environment (needs scikit-learn, already a dependency via svr_tgm.py).

Usage:
    python linear_decoding_categories_cell.py <subjID>
        [--bands theta alpha beta lowgamma highgamma]
        [--voxRes 8mm] [--rois visual parietal frontal]
        [--conditions ampOnly] [--schemes 2 4 6 10]
        [--points_per_category 10] [--win_ms 50] [--time_stride_ms 50]
        [--cv loo|kfold] [--n_splits 5] [--n_shuffle 0]
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
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.exceptions import ConvergenceWarning

from constants import (AMP_ONLY_BANDS, AMP_PHASE_BANDS, ROI_NAMES, get_bids_root,
                        CATEGORY_SCHEMES, category_labels_for_scheme, balance_categories)
from decoding_ts_cell import DEFAULT_WIN_MS, moving_window_mean
from features import build_features
from io_g04 import load_g04_band

DEFAULT_TIME_STRIDE_MS = 50.0
DEFAULT_N_SHUFFLE = 0
DEFAULT_POINTS_PER_CATEGORY = 10
DEFAULT_N_SPLITS = 5   # only used for --cv kfold

# Small fits on noisy/near-chance-separable data routinely fail to fully
# converge within max_iter -- expected and harmless here (this is an
# exploratory accuracy sweep, not a single carefully-tuned classifier), so
# silenced rather than spamming thousands of warnings into the log.
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# ── Output path ─────────────────────────────────────────────────────────────

def output_path(bids_root, subjID, band, roi, condition, voxRes, scheme, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'linDecodeCat')
    os.makedirs(base, exist_ok=True)
    fname = f'{subName}_task-mgs_linDecodeCat_{condition}_{band}_{roi}_{voxRes}_scheme{scheme}.npz'
    return os.path.join(base, fname)


# ── Core CV accuracy ─────────────────────────────────────────────────────────

def _cv_accuracy(X, y, cv, n_splits, seed):
    """
    One fold-loop of z-scored (train-fold stats only, no test leakage)
    LinearSVC, returns overall accuracy across all folds.
    """
    splitter = LeaveOneOut() if cv == 'loo' else \
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    correct, total = 0, 0
    for train_idx, test_idx in splitter.split(X, y):
        mu = X[train_idx].mean(axis=0, keepdims=True)
        sd = X[train_idx].std(axis=0, keepdims=True)
        sd[sd < 1e-10] = 1.0
        X_tr = (X[train_idx] - mu) / sd
        X_te = (X[test_idx] - mu) / sd

        clf = LinearSVC(max_iter=2000)
        clf.fit(X_tr, y[train_idx])
        pred = clf.predict(X_te)
        correct += int((pred == y[test_idx]).sum())
        total += test_idx.shape[0]

    return correct / total


# ── Core computation ─────────────────────────────────────────────────────────

def svm_ovr_timeseries(X_win, labels, tv, fsample, time_stride_ms=DEFAULT_TIME_STRIDE_MS,
                        cv='loo', n_splits=DEFAULT_N_SPLITS, n_shuffle=DEFAULT_N_SHUFFLE, seed=0):
    """
    X_win: (n_trials, n_times, n_feat) already moving-window-averaged.
    labels: (n_trials,) category labels (any hashable type, e.g. strings
    from category_labels_for_scheme).

    Returns dict: eval_time_vector, accuracy, shuffle_acc_mean, shuffle_acc_std,
    p_value (all length n_eval, p_value/shuffle_* all-NaN if n_shuffle == 0),
    chance_level (scalar), n_categories (scalar).
    """
    n_trials, n_times, n_feat = X_win.shape
    n_categories = len(np.unique(labels))
    chance_level = 1.0 / n_categories

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
        accuracy[i] = _cv_accuracy(X_t, labels, cv, n_splits, seed)

        if n_shuffle > 0:
            shuf_accs = np.empty(n_shuffle, dtype=np.float64)
            for k in range(n_shuffle):
                shuffled_labels = rng.permutation(labels)
                shuf_accs[k] = _cv_accuracy(X_t, shuffled_labels, cv, n_splits, seed + k + 1)
            shuffle_acc_mean[i] = shuf_accs.mean()
            shuffle_acc_std[i]  = shuf_accs.std()
            p_value[i] = (np.sum(shuf_accs >= accuracy[i]) + 1) / (n_shuffle + 1)

    return dict(eval_time_vector=tv[eval_idx], accuracy=accuracy,
                shuffle_acc_mean=shuffle_acc_mean, shuffle_acc_std=shuffle_acc_std,
                p_value=p_value, chance_level=chance_level, n_categories=n_categories)


# ── Per-subject cell runner ─────────────────────────────────────────────────

def run_cell(subjID, bands, voxRes, bids_root, rois, conditions, schemes,
             points_per_category, win_ms, time_stride_ms, cv, n_splits, n_shuffle,
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
                        balance_mask = balance_categories(
                            group_labels, points_per_category, seed=seed)

                        X_win_scheme  = X_win[keep_mask][balance_mask]
                        labels_scheme = group_labels[balance_mask]

                        n_trials, n_times, n_feat = X_win_scheme.shape
                        n_categories = len(CATEGORY_SCHEMES[scheme]['groups'])
                        print(f'  scheme={scheme}: shape=({n_trials},{n_times},{n_feat}) | '
                              f'fsample={fsample:.0f}Hz | win=+-{win_ms:.0f}ms | '
                              f'time_stride={time_stride_ms:.0f}ms | cv={cv} | '
                              f'n_shuffle={n_shuffle} | {n_categories} categories x '
                              f'{points_per_category} pts', flush=True)

                        result = svm_ovr_timeseries(
                            X_win_scheme, labels_scheme, tv, fsample,
                            time_stride_ms=time_stride_ms, cv=cv, n_splits=n_splits,
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
                            points_per_category = np.array([points_per_category]),
                            win_ms       = np.array([win_ms]),
                            time_stride_ms = np.array([time_stride_ms]),
                            cv           = np.array([cv]),
                            n_splits     = np.array([n_splits]),
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
        description='Linear (SVM, one-vs-rest) decoding-over-time at several category '
                     'granularities: one subject, one or more bands.')
    parser.add_argument('subjID', type=int)
    parser.add_argument('--bands',      nargs='+', default=list(AMP_ONLY_BANDS))
    parser.add_argument('--voxRes',     default='8mm')
    parser.add_argument('--rois',       nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions', nargs='+', default=['ampOnly'])
    parser.add_argument('--schemes',    nargs='+', type=int, default=sorted(CATEGORY_SCHEMES),
                         choices=sorted(CATEGORY_SCHEMES))
    parser.add_argument('--points_per_category', type=int, default=DEFAULT_POINTS_PER_CATEGORY)
    parser.add_argument('--win_ms',     type=float, default=DEFAULT_WIN_MS,
                         help=f'One-sided moving-window half-width in ms '
                              f'(default {DEFAULT_WIN_MS} -> {2*DEFAULT_WIN_MS:.0f}ms total window).')
    parser.add_argument('--time_stride_ms', type=float, default=DEFAULT_TIME_STRIDE_MS,
                         help=f'Evaluate every this-many ms instead of every native sample '
                              f'(default {DEFAULT_TIME_STRIDE_MS}ms -- see module docstring cost note).')
    parser.add_argument('--cv', choices=['loo', 'kfold'], default='loo',
                         help='loo = exhaustive leave-one-out (default, slower, exact). '
                              'kfold = StratifiedKFold (much cheaper, ~20-30x, see module docstring).')
    parser.add_argument('--n_splits', type=int, default=DEFAULT_N_SPLITS,
                         help='Only used when --cv kfold.')
    parser.add_argument('--n_shuffle', type=int, default=DEFAULT_N_SHUFFLE,
                         help=f'Label-permutation null repeats (default {DEFAULT_N_SHUFFLE} -- '
                              f'empirical null OFF by default given --cv loo cost; compare '
                              f'against the theoretical chance_level=1/n_categories instead. '
                              f'See module docstring cost table before setting this > 0 with --cv loo.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--force', action='store_true',
                         help='Overwrite existing .npz outputs instead of skipping them.')
    args = parser.parse_args()

    bids_root = get_bids_root()
    print(f'linear_decoding_categories_cell | sub-{args.subjID:02d} | bands={args.bands} | '
          f'{args.voxRes} | conditions={args.conditions} | rois={args.rois} | '
          f'schemes={args.schemes} | points_per_category={args.points_per_category} | '
          f'win_ms={args.win_ms} | time_stride_ms={args.time_stride_ms} | cv={args.cv} | '
          f'n_splits={args.n_splits} | n_shuffle={args.n_shuffle} | seed={args.seed} | '
          f'force={args.force}', flush=True)

    t0 = time.time()
    run_cell(args.subjID, list(args.bands), args.voxRes, bids_root,
              list(args.rois), list(args.conditions), list(args.schemes),
              args.points_per_category, args.win_ms, args.time_stride_ms,
              args.cv, args.n_splits, args.n_shuffle, args.seed,
              outdir=args.outdir, force=args.force)

    print(f'Done | sub-{args.subjID:02d} | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
