#!/usr/bin/env python3
"""
representational_distance_ts_cell.py

Decoder-independent, GLUE-independent sanity check: at every timepoint,
compare the average Euclidean distance between same-location trial pairs
("within") against different-location trial pairs ("between"), across the
FULL stim-locked timeline (not epoch-restricted) -- same moving-window
convention as decoding_ts_cell.py (imports its moving_window_mean directly,
same win_ms default), so this is directly time-aligned/comparable to the
decoding-error timeseries in plot_decoding_ts.py.

Motivation: manifold_capacity.py's ONE_VERSUS_REST result on all 10 target
locations was barely above its shuffle null (see chat history). Both actual
decoders in this repo (svr_tgm.py, decoding_ts_cell.py) decode a CONTINUOUS
circular angle, never treat the 10 locations as discrete unordered
categories -- so GLUE's categorical one-vs-rest framing may simply be
mismatched to a smooth circular code, rather than there being no real
structure. This script tests for structure directly, independent of both
GLUE's convex-optimization machinery and any decoder's assumptions: if
same-location trials sit reliably closer together in raw feature space than
different-location trials, real spatial structure exists regardless of
what any downstream method concludes about it.

Method, per (band, roi, condition, scheme), for one subject:
    0. Map the 10 raw target locations to this scheme's coarser categories
       (constants.CATEGORY_SCHEMES -- 2=left/right hemifield, 4=quadrants
       excluding the 2 axis locations, 6=quadrants + the 2 axis locations
       as singletons, 10=every raw location, no grouping), then randomly
       balance every category down to exactly --points_per_category points
       (constants.balance_categories) so schemes with different category
       counts stay apples-to-apples -- a difference in the within/between
       gap across schemes then reflects genuine granularity-dependent
       separability, not just different per-category sample sizes.
    1. Load stim-locked G04 features (build_features, shared with
       run_glue_cell.py/decoding_ts_cell.py) and apply the SAME +-win_ms
       moving-window average as decoding_ts_cell.py (imported, not
       duplicated, so the two timeseries are directly comparable).
    2. At each timepoint t, independently:
       a. Optionally z-score each source across trials at this timepoint
          (default OFF -- matches manifold_capacity.py's reasoning: GLUE's
          own preprocessing only mean-centers, and per-source amplitude
          variance is plausibly real signal, not a units artifact. Kept as
          an option since comparing z-scored vs. unscored gaps also
          empirically informs that same open question for manifold_capacity.py).
       b. Compute the (n_trials, n_trials) pairwise Euclidean distance
          matrix, then split its upper-triangle pairs into "within"
          (same target location, different trials) and "between"
          (different target locations), and record
          gap = mean(between) - mean(within). Positive gap = same-location
          trials are more similar than different-location trials.
       c. Build a null distribution for that same gap statistic via label
          permutation (shuffle target labels across trials, n_perm times,
          recompute the gap on the SAME distance matrix each time -- cheap,
          since only which pairs count as "within" changes, not the
          distances themselves). This deliberately mirrors GLUE's own
          shuffle-null logic, so it's a conceptually consistent reference.
          Saves null_mean/null_std/p_value (right-tailed: fraction of null
          gaps >= the real gap) per timepoint.

Output: one .npz per (subject, band, roi, condition, scheme) at
derivatives/sub-XX/sourceRecon/repDistTS/sub-XX_task-mgs_repDistTS_
{condition}_{band}_{roi}_{voxRes}_scheme{scheme}.npz (mirrors decodingTS's
per-cell layout).

Does NOT require the `glue` package -- pure numpy/scipy, runs in this
repo's normal Python environment (no special vader conda env needed).

Usage:
    python representational_distance_ts_cell.py <subjID>
        [--bands theta alpha beta lowgamma highgamma]
        [--voxRes 8mm] [--rois visual parietal frontal]
        [--conditions ampOnly] [--schemes 2 4 6 10]
        [--points_per_category 10] [--win_ms 50] [--n_perm 1000] [--zscore]
        [--seed 0] [--outdir <path>] [--force]
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

from constants import (AMP_ONLY_BANDS, AMP_PHASE_BANDS, ROI_NAMES, get_bids_root,
                        CATEGORY_SCHEMES, category_labels_for_scheme, balance_categories)
from decoding_ts_cell import DEFAULT_WIN_MS, moving_window_mean
from features import build_features
from io_g04 import load_g04_band

DEFAULT_N_PERM = 1000
DEFAULT_POINTS_PER_CATEGORY = 10

# ── Output path ─────────────────────────────────────────────────────────────

def output_path(bids_root, subjID, band, roi, condition, voxRes, scheme, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'repDistTS')
    os.makedirs(base, exist_ok=True)
    fname = f'{subName}_task-mgs_repDistTS_{condition}_{band}_{roi}_{voxRes}_scheme{scheme}.npz'
    return os.path.join(base, fname)


# ── Permutation structure (label-dependent, timepoint-independent) ─────────

def build_permutation_structure(target_labels, n_perm, seed):
    """
    Precomputes everything that depends on trial LABELS but not on the
    distances themselves, so it's computed once per cell and reused across
    every timepoint's distance matrix.

    Returns (iu, ju, same_real, same_perm, n_within_real, n_between_real,
    n_within_perm, n_between_perm):
        iu, ju            : upper-triangle pair indices (i < j), each (M,)
        same_real         : (M,) bool, True where trial i/j share a label
        same_perm         : (n_perm, M) bool, same but for each permuted labeling
        n_within/between_*: pair counts (real: scalars; perm: (n_perm,) arrays)
    """
    n = target_labels.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    same_real = (target_labels[iu] == target_labels[ju])

    rng = np.random.default_rng(seed)
    perm_labels = np.stack([rng.permutation(target_labels) for _ in range(n_perm)])  # (n_perm, n)
    same_perm = (perm_labels[:, iu] == perm_labels[:, ju])  # (n_perm, M)

    n_within_real  = int(same_real.sum())
    n_between_real = same_real.size - n_within_real
    n_within_perm  = same_perm.sum(axis=1).astype(np.float64)          # (n_perm,)
    n_between_perm = same_perm.shape[1] - n_within_perm                # (n_perm,)

    return (iu, ju, same_real, same_perm,
            n_within_real, n_between_real, n_within_perm, n_between_perm)


# ── Core computation ─────────────────────────────────────────────────────────

def representational_distance_timeseries(X_win, target_labels, n_perm=DEFAULT_N_PERM,
                                          zscore=False, seed=0):
    """
    X_win: (n_trials, n_times, n_feat) already moving-window-averaged.
    target_labels: (n_trials,) int.

    Returns dict with (n_times,) arrays: gap, within_mean, between_mean,
    null_mean, null_std, p_value.
    """
    n_trials, n_times, n_feat = X_win.shape

    (iu, ju, same_real, same_perm,
     n_within_real, n_between_real, n_within_perm, n_between_perm) = \
        build_permutation_structure(target_labels, n_perm, seed)

    same_real_f = same_real.astype(np.float64)
    same_perm_f = same_perm.astype(np.float64)   # (n_perm, M)

    gap          = np.zeros(n_times, dtype=np.float64)
    within_mean  = np.zeros(n_times, dtype=np.float64)
    between_mean = np.zeros(n_times, dtype=np.float64)
    null_mean    = np.zeros(n_times, dtype=np.float64)
    null_std     = np.zeros(n_times, dtype=np.float64)
    p_value      = np.zeros(n_times, dtype=np.float64)

    for t in range(n_times):
        X_t = X_win[:, t, :]   # (n_trials, n_feat)

        if zscore:
            mu = X_t.mean(axis=0, keepdims=True)
            sd = X_t.std(axis=0, keepdims=True)
            sd[sd < 1e-10] = 1.0
            X_t = (X_t - mu) / sd

        # Pairwise Euclidean distances via the Gram-matrix identity
        # (BLAS matmul, much faster than an explicit N^2 loop).
        sq = np.sum(X_t.astype(np.float64) ** 2, axis=1)               # (n_trials,)
        gram = X_t.astype(np.float64) @ X_t.astype(np.float64).T       # (n_trials, n_trials)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * gram, 0.0)
        d_vec = np.sqrt(d2[iu, ju])                                    # (M,) upper-triangle distances
        total_sum = d_vec.sum()

        # Real gap
        within_sum_real = same_real_f @ d_vec
        w_real = within_sum_real / n_within_real
        b_real = (total_sum - within_sum_real) / n_between_real
        within_mean[t]  = w_real
        between_mean[t] = b_real
        gap[t] = b_real - w_real

        # Permutation null (reuses d_vec -- only label->pair membership changes)
        within_sum_perm = same_perm_f @ d_vec                          # (n_perm,)
        w_perm = within_sum_perm / n_within_perm
        b_perm = (total_sum - within_sum_perm) / n_between_perm
        gap_perm = b_perm - w_perm

        null_mean[t] = gap_perm.mean()
        null_std[t]  = gap_perm.std()
        # Right-tailed: is the real gap bigger than the shuffled-label null?
        # +1/+1 smoothing avoids a p=0 claim from a finite permutation count.
        p_value[t] = (np.sum(gap_perm >= gap[t]) + 1) / (n_perm + 1)

    return dict(gap=gap, within_mean=within_mean, between_mean=between_mean,
                null_mean=null_mean, null_std=null_std, p_value=p_value)


# ── Per-subject cell runner ─────────────────────────────────────────────────

def run_cell(subjID, bands, voxRes, bids_root, rois, conditions, schemes,
             points_per_category, win_ms, n_perm, zscore, seed, outdir=None, force=False):
    lockType = 'stim'   # stim-locked only, matches decoding_ts_cell.py

    for band in bands:
        for condition in conditions:
            want_phase = (condition == 'ampPhase')
            if want_phase and band not in AMP_PHASE_BANDS:
                print(f'SKIP {condition}/{band}: phase not available '
                      f'(AMP_PHASE_BANDS={AMP_PHASE_BANDS})')
                continue

            for roi in rois:
                # Cell here = (band, condition, roi) -- loaded/windowed ONCE,
                # shared across every scheme below (scheme only changes the
                # label grouping/subsampling, not the underlying features).
                # Skip the load entirely if every scheme's output already exists.
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
                              f'fsample={fsample:.0f}Hz | win=+-{win_ms:.0f}ms | n_perm={n_perm} | '
                              f'zscore={zscore} | {n_categories} categories x '
                              f'{points_per_category} pts', flush=True)

                        result = representational_distance_timeseries(
                            X_win_scheme, labels_scheme, n_perm=n_perm, zscore=zscore, seed=seed)

                        np.savez_compressed(
                            out_path,
                            gap          = result['gap'].astype(np.float32),           # (T,)
                            within_mean  = result['within_mean'].astype(np.float32),   # (T,)
                            between_mean = result['between_mean'].astype(np.float32),  # (T,)
                            null_mean    = result['null_mean'].astype(np.float32),     # (T,)
                            null_std     = result['null_std'].astype(np.float32),      # (T,)
                            p_value      = result['p_value'].astype(np.float32),       # (T,)
                            time_vector  = tv.astype(np.float32),
                            group_labels = labels_scheme,
                            subjID       = np.array([subjID]),
                            band         = np.array([band]),
                            condition    = np.array([condition]),
                            roi          = np.array([roi]),
                            scheme       = np.array([scheme]),
                            n_categories = np.array([n_categories]),
                            points_per_category = np.array([points_per_category]),
                            win_ms       = np.array([win_ms]),
                            n_perm       = np.array([n_perm]),
                            zscore       = np.array([zscore]),
                            seed         = np.array([seed]),
                            fsample      = np.array([fsample]),
                        )
                        print(f'  Saved: {out_path}')
                    except ValueError as e:
                        print(f'  SKIP scheme={scheme}: {e}')
                    except Exception:
                        print(f'  FAILED sub-{subjID:02d} {band}/{condition}/{roi}/scheme{scheme} '
                              f'(skipping this scheme only):', flush=True)
                        traceback.print_exc()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Representational-distance-over-time cell: one subject, one or more bands.')
    parser.add_argument('subjID', type=int)
    parser.add_argument('--bands',      nargs='+', default=list(AMP_ONLY_BANDS))
    parser.add_argument('--voxRes',     default='8mm')
    parser.add_argument('--rois',       nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions', nargs='+', default=['ampOnly'],
                         help="Feature conditions (default: ['ampOnly'], matching "
                              "manifold_capacity.py's current scope; pass ampPhase too if needed).")
    parser.add_argument('--schemes',  nargs='+', type=int, default=sorted(CATEGORY_SCHEMES),
                         choices=sorted(CATEGORY_SCHEMES),
                         help='Category-grouping schemes to test (see constants.CATEGORY_SCHEMES): '
                              '2=left/right, 4=quadrants, 6=quadrants+axis, 10=every raw location.')
    parser.add_argument('--points_per_category', type=int, default=DEFAULT_POINTS_PER_CATEGORY,
                         help=f'Every category (in every scheme) is balanced down to exactly this '
                              f'many points (default {DEFAULT_POINTS_PER_CATEGORY}) -- see module docstring.')
    parser.add_argument('--win_ms',   type=float, default=DEFAULT_WIN_MS,
                         help=f'One-sided moving-window half-width in ms '
                              f'(default {DEFAULT_WIN_MS}, same as decoding_ts_cell.py).')
    parser.add_argument('--n_perm',   type=int, default=DEFAULT_N_PERM)
    parser.add_argument('--zscore',   action='store_true',
                         help='Z-score each source across trials at each timepoint before '
                              'computing distances (default OFF -- see module docstring).')
    parser.add_argument('--seed',     type=int, default=0)
    parser.add_argument('--outdir',   default=None)
    parser.add_argument('--force',    action='store_true',
                         help='Overwrite existing .npz outputs instead of skipping them.')
    args = parser.parse_args()

    bids_root = get_bids_root()
    print(f'representational_distance_ts_cell | sub-{args.subjID:02d} | bands={args.bands} | '
          f'{args.voxRes} | conditions={args.conditions} | rois={args.rois} | '
          f'schemes={args.schemes} | points_per_category={args.points_per_category} | '
          f'win_ms={args.win_ms} | n_perm={args.n_perm} | zscore={args.zscore} | '
          f'seed={args.seed} | force={args.force}', flush=True)

    run_cell(args.subjID, list(args.bands), args.voxRes, bids_root,
              list(args.rois), list(args.conditions), list(args.schemes),
              args.points_per_category, args.win_ms, args.n_perm, args.zscore, args.seed,
              outdir=args.outdir, force=args.force)

    print(f'Done | sub-{args.subjID:02d}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
