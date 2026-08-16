#!/usr/bin/env python3
"""
manifold_capacity.py

Runs glue's manifold-capacity analysis (glue.contrib.glue_analysis_dataframe)
on this project's stim-locked G04 features, for ONE subject across all
(band, roi, condition, scheme, time-window) combinations. NOT part of the
glue_decoding TGM pipeline (run_glue_cell.py et al.) -- "glue" here refers to
the separate manifold-capacity-theory package (github.com/cnchou/glue),
installed only where noted below; the name collision with this directory is
coincidental.

WHAT CHANGED FROM THE FIRST VERSION (and why)
Previously this script ran two fixed epochs (stim=[0,0.2], delay=[0.2,1.7]),
amplitude-only features, ERP kept, and swept four category schemes
(P=2/4/6/10). Given the decoding/geometry results so far it now runs:

  1. SLIDING WINDOW instead of two fixed epochs. Capacity/radius/dimension
     are computed in a moving --win_ms (default 100 ms) window stepped every
     --time_stride_ms (default 100 ms, i.e. non-overlapping by default)
     across [--tmin, --tmax]. The two fixed epochs averaged over 200 ms and
     1500 ms respectively, which cannot show WHEN geometry changes -- the
     delay window in particular collapsed 1.5 s of (very likely
     non-stationary) dynamics into one point per trial. The `epoch` index
     column is replaced by `t_center`/`t_start`/`t_stop`; downstream
     aggregation plots time courses, not bars (see
     aggregate_glue_capacity.py).
     COST: the window grid multiplies glue calls by n_windows, and each call
     is a cvxopt QP sweep. Defaults (2.2 s span, 100 ms non-overlapping
     windows -> 22 windows) x 3 bands x 4 (roi, condition) cells x 1 scheme
     = ~264 calls/subject. Widening the span or halving the stride scales
     that linearly -- check a single-subject run before launching the fleet.

  2. ERP REMOVED (default ON, --no_erp_removal to disable). The grand
     trial-average is subtracted per (band, roi, condition) cell at native
     time resolution, BEFORE windowing -- the same operation, computed the
     same way, as decoding_ts_cell.py / linear_decoding_categories_cell.py /
     circular_tgm_cell.py, so this script's preprocessing is line-for-line
     comparable with the decoders whose results motivated it. It is the grand
     mean across ALL trials in the cell, NOT per target location (that would
     remove exactly the location-specific structure the manifolds are built
     from). Note this shifts every manifold's centroid by a common vector,
     so it changes center_alignment/capacity in a way an ERP-kept run would
     not -- that is the point, not a side effect: the question here is the
     geometry of trial-to-trial deviations from the common evoked response.

  3. BANDS theta/alpha/beta only (the bands with saved phase), instead of
     all five.

  4. TWO FEATURE CONDITIONS, ampOnly and ampPhase (features.build_features,
     shared with every other cell script): ampOnly for every ROI, ampPhase
     for --phase_rois only (default: visual). Restricting ampPhase to visual
     matches where phase-carrying structure was actually found and keeps the
     window-grid cost bounded. Remember ampPhase is 2x the feature count of
     ampOnly by construction ([amp*cos, amp*sin]) -- with QR capping
     effective dimension at min(n_features, n_points) and n_points = n_trials
     here (see caveat below), that cap is what binds in both conditions, so
     the two are NOT confounded by raw feature count; they differ in what
     information those points carry.

  5. SCHEME 4 ONLY by default (P=4 quadrant manifolds, which is exactly
     "ignore location 0 deg and 180 deg" -- CATEGORY_SCHEMES[4] drops target
     labels 1 and 6, the two axis-aligned locations that sit on a quadrant
     boundary). P=10 raw-location manifolds are still selectable via
     --schemes but are no longer the default sweep.

For each (band, roi, condition, scheme):
    1. Load stim-locked G04 amplitude (+ phase when any requested condition
       for this ROI needs it) for one subject, once per (band, roi).
    2. build_features -> (n_trials, n_times, n_features).
    3. Subtract the ERP (see 2. above), at native resolution.
    4. Map trials to this scheme's categories (constants.category_labels_for_scheme
       -- drops trials whose raw location isn't in any category, e.g. labels
       1 and 6 for scheme=4) and balance every category to the same trial
       count. points_per_category defaults to None (auto: this subject's own
       smallest category trial count for this scheme); a fixed cross-scheme
       cap wastes usable data, so --points_per_category is opt-in. Balancing
       happens ONCE per cell, outside the time loop, so every window uses the
       SAME trials -- otherwise the time course would confound geometry
       changes with resampling noise.
    5. For each window, average features over [t_center - win/2,
       t_center + win/2] -> ONE point per trial, then split into one manifold
       per category. manifolds are (n_features, n_trials_for_that_category)
       arrays -- glue expects (n_neurons, n_points), the TRANSPOSE of every
       other array shape convention used in glue_decoding.
       CAVEAT (unchanged from the fixed-epoch version): glue_analysis's QR
       rotation caps effective ambient dimension at min(n_features,
       total_points); with one point per trial (~150 points) that cap sits
       BELOW every ROI's raw source count, so all ROIs collapse to the same
       ~150-dim space regardless of raw ROI size -- cross-ROI capacity
       comparisons should be read with that in mind, not as reflecting true
       per-ROI dimensionality. An intermediate version treated every
       timepoint as its own point to avoid this, but cvxopt's QP solves
       scale badly with point count (a 4-scheme x 5-band x 3-roi x
       21-subject stim-only run was still unfinished after 5+ hours);
       time-averaging within the window is what keeps this tractable, and is
       doubly necessary now that there are many windows.
    6. Optional per-source z-scoring across trials (--zscore, default OFF).
       glue's own preprocessing (reallocate_origin) only mean-centers, never
       rescales -- its implicit assumption is that raw feature magnitude is
       meaningful. For source-space amplitude, per-voxel variance differences
       plausibly ARE real signal (proximity to true source, genuine task
       modulation), not an arbitrary-units artifact.
    7. glue_analysis_dataframe(shuffle=True) so each cell reports both the
       real dichotomy geometry and a shuffled-points null in one call.

Output: one CSV per subject at derivatives/sub-XX/sourceRecon/glueFits/
sub-XX_task-mgs_glueFits_{lockType}_{voxRes}.csv (see constants.glue_fits_csv_path),
matching run_glue_cell.py's decodingGlue / decoding_ts_cell.py's decodingTS
per-subject layout. Index columns: subjID, band, condition, roi, scheme,
t_center. No dedicated log file -- prints only (with flush=True, so tail -f
on the redirected log shows real-time progress rather than sitting in
Python's stdout buffer), same pattern as every other glue_decoding script;
run_glue_capacity.sh redirects stdout into logs_glue_capacity_<voxRes>/ in
the code's own working directory.

Requires the `glue` package (github.com/cnchou/glue, distinct from PyPI
`glue-core`/glueviz), which is NOT part of this repo's normal Python
environment -- only installed in a separate env on vader. Run this with
whichever interpreter/conda env has `glue.contrib.glue_analysis_dataframe`
importable, e.g.:
    conda activate <env-with-glue> && python manifold_capacity.py

Usage:
    python manifold_capacity.py [--subjID 1] [--lockType stim] [--voxRes 8mm]
                                 [--bands theta alpha beta]
                                 [--rois visual parietal frontal]
                                 [--conditions ampOnly ampPhase]
                                 [--phase_rois visual]
                                 [--schemes 4]
                                 [--win_ms 100] [--time_stride_ms 100]
                                 [--tmin -0.5] [--tmax 1.7]
                                 [--points_per_category N] [--zscore]
                                 [--no_erp_removal]
                                 [--analysis_type ONE_VERSUS_REST]
                                 [--n_hyperplanes 200] [--seed 42]
                                 [--min_trials_per_class 2]
                                 [--no_shuffle] [--outdir <path>] [--force]

Meant to be launched as one background job per subject by
run_glue_capacity.sh (parallel across subjects, default concurrency =
n_subjects), the same pattern as run_decoding_ts.sh / run_glue_decoding.sh.
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from constants import (AMP_PHASE_BANDS, ROI_NAMES, get_bids_root, glue_fits_csv_path,
                        CATEGORY_SCHEMES, category_labels_for_scheme, balance_categories)
from features import VALID_CONDITIONS, build_features
from io_g04 import load_g04_band

try:
    from glue.contrib import glue_analysis_dataframe
except ImportError as e:
    raise ImportError(
        "Could not import glue.contrib.glue_analysis_dataframe -- this script "
        "needs the manifold-capacity-theory `glue` package (github.com/cnchou/glue), "
        "which lives in a separate environment (e.g. on vader), not this repo's "
        "normal Python env. Activate that environment first."
    ) from e

# Sliding-window defaults. WIN_MS is the FULL window width (not the one-sided
# half-width DEFAULT_WIN_MS in decoding_ts_cell.py, which is +/-50 ms = the
# same 100 ms total) -- windows here are explicit [start, stop] intervals
# rather than a per-timepoint centred average, since each window produces one
# glue fit rather than one value per native timepoint.
DEFAULT_WIN_MS         = 100.0
DEFAULT_TIME_STRIDE_MS = 100.0    # == WIN_MS -> non-overlapping windows
DEFAULT_TMIN           = -0.5     # some pre-stim baseline windows as a reference level
DEFAULT_TMAX           = 1.7      # end of the delay period

# ROIs for which phase-carrying conditions are run (see module docstring #4).
DEFAULT_PHASE_ROIS = ('visual',)

PHASE_CONDITIONS = ('ampPhase', 'phaseOnly')


# ── Time windows ───────────────────────────────────────────────────────────

def build_windows(tv, tmin, tmax, win_ms, stride_ms):
    """
    Sliding windows over [tmin, tmax] (both clipped to the data's own time
    vector). Only FULL-width windows are emitted -- a trailing partial window
    would average over fewer timepoints than the rest and give that one point
    a different noise level, which would show up as a spurious end-of-epoch
    change in radius/dimension.

    Returns a list of (t_start, t_stop, t_center) tuples in seconds.
    """
    win    = win_ms * 1e-3
    stride = stride_ms * 1e-3
    if win <= 0:
        raise ValueError(f'win_ms must be > 0, got {win_ms}')
    if stride <= 0:
        raise ValueError(f'time_stride_ms must be > 0, got {stride_ms}')

    lo = max(float(tmin), float(tv.min()))
    hi = min(float(tmax), float(tv.max()))
    if hi - lo < win:
        raise ValueError(
            f'Requested span [{lo}, {hi}] is shorter than one {win_ms} ms window.')

    windows = []
    # Half a sample of slack so floating-point edges don't drop the last window.
    eps = 0.5 / max(len(tv) - 1, 1) * (float(tv.max()) - float(tv.min()))
    start = lo
    while start + win <= hi + eps:
        stop = start + win
        windows.append((start, stop, start + win / 2.0))
        start += stride
    return windows


# ── Manifold construction ──────────────────────────────────────────────────

def select_scheme_trials(target_labels, scheme, points_per_category=None,
                          min_trials_per_class=2, seed=42, log=print):
    """
    Maps raw 1-10 target_labels to this scheme's categories and balances
    every category to the same trial count -- ONCE per cell, so every time
    window downstream is built from the SAME trials (see module docstring #4).

    Returns (trial_idx, labels_kept):
      trial_idx   : (n_kept,) int index into the ORIGINAL trial axis
      labels_kept : (n_kept,) category-name array, aligned with trial_idx
    Returns (None, None) if this scheme is unusable for this subject
    (no trials map to any category, or the smallest category is below
    min_trials_per_class).
    """
    group_labels, keep_mask = category_labels_for_scheme(target_labels, scheme)
    if group_labels.size == 0:
        log(f'    NOTE: scheme={scheme}: no trials map to any category -- skipping.')
        return None, None

    _, counts = np.unique(group_labels, return_counts=True)
    natural_min = int(counts.min())
    ppc_used = points_per_category if points_per_category is not None else natural_min
    if ppc_used < min_trials_per_class:
        log(f'    NOTE: scheme={scheme}: points_per_category={ppc_used} '
            f'(natural min={natural_min}) < min_trials_per_class={min_trials_per_class} '
            f'-- skipping this scheme.')
        return None, None

    try:
        balance_mask = balance_categories(group_labels, ppc_used, seed=seed)
    except ValueError as e:
        log(f'    NOTE: scheme={scheme}: {e} -- skipping this scheme.')
        return None, None

    trial_idx = np.where(keep_mask)[0][balance_mask]
    return trial_idx, group_labels[balance_mask]


def build_window_manifolds(X, tv, trial_idx, labels_kept, t_start, t_stop,
                            zscore=False):
    """
    X: (n_trials, n_times, n_features), already ERP-removed.
    trial_idx / labels_kept: from select_scheme_trials (shared across windows).

    Averages features over [t_start, t_stop] -> ONE point per trial, then
    splits into one manifold per category.

    Returns (manifolds, kept_labels) where each manifold is
    (n_features, n_trials_for_that_category) -- glue's (n_neurons, n_points)
    convention, the transpose of this repo's usual (n_trials, n_features).
    """
    win_mask = (tv >= t_start) & (tv <= t_stop)
    if not win_mask.any():
        raise ValueError(f'Window [{t_start}, {t_stop}] has no timepoints in this time_vector.')

    P = X[trial_idx][:, win_mask, :].mean(axis=1)      # (n_kept, n_features)

    if zscore:
        mu = P.mean(axis=0, keepdims=True)
        sd = P.std(axis=0, keepdims=True)
        sd = np.where(sd < 1e-10, np.asarray(1.0, dtype=sd.dtype), sd)
        P = (P - mu) / sd

    manifolds, kept_labels = [], []
    for cat in sorted(np.unique(labels_kept)):
        sel = labels_kept == cat
        manifolds.append(np.ascontiguousarray(P[sel].T))
        kept_labels.append(cat)
    return manifolds, kept_labels


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='glue manifold-capacity analysis in a sliding time window on '
                     'stim-locked, ERP-removed G04 features, one subject x all '
                     'bands x rois x conditions x schemes.')
    parser.add_argument('--subjID',   type=int, default=1)
    parser.add_argument('--lockType', default='stim', choices=['stim', 'resp'])
    parser.add_argument('--voxRes',   default='8mm')
    parser.add_argument('--bands',    nargs='+', default=list(AMP_PHASE_BANDS),
                         help='Default theta alpha beta (the bands with saved phase).')
    parser.add_argument('--rois',     nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'],
                         choices=list(VALID_CONDITIONS),
                         help='Feature conditions (features.build_features). Phase-carrying '
                              'conditions run only for --phase_rois.')
    parser.add_argument('--phase_rois', nargs='+', default=list(DEFAULT_PHASE_ROIS),
                         help='ROIs for which phase-carrying conditions (ampPhase/phaseOnly) '
                              'are run. Default: visual only. Pass "all" for every --rois.')
    parser.add_argument('--schemes',  nargs='+', type=int, default=[4],
                         choices=sorted(CATEGORY_SCHEMES),
                         help='Category-grouping schemes (see constants.CATEGORY_SCHEMES). '
                              'Default 4 = quadrants, which excludes the two axis-aligned '
                              'locations (0 deg and 180 deg).')
    parser.add_argument('--win_ms', type=float, default=DEFAULT_WIN_MS,
                         help='FULL sliding-window width in ms (default 100).')
    parser.add_argument('--time_stride_ms', type=float, default=DEFAULT_TIME_STRIDE_MS,
                         help='Step between window starts in ms (default 100 = non-overlapping). '
                              'Halving this doubles the number of glue fits.')
    parser.add_argument('--tmin', type=float, default=DEFAULT_TMIN)
    parser.add_argument('--tmax', type=float, default=DEFAULT_TMAX)
    parser.add_argument('--analysis_type', default='ONE_VERSUS_REST')
    parser.add_argument('--n_hyperplanes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_trials_per_class', type=int, default=2)
    parser.add_argument('--points_per_category', type=int, default=None,
                         help='Fixed trial count per category, applied identically across '
                              'every scheme. Default: None (auto -- balance each scheme to '
                              'that subject\'s own smallest category trial count for THAT scheme).')
    parser.add_argument('--zscore', action='store_true',
                         help='Z-score each feature across trials within each window '
                              'before building manifolds (default OFF -- see module docstring).')
    parser.add_argument('--no_erp_removal', action='store_false', dest='remove_erp',
                         help='Skip ERP (grand trial-average) subtraction '
                              '(default: ERP IS subtracted -- see module docstring).')
    parser.add_argument('--no_shuffle', action='store_true',
                         help='Skip the shuffled-manifolds null (glue_analysis_dataframe shuffle=False).')
    parser.add_argument('--outdir', default=None,
                         help='Directory for the results CSV, overriding the default '
                              'per-subject layout (<bids_root>/derivatives/sub-XX/'
                              'sourceRecon/glueFits/).')
    parser.add_argument('--force', action='store_true',
                         help='Overwrite an existing results CSV instead of skipping.')
    args = parser.parse_args()

    bids_root = get_bids_root()
    csv_path = glue_fits_csv_path(bids_root, args.subjID, args.lockType, args.voxRes,
                                   outdir=args.outdir)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path) and not args.force:
        print(f'SKIP (exists): {csv_path} -- pass --force to overwrite.')
        return

    # No dedicated log file here -- matches every other glue_decoding script
    # (run_glue_cell.py, decoding_ts_cell.py, intrinsic_dim_epochs.py, ...),
    # none of which write their own log; they just print() and rely on the
    # launcher shell script (run_glue_capacity.sh) to redirect stdout into
    # logs_glue_capacity_<voxRes>/ in the code's own working directory.
    # flush=True: stdout is block-buffered (not line-buffered) once
    # redirected to a file rather than a terminal, so without this, output
    # sits invisible in Python's internal buffer until it fills or the
    # process exits -- `tail -f` on the log otherwise shows nothing for a
    # long time even while the run is genuinely progressing.
    def log(msg=''):
        print(msg, flush=True)

    phase_rois = (list(args.rois) if len(args.phase_rois) == 1 and args.phase_rois[0] == 'all'
                  else list(args.phase_rois))

    log(f'manifold_capacity | sub-{args.subjID:02d} | {args.lockType} | {args.voxRes} | '
        f'bands={args.bands} | rois={args.rois} | conditions={args.conditions} | '
        f'phase_rois={phase_rois} | schemes={args.schemes} | '
        f'win={args.win_ms}ms stride={args.time_stride_ms}ms '
        f'span=[{args.tmin}, {args.tmax}]s | remove_erp={args.remove_erp} | '
        f'analysis_type={args.analysis_type} | n_hyperplanes={args.n_hyperplanes} | '
        f'seed={args.seed} | shuffle={not args.no_shuffle} | zscore={args.zscore} | '
        f'points_per_category={args.points_per_category or "auto"}')

    df_list = []
    t_run = time.time()

    for band in args.bands:
        for roi in args.rois:
            # Which conditions are actually runnable for this (band, roi)?
            conditions = []
            for cond in args.conditions:
                if cond in PHASE_CONDITIONS:
                    if roi not in phase_rois:
                        continue
                    if band not in AMP_PHASE_BANDS:
                        log(f'-- band={band} roi={roi}: SKIP {cond} '
                            f'(no saved phase; AMP_PHASE_BANDS={AMP_PHASE_BANDS})')
                        continue
                conditions.append(cond)
            if not conditions:
                continue

            want_phase = any(c in PHASE_CONDITIONS for c in conditions)

            log(f'\n-- band={band} roi={roi} conditions={conditions} --')
            try:
                g04 = load_g04_band(args.subjID, args.lockType, band, args.voxRes,
                                     bids_root, want_phase=want_phase, roi=roi)
            except (FileNotFoundError, ValueError) as e:
                log(f'  SKIP (load failed): {e}')
                continue

            amp   = g04['amp']                     # (n_trials, n_times, n_sources)
            phase = g04['phase'] if want_phase else None
            tv    = g04['time_vector']
            target_labels = g04['target_labels']

            log(f'  loaded: {amp.shape[0]} trials, {amp.shape[1]} timepoints, '
                f'{amp.shape[2]} sources')

            try:
                windows = build_windows(tv, args.tmin, args.tmax,
                                         args.win_ms, args.time_stride_ms)
            except ValueError as e:
                log(f'  SKIP (window grid): {e}')
                continue
            log(f'  windows: {len(windows)} x {args.win_ms:.0f}ms '
                f'(stride {args.time_stride_ms:.0f}ms), centers '
                f'{windows[0][2]:+.3f}..{windows[-1][2]:+.3f}s')

            for condition in conditions:
                X = build_features(condition, amp, phase)

                # ERP removal: subtract the grand trial-average (across ALL
                # trials in this cell, NOT per location -- see module
                # docstring) at native time resolution, before windowing.
                if args.remove_erp:
                    X = X - X.mean(axis=0, keepdims=True)

                log(f'  condition={condition}: F={X.shape[2]} features '
                    f'(remove_erp={args.remove_erp})')

                for scheme in args.schemes:
                    trial_idx, labels_kept = select_scheme_trials(
                        target_labels, scheme,
                        points_per_category=args.points_per_category,
                        min_trials_per_class=args.min_trials_per_class,
                        seed=args.seed, log=log)
                    if trial_idx is None:
                        continue

                    cats, cat_counts = np.unique(labels_kept, return_counts=True)
                    if cats.size < 2:
                        log(f'    SKIP scheme={scheme}: only {cats.size} usable manifold(s) '
                            f'(need >= 2 for a dichotomy).')
                        continue
                    log(f'    scheme={scheme}: P={cats.size} labels={list(cats)}, '
                        f'points per manifold={list(cat_counts)} '
                        f'({trial_idx.size} trials total, 1 point/trial/window)')

                    for t_start, t_stop, t_center in windows:
                        try:
                            manifolds, kept_labels = build_window_manifolds(
                                X, tv, trial_idx, labels_kept, t_start, t_stop,
                                zscore=args.zscore)
                        except ValueError as e:
                            log(f'      SKIP window {t_center:+.3f}s: {e}')
                            continue

                        t_cell = time.time()
                        try:
                            ret = glue_analysis_dataframe(
                                manifolds,
                                indices=(args.subjID, band, condition, roi, scheme,
                                         round(float(t_center), 4)),
                                indices_name=['subjID', 'band', 'condition', 'roi',
                                              'scheme', 't_center'],
                                analysis_type=args.analysis_type,
                                n_hyperplanes=args.n_hyperplanes,
                                shuffle=not args.no_shuffle,
                                seed=args.seed,
                            )
                        except Exception:
                            log(f'      FAILED glue_analysis_dataframe for band={band} '
                                f'roi={roi} condition={condition} scheme={scheme} '
                                f't_center={t_center:+.3f}s:')
                            log(traceback.format_exc())
                            continue

                        # Window bounds / preprocessing flags as plain columns so
                        # the CSV is self-describing without re-deriving the grid.
                        ret['t_start']    = round(float(t_start), 4)
                        ret['t_stop']     = round(float(t_stop), 4)
                        ret['win_ms']     = args.win_ms
                        ret['stride_ms']  = args.time_stride_ms
                        ret['remove_erp'] = bool(args.remove_erp)
                        ret['n_features'] = int(manifolds[0].shape[0])
                        ret['n_points']   = int(sum(m.shape[1] for m in manifolds))

                        cap = ret['capacity'].to_numpy() if 'capacity' in ret else np.array([np.nan])
                        log(f'      t={t_center:+.3f}s [{t_start:+.3f},{t_stop:+.3f}] '
                            f'capacity={np.array2string(cap, precision=4)} '
                            f'({time.time() - t_cell:.1f}s)')
                        df_list.append(ret)

                del X

            del amp, phase, g04

    if df_list:
        df_all = pd.concat(df_list)
        df_all.to_csv(csv_path)
        log(f'\nSaved combined results ({len(df_all)} rows): {csv_path}  '
            f'[total {time.time() - t_run:.1f}s]')
    else:
        log('\nNo results produced -- nothing to save.')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
