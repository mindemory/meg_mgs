#!/usr/bin/env python3
"""
manifold_capacity.py

Runs glue's manifold-capacity analysis (glue.contrib.glue_analysis_dataframe)
on this project's stim-locked G04 amplitude features, for ONE subject across
all (band, roi, epoch, scheme) combinations. NOT part of the glue_decoding TGM
pipeline (run_glue_cell.py et al.) -- "glue" here refers to the separate
manifold-capacity-theory package (github.com/cnchou/glue), installed only
where noted below; the name collision with this directory is coincidental.

SCHEME (P): which category-grouping to build manifolds from (see
constants.CATEGORY_SCHEMES, shared with representational_distance_ts_cell.py
/ linear_decoding_categories_cell.py, same rationale -- the standard linear
classifier's LOO accuracy was far more separable for the coarse P=2
left/right split than for P=10 raw locations, so it's worth asking GLUE the
same question at each granularity rather than only P=10):
    2  categories : left vs right hemifield
    4  categories : quadrants (excludes the 2 axis locations)
    6  categories : quadrants + the 2 axis locations as singletons
    10 categories : every raw location (the only option before)
Every scheme is balanced via constants.balance_categories at the TRIAL
level (before per-timepoint expansion below) -- points_per_category
defaults to None (auto: each subject's own smallest category TRIAL count
for that scheme), matching the other two scripts' fix (a fixed cross-scheme
cap wastes usable data; each scheme's own geometry is what it is for
whatever N that scheme naturally has, not something that needs matching
across schemes). Pass --points_per_category to force a fixed trial cap.

For each (band, roi, epoch, scheme):
    1. Load stim-locked G04 amplitude for one subject (want_phase=False --
       amplitude only, matches build_features's 'ampOnly' condition). Loaded
       once per (band, roi), shared across both epochs.
    2. Slice the epoch window (EPOCHS below -- stim=[0.0,0.2], delay=
       [0.2,1.7], matching intrinsic_dim_epochs.py) and treat EVERY
       timepoint in that window as its own point (no time-averaging) --
       each trial contributes n_epoch_timepoints points instead of 1. G04
       stores everything at a shared 200 Hz (TARGET_RATE in
       G04_BandAmplitudePhaseInSource.m, ~5ms spacing), so stim (~41
       timepoints/trial) and especially delay (~301 timepoints/trial) can
       generate a LOT of points -- e.g. ~6.3k points for stim and ~46k for
       delay at n_trials=154. This matters because glue_analysis's QR
       rotation caps effective ambient dimension at min(n_features,
       total_points); with time-averaging (n_trials points only, the
       previous version of this script) that cap was BELOW every ROI's raw
       source count, silently collapsing all ROIs to the same effective
       dimensionality regardless of ROI size -- per-timepoint points fixes
       that (total_points now comfortably exceeds every ROI's source count),
       at the cost of many more -- autocorrelated, since adjacent timepoints
       within a trial are highly correlated, not independent samples -- and
       much slower QP solves (cvxopt scales badly with point count). Use
       --points_per_trial to evenly subsample each trial's epoch window
       down to a manageable point count per epoch (recommended for delay
       especially); default is "use every timepoint" (no subsampling).
    3. Z-score each source (feature) across all (trial, timepoint) samples
       in the epoch (optional, default OFF). glue's own preprocessing
       (reallocate_origin) only mean-centers, never rescales -- its
       implicit assumption is that raw feature magnitude is meaningful.
       For source-space amplitude, per-voxel variance differences plausibly
       ARE real signal (proximity to true source, genuine task modulation),
       not an arbitrary-units artifact -- z-scoring would equalize a
       near-zero-variance noise voxel with a genuinely informative one.
       Pass --zscore to opt back in.
    4. Map trials to this scheme's categories (constants.category_labels_for_scheme
       -- drops trials whose raw location isn't in any category for this
       scheme, e.g. the 2 axis locations for scheme=4), balance every
       category to the same trial count (see SCHEME note above), THEN split
       (trial, timepoint) points into one manifold per category -- glue's
       ONE_VERSUS_REST dichotomy set then tests each category against the
       rest of the "concept" set, matching the classic
       manifold-capacity-theory setup (P object manifolds in shared neural
       feature space). If this scheme's smallest category has fewer than
       --min_trials_per_class DISTINCT TRIALS, the whole scheme is skipped
       for this cell (with a warning) -- trial count, not point count,
       since a class with 2 trials x 40 timepoints is still only 2
       independent samples repeated with heavy autocorrelation, not a
       genuinely diverse 80-point manifold.
    5. manifolds are (n_features, n_points_for_that_target) arrays -- glue
       expects (n_neurons, n_points), the TRANSPOSE of every other array
       shape convention used in glue_decoding (which is (n_trials, ...)).
    6. Run glue_analysis_dataframe(shuffle=True) so each cell reports both
       the real dichotomy geometry and a shuffled-points null in one call.

Output: one CSV per subject at derivatives/sub-XX/sourceRecon/glueFits/
sub-XX_task-mgs_glueFits_{lockType}_{voxRes}.csv (see constants.glue_fits_csv_path),
matching run_glue_cell.py's decodingGlue / decoding_ts_cell.py's decodingTS
per-subject layout. No dedicated log file -- prints only, same as every
other glue_decoding script; run_glue_capacity.sh redirects stdout into
logs_glue_capacity_<voxRes>/ in the code's own working directory.

Requires the `glue` package (github.com/cnchou/glue, distinct from PyPI
`glue-core`/glueviz), which is NOT part of this repo's normal Python
environment -- only installed in a separate env on vader. Run this with
whichever interpreter/conda env has `glue.contrib.glue_analysis_dataframe`
importable, e.g.:
    conda activate <env-with-glue> && python manifold_capacity.py

Usage:
    python manifold_capacity.py [--subjID 1] [--lockType stim] [--voxRes 8mm]
                                 [--bands theta alpha beta lowgamma highgamma]
                                 [--rois visual parietal frontal]
                                 [--epochs stim delay] [--schemes 2 4 6 10]
                                 [--points_per_trial N] [--points_per_category N] [--zscore]
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
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from constants import (AMP_ONLY_BANDS, ROI_NAMES, get_bids_root, glue_fits_csv_path,
                        CATEGORY_SCHEMES, category_labels_for_scheme, balance_categories)
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

# Epoch windows (stim-locked only) -- matches intrinsic_dim_epochs.py exactly,
# so results from the two scripts stay directly comparable.
EPOCHS = {
    'stim':  (0.0, 0.2),
    'delay': (0.2, 1.7),
}


# ── Manifold construction ──────────────────────────────────────────────────

def build_manifolds(amp, tv, target_labels, t0, t1, scheme, zscore=False,
                     points_per_trial=None, points_per_category=None,
                     min_trials_per_class=2, seed=42, log=print):
    """
    amp: (n_trials, n_times, n_sources); tv: (n_times,);
    target_labels: (n_trials,) values in ANGLE_MAPPING's keys.
    scheme: category-grouping scheme (2, 4, 6, or 10 -- see
    constants.CATEGORY_SCHEMES).

    Every timepoint in [t0, t1] becomes its own point (no time-averaging) --
    each trial contributes n_epoch_timepoints points. points_per_trial, if
    given, evenly subsamples each trial's epoch window down to that many
    timepoints (spanning the full window, not just its start) before
    building manifolds -- see module docstring for why this matters (G04's
    200Hz storage rate means the delay epoch alone is ~301 timepoints/trial).

    Trials are mapped to this scheme's categories and balanced to the same
    trial count BEFORE the per-timepoint expansion above (see module
    docstring's SCHEME note) -- points_per_category defaults to None (auto:
    this subject's own smallest category trial count for this scheme).

    Returns (manifolds, kept_labels, n_epoch_timepoints) where manifolds is
    a list of (n_sources, n_points_for_that_category) arrays -- glue's
    expected (n_neurons, n_points) convention, the transpose of this repo's
    usual (n_trials, n_features) convention -- one per category in
    kept_labels (sorted category-name order). Returns ([], [], n_epoch_pts)
    if this scheme's smallest category has fewer than min_trials_per_class
    DISTINCT TRIALS (trial count, not point count -- see module docstring).
    """
    epoch_idx = np.where((tv >= t0) & (tv <= t1))[0]
    if epoch_idx.size == 0:
        raise ValueError(f'Epoch [{t0}, {t1}] has no timepoints in this time_vector.')

    if points_per_trial is not None and points_per_trial < epoch_idx.size:
        sub = np.linspace(0, epoch_idx.size - 1, points_per_trial).round().astype(int)
        epoch_idx = epoch_idx[sub]

    X = amp[:, epoch_idx, :]   # (n_trials, n_epoch_pts, n_sources)
    n_epoch_pts = X.shape[1]

    if zscore:
        flat = X.reshape(-1, X.shape[2])
        mu = flat.mean(axis=0)
        sd = flat.std(axis=0)
        sd[sd < 1e-10] = 1.0
        X = (X - mu) / sd

    group_labels, keep_mask = category_labels_for_scheme(target_labels, scheme)
    if group_labels.size == 0:
        log(f'    NOTE: scheme={scheme}: no trials map to any category -- skipping.')
        return [], [], n_epoch_pts

    _, counts = np.unique(group_labels, return_counts=True)
    natural_min = int(counts.min())
    ppc_used = points_per_category if points_per_category is not None else natural_min
    if ppc_used < min_trials_per_class:
        log(f'    NOTE: scheme={scheme}: points_per_category={ppc_used} '
            f'(natural min={natural_min}) < min_trials_per_class={min_trials_per_class} '
            f'-- skipping this scheme.')
        return [], [], n_epoch_pts

    try:
        balance_mask = balance_categories(group_labels, ppc_used, seed=seed)
    except ValueError as e:
        log(f'    NOTE: scheme={scheme}: {e} -- skipping this scheme.')
        return [], [], n_epoch_pts

    X_kept      = X[keep_mask][balance_mask]        # (n_kept_trials, n_epoch_pts, n_sources)
    labels_kept = group_labels[balance_mask]

    manifolds, kept_labels = [], []
    for cat in sorted(np.unique(labels_kept)):
        sel = labels_kept == cat
        pts = X_kept[sel].reshape(-1, X_kept.shape[2])   # (n_trials_cat * n_epoch_pts, n_sources)
        manifolds.append(pts.T)                          # (n_sources, n_points) -- glue's convention
        kept_labels.append(cat)

    return manifolds, kept_labels, n_epoch_pts


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='glue manifold-capacity analysis on stim-locked G04 '
                     'amplitude features, one subject x all bands x rois x epochs.')
    parser.add_argument('--subjID',   type=int, default=1)
    parser.add_argument('--lockType', default='stim', choices=['stim', 'resp'])
    parser.add_argument('--voxRes',   default='8mm')
    parser.add_argument('--bands',    nargs='+', default=list(AMP_ONLY_BANDS))
    parser.add_argument('--rois',     nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--epochs',   nargs='+', default=list(EPOCHS),
                         choices=list(EPOCHS))
    parser.add_argument('--schemes',  nargs='+', type=int, default=sorted(CATEGORY_SCHEMES),
                         choices=sorted(CATEGORY_SCHEMES),
                         help='Category-grouping schemes to test (see constants.CATEGORY_SCHEMES): '
                              '2=left/right, 4=quadrants, 6=quadrants+axis, 10=every raw location.')
    parser.add_argument('--analysis_type', default='ONE_VERSUS_REST')
    parser.add_argument('--n_hyperplanes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_trials_per_class', type=int, default=2)
    parser.add_argument('--points_per_trial', type=int, default=None,
                         help='Evenly subsample each trial epoch window to this many '
                              'timepoints (default: use every timepoint -- see module '
                              'docstring for point-count cost, especially for delay).')
    parser.add_argument('--points_per_category', type=int, default=None,
                         help='Fixed trial count per category, applied identically across '
                              'every scheme. Default: None (auto -- balance each scheme to '
                              'that subject\'s own smallest category trial count for THAT scheme).')
    parser.add_argument('--zscore', action='store_true',
                         help='Z-score each source across (trial, timepoint) samples '
                              'before building manifolds (default OFF -- see module docstring).')
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
    log = print

    log(f'manifold_capacity | sub-{args.subjID:02d} | {args.lockType} | {args.voxRes} | '
        f'bands={args.bands} | rois={args.rois} | epochs={args.epochs} | schemes={args.schemes} | '
        f'analysis_type={args.analysis_type} | n_hyperplanes={args.n_hyperplanes} | '
        f'seed={args.seed} | shuffle={not args.no_shuffle} | zscore={args.zscore} | '
        f'points_per_trial={args.points_per_trial or "all"} | '
        f'points_per_category={args.points_per_category or "auto"}')

    df_list = []

    for band in args.bands:
        for roi in args.rois:
            log(f'\n-- band={band} roi={roi} --')
            try:
                g04 = load_g04_band(args.subjID, args.lockType, band, args.voxRes,
                                     bids_root, want_phase=False, roi=roi)
            except (FileNotFoundError, ValueError) as e:
                log(f'  SKIP (load failed): {e}')
                continue

            amp = g04['amp']                     # (n_trials, n_times, n_sources)
            tv  = g04['time_vector']
            target_labels = g04['target_labels']

            log(f'  loaded: {amp.shape[0]} trials, {amp.shape[1]} timepoints, '
                f'{amp.shape[2]} sources')

            for epoch in args.epochs:
                t0, t1 = EPOCHS[epoch]
                log(f'  epoch={epoch} [{t0}, {t1}]')

                for scheme in args.schemes:
                    manifolds, kept_labels, n_epoch_pts = build_manifolds(
                        amp, tv, target_labels, t0, t1, scheme,
                        zscore=args.zscore, points_per_trial=args.points_per_trial,
                        points_per_category=args.points_per_category,
                        min_trials_per_class=args.min_trials_per_class,
                        seed=args.seed, log=log)

                    total_points = sum(m.shape[1] for m in manifolds)
                    log(f'    scheme={scheme}: {n_epoch_pts} timepoints/trial in this epoch -> '
                        f'{total_points} total points across {len(manifolds)} manifolds')

                    if len(manifolds) < 2:
                        log(f'    SKIP scheme={scheme}: only {len(manifolds)} usable manifold(s) '
                            f'(need >= 2 for a dichotomy).')
                        continue

                    log(f'    manifolds: P={len(manifolds)} labels={kept_labels}, '
                        f'points per manifold={[m.shape[1] for m in manifolds]}, '
                        f'n_features={manifolds[0].shape[0]}')

                    try:
                        ret = glue_analysis_dataframe(
                            manifolds,
                            indices=(args.subjID, band, roi, epoch, scheme),
                            indices_name=['subjID', 'band', 'roi', 'epoch', 'scheme'],
                            analysis_type=args.analysis_type,
                            n_hyperplanes=args.n_hyperplanes,
                            shuffle=not args.no_shuffle,
                            seed=args.seed,
                        )
                    except Exception:
                        log(f'    FAILED glue_analysis_dataframe for band={band} roi={roi} '
                            f'epoch={epoch} scheme={scheme}:')
                        log(traceback.format_exc())
                        continue

                    log(ret.to_string())
                    df_list.append(ret)

    if df_list:
        df_all = pd.concat(df_list)
        df_all.to_csv(csv_path)
        log(f'\nSaved combined results ({len(df_all)} rows): {csv_path}')
    else:
        log('\nNo results produced -- nothing to save.')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
