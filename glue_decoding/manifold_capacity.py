#!/usr/bin/env python3
"""
manifold_capacity.py

Runs glue's manifold-capacity analysis (glue.contrib.glue_analysis_dataframe)
on this project's stim-locked G04 amplitude features, for ONE subject across
all (band, roi, epoch) combinations. NOT part of the glue_decoding TGM
pipeline (run_glue_cell.py et al.) -- "glue" here refers to the separate
manifold-capacity-theory package (github.com/cnchou/glue), installed only
where noted below; the name collision with this directory is coincidental.

For each (band, roi, epoch):
    1. Load stim-locked G04 amplitude for one subject (want_phase=False --
       amplitude only, matches build_features's 'ampOnly' condition). Loaded
       once per (band, roi), shared across both epochs.
    2. Slice the epoch window (EPOCHS below -- stim=[0.0,0.2], delay=
       [0.2,1.7], matching intrinsic_dim_epochs.py) and average over time
       -> (n_trials, n_sources). NOTE: this time-averaging is a deliberate
       choice, not a glue requirement -- it means total points per cell is
       capped at n_trials (~154 for a full subject), which is LESS than
       every ROI's raw source count (179/501/597). glue_analysis internally
       QR-rotates onto the subspace spanned by the data, whose rank can
       never exceed min(n_features, total_points) -- so with time-averaged
       features, every ROI's EFFECTIVE dimensionality collapses to the same
       ~n_trials-dim space regardless of its raw source count. (Switching to
       per-timepoint, non-averaged points would let each ROI use its own
       full dimensionality, at the cost of many more -- autocorrelated --
       points and much slower QP solves; not done here, see chat history.)
    3. Z-score each source (feature) across trials (optional, default on --
       glue only mean-centers the global manifold origin internally, it
       doesn't rescale features, so leaving very different-magnitude
       sources unscaled can dominate the QR-rotated feature space).
    4. Split trials into one manifold per target location (1-10, see
       ANGLE_MAPPING in constants.py) -- glue's ONE_VERSUS_REST dichotomy
       set then tests each location against the rest of the "concept" set,
       matching the classic manifold-capacity-theory setup (P object
       manifolds in shared neural feature space). Locations with fewer
       than --min_trials_per_class trials in this epoch are dropped (with
       a warning) rather than passed in as a near-empty manifold.
    5. manifolds are (n_features, n_trials_for_that_target) arrays -- glue
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
                                 [--epochs stim delay]
                                 [--analysis_type ONE_VERSUS_REST]
                                 [--n_hyperplanes 200] [--seed 42]
                                 [--min_trials_per_class 2] [--no_zscore]
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

from constants import AMP_ONLY_BANDS, ANGLE_MAPPING, ROI_NAMES, get_bids_root, glue_fits_csv_path
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


TARGET_LABELS = sorted(ANGLE_MAPPING)   # [1, 2, ..., 10]

# Epoch windows (stim-locked only) -- matches intrinsic_dim_epochs.py exactly,
# so results from the two scripts stay directly comparable.
EPOCHS = {
    'stim':  (0.0, 0.2),
    'delay': (0.2, 1.7),
}


# ── Manifold construction ──────────────────────────────────────────────────

def build_manifolds(amp, tv, target_labels, t0, t1, zscore=True,
                     min_trials_per_class=2, log=print):
    """
    amp: (n_trials, n_times, n_sources); tv: (n_times,);
    target_labels: (n_trials,) values in ANGLE_MAPPING's keys.

    Returns (manifolds, kept_labels) where manifolds is a list of
    (n_sources, n_trials_for_that_label) arrays -- glue's expected
    (n_neurons, n_points) convention, the transpose of this repo's usual
    (n_trials, n_features) convention -- one per label in kept_labels
    (labels with < min_trials_per_class trials in this epoch are dropped).
    """
    mask = (tv >= t0) & (tv <= t1)
    if not mask.any():
        raise ValueError(f'Epoch [{t0}, {t1}] has no timepoints in this time_vector.')

    X = amp[:, mask, :].mean(axis=1)   # (n_trials, n_sources)

    if zscore:
        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, keepdims=True)
        sd[sd < 1e-10] = 1.0
        X = (X - mu) / sd

    manifolds, kept_labels = [], []
    for label in TARGET_LABELS:
        sel = target_labels == label
        n = int(sel.sum())
        if n < min_trials_per_class:
            log(f'    NOTE: target {label} has only {n} trial(s) in this epoch '
                f'(< min_trials_per_class={min_trials_per_class}) -- dropping this manifold.')
            continue
        manifolds.append(X[sel].T)   # (n_sources, n) -- glue's convention
        kept_labels.append(label)

    return manifolds, kept_labels


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
    parser.add_argument('--analysis_type', default='ONE_VERSUS_REST')
    parser.add_argument('--n_hyperplanes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_trials_per_class', type=int, default=2)
    parser.add_argument('--no_zscore', action='store_true',
                         help='Skip per-source z-scoring before building manifolds.')
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
        f'bands={args.bands} | rois={args.rois} | epochs={args.epochs} | '
        f'analysis_type={args.analysis_type} | n_hyperplanes={args.n_hyperplanes} | '
        f'seed={args.seed} | shuffle={not args.no_shuffle} | zscore={not args.no_zscore}')

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

                manifolds, kept_labels = build_manifolds(
                    amp, tv, target_labels, t0, t1,
                    zscore=not args.no_zscore,
                    min_trials_per_class=args.min_trials_per_class, log=log)

                if len(manifolds) < 2:
                    log(f'    SKIP: only {len(manifolds)} usable manifold(s) '
                        f'(need >= 2 for a dichotomy).')
                    continue

                log(f'    manifolds: P={len(manifolds)} labels={kept_labels}, '
                    f'points per manifold={[m.shape[1] for m in manifolds]}, '
                    f'n_features={manifolds[0].shape[0]}')

                try:
                    ret = glue_analysis_dataframe(
                        manifolds,
                        indices=(args.subjID, band, roi, epoch),
                        indices_name=['subjID', 'band', 'roi', 'epoch'],
                        analysis_type=args.analysis_type,
                        n_hyperplanes=args.n_hyperplanes,
                        shuffle=not args.no_shuffle,
                        seed=args.seed,
                    )
                except Exception:
                    log(f'    FAILED glue_analysis_dataframe for band={band} roi={roi} epoch={epoch}:')
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
