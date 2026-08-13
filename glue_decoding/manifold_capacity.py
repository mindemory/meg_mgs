#!/usr/bin/env python3
"""
manifold_capacity.py

Feasibility runner for glue's manifold-capacity analysis
(glue.contrib.glue_analysis_dataframe) on this project's stim-locked G04
amplitude features. NOT part of the glue_decoding TGM pipeline (run_glue_cell.py
et al.) -- "glue" here refers to the separate manifold-capacity-theory package
(github.com/cnchou/glue), installed only where noted below; the name collision
with this directory is coincidental.

For each (band, roi):
    1. Load stim-locked G04 amplitude for one subject (want_phase=False --
       amplitude only, matches build_features's 'ampOnly' condition).
    2. Slice the stimulus epoch [--epoch_start, --epoch_end] (default
       0.0-0.2 s, matching intrinsic_dim_epochs.py's 'stim' epoch) and
       average over time -> (n_trials, n_sources).
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
       the real dichotomy geometry and a shuffled-points null in one call
       (mirrors the (name, shuffle, seed) MultiIndex shown in the plan).

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
                                 [--epoch_start 0.0] [--epoch_end 0.2]
                                 [--analysis_type ONE_VERSUS_REST]
                                 [--n_hyperplanes 200] [--seed 42]
                                 [--min_trials_per_class 2] [--no_zscore]
                                 [--no_shuffle] [--outdir <path>]
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

from constants import AMP_ONLY_BANDS, ANGLE_MAPPING, ROI_NAMES, get_bids_root
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
        description='Feasibility run of glue manifold-capacity analysis on '
                     'stim-locked G04 amplitude features.')
    parser.add_argument('--subjID',   type=int, default=1)
    parser.add_argument('--lockType', default='stim', choices=['stim', 'resp'])
    parser.add_argument('--voxRes',   default='8mm')
    parser.add_argument('--bands',    nargs='+', default=list(AMP_ONLY_BANDS))
    parser.add_argument('--rois',     nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--epoch_start', type=float, default=0.0)
    parser.add_argument('--epoch_end',   type=float, default=0.2)
    parser.add_argument('--analysis_type', default='ONE_VERSUS_REST')
    parser.add_argument('--n_hyperplanes', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_trials_per_class', type=int, default=2)
    parser.add_argument('--no_zscore', action='store_true',
                         help='Skip per-source z-scoring before building manifolds.')
    parser.add_argument('--no_shuffle', action='store_true',
                         help='Skip the shuffled-manifolds null (glue_analysis_dataframe shuffle=False).')
    parser.add_argument('--outdir', default=None,
                         help='Directory for the log + results CSV. '
                              'Default: <bids_root>/derivatives/glueDecoding/capacity')
    args = parser.parse_args()

    bids_root = get_bids_root()
    outdir = args.outdir or os.path.join(bids_root, 'derivatives', 'glueDecoding', 'capacity')
    os.makedirs(outdir, exist_ok=True)

    tag = f'sub-{args.subjID:02d}_{args.lockType}_{args.voxRes}'
    log_path = os.path.join(outdir, f'glue_capacity_{tag}.log')
    csv_path = os.path.join(outdir, f'glue_capacity_{tag}.csv')

    log_fh = open(log_path, 'w')

    def log(msg=''):
        print(msg)
        print(msg, file=log_fh, flush=True)

    log(f'manifold_capacity | sub-{args.subjID:02d} | {args.lockType} | {args.voxRes} | '
        f'bands={args.bands} | rois={args.rois} | epoch=[{args.epoch_start}, {args.epoch_end}] | '
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

            manifolds, kept_labels = build_manifolds(
                amp, tv, target_labels, args.epoch_start, args.epoch_end,
                zscore=not args.no_zscore,
                min_trials_per_class=args.min_trials_per_class, log=log)

            if len(manifolds) < 2:
                log(f'  SKIP: only {len(manifolds)} usable manifold(s) '
                    f'(need >= 2 for a dichotomy).')
                continue

            log(f'  manifolds: P={len(manifolds)} labels={kept_labels}, '
                f'points per manifold={[m.shape[1] for m in manifolds]}, '
                f'n_features={manifolds[0].shape[0]}')

            try:
                ret = glue_analysis_dataframe(
                    manifolds,
                    indices=(args.subjID, band, roi),
                    indices_name=['subjID', 'band', 'roi'],
                    analysis_type=args.analysis_type,
                    n_hyperplanes=args.n_hyperplanes,
                    shuffle=not args.no_shuffle,
                    seed=args.seed,
                )
            except Exception:
                log(f'  FAILED glue_analysis_dataframe for band={band} roi={roi}:')
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

    log(f'Log saved: {log_path}')
    log_fh.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
