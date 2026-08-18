#!/usr/bin/env python3
"""
intrinsic_dim_pooled_epochs.py

Per-LOCATION manifold geometry, pooled across ALL subjects, over the four
task epochs (fixation / stimulus / early_delay / late_delay) -- the pooled
counterpart of intrinsic_dim_by_location_epochs_cell.py, built because a
single subject only has ~10-40 trials per location per epoch, which is too
few for either quantity below to be a reliable estimate. Pooling across
subjects the same way manifold_capacity_epochs.load_pooled already does for
GLUE (anatomically-aligned common source set, each subject z-scored on its
OWN trials before pooling so no subject dominates via raw amplitude scale,
epoch-averaged before pooling) routinely gives 200-800 trials per location
instead, which is what makes both measures usable.

Computes BOTH of Will Slatton's suggested GLUE sanity-check quantities per
stimulus-location manifold (10 per cell), not just participation ratio:

    participation ratio        PR  = Tr(Sigma)^2 / Tr(Sigma^2)
    normalized total variation NTV = sqrt(Tr(Sigma)) / ||mu||

Sigma is that location's own (shrinkage-regularized) covariance and mu is
its own mean vector -- PR compares to GLUE's "dimension", NTV compares to
GLUE's "radius" (overall manifold size relative to how far its center sits
from the origin). NOTE: the standard, dimensionally-consistent PR has a
SQUARED numerator; Will's screenshot appeared to show Tr(Sigma)/Tr(Sigma^2)
but that may just be the exponent being obscured by an overlapping icon.
Using the squared form here for consistency with every other PR in this
repo (visual_geometry_cell.py, intrinsic_dim_epochs.py,
intrinsic_dim_by_location_epochs_cell.py) -- worth checking against Will's
actual intended formula.

CONDITIONING: same fix as intrinsic_dim_by_location_epochs_cell.py, reused
unchanged -- PCA-project each epoch's pooled trial cloud to a shared basis
(fit once per band/roi/epoch on ALL locations' pooled trials together, via
visual_geometry_cell.pca_project), then Ledoit-Wolf shrink each location's
own residual covariance (visual_geometry_cell._ledoit_wolf_cov) before
computing PR or NTV. pca_project mean-centers across ALL pooled trials
before projecting, so in the projected space the pooled grand mean is
exactly 0 -- each location's own mean mu is then its displacement from that
pooled population average, which is the origin ||mu|| is measured from.

Unlike pooled GLUE capacity, there is no linear-separability (Cover) ceiling
to respect here -- PR/NTV are covariance-based, not a linear-classifier
fit, so pooling more trials per location is unambiguously beneficial with
no analogous cap on how many points a manifold can use.

Both amplitude-only and amplitude+phase conditions are covered:
'ampOnly' runs on every band; 'ampPhase' is restricted to the bands that
actually have phase saved (theta/alpha/beta -- no phase for lowgamma/
highgamma, matching manifold_capacity_epochs.py's AMP_PHASE_BANDS) and to
--phase_rois (default just 'visual', again matching manifold_capacity_
epochs.py, since that's the only ROI phase pooling was validated for).
build_features already produces the correct 2x ampPhase representation
([amp*cos(phase), amp*sin(phase)]), not a redundant 3x [amp, cos, sin] --
confirmed by reading features.py directly, so no fix was needed there.

Output: one long-format CSV row per (band, roi, condition, epoch,
location), written to derivatives/glueDecoding/intrinsicDimPooledEpochs/.
Bar-plot summaries (mean +/- SEM across the 10 locations, matching the
"average with error bars" convention used elsewhere in this repo) are
produced separately by plot_intrinsic_dim_pooled_epochs.py.

Parallelism: each (band, roi, condition) cell pools all subjects and does
its own PCA/shrinkage independently of every other cell, so cells are
embarrassingly parallel -- run via joblib, one process per cell (default
n_jobs = number of cells, capped by --n_jobs).

Usage:
    python intrinsic_dim_pooled_epochs.py
        [--subjects 1 2 ...] [--bands theta alpha beta lowgamma highgamma]
        [--rois visual parietal frontal] [--conditions ampOnly ampPhase]
        [--phase_rois visual] [--voxRes 8mm]
        [--max_pca_dim 50] [--min_trials 2] [--outdir <path>] [--force]
        [--n_jobs 8]
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
import pandas as pd
from joblib import Parallel, delayed

from constants import AMP_ONLY_BANDS, AMP_PHASE_BANDS, SUBJECT_LIST, ANGLE_MAPPING, get_bids_root
from visual_geometry_cell import (
    LOCATIONS, MAX_PCA_DIM, MIN_TRIALS_PER_LOC, pca_project, _ledoit_wolf_cov,
)
from visual_geometry_epochs_cell import EPOCH_ORDER
from manifold_capacity_epochs import load_pooled, LOCK_TYPE, PHASE_CONDITIONS


def output_csv_path(bids_root, voxRes, outdir=None):
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', 'glueDecoding', 'intrinsicDimPooledEpochs')
    os.makedirs(base, exist_ok=True)
    return os.path.join(
        base, f'group_task-mgs_intrinsicDimPooledEpochs_{LOCK_TYPE}_{voxRes}.csv')


def participation_ratio(lam):
    s1 = lam.sum()
    return float(s1 ** 2 / (lam ** 2).sum()) if s1 > 1e-30 else np.nan


def per_location_geometry(Xp, y, min_trials):
    """
    Xp: (n_trials, k) PCA-projected pooled trials for one (band, roi, epoch),
    already mean-zero across all pooled trials (pca_project centers before
    projecting). y: (n_trials,) location labels.

    Returns a dict of per-location arrays: n, pr, trace_sigma, mu_norm, ntv,
    shrinkage.
    """
    n_loc = len(LOCATIONS)
    out = {k: np.full(n_loc, np.nan) for k in
           ('pr', 'trace_sigma', 'mu_norm', 'ntv', 'shrinkage')}
    out['n'] = np.zeros(n_loc, int)
    for li, loc in enumerate(LOCATIONS):
        idx = np.where(y == loc)[0]
        out['n'][li] = idx.size
        if idx.size < min_trials:
            continue
        cloud = Xp[idx]
        mu = cloud.mean(axis=0)                 # displacement from pooled origin
        resid = cloud - mu
        Sigma, shrink = _ledoit_wolf_cov(resid)
        lam = np.maximum(np.linalg.eigvalsh(Sigma), 0.0)
        trace_sigma = float(lam.sum())
        mu_norm = float(np.linalg.norm(mu))
        out['pr'][li] = participation_ratio(lam)
        out['trace_sigma'][li] = trace_sigma
        out['mu_norm'][li] = mu_norm
        out['shrinkage'][li] = shrink
        out['ntv'][li] = (np.sqrt(trace_sigma) / mu_norm) if mu_norm > 1e-30 else np.nan
    return out


def run_cell(subjects, band, roi, condition, voxRes, bids_root, max_pca_dim,
             min_trials, log=print):
    P_pooled, y_pooled, subj_pooled, info = load_pooled(
        subjects, band, condition, roi, voxRes, bids_root, log=log)
    if P_pooled is None:
        log(f'  {band}/{roi}/{condition}: nothing to pool, skipping.')
        return []

    rows = []
    for ei, ep in enumerate(EPOCH_ORDER):
        Xe = P_pooled[:, ei, :]
        Xp, k, explained = pca_project(Xe, max_pca_dim)
        geo = per_location_geometry(Xp, y_pooled, min_trials)
        for li, loc in enumerate(LOCATIONS):
            rows.append(dict(
                band=band, roi=roi, condition=condition, epoch=ep, epoch_index=ei,
                location=loc, angle_deg=ANGLE_MAPPING[loc],
                n_pooled=int(geo['n'][li]), n_subjects=info['n_subjects'],
                n_features=info['n_common_sources'], pca_dim=k,
                explained_var=explained, shrinkage=geo['shrinkage'][li],
                pr=geo['pr'][li], trace_sigma=geo['trace_sigma'][li],
                mu_norm=geo['mu_norm'][li], ntv=geo['ntv'][li],
            ))
        pr_vals = geo['pr'][~np.isnan(geo['pr'])]
        ntv_vals = geo['ntv'][~np.isnan(geo['ntv'])]
        log(f'    {ep}: k={k} N/loc={np.median(geo["n"]):.0f}-'
            f'[{geo["n"].min()}-{geo["n"].max()}] | '
            f'PR mean={np.nanmean(pr_vals) if pr_vals.size else np.nan:.2f} | '
            f'NTV mean={np.nanmean(ntv_vals) if ntv_vals.size else np.nan:.3f}')
    return rows


def _build_cell_list(bands, rois, conditions, phase_rois):
    """
    (band, roi, condition) triples to run, respecting the same restrictions
    manifold_capacity_epochs.py applies: ampPhase only for bands that have
    saved phase (AMP_PHASE_BANDS) and only for --phase_rois.
    """
    cells = []
    for condition in conditions:
        for band in bands:
            if condition in PHASE_CONDITIONS and band not in AMP_PHASE_BANDS:
                continue
            for roi in rois:
                if condition in PHASE_CONDITIONS and roi not in phase_rois:
                    continue
                cells.append((band, roi, condition))
    return cells


def _run_cell_worker(cell, subjects, voxRes, bids_root, max_pca_dim, min_trials):
    band, roi, condition = cell
    tag = f'{band}/{roi}/{condition}'
    log = lambda msg: print(f'[{tag}] {msg}', flush=True)
    t1 = time.time()
    log('-- start --')
    try:
        rows = run_cell(subjects, band, roi, condition, voxRes, bids_root,
                         max_pca_dim, min_trials, log=log)
        log(f'done in {time.time() - t1:.1f}s ({len(rows)} rows)')
        return rows
    except Exception:
        log('FAILED:')
        traceback.print_exc()
        return []


def main():
    ap = argparse.ArgumentParser(
        description='Pooled (cross-subject) per-location participation ratio '
                    'and normalized total variation over the four task epochs.')
    ap.add_argument('--subjects', nargs='+', type=int, default=list(SUBJECT_LIST))
    ap.add_argument('--bands', nargs='+', default=list(AMP_ONLY_BANDS))
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    ap.add_argument('--phase_rois', nargs='+', default=['visual'],
                     help='ROIs to run ampPhase (or other PHASE_CONDITIONS) on; '
                          'ampOnly always runs on all --rois.')
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--max_pca_dim', type=int, default=MAX_PCA_DIM)
    ap.add_argument('--min_trials', type=int, default=MIN_TRIALS_PER_LOC)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--n_jobs', type=int, default=None,
                     help='Parallel workers (one per (band, roi, condition) cell). '
                          'Default: number of cells.')
    args = ap.parse_args()

    bids_root = get_bids_root()
    out_csv = output_csv_path(bids_root, args.voxRes, args.outdir)
    if not args.force and os.path.exists(out_csv):
        print(f'SKIP (exists, use --force to overwrite): {out_csv}')
        return

    cells = _build_cell_list(args.bands, args.rois, args.conditions, args.phase_rois)
    n_jobs = args.n_jobs if args.n_jobs else max(1, len(cells))

    print(f'intrinsic_dim_pooled_epochs | subjects={args.subjects} | '
          f'bands={args.bands} | rois={args.rois} | conditions={args.conditions} | '
          f'phase_rois={args.phase_rois} | {args.voxRes} | min_trials={args.min_trials} | '
          f'{len(cells)} cells | n_jobs={n_jobs}', flush=True)

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=5)(
        delayed(_run_cell_worker)(cell, args.subjects, args.voxRes, bids_root,
                                   args.max_pca_dim, args.min_trials)
        for cell in cells
    )
    all_rows = [row for rows in results for row in rows]

    if not all_rows:
        print('Nothing computed -- no output written.')
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f'Wrote {len(df)} rows -> {out_csv}')
    print(f'Done | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
