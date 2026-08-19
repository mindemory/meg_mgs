#!/usr/bin/env python3
"""
ccgp_epochs_cell.py

CCGP (cross-condition generalization performance) and shattering
dimensionality per subject, over the four task epochs -- the abstraction /
dimensionality counterpart to the RDM+MDS geometry in
visual_geometry_epochs_cell.py.

WHY THIS SUPPLEMENTS THE RING RESULT
The MDS ring exists only as a GROUP average, and its inter-subject RDM
correlation is ~0.02-0.07: individuals barely agree, and the group ring is a
weak shared component recovered by averaging 21 noisy RDMs. CCGP and
shattering dimensionality are computed PER SUBJECT and cross-validated, so a
per-subject CCGP above its own null is materially stronger evidence for a
planar location code than anything the group MDS can provide. That is the
point of running this.

THE DESIGN -- this angle set contains a hidden 2x2x2 factorial
Dropping the two meridian locations (0 and 180 deg, where sin = 0 so "top vs
bottom" is undefined) leaves EIGHT locations that cross three binary
variables with EXACTLY ONE location per cell:

    horizontal  right {25,50,310,335}    vs  left   {130,155,205,230}
    vertical    top   {25,50,130,155}    vs  bottom {205,230,310,335}
    axis        nearH {25,155,205,335}   vs  nearV  {50,130,230,310}

("axis" = whether the location lies nearer the horizontal or the vertical
meridian; the two locations in each quadrant sit 25 and 40 deg from their
nearer axis.) That is the Bernardi et al. condition structure, which is what
CCGP and shattering dimensionality are defined on -- decodanda's shattering
dimensionality accepts BINARY variables only and returns a single dichotomy
for a 10-valued 'location' variable, so the factorial is not optional
packaging, it is what makes the analysis runnable at all.

WHAT EACH MEASURE TESTS, AND WHAT A RING PREDICTS
  CCGP(horizontal): train the left/right decoder on one value of the other
      variables, test on the held-out ones. High = the horizontal coding
      direction is the same regardless of vertical position, i.e. the code is
      FACTORIZED rather than a per-location lookup table.
  shattering dimensionality: decodability averaged over all 35 balanced
      dichotomies of the 8 conditions. High = many arbitrary groupings are
      linearly separable, i.e. high embedding dimensionality.

A clean 2-D ring makes a SPECIFIC joint prediction, verified here by
simulation at realistic SNR (40 trials/location, 40 features, noise 0.6):

                       CCGP horiz   CCGP vert   CCGP axis   SD frac>0.7
    2-D ring              1.00        0.99        0.52         0.23
    high-dimensional      0.54        0.43        0.49         1.00
    (shuffled null)       ~0.51       ~0.51       ~0.51         --

Note the THIRD variable is a built-in negative control: "near-horizontal vs
near-vertical" is not a linear coordinate on the ring plane, so a genuine
planar ring puts it at chance BY CONSTRUCTION. High CCGP on horizontal and
vertical TOGETHER WITH chance on axis is a far more specific signature than
either number alone -- a code that scored high on all three would not be a
ring.

Purely geometric reference (noiseless, from linear-separability of the 8
angles in a plane): only 4 of the 35 balanced dichotomies are separable, i.e.
SD = 0.114. The simulated 0.23 above is higher because finite trials and
noise let near-separable dichotomies decode above threshold, so the
noise-matched simulation, not the geometric bound, is the fair reference.

CONDITIONING: features are PCA-projected to a fixed k before decoding. This
does double duty -- it keeps the classifier out of the n_trials << n_features
regime, and it EQUALIZES CLASSIFIER CAPACITY ACROSS ROIs. Shattering
dimensionality rises with the dimensionality available to the decoder, and
the ROIs differ ~3x in source count (visual ~597, parietal ~501, frontal
~179), so without this an ROI would look higher-dimensional simply for being
bigger. Projecting every ROI to the same k removes that. (It does not
equalize the VARIANCE retained -- k=50 keeps more of frontal's structure than
of visual's -- so this controls the capacity confound, not every difference
between areas.)

Requires the `decodanda` package (github.com/lposani/decodanda), which lives
in its own environment -- NOT the eegmne env that has `glue`. Activate it
first (e.g. `conda activate decodanda`).

Usage:
    python ccgp_epochs_cell.py <subjID>
        [--bands theta alpha beta] [--conditions ampOnly ampPhase]
        [--rois visual parietal frontal] [--voxRes 8mm]
        [--max_pca_dim 50] [--n_shuffles 25] [--seed 0]
        [--outdir <path>] [--force]
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
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')          # decodanda imports pyplot; keep it headless

from constants import AMP_PHASE_BANDS, ANGLE_MAPPING, get_bids_root
from features import build_features
from io_g04 import load_g04_band
from visual_geometry_cell import (
    MAX_PCA_DIM, epoch_average, zscore_per_timepoint, pca_project,
)
from visual_geometry_epochs_cell import EPOCHS, EPOCH_ORDER

LOCK_TYPE = 'stim'

# The three binary variables and their level names, in decodanda's format.
CONDITIONS = {
    'horizontal': ['left', 'right'],
    'vertical':   ['bottom', 'top'],
    'axis':       ['nearV', 'nearH'],
}
DICHOTOMIES = tuple(CONDITIONS)

# Reference values (see module docstring). GEOMETRIC_RING_SD is the noiseless
# separability bound; the simulated, noise-matched figure is higher.
GEOMETRIC_RING_SD = 4 / 35


def _get_decodanda():
    """
    Lazy import, same reasoning as manifold_capacity_epochs._get_glue_...:
    decodanda lives in its own env, and the pure-numpy helpers here (the
    factorial construction, the PCA conditioning) are useful without it.
    """
    try:
        from decodanda import Decodanda
    except ImportError as e:
        raise ImportError(
            "Could not import decodanda -- this script needs the CCGP package "
            "(github.com/lposani/decodanda), which lives in its own environment "
            "and NOT in the eegmne env that carries `glue` (installing it there "
            "hits a numpy/sklearn ABI mismatch). Activate that env first, e.g. "
            "`conda activate decodanda`."
        ) from e
    return Decodanda


def output_path(bids_root, subjID, band, condition, roi, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'ccgpEpochs')
    os.makedirs(base, exist_ok=True)
    return os.path.join(
        base, f'{subName}_task-mgs_ccgp_{condition}_{band}_{roi}_{voxRes}.npz')


def factorial_labels(y):
    """
    (keep_mask, {variable: label array}) for the 2x2x2 described in the module
    docstring. Trials at the two meridian locations (sin == 0, so 'vertical'
    is undefined) are dropped rather than assigned arbitrarily.
    """
    ang = np.array([ANGLE_MAPPING[int(v)] for v in y], dtype=float)
    x, yy = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    keep = np.abs(yy) > 1e-9
    # Angular distance to the nearer horizontal meridian; the two locations in
    # each quadrant sit 25 and 40 deg from it, so 45 splits them cleanly.
    d_h = np.minimum(ang % 180.0, 180.0 - (ang % 180.0))
    labels = {
        'horizontal': np.where(x > 0, 'right', 'left')[keep],
        'vertical':   np.where(yy > 0, 'top', 'bottom')[keep],
        'axis':       np.where(d_h < 45.0, 'nearH', 'nearV')[keep],
    }
    return keep, labels


def run_epoch(Xe, y, max_pca_dim, n_shuffles, seed, log=print):
    """
    CCGP + shattering dimensionality for one epoch's (n_trials, n_features).
    Returns a dict, or None if the cell is not runnable.
    """
    Decodanda = _get_decodanda()
    keep, labels = factorial_labels(y)
    if keep.sum() < 16:
        log(f'      only {int(keep.sum())} non-meridian trials -- skipping')
        return None
    Xp, k, explained = pca_project(Xe[keep], max_pca_dim)

    session = {'raster': Xp, 'trial': np.arange(Xp.shape[0]), **labels}
    dec = Decodanda(data=session, conditions=CONDITIONS, verbose=False)

    out = dict(pca_dim=k, explained_var=explained, n_trials_used=int(keep.sum()))

    ccgp, ccgp_null = dec.CCGP(resamplings=5, nshuffles=n_shuffles, plot=False)
    for d in DICHOTOMIES:
        nl = np.asarray(ccgp_null.get(d, []), dtype=float).ravel()
        out[f'ccgp_{d}'] = float(np.mean(ccgp[d]))
        out[f'ccgp_{d}_null_mean'] = float(nl.mean()) if nl.size else np.nan
        out[f'ccgp_{d}_null_sd'] = float(nl.std()) if nl.size else np.nan
        # Permutation p against the geometric null: CCGP chance is NOT 0.5
        # (correlated conditions push it up), so significance has to be read
        # against this null rather than against a nominal 0.5.
        out[f'ccgp_{d}_p'] = (float((np.sum(nl >= out[f'ccgp_{d}']) + 1) / (nl.size + 1))
                              if nl.size else np.nan)

    sd_scalar, sd_acc, sd_null = dec.shattering_dimensionality(
        cross_validations=10, nshuffles=max(2, n_shuffles // 3), visualize=False)
    acc = np.array(list(sd_acc.values()), dtype=float)
    out['sd_frac_significant'] = float(sd_scalar)
    out['sd_mean_acc'] = float(acc.mean())
    out['sd_frac_above_0.7'] = float((acc > 0.7).mean())
    out['sd_n_dichotomies'] = int(acc.size)
    out['sd_acc'] = acc
    out['sd_dichotomy_keys'] = np.array(list(sd_acc.keys()), dtype=str)
    return out


def build_cell_list(bands, conditions, rois):
    """(band, condition, roi) triples to run. ampPhase is dropped for bands with
    no saved phase (constants.AMP_PHASE_BANDS)."""
    cells = []
    for band in bands:
        for condition in conditions:
            if condition == 'ampPhase' and band not in AMP_PHASE_BANDS:
                continue
            for roi in rois:
                cells.append((band, condition, roi))
    return cells


def _cell_worker(cell, subjID, voxRes, bids_root, max_pca_dim, n_shuffles, seed,
                 outdir, force):
    """
    One (band, condition, roi) cell, all four epochs. Cells load their own G04
    and share nothing, so they are embarrassingly parallel; the four epochs stay
    INSIDE a cell because they reuse that cell's single loaded/z-scored array
    (splitting them across processes would reload it four times).
    """
    band, condition, roi = cell
    tag = f'{band}/{condition}/{roi}'
    out_path = output_path(bids_root, subjID, band, condition, roi, voxRes, outdir)
    if not force and os.path.exists(out_path):
        print(f'[{tag}] SKIP (exists)', flush=True)
        return
    want_phase = (condition == 'ampPhase')
    t0 = time.time()
    try:
        g04 = load_g04_band(subjID, LOCK_TYPE, band, voxRes, bids_root,
                             want_phase=want_phase, roi=roi)
        X = build_features(condition, g04['amp'], g04['phase'] if want_phase else None)
        tv = g04['time_vector']
        y = g04['target_labels'].astype(int)
        del g04
        X = zscore_per_timepoint(X)

        per_epoch = {}
        for ep in EPOCH_ORDER:
            lo, hi = EPOCHS[ep]
            Xe, _ = epoch_average(X, tv, lo, hi, hi_inclusive=False)
            r = run_epoch(Xe, y, max_pca_dim, n_shuffles, seed,
                          log=lambda m: print(f'[{tag}] {m}', flush=True))
            if r is not None:
                per_epoch[ep] = r
        if not per_epoch:
            print(f'[{tag}] SKIP: no runnable epoch', flush=True)
            return

        save = dict(
            epochs=np.array(EPOCH_ORDER),
            epoch_bounds=np.array([EPOCHS[e] for e in EPOCH_ORDER]),
            dichotomies=np.array(DICHOTOMIES),
            geometric_ring_sd=np.array([GEOMETRIC_RING_SD]),
            n_trials=np.array([X.shape[0]]), n_features=np.array([X.shape[2]]),
            subjID=np.array([subjID]), band=np.array([band]),
            condition=np.array([condition]), roi=np.array([roi]),
            voxRes=np.array([voxRes]), seed=np.array([seed]),
            n_shuffles=np.array([n_shuffles]),
        )
        for ep, r in per_epoch.items():
            for key, val in r.items():
                save[f'{ep}__{key}'] = val
        np.savez_compressed(out_path, **save)

        # Report CCGP MINUS ITS OWN NULL, not raw CCGP: chance here is not 0.5
        # and drifts with the condition correlations, so a raw number cannot be
        # judged by eye. SD is reported as the null-based significant fraction,
        # which adapts to the noise level; sd_frac_above_0.7 saturates at 0.00
        # at real MEG SNR and says nothing.
        msg = '  '.join(
            f'{ep}: ' + ' '.join(
                f'{d[:4]}={per_epoch[ep][f"ccgp_{d}"] - per_epoch[ep][f"ccgp_{d}_null_mean"]:+.3f}'
                for d in DICHOTOMIES)
            + f' SDsig={per_epoch[ep]["sd_frac_significant"]:.2f}'
            + f' SDacc={per_epoch[ep]["sd_mean_acc"]:.3f}'
            for ep in EPOCH_ORDER if ep in per_epoch)
        print(f'[{tag}] {msg} | {time.time() - t0:.1f}s', flush=True)
        del X
    except (FileNotFoundError, ValueError) as e:
        print(f'[{tag}] SKIP: {e}', flush=True)
    except Exception:
        print(f'[{tag}] FAILED:', flush=True)
        traceback.print_exc()


def run_cell(subjID, bands, conditions, rois, voxRes, bids_root,
             max_pca_dim=MAX_PCA_DIM, n_shuffles=25, seed=0,
             outdir=None, force=False, n_jobs=None):
    cells = build_cell_list(bands, conditions, rois)
    n_jobs = n_jobs if n_jobs else max(1, len(cells))
    print(f'{len(cells)} cells | n_jobs={n_jobs}', flush=True)
    Parallel(n_jobs=n_jobs, prefer='processes', verbose=5)(
        delayed(_cell_worker)(c, subjID, voxRes, bids_root, max_pca_dim,
                              n_shuffles, seed, outdir, force) for c in cells)


def main():
    ap = argparse.ArgumentParser(
        description='Per-subject CCGP + shattering dimensionality over the four epochs.')
    ap.add_argument('subjID', type=int)
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--max_pca_dim', type=int, default=MAX_PCA_DIM,
                     help='Features are PCA-projected to this many dims before '
                          'decoding. Also the cross-ROI capacity control: shattering '
                          'dimensionality grows with the dimensionality available to '
                          'the classifier, and the ROIs differ ~3x in source count.')
    ap.add_argument('--n_shuffles', type=int, default=25,
                     help='Geometric-null shuffles for CCGP. CCGP chance is NOT 0.5, '
                          'so this null is what significance is read against.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_jobs', type=int, default=None,
                     help='Parallel workers, one per (band, condition, roi) cell. '
                          'Default: one per cell. Set to 1 when a runner is already '
                          'parallelising over SUBJECTS, or the two multiply.')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    bids_root = get_bids_root()
    print(f'ccgp_epochs_cell | sub-{args.subjID:02d} | bands={args.bands} | '
          f'conditions={args.conditions} | rois={args.rois} | {args.voxRes} | '
          f'k={args.max_pca_dim} | nshuffles={args.n_shuffles}', flush=True)
    t0 = time.time()
    run_cell(args.subjID, list(args.bands), list(args.conditions), list(args.rois),
             args.voxRes, bids_root, max_pca_dim=args.max_pca_dim,
             n_shuffles=args.n_shuffles, seed=args.seed,
             outdir=args.outdir, force=args.force, n_jobs=args.n_jobs)
    print(f'Done | sub-{args.subjID:02d} | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
