#!/usr/bin/env python3
"""
manifold_capacity_epochs.py

glue manifold-capacity on the four task epochs, with manifolds POOLED ACROSS
SUBJECTS: every subject contributes all of its trials for a location, and one
manifold per location is built from the whole pooled set.

    epochs   fixation -1.0..0.0 | stimulus 0.0..0.2
             early_delay 0.2..0.8 | late_delay 1.0..1.6   (half-open; the
             0.8-1.0 s gap is deliberate) -- identical to
             visual_geometry_epochs_cell.py, so the two analyses line up.
    scheme   4 quadrant manifolds (drops locations 1 and 6, i.e. 0 and 180 deg).
    rois     visual, parietal, frontal.

=============================================================================
POOLING ACROSS SUBJECTS -- why this is allowed here, and what it costs
=============================================================================
Every other cross-subject analysis in this repo refuses to pool raw trials,
because a column of subject A's source array is not the same thing as the same
column of subject B's. That objection is handled properly here rather than
waved away:

  * precompute_roi_splits.py stores, alongside each per-ROI cache, the
    TEMPLATE-GRID index of every ROI source in column order (inside_pos). Two
    subjects' columns refer to the same anatomical location exactly when those
    indices agree.
  * This script reads them, INTERSECTS the template indices across all
    subjects, and reindexes every subject onto that common, consistently
    ordered source set before anything is pooled. If the subjects already
    share an identical source list the intersection is a no-op; if they do
    not, the mismatch is reported and the non-shared sources are dropped
    rather than silently misaligned.
  * Each subject is z-scored across its OWN trials (per feature) before
    pooling. Without this, subjects with larger amplitude simply dominate the
    pooled manifolds and "capacity" would partly be measuring between-subject
    scale heterogeneity.

WHAT POOLING CHANGES, and how to read the result: each manifold now contains
trials from ~21 different brains, so BETWEEN-SUBJECT variability sits INSIDE
the manifolds and inflates their radius. Pooled capacity is therefore expected
to be LOWER than per-subject capacity, and the two are not comparable. The
meaningful contrast is pooled-REAL vs pooled-SHUFFLE (glue computes both), not
pooled-vs-per-subject. Read it as "is there a location code shared across
brains in a common anatomical space", which is the manifold-capacity analogue
of the inter-subject RDM correlation used elsewhere here.

COST: pooling multiplies points-per-manifold by ~n_subjects (from ~30 to
~600+), and glue's QP solves scale badly with point count -- an earlier sweep
in this repo ran 5+ hours before being killed for exactly that reason. Points
per manifold are therefore CAPPED (--points_per_category, default 200) with
the cap applied after pooling and reported in the output. Time one cell before
launching the grid.

Requires the `glue` package (github.com/cnchou/glue) importable -- activate the
env that has it first (e.g. `conda activate eegmne`); this script only calls
python3.

Usage:
    python manifold_capacity_epochs.py [--bands theta alpha beta]
        [--conditions ampOnly ampPhase] [--rois visual parietal frontal]
        [--phase_rois visual] [--schemes 4] [--voxRes 8mm]
        [--points_per_category 200] [--n_hyperplanes 200] [--seed 42]
        [--subjects 1 2 ...] [--outdir <path>] [--force]
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

from constants import (AMP_PHASE_BANDS, SUBJECT_LIST, get_bids_root,
                        CATEGORY_SCHEMES, category_labels_for_scheme, balance_categories)
from features import build_features
from io_g04 import load_g04_band
from visual_geometry_epochs_cell import EPOCHS, EPOCH_ORDER

try:
    from glue.contrib import glue_analysis_dataframe
except ImportError as e:
    raise ImportError(
        "Could not import glue.contrib.glue_analysis_dataframe -- this script needs "
        "the manifold-capacity-theory `glue` package (github.com/cnchou/glue), which "
        "lives in a separate environment, not this repo's normal Python env. "
        "Activate that environment first (e.g. `conda activate eegmne`)."
    ) from e

LOCK_TYPE = 'stim'
PHASE_CONDITIONS = ('ampPhase',)
DEFAULT_POINTS_PER_CATEGORY = 200


def output_csv_path(bids_root, voxRes, outdir=None):
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', 'glueDecoding', 'glueEpochsPooled')
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f'group_task-mgs_glueEpochsPooled_{LOCK_TYPE}_{voxRes}.csv')


def load_pooled(subjects, band, condition, roi, voxRes, bids_root, log=print):
    """
    Load every subject's ROI cache, align them onto a COMMON template-grid
    source set, z-score each subject's own trials, and return the pooled data.

    Returns (X_pooled (n_trials_total, n_times, n_common), tv, labels, subj_ids,
    info dict) or (None, ...) if nothing usable.
    """
    want_phase = condition in PHASE_CONDITIONS
    per_subj = []
    for s in subjects:
        try:
            g = load_g04_band(s, LOCK_TYPE, band, voxRes, bids_root,
                               want_phase=want_phase, roi=roi)
        except (FileNotFoundError, ValueError) as e:
            log(f'    sub-{s:02d}: SKIP ({e})')
            continue
        if g.get('inside_pos') is None:
            log(f'    sub-{s:02d}: SKIP -- ROI cache has no inside_pos, so its source '
                f'identity cannot be matched to other subjects. Rebuild the cache with '
                f'precompute_roi_splits.py to pool this subject.')
            continue
        per_subj.append((s, g))

    if len(per_subj) < 2:
        log(f'    only {len(per_subj)} usable subject(s) -- nothing to pool.')
        return None, None, None, None, None

    # --- common source set (template-grid indices), consistently ordered ---
    id_lists = [np.asarray(g['inside_pos']).astype(np.int64).ravel() for _, g in per_subj]
    common = id_lists[0]
    for ids in id_lists[1:]:
        common = np.intersect1d(common, ids)
    if common.size == 0:
        log('    no template-grid sources common to all subjects -- cannot pool.')
        return None, None, None, None, None
    identical = all(ids.size == common.size and np.array_equal(np.sort(ids), common)
                    for ids in id_lists)
    sizes = [ids.size for ids in id_lists]
    log(f'    sources: per-subject {min(sizes)}-{max(sizes)} | common {common.size} | '
        f'identical across subjects: {identical}')
    if not identical:
        log(f'    -> subjects do NOT share an identical source list; pooling on the '
            f'{common.size} shared template voxels and dropping the rest.')

    tv = per_subj[0][1]['time_vector']
    Xs, ys, subj_ids = [], [], []
    for s, g in per_subj:
        ids = np.asarray(g['inside_pos']).astype(np.int64).ravel()
        # Position of each common template id within THIS subject's columns.
        order = np.searchsorted(ids, common, sorter=np.argsort(ids))
        cols = np.argsort(ids)[order]

        X = build_features(condition, g['amp'], g['phase'] if want_phase else None)
        # build_features may double the feature axis (ampPhase = [real, imag]),
        # so the column map has to be applied per half, not once.
        n_src = ids.size
        if X.shape[2] == n_src:
            X = X[:, :, cols]
        elif X.shape[2] == 2 * n_src:
            X = np.concatenate([X[:, :, cols], X[:, :, n_src + cols]], axis=2)
        else:
            log(f'    sub-{s:02d}: SKIP -- unexpected feature count {X.shape[2]} '
                f'for {n_src} sources.')
            continue

        # ERP removal (grand trial-average, native resolution, before windowing).
        X = X - X.mean(axis=0, keepdims=True)
        # Per-subject z-score across its OWN trials, so no subject dominates the
        # pooled manifolds through sheer amplitude (see module docstring).
        sd = X.std(axis=0, keepdims=True)
        X = X / np.where(sd < 1e-10, np.asarray(1.0, dtype=sd.dtype), sd)

        Xs.append(X.astype(np.float32))
        ys.append(np.asarray(g['target_labels']).astype(int))
        subj_ids.append(np.full(X.shape[0], s, dtype=int))
        del g

    if len(Xs) < 2:
        return None, None, None, None, None

    X_pooled = np.concatenate(Xs, axis=0)
    y_pooled = np.concatenate(ys, axis=0)
    subj_pooled = np.concatenate(subj_ids, axis=0)
    info = dict(n_subjects=len(Xs), n_common_sources=int(common.size),
                identical_sources=bool(identical),
                n_features=int(X_pooled.shape[2]))
    log(f'    pooled: {X_pooled.shape[0]} trials from {len(Xs)} subjects, '
        f'{X_pooled.shape[2]} features')
    return X_pooled, tv, y_pooled, subj_pooled, info


def build_epoch_manifolds(X, tv, y, scheme, lo, hi, points_per_category, seed, log=print):
    """Epoch-average -> one point per trial -> one manifold per category."""
    mask = (tv >= lo) & (tv < hi)
    if not mask.any():
        raise ValueError(f'no timepoints in [{lo}, {hi})')
    P = X[:, mask, :].mean(axis=1)                      # (n_trials, n_features)

    group_labels, keep = category_labels_for_scheme(y, scheme)
    if group_labels.size == 0:
        return None, None, None
    P = P[keep]

    _, counts = np.unique(group_labels, return_counts=True)
    ppc = int(min(counts.min(), points_per_category)) if points_per_category else int(counts.min())
    if ppc < 2:
        return None, None, None
    bal = balance_categories(group_labels, ppc, seed=seed)
    P, labels = P[bal], group_labels[bal]

    manifolds, cats = [], []
    for cat in sorted(np.unique(labels)):
        manifolds.append(np.ascontiguousarray(P[labels == cat].T))   # (n_feat, n_points)
        cats.append(cat)
    return manifolds, cats, ppc


def main():
    ap = argparse.ArgumentParser(
        description='Epoch-based glue manifold capacity with manifolds pooled across subjects.')
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--phase_rois', nargs='+', default=['visual'],
                     help='ROIs for which ampPhase is run (default visual only -- '
                          'ampPhase doubles the feature count and the QP cost).')
    ap.add_argument('--schemes', nargs='+', type=int, default=[4],
                     choices=sorted(CATEGORY_SCHEMES))
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    ap.add_argument('--points_per_category', type=int, default=DEFAULT_POINTS_PER_CATEGORY,
                     help='CAP on points per manifold after pooling (default 200). '
                          'Pooling multiplies point count by ~n_subjects and glue QP '
                          'time scales badly with it. 0 = uncapped (use the pooled '
                          'minimum) -- time one cell first.')
    ap.add_argument('--n_hyperplanes', type=int, default=200)
    ap.add_argument('--analysis_type', default='ONE_VERSUS_REST')
    ap.add_argument('--no_shuffle', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    bids_root = get_bids_root()
    out_csv = output_csv_path(bids_root, args.voxRes, args.outdir)
    if os.path.exists(out_csv) and not args.force:
        print(f'SKIP (exists): {out_csv}  -- pass --force to overwrite.')
        return

    def log(m=''):
        print(m, flush=True)

    ppc_cap = args.points_per_category if args.points_per_category > 0 else None
    log(f'manifold_capacity_epochs (POOLED across subjects) | {args.voxRes} | '
        f'bands={args.bands} | conditions={args.conditions} | rois={args.rois} | '
        f'phase_rois={args.phase_rois} | schemes={args.schemes} | '
        f'epochs={list(EPOCH_ORDER)} | n_subjects={len(args.subjects)} | '
        f'points_per_category cap={ppc_cap} | n_hyperplanes={args.n_hyperplanes}')

    rows, t_run = [], time.time()
    for band in args.bands:
        for roi in args.rois:
            for condition in args.conditions:
                if condition in PHASE_CONDITIONS:
                    if roi not in args.phase_rois:
                        continue
                    if band not in AMP_PHASE_BANDS:
                        log(f'-- {band}/{roi}/{condition}: SKIP (no saved phase)')
                        continue

                log(f'\n-- band={band} roi={roi} condition={condition} --')
                X, tv, y, subj_ids, info = load_pooled(
                    args.subjects, band, condition, roi, args.voxRes, bids_root, log=log)
                if X is None:
                    continue

                for scheme in args.schemes:
                    for ep in EPOCH_ORDER:
                        lo, hi = EPOCHS[ep]
                        try:
                            manifolds, cats, ppc = build_epoch_manifolds(
                                X, tv, y, scheme, lo, hi, ppc_cap, args.seed, log=log)
                        except ValueError as e:
                            log(f'    SKIP {ep}: {e}')
                            continue
                        if manifolds is None or len(manifolds) < 2:
                            log(f'    SKIP {ep}: fewer than 2 usable manifolds.')
                            continue

                        t0 = time.time()
                        try:
                            ret = glue_analysis_dataframe(
                                manifolds,
                                indices=(band, condition, roi, scheme, ep),
                                indices_name=['band', 'condition', 'roi', 'scheme', 'epoch'],
                                analysis_type=args.analysis_type,
                                n_hyperplanes=args.n_hyperplanes,
                                shuffle=not args.no_shuffle,
                                seed=args.seed,
                            )
                        except Exception:
                            log(f'    FAILED glue for {band}/{roi}/{condition}/{ep}:')
                            log(traceback.format_exc())
                            continue

                        ret['t_start'] = lo
                        ret['t_stop'] = hi
                        ret['n_subjects'] = info['n_subjects']
                        ret['n_common_sources'] = info['n_common_sources']
                        ret['identical_sources'] = info['identical_sources']
                        ret['n_features'] = int(manifolds[0].shape[0])
                        ret['points_per_manifold'] = int(ppc)
                        ret['n_manifolds'] = len(manifolds)
                        ret['pooled'] = True
                        cap = ret['capacity'].to_numpy() if 'capacity' in ret else np.array([np.nan])
                        log(f'    {ep:12s} [{lo:+.2f},{hi:+.2f}) P={len(manifolds)} '
                            f'pts/manifold={ppc} F={manifolds[0].shape[0]} '
                            f'capacity={np.array2string(cap, precision=4)} '
                            f'({time.time() - t0:.1f}s)')
                        rows.append(ret)
                del X, y, subj_ids

    if not rows:
        log('\nNo results produced.')
        return
    df = pd.concat(rows)
    df.to_csv(out_csv)
    log(f'\nSaved ({len(df)} rows): {out_csv}  [total {time.time() - t_run:.1f}s]')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
