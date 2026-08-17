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
    scheme   10 -- ONE MANIFOLD PER LOCATION, all ten. Note this differs from
             the quadrant variant (scheme 4) in two ways that matter: locations
             1 and 6 (0 and 180 deg) are INCLUDED here rather than dropped, and
             with P=10 rather than P=4 there are ~2.5x fewer pooled trials per
             manifold (a manifold is now one location, not a pair) and 10
             one-versus-rest dichotomies per fit instead of 4.
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

POINTS PER MANIFOLD -- there is a HARD CEILING, and it is not a compute budget.
glue requires the manifolds to be LINEARLY SEPARABLE and raises
"Data is not linearly separable" otherwise. By Cover's theorem, points in
general position admit a separating homogeneous hyperplane only while the TOTAL
point count satisfies

        P * M  <=  2 * n_features        (P manifolds, M points each)

so M <= 2*F/P no matter how many subjects are pooled. Measured on real data
(visual, F=597, P=10 => 2F/P = 119): M=50 and M=100 fitted fine, M=200 raised
the separability error. This script therefore caps M at COVER_SAFETY (80%) of
that bound automatically, and reports the ceiling per cell.

The practical consequence: pooling still helps a lot -- it lifts M from ~30
(one subject) to the ceiling -- but NOT without limit, and past the ceiling
extra pooled trials are simply unusable. The only ways to raise the ceiling are
a larger ROI (more features) or a coarser scheme (fewer manifolds): at P=10 the
visual ROI allows ~119 points, while the P=4 quadrant scheme allows ~298. Small
ROIs bind much harder -- check each ROI's source count (inspect_data_summary.py)
before assuming a cell is runnable at P=10.

--benchmark times one real fit across sizes up to the ceiling and prints the
measured cost scaling; glue_with_retry additionally steps the point count down
if a cell turns out non-separable below the bound anyway.

Requires the `glue` package (github.com/cnchou/glue) importable -- activate the
env that has it first (e.g. `conda activate eegmne`); this script only calls
python3.

Usage:
    python manifold_capacity_epochs.py [--bands theta alpha beta]
        [--conditions ampOnly ampPhase] [--rois visual parietal frontal]
        [--phase_rois visual] [--schemes 10] [--voxRes 8mm]
        [--points_per_category 0] [--n_hyperplanes 200] [--seed 42]
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

# Fraction of the Cover bound to actually use. glue REQUIRES the manifolds to be
# linearly separable and raises RuntimeError otherwise; by Cover's theorem points
# in general position admit a separating homogeneous hyperplane only while the
# TOTAL number of points P*M stays under 2*n_features. Measured on real data
# (visual, N=597, P=10, so 2N/P = 119): M=50 and M=100 fitted, M=200 raised
# "Data is not linearly separable". Staying at 80% of the bound leaves room for
# the data not being in perfectly general position.
COVER_SAFETY = 0.8
# UNCAPPED by default. Maximising points per manifold is the entire reason to
# pool subjects in the first place, so the default must not throw that away.
# The flag remains for the case where a measured QP time turns out prohibitive
# -- use --benchmark to find out rather than guessing.
DEFAULT_POINTS_PER_CATEGORY = 0


def cover_max_points(n_features, n_manifolds, safety=COVER_SAFETY):
    """
    Largest points-per-manifold that keeps the pooled set linearly separable,
    P*M <= 2*n_features (Cover), scaled by `safety`. This is a HARD ceiling from
    the theory, not a compute budget: exceeding it makes glue raise rather than
    run slowly, and no amount of extra pooled data can get past it -- only more
    features (a bigger ROI) or fewer manifolds (a coarser scheme) can.
    """
    return max(2, int(safety * 2.0 * n_features / max(1, n_manifolds)))


def glue_with_retry(manifolds, log, retry_seed=42, min_points=20, **kwargs):
    """
    Call glue, and if it rejects the data as not linearly separable, subsample
    the points and try again rather than losing the cell. Returns
    (result_df, points_used) or (None, points_used_at_failure).

    The Cover bound is applied up front, so this is a safety net for data that
    is not in general position (e.g. near-duplicate trials), not the main
    mechanism.
    """
    # NB: named retry_seed, not seed -- `seed` is forwarded to glue via **kwargs
    # and having both would be a duplicate-keyword TypeError.
    rng = np.random.default_rng(retry_seed)
    cur = [np.asarray(m) for m in manifolds]
    while True:
        m_pts = cur[0].shape[1]
        try:
            return glue_analysis_dataframe(cur, **kwargs), m_pts
        except RuntimeError as e:
            if 'not linearly separable' not in str(e).lower() or m_pts <= min_points:
                raise
            new_pts = max(min_points, int(m_pts * 0.7))
            log(f'      not linearly separable at {m_pts} pts/manifold '
                f'-> retrying at {new_pts}')
            cur = [m[:, rng.choice(m.shape[1], new_pts, replace=False)] for m in cur]


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


def run_benchmark(args, bids_root, ppc_cap, log):
    """
    Time ONE real glue fit at increasing points-per-manifold on the first
    runnable cell, so the cost of running uncapped is measured rather than
    assumed. Prints elapsed time and the implied scaling exponent between
    successive sizes (t ~ M^k), plus a projection of the full grid.
    """
    for band in args.bands:
        for roi in args.rois:
            for condition in args.conditions:
                if condition in PHASE_CONDITIONS and (
                        roi not in args.phase_rois or band not in AMP_PHASE_BANDS):
                    continue
                log(f'\nBENCHMARK cell: band={band} roi={roi} condition={condition}')
                X, tv, y, _, info = load_pooled(
                    args.subjects, band, condition, roi, args.voxRes, bids_root, log=log)
                if X is None:
                    continue
                scheme = args.schemes[0]
                lo, hi = EPOCHS['early_delay']
                _, _, ppc_max = build_epoch_manifolds(X, tv, y, scheme, lo, hi,
                                                       None, args.seed, log=log)
                n_man = len(CATEGORY_SCHEMES[scheme]['groups'])
                cover = cover_max_points(info['n_features'], n_man)
                hard = int(2.0 * info['n_features'] / n_man)
                usable = min(int(ppc_max), cover)
                log(f'  pooled data allows {ppc_max} pts/manifold, but glue requires the '
                    f'manifolds to be LINEARLY SEPARABLE:')
                log(f'    P={n_man} manifolds, F={info["n_features"]} features '
                    f'-> Cover bound P*M <= 2F gives M <= {hard}; '
                    f'using {int(COVER_SAFETY*100)}% => {cover}')
                log(f'    so the usable size is {usable}, NOT {ppc_max} -- pooling past '
                    f'this point cannot help; only a larger ROI (more features) or a '
                    f'coarser scheme (fewer manifolds) can.')
                sizes = sorted({s for s in (50, 100, 200) if s < usable} | {usable})
                log(f'  timing sizes {sizes}')
                log(f'\n  {"pts/manifold":>13s} {"seconds":>9s} {"scaling t~M^k":>14s}')
                log('  ' + '-' * 40)
                prev = None
                for m in sizes:
                    manifolds, _, used = build_epoch_manifolds(
                        X, tv, y, scheme, lo, hi, m, args.seed, log=log)
                    t0 = time.time()
                    try:
                        glue_analysis_dataframe(
                            manifolds, indices=(band, condition, roi, scheme, 'bench'),
                            indices_name=['band', 'condition', 'roi', 'scheme', 'epoch'],
                            analysis_type=args.analysis_type,
                            n_hyperplanes=args.n_hyperplanes,
                            shuffle=False, seed=args.seed)
                    except RuntimeError as e:
                        if 'not linearly separable' in str(e).lower():
                            log(f'  {used:13d} {"NOT SEP.":>9s}   <- separability limit '
                                f'reached (Cover bound was {cover})')
                        else:
                            log(f'  {used:13d} {"FAILED":>9s}'); log(traceback.format_exc())
                        break
                    except Exception:
                        log(f'  {used:13d} {"FAILED":>9s}')
                        log(traceback.format_exc())
                        break
                    dt = time.time() - t0
                    k = ''
                    if prev is not None and prev[1] > 0 and used > prev[0]:
                        k = f'{np.log(dt / prev[1]) / np.log(used / prev[0]):14.2f}'
                    log(f'  {used:13d} {dt:9.1f} {k:>14s}')
                    prev = (used, dt)
                if prev is not None:
                    n_cells = (len(args.bands) * len(args.rois) *
                               len([c for c in args.conditions if c not in PHASE_CONDITIONS])
                               + len(args.bands) * len(args.phase_rois) *
                               len([c for c in args.conditions if c in PHASE_CONDITIONS]))
                    n_fits = n_cells * len(EPOCH_ORDER) * len(args.schemes)
                    log(f'\n  At the largest size, one fit took {prev[1]:.1f}s.')
                    log(f'  Full grid is ~{n_fits} fits (x2 with shuffle) => roughly '
                        f'{n_fits * 2 * prev[1] / 3600:.1f} h serial, '
                        f'{n_fits * 2 * prev[1] / 3600 / max(1, len(args.bands) * len(args.rois)):.1f} h '
                        f'at the runner\'s (band,roi) parallelism.')
                return
    log('No runnable cell found for benchmarking.')


def main():
    ap = argparse.ArgumentParser(
        description='Epoch-based glue manifold capacity with manifolds pooled across subjects.')
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--phase_rois', nargs='+', default=['visual'],
                     help='ROIs for which ampPhase is run (default visual only -- '
                          'ampPhase doubles the feature count and the QP cost).')
    ap.add_argument('--schemes', nargs='+', type=int, default=[10],
                     choices=sorted(CATEGORY_SCHEMES),
                     help='Default 10 = one manifold per location (all ten, including '
                          '0 and 180 deg). 4 = quadrants, which drops those two.')
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    ap.add_argument('--points_per_category', type=int, default=DEFAULT_POINTS_PER_CATEGORY,
                     help='Cap on points per manifold after pooling. DEFAULT 0 = '
                          'UNCAPPED, i.e. use every pooled trial up to the smallest '
                          'category -- maximising points is the point of pooling. Set a '
                          'positive value only if --benchmark shows the QP time is '
                          'prohibitive at full size.')
    ap.add_argument('--benchmark', action='store_true',
                     help='Do not run the grid. Instead take the first runnable cell and '
                          'time ONE glue fit at a range of points-per-manifold, printing '
                          'the measured scaling so the cap decision (if any) is made from '
                          'data rather than guessed.')
    ap.add_argument('--n_hyperplanes', type=int, default=200)
    ap.add_argument('--analysis_type', default='ONE_VERSUS_REST')
    ap.add_argument('--no_shuffle', action='store_true')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    bids_root = get_bids_root()

    def log(m=''):
        print(m, flush=True)

    ppc_cap = args.points_per_category if args.points_per_category > 0 else None

    # Benchmark writes nothing, so it must NOT be gated behind the
    # output-already-exists check -- otherwise it silently refuses to run for
    # exactly the cells you have already computed once and want to re-time.
    if args.benchmark:
        run_benchmark(args, bids_root, ppc_cap, log)
        return

    out_csv = output_csv_path(bids_root, args.voxRes, args.outdir)
    if os.path.exists(out_csv) and not args.force:
        print(f'SKIP (exists): {out_csv}  -- pass --force to overwrite.')
        return
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
                    n_man = len(CATEGORY_SCHEMES[scheme]['groups'])
                    cover = cover_max_points(info['n_features'], n_man)
                    cell_cap = cover if ppc_cap is None else min(ppc_cap, cover)
                    log(f'    scheme={scheme}: P={n_man} manifolds, F={info["n_features"]} '
                        f'features -> separability ceiling {cover} pts/manifold '
                        f'({int(COVER_SAFETY*100)}% of Cover 2N/P); using cap {cell_cap}')
                    for ep in EPOCH_ORDER:
                        lo, hi = EPOCHS[ep]
                        try:
                            manifolds, cats, ppc = build_epoch_manifolds(
                                X, tv, y, scheme, lo, hi, cell_cap, args.seed, log=log)
                        except ValueError as e:
                            log(f'    SKIP {ep}: {e}')
                            continue
                        if manifolds is None or len(manifolds) < 2:
                            log(f'    SKIP {ep}: fewer than 2 usable manifolds.')
                            continue

                        t0 = time.time()
                        try:
                            ret, ppc = glue_with_retry(
                                manifolds, log, retry_seed=args.seed,
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
