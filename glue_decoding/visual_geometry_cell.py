#!/usr/bin/env python3
"""
visual_geometry_cell.py

Per-subject, pre-capacity geometry of the spatial-WM location code in the
VISUAL ROI: an interpretable "signal vs noise vs alignment" decomposition
computed ENTIRELY WITHIN each subject's own source space, saved as
rotation-invariant scalars + a per-subject RDM for cross-subject-safe
aggregation by aggregate_visual_geometry.py (Option A -- raw trials are
NEVER pooled across subjects, whose source spaces are not in register).

Grid: 3 bands (theta/alpha/beta) x 2 feature reps (ampOnly / ampPhase) x
3 epochs (fixation/stim/delay) = 18 cells per subject. Stim-locked only.

PREPROCESSING -- features are z-scored across trials independently at each
timepoint (see zscore_per_timepoint) at native time resolution, BEFORE the
epoch average, matching what both decoders in this repo do at every
timepoint they evaluate. The mean-subtraction half of that step IS the ERP
removal ("ERP always removed" is therefore satisfied, and a separate
subtraction would be an exact no-op); the SD half puts the wildly different
source amplitudes on a common footing, without which a few loud sources
would dominate every distance, covariance, and alignment estimate below.

Per cell, in the subject's own space:

  SIGNAL -- 10x10 RDM over target locations, CROSS-VALIDATED so it is
  unbiased (expectation 0 for identical conditions; individual entries may
  legitimately go NEGATIVE). Naive Euclidean is deliberately NOT used: it is
  noise-biased, and the bias scales with trials-per-location, which varies
  12-43 across subjects -- i.e. it would manufacture systematic between-
  subject differences out of nothing. Two RDMs are saved:
    rdm_primary   : crossnobis (cross-validated + whitened by the pooled
                    within-location noise covariance) in the PCA-projected
                    space; falls back to unwhitened cross-validated
                    Euclidean if the conditioning check below fails
                    (rdm_primary_whitened flags which was used).
    rdm_cv_full   : unwhitened cross-validated Euclidean in the FULL source
                    space (no PCA). Kept as the bias-free reference -- see
                    the PCA caveat below.
  Both are averaged over `n_splits` independent random 2-fold partitions
  (a single split is very noisy at ~15 trials/location; averaging splits
  reduces variance without touching unbiasedness).

  NOISE (per location) -- mean distance of trials to their own centroid
  ("radius"), and effective dimensionality of the trial cloud as the
  participation ratio PR = (sum eig)^2 / sum(eig^2) of the within-location
  covariance.

  ALIGNMENT (per location) -- principal angle between the location cloud's
  top variance direction (first right-singular vector of the centered
  cloud) and the vector from the GLOBAL centroid to that location's
  centroid. Folded to [0, 90] deg, because a singular vector's sign is
  arbitrary (an axis, not a direction). ~90 deg = noise spread orthogonal
  to the separating axis (good for separability); ~0 deg = spread along it
  (bad).

CONDITIONING (the low-trial regime is the whole difficulty here)
  n_features (597, or 1194 for ampPhase) FAR exceeds trials-per-location
  (~15), so covariance-based quantities are ill-conditioned at raw
  dimension. Before the NOISE/ALIGNMENT computations and before whitening,
  each subject's cell is PCA-projected (fit on that subject's own trials,
  one fixed space reused consistently within the cell) to
      k = min(n_trials_total - 1, MAX_PCA_DIM, n_features).
  Whitening additionally requires residual dof >= WHITEN_DOF_FACTOR * k,
  else that cell falls back to unwhitened (reported per cell, not silent).
  The pooled noise covariance uses Ledoit-Wolf analytic shrinkage toward a
  scaled identity (implemented here in pure numpy -- see _ledoit_wolf_cov),
  which keeps it invertible and well-conditioned rather than merely
  invertible.

  PCA CAVEAT (why rdm_cv_full is also saved): the projection is fit on ALL
  trials of the cell, so it is not independent of either cross-validation
  fold. PCA is unsupervised (never sees the location labels) so it cannot
  invent between-location structure, but because the retained directions
  are chosen using both folds' noise, the projected crossnobis estimate can
  carry a small optimistic bias. rdm_cv_full is computed in the full,
  unprojected space and is therefore free of that particular concern --
  compare the two before trusting a marginal effect.

TRIAL-COUNT MATCHING (why NOISE/ALIGNMENT subsample)
  PR is hard-bounded by (n_trials_in_cloud - 1), so a location with 43
  trials can score a higher PR than one with 12 for reasons that are purely
  about sample size, not geometry. The same sample-size dependence makes
  the top-PC direction (hence the alignment angle) noisier for small
  clouds. So NOISE/ALIGNMENT are computed on locations subsampled to that
  subject's own minimum location count (n_per_loc, recorded in the output),
  averaged over `n_subsamples` random draws -- the same balancing logic
  constants.balance_categories already applies elsewhere in this repo.
  Trials-per-location is matched WITHIN a subject by this; it still differs
  ACROSS subjects, so n_per_loc is saved per cell and should be checked
  before reading much into between-subject PR differences.
  The RDM deliberately does NOT subsample -- crossnobis is already unbiased,
  so it can use all available trials for a lower-variance estimate.

SCALE (which scalars are safe to average across subjects)
  PR and the alignment angle are dimensionless/scale-free, so they average
  across subjects directly. The raw radius is NOT -- it inherits each
  subject's arbitrary amplitude units, so a raw-radius group mean is
  dominated by whichever subject has the largest scale. Therefore
  radius_norm = radius / mean_centroid_distance is also saved (mean over
  locations of ||centroid_l - global_centroid||, same cell, same space):
  dimensionless, comparable across subjects, and directly the
  noise-to-signal ratio manifold capacity actually depends on. Prefer
  radius_norm for group work; raw radius is kept for reference only.

Usage:
    python visual_geometry_cell.py <subjID> [--bands theta alpha beta]
                                    [--feature_reps ampOnly ampPhase]
                                    [--voxRes 8mm] [--roi visual]
                                    [--max_pca_dim 50] [--n_splits 10]
                                    [--n_subsamples 20] [--seed 0]
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

from constants import ANGLE_MAPPING, get_bids_root
from features import build_features
from io_g04 import load_g04_band

# ── Constants ────────────────────────────────────────────────────────────────

LOCK_TYPE = 'stim'
LOCATIONS = tuple(sorted(ANGLE_MAPPING))          # 1..10

# (lo, hi, hi_inclusive) -- same window definitions as the rest of the
# pipeline (manifold_capacity.py's EPOCHS for stim/delay, plot_timeseries.py's
# BASELINE_WINDOWS for the pre-stimulus fixation baseline).
EPOCHS = {
    'fixation': (-1.0, 0.0, False),
    'stim':     ( 0.0, 0.2, False),
    'delay':    ( 0.2, 1.7, True),
}
EPOCH_ORDER = ('fixation', 'stim', 'delay')

MAX_PCA_DIM        = 50     # see CONDITIONING in module docstring
WHITEN_DOF_FACTOR  = 2.0    # need residual dof >= this * k to whiten
N_SPLITS           = 10     # random 2-fold partitions averaged for the RDM
N_SUBSAMPLES       = 20     # random balanced draws averaged for NOISE/ALIGNMENT
MIN_TRIALS_PER_LOC = 4      # need >= 2 per fold for a cross-validated estimate


# ── Output path ──────────────────────────────────────────────────────────────

def output_path(bids_root, subjID, band, feature_rep, epoch, roi, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'visualGeometry')
    os.makedirs(base, exist_ok=True)
    fname = (f'{subName}_task-mgs_visGeom_{feature_rep}_{band}_{epoch}_'
             f'{roi}_{voxRes}.npz')
    return os.path.join(base, fname)


# ── Numerics ─────────────────────────────────────────────────────────────────

def zscore_per_timepoint(X):
    """
    Z-score every feature across TRIALS, independently at each timepoint:
    for each (t, f), mu/sd are taken over the trial axis. X: (N, T, F).

    Applied BEFORE epoch averaging, matching the convention used by both
    decoders in this repo (linear_decoding_categories_cell.py's
    ridge_ovr_timeseries and decoding_ts_cell.py's ridge_loocv_timeseries
    z-score exactly this way at each timepoint they evaluate), so geometry
    results here are computed on the same feature scaling the decoding
    results are, and the two are directly comparable.

    Two distinct jobs, both necessary here:
      - the mean subtraction IS the ERP removal (the across-trial mean at a
        timepoint is by definition the evoked response), so the spec's
        "ERP removed in all cases" is satisfied by this step and no separate
        subtraction is needed -- a separate one would be an exact no-op;
      - the division by SD is what the earlier epoch-average-only version was
        missing: MEG source amplitudes differ by orders of magnitude across
        sources (depth/leadfield-norm bias), and every quantity computed
        downstream (crossnobis distances, the pooled noise covariance,
        participation ratio, the top-variance direction behind the alignment
        angle) is variance-weighted. Unscaled, a handful of high-amplitude
        sources would dominate all of them, and each subject's geometry would
        partly reflect which of its sources happened to be loudest.

    sd < 1e-10 is set to 1.0 (dead/constant feature -> left at 0 rather than
    amplified into noise), same guard the decoders use.
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-10, np.asarray(1.0, dtype=sd.dtype), sd)
    return (X - mu) / sd


def epoch_average(X, tv, lo, hi, hi_inclusive):
    """(N, T, F) -> (N, F), averaging the timepoints inside the window."""
    mask = (tv >= lo) & (tv <= hi) if hi_inclusive else (tv >= lo) & (tv < hi)
    if not mask.any():
        raise ValueError(f'no timepoints in window [{lo}, {hi}] (tv spans '
                         f'[{tv[0]:.3f}, {tv[-1]:.3f}])')
    return X[:, mask, :].mean(axis=1), int(mask.sum())


def pca_project(Xe, max_pca_dim=MAX_PCA_DIM):
    """
    Fit a PCA basis on this cell's own trials and project into it.
    Returns (Xp, k, explained_var_ratio, global_mean).

    One fixed basis per cell, reused for every downstream quantity (see
    module docstring's CONDITIONING / PCA CAVEAT).
    """
    n_trials, n_feat = Xe.shape
    mu = Xe.mean(axis=0, keepdims=True)
    Xc = Xe - mu
    # full_matrices=False -> Vt is (min(N,F), F)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = int(min(n_trials - 1, max_pca_dim, n_feat, Vt.shape[0]))
    k = max(k, 1)
    Vk = Vt[:k].T                       # (F, k)
    Xp = Xc @ Vk                        # (N, k)
    total_var = float((S ** 2).sum())
    explained = float((S[:k] ** 2).sum() / total_var) if total_var > 0 else np.nan
    return Xp, k, explained


def _ledoit_wolf_cov(R):
    """
    Ledoit-Wolf analytically-shrunk covariance of already-centered residuals
    R (n, p), shrinking toward (trace(S)/p) * I. Pure numpy (no sklearn
    dependency) -- the closed form is:
        mu    = tr(S)/p
        d2    = ||S - mu I||_F^2 / p
        b2bar = (sum_i ||r_i||^4 - n ||S||_F^2) / (n^2 p)
        shrink = min(b2bar, d2) / d2
        Sigma = shrink * mu * I + (1 - shrink) * S
    Returns (Sigma, shrinkage).
    """
    n, p = R.shape
    S = (R.T @ R) / n                       # biased MLE covariance (R is centered)
    mu = float(np.trace(S) / p)
    S_fro2 = float(np.sum(S ** 2))
    d2 = (S_fro2 - 2.0 * mu * float(np.trace(S)) + p * mu ** 2) / p
    if d2 <= 0:
        return S, 0.0
    row_sq = np.sum(R ** 2, axis=1)         # ||r_i||^2
    b2bar = float((np.sum(row_sq ** 2) - n * S_fro2) / (n ** 2 * p))
    shrink = float(np.clip(min(b2bar, d2) / d2, 0.0, 1.0))
    Sigma = shrink * mu * np.eye(p) + (1.0 - shrink) * S
    return Sigma, shrink


def _two_fold_centroids(Xp, y, rng):
    """
    Split each location's trials into 2 disjoint folds and return
    (CA, CB, ok) -- (n_loc, k) fold centroids and a (n_loc,) bool mask of
    locations that had enough trials for both folds.
    """
    n_loc = len(LOCATIONS)
    CA = np.full((n_loc, Xp.shape[1]), np.nan)
    CB = np.full((n_loc, Xp.shape[1]), np.nan)
    ok = np.zeros(n_loc, dtype=bool)
    for li, loc in enumerate(LOCATIONS):
        idx = np.where(y == loc)[0]
        if idx.size < MIN_TRIALS_PER_LOC:
            continue
        perm = rng.permutation(idx)
        half = perm.size // 2
        CA[li] = Xp[perm[:half]].mean(axis=0)
        CB[li] = Xp[perm[half:]].mean(axis=0)
        ok[li] = True
    return CA, CB, ok


def crossvalidated_rdm(X, y, n_splits=N_SPLITS, Sigma_inv=None, seed=0):
    """
    Cross-validated (optionally whitened) representational dissimilarity
    matrix over LOCATIONS, averaged over `n_splits` random 2-fold splits.

    d[i,j] = (mA_i - mA_j)^T Sigma^-1 (mB_i - mB_j)

    with folds A/B disjoint, which makes E[d] = 0 when conditions i and j
    are identical -- so entries can be negative, by design. Sigma_inv=None
    gives the unwhitened cross-validated Euclidean version.

    Returns (rdm, n_valid_splits) -- rdm is (10, 10) with NaN rows/cols for
    locations that never had enough trials.
    """
    rng = np.random.default_rng(seed)
    n_loc = len(LOCATIONS)
    acc = np.zeros((n_loc, n_loc))
    cnt = np.zeros((n_loc, n_loc))

    for _ in range(n_splits):
        CA, CB, ok = _two_fold_centroids(X, y, rng)
        if ok.sum() < 2:
            continue
        CAo = np.nan_to_num(CA)
        CBo = np.nan_to_num(CB)
        WB = CBo if Sigma_inv is None else CBo @ Sigma_inv
        G = CAo @ WB.T                                   # G[i,j] = CA_i . (Sinv CB_j)
        g_diag = np.diag(G)
        # d[i,j] = G[i,i] - G[i,j] - G[j,i] + G[j,j]  (symmetric by construction)
        d = g_diag[:, None] - G - G.T + g_diag[None, :]
        valid = np.outer(ok, ok)
        acc[valid] += d[valid]
        cnt[valid] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        rdm = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    np.fill_diagonal(rdm, 0.0)
    return rdm, int(cnt.max()) if cnt.size else 0


def noise_and_alignment(Xp, y, n_subsamples=N_SUBSAMPLES, seed=0):
    """
    Per-location NOISE (radius, participation ratio) and ALIGNMENT
    (principal angle, deg) in the projected space, computed on locations
    subsampled to a common trial count (see module docstring's TRIAL-COUNT
    MATCHING) and averaged over `n_subsamples` draws.

    Returns dict of (10,) arrays + scalars. NaN for locations with too few
    trials.
    """
    rng = np.random.default_rng(seed)
    n_loc = len(LOCATIONS)
    idx_by_loc = {loc: np.where(y == loc)[0] for loc in LOCATIONS}
    usable = [loc for loc in LOCATIONS if idx_by_loc[loc].size >= MIN_TRIALS_PER_LOC]
    if len(usable) < 2:
        nan10 = np.full(n_loc, np.nan)
        return dict(radius=nan10, pr=nan10, align_deg=nan10, align_cos=nan10,
                     radius_norm=nan10, n_per_loc=0, mean_centroid_dist=np.nan)

    n_per_loc = int(min(idx_by_loc[loc].size for loc in usable))

    rad_acc   = np.zeros((n_subsamples, n_loc)) * np.nan
    pr_acc    = np.zeros((n_subsamples, n_loc)) * np.nan
    cos_acc   = np.zeros((n_subsamples, n_loc)) * np.nan
    cdist_acc = np.zeros((n_subsamples, n_loc)) * np.nan

    for s in range(n_subsamples):
        # Balanced draw, then a balanced global centroid (mean OF THE LOCATION
        # CENTROIDS, not of all trials -- so unequal trial counts can't tilt it).
        clouds = {}
        for loc in usable:
            sel = rng.choice(idx_by_loc[loc], size=n_per_loc, replace=False)
            clouds[loc] = Xp[sel]
        centroids = {loc: c.mean(axis=0) for loc, c in clouds.items()}
        global_centroid = np.mean(np.stack([centroids[loc] for loc in usable]), axis=0)

        for li, loc in enumerate(LOCATIONS):
            if loc not in clouds:
                continue
            cloud = clouds[loc]
            centroid = centroids[loc]
            centred = cloud - centroid

            rad_acc[s, li] = float(np.mean(np.linalg.norm(centred, axis=1)))

            # Participation ratio of the within-location covariance. Computed
            # from the cloud's singular values (eig of cov = sv^2 / (n-1)); the
            # (n-1) factor cancels in the ratio, so it is used directly.
            sv = np.linalg.svd(centred, compute_uv=False)
            ev = sv ** 2
            ev_sum = float(ev.sum())
            pr_acc[s, li] = (ev_sum ** 2 / float((ev ** 2).sum())) if ev_sum > 0 else np.nan

            # Alignment: top variance axis vs the global-centroid -> centroid axis.
            # abs() folds to [0, 90] deg because a singular vector's sign is
            # arbitrary (it defines an axis, not a direction).
            sep = centroid - global_centroid
            sep_norm = float(np.linalg.norm(sep))
            cdist_acc[s, li] = sep_norm
            if sep_norm > 1e-12 and sv[0] > 1e-12:
                top_pc = np.linalg.svd(centred, full_matrices=False)[2][0]
                cos_acc[s, li] = float(abs(np.dot(top_pc, sep / sep_norm)))

    with np.errstate(invalid='ignore'):
        radius    = np.nanmean(rad_acc,   axis=0)
        pr        = np.nanmean(pr_acc,    axis=0)
        align_cos = np.nanmean(cos_acc,   axis=0)
        cdist     = np.nanmean(cdist_acc, axis=0)

    align_deg = np.degrees(np.arccos(np.clip(align_cos, 0.0, 1.0)))
    mean_cdist = float(np.nanmean(cdist)) if np.isfinite(cdist).any() else np.nan
    # Dimensionless noise-to-signal radius -- the group-averageable one.
    radius_norm = radius / mean_cdist if (mean_cdist and np.isfinite(mean_cdist)
                                           and mean_cdist > 1e-12) else np.full(n_loc, np.nan)

    return dict(radius=radius, pr=pr, align_deg=align_deg, align_cos=align_cos,
                 radius_norm=radius_norm, n_per_loc=n_per_loc,
                 mean_centroid_dist=mean_cdist)


# ── Per-cell driver ──────────────────────────────────────────────────────────

def compute_cell(Xe, y, max_pca_dim, n_splits, n_subsamples, seed):
    """All geometry for one (subject, band, feature_rep, epoch) cell."""
    n_trials, n_feat = Xe.shape

    Xp, k, explained = pca_project(Xe, max_pca_dim)

    # Pooled within-location noise covariance (residuals about each location's
    # own centroid), in the projected space, Ledoit-Wolf shrunk.
    resid, n_loc_used = [], 0
    for loc in LOCATIONS:
        idx = np.where(y == loc)[0]
        if idx.size < MIN_TRIALS_PER_LOC:
            continue
        resid.append(Xp[idx] - Xp[idx].mean(axis=0))
        n_loc_used += 1
    R = np.concatenate(resid, axis=0) if resid else np.zeros((0, k))
    dof = max(0, R.shape[0] - n_loc_used)

    whiten = dof >= WHITEN_DOF_FACTOR * k and R.shape[0] > 1
    if whiten:
        Sigma, shrink = _ledoit_wolf_cov(R)
        Sigma_inv = np.linalg.pinv(Sigma)
    else:
        Sigma_inv, shrink = None, np.nan

    rdm_primary, n_ok = crossvalidated_rdm(Xp, y, n_splits=n_splits,
                                            Sigma_inv=Sigma_inv, seed=seed)
    # Bias-free reference: unwhitened, unprojected (see PCA CAVEAT).
    rdm_cv_full, _ = crossvalidated_rdm(Xe, y, n_splits=n_splits,
                                         Sigma_inv=None, seed=seed)

    noise = noise_and_alignment(Xp, y, n_subsamples=n_subsamples, seed=seed)

    return dict(
        rdm_primary=rdm_primary, rdm_cv_full=rdm_cv_full,
        rdm_primary_whitened=bool(whiten), lw_shrinkage=float(shrink),
        pca_dim=int(k), pca_explained_var=float(explained),
        whiten_dof=int(dof), n_trials=int(n_trials), n_features=int(n_feat),
        n_splits_effective=int(n_ok), **noise)


def run_cell(subjID, bands, feature_reps, voxRes, bids_root, roi,
             max_pca_dim=MAX_PCA_DIM, n_splits=N_SPLITS, n_subsamples=N_SUBSAMPLES,
             seed=0, outdir=None, force=False):
    for band in bands:
        need_phase = any(fr == 'ampPhase' for fr in feature_reps)

        pending = [(fr, ep) for fr in feature_reps for ep in EPOCH_ORDER
                   if force or not os.path.exists(
                       output_path(bids_root, subjID, band, fr, ep, roi, voxRes, outdir))]
        if not pending:
            print(f'sub-{subjID:02d} | {band}: SKIP (all cells exist)', flush=True)
            continue

        try:
            g04 = load_g04_band(subjID, LOCK_TYPE, band, voxRes, bids_root,
                                 want_phase=need_phase, roi=roi)
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f'sub-{subjID:02d} | {band}: SKIP ({e})', flush=True)
            continue

        amp   = g04['amp']
        phase = g04['phase'] if need_phase else None
        tv    = g04['time_vector']
        y     = g04['target_labels'].astype(int)

        for feature_rep in feature_reps:
            try:
                X = build_features(feature_rep, amp, phase)
            except ValueError as e:
                print(f'  sub-{subjID:02d} {band} {feature_rep}: SKIP ({e})', flush=True)
                continue

            # Per-timepoint z-scoring across trials at native time resolution,
            # BEFORE epoch averaging -- same convention as the two decoders.
            # Its mean-subtraction component performs the ERP removal the spec
            # asks for; its SD division is what puts wildly-differing source
            # amplitudes on a common footing. See zscore_per_timepoint.
            X = zscore_per_timepoint(X)

            for epoch in EPOCH_ORDER:
                out_path = output_path(bids_root, subjID, band, feature_rep, epoch,
                                        roi, voxRes, outdir)
                if not force and os.path.exists(out_path):
                    continue

                t_start = time.time()
                try:
                    lo, hi, hi_inc = EPOCHS[epoch]
                    Xe, n_win_times = epoch_average(X, tv, lo, hi, hi_inc)
                    res = compute_cell(Xe, y, max_pca_dim, n_splits, n_subsamples, seed)

                    np.savez_compressed(
                        out_path,
                        locations            = np.array(LOCATIONS),
                        location_angles_deg  = np.array([ANGLE_MAPPING[l] for l in LOCATIONS],
                                                         dtype=float),
                        rdm_primary          = res['rdm_primary'].astype(np.float64),
                        rdm_cv_full          = res['rdm_cv_full'].astype(np.float64),
                        rdm_primary_whitened = np.array([res['rdm_primary_whitened']]),
                        lw_shrinkage         = np.array([res['lw_shrinkage']]),
                        radius               = res['radius'].astype(np.float64),
                        radius_norm          = res['radius_norm'].astype(np.float64),
                        pr                   = res['pr'].astype(np.float64),
                        align_deg            = res['align_deg'].astype(np.float64),
                        align_cos            = res['align_cos'].astype(np.float64),
                        mean_centroid_dist   = np.array([res['mean_centroid_dist']]),
                        n_per_loc            = np.array([res['n_per_loc']]),
                        pca_dim              = np.array([res['pca_dim']]),
                        pca_explained_var    = np.array([res['pca_explained_var']]),
                        whiten_dof           = np.array([res['whiten_dof']]),
                        n_trials             = np.array([res['n_trials']]),
                        n_features           = np.array([res['n_features']]),
                        n_window_times       = np.array([n_win_times]),
                        n_splits_effective   = np.array([res['n_splits_effective']]),
                        subjID               = np.array([subjID]),
                        band                 = np.array([band]),
                        feature_rep          = np.array([feature_rep]),
                        epoch                = np.array([epoch]),
                        roi                  = np.array([roi]),
                        voxRes               = np.array([voxRes]),
                        seed                 = np.array([seed]),
                    )
                    print(f'  sub-{subjID:02d} | {band} | {feature_rep} | {epoch}: '
                          f'N={res["n_trials"]} F={res["n_features"]} k={res["pca_dim"]} '
                          f'(var {res["pca_explained_var"]:.2f}) | whiten={res["rdm_primary_whitened"]} '
                          f'(dof={res["whiten_dof"]}, shrink={res["lw_shrinkage"]:.3f}) | '
                          f'n_per_loc={res["n_per_loc"]} | {time.time() - t_start:.1f}s',
                          flush=True)
                except ValueError as e:
                    print(f'  sub-{subjID:02d} {band} {feature_rep} {epoch}: SKIP ({e})',
                          flush=True)
                except Exception:
                    print(f'  FAILED sub-{subjID:02d} {band}/{feature_rep}/{epoch}:', flush=True)
                    traceback.print_exc()

            del X


def main():
    parser = argparse.ArgumentParser(
        description='Per-subject pre-capacity geometry (signal/noise/alignment) of the '
                     'visual-ROI location code.')
    parser.add_argument('subjID', type=int)
    parser.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    parser.add_argument('--feature_reps', nargs='+', default=['ampOnly', 'ampPhase'],
                         choices=['ampOnly', 'ampPhase'])
    parser.add_argument('--voxRes', default='8mm')
    parser.add_argument('--roi', default='visual')
    parser.add_argument('--max_pca_dim', type=int, default=MAX_PCA_DIM)
    parser.add_argument('--n_splits', type=int, default=N_SPLITS)
    parser.add_argument('--n_subsamples', type=int, default=N_SUBSAMPLES)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    bids_root = get_bids_root()
    print(f'visual_geometry_cell | sub-{args.subjID:02d} | bands={args.bands} | '
          f'feature_reps={args.feature_reps} | roi={args.roi} | {args.voxRes} | '
          f'max_pca_dim={args.max_pca_dim} | n_splits={args.n_splits} | '
          f'n_subsamples={args.n_subsamples} | seed={args.seed} | force={args.force}',
          flush=True)

    t0 = time.time()
    run_cell(args.subjID, list(args.bands), list(args.feature_reps), args.voxRes,
             bids_root, args.roi, max_pca_dim=args.max_pca_dim, n_splits=args.n_splits,
             n_subsamples=args.n_subsamples, seed=args.seed,
             outdir=args.outdir, force=args.force)
    print(f'Done | sub-{args.subjID:02d} | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
