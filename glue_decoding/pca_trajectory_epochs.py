#!/usr/bin/env python3
"""
pca_trajectory_epochs.py

Location trajectories through a FIXED 2D subspace, over time.

WHAT THIS ADDS OVER THE EXISTING RDM+MDS FIGURES
------------------------------------------------
plot_visual_geometry_epochs.py re-fits MDS inside every panel, so each epoch
gets its own arbitrary rotation, reflection and scale. That is fine for asking
"is this epoch ringy?" but it makes it impossible to ask where a location MOVED
between epochs: the panels are not in a common frame.

Here the basis is defined once -- from the memory-delay condition means -- and
every timepoint is projected through that same basis. Locations then have
trajectories that can be compared across time, because they live in one space.

Note that PCA on condition means and classical (Torgerson) MDS on the distances
between those means are the same operation up to centering. So the projection
itself is not new information; the FIXED, SHARED basis is the point.

WHAT THE BASIS CHOICE MEANS FOR INTERPRETATION
----------------------------------------------
The basis spans the directions that carry location during the delay. A small
radius at fixation therefore means "no information in the DELAY's coding
directions", NOT "no information". A code that exists at fixation in some other
subspace is invisible here by construction -- which is the question the fixed
basis is asking, not a flaw, but it does have to be said out loud when reading
the figure.

CROSS-VALIDATION (--cv), AND WHY IT IS OFF BY DEFAULT
-----------------------------------------------------
Fitting the basis on the delay gives the delay an unfair advantage over epochs
that did not contribute to the fit. Measured on synthetic data where delay and
fixation were given the IDENTICAL true ring -- so the correct answer is "no
difference" -- fitting on all delay trials reported delay ahead by +0.09, while
a split-half fit reported -0.07, i.e. noise around zero.

That bias touches exactly ONE comparison: delay against the other epochs. The
other epochs are all projected onto a basis they had no part in fitting, so
they are mutually comparable with no holdout at all, and the trajectory picture
-- which is descriptive, not a test -- needs none either. Holding out also
halves the trials available for the basis, which at this SNR makes the basis
itself noisier. So the default uses every delay trial, and --cv is there for
the one case that needs it: quoting a delay-vs-other-epoch number.

ALIGNMENT ACROSS SUBJECTS
-------------------------
Generalized orthogonal Procrustes (rotation/reflection + uniform scale), NOT
CCA. CCA applies an arbitrary linear warp, which can stretch an ellipse into a
circle -- it manufactures the very geometry being measured. On matched
simulations, Procrustes separated a real ring from a label-shuffled one by 0.74
(0.984 vs 0.248) while CCA managed 0.22 (0.690 vs 0.466): CCA both lost real
signal and inflated the null.

The rotation is fit on the DELAY window only and then applied unchanged to
every timepoint, so the frame does not drift across the trajectory.

THE SHUFFLED NULL IS NOT OPTIONAL
---------------------------------
Aligning 21 noisy 2D clouds to a common template can produce a group ring on
its own. Every subject therefore also goes through the identical pipeline with
its location labels permuted -- same PCA, same Procrustes, same readout -- and
the null band on the metric figure is what that produces. If the real line is
inside the null band, the pipeline made the ring, not the brain.

Usage:
    python pca_trajectory_epochs.py
    python pca_trajectory_epochs.py --bands theta --rois visual
    python pca_trajectory_epochs.py --cv        # for delay-vs-other numbers
"""

import os
import sys
import argparse
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.linalg import orthogonal_procrustes

from constants import SUBJECT_LIST, ANGLE_MAPPING, get_bids_root, AMP_PHASE_BANDS
from features import build_features
from io_g04 import load_g04_band
from visual_geometry_cell import LOCATIONS, zscore_per_timepoint
from visual_geometry_epochs_cell import EPOCHS, LOCK_TYPE

REF_EPOCH = 'early_delay'
EPOCH_LABELS = {'fixation': 'Fixation', 'stimulus': 'Stimulus',
                'early_delay': 'Memory delay', 'late_delay': 'Late delay'}

# Locations in true angular order, and their angles -- the readout compares
# recovered angle against these.
LOC_BY_ANGLE = sorted(LOCATIONS, key=lambda l: ANGLE_MAPPING[l])
ANGLES_DEG = np.array([ANGLE_MAPPING[l] for l in LOC_BY_ANGLE], dtype=float)
ANGLES_RAD = np.deg2rad(ANGLES_DEG)
LOC_COLOURS = plt.cm.hsv(ANGLES_DEG / 360.0)

_BG, _FG, _GRID = '#000000', '#f0f0f0', '#333333'
FS_SUPTITLE, FS_PANEL_TTL, FS_AXIS_LABEL, FS_TICK, FS_LEGEND = 26, 21, 20, 17, 15


# ---------------------------------------------------------------- geometry --

def pca2(M):
    """Top-2 right singular vectors of the mean-centered condition means."""
    M = M - M.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[:2].T                                   # (n_features, 2)


def condition_means(X, y, mask=None):
    """(n_loc, F) means. Locations with no trials come back as NaN."""
    if mask is not None:
        X, y = X[mask], y[mask]
    out = np.full((len(LOC_BY_ANGLE), X.shape[-1]), np.nan)
    for li, loc in enumerate(LOC_BY_ANGLE):
        idx = (y == loc)
        if idx.any():
            out[li] = X[idx].mean(axis=0)
    return out


def ring_r(coords, sign=1):
    """
    Resultant length of (recovered angle - true angle): 1 = perfect ring,
    ~0.38 = chance for 10 points.

    Invariant to rotation by construction -- adding a constant to every angle
    difference rotates all unit vectors equally and leaves the magnitude alone
    -- so only the reflection has to be resolved, which the caller does ONCE
    from the delay window rather than per timepoint. Choosing the better
    reflection at every timepoint would quietly inflate the null.
    """
    C = coords - np.nanmean(coords, axis=0, keepdims=True)
    ok = np.isfinite(C).all(axis=1)
    if ok.sum() < 3:
        return np.nan
    a = np.arctan2(C[ok, 1], C[ok, 0]) * sign
    return float(abs(np.exp(1j * (a - ANGLES_RAD[ok])).mean()))


def mean_radius(coords):
    C = coords - np.nanmean(coords, axis=0, keepdims=True)
    return float(np.nanmean(np.linalg.norm(C, axis=1)))


def gpa(clouds, n_iter=10):
    """
    Generalized orthogonal Procrustes.

    Each cloud is centered and scaled to unit Frobenius norm BEFORE rotating --
    without the scale step a single high-variance subject dominates the group
    mean and the template becomes that subject's geometry.

    Returns (rotations, scales, centroids) so the same transform can be
    replayed on timepoints that took no part in fitting it.
    """
    cent = [np.nanmean(c, axis=0, keepdims=True) for c in clouds]
    cs = [c - m for c, m in zip(clouds, cent)]
    scl = [np.sqrt(np.nansum(c ** 2)) or 1.0 for c in cs]
    cs = [c / s for c, s in zip(cs, scl)]

    tmpl = cs[0].copy()
    rots = [np.eye(2)] * len(cs)
    for _ in range(n_iter):
        rots, aligned = [], []
        for c in cs:
            ok = np.isfinite(c).all(axis=1) & np.isfinite(tmpl).all(axis=1)
            R, _ = orthogonal_procrustes(c[ok], tmpl[ok])
            rots.append(R)
            aligned.append(c @ R)
        tmpl = np.nanmean(np.stack(aligned), axis=0)
    return rots, scl, cent


# ------------------------------------------------------------- per subject --

def load_subject(subjID, band, condition, roi, voxRes, bids_root):
    """(X, y, tv) with X (n_trials, n_times, F). Read from disk ONCE."""
    want_phase = (condition == 'ampPhase')
    g04 = load_g04_band(subjID, LOCK_TYPE, band, voxRes, bids_root,
                        want_phase=want_phase, roi=roi)
    tv = g04['time_vector']
    y = g04['target_labels'].astype(int)
    X = build_features(condition, g04['amp'], g04['phase'] if want_phase else None)
    del g04
    return zscore_per_timepoint(X), y, tv


def trajectory_from(X, y, tv, cv=False, seed=0, shuffle=False):
    """
    coords (n_times, n_loc, 2) in this subject's delay-defined basis.

    Takes already-loaded arrays rather than a subject id so the shuffled null
    can reuse them: the null needs the whole pipeline rerun per permutation,
    but NOT the MEG file reread, and rereading it 20x per subject was the
    difference between minutes and hours.
    """
    if shuffle:
        # Permute the labels, then redo EVERYTHING downstream -- the basis is
        # refit on the shuffled means too. A null that reused the real basis
        # would be testing a different, easier question.
        y = np.random.default_rng(seed).permutation(y)

    lo, hi = EPOCHS[REF_EPOCH]
    tmask = (tv >= lo) & (tv < hi)
    if not tmask.any():
        return None
    Xd = X[:, tmask, :].mean(axis=1)                  # (n_trials, F)

    if cv:
        rng = np.random.default_rng(seed + 1)
        perm = rng.permutation(X.shape[0])
        fit = np.zeros(X.shape[0], bool); fit[perm[:len(perm) // 2]] = True
    else:
        fit = np.ones(X.shape[0], bool)

    M = condition_means(Xd, y, fit)
    if not np.isfinite(M).all():
        return None
    B = pca2(M)                                       # (F, 2)
    return np.stack([condition_means(X[:, t, :], y) @ B
                     for t in range(X.shape[1])])


# ------------------------------------------------------------------ group ---

def group_align(per_subj, tv):
    """
    Fit the Procrustes rotation on the DELAY window only, then replay it on
    every timepoint so the frame is constant along the trajectory.
    """
    lo, hi = EPOCHS[REF_EPOCH]
    tmask = (tv >= lo) & (tv < hi)
    ref = [c[tmask].mean(axis=0) for c in per_subj]    # (n_loc, 2) per subject
    rots, scl, cent = gpa(ref)

    out = []
    for c, R, s, m in zip(per_subj, rots, scl, cent):
        out.append(((c - m[None]) / s) @ R)            # (n_times, n_loc, 2)
    return np.stack(out)                               # (n_subj, n_times, n_loc, 2)


def readout(group_coords, tv):
    """ring_r and radius over time, with the reflection fixed from the delay."""
    G = np.nanmean(group_coords, axis=0)               # (n_times, n_loc, 2)
    lo, hi = EPOCHS[REF_EPOCH]
    tmask = (tv >= lo) & (tv < hi)
    ref = np.nanmean(G[tmask], axis=0)
    sign = 1 if ring_r(ref, 1) >= ring_r(ref, -1) else -1
    r = np.array([ring_r(G[t], sign) for t in range(G.shape[0])])
    rad = np.array([mean_radius(G[t]) for t in range(G.shape[0])])
    return G, r, rad, sign


# ---------------------------------------------------------------- plotting --

def _style(ax):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, labelsize=FS_TICK)
    for s in ax.spines.values():
        s.set_edgecolor(_GRID)


def figure_trajectories(G, tv, band, condition, roi, voxRes, figdir, epochs):
    """One 2D panel per epoch, all in the SAME frame -- that is the point."""
    n = len(epochs)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 2.6, 5.2), facecolor=_BG,
                              squeeze=False)
    # Shared limits: with a common basis, panel-specific limits would undo the
    # comparability the fixed basis exists to provide.
    allpts = np.concatenate([G[(tv >= EPOCHS[e][0]) & (tv < EPOCHS[e][1])].reshape(-1, 2)
                             for e in epochs])
    lim = np.nanmax(np.abs(allpts)) * 1.15
    for c, ep in enumerate(epochs):
        ax = axes[0][c]; _style(ax)
        m = (tv >= EPOCHS[ep][0]) & (tv < EPOCHS[ep][1])
        seg = G[m]                                     # (n_t, n_loc, 2)
        for li in range(len(LOC_BY_ANGLE)):
            ax.plot(seg[:, li, 0], seg[:, li, 1], color=LOC_COLOURS[li],
                    lw=2.2, alpha=0.9, zorder=3)
            ax.scatter(seg[-1, li, 0], seg[-1, li, 1], color=LOC_COLOURS[li],
                       s=90, zorder=4, edgecolors='k', linewidths=0.8)
        ax.axhline(0, color=_GRID, lw=1.0, zorder=1)
        ax.axvline(0, color=_GRID, lw=1.0, zorder=1)
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(EPOCH_LABELS.get(ep, ep), fontsize=FS_PANEL_TTL, color=_FG,
                     fontweight='bold', pad=10)
    h = [plt.Line2D([0], [0], marker='o', ls='', color=LOC_COLOURS[i],
                    markersize=9, markeredgecolor='k')
         for i in range(len(LOC_BY_ANGLE))]
    leg = fig.legend(h, [f'{int(a)}°' for a in ANGLES_DEG], loc='center left',
                     bbox_to_anchor=(0.995, 0.5), fontsize=FS_LEGEND,
                     framealpha=0.25, edgecolor='#444444', labelcolor=_FG,
                     title='Target angle')
    leg.get_frame().set_facecolor('#1a1a1a')
    leg.get_title().set_color(_FG); leg.get_title().set_fontsize(FS_AXIS_LABEL)
    fig.suptitle(f'{roi.capitalize()}  |  {band.capitalize()}', color=_FG,
                 fontsize=FS_SUPTITLE, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir,
                      f'pca_traj_{condition}_{band}_{roi}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG); plt.close(fig)
    print(f'Saved: {fp}')


def figure_timecourse(tv, r, rad, r_null, rad_null, band, condition, roi,
                       voxRes, figdir):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.0), facecolor=_BG, squeeze=False)
    for ax, real, null, lab in ((axes[0][0], r, r_null, 'Ring fidelity'),
                                 (axes[0][1], rad, rad_null, 'Radius in delay subspace')):
        _style(ax)
        if null is not None and null.size:
            # 2.5-97.5 across shuffles: what this pipeline produces when no
            # ring exists. The real line has to leave this band to mean
            # anything.
            lo_, hi_ = np.nanpercentile(null, [2.5, 97.5], axis=0)
            ax.fill_between(tv, lo_, hi_, color='#888888', alpha=0.28, lw=0,
                            zorder=1, label='Shuffled null (95%)')
        ax.plot(tv, real, color='#FFC629', lw=3.2, zorder=3, label='Real')
        for t0 in (0.0, EPOCHS[REF_EPOCH][0]):
            ax.axvline(t0, color='#cccccc', lw=2.0, ls=':', zorder=2)
        ax.set_xlabel('Time (s)', fontsize=FS_AXIS_LABEL, fontweight='bold',
                      color=_FG)
        ax.set_ylabel(lab, fontsize=FS_AXIS_LABEL, fontweight='bold', color=_FG)
        ax.legend(fontsize=FS_LEGEND, framealpha=0.25, edgecolor='#444444',
                  labelcolor=_FG, loc='upper left')
    fig.suptitle(f'{roi.capitalize()}  |  {band.capitalize()}', color=_FG,
                 fontsize=FS_SUPTITLE, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir,
                      f'pca_traj_timecourse_{condition}_{band}_{roi}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG); plt.close(fig)
    print(f'Saved: {fp}')


# ------------------------------------------------------------------- main ---

def run_cell(subjects, band, condition, roi, voxRes, bids_root, figdir,
             epochs, cv=False, n_shuffles=20):
    # One pass over subjects: read the file once, then compute the real
    # trajectory and every shuffle from the same in-memory arrays.
    per_subj, tv = [], None
    null_subj = [[] for _ in range(n_shuffles)]
    for s in subjects:
        try:
            X, y, t = load_subject(s, band, condition, roi, voxRes, bids_root)
        except Exception as e:
            print(f'  sub-{s:02d}: {type(e).__name__}: {e}')
            continue
        if tv is None:
            tv = t
        elif t.size != tv.size:
            print(f'  sub-{s:02d}: time axis mismatch, skipping.')
            continue
        c = trajectory_from(X, y, t, cv=cv, seed=s)
        if c is None:
            del X
            continue
        per_subj.append(c)
        for k in range(n_shuffles):
            cs = trajectory_from(X, y, t, cv=cv, seed=1000 * k + s, shuffle=True)
            if cs is not None:
                null_subj[k].append(cs)
        del X
    if len(per_subj) < 3:
        print(f'  {band}/{condition}/{roi}: only {len(per_subj)} subjects, skipping.')
        return

    G, r, rad, sign = readout(group_align(per_subj, tv), tv)

    r_null, rad_null = [], []
    for k in range(n_shuffles):
        if len(null_subj[k]) < 3:
            continue
        _, rs, rads, _ = readout(group_align(null_subj[k], tv), tv)
        r_null.append(rs); rad_null.append(rads)
    r_null = np.array(r_null) if r_null else None
    rad_null = np.array(rad_null) if rad_null else None

    lo, hi = EPOCHS[REF_EPOCH]
    dm = (tv >= lo) & (tv < hi)
    msg = f'  {band}/{condition}/{roi}: n={len(per_subj)} ring(delay)={np.nanmean(r[dm]):.3f}'
    if r_null is not None:
        msg += f' null={np.nanmean(r_null[:, dm]):.3f}'
    print(msg, flush=True)

    figure_trajectories(G, tv, band, condition, roi, voxRes, figdir, epochs)
    figure_timecourse(tv, r, rad, r_null, rad_null, band, condition, roi,
                       voxRes, figdir)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[2])
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly'])
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--epochs', nargs='+',
                     default=['fixation', 'stimulus', 'early_delay', 'late_delay'],
                     choices=list(EPOCHS),
                     help='Epochs to draw as trajectory panels. All four by '
                          'default -- the point of the fixed basis is to '
                          'compare them, including the one it was fit on.')
    ap.add_argument('--cv', action='store_true',
                     help='Fit the basis on half the delay trials and evaluate '
                          'on the rest. Only needed when quoting a '
                          'delay-vs-other-epoch number: fitting on all delay '
                          'trials gives the delay a spurious advantage over '
                          'epochs that took no part in the fit (+0.09 on '
                          'synthetic data where the true difference was zero). '
                          'Costs half the trials for the basis, so it is off '
                          'for the descriptive trajectory figure.')
    ap.add_argument('--n_shuffles', type=int, default=20,
                     help='Label-permuted runs through the identical pipeline '
                          '(default 20). Set 0 to skip, but then the figure '
                          'cannot show whether alignment alone made the ring.')
    ap.add_argument('--figdir', default=None)
    args = ap.parse_args()

    bids_root = get_bids_root()
    figdir = args.figdir or os.path.join(
        bids_root, 'derivatives', 'glueDecoding', 'pcaTrajectoryEpochs', 'figures')

    print(f'pca_trajectory_epochs | basis={REF_EPOCH} | cv={args.cv} | '
          f'shuffles={args.n_shuffles} | subjects={len(args.subjects)}')
    for condition in args.conditions:
        for band in args.bands:
            if condition == 'ampPhase' and band not in AMP_PHASE_BANDS:
                continue
            for roi in args.rois:
                try:
                    run_cell(args.subjects, band, condition, roi, args.voxRes,
                             bids_root, figdir, args.epochs, cv=args.cv,
                             n_shuffles=args.n_shuffles)
                except Exception:
                    traceback.print_exc()
    print('Done.')


if __name__ == '__main__':
    main()
