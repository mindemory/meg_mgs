#!/usr/bin/env python3
"""
plot_visual_geometry_ts.py

Group aggregation + time-resolved MDS geometry metrics for
visual_geometry_ts_cell.py.

At each timepoint: every subject's RDM is z-scored across its own
off-diagonal entries and averaged (raw distances are not comparable across
subjects), classical MDS is run on the group RDM, and the spectrum is
summarized by the metrics below. The identical pipeline is applied to the
label-shuffled RDMs to build a null band, because NONE of these metrics has
a chance level of zero.

METRICS (all from the eigenvalues lambda_1 >= lambda_2 >= ... of the
double-centered group RDM, positive eigenvalues only)

  ring_r        Rotation/reflection-invariant agreement between each
                location's embedded angle and its true polar angle. 1 = a
                perfect ring; chance ~0.38, p95 ~0.61 (10 points).

  lam2_over_lam1  lambda_2/lambda_1. Distinguishes a genuine 2-D ring from a
                1-D line dressed up as one.
                *** READ THE REFERENCE LINE, NOT "IS IT NEAR 1" ***
                The familiar "a ring gives ~1" rule assumes UNIFORMLY spaced
                angles. This study's 10 locations are not uniform -- gaps are
                25,25,80,25,25,25,25,80,25,25 deg, i.e. two clusters -- and a
                PERFECT ring sampled at exactly these angles gives
                lambda_2/lambda_1 = 0.441, not 1.0 (verified analytically).
                Worse, random 10-point geometry averages 0.641, ABOVE the
                perfect-ring value, so scoring "closer to 1 = more ring-like"
                would rank noise above a true ring here. Compare against
                IDEAL_RING['lam2_over_lam1'] and the null band instead; ~0
                means collapse toward a line.

  top2_var_frac (lambda_1+lambda_2)/sum(lambda_positive). How much of the
                geometry actually lives in the plotted plane. 1.0 for a clean
                ring at any sampling; null ~0.67. Low values mean the 2-D MDS
                picture is hiding higher-dimensional structure -- the honest
                check on being fooled by plotting only 2 dims.

  pr_mds        Participation ratio of the MDS spectrum. Perfect ring at THIS
                study's angles = 1.74 (not 2.0, same non-uniform-sampling
                reason as above); a line = 1.0; null ~3.56. Rising values
                mean genuinely higher-dimensional geometry.

  radial_cv     std/mean of each location's radius in the 2-D embedding.
                Added because it is the one ring-vs-line measure that is
                INVARIANT to angular sampling: a perfect ring gives exactly
                0.000 whether angles are uniform or clustered as they are
                here, whereas lam2_over_lam1's reference shifts with sampling.
                Ellipse ~0.12, line ~0.18, null ~0.46.

  neg_eig_frac  sum|negative eigenvalues| / sum|all eigenvalues|, computed on
                each subject's RAW RDM and averaged -- NOT on the group RDM.
                On the GROUP RDM this quantity is mathematically pinned to
                exactly 0.500 and carries no information: z-scoring forces
                sum(off-diagonals)=0 with a zero diagonal, so
                trace(B) = -0.5*[trace(D) - sum(D)/n] = 0, i.e. the eigenvalues
                sum to zero and positive mass always equals negative mass.
                (An earlier version reported the group value and duly printed
                0.50 for every single cell.) On the raw per-subject RDMs it is
                informative: crossnobis is unbiased and legitimately yields
                negative entries, and a large value means the dissimilarities
                are far from Euclidean, so the 2-D embedding and every ratio
                above should be treated as approximate.

  intersubj_r   Mean pairwise Spearman between subjects' RDMs. The GATE: a
                clean-looking group geometry with intersubj_r ~ 0 is an
                artifact of averaging, not a shared code.

IMPORTANT -- no additive constant is applied before eigendecomposition.
visual_geometry_cell's classical_mds shifted off-diagonals to be non-negative,
which is harmless for drawing coordinates but adds exactly c/2 to EVERY
centered eigenvalue (verified numerically), thereby pushing lam2/lam1 toward
1, top2_var_frac down and pr_mds up -- i.e. it would make a line look like a
ring. The spectrum here is taken from the unshifted double-centered matrix;
coordinates use the top-2 positive eigenpairs directly, which is the same
embedding up to scale since the shift leaves eigenVECTORS unchanged.

Usage:
    python plot_visual_geometry_ts.py [--voxRes 8mm] [--bands theta alpha beta]
                                       [--feature_reps ampOnly ampPhase]
                                       [--rois visual] [--subjects 1 2 ...]
                                       [--outdir <path>] [--figdir <path>]
                                       [--csvdir <path>]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from constants import SUBJECT_LIST, ANGLE_MAPPING, get_bids_root
from visual_geometry_ts_cell import output_path
from visual_geometry_cell import LOCATIONS

_BG, _FG, _GRID, _FLAG = '#000000', '#e0e0e0', '#1c1c1c', '#888888'

# Font sizes -- shared with plot_visual_geometry_epochs.py (which imports
# _style_ax from here), matching plot_circular_tgm.py's scale so every
# figure across the two scripts reads consistently: big, bold, legible.
FS_SUPTITLE   = 18
FS_PANEL_TTL  = 14
FS_AXIS_LABEL = 13
FS_ROW_LABEL  = 14
FS_TICK       = 10
FS_CBAR_LABEL = 12
FS_CBAR_TICK  = 9.5

BAND_LABELS = {'theta': 'Theta (4-8 Hz)', 'alpha': 'Alpha (8-12 Hz)',
               'beta': 'Beta (13-30 Hz)'}
FEATURE_LABELS = {'ampOnly': 'Amplitude', 'ampPhase': 'Amplitude + Phase'}
EVENT_TIMES = [(0.0, 'Stim'), (0.2, 'Delay')]

LOC_BY_ANGLE = sorted(LOCATIONS, key=lambda l: ANGLE_MAPPING[l])
ANGLES_DEG = np.array([ANGLE_MAPPING[l] for l in LOC_BY_ANGLE], dtype=float)
_ORDER = [list(LOCATIONS).index(l) for l in LOC_BY_ANGLE]

# Values a PERFECT ring sampled at THIS study's 10 (non-uniform) angles gives.
# Computed analytically from the exact angle set -- see module docstring for
# why the textbook 1.0 / 1.0 / 2.0 do not apply here.
IDEAL_RING = {'ring_r': 1.000, 'lam2_over_lam1': 0.441, 'top2_var_frac': 1.000,
              'pr_mds': 1.739, 'radial_cv': 0.000, 'neg_eig_frac': 0.0,
              'intersubj_r': None}

METRICS = [
    ('ring_r',         'Ring-ness',            'higher = more ring-like (chance ~0.38)'),
    ('lam2_over_lam1', r'$\lambda_2/\lambda_1$', 'ring=0.44 HERE (not 1); ~0 = line'),
    ('top2_var_frac',  'Top-2 var. fraction',  'is the geometry really 2-D?'),
    ('pr_mds',         'PR (MDS spectrum)',    'ring=1.74 here; higher = higher-D'),
    ('radial_cv',      'Radial CV',            'sampling-invariant; ring=0'),
    ('intersubj_r',    'Inter-subject r',      'GATE: ~0 => no shared geometry'),
]


def _style_ax(ax):
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color('#333333')
    ax.tick_params(colors=_FG, labelsize=FS_TICK)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)


# ── Geometry ─────────────────────────────────────────────────────────────────

def mds_spectrum(D):
    """
    Classical-MDS eigendecomposition of a dissimilarity matrix, WITHOUT any
    additive constant (see module docstring). Returns (eigvals_desc, coords_2d)
    where coords_2d is None if fewer than 2 positive eigenvalues exist.
    """
    D = np.array(D, dtype=float, copy=True)
    n = D.shape[0]
    np.fill_diagonal(D, 0.0)
    if not np.isfinite(D).all():
        return None, None
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D @ J
    B = (B + B.T) / 2.0
    w, v = np.linalg.eigh(B)
    o = np.argsort(w)[::-1]
    w, v = w[o], v[:, o]
    n_pos = int((w > 1e-12).sum())
    if n_pos < 1:
        return w, None
    # Pad a rank-deficient embedding with zeros rather than refusing to embed:
    # a collapsed 1-D geometry is precisely the "line, not ring" case these
    # metrics exist to detect, and returning None there would blank out
    # ring_r/radial_cv at exactly the timepoints of interest. lam2_over_lam1
    # still reports the degeneracy explicitly (it goes to 0).
    k = min(2, n_pos)
    coords = np.zeros((n, 2))
    coords[:, :k] = v[:, :k] * np.sqrt(w[:k])
    return w, coords


def ring_alignment(emb_rad, true_rad):
    """Rotation-invariant by construction; max over reflection (both are free
    parameters of MDS). See visual_geometry_cell/aggregate_visual_geometry."""
    best = -np.inf
    for s in (1.0, -1.0):
        best = max(best, float(abs(np.mean(np.exp(1j * (s * emb_rad - true_rad))))))
    return best


def geometry_metrics(D):
    """All MDS metrics for one (already angle-ordered) dissimilarity matrix."""
    out = {k: np.nan for k, _, _ in METRICS}
    out['neg_eig_frac'] = np.nan
    w, coords = mds_spectrum(D)
    if w is None:
        return out
    pos = w[w > 1e-12]
    neg = w[w < -1e-12]
    tot = pos.sum() + np.abs(neg).sum()
    out['neg_eig_frac'] = float(np.abs(neg).sum() / tot) if tot > 0 else np.nan
    if pos.size >= 1:
        out['pr_mds'] = float(pos.sum() ** 2 / (pos ** 2).sum())
    if pos.size >= 2:
        out['lam2_over_lam1'] = float(pos[1] / pos[0])
        out['top2_var_frac'] = float(pos[:2].sum() / pos.sum())
    elif pos.size == 1:
        out['lam2_over_lam1'] = 0.0
        out['top2_var_frac'] = 1.0
    if coords is not None:
        emb = np.arctan2(coords[:, 1], coords[:, 0])
        out['ring_r'] = ring_alignment(emb, np.radians(ANGLES_DEG))
        r = np.linalg.norm(coords - coords.mean(axis=0), axis=1)
        out['radial_cv'] = float(r.std() / r.mean()) if r.mean() > 1e-12 else np.nan
    return out


def _zscore_rdm(M):
    off = M[~np.eye(M.shape[0], dtype=bool)]
    if not np.isfinite(off).all() or np.nanstd(off) < 1e-12:
        return None
    Z = (M - np.nanmean(off)) / np.nanstd(off)
    np.fill_diagonal(Z, 0.0)
    return Z


def group_rdm_and_consistency(mats):
    """z-score each subject's RDM, average -> (group, mean pairwise Spearman)."""
    zs = [z for z in (_zscore_rdm(m) for m in mats) if z is not None]
    if not zs:
        return None, np.nan
    grp = np.mean(zs, axis=0)
    np.fill_diagonal(grp, 0.0)
    offm = ~np.eye(grp.shape[0], dtype=bool)
    rs = []
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            r, _ = stats.spearmanr(zs[i][offm], zs[j][offm])
            if np.isfinite(r):
                rs.append(r)
    return grp, (float(np.mean(rs)) if rs else np.nan)


# ── Loading / aggregation ────────────────────────────────────────────────────

def load_cell(subjects, bids_root, voxRes, band, fr, roi, outdir):
    rdms, nulls, tv = [], [], None
    for s in subjects:
        fp = output_path(bids_root, s, band, fr, roi, voxRes, outdir)
        if not os.path.exists(fp):
            continue
        with np.load(fp, allow_pickle=True) as npz:
            rdms.append(np.asarray(npz['rdm'], float)[:, _ORDER][:, :, _ORDER])
            if 'rdm_null' in npz.files:
                nulls.append(np.asarray(npz['rdm_null'], float)[:, :, _ORDER][:, :, :, _ORDER])
            if tv is None:
                tv = np.asarray(npz['eval_time_vector'], float)
    if not rdms:
        return None, None, None, 0
    return np.stack(rdms), (np.stack(nulls) if nulls else None), tv, len(rdms)


def metrics_over_time(rdm_stack, null_stack):
    """
    rdm_stack : (n_subj, T, 10, 10);  null_stack : (n_subj, T, n_null, 10, 10)
    Returns (real dict of (T,) arrays, null dict of (T, n_null) arrays).

    The null is built by aggregating shuffle j across ALL subjects into one
    group RDM, so it is matched to the real analysis at the same level (group
    of n_subj), not at the single-subject level.
    """
    n_subj, T = rdm_stack.shape[0], rdm_stack.shape[1]
    _KEYS = [k for k, _, _ in METRICS] + ['neg_eig_frac']
    real = {k: np.full(T, np.nan) for k in _KEYS}

    n_null = null_stack.shape[2] if null_stack is not None else 0
    # Same key set as `real` -- these two drifting apart is what previously
    # raised a KeyError in write_csv.
    null = ({k: np.full((T, n_null), np.nan) for k in _KEYS} if n_null else None)

    for t in range(T):
        grp, cons = group_rdm_and_consistency([rdm_stack[s, t] for s in range(n_subj)])
        if grp is not None:
            for k, v in geometry_metrics(grp).items():
                if k in real:
                    real[k][t] = v
        real['intersubj_r'][t] = cons
        # neg_eig_frac from the RAW per-subject RDMs -- on the group RDM it is
        # pinned to 0.5 by z-scoring and says nothing (see module docstring).
        subj_neg = [geometry_metrics(rdm_stack[s, t])['neg_eig_frac']
                    for s in range(n_subj)]
        real['neg_eig_frac'][t] = float(np.nanmean(subj_neg)) if subj_neg else np.nan

        for j in range(n_null):
            gj, cj = group_rdm_and_consistency([null_stack[s, t, j] for s in range(n_subj)])
            if gj is None:
                continue
            for k, v in geometry_metrics(gj).items():
                if k in null:
                    null[k][t, j] = v
            null['intersubj_r'][t, j] = cj
    return real, null


# ── Figures ──────────────────────────────────────────────────────────────────

def figure_metrics(results, bands, fr, roi, voxRes, figdir):
    """rows = metrics, cols = bands. Real line + shuffle null band + ideal-ring ref."""
    n_r, n_c = len(METRICS), len(bands)
    fig_h = 1.85 * n_r + 1.5
    fig = plt.figure(figsize=(4.0 * n_c + 1.0, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_r, n_c, figure=fig, hspace=0.35, wspace=0.26,
                            left=0.125, right=0.985,
                            top=1 - 1.15 / fig_h, bottom=0.55 / fig_h)

    for r, (key, label, hint) in enumerate(METRICS):
        for c, band in enumerate(bands):
            ax = fig.add_subplot(gs[r, c])
            entry = results.get((band, fr, roi))
            if entry is None or entry['tv'] is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color=_FG, fontsize=9)
                _style_ax(ax)
                continue
            tv, real, null, n_subj = entry['tv'], entry['real'], entry['null'], entry['n']

            if null is not None and np.isfinite(null[key]).any():
                lo = np.nanpercentile(null[key], 2.5, axis=1)
                hi = np.nanpercentile(null[key], 97.5, axis=1)
                ax.fill_between(tv, lo, hi, color='#666666', alpha=0.35, zorder=1,
                                 label='shuffle null (95%)')
            ideal = IDEAL_RING.get(key)
            if ideal is not None:
                ax.axhline(ideal, color='#4EA1F3', lw=0.9, ls='--', zorder=2,
                           label='perfect ring (this angle set)')
            if key == 'intersubj_r':
                ax.axhline(0.0, color='#555555', lw=0.8, ls=':', zorder=2)
            ax.plot(tv, real[key], color='#FFC629', lw=1.7, zorder=4, label='observed')

            for t_ev, _l in EVENT_TIMES:
                ax.axvline(t_ev, color=_FLAG, lw=0.7, ls='--', zorder=3)
            ax.set_xlim(tv[0], tv[-1])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax.grid(True, color=_GRID, lw=0.4, zorder=0)
            if r == 0:
                ax.set_title(BAND_LABELS.get(band, band), fontsize=FS_PANEL_TTL,
                             color=_FG, fontweight='bold')
            if r == n_r - 1:
                ax.set_xlabel('Time (s)', fontsize=FS_AXIS_LABEL, fontweight='bold')
            if c == 0:
                ax.set_ylabel(label, fontsize=FS_AXIS_LABEL, color=_FG,
                               fontweight='bold', labelpad=10)
                ax.annotate(hint, xy=(-0.32, 0.5), xycoords='axes fraction',
                            fontsize=7.5, color='#9a9a9a', ha='right', va='center',
                            rotation=90)
            if r == 0 and c == n_c - 1:
                leg = ax.legend(fontsize=9, loc='upper right', framealpha=0.25,
                                edgecolor='#444444', labelcolor=_FG)
                leg.get_frame().set_facecolor('#1a1a1a')
            _style_ax(ax)

    n_show = results.get((bands[0], fr, roi), {}).get('n', 0)
    fig.suptitle(f'Time-resolved MDS geometry  |  {FEATURE_LABELS.get(fr, fr)}  |  '
                 f'{roi.capitalize()}  (n={n_show})\n'
                 r'NB: $\lambda_2/\lambda_1$ for a perfect ring at these angles is 0.44, '
                 'not 1.0 -- read against the dashed reference line, not against 1',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1 - 0.13 / fig_h)
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'visual_geometry_ts_metrics_{fr}_{roi}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')
    return fp


def figure_ring_snapshots(results, bands, fr, roi, voxRes, figdir, n_snap=6):
    """rows = bands, cols = evenly spaced timepoints: the MDS embedding itself."""
    entry0 = results.get((bands[0], fr, roi))
    if entry0 is None or entry0['tv'] is None:
        return None
    tv = entry0['tv']
    idx = np.linspace(0, len(tv) - 1, n_snap).astype(int)
    n_r, n_c = len(bands), n_snap
    fig_h = 2.5 * n_r + 1.2
    fig = plt.figure(figsize=(2.5 * n_c + 1.0, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_r, n_c, figure=fig, hspace=0.25, wspace=0.22,
                            left=0.07, right=0.99,
                            top=1 - 0.95 / fig_h, bottom=0.35 / fig_h)
    cols = plt.cm.hsv(ANGLES_DEG / 360.0)
    for r, band in enumerate(bands):
        entry = results.get((band, fr, roi))
        for c, ti in enumerate(idx):
            ax = fig.add_subplot(gs[r, c])
            if entry is None:
                _style_ax(ax); continue
            grp = entry['group_rdm'][ti]
            _, coords = mds_spectrum(grp)
            if coords is None:
                ax.text(0.5, 0.5, 'degenerate', ha='center', va='center',
                        transform=ax.transAxes, color=_FG, fontsize=7)
                _style_ax(ax); continue
            o = np.argsort(ANGLES_DEG)
            ax.plot(coords[o, 0], coords[o, 1], '-', color='#555555', lw=0.7, zorder=1)
            ax.plot(coords[o[[-1, 0]], 0], coords[o[[-1, 0]], 1], '-',
                    color='#555555', lw=0.7, zorder=1)
            ax.scatter(coords[:, 0], coords[:, 1], c=cols, s=26, zorder=3,
                       edgecolors='k', linewidths=0.3)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"t={tv[ti]:+.2f}s  ring={entry['real']['ring_r'][ti]:.2f}",
                         fontsize=10, color=_FG, fontweight='bold', pad=3)
            if c == 0:
                ax.annotate(BAND_LABELS.get(band, band), xy=(-0.20, 0.5),
                            xycoords='axes fraction', fontsize=FS_ROW_LABEL, color=_FG,
                            ha='right', va='center', rotation=90, fontweight='bold')
            _style_ax(ax)
    fig.suptitle(f'MDS embeddings over time  |  {FEATURE_LABELS.get(fr, fr)}  |  '
                 f'{roi.capitalize()}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1 - 0.10 / fig_h)
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'visual_geometry_ts_rings_{fr}_{roi}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')
    return fp


def write_csv(results, csvdir, roi, voxRes):
    os.makedirs(csvdir, exist_ok=True)
    fp = os.path.join(csvdir, f'visual_geometry_ts_{roi}_{voxRes}.csv')
    keys = [k for k, _, _ in METRICS] + ['neg_eig_frac']
    lines = ['band,feature_rep,roi,time_s,n_subj,' + ','.join(keys) +
             ',' + ','.join(f'{k}_null_p95' for k in keys)]
    for (band, fr, r_), e in sorted(results.items()):
        if e['tv'] is None:
            continue
        for i, t in enumerate(e['tv']):
            vals = [f"{e['real'][k][i]:.6g}" for k in keys]
            if e['null'] is not None:
                nulls = [f"{np.nanpercentile(e['null'][k][i], 95):.6g}" for k in keys]
            else:
                nulls = ['' for _ in keys]
            lines.append(f'{band},{fr},{r_},{t:.4f},{e["n"]},' +
                         ','.join(vals) + ',' + ','.join(nulls))
    with open(fp, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Saved: {fp}')
    return fp


def spearman_brown(rho, n):
    """Reliability of the MEAN of n raters given mean pairwise reliability rho.
    interR is a ONE-subject-vs-ONE-subject correlation; the group average is
    far more reliable than that, which is why a pairwise r of 0.03 does not by
    itself mean the group RDM is noise."""
    if not np.isfinite(rho) or n < 2 or rho <= 0:
        return np.nan
    return float(n * rho / (1.0 + (n - 1) * rho))


def print_summary(results, bands, feature_reps, rois):
    hdr = (f"{'band':6s} {'featrep':9s} {'roi':8s} {'n':>3s} {'peak_ring':>9s} "
           f"{'@t':>7s} {'p_corr':>7s} {'l2/l1':>6s} {'top2':>5s} {'PR':>5s} "
           f"{'radCV':>6s} {'interR':>7s} {'Rgroup':>7s} {'negEig':>6s}")
    print('\n' + '=' * len(hdr))
    print('SUMMARY -- values at each cell\'s PEAK ring-ness timepoint')
    print(f"  perfect ring at THIS angle set: l2/l1={IDEAL_RING['lam2_over_lam1']:.2f} "
          f"top2={IDEAL_RING['top2_var_frac']:.2f} PR={IDEAL_RING['pr_mds']:.2f} "
          f"radCV={IDEAL_RING['radial_cv']:.2f}   (NOT 1/1/2/0 -- angles are non-uniform)")
    print('  p_corr = permutation p for the PEAK, against the max-over-TIME null')
    print('           (each shuffle contributes its own max across all timepoints).')
    print('           This is the honest test: comparing a peak picked over ~84')
    print('           windows against a PER-timepoint null is cherry-picking -- the')
    print('           per-timepoint p95 sits near 0.62 but the max-over-time p95 is')
    print('           near 0.85. Floor is 1/(n_null+1).')
    print('  interR = mean PAIRWISE (one subject vs one subject) RDM correlation.')
    print('  Rgroup = Spearman-Brown reliability of the GROUP MEAN given interR and n.')
    print('           Low interR with high Rgroup means: trust the group geometry,')
    print('           do NOT trust any single subject.')
    print('  negEig = mean over subjects of each RAW RDM\'s negative-eigenvalue mass')
    print('           (>0.4 => strongly non-Euclidean; read the 2-D picture loosely).')
    print('=' * len(hdr))
    print(hdr); print('-' * len(hdr))
    for fr in feature_reps:
        for roi in rois:
            for band in bands:
                e = results.get((band, fr, roi))
                if e is None or e['tv'] is None:
                    continue
                rr = e['real']['ring_r']
                if not np.isfinite(rr).any():
                    continue
                i = int(np.nanargmax(rr))
                # Max-over-TIME null: each shuffle contributes its own peak
                # across all timepoints, matching how the observed peak was chosen.
                if e['null'] is not None and np.isfinite(e['null']['ring_r']).any():
                    null_max = np.nanmax(e['null']['ring_r'], axis=0)   # (n_null,)
                    p_corr = (np.sum(null_max >= rr[i]) + 1) / (null_max.size + 1)
                else:
                    p_corr = np.nan
                rg = spearman_brown(e['real']['intersubj_r'][i], e['n'])
                f = lambda x, w, p=2: (f'{x:{w}.{p}f}' if np.isfinite(x) else f'{"-":>{w}}')
                print(f"{band:6s} {fr:9s} {roi:8s} {e['n']:3d} {f(rr[i],9)} "
                      f"{e['tv'][i]:+7.2f} {f(p_corr,7,3)} {f(e['real']['lam2_over_lam1'][i],6)} "
                      f"{f(e['real']['top2_var_frac'][i],5)} {f(e['real']['pr_mds'][i],5)} "
                      f"{f(e['real']['radial_cv'][i],6)} {f(e['real']['intersubj_r'][i],7)} "
                      f"{f(rg,7)} {f(e['real']['neg_eig_frac'][i],6)}")


def main():
    ap = argparse.ArgumentParser(description='Time-resolved MDS geometry across subjects.')
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--feature_reps', nargs='+', default=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual'])
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--figdir', required=True)
    ap.add_argument('--csvdir', required=True)
    args = ap.parse_args()

    bids_root = get_bids_root()
    results = {}
    for band in args.bands:
        for fr in args.feature_reps:
            for roi in args.rois:
                stack, nulls, tv, n = load_cell(args.subjects, bids_root, args.voxRes,
                                                 band, fr, roi, args.outdir)
                if stack is None:
                    results[(band, fr, roi)] = dict(tv=None, real=None, null=None,
                                                     n=0, group_rdm=None)
                    continue
                real, null = metrics_over_time(stack, nulls)
                grp = np.stack([group_rdm_and_consistency(
                    [stack[s, t] for s in range(stack.shape[0])])[0]
                    for t in range(stack.shape[1])])
                results[(band, fr, roi)] = dict(tv=tv, real=real, null=null,
                                                 n=n, group_rdm=grp)
                print(f'  aggregated {band}/{fr}/{roi}: n={n}, T={tv.size}', flush=True)

    if all(v['tv'] is None for v in results.values()):
        print('Nothing to plot -- run visual_geometry_ts_cell.py first.')
        return

    for fr in args.feature_reps:
        for roi in args.rois:
            figure_metrics(results, args.bands, fr, roi, args.voxRes, args.figdir)
            figure_ring_snapshots(results, args.bands, fr, roi, args.voxRes, args.figdir)
    write_csv(results, args.csvdir, args.rois[0], args.voxRes)
    print_summary(results, args.bands, args.feature_reps, args.rois)
    print('\nDone.')


if __name__ == '__main__':
    main()
