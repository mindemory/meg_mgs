#!/usr/bin/env python3
"""
aggregate_visual_geometry.py

Cross-subject aggregation + figures for visual_geometry_cell.py, following
Option A: raw trials are NEVER pooled across subjects (their 597-dim source
spaces share a dimensionality but not an identity, so pooling would be
meaningless). Only rotation-invariant scalars and cross-subject-safe RDM
summaries are combined.

SCALARS (radius_norm, PR, alignment angle) are averaged directly across
subjects (mean +/- SEM), per location and location-averaged. Note the
default radius reported everywhere is radius_norm (radius / mean centroid
distance) -- dimensionless, hence comparable across subjects; the raw radius
carries each subject's arbitrary amplitude units and is loaded but not
plotted (see visual_geometry_cell.py's SCALE note).

RDM is combined two ways, never by averaging raw distances (units differ
per subject):
  (a) each subject's RDM is z-scored across its own off-diagonal entries,
      then averaged -> group RDM, for visualization;
  (b) second-order RSA: every subject's RDM is Spearman-correlated with
      every other subject's (off-diagonal entries only), and the mean of
      those pairwise correlations is reported as a consistency measure --
      i.e. "are subjects' geometries even similar to each other?". A group
      RDM with near-zero inter-subject consistency is not a group effect,
      however clean its heatmap looks, so read (b) before believing (a).

RING TEST -- the 10 locations are 10 polar angles on a ring at fixed
eccentricity, so a code that respects task geometry should embed as a ring.
Classical MDS (Torgerson) on the group RDM gives a 2D embedding; ring-ness
is the circular-circular correlation (Jammalamadaka-SenGupta) between each
location's embedded angular position and its true polar angle.
  - Crossnobis estimates SQUARED distances, which is what classical MDS
    expects, so the RDM is fed in directly (not re-squared).
  - The group RDM is z-scored, so roughly half its entries are negative by
    construction. Classical MDS needs non-negative dissimilarities, so a
    constant is added to the off-diagonals to bring the minimum to 0 (the
    classic MDS additive-constant fix). This preserves the ORDER of
    dissimilarities and hence the ring test, but does distort absolute
    embedded distances -- so the embedding is read qualitatively and the
    quantitative claim rests on the angular correlation, which depends only
    on ordering around the ring.
  - MDS is invariant to rotation AND reflection, so a mirrored ring is just
    as good a recovery as an unmirrored one; the SIGNED circular
    correlation flips sign under reflection. Ring-ness is therefore
    reported as |r|, with the signed value kept alongside it.

Usage:
    python aggregate_visual_geometry.py [--voxRes 8mm] [--roi visual]
                                         [--bands theta alpha beta]
                                         [--feature_reps ampOnly ampPhase]
                                         [--subjects 1 2 ...]
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

from constants import SUBJECT_LIST, ANGLE_MAPPING, get_bids_root
from visual_geometry_cell import output_path, EPOCH_ORDER, LOCATIONS

# ── Design constants (matches the repo's other figures) ──────────────────────
_BG   = '#000000'
_FG   = '#e0e0e0'
_GRID = '#1c1c1c'

BAND_LABELS = {
    'theta': 'Theta (4-8 Hz)',
    'alpha': 'Alpha (8-12 Hz)',
    'beta':  'Beta (13-30 Hz)',
}
FEATURE_LABELS = {'ampOnly': 'Amplitude', 'ampPhase': 'Amplitude + Phase'}
EPOCH_LABELS   = {'fixation': 'Fixation', 'stim': 'Stimulus', 'delay': 'Delay'}
EPOCH_COLOURS  = {'fixation': '#8a8a8a', 'stim': '#FFC629', 'delay': '#4EA1F3'}

SCALARS = [
    ('radius_norm', 'Radius (norm.)',   'within-loc spread / centroid spread'),
    ('pr',          'Participation ratio', 'effective noise dimensionality'),
    ('align_deg',   'Alignment (deg)',  '90 = noise orthogonal to signal axis'),
]

# Locations ordered by true polar angle (ANGLE_MAPPING is already monotonic in
# location id, but this is made explicit so the RDM axes are unambiguous).
LOC_BY_ANGLE = sorted(LOCATIONS, key=lambda l: ANGLE_MAPPING[l])
ANGLES_DEG   = np.array([ANGLE_MAPPING[l] for l in LOC_BY_ANGLE], dtype=float)

# Chance-level reference for ring_alignment's rho, obtained by simulation
# (3000 draws of 10 random points in 5-D, classical-MDS-embedded and scored
# exactly as real cells are): mean 0.38, 95th pct 0.61, 99th pct 0.72.
# Ring-ness is bounded in [0, 1] and is NOT zero under the null -- 10 points
# land at ~0.38 by chance alone -- so a value must clear ~0.61 before it is
# even nominally better than random geometry. Printed alongside the summary
# so the number is never read as if 0 were its baseline.
RINGNESS_NULL_MEAN = 0.38
RINGNESS_NULL_P95  = 0.61
RINGNESS_NULL_P99  = 0.72


def _style_ax(ax, spine_col='#333333'):
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color(spine_col)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)


# ── Loading ──────────────────────────────────────────────────────────────────

def load_group(subjects, bids_root, voxRes, roi, bands, feature_reps, outdir):
    """Returns data[(band, feature_rep, epoch)] = list of per-subject dicts."""
    data = {}
    for band in bands:
        for fr in feature_reps:
            for epoch in EPOCH_ORDER:
                cells = []
                for subjID in subjects:
                    fp = output_path(bids_root, subjID, band, fr, epoch, roi, voxRes, outdir)
                    if not os.path.exists(fp):
                        continue
                    with np.load(fp, allow_pickle=True) as npz:
                        cells.append({k: npz[k] for k in npz.files})
                data[(band, fr, epoch)] = cells
    return data


# ── Group RDM / second-order RSA ─────────────────────────────────────────────

def _offdiag(M):
    return M[~np.eye(M.shape[0], dtype=bool)]


def group_rdm(cells, key='rdm_primary'):
    """
    z-score each subject's RDM across its own off-diagonal entries, then
    average -> (group_rdm, n_subj, mean_pairwise_spearman, sem_pairwise).
    Returns (None, 0, nan, nan) if nothing usable.
    """
    mats = []
    for c in cells:
        M = np.asarray(c[key], dtype=float)
        if M.shape != (len(LOCATIONS), len(LOCATIONS)):
            continue
        # reorder to polar-angle order
        order = [list(LOCATIONS).index(l) for l in LOC_BY_ANGLE]
        M = M[np.ix_(order, order)]
        off = _offdiag(M)
        if not np.isfinite(off).all() or np.nanstd(off) < 1e-12:
            continue
        Mz = M.copy().astype(float)
        mu, sd = np.nanmean(off), np.nanstd(off)
        Mz = (Mz - mu) / sd
        np.fill_diagonal(Mz, 0.0)
        mats.append(Mz)

    if not mats:
        return None, 0, np.nan, np.nan

    stack = np.stack(mats)
    grp = stack.mean(axis=0)
    np.fill_diagonal(grp, 0.0)

    # Second-order RSA: pairwise Spearman across subjects (off-diagonals).
    rs = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            r, _ = stats.spearmanr(_offdiag(mats[i]), _offdiag(mats[j]))
            if np.isfinite(r):
                rs.append(r)
    if rs:
        mean_r = float(np.mean(rs))
        sem_r = float(np.std(rs) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0
    else:
        mean_r, sem_r = np.nan, np.nan
    return grp, len(mats), mean_r, sem_r


# ── Ring test ────────────────────────────────────────────────────────────────

def classical_mds(D, n_components=2):
    """
    Torgerson classical MDS on a (possibly z-scored, hence partly negative)
    SQUARED-distance matrix. Off-diagonals are shifted to a minimum of 0
    first (MDS additive-constant fix; preserves dissimilarity ORDER -- see
    module docstring). Returns (coords (n, n_components), eigvals) or
    (None, None) if degenerate.
    """
    D = np.array(D, dtype=float, copy=True)
    n = D.shape[0]
    np.fill_diagonal(D, 0.0)
    off_mask = ~np.eye(n, dtype=bool)
    if not np.isfinite(D[off_mask]).all():
        return None, None
    m = D[off_mask].min()
    if m < 0:
        D[off_mask] -= m
    np.fill_diagonal(D, 0.0)

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D @ J
    B = (B + B.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    pos = eigvals[:n_components] > 1e-12
    if pos.sum() < n_components:
        return None, eigvals
    coords = eigvecs[:, :n_components] * np.sqrt(eigvals[:n_components])
    return coords, eigvals


def circ_corr(a_rad, b_rad):
    """
    Jammalamadaka-SenGupta circular-circular correlation. Kept as a
    secondary diagnostic ONLY -- it is NOT used for ring-ness, because it
    depends on deviations from each set's circular mean, and this study's 10
    target angles are almost perfectly spread around the circle (resultant
    length ~0), leaving their circular mean numerically undefined. Under
    that degeneracy the statistic swings with the arbitrary orientation MDS
    happens to return (measured: 0.09-1.00 across rotations of one and the
    same perfect ring), which would make it meaningless here. See
    ring_alignment for the metric actually used.
    """
    a_bar = np.arctan2(np.mean(np.sin(a_rad)), np.mean(np.cos(a_rad)))
    b_bar = np.arctan2(np.mean(np.sin(b_rad)), np.mean(np.cos(b_rad)))
    sa, sb = np.sin(a_rad - a_bar), np.sin(b_rad - b_bar)
    den = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2))
    return float(np.sum(sa * sb) / den) if den > 1e-12 else np.nan


def ring_alignment(emb_rad, true_rad):
    """
    Ring-ness = mean resultant length of the angular discrepancies
    (embedded angle - true polar angle), maximized over reflection:

        rho = max_over_sign | mean_l exp( i * (sign*emb_l - true_l) ) |

    Rotation-invariant BY CONSTRUCTION (a constant offset shifts every
    discrepancy equally and so cannot change the resultant length), which is
    exactly what is required here since classical MDS returns an arbitrary
    rotation; reflection is likewise a free parameter of MDS, hence the max
    over sign. rho = 1 means the embedding tracks true polar angle perfectly
    up to rigid rotation/reflection; rho ~ 1/sqrt(10) ~ 0.32 is the
    chance-level scale for 10 points.

    Returns (rho, reflected_flag).
    """
    best_rho, best_sign = -np.inf, 1
    for sign in (1.0, -1.0):
        d = sign * emb_rad - true_rad
        rho = float(abs(np.mean(np.exp(1j * d))))
        if rho > best_rho:
            best_rho, best_sign = rho, sign
    return best_rho, bool(best_sign < 0)


def ring_test(grp_rdm):
    """
    Returns (coords, ringness, ringness_signed_circcorr, embed_angles_deg).
    ringness is ring_alignment's rotation-invariant rho (the quantity to
    report); the Jammalamadaka correlation is returned alongside it purely
    as a diagnostic and should not be interpreted on its own here.
    """
    coords, _ = classical_mds(grp_rdm, n_components=2)
    if coords is None:
        return None, np.nan, np.nan, None
    emb = np.arctan2(coords[:, 1], coords[:, 0])
    true_rad = np.radians(ANGLES_DEG)
    rho, _ = ring_alignment(emb, true_rad)
    return coords, rho, circ_corr(emb, true_rad), np.degrees(np.mod(emb, 2 * np.pi))


# ── Scalar aggregation ───────────────────────────────────────────────────────

def scalar_stats(cells, field):
    """
    Returns (per_loc_mean (10,), per_loc_sem (10,), subj_locavg (n_subj,)).
    subj_locavg is each subject's location-averaged value -- the unit of
    analysis for the paired epoch tests.
    """
    order = [list(LOCATIONS).index(l) for l in LOC_BY_ANGLE]
    rows = []
    for c in cells:
        v = np.asarray(c[field], dtype=float)
        if v.shape != (len(LOCATIONS),):
            continue
        rows.append(v[order])
    if not rows:
        return (np.full(len(LOCATIONS), np.nan), np.full(len(LOCATIONS), np.nan),
                np.array([]))
    M = np.stack(rows)
    n = M.shape[0]
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(M, axis=0)
        sem = np.nanstd(M, axis=0) / np.sqrt(max(n, 1))
        locavg = np.nanmean(M, axis=1)
    return mean, sem, locavg


def paired_epoch_test(data, band, fr, field, epoch_a='fixation', epoch_b='delay'):
    """Paired test of location-averaged `field`, matched by subjID. Descriptive."""
    ca = {int(c['subjID'][0]): c for c in data.get((band, fr, epoch_a), [])}
    cb = {int(c['subjID'][0]): c for c in data.get((band, fr, epoch_b), [])}
    common = sorted(set(ca) & set(cb))
    if len(common) < 3:
        return np.nan, np.nan, len(common)
    order = [list(LOCATIONS).index(l) for l in LOC_BY_ANGLE]
    a = np.array([np.nanmean(np.asarray(ca[s][field], float)[order]) for s in common])
    b = np.array([np.nanmean(np.asarray(cb[s][field], float)[order]) for s in common])
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3:
        return np.nan, np.nan, int(good.sum())
    t, p = stats.ttest_rel(a[good], b[good])
    return float(t), float(p), int(good.sum())


# ── Figures ──────────────────────────────────────────────────────────────────

def _draw_rdm(ax, grp, title, show_ticks=True):
    if grp is None:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, color=_FG, fontsize=9)
        _style_ax(ax)
        return None
    v = np.nanmax(np.abs(_offdiag(grp))) if np.isfinite(_offdiag(grp)).any() else 1.0
    im = ax.imshow(grp, cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
    if show_ticks:
        ax.set_xticks(range(len(LOC_BY_ANGLE)))
        ax.set_yticks(range(len(LOC_BY_ANGLE)))
        ax.set_xticklabels([f'{int(a)}' for a in ANGLES_DEG], fontsize=5.5, rotation=90)
        ax.set_yticklabels([f'{int(a)}' for a in ANGLES_DEG], fontsize=5.5)
    else:
        ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, color=_FG, pad=3)
    _style_ax(ax)
    return im


def _draw_ring(ax, coords, ringness, title):
    if coords is None:
        ax.text(0.5, 0.5, 'MDS degenerate', ha='center', va='center',
                transform=ax.transAxes, color=_FG, fontsize=8)
        _style_ax(ax)
        return
    # Colour by TRUE polar angle so a recovered ring shows as a smooth hue cycle.
    cols = plt.cm.hsv(ANGLES_DEG / 360.0)
    order = np.argsort(ANGLES_DEG)
    ax.plot(coords[order, 0], coords[order, 1], '-', color='#555555', lw=0.8, zorder=1)
    ax.plot(coords[order[[-1, 0]], 0], coords[order[[-1, 0]], 1], '-',
            color='#555555', lw=0.8, zorder=1)
    ax.scatter(coords[:, 0], coords[:, 1], c=cols, s=42, zorder=3, edgecolors='k', linewidths=0.4)
    for i, a in enumerate(ANGLES_DEG):
        ax.annotate(f'{int(a)}', (coords[i, 0], coords[i, 1]), fontsize=5.5,
                    color=_FG, ha='left', va='bottom', zorder=4)
    ax.axhline(0, color=_GRID, lw=0.5, zorder=0)
    ax.axvline(0, color=_GRID, lw=0.5, zorder=0)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{title}\nring r={ringness:.2f}' if np.isfinite(ringness) else title,
                 fontsize=8, color=_FG, pad=3)
    _style_ax(ax)


def figure_grid(data, bands, fr, voxRes, roi, figdir, kind='rdm'):
    """rows = bands, cols = epochs; one figure per feature rep. kind: 'rdm'|'ring'."""
    n_rows, n_cols = len(bands), len(EPOCH_ORDER)
    fig_w, fig_h = 3.0 * n_cols + 1.2, 3.0 * n_rows + 1.0
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.3,
                            left=0.10, right=0.93, top=0.88, bottom=0.06)
    im = None
    for r, band in enumerate(bands):
        for c, epoch in enumerate(EPOCH_ORDER):
            ax = fig.add_subplot(gs[r, c])
            cells = data.get((band, fr, epoch), [])
            grp, n_subj, mean_r, _ = group_rdm(cells)
            if kind == 'rdm':
                im = _draw_rdm(ax, grp, f'{EPOCH_LABELS[epoch]}  (n={n_subj}, r={mean_r:.2f})'
                               if n_subj else f'{EPOCH_LABELS[epoch]}  (n=0)',
                               show_ticks=True) or im
            else:
                coords, ringness, _, _ = ring_test(grp) if grp is not None else (None, np.nan, np.nan, None)
                _draw_ring(ax, coords, ringness, f'{EPOCH_LABELS[epoch]}  (n={n_subj})')
            if c == 0:
                ax.annotate(BAND_LABELS.get(band, band), xy=(-0.30, 0.5),
                            xycoords='axes fraction', fontsize=10, color=_FG,
                            ha='right', va='center', rotation=90, fontweight='bold')

    kind_title = ('Group RDM (z-scored per subject, then averaged)' if kind == 'rdm'
                  else 'Ring test -- classical MDS of the group RDM')
    fig.suptitle(f'{kind_title}\n{FEATURE_LABELS.get(fr, fr)}  |  {roi.capitalize()}  |  '
                 f'{voxRes}  |  axes/labels = true polar angle (deg)'
                 + ('  |  r = mean inter-subject RDM correlation' if kind == 'rdm' else ''),
                 color=_FG, fontsize=11, fontweight='bold', y=0.975)
    if kind == 'rdm' and im is not None:
        cax = fig.add_axes([0.945, 0.20, 0.012, 0.5])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('z-scored dissimilarity', color=_FG, fontsize=8)
        cb.ax.tick_params(colors=_FG, labelsize=7)
        cb.outline.set_edgecolor('#333333')

    os.makedirs(figdir, exist_ok=True)
    fpath = os.path.join(figdir, f'visual_geometry_{kind}_grid_{fr}_{roi}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


def figure_scalars(data, bands, feature_reps, voxRes, roi, figdir):
    """rows = scalars, cols = bands; grouped bars over epochs, one line per feature rep."""
    n_rows, n_cols = len(SCALARS), len(bands)
    fig = plt.figure(figsize=(4.2 * n_cols + 1.0, 3.0 * n_rows + 0.8), facecolor=_BG)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.32,
                            left=0.09, right=0.98, top=0.89, bottom=0.07)

    width = 0.36
    for r, (field, label, subtitle) in enumerate(SCALARS):
        for c, band in enumerate(bands):
            ax = fig.add_subplot(gs[r, c])
            xs = np.arange(len(EPOCH_ORDER))
            for fi, fr in enumerate(feature_reps):
                means, sems = [], []
                for epoch in EPOCH_ORDER:
                    _, _, locavg = scalar_stats(data.get((band, fr, epoch), []), field)
                    if locavg.size:
                        means.append(np.nanmean(locavg))
                        sems.append(np.nanstd(locavg) / np.sqrt(max(locavg.size, 1)))
                    else:
                        means.append(np.nan); sems.append(np.nan)
                offs = (fi - (len(feature_reps) - 1) / 2) * width
                colour = '#FFC629' if fr == 'ampOnly' else '#4EA1F3'
                ax.bar(xs + offs, means, width=width, yerr=sems, capsize=2.5,
                       color=colour, alpha=0.85, label=FEATURE_LABELS.get(fr, fr),
                       error_kw=dict(ecolor=_FG, lw=0.9))
            if field == 'align_deg':
                ax.axhline(90, color='#888888', ls=':', lw=0.9, zorder=0)
            ax.set_xticks(xs)
            ax.set_xticklabels([EPOCH_LABELS[e] for e in EPOCH_ORDER], fontsize=8)
            ax.set_ylabel(label, fontsize=8, color=_FG)
            ax.grid(True, axis='y', color=_GRID, lw=0.4, zorder=0)
            if r == 0:
                ax.set_title(BAND_LABELS.get(band, band), fontsize=10,
                             color=_FG, fontweight='bold')
            if r == 0 and c == 0:
                leg = ax.legend(fontsize=7.5, loc='best', framealpha=0.2,
                                edgecolor='#444444', labelcolor=_FG)
                leg.get_frame().set_facecolor('#1a1a1a')
            if c == 0:
                ax.annotate(subtitle, xy=(-0.16, 0.5), xycoords='axes fraction',
                            fontsize=6.5, color='#999999', ha='right', va='center',
                            rotation=90)
            _style_ax(ax)

    fig.suptitle(f'Noise / alignment scalars across epochs  |  {roi.capitalize()}  |  {voxRes}\n'
                 f'mean +/- SEM across subjects (each subject location-averaged first)',
                 color=_FG, fontsize=11, fontweight='bold', y=0.965)
    os.makedirs(figdir, exist_ok=True)
    fpath = os.path.join(figdir, f'visual_geometry_scalars_{roi}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


# ── CSV / summary ────────────────────────────────────────────────────────────

def write_long_csv(data, csvdir, roi, voxRes):
    os.makedirs(csvdir, exist_ok=True)
    fpath = os.path.join(csvdir, f'visual_geometry_per_subject_{roi}_{voxRes}.csv')
    cols = ['subjID', 'band', 'feature_rep', 'epoch', 'location', 'angle_deg',
            'radius', 'radius_norm', 'pr', 'align_deg', 'n_per_loc', 'pca_dim',
            'pca_explained_var', 'whitened', 'n_trials', 'n_features']
    lines = [','.join(cols)]
    for (band, fr, epoch), cells in sorted(data.items()):
        for c in cells:
            subjID = int(c['subjID'][0])
            for li, loc in enumerate(LOCATIONS):
                lines.append(','.join([
                    str(subjID), band, fr, epoch, str(loc), str(ANGLE_MAPPING[loc]),
                    f"{float(c['radius'][li]):.6g}", f"{float(c['radius_norm'][li]):.6g}",
                    f"{float(c['pr'][li]):.6g}", f"{float(c['align_deg'][li]):.6g}",
                    str(int(c['n_per_loc'][0])), str(int(c['pca_dim'][0])),
                    f"{float(c['pca_explained_var'][0]):.6g}",
                    str(bool(c['rdm_primary_whitened'][0])),
                    str(int(c['n_trials'][0])), str(int(c['n_features'][0])),
                ]))
    with open(fpath, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Saved: {fpath}')
    return fpath


def build_summary(data, bands, feature_reps):
    """One row per cell: consistency, ring-ness, and location-averaged scalars."""
    rows = []
    for band in bands:
        for fr in feature_reps:
            for epoch in EPOCH_ORDER:
                cells = data.get((band, fr, epoch), [])
                grp, n_subj, mean_r, sem_r = group_rdm(cells)
                _, ringness, ring_signed, _ = ring_test(grp) if grp is not None else (None, np.nan, np.nan, None)
                row = dict(band=band, feature_rep=fr, epoch=epoch, n_subj=n_subj,
                           intersubj_r=mean_r, intersubj_r_sem=sem_r,
                           ringness=ringness, ringness_signed=ring_signed)
                for field, _, _ in SCALARS:
                    _, _, locavg = scalar_stats(cells, field)
                    row[f'{field}_mean'] = float(np.nanmean(locavg)) if locavg.size else np.nan
                    row[f'{field}_sem'] = (float(np.nanstd(locavg) / np.sqrt(locavg.size))
                                            if locavg.size else np.nan)
                whit = [bool(c['rdm_primary_whitened'][0]) for c in cells]
                pcad = [int(c['pca_dim'][0]) for c in cells]
                nploc = [int(c['n_per_loc'][0]) for c in cells]
                row['frac_whitened'] = float(np.mean(whit)) if whit else np.nan
                row['pca_dim_median'] = float(np.median(pcad)) if pcad else np.nan
                row['n_per_loc_median'] = float(np.median(nploc)) if nploc else np.nan
                rows.append(row)
    return rows


def write_summary_csv(rows, csvdir, roi, voxRes):
    os.makedirs(csvdir, exist_ok=True)
    fpath = os.path.join(csvdir, f'visual_geometry_summary_{roi}_{voxRes}.csv')
    if not rows:
        return None
    cols = list(rows[0].keys())
    lines = [','.join(cols)]
    for r in rows:
        lines.append(','.join(
            (f'{r[c]:.6g}' if isinstance(r[c], float) else str(r[c])) for c in cols))
    with open(fpath, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Saved: {fpath}')
    return fpath


def print_summary(rows, data, bands, feature_reps):
    hdr = (f"{'band':6s} {'featrep':9s} {'epoch':9s} {'n':>3s} {'interR':>7s} "
           f"{'ring':>6s} {'radN':>7s} {'PR':>6s} {'align':>6s} {'k':>4s} {'nLoc':>5s} {'whit':>5s}")
    print('\n' + '=' * len(hdr))
    print('SUMMARY -- one row per cell')
    print('  interR = mean inter-subject RDM Spearman (consistency; ~0 => no shared geometry)')
    print(f'  ring   = rotation/reflection-invariant ring-ness in [0,1]. CHANCE IS NOT 0: '
          f'null mean={RINGNESS_NULL_MEAN:.2f}, 95th pct={RINGNESS_NULL_P95:.2f}, '
          f'99th pct={RINGNESS_NULL_P99:.2f}')
    print('  radN   = radius_norm (dimensionless), PR = participation ratio,')
    print('  align  = alignment angle in deg (90 = noise orthogonal to signal axis)')
    print('  k      = median PCA dim, nLoc = median trials/location used, whit = frac whitened')
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        def f(x, w, p=2):
            return (f'{x:{w}.{p}f}' if isinstance(x, float) and np.isfinite(x) else f'{"-":>{w}}')
        print(f"{r['band']:6s} {r['feature_rep']:9s} {r['epoch']:9s} {r['n_subj']:3d} "
              f"{f(r['intersubj_r'],7)} {f(r['ringness'],6)} {f(r['radius_norm_mean'],7)} "
              f"{f(r['pr_mean'],6,1)} {f(r['align_deg_mean'],6,1)} "
              f"{f(r['pca_dim_median'],4,0)} {f(r['n_per_loc_median'],5,0)} "
              f"{f(r['frac_whitened'],5)}")

    print('\n' + '=' * 78)
    print('PAIRED delay vs fixation (location-averaged per subject; descriptive, uncorrected)')
    print('=' * 78)
    print(f"{'band':6s} {'featrep':9s} {'scalar':16s} {'t':>8s} {'p':>9s} {'n':>4s}")
    print('-' * 78)
    for band in bands:
        for fr in feature_reps:
            for field, label, _ in SCALARS:
                t, p, n = paired_epoch_test(data, band, fr, field)
                ts = f'{t:8.2f}' if np.isfinite(t) else f'{"-":>8}'
                ps = f'{p:9.4f}' if np.isfinite(p) else f'{"-":>9}'
                print(f'{band:6s} {fr:9s} {label:16s} {ts} {ps} {n:4d}')


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate + plot visual_geometry_cell.py across subjects (Option A).')
    parser.add_argument('--voxRes', default='8mm')
    parser.add_argument('--roi', default='visual')
    parser.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    parser.add_argument('--feature_reps', nargs='+', default=['ampOnly', 'ampPhase'])
    parser.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    parser.add_argument('--outdir', default=None,
                         help='Directory holding the per-subject .npz files.')
    parser.add_argument('--figdir', required=True)
    parser.add_argument('--csvdir', required=True)
    args = parser.parse_args()

    bids_root = get_bids_root()
    print(f'Loading | roi={args.roi} | {args.voxRes} | bands={args.bands} | '
          f'feature_reps={args.feature_reps} | {len(args.subjects)} subjects')
    data = load_group(args.subjects, bids_root, args.voxRes, args.roi,
                      args.bands, args.feature_reps, args.outdir)
    total = sum(len(v) for v in data.values())
    print(f'Loaded {total} subject-cells across {len(data)} cells '
          f'({len(args.bands)} bands x {len(args.feature_reps)} feature reps x '
          f'{len(EPOCH_ORDER)} epochs).')
    if total == 0:
        print('Nothing to aggregate -- run visual_geometry_cell.py first.')
        return

    for fr in args.feature_reps:
        figure_grid(data, args.bands, fr, args.voxRes, args.roi, args.figdir, kind='rdm')
        figure_grid(data, args.bands, fr, args.voxRes, args.roi, args.figdir, kind='ring')
    figure_scalars(data, args.bands, args.feature_reps, args.voxRes, args.roi, args.figdir)

    write_long_csv(data, args.csvdir, args.roi, args.voxRes)
    rows = build_summary(data, args.bands, args.feature_reps)
    write_summary_csv(rows, args.csvdir, args.roi, args.voxRes)
    print_summary(rows, data, args.bands, args.feature_reps)

    print('\nDone.')


if __name__ == '__main__':
    main()
