#!/usr/bin/env python3
"""
plot_visual_geometry_epochs.py

Group RDM + MDS geometry figures for visual_geometry_epochs_cell.py, over the
four task epochs (fixation / stimulus / early delay / late delay).

Per condition it writes two figures, each rows = bands x cols = epochs:
  1. ..._rdm_...png  -- the group RDM heatmaps (locations ordered by polar
     angle, so a ring code shows as blue near the diagonal and red in the
     corners).
  2. ..._mds_...png  -- the classical-MDS embedding of each group RDM. Each
     panel title carries ring-ness, lambda2/lambda1 and the top-2 variance
     fraction, because the embedding alone cannot tell you whether you are
     looking at a real ring, a line, or a 2-D shadow of higher-dimensional
     geometry.

READING THE MDS PANELS -- the three numbers in each title, in order:
  ring   ring-ness in [0,1]. CHANCE IS ~0.38, p95 ~0.61, NOT 0. And it can be
         fooled: a pure 1-D line scores ~0.82 in simulation, which is why the
         next two numbers are printed beside it rather than left in a table.
  l2/l1  lambda_2/lambda_1. For a PERFECT ring at THIS study's 10 non-uniform
         angles this is 0.441, NOT 1.0 (the textbook "~1" assumes uniform
         spacing; random geometry here averages 0.641, i.e. ABOVE the
         perfect-ring value). Near 0 = collapsed toward a line.
  top2   (lambda_1+lambda_2)/sum(positive lambda). How much of the geometry
         actually lives in the plotted plane; 1.0 for a clean ring, ~0.67
         under the null. Low values mean the 2-D picture is hiding structure.
The reference values for a perfect ring at these angles are printed in each
figure's subtitle so the panel numbers can be read against them directly.

Colour encodes TRUE polar angle and is shown once as a shared legend, so the
per-point text labels of the earlier version are gone. The connecting path
runs through the locations in TRUE angular order with its colour progressing
along that order: if the embedding preserves the ring, the path traces a
simple loop and its colour cycles smoothly around it; if not, the path
self-crosses and the colour order scrambles.

Usage:
    python plot_visual_geometry_epochs.py [--voxRes 8mm]
        [--bands theta alpha beta lowgamma highgamma]
        [--conditions ampOnly ampPhase] [--rois visual parietal frontal]
        [--subjects 1 2 ...] [--outdir <path>] [--figdir <path>] [--csvdir <path>]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from constants import SUBJECT_LIST, ANGLE_MAPPING, get_bids_root
from visual_geometry_epochs_cell import output_path, EPOCHS, EPOCH_ORDER
from visual_geometry_cell import LOCATIONS
from plot_visual_geometry_ts import (
    mds_spectrum, geometry_metrics, group_rdm_and_consistency,
    spearman_brown, IDEAL_RING, LOC_BY_ANGLE, ANGLES_DEG, _ORDER, _style_ax,
    _BG, _FG, _GRID,
    FS_SUPTITLE, FS_PANEL_TTL, FS_AXIS_LABEL, FS_ROW_LABEL, FS_TICK,
    FS_CBAR_LABEL, FS_CBAR_TICK,
)

# The MDS panel titles pack 3 metrics into 2 lines inside a small (~2.5in)
# panel -- FS_PANEL_TTL is too big to fit there without overlapping the plot
# above it, so this gets its own, still-legible-but-denser size.
FS_MDS_METRICS = 10.5

BAND_LABELS = {'theta': 'Theta (4-8 Hz)', 'alpha': 'Alpha (8-12 Hz)',
               'beta': 'Beta (13-30 Hz)', 'lowgamma': 'Low gamma (30-80 Hz)',
               'highgamma': 'High gamma (80-150 Hz)'}
COND_LABELS = {'ampOnly': 'Amplitude', 'ampPhase': 'Amplitude + Phase'}
EPOCH_LABELS = {'fixation': 'Fixation', 'stimulus': 'Stimulus',
                'early_delay': 'Early delay', 'late_delay': 'Late delay'}

# Colour = true polar angle. hsv is cyclic, which is what an angular variable
# needs -- a sequential map would put 0 and 335 deg at opposite ends of the
# scale even though they are 25 deg apart on the screen.
LOC_COLOURS = plt.cm.hsv(ANGLES_DEG / 360.0)


def load_group(subjects, bids_root, voxRes, band, cond, roi, outdir, bin_name='all'):
    """
    Returns (rdm_stack (n_subj,n_ep,10,10), null_stack or None, n_subj) for one
    performance bin. Files written before binning existed have no bin axis and
    are treated as the single 'all' bin.
    """
    rdms, nulls, dropped = [], [], []
    bin_frac = np.nan
    for s in subjects:
        fp = output_path(bids_root, s, band, cond, roi, voxRes, outdir)
        if not os.path.exists(fp):
            continue
        with np.load(fp, allow_pickle=True) as npz:
            if 'bin_frac' in npz.files:
                bin_frac = float(np.asarray(npz['bin_frac']).ravel()[0])
            r = np.asarray(npz['rdm'], float)
            nl = np.asarray(npz['rdm_null'], float) if 'rdm_null' in npz.files else None
            if 'bins' in npz.files and r.ndim == 4:
                names = [str(b) for b in npz['bins']]
                if bin_name not in names:
                    continue
                bi = names.index(bin_name)
                r = r[bi]
                nl = nl[bi] if nl is not None else None
            elif bin_name != 'all':
                continue          # pre-binning file: only 'all' is available
            if not np.isfinite(r).all():
                # A location fell below the crossnobis minimum for this
                # subject/bin, so its RDM row/col is NaN. Counted and reported
                # rather than silently skipped -- with 3 bins this drops a large
                # share of subjects, which is invisible otherwise.
                dropped.append(s)
                continue
            rdms.append(r[:, _ORDER][:, :, _ORDER])
            if nl is not None:
                nulls.append(nl[:, :, _ORDER][:, :, :, _ORDER])
    if dropped:
        print(f'    [{band}/{cond}/{roi}/{bin_name}] dropped {len(dropped)} subject(s) '
              f'with an under-populated location: '
              f'{", ".join(f"sub-{d:02d}" for d in dropped)}', flush=True)
    if not rdms:
        return None, None, 0, bin_frac
    return np.stack(rdms), (np.stack(nulls) if nulls else None), len(rdms), bin_frac


def cell_metrics(rdm_stack, null_stack):
    """Per-epoch group RDM + metrics + max-over-nothing p (epochs judged singly)."""
    n_subj, n_ep = rdm_stack.shape[0], rdm_stack.shape[1]
    out = []
    for e in range(n_ep):
        grp, cons = group_rdm_and_consistency([rdm_stack[s, e] for s in range(n_subj)])
        m = geometry_metrics(grp) if grp is not None else {}
        m['intersubj_r'] = cons
        m['Rgroup'] = spearman_brown(cons, n_subj)
        p = np.nan
        if null_stack is not None and grp is not None:
            nulls = []
            for j in range(null_stack.shape[2]):
                gj, _ = group_rdm_and_consistency(
                    [null_stack[s, e, j] for s in range(n_subj)])
                if gj is not None:
                    nulls.append(geometry_metrics(gj)['ring_r'])
            if nulls:
                nulls = np.array(nulls)
                p = (np.sum(nulls >= m.get('ring_r', np.nan)) + 1) / (nulls.size + 1)
        m['p_ring'] = p
        out.append((grp, m))
    return out


# ── Figures ──────────────────────────────────────────────────────────────────

def _legend_handles():
    return [Line2D([0], [0], marker='o', linestyle='', markersize=6,
                   markerfacecolor=LOC_COLOURS[i], markeredgecolor='k',
                   markeredgewidth=0.4, label=f'{int(a)}°')
            for i, a in enumerate(ANGLES_DEG)]


def _draw_ring(ax, coords, title):
    if coords is None:
        ax.text(0.5, 0.5, 'degenerate', ha='center', va='center',
                transform=ax.transAxes, color=_FG, fontsize=10)
        _style_ax(ax); ax.set_xticks([]); ax.set_yticks([])
        return
    o = np.argsort(ANGLES_DEG)
    loop = np.append(o, o[0])                       # close the ring
    pts = coords[loop]
    # Colour the path along TRUE angular order: a preserved ring gives a simple
    # loop whose hue cycles smoothly; a scrambled one self-crosses and the hue
    # order breaks. Drawn as a dark halo + coloured line so it reads on black.
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    ax.add_collection(LineCollection(segs, colors='#000000', linewidths=3.4,
                                      zorder=2, capstyle='round'))
    ax.add_collection(LineCollection(segs, colors=LOC_COLOURS[loop[:-1]],
                                      linewidths=1.8, alpha=0.95, zorder=3,
                                      capstyle='round'))
    ax.scatter(coords[:, 0], coords[:, 1], c=LOC_COLOURS, s=64, zorder=4,
               edgecolors='k', linewidths=0.6)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=FS_MDS_METRICS, color=_FG, fontweight='bold', pad=4)
    ax.margins(0.16)
    _style_ax(ax)


def figure_mds(results, bands, cond, roi, voxRes, figdir, bin_name='all'):
    bands = [b for b in bands if results.get((b, cond, roi, bin_name), {}).get('n', 0) > 0]
    if not bands:
        return None
    n_r, n_c = len(bands), len(EPOCH_ORDER)
    # Extra header room (2.6in vs the old 1.7in) for the bigger fonts below --
    # a brief bold suptitle, a smaller reference-values line, and the epoch
    # column headers all have to fit without colliding.
    fig_h = 2.55 * n_r + 2.6
    fig = plt.figure(figsize=(2.55 * n_c + 2.4, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_r, n_c, figure=fig, hspace=0.34, wspace=0.16,
                            left=0.075, right=0.845,
                            top=1 - 2.0 / fig_h, bottom=0.35 / fig_h)
    for r, band in enumerate(bands):
        e = results[(band, cond, roi, bin_name)]
        for c, ep in enumerate(EPOCH_ORDER):
            ax = fig.add_subplot(gs[r, c])
            grp, m = e['cells'][c]
            _, coords = mds_spectrum(grp) if grp is not None else (None, None)
            ttl = (f"ring={m.get('ring_r', np.nan):.2f}  "
                   f"$\\lambda_2/\\lambda_1$={m.get('lam2_over_lam1', np.nan):.2f}\n"
                   f"top2={m.get('top2_var_frac', np.nan):.2f}")
            _draw_ring(ax, coords, ttl)
            if r == 0:
                ax.text(0.5, 1.16, EPOCH_LABELS[ep], transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=FS_PANEL_TTL, color=_FG,
                        fontweight='bold')
            if c == 0:
                ax.annotate(f"{BAND_LABELS.get(band, band)}\n(n={e['n']})",
                            xy=(-0.16, 0.5), xycoords='axes fraction',
                            fontsize=FS_ROW_LABEL, color=_FG, ha='right', va='center',
                            rotation=90, fontweight='bold')

    leg = fig.legend(handles=_legend_handles(), loc='center left',
                     bbox_to_anchor=(0.855, 0.5), fontsize=10, framealpha=0.25,
                     edgecolor='#444444', labelcolor=_FG, title='Target angle')
    leg.get_frame().set_facecolor('#1a1a1a')
    leg.get_title().set_color(_FG)
    leg.get_title().set_fontsize(FS_AXIS_LABEL)

    # Brief bold headline, then the ring/lambda2/lambda1/top2 reference values
    # as a smaller, separate line -- keeping both (per feedback: brief titles,
    # but keep the metric reference numbers), just not sharing one font size.
    fig.suptitle(
        f'MDS geometry by epoch  |  {COND_LABELS.get(cond, cond)}  |  '
        f'{"" if bin_name=="all" else "bin=" + bin_name + "  |  "}'
        f'{roi.capitalize()}',
        color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1 - 0.16 / fig_h)
    fig.text(0.5, 1 - 0.60 / fig_h,
             f"perfect ring: ring=1.00, "
             f"$\\lambda_2/\\lambda_1$={IDEAL_RING['lam2_over_lam1']:.2f}, "
             f"top2={IDEAL_RING['top2_var_frac']:.2f}   |   "
             f"null: ring~0.38, $\\lambda_2/\\lambda_1$~0.64, top2~0.67",
             ha='center', va='top', color='#aaaaaa', fontsize=11)
    os.makedirs(figdir, exist_ok=True)
    tag = '' if bin_name == 'all' else f'_{bin_name}'
    fp = os.path.join(figdir, f'visual_geometry_epochs_mds_{cond}_{roi}{tag}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')
    return fp


def figure_rdm(results, bands, cond, roi, voxRes, figdir, bin_name='all', clim=0.3):
    bands = [b for b in bands if results.get((b, cond, roi, bin_name), {}).get('n', 0) > 0]
    if not bands:
        return None
    n_r, n_c = len(bands), len(EPOCH_ORDER)
    fig_h = 2.5 * n_r + 2.1
    fig = plt.figure(figsize=(2.5 * n_c + 2.0, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_r, n_c, figure=fig, hspace=0.30, wspace=0.22,
                            left=0.085, right=0.86,
                            top=1 - 1.5 / fig_h, bottom=0.40 / fig_h)
    im = None
    for r, band in enumerate(bands):
        e = results[(band, cond, roi, bin_name)]
        for c, ep in enumerate(EPOCH_ORDER):
            ax = fig.add_subplot(gs[r, c])
            grp, m = e['cells'][c]
            if grp is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color=_FG, fontsize=10)
                _style_ax(ax); continue
            # FIXED colour limits (-0.3 to 0.3 by default) rather than
            # per-panel data-driven ones, so every band/epoch/condition panel
            # is directly comparable by eye -- matches plot_circular_tgm.py's
            # fixed-clim convention for the same reason.
            im = ax.imshow(grp, cmap='RdBu_r', vmin=-clim, vmax=clim,
                           interpolation='nearest')
            ticks = range(len(LOC_BY_ANGLE))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            lbl = [f'{int(a)}' for a in ANGLES_DEG]
            ax.set_xticklabels(lbl, fontsize=FS_TICK, rotation=90)
            ax.set_yticklabels(lbl, fontsize=FS_TICK)
            ax.set_title(f"r={m.get('intersubj_r', np.nan):.2f}", fontsize=FS_PANEL_TTL,
                         color=_FG, fontweight='bold', pad=3)
            if r == 0:
                ax.text(0.5, 1.16, EPOCH_LABELS[ep], transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=FS_PANEL_TTL, color=_FG,
                        fontweight='bold')
            if c == 0:
                ax.annotate(f"{BAND_LABELS.get(band, band)}\n(n={e['n']})",
                            xy=(-0.32, 0.5), xycoords='axes fraction',
                            fontsize=FS_ROW_LABEL, color=_FG, ha='right', va='center',
                            rotation=90, fontweight='bold')
            _style_ax(ax)
    if im is not None:
        cax = fig.add_axes([0.875, 0.25, 0.014, 0.45])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('z-scored dissimilarity', color=_FG, fontsize=FS_CBAR_LABEL,
                     fontweight='bold')
        cb.ax.tick_params(colors=_FG, labelsize=FS_CBAR_TICK)
        cb.outline.set_edgecolor('#333333')
    fig.suptitle(f'Group RDM by epoch  |  {COND_LABELS.get(cond, cond)}  |  '
                 f'{"" if bin_name=="all" else "bin=" + bin_name + "  |  "}'
                 f'{roi.capitalize()}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1 - 0.28 / fig_h)
    os.makedirs(figdir, exist_ok=True)
    tag = '' if bin_name == 'all' else f'_{bin_name}'
    fp = os.path.join(figdir, f'visual_geometry_epochs_rdm_{cond}_{roi}{tag}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')
    return fp


def write_csv(results, csvdir, rois, voxRes, bins=('all',)):
    """One tidy table covering EVERY roi in `results`. The filename is built
    from the full roi list, not rois[0] -- naming a three-ROI table after just
    the first one invites it being read as visual-only later."""
    os.makedirs(csvdir, exist_ok=True)
    tag = rois[0] if len(rois) == 1 else '-'.join(rois)
    fp = os.path.join(csvdir, f'visual_geometry_epochs_{tag}_{voxRes}.csv')
    keys = ['ring_r', 'p_ring', 'lam2_over_lam1', 'top2_var_frac', 'pr_mds',
            'radial_cv', 'intersubj_r', 'Rgroup', 'neg_eig_frac']
    lines = ['band,condition,roi,bin,epoch,t_start,t_stop,n_subj,' + ','.join(keys)]
    for key_tuple, e in sorted(results.items(), key=lambda kv: str(kv[0])):
        band, cond, r_, bn = key_tuple
        if e['n'] == 0:
            continue
        for c, ep in enumerate(EPOCH_ORDER):
            _, m = e['cells'][c]
            lo, hi = EPOCHS[ep]
            vals = ','.join(f"{m.get(k, np.nan):.6g}" for k in keys)
            lines.append(f'{band},{cond},{r_},{bn},{ep},{lo},{hi},{e["n"]},{vals}')
    with open(fp, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Saved: {fp}')
    return fp


def print_summary(results, bands, conditions, rois, bins=('all',)):
    hdr = (f"{'band':10s} {'cond':9s} {'bin':6s} {'epoch':12s} {'n':>3s} {'ring':>6s} {'p':>6s} "
           f"{'l2/l1':>6s} {'top2':>5s} {'PR':>5s} {'radCV':>6s} {'interR':>7s} {'Rgrp':>6s}")
    print('\n' + '=' * len(hdr))
    print('SUMMARY by epoch')
    print(f"  perfect ring at THESE angles: l2/l1={IDEAL_RING['lam2_over_lam1']:.2f} "
          f"top2={IDEAL_RING['top2_var_frac']:.2f} PR={IDEAL_RING['pr_mds']:.2f} "
          f"radCV={IDEAL_RING['radial_cv']:.2f}  (NOT 1/1/2/0 -- angles are non-uniform)")
    print('  null: ring~0.38 (p95 0.61) | l2/l1~0.64 | top2~0.67 | PR~3.56 | radCV~0.46')
    print('  p    = label-shuffle permutation p for ring-ness (floor 1/(n_null+1))')
    print('  Rgrp = Spearman-Brown reliability of the GROUP mean given interR and n')
    print('=' * len(hdr))
    print(hdr); print('-' * len(hdr))
    for cond in conditions:
        for roi in rois:
            for band in bands:
              for bn in bins:
                e = results.get((band, cond, roi, bn))
                if e is None or e['n'] == 0:
                    continue
                for c, ep in enumerate(EPOCH_ORDER):
                    _, m = e['cells'][c]
                    f = lambda k, w, p=2: (f"{m[k]:{w}.{p}f}"
                                            if np.isfinite(m.get(k, np.nan)) else f'{"-":>{w}}')
                    print(f"{band:10s} {cond:9s} {bn:6s} {ep:12s} {e['n']:3d} {f('ring_r',6)} "
                          f"{f('p_ring',6,3)} {f('lam2_over_lam1',6)} {f('top2_var_frac',5)} "
                          f"{f('pr_mds',5)} {f('radial_cv',6)} {f('intersubj_r',7)} "
                          f"{f('Rgroup',6)}")


def discover_bins(subjects, bids_root, voxRes, bands, conds, rois, outdir):
    """Bin names present in the saved files (falls back to ('all',))."""
    for band in bands:
        for cond in conds:
            for roi in rois:
                for s in subjects:
                    fp = output_path(bids_root, s, band, cond, roi, voxRes, outdir)
                    if os.path.exists(fp):
                        with np.load(fp, allow_pickle=True) as npz:
                            if 'bins' in npz.files:
                                return tuple(str(b) for b in npz['bins'])
                        return ('all',)
    return ('all',)


def figure_bin_comparison(results, bands, bins, cond, roi, voxRes, figdir):
    """
    The headline binning figure: each metric against performance bin, one line
    per band, one column per epoch. This is what answers "does the geometry
    depend on how well the trial was remembered" -- the per-bin MDS grids show
    what the geometry looks like, this shows whether it moves.
    """
    plot_bins = [b for b in bins if b != 'all']
    if len(plot_bins) < 2:
        return None
    METS = [('ring_r', 'Ring-ness'), ('lam2_over_lam1', r'$\lambda_2/\lambda_1$'),
            ('top2_var_frac', 'Top-2 var. frac.'), ('radial_cv', 'Radial CV'),
            ('intersubj_r', 'Inter-subject r')]
    n_r, n_c = len(METS), len(EPOCH_ORDER)
    # Taller per row and more header room than the original 1.9/1.15, to fit the
    # larger fonts below without the suptitle colliding with the column titles.
    fig_h = 2.3 * n_r + 2.0
    fig = plt.figure(figsize=(3.6 * n_c + 2.0, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_r, n_c, figure=fig, hspace=0.42, wspace=0.32,
                            left=0.10, right=0.855,
                            top=1 - 1.5 / fig_h, bottom=0.45 / fig_h)
    cmap = plt.cm.viridis(np.linspace(0.15, 0.9, max(1, len(bands))))
    x = np.arange(len(plot_bins))
    for r, (key, lab) in enumerate(METS):
        for c, ep in enumerate(EPOCH_ORDER):
            ax = fig.add_subplot(gs[r, c])
            for bi, band in enumerate(bands):
                ys = []
                for bn in plot_bins:
                    e = results.get((band, cond, roi, bn))
                    ys.append(e['cells'][c][1].get(key, np.nan)
                              if e and e['n'] else np.nan)
                if not np.isfinite(ys).any():
                    continue
                ax.plot(x, ys, 'o-', color=cmap[bi], lw=1.5, ms=4,
                        label=BAND_LABELS.get(band, band) if (r == 0 and c == 0) else None)
            ideal = IDEAL_RING.get(key)
            if ideal is not None:
                ax.axhline(ideal, color='#4EA1F3', lw=0.9, ls='--', zorder=0)
            if key == 'intersubj_r':
                ax.axhline(0.0, color='#555555', lw=0.8, ls=':', zorder=0)
            ax.set_xticks(x); ax.set_xticklabels(plot_bins, fontsize=FS_TICK)
            ax.grid(True, color=_GRID, lw=0.4)
            if r == 0:
                ax.set_title(EPOCH_LABELS[ep], fontsize=FS_PANEL_TTL, color=_FG,
                             fontweight='bold')
            if c == 0:
                ax.set_ylabel(lab, fontsize=FS_AXIS_LABEL, color=_FG, fontweight='bold')
            if r == n_r - 1:
                ax.set_xlabel('Performance bin', fontsize=FS_AXIS_LABEL, fontweight='bold')
            _style_ax(ax)
    h, l = fig.axes[0].get_legend_handles_labels()
    if h:
        leg = fig.legend(h, l, loc='center left', bbox_to_anchor=(0.865, 0.5),
                         fontsize=11, framealpha=0.25, edgecolor='#444444',
                         labelcolor=_FG)
        leg.get_frame().set_facecolor('#1a1a1a')
    fig.suptitle(f'Geometry vs memory performance  |  {COND_LABELS.get(cond, cond)}  |  '
                 f'{roi.capitalize()}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1 - 0.28 / fig_h)
    # Kept as a separate smaller line rather than folded into the bold title:
    # "within each location" is the thing that makes these bins interpretable
    # (they are not confounded by location difficulty), so it should not be
    # dropped just to shorten the headline.
    bfrac = next((results[k].get('bin_frac') for k in results
                  if k[1] == cond and k[2] == roi
                  and np.isfinite(results[k].get('bin_frac', np.nan))), np.nan)
    if np.isfinite(bfrac):
        # OVERLAPPING windows: say so on the figure. Adjacent bins share trials,
        # so the bin-to-bin contrast here is diluted and the bins are not
        # independent -- reading this plot without that is the main way to
        # over-interpret it.
        bin_txt = (f'bins = overlapping {bfrac:.0%}-wide windows of initial-saccade '
                   f'error within each target location (adjacent bins share trials)')
    else:
        bin_txt = ('bins = disjoint quantiles of initial-saccade error within each '
                   'target location')
    fig.text(0.5, 1 - 0.85 / fig_h, bin_txt + '  |  dashed = perfect-ring reference',
             ha='center', va='top', color='#aaaaaa', fontsize=11)
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'visual_geometry_epochs_bincompare_{cond}_{roi}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')
    return fp


def main():
    ap = argparse.ArgumentParser(description='Epoch-based group RDM + MDS geometry figures.')
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'],
                     help='Default theta/alpha/beta for both ampOnly and ampPhase; '
                          'pass lowgamma/highgamma explicitly for an ampOnly-only run.')
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--bins', nargs='+', default=['all'],
                     help="Performance bins to plot. Default ['all'] only -- trials are "
                          "NOT split by performance bin (every cache file also carries "
                          "an unsplit 'all' bin alongside its tertiles). Pass "
                          "--bins auto to discover and plot every bin saved in the "
                          "files instead (e.g. the performance tertiles), which also "
                          "turns on figure_bin_comparison.")
    ap.add_argument('--rdm_clim', type=float, default=0.3,
                     help='Fixed RDM colour limit in z-scored dissimilarity units '
                          '(default 0.3, i.e. scale runs -0.3 to +0.3) so every band/ '
                          'epoch/condition panel is comparable by eye.')
    ap.add_argument('--outdir', default=None,
                     help='Where the per-subject visual_geometry_epochs_cell.py .npz '
                          'caches live. Default: derivatives/glueDecoding/'
                          'visualGeometryEpochs/data under bids_root -- the shared '
                          'group-level location actually used in production (NOT '
                          'visual_geometry_epochs_cell.py\'s own per-subject default '
                          'of derivatives/sub-XX/sourceRecon/visualGeometryEpochs, '
                          'which is only what you get running it without --outdir).')
    ap.add_argument('--figdir', default=None,
                     help='Default: derivatives/glueDecoding/visualGeometryEpochs/'
                          'figures under bids_root, sibling to --outdir\'s default.')
    ap.add_argument('--csvdir', default=None,
                     help='Default: derivatives/glueDecoding/visualGeometryEpochs/'
                          'csv under bids_root, sibling to --outdir\'s default.')
    args = ap.parse_args()

    bids_root = get_bids_root()
    _default_base = os.path.join(bids_root, 'derivatives', 'glueDecoding',
                                  'visualGeometryEpochs')
    if args.outdir is None:
        args.outdir = os.path.join(_default_base, 'data')
    if args.figdir is None:
        args.figdir = os.path.join(_default_base, 'figures')
    if args.csvdir is None:
        args.csvdir = os.path.join(_default_base, 'csv')
    print(f'  outdir (reading caches from) = {args.outdir}\n'
          f'  figdir (writing figures to)  = {args.figdir}\n'
          f'  csvdir (writing csv to)      = {args.csvdir}')

    if args.bins == ['auto']:
        bins = discover_bins(args.subjects, bids_root, args.voxRes,
                             args.bands, args.conditions, args.rois, args.outdir)
        print(f'performance bins found (auto): {list(bins)}')
    else:
        bins = args.bins
        print(f'bins: {list(bins)} (pass --bins auto for the performance tertiles too)')
    results = {}
    for band in args.bands:
        for cond in args.conditions:
            for roi in args.rois:
                for bn in bins:
                    stack, nulls, n, bfrac = load_group(
                        args.subjects, bids_root, args.voxRes, band, cond, roi,
                        args.outdir, bin_name=bn)
                    if stack is None:
                        results[(band, cond, roi, bn)] = dict(cells=None, n=0,
                                                              bin_frac=bfrac)
                        continue
                    results[(band, cond, roi, bn)] = dict(
                        cells=cell_metrics(stack, nulls), n=n, bin_frac=bfrac)
                    print(f'  aggregated {band}/{cond}/{roi}/{bn}: n={n}', flush=True)

    if all(v['n'] == 0 for v in results.values()):
        print('Nothing to plot -- run visual_geometry_epochs_cell.py first.')
        return

    for cond in args.conditions:
        for roi in args.rois:
            for bn in bins:
                figure_rdm(results, args.bands, cond, roi, args.voxRes, args.figdir,
                           bin_name=bn, clim=args.rdm_clim)
                figure_mds(results, args.bands, cond, roi, args.voxRes, args.figdir, bin_name=bn)
            figure_bin_comparison(results, args.bands, bins, cond, roi,
                                   args.voxRes, args.figdir)
    write_csv(results, args.csvdir, args.rois, args.voxRes, bins)
    print_summary(results, args.bands, args.conditions, args.rois, bins)
    print('\nDone.')


if __name__ == '__main__':
    main()
