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
        [--conditions ampOnly ampPhase] [--rois visual]
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
)

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


def load_group(subjects, bids_root, voxRes, band, cond, roi, outdir):
    """Returns (rdm_stack (n_subj,n_ep,10,10), null_stack or None, n_subj)."""
    rdms, nulls = [], []
    for s in subjects:
        fp = output_path(bids_root, s, band, cond, roi, voxRes, outdir)
        if not os.path.exists(fp):
            continue
        with np.load(fp, allow_pickle=True) as npz:
            rdms.append(np.asarray(npz['rdm'], float)[:, _ORDER][:, :, _ORDER])
            if 'rdm_null' in npz.files:
                nulls.append(np.asarray(npz['rdm_null'], float)[:, :, _ORDER][:, :, :, _ORDER])
    if not rdms:
        return None, None, 0
    return np.stack(rdms), (np.stack(nulls) if nulls else None), len(rdms)


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
                transform=ax.transAxes, color=_FG, fontsize=8)
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
    ax.set_title(title, fontsize=7.2, color=_FG, pad=3)
    ax.margins(0.16)
    _style_ax(ax)


def figure_mds(results, bands, cond, roi, voxRes, figdir):
    bands = [b for b in bands if results.get((b, cond, roi), {}).get('n', 0) > 0]
    if not bands:
        return None
    n_r, n_c = len(bands), len(EPOCH_ORDER)
    fig_h = 2.55 * n_r + 1.7
    fig = plt.figure(figsize=(2.55 * n_c + 2.4, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_r, n_c, figure=fig, hspace=0.34, wspace=0.16,
                            left=0.075, right=0.845,
                            top=1 - 1.25 / fig_h, bottom=0.35 / fig_h)
    for r, band in enumerate(bands):
        e = results[(band, cond, roi)]
        for c, ep in enumerate(EPOCH_ORDER):
            ax = fig.add_subplot(gs[r, c])
            grp, m = e['cells'][c]
            _, coords = mds_spectrum(grp) if grp is not None else (None, None)
            ttl = (f"ring={m.get('ring_r', np.nan):.2f}  "
                   f"$\\lambda_2/\\lambda_1$={m.get('lam2_over_lam1', np.nan):.2f}\n"
                   f"top2={m.get('top2_var_frac', np.nan):.2f}")
            _draw_ring(ax, coords, ttl)
            if r == 0:
                ax.text(0.5, 1.30, EPOCH_LABELS[ep], transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=10, color=_FG,
                        fontweight='bold')
                ax.text(0.5, 1.21, f'{EPOCHS[ep][0]:+.1f} to {EPOCHS[ep][1]:+.1f}s',
                        transform=ax.transAxes, ha='center', va='bottom',
                        fontsize=6.5, color='#999999')
            if c == 0:
                ax.annotate(f"{BAND_LABELS.get(band, band)}\n(n={e['n']})",
                            xy=(-0.14, 0.5), xycoords='axes fraction',
                            fontsize=8.5, color=_FG, ha='right', va='center',
                            rotation=90, fontweight='bold')

    leg = fig.legend(handles=_legend_handles(), loc='center left',
                     bbox_to_anchor=(0.855, 0.5), fontsize=8, framealpha=0.25,
                     edgecolor='#444444', labelcolor=_FG, title='Target angle')
    leg.get_frame().set_facecolor('#1a1a1a')
    leg.get_title().set_color(_FG)
    leg.get_title().set_fontsize(8.5)

    fig.suptitle(
        f'MDS geometry by epoch  |  {COND_LABELS.get(cond, cond)}  |  '
        f'{roi.capitalize()}  |  {voxRes}\n'
        f"perfect ring at THESE 10 non-uniform angles: ring=1.00, "
        f"$\\lambda_2/\\lambda_1$={IDEAL_RING['lam2_over_lam1']:.2f} (NOT 1.0), "
        f"top2={IDEAL_RING['top2_var_frac']:.2f}   |   "
        f"null: ring~0.38 (p95 0.61), $\\lambda_2/\\lambda_1$~0.64, top2~0.67",
        color=_FG, fontsize=10, fontweight='bold', y=1 - 0.14 / fig_h)
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'visual_geometry_epochs_mds_{cond}_{roi}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')
    return fp


def figure_rdm(results, bands, cond, roi, voxRes, figdir):
    bands = [b for b in bands if results.get((b, cond, roi), {}).get('n', 0) > 0]
    if not bands:
        return None
    n_r, n_c = len(bands), len(EPOCH_ORDER)
    fig_h = 2.5 * n_r + 1.5
    fig = plt.figure(figsize=(2.5 * n_c + 2.0, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_r, n_c, figure=fig, hspace=0.30, wspace=0.22,
                            left=0.085, right=0.86,
                            top=1 - 1.05 / fig_h, bottom=0.40 / fig_h)
    im = None
    for r, band in enumerate(bands):
        e = results[(band, cond, roi)]
        for c, ep in enumerate(EPOCH_ORDER):
            ax = fig.add_subplot(gs[r, c])
            grp, m = e['cells'][c]
            if grp is None:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color=_FG, fontsize=8)
                _style_ax(ax); continue
            off = grp[~np.eye(grp.shape[0], dtype=bool)]
            v = np.nanmax(np.abs(off)) if np.isfinite(off).any() else 1.0
            im = ax.imshow(grp, cmap='RdBu_r', vmin=-v, vmax=v, interpolation='nearest')
            ticks = range(len(LOC_BY_ANGLE))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            lbl = [f'{int(a)}' for a in ANGLES_DEG]
            ax.set_xticklabels(lbl, fontsize=5, rotation=90)
            ax.set_yticklabels(lbl, fontsize=5)
            ax.set_title(f"r={m.get('intersubj_r', np.nan):.2f}", fontsize=7,
                         color=_FG, pad=2)
            if r == 0:
                ax.text(0.5, 1.30, EPOCH_LABELS[ep], transform=ax.transAxes,
                        ha='center', va='bottom', fontsize=10, color=_FG,
                        fontweight='bold')
            if c == 0:
                ax.annotate(f"{BAND_LABELS.get(band, band)}\n(n={e['n']})",
                            xy=(-0.30, 0.5), xycoords='axes fraction',
                            fontsize=8.5, color=_FG, ha='right', va='center',
                            rotation=90, fontweight='bold')
            _style_ax(ax)
    if im is not None:
        cax = fig.add_axes([0.875, 0.25, 0.014, 0.45])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label('z-scored dissimilarity', color=_FG, fontsize=8)
        cb.ax.tick_params(colors=_FG, labelsize=7)
        cb.outline.set_edgecolor('#333333')
    fig.suptitle(f'Group RDM by epoch  |  {COND_LABELS.get(cond, cond)}  |  '
                 f'{roi.capitalize()}  |  {voxRes}\n'
                 f'axes = true polar angle (deg); panel r = mean inter-subject '
                 f'RDM correlation (the gate)',
                 color=_FG, fontsize=10, fontweight='bold', y=1 - 0.12 / fig_h)
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'visual_geometry_epochs_rdm_{cond}_{roi}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fp}')
    return fp


def write_csv(results, csvdir, roi, voxRes):
    os.makedirs(csvdir, exist_ok=True)
    fp = os.path.join(csvdir, f'visual_geometry_epochs_{roi}_{voxRes}.csv')
    keys = ['ring_r', 'p_ring', 'lam2_over_lam1', 'top2_var_frac', 'pr_mds',
            'radial_cv', 'intersubj_r', 'Rgroup', 'neg_eig_frac']
    lines = ['band,condition,roi,epoch,t_start,t_stop,n_subj,' + ','.join(keys)]
    for (band, cond, r_), e in sorted(results.items()):
        if e['n'] == 0:
            continue
        for c, ep in enumerate(EPOCH_ORDER):
            _, m = e['cells'][c]
            lo, hi = EPOCHS[ep]
            vals = ','.join(f"{m.get(k, np.nan):.6g}" for k in keys)
            lines.append(f'{band},{cond},{r_},{ep},{lo},{hi},{e["n"]},{vals}')
    with open(fp, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Saved: {fp}')
    return fp


def print_summary(results, bands, conditions, rois):
    hdr = (f"{'band':10s} {'cond':9s} {'epoch':12s} {'n':>3s} {'ring':>6s} {'p':>6s} "
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
                e = results.get((band, cond, roi))
                if e is None or e['n'] == 0:
                    continue
                for c, ep in enumerate(EPOCH_ORDER):
                    _, m = e['cells'][c]
                    f = lambda k, w, p=2: (f"{m[k]:{w}.{p}f}"
                                            if np.isfinite(m.get(k, np.nan)) else f'{"-":>{w}}')
                    print(f"{band:10s} {cond:9s} {ep:12s} {e['n']:3d} {f('ring_r',6)} "
                          f"{f('p_ring',6,3)} {f('lam2_over_lam1',6)} {f('top2_var_frac',5)} "
                          f"{f('pr_mds',5)} {f('radial_cv',6)} {f('intersubj_r',7)} "
                          f"{f('Rgroup',6)}")


def main():
    ap = argparse.ArgumentParser(description='Epoch-based group RDM + MDS geometry figures.')
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    ap.add_argument('--bands', nargs='+',
                     default=['theta', 'alpha', 'beta', 'lowgamma', 'highgamma'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual'])
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--figdir', required=True)
    ap.add_argument('--csvdir', required=True)
    args = ap.parse_args()

    bids_root = get_bids_root()
    results = {}
    for band in args.bands:
        for cond in args.conditions:
            for roi in args.rois:
                stack, nulls, n = load_group(args.subjects, bids_root, args.voxRes,
                                              band, cond, roi, args.outdir)
                if stack is None:
                    results[(band, cond, roi)] = dict(cells=None, n=0)
                    continue
                results[(band, cond, roi)] = dict(cells=cell_metrics(stack, nulls), n=n)
                print(f'  aggregated {band}/{cond}/{roi}: n={n}', flush=True)

    if all(v['n'] == 0 for v in results.values()):
        print('Nothing to plot -- run visual_geometry_epochs_cell.py first.')
        return

    for cond in args.conditions:
        for roi in args.rois:
            figure_rdm(results, args.bands, cond, roi, args.voxRes, args.figdir)
            figure_mds(results, args.bands, cond, roi, args.voxRes, args.figdir)
    write_csv(results, args.csvdir, args.rois[0], args.voxRes)
    print_summary(results, args.bands, args.conditions, args.rois)
    print('\nDone.')


if __name__ == '__main__':
    main()
