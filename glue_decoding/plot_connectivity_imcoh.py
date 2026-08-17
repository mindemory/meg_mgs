#!/usr/bin/env python3
"""
plot_connectivity_imcoh.py

Group figures for connectivity_imcoh_epochs.py: rows = bands, columns = ROI
pairs, x-axis = epoch.

WHAT IS PLOTTED. The solid line is mean|ImCoh| and the dashed line is the
trial-shuffle floor for the same cell. Read the GAP between them, not the
absolute height: mean|ImCoh| is noise-biased and never reaches 0 even with no
coupling at all, so the floor is the actual reference. A second row of panels
plots that gap (excess over null) directly, which is the quantity to compare
across epochs and bands.

Sample counts are equalised across epochs upstream (see
connectivity_imcoh_epochs.py) -- without that the 5x shorter stimulus epoch
would sit ~2.2x higher purely from having fewer samples. The equalisation flag
is checked here and a warning is printed if any loaded cell was computed
without it.

Usage:
    python plot_connectivity_imcoh.py [--voxRes 8mm] [--bands theta alpha beta]
        [--pairs visual-parietal visual-frontal parietal-frontal]
        [--subjects 1 2 ...] [--outdir <path>] [--figdir <path>] [--csvdir <path>]
"""

import os, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from constants import SUBJECT_LIST, get_bids_root
from connectivity_imcoh_epochs import output_path, DEFAULT_PAIRS
from visual_geometry_epochs_cell import EPOCH_ORDER

_BG, _FG, _GRID = '#000000', '#e0e0e0', '#1c1c1c'
BAND_LABELS = {'theta': 'Theta (4-8 Hz)', 'alpha': 'Alpha (8-12 Hz)', 'beta': 'Beta (13-30 Hz)'}
EPOCH_LABELS = {'fixation': 'Fix', 'stimulus': 'Stim', 'early_delay': 'Early\ndelay',
                'late_delay': 'Late\ndelay'}
PAIR_COLOURS = {'visual-parietal': '#FFC629', 'visual-frontal': '#34D399',
                'parietal-frontal': '#A78BFA'}


def _style(ax):
    ax.set_facecolor(_BG)
    for sp in ax.spines.values(): sp.set_color('#333333')
    ax.tick_params(colors=_FG, labelsize=7.5)
    ax.xaxis.label.set_color(_FG); ax.yaxis.label.set_color(_FG); ax.title.set_color(_FG)


def load_group(subjects, bids_root, voxRes, band, pair, outdir):
    """(real (n_subj,n_ep), null (n_subj,n_ep), n_subj, all_equalised)."""
    R, N, eq = [], [], True
    for s in subjects:
        fp = output_path(bids_root, s, band, pair, voxRes, outdir)
        if not os.path.exists(fp):
            continue
        with np.load(fp, allow_pickle=True) as z:
            R.append(np.asarray(z['mean_abs_imcoh'], float))
            N.append(np.asarray(z['null_mean'], float))
            if 'equalised' in z.files and not bool(z['equalised'][0]):
                eq = False
    if not R:
        return None, None, 0, eq
    return np.stack(R), np.stack(N), len(R), eq


def figure(results, bands, pairs, voxRes, figdir):
    n_r, n_c = 2 * len(bands), len(pairs)
    fig_h = 2.0 * n_r + 1.4
    fig = plt.figure(figsize=(3.2 * n_c + 1.0, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(n_r, n_c, figure=fig, hspace=0.45, wspace=0.30,
                            left=0.10, right=0.98, top=1 - 1.05 / fig_h, bottom=0.40 / fig_h)
    x = np.arange(len(EPOCH_ORDER))
    for bi, band in enumerate(bands):
        for pi, pair in enumerate(pairs):
            R, N, n, _ = results.get((band, pair), (None, None, 0, True))
            col = PAIR_COLOURS.get(pair, '#ffffff')
            # row 1: raw + null
            ax = fig.add_subplot(gs[2 * bi, pi])
            if R is not None:
                m, sd = R.mean(0), R.std(0, ddof=1) / np.sqrt(n)
                nm = N.mean(0)
                ax.fill_between(x, m - sd, m + sd, color=col, alpha=0.28)
                ax.plot(x, m, 'o-', color=col, lw=1.6, ms=4, label='observed')
                ax.plot(x, nm, 's--', color='#888888', lw=1.2, ms=3, label='shuffle floor')
                if bi == 0 and pi == n_c - 1:
                    leg = ax.legend(fontsize=6.5, framealpha=.25, edgecolor='#444', labelcolor=_FG)
                    leg.get_frame().set_facecolor('#1a1a1a')
            else:
                ax.text(.5, .5, 'No data', ha='center', va='center', transform=ax.transAxes,
                        color=_FG, fontsize=8)
            ax.set_xticks(x); ax.set_xticklabels([EPOCH_LABELS[e] for e in EPOCH_ORDER], fontsize=6.5)
            ax.grid(True, color=_GRID, lw=.4)
            if bi == 0: ax.set_title(f'{pair}  (n={n})', fontsize=9, color=_FG, fontweight='bold')
            if pi == 0: ax.set_ylabel(f'{BAND_LABELS.get(band,band)}\nmean|ImCoh|', fontsize=7.5)
            _style(ax)
            # row 2: excess over null (the comparable quantity)
            ax2 = fig.add_subplot(gs[2 * bi + 1, pi])
            if R is not None:
                E = R - N
                m, sd = E.mean(0), E.std(0, ddof=1) / np.sqrt(n)
                ax2.axhline(0, color='#555555', lw=.8, ls=':')
                ax2.fill_between(x, m - sd, m + sd, color=col, alpha=.28)
                ax2.plot(x, m, 'o-', color=col, lw=1.6, ms=4)
                for i in range(len(EPOCH_ORDER)):
                    t, p = stats.ttest_1samp(E[:, i], 0.0)
                    if np.isfinite(p) and p < 0.05:
                        ax2.plot(i, m[i] + sd[i] * 1.6, '*', color='#ffffff', ms=6)
            ax2.set_xticks(x); ax2.set_xticklabels([EPOCH_LABELS[e] for e in EPOCH_ORDER], fontsize=6.5)
            ax2.grid(True, color=_GRID, lw=.4)
            if pi == 0: ax2.set_ylabel('excess over\nshuffle', fontsize=7.5)
            _style(ax2)
    fig.suptitle(f'ROI-pair imaginary coherence by epoch  |  {voxRes}\n'
                 f'top rows: mean|ImCoh| with its trial-shuffle floor (mean|ImCoh| is '
                 f'noise-biased and never reaches 0)  |  bottom rows: the gap, which is '
                 f'the comparable quantity  |  * = p<0.05 vs 0, uncorrected',
                 color=_FG, fontsize=10, fontweight='bold', y=1 - 0.12 / fig_h)
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'connectivity_imcoh_epochs_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig); print(f'Saved: {fp}'); return fp


def main():
    ap = argparse.ArgumentParser(description='Group ImCoh figures by epoch.')
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--pairs', nargs='+', default=list(DEFAULT_PAIRS))
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--figdir', required=True)
    ap.add_argument('--csvdir', required=True)
    args = ap.parse_args()

    bids_root = get_bids_root()
    results, warn = {}, False
    for band in args.bands:
        for pair in args.pairs:
            R, N, n, eq = load_group(args.subjects, bids_root, args.voxRes, band, pair, args.outdir)
            results[(band, pair)] = (R, N, n, eq)
            if n: print(f'  {band}/{pair}: n={n}')
            if not eq: warn = True
    if warn:
        print('\nWARNING: some cells were computed WITHOUT sample equalisation. The '
              'epochs differ up to 5-fold in length, so their noise floors differ ~2.2x '
              'and the epoch comparison is not interpretable for those cells.')
    if all(v[2] == 0 for v in results.values()):
        print('Nothing to plot -- run connectivity_imcoh_epochs.py first.'); return

    figure(results, args.bands, args.pairs, args.voxRes, args.figdir)

    os.makedirs(args.csvdir, exist_ok=True)
    fp = os.path.join(args.csvdir, f'connectivity_imcoh_epochs_{args.voxRes}.csv')
    lines = ['band,pair,epoch,n_subj,imcoh_mean,imcoh_sem,null_mean,excess_mean,excess_sem,t,p']
    for (band, pair), (R, N, n, _) in sorted(results.items()):
        if not n: continue
        E = R - N
        for i, ep in enumerate(EPOCH_ORDER):
            t, p = stats.ttest_1samp(E[:, i], 0.0)
            lines.append(f'{band},{pair},{ep},{n},{R[:,i].mean():.6g},'
                         f'{R[:,i].std(ddof=1)/np.sqrt(n):.6g},{N[:,i].mean():.6g},'
                         f'{E[:,i].mean():.6g},{E[:,i].std(ddof=1)/np.sqrt(n):.6g},'
                         f'{t:.4g},{p:.4g}')
    open(fp, 'w').write('\n'.join(lines) + '\n'); print(f'Saved: {fp}')
    print('\nDone.')


if __name__ == '__main__':
    main()
