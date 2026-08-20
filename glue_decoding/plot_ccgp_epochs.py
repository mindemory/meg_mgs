#!/usr/bin/env python3
"""
plot_ccgp_epochs.py

Cross-subject aggregation + figures for ccgp_epochs_cell.py: mean +/- SEM
ACROSS SUBJECTS of CCGP and shattering dimensionality, per band x ROI x epoch.

The subject axis is the point of this analysis. Unlike the RDM/MDS geometry --
which only exists as a group average, with inter-subject RDM correlation
~0.02-0.07 -- CCGP and SD are computed independently per subject, so here the
error bar is a genuine across-subject SEM and the group claim rests on
consistency between subjects rather than on averaging their noise together.

TWO FIGURES per condition:
  ..._ccgp_...png  three lines per panel (horizontal / vertical / axis) over
      the four epochs, with each dichotomy's own shuffled null as a shaded
      band. CCGP chance is NOT 0.5 -- correlated conditions push it up -- so
      every value is read against its own null, never against 0.5.
  ..._sd_...png    shattering dimensionality over the epochs, against the
      noiseless ring reference (4/35 of balanced dichotomies separable for 8
      points in a plane).

READING THEM -- the ring makes a JOINT prediction, not three separate ones:
'axis' (near-horizontal vs near-vertical) is not a linear coordinate on the
ring plane, so a genuine planar code puts it AT CHANCE by construction. The
signature is horizontal and vertical above their nulls WHILE axis sits on its
null, together with low SD. High CCGP on all three would not be a ring, and
high SD would mean structure beyond the plane.

Usage:
    python plot_ccgp_epochs.py [--voxRes 8mm] [--bands theta alpha beta]
        [--conditions ampOnly ampPhase] [--rois visual parietal frontal]
        [--subjects 1 2 ...] [--outdir <path>] [--figdir <path>] [--csvdir <path>]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from constants import SUBJECT_LIST, get_bids_root
from visual_geometry_epochs_cell import EPOCH_ORDER
from ccgp_epochs_cell import output_path, DICHOTOMIES, GEOMETRIC_RING_SD

_BG, _FG, _GRID = '#000000', '#e0e0e0', '#1c1c1c'
FS_SUPTITLE, FS_PANEL_TTL, FS_AXIS_LABEL, FS_ROW_LABEL, FS_TICK = 22, 17, 16, 17, 13
FS_LEGEND = 13

ROI_COLOURS = {'visual': '#FFC629', 'parietal': '#A78BFA', 'frontal': '#34D399'}
DICH_COLOURS = {'horizontal': '#FFC629', 'vertical': '#4EA1F3', 'axis': '#FF6B6B'}
DICH_LABELS = {'horizontal': 'Left / right', 'vertical': 'Top / bottom',
                'axis': 'Near-H / near-V  (control)'}
BAND_LABELS = {'theta': 'Theta (4-8 Hz)', 'alpha': 'Alpha (8-12 Hz)',
               'beta': 'Beta (13-30 Hz)'}
COND_LABELS = {'ampOnly': 'Amplitude', 'ampPhase': 'Amplitude + Phase'}
EPOCH_LABELS = {'fixation': 'Fixation', 'stimulus': 'Stimulus',
                 'early_delay': 'Early delay', 'late_delay': 'Late delay'}

# CONVENTIONAL shattering dimensionality: mean decoding accuracy over all 35
# balanced dichotomies, on the familiar 0.5-1.0 scale. Reporting a
# fraction-of-dichotomies instead invites the reading "only two conditions are
# decodable", which is not what the term denotes.
SD_KEY = 'sd_mean_acc'


def load_all(subjects, bids_root, voxRes, bands, conditions, rois, outdir):
    """Long-format table, one row per (subject, band, condition, roi, epoch)."""
    rows = []
    for s in subjects:
        for band in bands:
            for cond in conditions:
                for roi in rois:
                    fp = output_path(bids_root, s, band, cond, roi, voxRes, outdir)
                    if not os.path.exists(fp):
                        continue
                    with np.load(fp, allow_pickle=True) as z:
                        for ep in EPOCH_ORDER:
                            if f'{ep}__pca_dim' not in z.files:
                                continue
                            r = dict(subjID=s, band=band, condition=cond, roi=roi,
                                     epoch=ep, pca_dim=int(z[f'{ep}__pca_dim']),
                                     n_trials_used=int(z[f'{ep}__n_trials_used']),
                                     sd=float(z[f'{ep}__{SD_KEY}']),
                                     sd_frac_sig=float(z[f'{ep}__sd_frac_significant']),
                                     sd_frac_above_07=float(z[f'{ep}__sd_frac_above_0.7']))
                            for d in DICHOTOMIES:
                                cc = float(z[f'{ep}__ccgp_{d}'])
                                nu = float(z[f'{ep}__ccgp_{d}_null_mean'])
                                r[f'ccgp_{d}'] = cc
                                r[f'null_{d}'] = nu
                                r[f'p_{d}'] = float(z[f'{ep}__ccgp_{d}_p'])
                                # CCGP MINUS THAT SUBJECT'S OWN NULL is the
                                # quantity to average. Chance is not 0.5 and
                                # varies with the condition correlations, which
                                # differ per subject/cell, so averaging raw CCGP
                                # mixes signal with per-subject chance level.
                                r[f'delta_{d}'] = cc - nu
                                # CCGP's ceiling: standard (non-cross-condition)
                                # decoding of the same dichotomy. CCGP cannot
                                # exceed it, so raw CCGP alone cannot separate
                                # "not abstract" from "not decodable".
                                if f'{ep}__decode_{d}' in z.files:
                                    dv = float(z[f'{ep}__decode_{d}'])
                                    dn = float(z[f'{ep}__decode_{d}_null_mean'])
                                    r[f'decode_{d}'] = dv
                                    r[f'decode_delta_{d}'] = dv - dn
                                    r[f'abstraction_{d}'] = float(
                                        z[f'{ep}__abstraction_{d}'])
                            rows.append(r)
    return pd.DataFrame(rows)


def mean_sem(df, col):
    """(mean, sem, n) across SUBJECTS -- the unit the error bar represents."""
    v = df[col].values
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan, np.nan, 0
    return (float(v.mean()),
            float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0,
            int(v.size))


def bh_fdr(pvals, q=0.05):
    """
    Benjamini-Hochberg. Returns a bool mask of which tests survive at FDR q.

    Correction is applied because these figures run one test per
    (band, roi, epoch, dichotomy) -- 108 for the CCGP figure -- and a reviewer
    has already flagged uncorrected multiple comparisons on this dataset's
    connectivity claim. FDR rather than Bonferroni: the tests are positively
    correlated (neighbouring epochs, overlapping ROIs) so Bonferroni would be
    needlessly conservative, and FDR is the standard choice for a family this
    size where some true effects are expected.
    """
    pv = np.asarray(pvals, float)
    out = np.zeros(pv.shape, bool)
    ok = np.isfinite(pv)
    v = pv[ok]
    if v.size == 0:
        return out
    order = np.argsort(v)
    ranked = v[order]
    passed = ranked <= q * (np.arange(1, v.size + 1) / v.size)
    if not passed.any():
        return out
    sig = np.zeros(v.size, bool)
    sig[order[:np.max(np.flatnonzero(passed)) + 1]] = True
    out[ok] = sig
    return out


SCOPE_LABEL = {'panel': 'FDR per panel', 'figure': 'FDR figure-wide',
               'none': 'uncorrected'}


def apply_correction(fam, scope, q):
    """
    fam: {panel_key: (test_keys, pvals)} -> {test_key: bool significant}.

    The FAMILY you correct over is a judgment call, not a fact, so it is a
    knob rather than a hardcoded choice:
      'panel'  (default) BH within each (band, roi) panel separately -- each
               panel treated as its own pre-specified analysis. 12 tests per
               family here, so markedly more lenient than figure-wide.
      'figure' BH over every test in the figure at once. Strictest, and the
               safest against the "uncorrected multiple comparisons" objection
               this dataset has already drawn once.
      'none'   raw p < q. Report the scope if you use this.
    Whichever is used is written into the figure legend, so a reader never has
    to guess how strict the marks are.
    """
    sig = {}
    if scope == 'figure':
        ks = [k for v in fam.values() for k in v[0]]
        ps = [pp for v in fam.values() for pp in v[1]]
        sig.update(zip(ks, bh_fdr(ps, q)))
        return sig
    for ks, ps in fam.values():
        arr = np.asarray(ps, float)
        if scope == 'none':
            m = np.isfinite(arr) & (arr < q)
        else:
            m = bh_fdr(arr, q)
        sig.update(zip(ks, m))
    return sig


def ttest_vs(df, col, mu=0.0):
    """One-sample t-test ACROSS SUBJECTS against mu. Returns (mean, p)."""
    v = df[col].values
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (float(v.mean()) if v.size else np.nan), np.nan
    return float(v.mean()), float(stats.ttest_1samp(v, mu).pvalue)


def _thin_yticks(ax, n=5):
    """Fewer y ticks: the default put 9 labels on a 0.08-wide axis."""
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=n, prune=None))


def style_ax(ax):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, which='both', labelsize=FS_TICK)
    ax.xaxis.label.set_color(_FG); ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, lw=0.5, ls='--', alpha=0.6)
    ax.set_axisbelow(True)


def figure_ccgp(df, condition, bands, rois, voxRes, figdir, sharey=False, q=0.05,
                scope='panel'):
    cdf = df[df.condition == condition]
    bands = [b for b in bands if b in set(cdf.band.unique())]
    rois = [r for r in rois if r in set(cdf.roi.unique())]
    if not bands or not rois:
        print(f'  no data for {condition}'); return

    # PASS 1 -- collect every test, grouped by panel so the correction family
    # can be chosen (see apply_correction).
    fam = {}
    for band in bands:
        for roi in rois:
            ks, ps = [], []
            for d in DICHOTOMIES:
                for ep in EPOCH_ORDER:
                    ss = cdf[(cdf.band == band) & (cdf.roi == roi) & (cdf.epoch == ep)]
                    _, pp = ttest_vs(ss, f'delta_{d}', 0.0)
                    ks.append((band, roi, d, ep)); ps.append(pp)
            fam[(band, roi)] = (ks, ps)
    sig = apply_correction(fam, scope, q)
    n_tested = sum(int(np.isfinite(np.asarray(v[1], float)).sum()) for v in fam.values())
    per_fam = len(next(iter(fam.values()))[1]) if fam else 0
    print(f'  CCGP: {sum(sig.values())}/{n_tested} significant | {SCOPE_LABEL[scope]} '
          f'q<{q} ({per_fam} tests per family)')

    n_r, n_c = len(bands), len(rois)
    fig, axes = plt.subplots(n_r, n_c, figsize=(5.0 * n_c + 2.4, 3.8 * n_r + 1.6),
                              facecolor=_BG, squeeze=False, sharey=sharey)
    x = np.arange(len(EPOCH_ORDER))
    for r, band in enumerate(bands):
        for c, roi in enumerate(rois):
            ax = axes[r][c]; style_ax(ax)
            sub = cdf[(cdf.band == band) & (cdf.roi == roi)]
            nn = []
            for d in DICHOTOMIES:
                m, e = [], []
                for ep in EPOCH_ORDER:
                    ss = sub[sub.epoch == ep]
                    mm, ee, n = mean_sem(ss, f'delta_{d}')
                    m.append(mm); e.append(ee); nn.append(n)
                m, e = np.array(m, float), np.array(e, float)
                col = DICH_COLOURS[d]
                # Hollow markers everywhere, FILLED where the test survives FDR:
                # the significance lives on the marker rather than on a separate
                # row of stars, which would collide with three overlapping lines.
                ax.errorbar(x, m, yerr=e, color=col, lw=2.4, marker='o', ms=7,
                            markerfacecolor=_BG, markeredgewidth=1.8, capsize=4,
                            zorder=4,
                            label=DICH_LABELS[d] if (r == 0 and c == 0) else None)
                msk = np.array([bool(sig.get((band, roi, d, ep), False))
                                for ep in EPOCH_ORDER])
                if msk.any():
                    ax.plot(x[msk], m[msk], 'o', color=col, ms=10, zorder=5,
                            markeredgecolor=col)
                # Dashed = the same dichotomy's standard decoding above chance,
                # i.e. CCGP's ceiling. The GAP between solid and dashed is the
                # part of the decodable signal that fails to generalize.
                if f'decode_delta_{d}' in sub.columns:
                    md = [mean_sem(sub[sub.epoch == ep], f'decode_delta_{d}')[0]
                          for ep in EPOCH_ORDER]
                    ax.plot(x, md, color=col, lw=1.5, ls='--', alpha=0.75, zorder=3,
                            label=('Decoding ceiling' if (r == 0 and c == 0
                                   and d == DICHOTOMIES[0]) else None))
            ax.axhline(0.0, color='#888888', lw=1.3, ls=':', zorder=2)
            ax.set_xticks(x)
            ax.set_xticklabels([EPOCH_LABELS[e] for e in EPOCH_ORDER],
                                rotation=30, ha='right', fontsize=FS_TICK)
            _thin_yticks(ax)
            if r == 0:
                ax.set_title(f'{roi.capitalize()} (n={max(nn) if nn else 0})',
                             fontsize=FS_PANEL_TTL,
                             color=ROI_COLOURS.get(roi, _FG), fontweight='bold', pad=10)
            if c == 0:
                ax.set_ylabel('CCGP $-$ chance', fontsize=FS_AXIS_LABEL,
                               fontweight='bold')
                ax.text(-0.32, 0.5, BAND_LABELS.get(band, band), transform=ax.transAxes,
                        fontsize=FS_ROW_LABEL, color=_FG, ha='right', va='center',
                        rotation=90, fontweight='bold')
    h, l = axes[0][0].get_legend_handles_labels()
    h += [plt.Line2D([0], [0], marker='o', ls='', color='#dddddd', ms=10),
          plt.Line2D([0], [0], marker='o', ls='', color='#dddddd', ms=7,
                     markerfacecolor=_BG, markeredgewidth=1.8)]
    l += [f'p < {q} ({SCOPE_LABEL[scope]})', 'n.s.']
    leg = fig.legend(h, l, loc='center left', bbox_to_anchor=(0.99, 0.5),
                     fontsize=FS_LEGEND, framealpha=0.25, edgecolor='#444444',
                     labelcolor=_FG)
    leg.get_frame().set_facecolor('#1a1a1a')
    fig.suptitle(f'CCGP  |  {COND_LABELS.get(condition, condition)}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1.005)
    fig.tight_layout(rect=[0.02, 0, 1, 0.98])
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'ccgp_epochs_ccgp_{condition}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG); plt.close(fig)
    print(f'Saved: {fp}')


def figure_sd(df, condition, bands, rois, voxRes, figdir, sharey=False, q=0.05,
              scope='panel'):
    cdf = df[df.condition == condition]
    bands = [b for b in bands if b in set(cdf.band.unique())]
    rois = [r for r in rois if r in set(cdf.roi.unique())]
    if not bands or not rois:
        return

    # SD is a mean decoding ACCURACY, so the null is 0.5, not 0. A panel here
    # is one BAND (its three ROI lines), matching the figure's layout.
    fam = {}
    for band in bands:
        ks, ps = [], []
        for roi in rois:
            for ep in EPOCH_ORDER:
                ss = cdf[(cdf.band == band) & (cdf.roi == roi) & (cdf.epoch == ep)]
                _, pp = ttest_vs(ss, 'sd', 0.5)
                ks.append((band, roi, ep)); ps.append(pp)
        fam[band] = (ks, ps)
    sig = apply_correction(fam, scope, q)
    n_tested = sum(int(np.isfinite(np.asarray(v[1], float)).sum()) for v in fam.values())
    per_fam = len(next(iter(fam.values()))[1]) if fam else 0
    print(f'  SD:   {sum(sig.values())}/{n_tested} significant | {SCOPE_LABEL[scope]} '
          f'q<{q} ({per_fam} tests per family)')

    fig, axes = plt.subplots(1, len(bands), figsize=(5.2 * len(bands) + 2.6, 5.2),
                              facecolor=_BG, squeeze=False, sharey=sharey)
    x = np.arange(len(EPOCH_ORDER))
    for c, band in enumerate(bands):
        ax = axes[0][c]; style_ax(ax)
        for roi in rois:
            sub = cdf[(cdf.band == band) & (cdf.roi == roi)]
            m, e = [], []
            for ep in EPOCH_ORDER:
                mm, ee, _ = mean_sem(sub[sub.epoch == ep], 'sd')
                m.append(mm); e.append(ee)
            m, e = np.array(m, float), np.array(e, float)
            col = ROI_COLOURS.get(roi, '#fff')
            ax.errorbar(x, m, yerr=e, color=col, lw=2.4, marker='o', ms=7,
                        markerfacecolor=_BG, markeredgewidth=1.8, capsize=4,
                        zorder=4, label=roi.capitalize() if c == 0 else None)
            msk = np.array([bool(sig.get((band, roi, ep), False)) for ep in EPOCH_ORDER])
            if msk.any():
                ax.plot(x[msk], m[msk], 'o', color=col, ms=10, zorder=5,
                        markeredgecolor=col)
        # 0.5 is chance for a mean-accuracy SD; the planar-ring bound (4/35) no
        # longer applies on this scale -- it was a fraction-of-dichotomies
        # quantity, not an accuracy.
        ax.axhline(0.5, color='#888888', lw=1.3, ls=':', zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([EPOCH_LABELS[e] for e in EPOCH_ORDER],
                            rotation=30, ha='right', fontsize=FS_TICK)
        _thin_yticks(ax)
        ax.set_title(BAND_LABELS.get(band, band), fontsize=FS_PANEL_TTL,
                     color=_FG, fontweight='bold', pad=10)
        if c == 0:
            ax.set_ylabel('Shattering dimensionality\n(mean decoding acc.)',
                          fontsize=FS_AXIS_LABEL, fontweight='bold')
    h, l = axes[0][0].get_legend_handles_labels()
    h += [plt.Line2D([0], [0], color='#888888', lw=1.3, ls=':'),
          plt.Line2D([0], [0], marker='o', ls='', color='#dddddd', ms=10),
          plt.Line2D([0], [0], marker='o', ls='', color='#dddddd', ms=7,
                     markerfacecolor=_BG, markeredgewidth=1.8)]
    l += ['Chance (0.5)', f'p < {q} ({SCOPE_LABEL[scope]})', 'n.s.']
    leg = fig.legend(h, l, loc='center left', bbox_to_anchor=(0.99, 0.5),
                     fontsize=FS_LEGEND, framealpha=0.25, edgecolor='#444444',
                     labelcolor=_FG)
    leg.get_frame().set_facecolor('#1a1a1a')
    fig.suptitle(f'Shattering dimensionality  |  {COND_LABELS.get(condition, condition)}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(figdir, exist_ok=True)
    fp = os.path.join(figdir, f'ccgp_epochs_sd_{condition}_{voxRes}.png')
    fig.savefig(fp, dpi=150, bbox_inches='tight', facecolor=_BG); plt.close(fig)
    print(f'Saved: {fp}')


def print_summary(df):
    if df.empty:
        print('No data.'); return
    print('\n' + '=' * 78)
    print('CCGP / SD -- mean +/- SEM ACROSS SUBJECTS')
    print('  values are CCGP MINUS that subject\'s own null (0 = chance), with a')
    print('  one-sample t-test across subjects; p is that t-test, not the shuffle p.')
    print('  the ring signature is L/R and T/B above null WITH the control at null.')
    print('=' * 78)
    hdr = (f"{'band':6s} {'cond':9s} {'roi':9s} {'epoch':12s} {'n':>3s} "
           + ' '.join(f'{d[:4]:>13s}' for d in DICHOTOMIES) + f" {'SD':>12s}")
    print(hdr); print('-' * len(hdr))
    for (band, cond, roi), g in df.groupby(['band', 'condition', 'roi']):
        for ep in EPOCH_ORDER:
            s = g[g.epoch == ep]
            if s.empty:
                continue
            cells = []
            for d in DICHOTOMIES:
                v = s[f'delta_{d}'].values
                v = v[np.isfinite(v)]
                m, e, n = mean_sem(s, f'delta_{d}')
                pv = stats.ttest_1samp(v, 0).pvalue if v.size > 1 else np.nan
                cells.append(f'{m:+.3f}p{pv:.3f}')
            sm, se, n = mean_sem(s, 'sd')
            print(f'{band:6s} {cond:9s} {roi:9s} {ep:12s} {n:3d} '
                  + ' '.join(f'{c:>13s}' for c in cells) + f' {sm:.2f}+/-{se:.2f}')


def main():
    ap = argparse.ArgumentParser(description='Aggregate + plot CCGP / SD across subjects.')
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--fdr_q', type=float, default=0.05,
                     help='FDR level for the across-subject t-tests marked on the '
                          'figures (default 0.05; raise to 0.1 for a more lenient '
                          'threshold).')
    ap.add_argument('--fdr_scope', default='panel',
                     choices=['panel', 'figure', 'none'],
                     help="Family the correction is applied over. 'panel' (default) "
                          "corrects within each subplot separately -- 12 tests per "
                          "family, the most lenient corrected option. 'figure' "
                          "corrects over every test in the figure (strictest). "
                          "'none' is uncorrected. Whichever is used is written into "
                          "the figure legend.")
    ap.add_argument('--sharey', action='store_true',
                     help='Force one y-scale across all panels (off by default, so '
                          'each panel autoscales to its own range).')
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--figdir', default=None)
    ap.add_argument('--csvdir', default=None)
    args = ap.parse_args()

    bids_root = get_bids_root()
    base = os.path.join(bids_root, 'derivatives', 'glueDecoding', 'ccgpEpochs')
    outdir = args.outdir or os.path.join(base, 'data')
    figdir = args.figdir or os.path.join(base, 'figures')
    csvdir = args.csvdir or os.path.join(base, 'tables')
    print(f'  outdir (reading) = {outdir}\n  figdir (writing) = {figdir}\n'
          f'  csvdir (writing) = {csvdir}')

    df = load_all(args.subjects, bids_root, args.voxRes, args.bands,
                  args.conditions, args.rois, outdir)
    if df.empty:
        print('Nothing loaded -- run ccgp_epochs_cell.py first.')
        sys.exit(1)
    print(f'Loaded {len(df)} rows from {df.subjID.nunique()} subjects.')

    os.makedirs(csvdir, exist_ok=True)
    fp = os.path.join(csvdir, f'ccgp_epochs_{args.voxRes}.csv')
    df.to_csv(fp, index=False)
    print(f'Saved: {fp}')

    for cond in args.conditions:
        if cond not in set(df.condition.unique()):
            continue
        figure_ccgp(df, cond, args.bands, args.rois, args.voxRes, figdir,
                    sharey=args.sharey, q=args.fdr_q, scope=args.fdr_scope)
        figure_sd(df, cond, args.bands, args.rois, args.voxRes, figdir,
                  sharey=args.sharey, q=args.fdr_q, scope=args.fdr_scope)
    print_summary(df)


if __name__ == '__main__':
    main()
