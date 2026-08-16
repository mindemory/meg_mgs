#!/usr/bin/env python3
"""
aggregate_glue_capacity.py

Cross-subject aggregation + plotting for manifold_capacity.py's per-subject
CSVs (derivatives/sub-XX/sourceRecon/glueFits/sub-XX_task-mgs_glueFits_
{lockType}_{voxRes}.csv -- see constants.glue_fits_csv_path -- produced by
run_glue_capacity.sh).

TIME COURSES, NOT BARS
manifold_capacity.py now computes each glue fit in a sliding time window
(default 100 ms, non-overlapping) instead of two fixed epochs, so the
`epoch` index column is gone and `t_center` takes its place. Each panel is
therefore a TIME COURSE (mean +/- SEM across subjects), Real as a solid line
and the shuffled-points null as a dashed line, rather than a Real-vs-Shuffle
bar pair. Figure layout per (metric, scheme):
    rows = bands, columns = (roi, condition) cells present in the data
(so e.g. visual appears twice, once for ampOnly and once for ampPhase, when
manifold_capacity.py was run with its defaults -- ampPhase for visual only).

Significance: a per-timepoint paired t-test (real vs shuffle, matched by
subjID -- each subject contributes one real and one shuffle value from the
same manifolds/seed, so paired, not independent-samples), marked as dots
along the bottom of each panel for p<0.05. UNCORRECTED across timepoints --
with ~22 windows per panel this is a descriptive marker of where the effect
lives, not a family-wise-controlled test; read it as such.

y-axis is per-row (band) and purely data-driven, NOT clamped to include 0,
since capacity-type metrics sit in a narrow band (e.g. ~0.2-0.3) and a
forced 0 floor made the Real-vs-Shuffle difference nearly invisible.

Metrics plotted (see glue's ManifoldAnalysisResults / glue_analysis.py):
    capacity, dimension, radius, utility, center_alignment, axis_alignment

schemes: 4=quadrants (the default -- excludes the two axis-aligned locations,
0 deg and 180 deg), 2=left/right hemifield, 6=quadrants+axis, 10=every raw
location -- see constants.CATEGORY_SCHEMES, manifold_capacity.py's docstring.

Usage:
    python aggregate_glue_capacity.py [--voxRes 8mm] [--lockType stim]
                                       [--subjects 1 2 ...]
                                       [--bands theta alpha beta]
                                       [--rois visual parietal frontal]
                                       [--conditions ampOnly ampPhase]
                                       [--schemes 4]
                                       [--metrics capacity dimension radius ...]
                                       [--outdir <bids_root>/derivatives/glueDecoding/glueFits]
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import stats

from constants import SUBJECT_LIST, AMP_PHASE_BANDS, ROI_NAMES, CATEGORY_SCHEMES, \
                       get_bids_root, glue_fits_csv_path

# NOTE: category count uses len(groups), NOT the scheme key `s` -- they
# diverge as of scheme 3 (top_bottom, 2 categories, not 3; see constants.py).
SCHEME_LABELS = {s: f"{CATEGORY_SCHEMES[s]['name']} ({len(CATEGORY_SCHEMES[s]['groups'])} categories)"
                  for s in CATEGORY_SCHEMES}

# -- Visual design (mirrors intrinsic_dim_epochs.py / plot_timeseries.py) -----

_BG        = '#000000'
_FG        = '#e0e0e0'
_GRID      = '#1c1c1c'

ROI_COLOURS = {
    'visual':   '#FFC629',
    'parietal': '#A78BFA',
    'frontal':  '#34D399',
    'whole':    '#E76F51',
}

BAND_LABELS = {
    'theta':     'Theta\n(4-8 Hz)',
    'alpha':     'Alpha\n(8-12 Hz)',
    'beta':      'Beta\n(13-30 Hz)',
    'lowgamma':  'Low gamma\n(30-80 Hz)',
    'highgamma': 'High gamma\n(80-150 Hz)',
}

COND_LABELS = {
    'ampOnly':    'amp',
    'ampPhase':   'amp+phase',
    'phaseOnly':  'phase',
    'unfiltered': 'unfiltered',
}

METRICS = ['capacity', 'dimension', 'radius', 'utility',
           'center_alignment', 'axis_alignment']
METRIC_LABELS = {
    'capacity':          'Capacity',
    'dimension':         'Dimension',
    'radius':            'Radius',
    'utility':           'Utility',
    'center_alignment':  'Center alignment',
    'axis_alignment':    'Axis alignment',
}

STATE_ORDER  = [False, True]   # shuffle column: False=Real, True=Shuffle
STATE_LABELS = {False: 'Real', True: 'Shuffle'}

# Task-event markers (stim onset / end of the 0.2 s stimulus), matching the
# epoch boundaries the fixed-epoch version of this analysis used.
EVENT_TIMES = (0.0, 0.2)


def state_shades(roi_name):
    """(real_colour, shuffle_colour): full-saturation ROI hue for Real,
    lighter/desaturated variant for Shuffle -- same recipe as
    intrinsic_dim_epochs.py's epoch_shades, just relabeled."""
    base = ROI_COLOURS.get(roi_name, '#ffffff')
    h, s, v = mcolors.rgb_to_hsv(mcolors.to_rgb(base))
    real_colour    = base
    shuffle_colour = mcolors.hsv_to_rgb((h, s * 0.4, min(1.0, v * 1.15 + 0.1)))
    return real_colour, shuffle_colour


# -- Loading -------------------------------------------------------------------

def load_all_subjects(subjects, lockType, voxRes, bids_root):
    """Loads + concatenates every subject's glueFits CSV that exists (see
    constants.glue_fits_csv_path for the per-subject sourceRecon/glueFits
    layout). Returns one long DataFrame with subjID/band/condition/roi/scheme/
    t_center/shuffle as plain columns, or an empty DataFrame if none found."""
    dfs = []
    for subjID in subjects:
        fpath = glue_fits_csv_path(bids_root, subjID, lockType, voxRes)
        if not os.path.exists(fpath):
            print(f'  missing: {fpath}')
            continue
        dfs.append(pd.read_csv(fpath))
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# -- Aggregation -----------------------------------------------------------------

def timecourse(df, band, roi, condition, scheme, shuffle_val, metric):
    """
    Cross-subject mean +/- SEM time course for one
    (band, roi, condition, scheme, shuffle) cell.

    Returns (times, mean, sem, n_per_time) as numpy arrays sorted by time,
    or (None, None, None, None) if no rows match.
    """
    sel = df[(df['band'] == band) & (df['roi'] == roi) &
             (df['condition'] == condition) & (df['scheme'] == scheme) &
             (df['shuffle'] == shuffle_val)].dropna(subset=[metric])
    if sel.empty:
        return None, None, None, None

    grouped = sel.groupby('t_center')[metric]
    times = np.array(sorted(grouped.groups.keys()), dtype=float)
    mean  = grouped.mean().reindex(times).to_numpy()
    n     = grouped.count().reindex(times).to_numpy()
    # ddof=1 sample SD (pandas default) -- NaN where a timepoint has n=1.
    sd    = grouped.std().reindex(times).to_numpy()
    sem   = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), 0.0)
    return times, mean, sem, n


def paired_ttest_by_time(df, band, roi, condition, scheme, metric):
    """
    Per-timepoint paired (real vs shuffle, matched by subjID) t-test.
    Returns (times, p_values); p is NaN where fewer than 2 subjects have both
    a real and a shuffle value at that timepoint. UNCORRECTED across
    timepoints -- see module docstring.
    """
    sel = df[(df['band'] == band) & (df['roi'] == roi) &
             (df['condition'] == condition) & (df['scheme'] == scheme)]
    if sel.empty:
        return np.array([]), np.array([])

    times, pvals = [], []
    for t, sub in sel.groupby('t_center'):
        real = sub[sub['shuffle'] == False].dropna(subset=[metric]).set_index('subjID')[metric]
        shuf = sub[sub['shuffle'] == True].dropna(subset=[metric]).set_index('subjID')[metric]
        common = real.index.intersection(shuf.index)
        times.append(float(t))
        if len(common) < 2:
            pvals.append(np.nan)
            continue
        _, p = stats.ttest_rel(real.loc[common].to_numpy(), shuf.loc[common].to_numpy())
        pvals.append(float(p))
    order = np.argsort(times)
    return np.array(times)[order], np.array(pvals)[order]


def present_cells(df, rois, conditions):
    """(roi, condition) pairs actually present in the data, in --rois x
    --conditions order. manifold_capacity.py runs phase conditions for a
    subset of ROIs only (default: visual), so the column set is data-driven
    rather than the full cross product."""
    have = set(zip(df['roi'], df['condition']))
    return [(r, c) for r in rois for c in conditions if (r, c) in have]


# -- Plotting --------------------------------------------------------------------

def _apply_black_style(fig, axes_flat):
    fig.patch.set_facecolor(_BG)
    for ax in axes_flat:
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_FG, which='both', labelsize=10)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)
        ax.grid(True, color=_GRID, linewidth=0.5, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)


def plot_metric_figure(df, metric, scheme, bands, cells, voxRes, outdir):
    """Time-course figure for one metric x scheme: rows = bands, columns =
    (roi, condition) cells. Within each panel: Real (solid) vs Shuffle
    (dashed), cross-subject mean with +/- SEM ribbon."""
    n_rows, n_cols = len(bands), len(cells)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(max(4.2 * n_cols, 10), max(2.9 * n_rows, 7)),
                              sharex=True, squeeze=False)
    _apply_black_style(fig, axes.flatten())

    # Per-row (band) y-limits shared across columns, purely data-driven (see
    # module docstring for why 0 is deliberately not forced into range).
    row_ylim = {}
    for band in bands:
        vmin, vmax = np.inf, -np.inf
        for roi, cond in cells:
            for sh in STATE_ORDER:
                _, mean, sem, _ = timecourse(df, band, roi, cond, scheme, sh, metric)
                if mean is None:
                    continue
                vmin = min(vmin, np.nanmin(mean - sem))
                vmax = max(vmax, np.nanmax(mean + sem))
        if np.isfinite(vmin) and np.isfinite(vmax):
            pad = max(1e-6, (vmax - vmin) * 0.12)
            # Extra bottom room for the significance dot strip.
            row_ylim[band] = (vmin - pad * 2.2, vmax + pad)
        else:
            row_ylim[band] = None

    for r_idx, band in enumerate(bands):
        for c_idx, (roi, cond) in enumerate(cells):
            ax = axes[r_idx, c_idx]
            if row_ylim[band] is not None:
                ax.set_ylim(*row_ylim[band])

            real_colour, shuffle_colour = state_shades(roi)
            state_colour = {False: real_colour, True: shuffle_colour}

            has_data = False
            n_max = 0
            for sh in STATE_ORDER:
                times, mean, sem, n = timecourse(df, band, roi, cond, scheme, sh, metric)
                if mean is None:
                    continue
                has_data = True
                n_max = max(n_max, int(np.nanmax(n)))
                colour = state_colour[sh]
                ax.plot(times, mean, color=colour, linewidth=2.0,
                        linestyle='-' if sh is False else '--',
                        label=STATE_LABELS[sh], zorder=4)
                ax.fill_between(times, mean - sem, mean + sem, color=colour,
                                alpha=0.20, linewidth=0, zorder=3)

            if not has_data:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', color='#555555', fontsize=9)
                continue

            for t_ev in EVENT_TIMES:
                ax.axvline(t_ev, color='#666666', linewidth=0.9, linestyle=':', zorder=2)

            # Uncorrected per-timepoint paired test, drawn as a dot strip near
            # the panel floor (see module docstring).
            t_p, pvals = paired_ttest_by_time(df, band, roi, cond, scheme, metric)
            sig = np.isfinite(pvals) & (pvals < 0.05)
            if sig.any():
                y_lo, y_hi = ax.get_ylim()
                ax.scatter(t_p[sig], np.full(sig.sum(), y_lo + 0.04 * (y_hi - y_lo)),
                           s=10, color=real_colour, alpha=0.9, linewidths=0, zorder=6)

            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))

            if r_idx == 0:
                roi_lbl = roi.capitalize() if roi != 'whole' else 'Whole brain'
                ax.set_title(f'{roi_lbl} | {COND_LABELS.get(cond, cond)}  (n={n_max})',
                             fontsize=12, fontweight='bold', pad=6)
            if r_idx == n_rows - 1:
                ax.set_xlabel('Time from stimulus onset (s)', fontsize=10)
            if c_idx == 0:
                ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=11)
                ax.annotate(BAND_LABELS.get(band, band),
                            xy=(-0.30, 0.5), xycoords='axes fraction',
                            fontsize=12, color=_FG, ha='right', va='center',
                            rotation=90, fontweight='bold')
            if r_idx == 0 and c_idx == n_cols - 1:
                leg = ax.legend(fontsize=9, loc='upper right', framealpha=0.2,
                                 edgecolor='#444444', labelcolor=_FG)
                leg.get_frame().set_facecolor('#1a1a1a')

    win_ms = df['win_ms'].dropna().unique()
    win_txt = f'{win_ms[0]:.0f} ms window' if win_ms.size == 1 else 'sliding window'
    erp = df['remove_erp'].dropna().unique()
    erp_txt = ('ERP removed' if (erp.size == 1 and bool(erp[0]))
               else 'ERP kept' if erp.size == 1 else 'mixed ERP state')

    fig.suptitle(
        f'Manifold Capacity -- {METRIC_LABELS.get(metric, metric)}  |  '
        f'{SCHEME_LABELS.get(scheme, scheme)}  |  {win_txt}  |  {erp_txt}  |  {voxRes}',
        color=_FG, fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0.05, 0, 1, 1])

    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f'glue_capacity_ts_{metric}_scheme{scheme}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


# -- Main ------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate + plot sliding-window glue manifold-capacity results '
                     'across subjects.')
    parser.add_argument('--voxRes',   default='8mm')
    parser.add_argument('--lockType', default='stim', choices=['stim', 'resp'])
    parser.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    parser.add_argument('--bands',    nargs='+', default=list(AMP_PHASE_BANDS))
    parser.add_argument('--rois',     nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'])
    parser.add_argument('--schemes',  nargs='+', type=int, default=[4],
                         choices=sorted(CATEGORY_SCHEMES))
    parser.add_argument('--metrics',  nargs='+', default=METRICS)
    parser.add_argument('--outdir',   default=None,
                         help='Directory for figures. '
                              'Default: <bids_root>/derivatives/glueDecoding/glueFits')
    args = parser.parse_args()

    bids_root = get_bids_root()
    outdir = args.outdir or os.path.join(bids_root, 'derivatives', 'glueDecoding', 'glueFits')

    print(f'aggregate_glue_capacity | voxRes={args.voxRes} | lockType={args.lockType} | '
          f'subjects={args.subjects} | bands={args.bands} | rois={args.rois} | '
          f'conditions={args.conditions} | schemes={args.schemes} | metrics={args.metrics}')
    print(f'Loading per-subject glueFits CSVs from: {bids_root}/derivatives/sub-XX/sourceRecon/glueFits/')

    df = load_all_subjects(args.subjects, args.lockType, args.voxRes, bids_root)
    if df.empty:
        print('No per-subject CSVs found -- nothing to plot.')
        return

    missing = [c for c in ('condition', 't_center') if c not in df.columns]
    if missing:
        print(f'ERROR: loaded CSVs are missing {missing} -- these look like output from the '
              'OLD fixed-epoch manifold_capacity.py (epoch column, amplitude only). '
              'Re-run manifold_capacity.py with --force to regenerate them.')
        return

    print(f'Loaded {df["subjID"].nunique()}/{len(args.subjects)} subjects, {len(df)} rows total.')

    cells = present_cells(df, args.rois, args.conditions)
    if not cells:
        print('No (roi, condition) cells present in the data -- nothing to plot.')
        return
    print(f'(roi, condition) cells: {cells}')

    for metric in args.metrics:
        if metric not in df.columns:
            print(f'  SKIP metric {metric!r}: not a column in the loaded results.')
            continue
        for scheme in args.schemes:
            plot_metric_figure(df, metric, scheme, args.bands, cells, args.voxRes, outdir)

    print('\nDone.')


if __name__ == '__main__':
    main()
