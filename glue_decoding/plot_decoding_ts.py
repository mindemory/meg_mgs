#!/usr/bin/env python3
"""
plot_decoding_ts.py

Aggregates per-subject decoding timeseries .npz files and produces
publication-quality figures.

Figure layout (one figure per condition: ampOnly / ampPhase):
  rows = frequency bands  (theta / alpha / beta / lowgamma)
  cols = ROIs             (visual / parietal / frontal)

  Each cell contains:
    [Main panel, 80% height]
      Mean +/- SEM circular decoding error across subjects (lower = better).
      Dashed line: mean +/- SEM shuffle baseline.
      Horizontal line at 90 deg: theoretical chance for circular decoding.
      Event flag lines: Stim at 0.0 s (top), Delay Onset at 0.2 s (mid).

    [Bar panel, 20% height]
      Trials binned per-subject into 4 quartiles by mean circular error
      in two epochs: Stim (0.0-0.2 s) | Delay (0.2-1.7 s).
      Bar = mean +/- SEM across subjects per quartile.
      Scatter dots = individual-subject mean error per quartile.
      Quartile colours: ROI colour shaded Q1 (darkest) -> Q4 (lightest).

Color scheme mirrors plot_timeseries.py:
  visual  = #FFC629  mango/Bumble
  parietal= #A78BFA  soft violet
  frontal = #34D399  emerald mint
  Background = #000000 true black
"""

import os, sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgb

from constants import get_bids_root, ROI_NAMES

# ── Design constants (mirror plot_timeseries.py) ──────────────────────────────

_BG        = '#000000'
_FG        = '#e0e0e0'
_GRID      = '#1c1c1c'
_FLAG_LINE = '#777777'
_FLAG_TXT  = '#cccccc'
_CHANCE    = '#444444'     # horizontal chance line

ROI_COLOURS = {
    'visual':   '#FFC629',
    'parietal': '#A78BFA',
    'frontal':  '#34D399',
}

BAND_LABELS = {
    'theta':     'Theta\n(4-8 Hz)',
    'alpha':     'Alpha\n(8-12 Hz)',
    'beta':      'Beta\n(13-30 Hz)',
    'lowgamma':  'Low gamma\n(30-80 Hz)',
}

# Event flags: (time_s, label, label_y_frac)
EVENT_FLAGS = [
    (0.0,  'Stim',        0.93),
    (0.2,  'Delay Onset', 0.55),
]

EPOCH_WINDOWS = {
    'stim':  (0.0, 0.2),
    'delay': (0.2, 1.7),
}
EPOCH_LABELS = {'stim': 'Stim\n(0-0.2 s)', 'delay': 'Delay\n(0.2-1.7 s)'}

N_QUANTILES = 4    # quartiles

# ── I/O helpers ───────────────────────────────────────────────────────────────

def cell_path(bids_root, subjID, band, roi, condition, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base    = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'decodingTS')
    return os.path.join(
        base,
        f'{subName}_task-mgs_decodingTS_{condition}_{band}_{roi}_{voxRes}.npz')


def load_all_subjects(subjects, bids_root, voxRes, bands, rois, conditions, outdir=None):
    """
    Returns list (one entry per subject) of dicts:
        data[(band, roi, condition)] = loaded npz dict or None
    """
    all_data = []
    for subjID in subjects:
        d = {}
        for band in bands:
            for roi in rois:
                for condition in conditions:
                    fp = cell_path(bids_root, subjID, band, roi,
                                    condition, voxRes, outdir)
                    if os.path.exists(fp):
                        d[(band, roi, condition)] = dict(np.load(fp, allow_pickle=True))
        all_data.append(d if d else None)
    return all_data


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_timeseries(all_data, band, roi, condition):
    """
    Returns (tv, mean_err, sem_err, mean_shuf, sem_shuf) or Nones.
    err/shuf are trial-averaged per subject, then subject-averaged.
    """
    errors_list  = []
    shuffle_list = []
    tv           = None

    for d in all_data:
        if d is None:
            continue
        k = (band, roi, condition)
        if k not in d:
            continue
        npz = d[k]
        errors_list.append(npz['errors'].mean(axis=0))        # (T,)
        shuffle_list.append(npz['shuffle_errors'].mean(axis=0))
        if tv is None:
            tv = npz['time_vector']

    if not errors_list:
        return None, None, None, None, None

    err  = np.stack(errors_list)   # (n_subj, T)
    shuf = np.stack(shuffle_list)
    n    = err.shape[0]

    return (tv,
            err.mean(0),  err.std(0) / np.sqrt(n),
            shuf.mean(0), shuf.std(0) / np.sqrt(n))


def compute_epoch_quartiles(all_data, band, roi, condition):
    """
    Per-subject quartile binning of trial-level circular errors.

    For each epoch (stim / delay):
      - Compute per-trial mean circular error within the epoch window.
      - Bin trials into N_QUANTILES quartiles.
      - Record the mean error of trials in each quartile.
    Aggregate across subjects: mean +/- SEM per quartile.

    Returns:
        dict: epoch_name -> (q_means, q_sems, q_subj_vals)
          q_means, q_sems : (N_QUANTILES,)
          q_subj_vals     : list of length N_QUANTILES, each entry is an
                            array of subject values for scatter plotting.
    """
    ep_collector = {ep: [[] for _ in range(N_QUANTILES)]
                    for ep in EPOCH_WINDOWS}

    for d in all_data:
        if d is None:
            continue
        k = (band, roi, condition)
        if k not in d:
            continue
        npz = d[k]
        tv     = npz['time_vector']   # (T,)
        errors = npz['errors']        # (N, T)

        for ep_name, (t0, t1) in EPOCH_WINDOWS.items():
            mask = (tv >= t0) & (tv <= t1)
            if not mask.any():
                continue

            # Per-trial mean circular error in this epoch (N,)
            trial_err = errors[:, mask].mean(axis=1)

            # Bin into quartiles (per subject). Lower bound is exclusive
            # except for the first bin, so a trial sitting exactly on an
            # internal percentile boundary lands in exactly one quartile
            # (previously both bounds were inclusive everywhere, so boundary
            # trials were double-counted in two adjacent quartiles).
            q_bounds = np.percentile(trial_err, np.linspace(0, 100, N_QUANTILES + 1))
            for q in range(N_QUANTILES):
                lo = q_bounds[q]
                hi = q_bounds[q + 1]
                if q == 0:
                    in_bin = (trial_err >= lo) & (trial_err <= hi)
                else:
                    in_bin = (trial_err > lo) & (trial_err <= hi)
                if in_bin.any():
                    ep_collector[ep_name][q].append(float(trial_err[in_bin].mean()))

    result = {}
    for ep_name in EPOCH_WINDOWS:
        q_means   = []
        q_sems    = []
        q_subj    = []
        for q in range(N_QUANTILES):
            vals = np.asarray(ep_collector[ep_name][q])
            q_subj.append(vals)
            if len(vals) >= 1:
                q_means.append(vals.mean())
                q_sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            else:
                q_means.append(np.nan)
                q_sems.append(np.nan)
        result[ep_name] = (np.array(q_means), np.array(q_sems), q_subj)

    return result


# ── Colour helpers ────────────────────────────────────────────────────────────

def quartile_colours(roi, n=N_QUANTILES):
    """
    Return n shades of the ROI colour from darkest (Q1) to lightest (Q_n) --
    Q1 blended toward black, Q_n blended toward white, hue preserved
    throughout. (Previously this blended everything toward black with
    DEcreasing weight, which made Q1 the brightest/most-saturated shade and
    Q_n the darkest -- the opposite of the intended darkest->lightest order.)
    """
    base = np.array(to_rgb(ROI_COLOURS[roi]))
    black, white = np.zeros(3), np.ones(3)
    t_vals = np.linspace(0.15, 0.85, n)   # 0=black, 0.5=base colour, 1=white
    colours = []
    for t in t_vals:
        if t <= 0.5:
            c = black + (base - black) * (t / 0.5)
        else:
            c = base + (white - base) * ((t - 0.5) / 0.5)
        colours.append(tuple(np.clip(c, 0, 1)))
    return colours


# ── Plotting ──────────────────────────────────────────────────────────────────

def _style_ax(ax, spine_col='#333333'):
    """Apply dark-style shared formatting."""
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color(spine_col)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)


def plot_timeseries_panel(ax, tv, mean_err, sem_err, mean_shuf, sem_shuf,
                           roi, is_bottom_row, is_left_col, show_title, title_str):
    """Draw decoding error timeseries + shuffle into ax."""
    col  = ROI_COLOURS[roi]
    t0   = float(tv[0])
    t1   = float(tv[-1])

    # Chance line
    ax.axhline(90, color=_CHANCE, lw=0.8, ls=':', zorder=1)

    # Shuffle baseline (dashed, same colour, dimmer)
    ax.fill_between(tv, mean_shuf - sem_shuf, mean_shuf + sem_shuf,
                     alpha=0.15, color=col, zorder=2)
    ax.plot(tv, mean_shuf, color=col, lw=0.9, ls='--', alpha=0.55, zorder=2)

    # Real decoding
    ax.fill_between(tv, mean_err - sem_err, mean_err + sem_err,
                     alpha=0.30, color=col, zorder=3)
    ax.plot(tv, mean_err, color=col, lw=1.5, zorder=3)

    # xlim/ylim must be set BEFORE reading ax.get_ylim() for flag placement --
    # otherwise the flags are positioned against matplotlib's autoscaled
    # limits (whatever they were before set_ylim(0, 180) below), not the
    # actual [0, 180] range the panel ends up with.
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, 180)

    # Event flags
    y_lo, y_hi = ax.get_ylim()
    for (t_flag, label, y_frac) in EVENT_FLAGS:
        ax.axvline(t_flag, color=_FLAG_LINE, lw=0.8, ls='--', zorder=4)
        ax.text(t_flag, y_lo + y_frac * (y_hi - y_lo), label,
                rotation=90, va='top', ha='right',
                fontsize=6.5, color=_FLAG_TXT, zorder=5)

    if is_left_col:
        ax.set_ylabel('Circular error (deg)', fontsize=8, color=_FG)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(45))
    else:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(45))
        ax.set_yticklabels([])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    if is_bottom_row:
        ax.set_xlabel('Time (s)', fontsize=8, color=_FG)
    else:
        ax.set_xticklabels([])

    ax.grid(True, color=_GRID, lw=0.4, zorder=0)
    _style_ax(ax)

    if show_title:
        ax.set_title(title_str, color=_FG, fontsize=10, fontweight='bold', pad=4)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))


def plot_bar_panel(ax, epoch_quartiles, roi, show_ylabel):
    """
    Draw quartile bar strip below timeseries.
    Two groups (stim | delay), N_QUANTILES bars each.
    Bars: mean circular error per quartile across subjects.
    Dots: individual subject values.
    """
    col     = ROI_COLOURS[roi]
    q_cols  = quartile_colours(roi, N_QUANTILES)
    bar_w   = 0.6
    gap     = 1.5    # gap between stim and delay groups
    eps     = 0.12   # jitter for scatter

    group_offsets = {'stim': 0, 'delay': N_QUANTILES + gap}
    rng = np.random.default_rng(42)

    all_vals = []
    for ep_name in EPOCH_WINDOWS:
        q_means, q_sems, q_subj = epoch_quartiles[ep_name]
        x0 = group_offsets[ep_name]
        for q in range(N_QUANTILES):
            x = x0 + q
            v = q_means[q]
            s = q_sems[q]
            c = q_cols[q]
            if not np.isnan(v):
                ax.bar(x, v, width=bar_w, color=c, alpha=0.85, zorder=3,
                        linewidth=0, align='center')
                ax.errorbar(x, v, yerr=s, fmt='none', ecolor=_FG,
                             elinewidth=1.0, capsize=2, zorder=4)
                # Subject scatter
                subj_vals = q_subj[q]
                if len(subj_vals):
                    jitter = rng.uniform(-eps, eps, len(subj_vals))
                    ax.scatter(x + jitter, subj_vals,
                                s=8, color='white', alpha=0.7,
                                edgecolors='none', zorder=5)
                    all_vals.extend(subj_vals.tolist())
            all_vals.append(v)

    # Group labels
    for ep_name, label in EPOCH_LABELS.items():
        x0 = group_offsets[ep_name]
        ax.text(x0 + (N_QUANTILES - 1) / 2, -0.05,
                label, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=6.5, color=_FG)

    ax.set_xlim(-0.6, max(group_offsets.values()) + N_QUANTILES - 0.4)
    y_max = max((v for v in all_vals if not np.isnan(v)), default=90)
    ax.set_ylim(0, y_max * 1.25)
    ax.set_xticks([])

    if show_ylabel:
        ax.set_ylabel('Circ. err. (deg)', fontsize=7, color=_FG)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, integer=True))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.grid(axis='y', color=_GRID, lw=0.4, zorder=0)
    _style_ax(ax)


def make_figure(all_data, bands, rois, condition, voxRes, outdir_fig):
    """Build and save one figure for a given condition."""

    n_bands = len(bands)
    n_rois  = len(rois)

    # Height ratios: for each band row, [4 (timeseries), 1 (bar)]
    ts_ratio  = 4
    bar_ratio = 1
    all_ratios = [ts_ratio if row % 2 == 0 else bar_ratio
                  for row in range(n_bands * 2)]

    fig_w = 3.8 * n_rois
    # 1.5 in per band-row's combined (timeseries + bar) height -- NOT
    # squared in n_bands (a previous version had `* n_bands` twice here,
    # making a 4-band figure ~30in tall instead of the intended ~7.5in).
    fig_h = 1.5 * n_bands * (ts_ratio + bar_ratio) / ts_ratio
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG)

    gs = gridspec.GridSpec(
        n_bands * 2, n_rois,
        figure       = fig,
        height_ratios= all_ratios,
        hspace       = 0.05,
        wspace       = 0.25,
        left         = 0.10,
        right        = 0.97,
        top          = 0.93,
        bottom       = 0.07,
    )

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois):
            ax_ts  = fig.add_subplot(gs[r_idx * 2,     c_idx])
            ax_bar = fig.add_subplot(gs[r_idx * 2 + 1, c_idx])

            tv, mean_err, sem_err, mean_shuf, sem_shuf = aggregate_timeseries(
                all_data, band, roi, condition)
            epoch_quartiles = compute_epoch_quartiles(
                all_data, band, roi, condition)

            is_left_col   = (c_idx == 0)
            is_bottom_row = (r_idx == n_bands - 1)
            show_title    = (r_idx == 0)
            title_str     = roi.capitalize()

            if tv is not None:
                plot_timeseries_panel(
                    ax_ts, tv, mean_err, sem_err, mean_shuf, sem_shuf,
                    roi, is_bottom_row, is_left_col, show_title, title_str)
            else:
                ax_ts.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax_ts.transAxes, color=_FG, fontsize=9)
                _style_ax(ax_ts)

            if epoch_quartiles:
                plot_bar_panel(ax_bar, epoch_quartiles, roi,
                                show_ylabel=is_left_col)
            else:
                _style_ax(ax_bar)

            # Band label on left edge
            if c_idx == 0:
                ax_ts.annotate(BAND_LABELS.get(band, band),
                                xy=(-0.38, 0.5), xycoords='axes fraction',
                                fontsize=9, color=_FG,
                                ha='right', va='center',
                                rotation=90, fontweight='bold')

    cond_label = 'Amplitude only' if condition == 'ampOnly' \
                 else 'Amplitude + Phase'
    fig.suptitle(f'Stim-locked Decoding  |  {cond_label}  |  {voxRes}',
                  color=_FG, fontsize=14, fontweight='bold', y=0.97)

    os.makedirs(outdir_fig, exist_ok=True)
    fpath = os.path.join(outdir_fig, f'decoding_ts_{condition}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and plot decoding timeseries across subjects.')
    parser.add_argument('--voxRes',      default='8mm')
    parser.add_argument('--subjects',    nargs='+', type=int,
                        default=[1,2,3,4,5,6,7,9,10,12,13,15,17,18,19,
                                  23,24,25,29,31,32])
    parser.add_argument('--bands',       nargs='+',
                        default=['theta', 'alpha', 'beta', 'lowgamma'])
    parser.add_argument('--rois',        nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions',  nargs='+', default=['ampOnly', 'ampPhase'])
    parser.add_argument('--outdir',      default=None,
                        help='Directory containing the per-subject .npz files.')
    parser.add_argument('--figdir',      default=None,
                        help='Directory to save figures (default: same as outdir '
                             'or BIDS derivatives/decoding_figures).')
    args = parser.parse_args()

    bids_root = get_bids_root()
    figdir    = args.figdir or (args.outdir if args.outdir else
                                 os.path.join(bids_root, 'derivatives',
                                              'decoding_figures'))

    print(f'Loading data | {args.voxRes} | subjects={args.subjects} | '
          f'bands={args.bands} | rois={args.rois} | conditions={args.conditions}')

    all_data = load_all_subjects(
        args.subjects, bids_root, args.voxRes,
        args.bands, args.rois, args.conditions, args.outdir)

    n_loaded = sum(1 for d in all_data if d)
    print(f'Loaded data for {n_loaded}/{len(args.subjects)} subjects.')

    for condition in args.conditions:
        print(f'\n-- Plotting condition: {condition} --')
        make_figure(all_data, args.bands, args.rois, condition,
                     args.voxRes, figdir)

    print('\nAll figures saved.')


if __name__ == '__main__':
    main()
