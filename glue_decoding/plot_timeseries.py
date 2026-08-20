#!/usr/bin/env python3
"""
plot_timeseries.py

Plots mean +/- SEM timeseries of source-space MEG activity averaged across
subjects, for stim-locked and resp-locked epochs, for all G04 frequency
bands (theta/alpha/beta/lowgamma/highgamma amplitude) and the requested
ROIs (visual, parietal, frontal by default; whole-brain is opt-in --
pass `--rois visual parietal frontal whole`).

Data source: G04 Hilbert amplitude, mean across sources in ROI, mean across
trials. (G03 raw broadband ["unfiltered"] is no longer plotted -- that row
wasn't useful -- though it's still loaded internally when 'whole' is
requested, to derive per-ROI source indices for G04's whole-grid slicing.)

By default, each subject's (band, ROI) curve is z-scored against its own
mean/std within a baseline window (see BASELINE_WINDOWS: pre-stim fixation
for stim-locked, pre-response tail segment for resp-locked) before being
averaged across subjects -- otherwise raw voltage/amplitude scale differs
enough across subjects/bands/ROIs that a cross-subject mean is dominated by
whichever subject happens to have the largest scale. Pass --no_baseline to
plot raw units instead.

'whole' is opt-in rather than default because including it forces a full
whole-grid load of every G03/G04 file (8-10GB each); without it, each ROI is
read directly from precompute_roi_splits.py's small precomputed per-ROI
cache instead (see load_subject_timeseries) -- run that script first if the
caches don't exist yet.

Time windows & event flags (hard-coded per lock type):
  stim-locked : -1.0 to +1.7 s   | Stim at 0 s, Delay Onset at +0.2 s
  resp-locked : -4.5 to -0.5 s   | Delay Onset at -4 s, R Onset at -2.5 s,
                                    Feedback at -2 s

Parallelism:
  Subjects are processed in parallel using joblib (processes, not threads,
  to avoid GIL contention in numpy operations). Default n_jobs = len(subjects)
  -- one worker per requested subject. Without 'whole', per-subject IO is
  small (ROI caches only), so OMP/MKL/OPENBLAS_NUM_THREADS are pinned to 1
  below (same as run_glue_cell.py) -- otherwise each of these many
  concurrent processes would spawn its own multi-threaded BLAS pool for
  numpy's tiny .mean() reductions, oversubscribing the machine's cores far
  more than the actual compute needs.

Usage:
    python plot_timeseries.py [--voxRes 8mm] [--lockTypes stim resp]
                              [--rois visual parietal frontal]
                              [--subjects 1 2 ...]
                              [--outdir <path>]
                              [--n_jobs 8]
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from joblib import Parallel, delayed

from atlas import load_atlas_masks, roi_local_indices
from constants import AMP_ONLY_BANDS, SUBJECT_LIST, ROI_NAMES, get_bids_root
from io_g03 import load_g03_unfiltered
from io_g04 import load_g04_band

# ── Time windows and event flags per lock type ────────────────────────────────
TIME_WINDOWS = {
    'stim': (-1.0, 1.7),
    'resp': (-4.5, -0.5),
}

# Baseline window used to z-score each subject's curve before cross-subject
# averaging (per (subject, band, ROI) independently -- amplitude scale
# differs hugely across bands/ROIs/subjects, so raw units aren't comparable
# in a cross-subject mean). stim uses the pre-stimulus fixation period;
# resp has no true ITI within its cropped window (it ends at -0.5s, well
# before the response), so the tail segment just before that crop boundary
# is used as the closest available baseline proxy. Both windows are subsets
# of TIME_WINDOWS above, so they can be computed straight from a subject's
# already-cropped curve.
BASELINE_WINDOWS = {
    'stim': (-1.0, 0.0),
    'resp': (-1.0, -0.5),
}

# Each flag: (time_s, label, label_y_frac)
# label_y_frac controls vertical placement of the text label (0=bottom, 1=top).
# Adjacent flags are staggered (alternating high / mid) so their rotated
# labels don't overlap when the time gap is small (e.g. Stim/Delay at 0.0/0.2s).
EVENT_FLAGS = {
    'stim': [
        (0.0,  'Stim',         0.93),   # top
        (0.2,  'Delay Onset',  0.55),   # mid -- staggered away from Stim label
    ],
    'resp': [
        (-4.0, 'Delay Onset',  0.93),   # top
        (-2.5, 'R Onset',      0.75),   # mid-high
        (-2.0, 'Feedback',     0.55),   # mid
    ],
}

# ── Visual design ────────────────────────────────────────────────────────────────────────────────
_BG        = '#000000'   # true black
_FG        = '#e0e0e0'
_GRID      = '#1c1c1c'
_FLAG_LINE = '#777777'
_FLAG_TXT  = '#cccccc'

# Font sizes -- same scale as plot_circular_tgm.py / plot_visual_geometry_*.py /
# plot_ccgp_epochs.py, so every figure in this project reads consistently.
FS_SUPTITLE   = 18
FS_PANEL_TTL  = 14
FS_AXIS_LABEL = 13
FS_ROW_LABEL  = 14
FS_TICK       = 10
FS_FLAG       = 11
LW_MEAN       = 2.6   # the curve itself, was 1.6

# ROI colours: mango/bumble for visual, soft violet for parietal,
# emerald mint for frontal -- all vivid on true black.
ROI_COLOURS = {
    'visual':   '#FFC629',   # mango / Bumble amber
    'parietal': '#A78BFA',   # soft violet
    'frontal':  '#34D399',   # emerald mint
    'whole':    '#E76F51',   # coral (unchanged)
}

# 'unfiltered' (G03 broadband) is intentionally excluded -- that row wasn't
# useful and has been dropped from the plotted grid (see load_subject_timeseries).
BAND_ORDER = ['theta', 'alpha', 'beta', 'lowgamma', 'highgamma']
BAND_LABELS = {
    'theta':      'Theta\n(4-8 Hz)',
    'alpha':      'Alpha\n(8-12 Hz)',
    'beta':       'Beta\n(13-30 Hz)',
    'lowgamma':   'Low gamma\n(30-80 Hz)',
    'highgamma':  'High gamma\n(80-150 Hz)',
}


# ── Data loading (one worker per subject, runs in parallel) ───────────────────

def _filter_inside_pos(inside_pos, data, atlas_masks):
    """Drop source columns whose 1-based inside_pos exceeds atlas grid."""
    n_grid     = len(next(iter(atlas_masks.values())))
    valid_mask = (inside_pos >= 1) & (inside_pos <= n_grid)
    if valid_mask.all():
        return inside_pos, data
    valid_cols = np.where(valid_mask)[0]
    return inside_pos[valid_cols], data[:, :, valid_cols]


def _crop_to_window(tv, curve, t_min, t_max):
    """Crop (time_vector, curve) to [t_min, t_max]."""
    mask = (tv >= t_min) & (tv <= t_max)
    return tv[mask], curve[mask]


def _reduce_curve(data, tv_crop_mask, idx=None):
    """
    Mean across trials then sources -> (n_crop_times,) cropped to tv_crop_mask.
    idx, if given, first slices data[:, :, idx] (whole-grid path); omit it
    when data is already ROI-sliced (roi-cache fast path).
    """
    if idx is not None:
        if idx.size == 0:
            return None
        data = data[:, :, idx]
    if data.shape[2] == 0:
        return None
    curve = data.mean(axis=0).mean(axis=1)
    return curve[tv_crop_mask]


def _baseline_zscore(curve, tv_crop, baseline_window):
    """
    Z-score curve using its own mean/std within baseline_window of tv_crop --
    per (subject, band, ROI) independently, since amplitude scale differs
    across bands and ROIs and raw units aren't comparable across subjects.

    baseline_window: (b_min, b_max), or None to disable (returns curve as-is).
    Falls back to a mean-subtract only (no /std) if the baseline segment has
    ~zero variance, to avoid dividing by ~0.
    """
    if curve is None or tv_crop is None or baseline_window is None:
        return curve
    b_min, b_max = baseline_window
    b_mask = (tv_crop >= b_min) & (tv_crop <= b_max)
    if not b_mask.any():
        return curve
    b_mean = curve[b_mask].mean()
    b_std = curve[b_mask].std()
    if b_std < 1e-12:
        return curve - b_mean
    return (curve - b_mean) / b_std


def load_subject_timeseries(subjID, lockType, voxRes, bids_root,
                             atlas_masks, rois_all, t_min, t_max,
                             baseline_window=None, bands=None):
    """
    Load and reduce one subject's data to per-ROI mean timeseries curves,
    cropped to [t_min, t_max].

    If 'whole' is NOT in rois_all (the default), each ROI is loaded directly
    from precompute_roi_splits.py's small per-ROI cache -- the whole-grid
    G03/G04 files are never touched. If 'whole' IS requested, the whole-grid
    files are loaded once and sliced in-memory for every ROI (including
    'whole'), same as before -- there's no cache-based shortcut once the
    whole-grid load is unavoidable anyway.

    baseline_window: (b_min, b_max) or None. If given, each (band, ROI)
    curve is independently z-scored against its own mean/std within this
    window (see _baseline_zscore) before being returned -- this happens
    per subject, so cross-subject amplitude-scale differences don't distort
    the aggregate() mean/SEM computed afterwards.

    Returns dict:
        result[band][roi]           = (n_crop_times,) float, or None
        result['time_vectors'][band] = (n_crop_times,) float, or None

    Called in parallel via joblib -- must be importable at top level.
    """
    need_whole = 'whole' in rois_all
    result = {'time_vectors': {}}
    roi_idx = None  # only populated (and only needed) on the whole-grid path

    # G03 (broadband/"unfiltered") is no longer plotted -- that row wasn't
    # useful -- but on the whole-grid path we still need one G03 load to
    # derive inside_pos -> per-ROI source indices for slicing G04's
    # whole-grid arrays (G04 has no inside_pos of its own; see io_g04.py).
    # On the ROI-only fast path, G04's per-ROI caches are already sliced
    # offline by precompute_roi_splits.py, so G03 isn't needed at all there.
    if need_whole:
        try:
            g03 = load_g03_unfiltered(subjID, lockType, voxRes, bids_root)
        except (FileNotFoundError, OSError) as e:
            # OSError: open_h5 exhausted its retries (e.g. file mid-write).
            # Skip just G03 for this subject rather than crashing the whole
            # joblib batch and losing every other subject's results.
            print(f'  sub-{subjID:02d}: G03 missing/failed to open: {e}', flush=True)
            g03 = None

        if g03 is not None:
            inside_pos, g03_data = _filter_inside_pos(
                g03['inside_pos'], g03['data'], atlas_masks)
            roi_idx = {}
            for roi in rois_all:
                roi_idx[roi] = (np.arange(g03_data.shape[2]) if roi == 'whole'
                                 else roi_local_indices(atlas_masks, inside_pos, roi))
            del g03_data, g03

    # ── G04 band amplitudes ───────────────────────────────────────────────────
    # Only the requested bands: this loop is the expensive part (one G04 load
    # per band), so restricting it here rather than at plot time is what makes
    # a theta/alpha/beta run ~40% cheaper than an all-five run.
    for band in (AMP_ONLY_BANDS if bands is None else bands):
        result[band] = {}

        if need_whole:
            try:
                g04 = load_g04_band(subjID, lockType, band, voxRes, bids_root,
                                     want_phase=False)
            except (FileNotFoundError, ValueError, OSError) as e:
                # OSError: open_h5 exhausted its retries for this one band
                # file -- skip just this band for this subject rather than
                # crashing the whole joblib batch (see G03 handling above).
                print(f'  sub-{subjID:02d} {band}: {e}', flush=True)
                result['time_vectors'][band] = None
                for roi in rois_all:
                    result[band][roi] = None
                continue

            amp        = g04['amp']
            tv_g04     = g04['time_vector']
            tv_g04_mask = (tv_g04 >= t_min) & (tv_g04 <= t_max)
            tv_g04_crop = tv_g04[tv_g04_mask]
            result['time_vectors'][band] = tv_g04_crop
            for roi in rois_all:
                # roi_idx may be None if G03 failed above (need_whole=True but
                # G03 load errored) -- without it we can only still resolve
                # the 'whole' ROI (all columns, no inside_pos mapping needed).
                if roi_idx is not None:
                    idx = roi_idx[roi]
                elif roi == 'whole':
                    idx = np.arange(amp.shape[2])
                else:
                    result[band][roi] = None
                    continue
                curve = _reduce_curve(amp, tv_g04_mask, idx)
                result[band][roi] = _baseline_zscore(curve, tv_g04_crop, baseline_window)
            del amp, g04
            continue

        # ROI-only fast path (mirrors the G03 branch above).
        tv_g04_mask = None
        tv_g04_crop = None
        for roi in rois_all:
            try:
                g04_roi = load_g04_band(subjID, lockType, band, voxRes, bids_root,
                                         want_phase=False, roi=roi)
            except (FileNotFoundError, ValueError, OSError) as e:
                print(f'  sub-{subjID:02d} {band} roi={roi}: {e}', flush=True)
                result[band][roi] = None
                continue
            if tv_g04_mask is None:
                tv_g04 = g04_roi['time_vector']
                tv_g04_mask = (tv_g04 >= t_min) & (tv_g04 <= t_max)
                tv_g04_crop = tv_g04[tv_g04_mask]
                result['time_vectors'][band] = tv_g04_crop
            curve = _reduce_curve(g04_roi['amp'], tv_g04_mask)
            result[band][roi] = _baseline_zscore(curve, tv_g04_crop, baseline_window)
        if tv_g04_mask is None:
            result['time_vectors'][band] = None

    print(f'  sub-{subjID:02d} done', flush=True)
    return result


# ── Cross-subject aggregation ─────────────────────────────────────────────────

def aggregate(all_results, band, roi):
    """Returns (time_vector, mean, sem) or (None, None, None)."""
    curves = []
    tv     = None
    for r in all_results:
        if r is None:
            continue
        curve = r.get(band, {}).get(roi)
        if curve is None:
            continue
        curves.append(curve)
        if tv is None:
            tv = r['time_vectors'].get(band)
    if not curves or tv is None:
        return None, None, None
    stacked = np.stack(curves, axis=0)
    return tv, stacked.mean(axis=0), stacked.std(axis=0) / np.sqrt(len(curves))


# ── Plotting ──────────────────────────────────────────────────────────────────

def _apply_black_style(fig, axes_flat):
    fig.patch.set_facecolor(_BG)
    for ax in axes_flat:
        ax.set_facecolor(_BG)
        ax.tick_params(colors=_FG, which='both', labelsize=FS_TICK)
        ax.xaxis.label.set_color(_FG)
        ax.yaxis.label.set_color(_FG)
        ax.title.set_color(_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(_GRID)


def _draw_event_flags(ax, flags, y_lim):
    """Draw vertical flag lines and rotated text labels."""
    for t_flag, label, y_frac in flags:
        ax.axvline(t_flag, color=_FLAG_LINE, linewidth=1.3,
                   linestyle=':', alpha=0.85, zorder=3)
        y_pos = y_lim[0] + y_frac * (y_lim[1] - y_lim[0])
        ax.text(t_flag, y_pos, label,
                color=_FLAG_TXT, fontsize=FS_FLAG, ha='left', va='top',
                rotation=90, rotation_mode='anchor',
                fontweight='bold', zorder=4,
                transform=ax.get_xaxis_transform() if False else ax.transData)


def plot_timeseries_figure(all_results, rois_all, lockType, voxRes, outdir,
                            baselined=True, bands=None):
    """
    One figure per lockType:
        rows = bands (6)
        cols = ROIs  (4)
    """
    bands  = [b for b in (bands if bands is not None else BAND_ORDER)
              if b in BAND_ORDER]
    n_rows = len(bands)
    n_cols = len(rois_all)
    t_min, t_max = TIME_WINDOWS.get(lockType, (-1.0, 2.0))
    flags        = EVENT_FLAGS.get(lockType, [])

    fig_w = max(4.8 * n_cols, 14)
    # 3.4 per row, not 2.8, and no 12-inch floor: with three bands instead of
    # five a fixed floor stretched each panel vertically and thinned the curve.
    fig_h = 3.4 * n_rows + 1.2

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h),
                              sharex=False, sharey=False,
                              squeeze=False)
    _apply_black_style(fig, axes.flatten())

    # First pass: aggregate every (band, ROI) cell once, and find each row's
    # (band's) symmetric y-limit -- the max abs value across the mean+/-SEM
    # band over ALL ROI columns in that row -- so every column in a row
    # shares one consistent, zero-centered scale.
    curves = {}
    row_ylim = {}
    for band in bands:
        row_max_abs = 0.0
        for roi in rois_all:
            tv, mean_curve, sem_curve = aggregate(all_results, band, roi)
            curves[(band, roi)] = (tv, mean_curve, sem_curve)
            if tv is None:
                continue
            row_max_abs = max(row_max_abs,
                               np.max(np.abs(mean_curve + sem_curve)),
                               np.max(np.abs(mean_curve - sem_curve)))
        row_ylim[band] = (-row_max_abs * 1.1, row_max_abs * 1.1) if row_max_abs > 0 else None

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois_all):
            ax     = axes[r_idx, c_idx]
            colour = ROI_COLOURS.get(roi, '#ffffff')

            tv, mean_curve, sem_curve = curves[(band, roi)]

            ax.set_xlim(t_min, t_max)
            if row_ylim[band] is not None:
                ax.set_ylim(*row_ylim[band])

            if tv is None:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', color='#555555', fontsize=11)
                continue

            ax.fill_between(tv,
                             mean_curve - sem_curve,
                             mean_curve + sem_curve,
                             color=colour, alpha=0.30)
            ax.plot(tv, mean_curve, color=colour, linewidth=LW_MEAN)

            # Reference line at z(or amplitude)=0
            ax.axhline(0, color=_FLAG_LINE, linewidth=1.1, alpha=0.6, zorder=2)

            # Draw event flags
            _draw_event_flags(ax, flags, ax.get_ylim())

            # Count valid subjects
            n_subj = sum(
                1 for r in all_results
                if r is not None and r.get(band, {}).get(roi) is not None
            )

            if r_idx == 0:
                roi_lbl = roi.capitalize() if roi != 'whole' else 'Whole brain'
                ax.set_title(f'{roi_lbl}  (n={n_subj})',
                             fontsize=FS_PANEL_TTL, fontweight='bold', pad=8)

            if r_idx == n_rows - 1:
                ax.set_xlabel('Time (s)', fontsize=FS_AXIS_LABEL,
                              fontweight='bold')

            if c_idx == 0:
                ylabel = 'Normalized activity'
                ax.set_ylabel(ylabel, fontsize=FS_AXIS_LABEL, fontweight='bold')

            if c_idx == 0:
                ax.annotate(BAND_LABELS.get(band, band),
                             xy=(-0.36, 0.5), xycoords='axes fraction',
                             fontsize=FS_ROW_LABEL, color=_FG,
                             ha='right', va='center',
                             rotation=90, fontweight='bold')

            # Tick spacing: 1 unit on the base grid (1 s on x, 1 z-score/
            # amplitude unit on y), PLUS an explicit x-tick at every epoch
            # transition (event flag time) so its exact time is readable
            # directly off the axis, not just from the floating flag label.
            base_xticks = np.arange(np.ceil(t_min), np.floor(t_max) + 1.0, 1.0)
            flag_times  = [f[0] for f in flags]
            xticks = sorted(set(np.round(np.concatenate([base_xticks, flag_times]), 3))) \
                if flag_times else base_xticks
            ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5, symmetric=True))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))

    title = 'Stim-locked Activity' if lockType == 'stim' else 'Response-locked Activity'
    fig.suptitle(f'{title}  |  {voxRes}',
                 color=_FG, fontsize=FS_SUPTITLE, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0.07, 0, 1, 1])

    os.makedirs(outdir, exist_ok=True)
    fpath = os.path.join(outdir, f'timeseries_{lockType}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot mean +/- SEM timeseries for all bands and ROIs.')
    parser.add_argument('--voxRes',    default='8mm')
    parser.add_argument('--lockTypes', nargs='+', default=['stim', 'resp'])
    parser.add_argument('--rois',      nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--bands',     nargs='+', default=['theta', 'alpha', 'beta'],
                        help='Bands to load AND plot (default theta alpha beta). '
                             'Restricting here skips their G04 loads entirely, so it '
                             'is a real speedup, not just a plotting filter. Pass '
                             'lowgamma/highgamma explicitly to include them.')
    parser.add_argument('--subjects',  nargs='+', type=int,
                        default=SUBJECT_LIST)
    parser.add_argument('--outdir',    default=None)
    parser.add_argument('--n_jobs',    type=int, default=None,
                        help='Parallel workers (subjects). Default: len(--subjects) '
                             '-- one worker per requested subject.')
    parser.add_argument('--no_baseline', action='store_true',
                        help='Disable per-subject baseline z-scoring (see BASELINE_WINDOWS). '
                             'Default: on -- each (subject, band, ROI) curve is z-scored '
                             'against its own fixation/pre-response baseline period before '
                             'cross-subject averaging.')
    args = parser.parse_args()

    if args.n_jobs is None:
        args.n_jobs = len(args.subjects)

    bids_root = get_bids_root()
    outdir    = args.outdir or os.path.join(
        bids_root, 'derivatives', 'glueDecoding', 'timeseries')

    # 'whole' is opt-in (pass --rois visual parietal frontal whole), not
    # forced on by default: including it forces a full whole-grid G03/G04
    # load per subject, whereas the default ROI set (visual/parietal/frontal)
    # can be served entirely from precompute_roi_splits.py's small per-ROI
    # caches -- see load_subject_timeseries's need_whole branch.
    rois_all = list(args.rois)

    n_jobs = max(1, args.n_jobs)
    print(f'plot_timeseries | voxRes={args.voxRes} | subjects={args.subjects} | '
          f'rois={rois_all} | bands={args.bands} | n_jobs={n_jobs}')

    atlas_masks = load_atlas_masks(args.voxRes, bids_root)

    for lockType in args.lockTypes:
        t_min, t_max = TIME_WINDOWS.get(lockType, (-1.0, 2.0))
        baseline_window = None if args.no_baseline else BASELINE_WINDOWS.get(lockType)
        print(f'\n=== lockType: {lockType}  window: [{t_min}, {t_max}] s  '
              f'baseline: {baseline_window} ===')

        all_results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=5)(
            delayed(load_subject_timeseries)(
                subjID, lockType, args.voxRes, bids_root,
                atlas_masks, rois_all, t_min, t_max, baseline_window,
                list(args.bands)
            )
            for subjID in args.subjects
        )

        plot_timeseries_figure(all_results, rois_all, lockType,
                                args.voxRes, outdir,
                                baselined=baseline_window is not None,
                                bands=list(args.bands))

    print('\nDone.')


if __name__ == '__main__':
    main()
