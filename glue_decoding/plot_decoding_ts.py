#!/usr/bin/env python3
"""
plot_decoding_ts.py

Aggregates per-subject decoding timeseries .npz files and produces two
publication-quality figures per condition (ampOnly / ampPhase), each a
bands x ROIs grid (rows = theta/alpha/beta/lowgamma, cols = visual/parietal/
frontal):

  1) Timeseries figure (make_timeseries_figure):
       Mean +/- SEM circular decoding error across subjects (lower = better).
       Dashed line: mean +/- SEM shuffle baseline.
       Horizontal line at 90 deg: theoretical chance for circular decoding.
       Event flag lines: Stim at 0.0 s (top), Delay Onset at 0.2 s (mid).
       Y-axis is data-driven per panel (not a fixed 0-180 clamp), always
       keeping the 90 deg chance line in view.

  2) Quartile figure (make_quartile_figure):
       Trials binned per-subject into 4 quartiles of REAL BEHAVIORAL
       PERFORMANCE (i_sacc_err, the initial-saccade/memory-report error --
       see get_subject_i_sacc_err/compute_epoch_quartiles), NOT by decoding
       error itself -- binning trials by an outcome and then showing the
       bins differ in that outcome would be circular and would look
       "significant" even for a decoder with zero real signal.
       Two epochs shown per panel: Stim (0.0-0.2 s) | Delay (0.2-1.7 s).
       Bar = mean +/- SEM decoding error across subjects per quartile.
       Scatter dots = individual-subject mean error per quartile.
       Dashed ticks = matched-trial shuffle baseline per quartile.
       Text = per-subject slope/correlation of decoding error vs.
       performance quartile, tested against 0 across subjects.
       Quartile colours: ROI colour shaded Q1 (best performance, darkest) ->
       Q4 (worst performance, lightest).

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
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import to_rgb

from align import load_behav, verify_alignment, attach_behav
from constants import get_bids_root, ROI_NAMES
from io_g03 import load_g03_unfiltered

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
    'highgamma': 'High gamma\n(80-150 Hz)',
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

# Minimum initial-saccade error to count as a valid behavioral trial (matches
# megScripts/plotDecodBehav.py's errThresh) -- excludes trials whose i_sacc_err
# is ~0, which tends to indicate missing/invalid saccade data rather than a
# genuinely perfect response.
I_SACC_ERR_THRESH = 0.001

# ── I/O helpers ───────────────────────────────────────────────────────────────

def cell_path(bids_root, subjID, band, roi, condition, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base    = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'decodingTS')
    return os.path.join(
        base,
        f'{subName}_task-mgs_decodingTS_{condition}_{band}_{roi}_{voxRes}.npz')


def get_subject_i_sacc_err(subjID, bids_root, voxRes, rois):
    """
    Loads this subject's initial-saccade error (i_sacc_err, real behavioral
    performance) and verifies it's positionally aligned with G04's stim-locked
    trial order via the same checksum megScripts/align.py already implements
    (verify_alignment). Returns None if the behavioral file is missing or
    alignment can't be verified -- callers should then fall back to skipping
    behavior-based analysis for this subject rather than risk a silent
    mis-join.

    Note: this attaches behavior at PLOT time using each cell's own saved
    'trial_idx' (the G04-row -> original-sourcedataCombined-row map), rather
    than baking i_sacc_err into decoding_ts_cell.py's .npz cache -- avoids
    having to regenerate the whole per-subject/band/roi cache just to add one
    field.
    """
    behav = load_behav(subjID, bids_root)
    if behav is None:
        return None
    try:
        g03_meta = load_g03_unfiltered(subjID, 'stim', voxRes, bids_root, roi=rois[0])
    except (FileNotFoundError, OSError):
        return None
    if not verify_alignment(g03_meta['trialinfo_col2'], behav['tarlocCode']):
        print(f'  WARNING: sub-{subjID:02d}: behavioral alignment check failed '
              f'-- skipping i_sacc_err for this subject.', flush=True)
        return None
    return behav


def load_all_subjects(subjects, bids_root, voxRes, bands, rois, conditions, outdir=None):
    """
    Returns list (one entry per subject) of dicts:
        data[(band, roi, condition)] = loaded npz dict or None

    Each loaded cell also gets an 'i_sacc_err' key attached (real behavioral
    performance, NOT decoding error -- see get_subject_i_sacc_err), row-aligned
    to that cell's own trial order via its saved 'trial_idx'. NaN where
    behavior is unavailable/unverified for that subject, so downstream code
    can just filter NaNs rather than branch on missing behavior.
    """
    all_data = []
    for subjID in subjects:
        d = {}
        behav = get_subject_i_sacc_err(subjID, bids_root, voxRes, rois)
        for band in bands:
            for roi in rois:
                for condition in conditions:
                    fp = cell_path(bids_root, subjID, band, roi,
                                    condition, voxRes, outdir)
                    if os.path.exists(fp):
                        cell = dict(np.load(fp, allow_pickle=True))
                        n_trials = cell['trial_idx'].shape[0]
                        if behav is not None:
                            cell['i_sacc_err'] = attach_behav(
                                cell['trial_idx'], behav)['i_sacc_err']
                        else:
                            cell['i_sacc_err'] = np.full(n_trials, np.nan)
                        d[(band, roi, condition)] = cell
        all_data.append(d if d else None)
    return all_data


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_timeseries(all_data, band, roi, condition):
    """
    Returns (tv, mean_err, sem_err, mean_shuf, sem_shuf) or Nones.

    Real error is now computed the way megScripts/plotDecodBehav.py computes
    "decoding error" (its top-row scatter is unweighted circmean with no
    SEM; this ports the fuller version its own quantile/bottom-row uses,
    which is the same operation plus proper across-subject stats):
      1. Per trial, per timepoint: SIGNED circular error
         ((pred - true + 180) % 360) - 180, in [-180, 180] -- NOT
         glue_decoding's original per-trial UNSIGNED circular_dist.
      2. Per subject, per timepoint: circular mean of that signed error
         across trials (scipy.stats.circmean), then abs(). This lets
         same-magnitude opposite-direction trial errors partially cancel
         before the abs() is taken, instead of every trial contributing its
         full unsigned distance regardless of direction. Under chance-level
         decoding this quantity is ~Uniform[0, 180] with mean 90 -- the same
         chance level as the unsigned-distance metric it replaces (a
         resultant vector with no real angular signal points in an
         essentially random direction, and abs() folds that uniformly onto
         [0, 180]).
      3. Arithmetic mean/SEM of that per-subject quantity across subjects
         (unchanged from before).

    Shuffle baseline is method-matched to the real-error statistic above:
    decoding_ts_cell.py now saves shuffle_signed_circmean (n_shuffle, T) --
    the signed circular mean across trials for each label-permutation --
    and here we take abs() then average over permutations, i.e. the same
    "signed circmean over trials, then abs" statistic evaluated under the
    permuted-label null. Falls back to the old per-trial unsigned
    circular_dist shuffle_errors (arithmetic trial mean) for any cached
    .npz predating this field -- not method-matched, but same ~90 deg
    chance level, so still a valid approximate reference.
    """
    subj_err_list = []
    shuffle_list  = []
    tv            = None
    warned_legacy_shuffle = False

    for d in all_data:
        if d is None:
            continue
        k = (band, roi, condition)
        if k not in d:
            continue
        npz = d[k]
        pred_angles = npz['pred_angles']            # (N, T) deg
        true_angles = npz['true_angles']             # (N,) deg
        signed_err  = ((pred_angles - true_angles[:, None] + 180) % 360) - 180
        subj_signed_mean = stats.circmean(signed_err, high=180, low=-180, axis=0)  # (T,)
        subj_err_list.append(np.abs(subj_signed_mean))

        if 'shuffle_signed_circmean' in npz:
            shuffle_list.append(np.abs(npz['shuffle_signed_circmean']).mean(axis=0))  # (T,)
        else:
            if not warned_legacy_shuffle:
                print(f'  NOTE: {band}/{roi}/{condition}: cached .npz predates '
                      f'shuffle_signed_circmean -- using legacy unsigned-distance '
                      f'shuffle baseline (not method-matched). Re-run with --force '
                      f'to regenerate.', flush=True)
                warned_legacy_shuffle = True
            shuffle_list.append(npz['shuffle_errors'].mean(axis=0))
        if tv is None:
            tv = npz['time_vector']

    if not subj_err_list:
        return None, None, None, None, None

    err  = np.stack(subj_err_list)   # (n_subj, T)
    shuf = np.stack(shuffle_list)
    n    = err.shape[0]

    return (tv,
            err.mean(0),  err.std(0) / np.sqrt(n),
            shuf.mean(0), shuf.std(0) / np.sqrt(n))


def compute_epoch_quartiles(all_data, band, roi, condition):
    """
    Per-subject quartile binning of trial-level circular errors BY REAL
    BEHAVIORAL PERFORMANCE (i_sacc_err, initial-saccade/memory-report error --
    see get_subject_i_sacc_err / align.py), not by the decoding error itself.

    Binning by the decoding error itself would be circular: sorting trials by
    an outcome and then reporting that the sorted bins differ in that outcome
    is guaranteed by construction, even for a decoder with zero real signal.
    Binning by an independent behavioral measure is what actually tests
    "does decoding error track how well the subject performed."

    Quartile membership is computed ONCE per subject (i_sacc_err doesn't
    depend on epoch), then the epoch-specific mean decoding error / shuffle
    error is computed within each behavioral quartile:
      - Bin trials into N_QUANTILES quartiles of i_sacc_err (NaN and
        near-zero trials excluded -- see I_SACC_ERR_THRESH).
      - For each epoch (stim / delay), record the mean decoding error of
        trials in each quartile, plus the mean shuffle-baseline error
        (npz['shuffle_errors']) over the SAME trials, so the shuffle level
        shown per quartile is a matched, not just an overall, control.
      - Fit a per-subject linear trend (decoding error ~ performance
        quartile rank) so we can test whether decoding error tracks
        behavior, rather than just whether it differs from the 90 deg
        chance level.
    Aggregate across subjects: mean +/- SEM per quartile, plus group-level
    (one-sample t-test vs 0) stats on the per-subject slopes/correlations.

    Returns:
        dict: epoch_name -> dict with keys
          q_means, q_sems       : (N_QUANTILES,) real error, across subjects
          q_subj                : list (len N_QUANTILES) of per-subject arrays,
                                   for scatter plotting
          shuf_means, shuf_sems : (N_QUANTILES,) matched-trial shuffle baseline
          slope_mean, slope_p   : mean per-subject slope (deg/quartile rank)
                                   and one-sample t-test p-value vs 0
          r_mean, r_p           : mean per-subject Pearson r (error vs
                                   quartile rank) and one-sample t-test
                                   p-value vs 0
          n_complete            : # subjects with all N_QUANTILES bins
                                   populated (basis for slope/r stats)
    """
    ep_collector      = {ep: [[] for _ in range(N_QUANTILES)] for ep in EPOCH_WINDOWS}
    ep_shuf_collector = {ep: [[] for _ in range(N_QUANTILES)] for ep in EPOCH_WINDOWS}
    ep_subj_rows      = {ep: [] for ep in EPOCH_WINDOWS}   # per-subject (N_QUANTILES,) rows, nan where empty

    for d in all_data:
        if d is None:
            continue
        k = (band, roi, condition)
        if k not in d:
            continue
        npz        = d[k]
        tv         = npz['time_vector']        # (T,)
        errors     = npz['errors']             # (N, T)
        shuffle    = npz['shuffle_errors']     # (N, T)
        i_sacc_err = npz['i_sacc_err']         # (N,) real behavioral performance, NaN if unavailable

        # Behavioral quartile assignment, computed once per subject (not per
        # epoch -- i_sacc_err is a single per-trial value, epoch-invariant).
        # Skip this subject/cell entirely if too few trials have usable
        # behavior to form N_QUANTILES bins.
        valid = ~np.isnan(i_sacc_err) & (i_sacc_err > I_SACC_ERR_THRESH)
        if valid.sum() < N_QUANTILES:
            continue

        valid_idx = np.where(valid)[0]
        perf      = i_sacc_err[valid_idx]
        # Lower bound is exclusive except for the first bin, so a trial
        # sitting exactly on an internal percentile boundary lands in
        # exactly one quartile.
        q_bounds  = np.percentile(perf, np.linspace(0, 100, N_QUANTILES + 1))
        q_of_trial = np.full(valid_idx.shape[0], -1, dtype=int)
        for q in range(N_QUANTILES):
            lo, hi = q_bounds[q], q_bounds[q + 1]
            if q == 0:
                in_bin = (perf >= lo) & (perf <= hi)
            else:
                in_bin = (perf > lo) & (perf <= hi)
            q_of_trial[in_bin] = q

        for ep_name, (t0, t1) in EPOCH_WINDOWS.items():
            mask = (tv >= t0) & (tv <= t1)
            if not mask.any():
                continue

            # Per-trial mean circular error in this epoch (N,)
            trial_err  = errors[:, mask].mean(axis=1)
            trial_shuf = shuffle[:, mask].mean(axis=1)

            subj_row = np.full(N_QUANTILES, np.nan)
            for q in range(N_QUANTILES):
                sel = valid_idx[q_of_trial == q]
                if sel.size:
                    q_real = float(trial_err[sel].mean())
                    ep_collector[ep_name][q].append(q_real)
                    ep_shuf_collector[ep_name][q].append(float(trial_shuf[sel].mean()))
                    subj_row[q] = q_real
            ep_subj_rows[ep_name].append(subj_row)

    x_ranks = np.arange(1, N_QUANTILES + 1)
    result  = {}
    for ep_name in EPOCH_WINDOWS:
        q_means, q_sems       = [], []
        shuf_means, shuf_sems = [], []
        q_subj                = []
        for q in range(N_QUANTILES):
            vals  = np.asarray(ep_collector[ep_name][q])
            svals = np.asarray(ep_shuf_collector[ep_name][q])
            q_subj.append(vals)
            if len(vals) >= 1:
                q_means.append(vals.mean())
                q_sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
            else:
                q_means.append(np.nan)
                q_sems.append(np.nan)
            if len(svals) >= 1:
                shuf_means.append(svals.mean())
                shuf_sems.append(svals.std() / np.sqrt(len(svals)) if len(svals) > 1 else 0.0)
            else:
                shuf_means.append(np.nan)
                shuf_sems.append(np.nan)

        # Per-subject linear trend of error vs. quartile rank. Restricted to
        # subjects with all N_QUANTILES bins populated -- a slope fit through
        # 2-3 points per subject is too noisy to be a meaningful trend.
        rows = np.array(ep_subj_rows[ep_name]) if ep_subj_rows[ep_name] else np.empty((0, N_QUANTILES))
        complete = rows[~np.isnan(rows).any(axis=1)] if rows.size else rows

        slopes, rs = [], []
        for row in complete:
            lr = stats.linregress(x_ranks, row)
            slopes.append(lr.slope)
            rs.append(lr.rvalue)
        slopes, rs = np.array(slopes), np.array(rs)

        if len(slopes) > 1:
            slope_mean, slope_p = float(slopes.mean()), float(stats.ttest_1samp(slopes, 0).pvalue)
            r_mean, r_p         = float(rs.mean()),     float(stats.ttest_1samp(rs, 0).pvalue)
        elif len(slopes) == 1:
            slope_mean, slope_p = float(slopes[0]), np.nan
            r_mean, r_p         = float(rs[0]), np.nan
        else:
            slope_mean = slope_p = r_mean = r_p = np.nan

        result[ep_name] = dict(
            q_means=np.array(q_means), q_sems=np.array(q_sems), q_subj=q_subj,
            shuf_means=np.array(shuf_means), shuf_sems=np.array(shuf_sems),
            slope_mean=slope_mean, slope_p=slope_p,
            r_mean=r_mean, r_p=r_p, n_complete=len(complete),
        )

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
                           roi, is_bottom_row, is_left_col, show_title, title_str,
                           show_flag_labels=True):
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

    # ylim: no longer a hard 0-180 clamp -- that flattened real effects
    # against the full theoretical range and made everything look like it
    # hovers at chance. Instead scale to the actual data (err/shuf +/- SEM),
    # but always keep the 90 deg chance line in view so the effect size is
    # still interpretable relative to chance, and clip to the physical
    # [0, 180] range.
    lo = min(np.nanmin(mean_err - sem_err), np.nanmin(mean_shuf - sem_shuf), 90)
    hi = max(np.nanmax(mean_err + sem_err), np.nanmax(mean_shuf + sem_shuf), 90)
    pad = max(5.0, (hi - lo) * 0.15)
    lo, hi = max(0, lo - pad), min(180, hi + pad)

    # xlim/ylim must be set BEFORE reading ax.get_ylim() for flag placement --
    # otherwise the flags are positioned against matplotlib's autoscaled
    # limits, not the actual range the panel ends up with.
    ax.set_xlim(t0, t1)
    ax.set_ylim(lo, hi)

    # Event flags -- the vertical line is drawn on every row for alignment,
    # but the text label only on the top row (show_flag_labels), since
    # repeating it on every band row was a big chunk of the clutter in a
    # multi-band grid.
    y_lo, y_hi = ax.get_ylim()
    for (t_flag, label, y_frac) in EVENT_FLAGS:
        ax.axvline(t_flag, color=_FLAG_LINE, lw=0.8, ls='--', zorder=4)
        if show_flag_labels:
            ax.text(t_flag, y_lo + y_frac * (y_hi - y_lo), label,
                    rotation=90, va='top', ha='right',
                    fontsize=6.5, color=_FLAG_TXT, zorder=5)

    # Each panel now has its own data-driven range (no longer a shared
    # 0-180 scale), so tick labels are shown on every column, not just the
    # left one -- hiding them would hide real information now.
    if is_left_col:
        ax.set_ylabel('Circular error (deg)', fontsize=8, color=_FG)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    if not is_bottom_row:
        ax.set_xticklabels([])
    # 'Time (s)' axis title is drawn once for the whole figure
    # (make_timeseries_figure) rather than per-panel, since it was
    # redundant across ROI columns.

    ax.grid(True, color=_GRID, lw=0.4, zorder=0)
    _style_ax(ax)

    if show_title:
        ax.set_title(title_str, color=_FG, fontsize=10, fontweight='bold', pad=4)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))


def _quartile_panel_values(epoch_quartiles):
    """
    Collect every value plot_bar_panel would actually draw for one
    (band, roi, condition) cell -- bar heights +/- SEM, subject scatter
    dots, and shuffle ticks -- so a shared y-range can be computed across
    cells/conditions before any panel is drawn.
    """
    vals = []
    for ep_name in EPOCH_WINDOWS:
        ep_stats = epoch_quartiles[ep_name]
        q_means, q_sems = ep_stats['q_means'], ep_stats['q_sems']
        for v, s in zip(q_means, q_sems):
            if np.isnan(v):
                continue
            s = 0.0 if np.isnan(s) else s
            vals.append(v - s)
            vals.append(v + s)
        for subj_vals in ep_stats['q_subj']:
            vals.extend(float(x) for x in subj_vals)
        vals.extend(v for v in ep_stats['shuf_means'] if not np.isnan(v))
    return vals


def compute_shared_bar_ylims(all_data, bands, rois, conditions):
    """
    One y-axis range per (band, roi), shared across BOTH conditions'
    quartile figures -- so the same panel position (e.g. beta/visual) uses
    an identical y-scale whether it's drawn on the ampOnly or the ampPhase
    figure, and the two are directly comparable by eye.

    Narrow, data-driven range around the actual bars/scatter/shuffle
    values (min..max +/- a small pad), rather than the previous per-panel
    0-to-(1.55x max) clamp, which forced every bar to share a zero
    baseline and made the (usually small) real differences between
    quartiles hard to see.
    """
    ylims = {}
    for band in bands:
        for roi in rois:
            vals = []
            for condition in conditions:
                epoch_quartiles = compute_epoch_quartiles(all_data, band, roi, condition)
                if epoch_quartiles:
                    vals.extend(_quartile_panel_values(epoch_quartiles))
            if vals:
                lo, hi = min(vals), max(vals)
                pad = max(1.5, (hi - lo) * 0.15)
                ylims[(band, roi)] = (max(0, lo - pad), min(180, hi + pad))
            else:
                ylims[(band, roi)] = (0, 180)
    return ylims


def plot_bar_panel(ax, epoch_quartiles, roi, ylim):
    """
    Draw quartile bar strip below timeseries.
    Two groups (stim | delay), N_QUANTILES bars each.
    Bars: mean circular error per quartile across subjects.
    Dots: individual subject values.
    Dashed ticks: matched-trial shuffle baseline per quartile.
    Text: per-subject linear trend of error vs. quartile rank (slope +
    Pearson r), tested against 0 -- this asks whether error tracks
    performance, which is more informative than a chance-level test.

    ylim: (lo, hi) shared across conditions for this (band, roi) -- see
    compute_shared_bar_ylims.
    """
    q_cols  = quartile_colours(roi, N_QUANTILES)
    bar_w   = 0.6
    gap     = 1.5    # gap between stim and delay groups
    eps     = 0.12   # jitter for scatter

    group_offsets = {'stim': 0, 'delay': N_QUANTILES + gap}
    rng = np.random.default_rng(42)

    # Bars are grounded at ylim's floor, not 0 -- with a narrow, non-zero-based
    # range (see compute_shared_bar_ylims) a 0-baseline bar would mostly sit
    # off-screen below the visible panel.
    bar_bottom = ylim[0]

    for ep_name in EPOCH_WINDOWS:
        ep_stats   = epoch_quartiles[ep_name]
        q_means    = ep_stats['q_means']
        q_sems     = ep_stats['q_sems']
        q_subj     = ep_stats['q_subj']
        shuf_means = ep_stats['shuf_means']
        x0 = group_offsets[ep_name]
        for q in range(N_QUANTILES):
            x = x0 + q
            v = q_means[q]
            s = q_sems[q]
            c = q_cols[q]
            if not np.isnan(v):
                ax.bar(x, v - bar_bottom, bottom=bar_bottom, width=bar_w,
                        color=c, alpha=0.85, zorder=3,
                        linewidth=0, align='center')
                ax.errorbar(x, v, yerr=s, fmt='none', ecolor=_FG,
                             elinewidth=1.0, capsize=2, zorder=4)
                # Subject scatter
                subj_vals = q_subj[q]
                if len(subj_vals):
                    jitter = rng.uniform(-eps, eps, len(subj_vals))
                    ax.scatter(x + jitter, subj_vals,
                                s=6, color='white', alpha=0.55,
                                edgecolors='none', zorder=5)

            # Matched-trial shuffle baseline for this quartile
            sv = shuf_means[q]
            if not np.isnan(sv):
                ax.hlines(sv, x - bar_w / 2, x + bar_w / 2,
                           color=_FG, lw=1.1, ls='--', alpha=0.7, zorder=6)

        # Trend annotation: per-subject slope (deg/quartile rank) + Pearson r,
        # group-tested against 0.
        slope_mean, slope_p = ep_stats['slope_mean'], ep_stats['slope_p']
        r_mean, r_p         = ep_stats['r_mean'], ep_stats['r_p']
        if not np.isnan(slope_mean):
            sig  = '*' if (not np.isnan(slope_p) and slope_p < 0.05) else ''
            rsig = '*' if (not np.isnan(r_p) and r_p < 0.05) else ''
            txt = f'β={slope_mean:+.1f}°/Q{sig}\nr={r_mean:+.2f}{rsig}'
            ax.text(x0 + (N_QUANTILES - 1) / 2, 0.97, txt,
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='top', fontsize=5.8, color=_FG,
                    linespacing=1.3, zorder=7)

    # Group labels
    for ep_name, label in EPOCH_LABELS.items():
        x0 = group_offsets[ep_name]
        ax.text(x0 + (N_QUANTILES - 1) / 2, -0.05,
                label, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=6.5, color=_FG)

    ax.set_xlim(-0.6, max(group_offsets.values()) + N_QUANTILES - 0.4)
    # Shared across both conditions' figures for this (band, roi) -- see
    # compute_shared_bar_ylims -- so the same panel position is directly
    # comparable between the ampOnly and ampPhase quartile figures.
    ax.set_ylim(*ylim)
    ax.set_xticks([])

    # No y-label here: it shares units/scale with the timeseries panel
    # directly above it, which already labels the axis for the left column.
    # A second label on the much-shorter bar panel had no room and
    # collided with it.
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, integer=True))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.grid(axis='y', color=_GRID, lw=0.4, zorder=0)
    _style_ax(ax)


def _band_row_grid(n_bands, n_rois, row_h_in, fig_w_per_roi=4.2,
                    hspace=0.45, title_margin_in=0.65, bottom_margin_in=0.15):
    """
    Shared figure/GridSpec setup for a flat bands x rois grid: fixed
    inch-based top/bottom margins (independent of n_bands) so the suptitle
    doesn't collide with the top row for small n_bands, plus a generous
    hspace so band rows read as distinct groups.
    """
    fig_w = fig_w_per_roi * n_rois
    fig_h = row_h_in * n_bands + title_margin_in + bottom_margin_in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(
        n_bands, n_rois,
        figure = fig,
        hspace = hspace,
        wspace = 0.30,
        left   = 0.11,
        right  = 0.97,
        top    = 1 - title_margin_in / fig_h,
        bottom = bottom_margin_in / fig_h,
    )
    return fig, gs


def make_timeseries_figure(all_data, bands, rois, condition, voxRes, outdir_fig):
    """Build and save the timeseries figure (mean +/- SEM error/shuffle vs. time)."""

    n_bands = len(bands)
    n_rois  = len(rois)
    fig, gs = _band_row_grid(n_bands, n_rois, row_h_in=1.3,
                              bottom_margin_in=0.40)

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois):
            ax_ts = fig.add_subplot(gs[r_idx, c_idx])

            tv, mean_err, sem_err, mean_shuf, sem_shuf = aggregate_timeseries(
                all_data, band, roi, condition)

            is_left_col   = (c_idx == 0)
            is_bottom_row = (r_idx == n_bands - 1)
            show_title    = (r_idx == 0)
            title_str     = roi.capitalize()

            if tv is not None:
                plot_timeseries_panel(
                    ax_ts, tv, mean_err, sem_err, mean_shuf, sem_shuf,
                    roi, is_bottom_row, is_left_col, show_title, title_str,
                    show_flag_labels=show_title)
            else:
                ax_ts.text(0.5, 0.5, 'No data', ha='center', va='center',
                            transform=ax_ts.transAxes, color=_FG, fontsize=9)
                _style_ax(ax_ts)

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
    fig.text(0.5, 0.005, 'Time (s)', ha='center', va='bottom',
              color=_FG, fontsize=9)

    os.makedirs(outdir_fig, exist_ok=True)
    fpath = os.path.join(outdir_fig, f'decoding_ts_{condition}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


def make_quartile_figure(all_data, bands, rois, condition, voxRes, outdir_fig, ylims):
    """Build and save the quartile bar figure (error vs. performance quartile),
    as its own figure separate from the timeseries -- previously this was
    squeezed into a 20%-height strip under each timeseries panel, which left
    no room for the shuffle ticks / trend annotation without crowding.

    ylims: dict (band, roi) -> (lo, hi), shared across conditions -- see
    compute_shared_bar_ylims -- so this figure and the other condition's
    figure use identical y-scales panel-for-panel."""

    n_bands = len(bands)
    n_rois  = len(rois)
    fig, gs = _band_row_grid(n_bands, n_rois, row_h_in=1.5,
                              bottom_margin_in=0.15)

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois):
            ax_bar = fig.add_subplot(gs[r_idx, c_idx])

            epoch_quartiles = compute_epoch_quartiles(
                all_data, band, roi, condition)

            show_title = (r_idx == 0)

            if epoch_quartiles:
                plot_bar_panel(ax_bar, epoch_quartiles, roi, ylims[(band, roi)])
            else:
                ax_bar.text(0.5, 0.5, 'No data', ha='center', va='center',
                             transform=ax_bar.transAxes, color=_FG, fontsize=9)
                _style_ax(ax_bar)

            if show_title:
                ax_bar.set_title(roi.capitalize(), color=_FG, fontsize=10,
                                   fontweight='bold', pad=4)

            if c_idx == 0:
                ax_bar.annotate(BAND_LABELS.get(band, band),
                                 xy=(-0.38, 0.5), xycoords='axes fraction',
                                 fontsize=9, color=_FG,
                                 ha='right', va='center',
                                 rotation=90, fontweight='bold')

    cond_label = 'Amplitude only' if condition == 'ampOnly' \
                 else 'Amplitude + Phase'
    fig.suptitle(f'Decoding by Performance Quartile  |  {cond_label}  |  {voxRes}',
                  color=_FG, fontsize=14, fontweight='bold', y=0.97)

    os.makedirs(outdir_fig, exist_ok=True)
    fpath = os.path.join(outdir_fig, f'decoding_quartiles_{condition}_{voxRes}.png')
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
                        default=['theta', 'alpha', 'beta', 'lowgamma', 'highgamma'])
    parser.add_argument('--rois',        nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions',  nargs='+', default=['ampOnly', 'ampPhase'])
    parser.add_argument('--outdir',      default=None,
                        help='Directory containing the per-subject .npz files.')
    parser.add_argument('--figdir',      default=None,
                        help='Directory to save figures (default: same as outdir '
                             'or BIDS derivatives/glueDecoding/decodingTS).')
    args = parser.parse_args()

    bids_root = get_bids_root()
    figdir    = args.figdir or (args.outdir if args.outdir else
                                 os.path.join(bids_root, 'derivatives',
                                              'glueDecoding', 'decodingTS'))

    print(f'Loading data | {args.voxRes} | subjects={args.subjects} | '
          f'bands={args.bands} | rois={args.rois} | conditions={args.conditions}')

    all_data = load_all_subjects(
        args.subjects, bids_root, args.voxRes,
        args.bands, args.rois, args.conditions, args.outdir)

    n_loaded = sum(1 for d in all_data if d)
    print(f'Loaded data for {n_loaded}/{len(args.subjects)} subjects.')

    # Computed once across ALL conditions so the quartile bar figures share
    # an identical y-scale panel-for-panel (see compute_shared_bar_ylims).
    bar_ylims = compute_shared_bar_ylims(all_data, args.bands, args.rois, args.conditions)

    for condition in args.conditions:
        print(f'\n-- Plotting condition: {condition} --')
        make_timeseries_figure(all_data, args.bands, args.rois, condition,
                                 args.voxRes, figdir)
        make_quartile_figure(all_data, args.bands, args.rois, condition,
                               args.voxRes, figdir, bar_ylims)

    print('\nAll figures saved.')


if __name__ == '__main__':
    main()
