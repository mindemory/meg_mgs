#!/usr/bin/env python3
"""
plot_linear_decoding_categories.py

Aggregates linear_decoding_categories_cell.py's per-subject .npz files and
plots LOO (or k-fold) SVM one-vs-rest classification accuracy over time,
bands x rois grid, one figure per (condition, scheme) -- mirrors
plot_decoding_ts.py / plot_representational_distance_ts.py's layout/styling
so all three are visually and temporally comparable.

Two figures per (condition, scheme):
  lindecode_cat_{condition}_scheme{scheme}_{voxRes}.png
      Raw accuracy: mean +/- SEM across subjects, theoretical chance line
      (1/n_categories), optional empirical-null band (only if cells were
      run with --n_shuffle > 0 -- same "typical single-subject null"
      caveat as plot_representational_distance_ts.py, not a group-level
      null). Most directly interpretable ("62% accuracy").
  lindecode_cat_norm_{condition}_scheme{scheme}_{voxRes}.png
      Normalized: (accuracy - chance) / (1 - chance), bounded [0, 1],
      comparable across schemes with different chance floors (0.5 for
      scheme 2 vs. 0.1 for scheme 10) the same way accuracy alone isn't --
      exact linear rescaling of the raw accuracy line/SEM, so it doesn't
      need re-deriving from the per-subject data.

Both figures share the same significance overlay: a one-sample CLUSTER-BASED
PERMUTATION test (Maris & Oostenveld, 2007; sign-flipping) of accuracy(t)
against chance_level across subjects -- replaces an earlier per-timepoint
Wilcoxon+FDR approach (see chat history): FDR treats every timepoint as an
independent test, throwing away the temporal correlation between adjacent
timepoints; cluster permutation tests whole contiguous runs against a null
built from the same kind of statistic, which is both more powerful (a real
effect should show up as a sustained run, not isolated blips) and still
properly controls family-wise error (via the max-cluster-statistic null).
Shown as a dot row under each panel. This REPLACES the empirical shuffle
band as the group-level significance readout; the shuffle band (when
present) is kept only as an additional single-subject-null visual
reference, same caveat as before.

Usage:
    python plot_linear_decoding_categories.py [--voxRes 8mm]
                                               [--bands theta alpha beta lowgamma highgamma]
                                               [--rois visual parietal frontal]
                                               [--conditions ampOnly]
                                               [--schemes 2 4 6 10]
                                               [--subjects 1 2 ...]
                                               [--n_perm 1000] [--cluster_alpha 0.05] [--alpha 0.05]
                                               [--outdir <path>] [--figdir <path>]
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

from constants import SUBJECT_LIST, ROI_NAMES, CATEGORY_SCHEMES, get_bids_root
from linear_decoding_categories_cell import output_path

SCHEME_LABELS = {s: f"{CATEGORY_SCHEMES[s]['name']} ({s} categories)" for s in CATEGORY_SCHEMES}

# ── Design constants (mirrors plot_decoding_ts.py / plot_representational_distance_ts.py) ──

_BG        = '#000000'
_FG        = '#e0e0e0'
_GRID      = '#1c1c1c'
_FLAG_LINE = '#777777'
_FLAG_TXT  = '#cccccc'
_CHANCE    = '#444444'
_SIG       = '#ffffff'     # significance marker dots

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

EVENT_FLAGS = [
    (0.0,  'Stim',        0.93),
    (0.2,  'Delay Onset', 0.55),
]

# ── I/O ─────────────────────────────────────────────────────────────────────

def load_all_subjects(subjects, bids_root, voxRes, bands, rois, conditions, schemes, outdir=None):
    """Returns list (one entry per subject) of dicts: data[(band, roi, condition, scheme)] = npz dict or None."""
    all_data = []
    for subjID in subjects:
        d = {}
        for band in bands:
            for roi in rois:
                for condition in conditions:
                    for scheme in schemes:
                        fp = output_path(bids_root, subjID, band, roi, condition, voxRes, scheme, outdir)
                        if os.path.exists(fp):
                            d[(band, roi, condition, scheme)] = dict(np.load(fp, allow_pickle=True))
        all_data.append(d if d else None)
    return all_data


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_lindecode(all_data, band, roi, condition, scheme):
    """
    Returns (tv, acc_matrix, mean_acc, sem_acc, chance_level, mean_shuf,
    mean_shuf_std, has_shuffle) or (None,)*8.

    acc_matrix: (n_subj, T) raw per-subject accuracy(t), returned (not just
    its mean/SEM) so compute_significance_vs_chance can run the actual
    across-subject test on it.
    mean_acc/sem_acc: arithmetic mean/SEM across subjects.
    chance_level: 1/n_categories (identical across subjects for the same
    scheme, just read from the first available cell).
    mean_shuf/mean_shuf_std: same "typical single-subject null" caveat as
    plot_representational_distance_ts.py's aggregate_repdist -- only
    meaningful if cells were run with --n_shuffle > 0 (all-NaN otherwise,
    has_shuffle=False); NOT used for group-level significance any more (see
    module docstring) -- kept only as an optional visual reference.
    """
    acc_list, shuf_mean_list, shuf_std_list = [], [], []
    tv = None
    chance_level = None
    for d in all_data:
        if d is None:
            continue
        k = (band, roi, condition, scheme)
        if k not in d:
            continue
        npz = d[k]
        acc_list.append(npz['accuracy'])
        shuf_mean_list.append(npz['shuffle_acc_mean'])
        shuf_std_list.append(npz['shuffle_acc_std'])
        if tv is None:
            tv = npz['eval_time_vector']
            chance_level = float(npz['chance_level'][0])

    if not acc_list:
        return None, None, None, None, None, None, None, False

    acc  = np.stack(acc_list)          # (n_subj, T)
    smu  = np.stack(shuf_mean_list)    # (n_subj, T)
    ssd  = np.stack(shuf_std_list)     # (n_subj, T)
    n    = acc.shape[0]

    has_shuffle = not np.all(np.isnan(smu))
    mean_shuf     = np.nanmean(smu, axis=0) if has_shuffle else None
    mean_shuf_std = np.nanmean(ssd, axis=0) if has_shuffle else None

    return (tv, acc, acc.mean(0), acc.std(0) / np.sqrt(n), chance_level,
            mean_shuf, mean_shuf_std, has_shuffle)


def _cluster_tstat(diff):
    """One-sample t-stat per timepoint. diff: (n_subj, T) -> (T,)."""
    n_subj = diff.shape[0]
    m = diff.mean(axis=0)
    s = diff.std(axis=0, ddof=1)
    s = np.where(s < 1e-12, 1e-12, s)
    return m / (s / np.sqrt(n_subj))


def _find_clusters(tvals, t_crit):
    """
    Contiguous runs where |t| > t_crit. Returns list of (start, end,
    cluster_stat) with end EXCLUSIVE, cluster_stat = sum(|t|) within the run.
    """
    sig = np.abs(tvals) > t_crit
    clusters = []
    start = None
    for i, s in enumerate(sig):
        if s and start is None:
            start = i
        elif not s and start is not None:
            clusters.append((start, i, float(np.sum(np.abs(tvals[start:i])))))
            start = None
    if start is not None:
        clusters.append((start, len(tvals), float(np.sum(np.abs(tvals[start:])))))
    return clusters


def cluster_permutation_test(acc_matrix, chance_level, n_perm=1000,
                              cluster_alpha=0.05, alpha=0.05, seed=0):
    """
    One-sample cluster-based permutation test (Maris & Oostenveld, 2007) of
    accuracy(t) against chance_level across subjects, via sign-flipping --
    replaces the earlier per-timepoint Wilcoxon+FDR approach (see chat
    history): FDR treats each timepoint as an independent test, discarding
    the temporal correlation between adjacent timepoints; cluster
    permutation instead tests whole contiguous runs of a candidate effect
    against a null built from the SAME kind of contiguous-run statistic, so
    it's both more powerful (exploits temporal structure) and still
    controls family-wise error properly (via the max-cluster-statistic null).

    acc_matrix: (n_subj, T) raw per-subject accuracy(t).
    n_perm: sign-flip permutations for the null distribution of the max
    cluster statistic.
    cluster_alpha: two-sided per-timepoint threshold (as a t-distribution
    critical value, df=n_subj-1) used only to FORM candidate clusters --
    not itself a significance claim.
    alpha: cluster-level significance threshold on each observed cluster's
    permutation p-value.

    Returns a (T,) bool mask -- True for timepoints belonging to a cluster
    with p < alpha. Returns all-False if n_subj < 2 or no candidate cluster
    forms in the observed data.
    """
    n_subj, n_times = acc_matrix.shape
    if n_subj < 2:
        return np.zeros(n_times, dtype=bool)

    diff = acc_matrix - chance_level   # (n_subj, T)
    t_crit = stats.t.ppf(1 - cluster_alpha / 2, df=n_subj - 1)

    t_obs = _cluster_tstat(diff)
    obs_clusters = _find_clusters(t_obs, t_crit)
    if not obs_clusters:
        return np.zeros(n_times, dtype=bool)

    rng = np.random.default_rng(seed)
    max_stat_null = np.zeros(n_perm)
    for p in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n_subj)
        t_perm = _cluster_tstat(diff * signs[:, None])
        perm_clusters = _find_clusters(t_perm, t_crit)
        max_stat_null[p] = max((c[2] for c in perm_clusters), default=0.0)

    sig_mask = np.zeros(n_times, dtype=bool)
    for (start, end, stat) in obs_clusters:
        p_val = (np.sum(max_stat_null >= stat) + 1) / (n_perm + 1)
        if p_val < alpha:
            sig_mask[start:end] = True
    return sig_mask


# ── Plotting ──────────────────────────────────────────────────────────────────

def _style_ax(ax, spine_col='#333333'):
    ax.set_facecolor(_BG)
    for sp in ax.spines.values():
        sp.set_color(spine_col)
    ax.tick_params(colors=_FG, labelsize=8)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)


def plot_lindecode_panel(ax, tv, mean_acc, sem_acc, chance_level, mean_shuf, mean_shuf_std,
                          sig_mask, y_label, roi, is_bottom_row, is_left_col, show_title,
                          title_str, show_flag_labels=True):
    """
    Plots whatever (mean_acc, sem_acc, chance_level[, mean_shuf, mean_shuf_std])
    it's given -- caller decides raw accuracy vs. normalized
    (accuracy-chance)/(1-chance) by passing the already-transformed values
    (see make_timeseries_figure) and the matching y_label/chance_level (0
    for the normalized variant, 1/n_categories for raw).
    """
    col = ROI_COLOURS[roi]
    t0, t1 = float(tv[0]), float(tv[-1])

    ax.axhline(chance_level, color=_CHANCE, lw=0.8, ls=':', zorder=1)

    if mean_shuf is not None:
        ax.fill_between(tv, mean_shuf - mean_shuf_std, mean_shuf + mean_shuf_std,
                         alpha=0.15, color=col, zorder=2)
        ax.plot(tv, mean_shuf, color=col, lw=0.9, ls='--', alpha=0.55, zorder=2)

    ax.fill_between(tv, mean_acc - sem_acc, mean_acc + sem_acc,
                     alpha=0.30, color=col, zorder=3)
    ax.plot(tv, mean_acc, color=col, lw=1.5, zorder=3)

    lo_candidates = [np.nanmin(mean_acc - sem_acc), chance_level]
    hi_candidates = [np.nanmax(mean_acc + sem_acc), chance_level]
    if mean_shuf is not None:
        lo_candidates.append(np.nanmin(mean_shuf - mean_shuf_std))
        hi_candidates.append(np.nanmax(mean_shuf + mean_shuf_std))
    lo, hi = min(lo_candidates), max(hi_candidates)
    pad = max(1e-6, (hi - lo) * 0.15)
    lo, hi = lo - pad, hi + pad

    # Significance dots (Wilcoxon vs chance, FDR-corrected -- see
    # compute_significance_vs_chance) along a fixed row near the panel
    # bottom, same convention as plot_representational_distance_ts.py.
    if sig_mask is not None and sig_mask.any():
        y_sig = lo + 0.06 * (hi - lo)
        ax.plot(tv[sig_mask], np.full(sig_mask.sum(), y_sig),
                '.', color=_SIG, ms=3.0, zorder=6)

    ax.set_xlim(t0, t1)
    ax.set_ylim(lo, hi)

    y_lo, y_hi = ax.get_ylim()
    for (t_flag, label, y_frac) in EVENT_FLAGS:
        ax.axvline(t_flag, color=_FLAG_LINE, lw=0.8, ls='--', zorder=4)
        if show_flag_labels:
            ax.text(t_flag, y_lo + y_frac * (y_hi - y_lo), label,
                    rotation=90, va='top', ha='right',
                    fontsize=6.5, color=_FLAG_TXT, zorder=5)

    if is_left_col:
        ax.set_ylabel(y_label, fontsize=8, color=_FG)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    if not is_bottom_row:
        ax.set_xticklabels([])

    ax.grid(True, color=_GRID, lw=0.4, zorder=0)
    _style_ax(ax)

    if show_title:
        ax.set_title(title_str, color=_FG, fontsize=10, fontweight='bold', pad=4)

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))


def _band_row_grid(n_bands, n_rois, row_h_in, fig_w_per_roi=4.2,
                    hspace=0.45, title_margin_in=0.65, bottom_margin_in=0.15):
    fig_w = fig_w_per_roi * n_rois
    fig_h = row_h_in * n_bands + title_margin_in + bottom_margin_in
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=_BG)
    gs = gridspec.GridSpec(
        n_bands, n_rois, figure=fig, hspace=hspace, wspace=0.30,
        left=0.13, right=0.97,
        top=1 - title_margin_in / fig_h, bottom=bottom_margin_in / fig_h,
    )
    return fig, gs


def make_timeseries_figure(all_data, bands, rois, condition, scheme, voxRes, outdir_fig,
                            normalize=False, n_perm=1000, cluster_alpha=0.05, alpha=0.05):
    """
    normalize=False (default): raw accuracy, chance line at 1/n_categories.
    normalize=True: (accuracy-chance)/(1-chance), bounded [0,1], chance line
    at 0 -- an exact linear rescaling of the raw mean/SEM (see module
    docstring), for cross-scheme comparability. Same significance mask
    either way (a positive linear rescaling doesn't change the cluster
    test's sign/magnitude -- t-stats are scale-invariant under a positive
    linear transform).
    """
    n_bands, n_rois = len(bands), len(rois)
    fig, gs = _band_row_grid(n_bands, n_rois, row_h_in=1.3, bottom_margin_in=0.40)

    for r_idx, band in enumerate(bands):
        for c_idx, roi in enumerate(rois):
            ax_ts = fig.add_subplot(gs[r_idx, c_idx])

            (tv, acc_matrix, mean_acc, sem_acc, chance_level,
             mean_shuf, mean_shuf_std, has_shuffle) = aggregate_lindecode(
                all_data, band, roi, condition, scheme)

            is_left_col   = (c_idx == 0)
            is_bottom_row = (r_idx == n_bands - 1)
            show_title    = (r_idx == 0)
            title_str     = roi.capitalize()

            if tv is not None:
                sig_mask = cluster_permutation_test(
                    acc_matrix, chance_level, n_perm=n_perm,
                    cluster_alpha=cluster_alpha, alpha=alpha)

                if normalize:
                    scale = 1.0 - chance_level
                    plot_mean, plot_sem   = (mean_acc - chance_level) / scale, sem_acc / scale
                    plot_chance = 0.0
                    plot_shuf, plot_shuf_std = (
                        ((mean_shuf - chance_level) / scale, mean_shuf_std / scale)
                        if mean_shuf is not None else (None, None))
                    y_label = 'Normalized accuracy\n(acc-chance)/(1-chance)'
                else:
                    plot_mean, plot_sem = mean_acc, sem_acc
                    plot_chance = chance_level
                    plot_shuf, plot_shuf_std = mean_shuf, mean_shuf_std
                    y_label = 'LOO accuracy'

                plot_lindecode_panel(
                    ax_ts, tv, plot_mean, plot_sem, plot_chance, plot_shuf, plot_shuf_std,
                    sig_mask, y_label, roi, is_bottom_row, is_left_col, show_title, title_str,
                    show_flag_labels=show_title)
            else:
                ax_ts.text(0.5, 0.5, 'No data', ha='center', va='center',
                           transform=ax_ts.transAxes, color=_FG, fontsize=9)
                _style_ax(ax_ts)

            if c_idx == 0:
                ax_ts.annotate(BAND_LABELS.get(band, band),
                               xy=(-0.42, 0.5), xycoords='axes fraction',
                               fontsize=9, color=_FG, ha='right', va='center',
                               rotation=90, fontweight='bold')

    cond_label   = 'Amplitude only' if condition == 'ampOnly' else 'Amplitude + Phase'
    scheme_label = SCHEME_LABELS.get(scheme, str(scheme))
    metric_label = 'Normalized Accuracy' if normalize else 'Accuracy'
    fig.suptitle(f'Linear (Ridge, OvR, LOO) Decoding {metric_label}  |  {cond_label}  |  '
                 f'{scheme_label}  |  {voxRes}  |  dots = cluster permutation vs chance, p<{alpha}',
                 color=_FG, fontsize=13, fontweight='bold', y=0.97)
    fig.text(0.5, 0.005, 'Time (s)', ha='center', va='bottom', color=_FG, fontsize=9)

    os.makedirs(outdir_fig, exist_ok=True)
    tag = 'lindecode_cat_norm' if normalize else 'lindecode_cat'
    fpath = os.path.join(outdir_fig, f'{tag}_{condition}_scheme{scheme}_{voxRes}.png')
    fig.savefig(fpath, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.close(fig)
    print(f'Saved: {fpath}')
    return fpath


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Aggregate and plot linear decoding-by-category accuracy timeseries across subjects.')
    parser.add_argument('--voxRes',     default='8mm')
    parser.add_argument('--subjects',   nargs='+', type=int, default=SUBJECT_LIST)
    parser.add_argument('--bands',      nargs='+',
                        default=['theta', 'alpha', 'beta', 'lowgamma', 'highgamma'])
    parser.add_argument('--rois',       nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions', nargs='+', default=['ampOnly'])
    parser.add_argument('--schemes',    nargs='+', type=int, default=sorted(CATEGORY_SCHEMES),
                        choices=sorted(CATEGORY_SCHEMES))
    parser.add_argument('--n_perm',       type=int, default=1000,
                        help='Sign-flip permutations for the cluster-permutation null (default 1000).')
    parser.add_argument('--cluster_alpha', type=float, default=0.05,
                        help='Per-timepoint threshold (t-distribution) used only to FORM '
                             'candidate clusters, not a significance claim itself (default 0.05).')
    parser.add_argument('--alpha',      type=float, default=0.05,
                        help='Cluster-level significance threshold on each observed cluster\'s '
                             'permutation p-value (default 0.05).')
    parser.add_argument('--outdir',     default=None,
                        help='Directory containing the per-subject .npz files.')
    parser.add_argument('--figdir',     default=None,
                        help='Directory to save figures (default: same as outdir '
                             'or BIDS derivatives/glueDecoding/linDecodeCat).')
    args = parser.parse_args()

    bids_root = get_bids_root()
    figdir = args.figdir or (args.outdir if args.outdir else
                              os.path.join(bids_root, 'derivatives', 'glueDecoding', 'linDecodeCat'))

    print(f'Loading data | {args.voxRes} | subjects={args.subjects} | '
          f'bands={args.bands} | rois={args.rois} | conditions={args.conditions} | '
          f'schemes={args.schemes}')

    all_data = load_all_subjects(
        args.subjects, bids_root, args.voxRes,
        args.bands, args.rois, args.conditions, args.schemes, args.outdir)

    n_loaded = sum(1 for d in all_data if d)
    print(f'Loaded data for {n_loaded}/{len(args.subjects)} subjects.')

    for condition in args.conditions:
        for scheme in args.schemes:
            print(f'\n-- Plotting condition: {condition} | scheme: {scheme} --')
            make_timeseries_figure(all_data, args.bands, args.rois, condition, scheme,
                                    args.voxRes, figdir, normalize=False,
                                    n_perm=args.n_perm, cluster_alpha=args.cluster_alpha, alpha=args.alpha)
            make_timeseries_figure(all_data, args.bands, args.rois, condition, scheme,
                                    args.voxRes, figdir, normalize=True,
                                    n_perm=args.n_perm, cluster_alpha=args.cluster_alpha, alpha=args.alpha)

    print('\nAll figures saved.')


if __name__ == '__main__':
    main()
