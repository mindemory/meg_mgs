#!/usr/bin/env python3
"""
visual_geometry_epochs_cell.py

EPOCH-based crossnobis RDMs for MDS geometry -- the four-epoch counterpart of
visual_geometry_ts_cell.py's sliding window. One RDM per
(band, condition, roi, epoch), plus a label-shuffle null.

EPOCHS (half-open [lo, hi), so adjacent epochs never share a sample):
    fixation    -1.0 .. 0.0 s
    stimulus     0.0 .. 0.2 s
    early_delay  0.2 .. 0.8 s
    late_delay   1.0 .. 1.6 s
Note the deliberate 0.8-1.0 s gap: early and late delay are separated rather
than contiguous, so any difference between them reflects two clearly distinct
periods rather than a boundary drawn through the middle of one.

CAVEAT worth carrying into interpretation -- the epochs are NOT equal length
(1.0 / 0.2 / 0.6 / 0.6 s). The stimulus epoch averages over ~5x fewer
timepoints than fixation, so its RDM is intrinsically noisier, and a lower
ring-ness there is partly a sample-size effect rather than purely a geometric
one. The per-epoch label-shuffle null absorbs this (each epoch is compared to
a null computed with that epoch's own trial/timepoint count), which is exactly
why the null is computed per cell rather than assumed.

BANDS / CONDITIONS: theta/alpha/beta carry saved phase and so run both
ampOnly and ampPhase; lowgamma/highgamma have no phase
(constants.AMP_PHASE_BANDS) and are skipped for ampPhase automatically, giving
amplitude-only cells for them.

Everything else matches visual_geometry_ts_cell.py and is reused from the
already-validated helpers: per-timepoint z-scoring before epoch averaging (its
mean-subtraction half IS the ERP removal), PCA conditioning to
min(n_trials-1, max_pca_dim, n_features), Ledoit-Wolf shrunk whitening applied
only when residual dof >= 2k, and crossnobis averaged over n_splits random
2-fold partitions. The PCA basis is held fixed across the shuffles (it is
unsupervised) while the noise covariance is recomputed per shuffle (it is
label-defined).

Usage:
    python visual_geometry_epochs_cell.py <subjID>
        [--bands theta alpha beta lowgamma highgamma]
        [--conditions ampOnly ampPhase] [--rois visual parietal frontal]
        [--voxRes 8mm] [--n_splits 10] [--n_null 100]
        [--max_pca_dim 50] [--seed 0] [--outdir <path>] [--force]
"""

import os

os.environ.setdefault('OMP_NUM_THREADS',      '1')
os.environ.setdefault('MKL_NUM_THREADS',      '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import time
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from align import (load_behav, verify_alignment, g04_orig_row_index, attach_behav)
from constants import AMP_PHASE_BANDS, ANGLE_MAPPING, get_bids_root
from features import build_features
from io_g03 import load_g03_unfiltered
from io_g04 import load_g04_band
from visual_geometry_cell import (
    LOCATIONS, MAX_PCA_DIM, MIN_TRIALS_PER_LOC, epoch_average, zscore_per_timepoint,
)
from visual_geometry_ts_cell import run_timepoint

LOCK_TYPE = 'stim'
DEFAULT_N_SPLITS = 10
DEFAULT_N_NULL = 100

# Behavioural binning. i_sacc_err is initial-saccade error, so LOW = GOOD; bin 0
# is therefore the best-performance third. Trials at or below the threshold are
# excluded as invalid rather than counted as perfect -- a ~0 error almost always
# means a missing/unparsed saccade, and the same threshold is used by
# plot_decoding_ts.py's quantile analysis.
I_SACC_ERR_THRESH = 0.001
DEFAULT_N_BINS = 3
BIN_NAMES = {2: ('better', 'worse'),
             3: ('best', 'mid', 'worst'),
             4: ('best', 'good', 'fair', 'worst')}


def bin_labels(n_bins):
    return BIN_NAMES.get(n_bins, tuple(f'q{i+1}' for i in range(n_bins)))


def performance_bins(err, y, n_bins, scope='location', log=print):
    """
    Split trials into n_bins performance bins. Returns (n_trials,) int with the
    bin index, or -1 for trials excluded as invalid.

    scope='location' (default) bins WITHIN each target location, so every bin
    gets ~1/n_bins of every location's trials. This matters: saccade error
    varies systematically with target location (some locations are simply
    harder), so a global split would load the "best" bin with easy locations and
    the "worst" bin with hard ones, and any geometry difference between bins
    would partly be a difference in which locations dominate them. Binning
    within location isolates trial-to-trial performance from location
    difficulty, and as a side effect keeps the per-location counts balanced,
    which the RDM needs anyway.

    scope='global' reproduces the simpler whole-session split if wanted.
    """
    err = np.asarray(err, dtype=float)
    valid = np.isfinite(err) & (err > I_SACC_ERR_THRESH)
    bins = np.full(err.shape, -1, dtype=int)
    if valid.sum() == 0:
        return bins

    def _assign(mask):
        vals = err[mask]
        if vals.size < n_bins:
            return
        edges = np.quantile(vals, np.linspace(0, 1, n_bins + 1)[1:-1])
        bins[mask] = np.clip(np.searchsorted(edges, vals, side='right'), 0, n_bins - 1)

    if scope == 'location':
        for loc in np.unique(y):
            _assign(valid & (y == loc))
    else:
        _assign(valid)
    return bins


def subject_trial_behaviour(subjID, voxRes, bids_root, roi, n_trials, log=print):
    """
    i_sacc_err row-aligned to the G04 trial order, or None if unavailable.

    The alignment is not optional bookkeeping: G04 concatenates trials grouped
    by target location, so its row order differs from the original session
    order that the behavioural file follows. g04_orig_row_index reconstructs the
    mapping from G03's trialinfo, and verify_alignment runs the same checksum
    S02B_organizeBehavForSource.m uses before anything is joined.
    """
    behav = load_behav(subjID, bids_root)
    if behav is None:
        log(f'    no behavioural file for sub-{subjID:02d} -- skipping bins.')
        return None
    try:
        g03 = load_g03_unfiltered(subjID, LOCK_TYPE, voxRes, bids_root, roi=roi)
    except (FileNotFoundError, OSError) as e:
        log(f'    sub-{subjID:02d}: no G03 metadata ({e}) -- skipping bins.')
        return None
    if not verify_alignment(g03['trialinfo_col2'], behav['tarlocCode']):
        log(f'    sub-{subjID:02d}: behavioural alignment CHECK FAILED -- skipping bins '
            f'rather than risking a silent mis-join.')
        return None
    idx = g04_orig_row_index(g03['trialinfo_col2'])
    if idx.shape[0] != n_trials:
        log(f'    sub-{subjID:02d}: trial_idx length {idx.shape[0]} != {n_trials} '
            f'-- skipping bins.')
        return None
    return attach_behav(idx, behav)['i_sacc_err']

# (lo, hi) half-open. See module docstring for the 0.8-1.0 s gap.
EPOCHS = {
    'fixation':    (-1.0, 0.0),
    'stimulus':    ( 0.0, 0.2),
    'early_delay': ( 0.2, 0.8),
    'late_delay':  ( 1.0, 1.6),
}
EPOCH_ORDER = ('fixation', 'stimulus', 'early_delay', 'late_delay')


def output_path(bids_root, subjID, band, condition, roi, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'visualGeometryEpochs')
    os.makedirs(base, exist_ok=True)
    return os.path.join(
        base, f'{subName}_task-mgs_visGeomEp_{condition}_{band}_{roi}_{voxRes}.npz')


def run_cell(subjID, bands, conditions, rois, voxRes, bids_root,
             n_splits=DEFAULT_N_SPLITS, n_null=DEFAULT_N_NULL,
             max_pca_dim=MAX_PCA_DIM, seed=0, n_bins=DEFAULT_N_BINS,
             bin_scope='location', min_trials=None, outdir=None, force=False):
    for band in bands:
        for condition in conditions:
            want_phase = (condition == 'ampPhase')
            if want_phase and band not in AMP_PHASE_BANDS:
                print(f'SKIP {condition}/{band}: no saved phase '
                      f'(AMP_PHASE_BANDS={AMP_PHASE_BANDS})', flush=True)
                continue

            for roi in rois:
                out_path = output_path(bids_root, subjID, band, condition, roi,
                                        voxRes, outdir)
                if not force and os.path.exists(out_path):
                    print(f'SKIP (exists): {out_path}', flush=True)
                    continue

                t_start = time.time()
                try:
                    g04 = load_g04_band(subjID, LOCK_TYPE, band, voxRes, bids_root,
                                         want_phase=want_phase, roi=roi)
                    amp   = g04['amp']
                    phase = g04['phase'] if want_phase else None
                    tv    = g04['time_vector']
                    y     = g04['target_labels'].astype(int)

                    X = build_features(condition, amp, phase)
                    del amp, phase, g04
                    X = zscore_per_timepoint(X)

                    # Bin 0 is always ALL trials, so every binned result has a
                    # same-pipeline reference to be read against.
                    err = subject_trial_behaviour(subjID, voxRes, bids_root, roi,
                                                   X.shape[0], log=print)
                    if err is not None and n_bins > 1:
                        bidx = performance_bins(err, y, n_bins, scope=bin_scope)
                        bnames = ('all',) + bin_labels(n_bins)
                        trial_sets = [np.ones(X.shape[0], bool)] + \
                                     [bidx == b for b in range(n_bins)]
                    else:
                        bnames = ('all',)
                        trial_sets = [np.ones(X.shape[0], bool)]

                    n_ep, n_b = len(EPOCH_ORDER), len(bnames)
                    rdms = np.full((n_b, n_ep, len(LOCATIONS), len(LOCATIONS)), np.nan)
                    rdms_null = (np.full((n_b, n_ep, n_null, len(LOCATIONS), len(LOCATIONS)),
                                          np.nan) if n_null > 0 else None)
                    pca_dims = np.zeros((n_b, n_ep), int)
                    whit = np.zeros((n_b, n_ep), bool)
                    n_win = np.zeros(n_ep, int)
                    n_bin_trials = np.array([int(m.sum()) for m in trial_sets])
                    # Smallest per-location count in each bin -- crossnobis needs
                    # >= MIN_TRIALS_PER_LOC per location or that location drops out.
                    min_per_loc = np.array([
                        int(min((int(((y == l) & m).sum()) for l in LOCATIONS), default=0))
                        for m in trial_sets])

                    for i, ep in enumerate(EPOCH_ORDER):
                        lo, hi = EPOCHS[ep]
                        Xe, n_times = epoch_average(X, tv, lo, hi, hi_inclusive=False)
                        n_win[i] = n_times
                        min_tr = MIN_TRIALS_PER_LOC if min_trials is None else min_trials
                        for bi, m in enumerate(trial_sets):
                            if m.sum() < len(LOCATIONS) * min_tr:
                                continue
                            rdm, rdm_null, meta = run_timepoint(
                                Xe[m], y[m], max_pca_dim, n_splits, n_null, seed,
                                min_trials=min_tr)
                            rdms[bi, i] = rdm
                            if n_null > 0:
                                rdms_null[bi, i] = rdm_null
                            pca_dims[bi, i], whit[bi, i] = meta['pca_dim'], meta['whitened']

                    save_kw = dict(
                        rdm                 = rdms.astype(np.float32),
                        bins                = np.array(bnames),
                        n_bin_trials        = n_bin_trials,
                        min_trials_per_loc  = min_per_loc,
                        bin_scope           = np.array([bin_scope]),
                        epochs              = np.array(EPOCH_ORDER),
                        epoch_bounds        = np.array([EPOCHS[e] for e in EPOCH_ORDER]),
                        n_window_times      = n_win,
                        locations           = np.array(LOCATIONS),
                        location_angles_deg = np.array([ANGLE_MAPPING[l] for l in LOCATIONS],
                                                        dtype=float),
                        pca_dim             = pca_dims,
                        whitened            = whit,
                        target_labels       = y.astype(np.int32),
                        n_trials            = np.array([X.shape[0]]),
                        n_features          = np.array([X.shape[2]]),
                        n_splits            = np.array([n_splits]),
                        n_null              = np.array([n_null]),
                        subjID = np.array([subjID]), band = np.array([band]),
                        condition = np.array([condition]), roi = np.array([roi]),
                        voxRes = np.array([voxRes]), seed = np.array([seed]),
                    )
                    if n_null > 0:
                        save_kw['rdm_null'] = rdms_null.astype(np.float32)
                    np.savez_compressed(out_path, **save_kw)

                    thin = [f'{b}({t} tr, min {mp}/loc)'
                            for b, t, mp in zip(bnames, n_bin_trials, min_per_loc)]
                    print(f'sub-{subjID:02d} | {band} | {condition} | {roi}: '
                          f'N={X.shape[0]} F={X.shape[2]} | bins: {", ".join(thin)} | '
                          f'k={int(np.median(pca_dims))} | whitened={whit.mean()*100:.0f}% | '
                          f'n_null={n_null} | {time.time() - t_start:.1f}s', flush=True)
                    _mt = MIN_TRIALS_PER_LOC if min_trials is None else min_trials
                    for b, mp in zip(bnames, min_per_loc):
                        if b != 'all' and mp < _mt:
                            print(f'    NOTE bin {b!r}: smallest location has only {mp} '
                                  f'trials (< {_mt}); that location is '
                                  f'dropped from this RDM.', flush=True)
                    del X
                except (FileNotFoundError, ValueError) as e:
                    print(f'  SKIP sub-{subjID:02d} {band}/{condition}/{roi}: {e}', flush=True)
                except Exception:
                    print(f'  FAILED sub-{subjID:02d} {band}/{condition}/{roi}:', flush=True)
                    traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(
        description='Epoch-based crossnobis RDMs (+ label-shuffle null) for MDS geometry.')
    ap.add_argument('subjID', type=int)
    ap.add_argument('--bands', nargs='+',
                     default=['theta', 'alpha', 'beta', 'lowgamma', 'highgamma'])
    ap.add_argument('--conditions', nargs='+', default=['ampOnly', 'ampPhase'],
                     choices=['ampOnly', 'ampPhase'])
    ap.add_argument('--rois', nargs='+', default=['visual', 'parietal', 'frontal'])
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--n_splits', type=int, default=DEFAULT_N_SPLITS)
    ap.add_argument('--n_null', type=int, default=DEFAULT_N_NULL)
    ap.add_argument('--max_pca_dim', type=int, default=MAX_PCA_DIM)
    ap.add_argument('--n_bins', type=int, default=DEFAULT_N_BINS,
                     help='Performance bins from i_sacc_err (default 3; 1 disables). '
                          'Bin 0 of the output is always ALL trials as a reference.')
    ap.add_argument('--bin_scope', default='location', choices=['location', 'global'],
                     help="'location' (default) bins within each target location, so "
                          "bins are not confounded by location difficulty; 'global' "
                          "splits the whole session at once.")
    ap.add_argument('--min_trials_per_loc', type=int, default=None,
                     help=f'Minimum trials per location for a bin to use that location '
                          f'(default {MIN_TRIALS_PER_LOC} = one per cross-validation fold, '
                          f'the true minimum; crossnobis stays unbiased there).')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    bids_root = get_bids_root()
    print(f'visual_geometry_epochs_cell | sub-{args.subjID:02d} | bands={args.bands} | '
          f'conditions={args.conditions} | rois={args.rois} | {args.voxRes} | '
          f'epochs={ {e: EPOCHS[e] for e in EPOCH_ORDER} } | n_splits={args.n_splits} | '
          f'n_null={args.n_null} | force={args.force}', flush=True)

    t0 = time.time()
    run_cell(args.subjID, list(args.bands), list(args.conditions), list(args.rois),
             args.voxRes, bids_root, n_splits=args.n_splits, n_null=args.n_null,
             max_pca_dim=args.max_pca_dim, seed=args.seed,
             n_bins=args.n_bins, bin_scope=args.bin_scope,
             min_trials=args.min_trials_per_loc,
             outdir=args.outdir, force=args.force)
    print(f'Done | sub-{args.subjID:02d} | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
