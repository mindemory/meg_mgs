#!/usr/bin/env python3
"""
connectivity_imcoh_epochs.py

Imaginary coherence between ROI pairs (visual-parietal, visual-frontal,
parietal-frontal) over the four task epochs, per band, per subject.

Coherency between two sources is estimated from the analytic signal G04
already provides -- z = amp * exp(i*phase) -- pooled over trials and over the
timepoints inside an epoch:

    C_ij = <z_i conj(z_j)> / sqrt(<|z_i|^2> <|z_j|^2>),   ImCoh_ij = Im(C_ij)

ImCoh is used rather than plain coherence because it is insensitive to
zero-lag coupling, which is what spatial leakage between nearby sources
produces; only genuinely lagged interaction contributes.

=============================================================================
TWO THINGS THAT WOULD OTHERWISE WRECK THIS, both verified numerically
=============================================================================
1. NEVER AVERAGE SIGNED ImCoh ACROSS SOURCE PAIRS. Im(C) is signed by the
   direction of the phase lag, so pooling pairs where A leads with pairs where
   A lags cancels them out. Simulated with true coupling of 0.65: averaging the
   signed values across pairs with mixed lag directions gave 0.0006, while the
   mean of |ImCoh| correctly gave 0.648. This script therefore reports
   mean|ImCoh|, matching the convention already used in
   megScripts/inSourceSpaceConnectivity.py.

2. mean|ImCoh| IS NOISE-BIASED, AND THE BIAS SCALES AS 1/sqrt(n_samples).
   With no coupling whatsoever it is not 0: simulated 0.0255 at 500 samples
   and 0.0040 at 20000. That matters enormously here because the four epochs
   have very different lengths (1.0 / 0.2 / 0.6 / 0.6 s), so at 250 Hz the
   stimulus epoch has 5x fewer samples than fixation and a noise floor 2.2x
   higher. Comparing raw mean|ImCoh| across these epochs would manufacture a
   stimulus-onset "increase" out of nothing. Two defences, both applied:

     * SAMPLE EQUALISATION (default on): every epoch is subsampled to the same
       number of (trial x timepoint) observations as the SHORTEST epoch, so all
       four have identical statistical power by construction. This is the main
       fix; without it the epoch comparison is not interpretable.
     * A TRIAL-SHUFFLE SURROGATE NULL: the trial order of ROI B is permuted
       relative to ROI A, which destroys genuine trial-locked coupling while
       preserving each ROI's own spectrum and within-trial temporal structure.
       The resulting mean|ImCoh| is the floor for that cell, reported alongside
       the real value so the two can be compared directly rather than the real
       value being read against an assumed zero.

   The null also absorbs a subtlety the analytic 1/sqrt(n) formula misses:
   timepoints within a trial are autocorrelated at the band's timescale, so the
   effective degrees of freedom are well below the nominal sample count. The
   surrogate inherits exactly the same autocorrelation, so its floor is the
   right one.

For speed the null is computed on a random subset of source pairs
(--null_pairs, default 200 per ROI): the floor is a property of the sample
count and the spectra, not of which particular pairs are chosen, so a subset
estimates it to plenty of precision at a fraction of the cost.

Bands: theta/alpha/beta only -- ImCoh needs phase, and lowgamma/highgamma have
none saved (constants.AMP_PHASE_BANDS).

Usage:
    python connectivity_imcoh_epochs.py <subjID> [--bands theta alpha beta]
        [--pairs visual-parietal visual-frontal parietal-frontal]
        [--voxRes 8mm] [--n_null 20] [--null_pairs 200]
        [--no_equalise] [--seed 0] [--outdir <path>] [--force]
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

from constants import AMP_PHASE_BANDS, get_bids_root
from io_g04 import load_g04_band
from visual_geometry_epochs_cell import EPOCHS, EPOCH_ORDER

LOCK_TYPE = 'stim'
DEFAULT_PAIRS = ('visual-parietal', 'visual-frontal', 'parietal-frontal')
DEFAULT_N_NULL = 20
DEFAULT_NULL_PAIRS = 200


def output_path(bids_root, subjID, band, pair, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    base = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'imcohEpochs')
    os.makedirs(base, exist_ok=True)
    return os.path.join(
        base, f'{subName}_task-mgs_imcoh_{pair}_{band}_{voxRes}.npz')


def analytic(amp, phase):
    """G04 amplitude/phase -> analytic signal, complex64 to halve memory."""
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def mean_abs_imcoh(ZA, ZB, block=64):
    """
    mean |Im(coherency)| over all pairs between two ROIs.

    ZA (n_samp, nA), ZB (n_samp, nB) analytic signals, already flattened over
    trials x timepoints. Computed in blocks over ROI A so the (nA, nB)
    cross-spectrum never has to exist in full -- for visual x parietal that
    would be ~300k complex entries per cell.
    """
    n = ZA.shape[0]
    pa = np.mean(np.abs(ZA) ** 2, axis=0)
    pb = np.mean(np.abs(ZB) ** 2, axis=0)
    pa = np.where(pa < 1e-20, 1e-20, pa)
    pb = np.where(pb < 1e-20, 1e-20, pb)
    tot, cnt = 0.0, 0
    for i0 in range(0, ZA.shape[1], block):
        i1 = min(i0 + block, ZA.shape[1])
        S = ZA[:, i0:i1].conj().T @ ZB / n              # (blk, nB)
        C = S / np.sqrt(np.outer(pa[i0:i1], pb))
        tot += float(np.abs(np.imag(C)).sum())
        cnt += C.size
    return tot / max(cnt, 1)


def epoch_samples(z, tv, lo, hi, n_keep, rng):
    """
    (n_trials, n_times, n_src) -> (n_samp, n_src) flattened over trials x time,
    subsampled to exactly n_keep TIMEPOINTS per trial so every epoch
    contributes the same number of observations (see module docstring).
    """
    m = (tv >= lo) & (tv < hi)
    idx = np.where(m)[0]
    if idx.size == 0:
        raise ValueError(f'no timepoints in [{lo}, {hi})')
    if n_keep is not None and idx.size > n_keep:
        idx = np.sort(rng.choice(idx, size=n_keep, replace=False))
    zz = z[:, idx, :]
    return zz.reshape(-1, zz.shape[2]), idx.size


def run_cell(subjID, bands, pairs, voxRes, bids_root, n_null=DEFAULT_N_NULL,
             null_pairs=DEFAULT_NULL_PAIRS, equalise=True, seed=0,
             outdir=None, force=False):
    rng = np.random.default_rng(seed)

    for band in bands:
        if band not in AMP_PHASE_BANDS:
            print(f'SKIP {band}: ImCoh needs phase, which is not saved for this band '
                  f'(AMP_PHASE_BANDS={AMP_PHASE_BANDS})', flush=True)
            continue

        for pair in pairs:
            roi_a, roi_b = pair.split('-')
            out_path = output_path(bids_root, subjID, band, pair, voxRes, outdir)
            if not force and os.path.exists(out_path):
                print(f'SKIP (exists): {out_path}', flush=True)
                continue

            t_start = time.time()
            try:
                Zs, tv, n_trials = {}, None, None
                for roi in (roi_a, roi_b):
                    g = load_g04_band(subjID, LOCK_TYPE, band, voxRes, bids_root,
                                       want_phase=True, roi=roi)
                    if g['phase'] is None:
                        raise ValueError(f'no phase for {roi}/{band}')
                    Zs[roi] = analytic(g['amp'], g['phase'])
                    tv = g['time_vector']
                    n_trials = g['amp'].shape[0]
                    del g

                # Equalise observations across epochs to the SHORTEST epoch's
                # timepoint count -- the whole epoch comparison rests on this.
                n_ep_times = [int(((tv >= EPOCHS[e][0]) & (tv < EPOCHS[e][1])).sum())
                              for e in EPOCH_ORDER]
                n_keep = min(n_ep_times) if equalise else None

                real = np.full(len(EPOCH_ORDER), np.nan)
                null_m = np.full(len(EPOCH_ORDER), np.nan)
                null_s = np.full(len(EPOCH_ORDER), np.nan)
                n_samp = np.zeros(len(EPOCH_ORDER), int)

                for i, ep in enumerate(EPOCH_ORDER):
                    lo, hi = EPOCHS[ep]
                    ZA, kept = epoch_samples(Zs[roi_a], tv, lo, hi, n_keep, rng)
                    ZB, _ = epoch_samples(Zs[roi_b], tv, lo, hi, n_keep, rng)
                    n_samp[i] = ZA.shape[0]
                    real[i] = mean_abs_imcoh(ZA, ZB)

                    if n_null > 0:
                        # Subset of pairs for the null -- the floor depends on the
                        # sample count and spectra, not on which pairs (docstring).
                        ia = rng.choice(ZA.shape[1], min(null_pairs, ZA.shape[1]),
                                         replace=False)
                        ib = rng.choice(ZB.shape[1], min(null_pairs, ZB.shape[1]),
                                         replace=False)
                        za = Zs[roi_a][:, :, ia]
                        zb = Zs[roi_b][:, :, ib]
                        vals = []
                        for _ in range(n_null):
                            perm = rng.permutation(n_trials)
                            a, _ = epoch_samples(za, tv, lo, hi, n_keep, rng)
                            b, _ = epoch_samples(zb[perm], tv, lo, hi, n_keep, rng)
                            vals.append(mean_abs_imcoh(a, b))
                        null_m[i], null_s[i] = float(np.mean(vals)), float(np.std(vals))
                        del za, zb

                np.savez_compressed(
                    out_path,
                    epochs           = np.array(EPOCH_ORDER),
                    epoch_bounds     = np.array([EPOCHS[e] for e in EPOCH_ORDER]),
                    mean_abs_imcoh   = real.astype(np.float64),
                    null_mean        = null_m.astype(np.float64),
                    null_std         = null_s.astype(np.float64),
                    n_samples        = n_samp,
                    n_epoch_times    = np.array(n_ep_times),
                    timepts_used     = np.array([n_keep if n_keep else -1]),
                    equalised        = np.array([bool(equalise)]),
                    n_sources        = np.array([Zs[roi_a].shape[2], Zs[roi_b].shape[2]]),
                    n_trials         = np.array([n_trials]),
                    n_null           = np.array([n_null]),
                    subjID = np.array([subjID]), band = np.array([band]),
                    pair = np.array([pair]), roi_a = np.array([roi_a]),
                    roi_b = np.array([roi_b]), voxRes = np.array([voxRes]),
                    seed = np.array([seed]),
                )
                excess = real - null_m
                print(f'sub-{subjID:02d} | {band} | {pair}: '
                      f'{Zs[roi_a].shape[2]}x{Zs[roi_b].shape[2]} sources, '
                      f'{n_trials} trials, {n_keep} timepts/epoch '
                      f'({n_samp[0]} samples) | ' +
                      '  '.join(f'{e}={r:.4f}(null {n:.4f}, +{x:.4f})'
                                for e, r, n, x in zip(EPOCH_ORDER, real, null_m, excess)) +
                      f' | {time.time() - t_start:.0f}s', flush=True)
                del Zs
            except (FileNotFoundError, ValueError) as e:
                print(f'  SKIP sub-{subjID:02d} {band}/{pair}: {e}', flush=True)
            except Exception:
                print(f'  FAILED sub-{subjID:02d} {band}/{pair}:', flush=True)
                traceback.print_exc()


def main():
    ap = argparse.ArgumentParser(
        description='ROI-pair imaginary coherence over the four task epochs.')
    ap.add_argument('subjID', type=int)
    ap.add_argument('--bands', nargs='+', default=['theta', 'alpha', 'beta'])
    ap.add_argument('--pairs', nargs='+', default=list(DEFAULT_PAIRS))
    ap.add_argument('--voxRes', default='8mm')
    ap.add_argument('--n_null', type=int, default=DEFAULT_N_NULL,
                     help='Trial-shuffle surrogates per epoch (0 disables). mean|ImCoh| '
                          'has a strongly non-zero noise floor, so keep this on.')
    ap.add_argument('--null_pairs', type=int, default=DEFAULT_NULL_PAIRS,
                     help='Source pairs sampled per ROI for the null (default 200).')
    ap.add_argument('--no_equalise', action='store_false', dest='equalise',
                     help='Do NOT equalise sample counts across epochs. The epochs '
                          'differ 5-fold in length, so this makes the epoch comparison '
                          'uninterpretable -- for diagnostics only.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    bids_root = get_bids_root()
    print(f'connectivity_imcoh_epochs | sub-{args.subjID:02d} | bands={args.bands} | '
          f'pairs={args.pairs} | {args.voxRes} | n_null={args.n_null} | '
          f'equalise={args.equalise} | force={args.force}', flush=True)

    t0 = time.time()
    run_cell(args.subjID, list(args.bands), list(args.pairs), args.voxRes, bids_root,
             n_null=args.n_null, null_pairs=args.null_pairs, equalise=args.equalise,
             seed=args.seed, outdir=args.outdir, force=args.force)
    print(f'Done | sub-{args.subjID:02d} | total {time.time() - t0:.1f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
