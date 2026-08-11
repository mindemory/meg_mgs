#!/usr/bin/env python3
"""
intrinsic_dim_cell.py

CLI entry point for parallelised intrinsic-dimensionality computation.
Processes ONE (subjID, lockType, band) cell and saves per-ROI PR/n_pcs
curves to a .npz file under:

    <bids_root>/derivatives/<subName>/sourceRecon/intrinsicDim/
        <subName>_task-mgs_intrinsicDim_<band>_<lockType>_<voxRes>.npz

Each .npz contains, for every ROI key (e.g. 'pr_visual', 'npcs_visual',
'time_vector_visual'):
    pr_<roi>          : (n_times,) float  -- participation ratio
    npcs_<roi>        : (n_times,) int    -- n PCs for >= var_threshold var
    time_vector_<roi> : (n_times,) float  -- seconds

Meant to be fanned out by run_intrinsic_dim.sh, one background job per
(subjID, lockType, band). The plotter (intrinsic_dimensionality.py --plot_only)
reads these files after all jobs finish.

Usage:
    python intrinsic_dim_cell.py <subjID> <lockType> <band>
                                 [--voxRes 8mm]
                                 [--rois visual parietal frontal]
                                 [--var_threshold 0.90]
                                 [--outdir <custom_dir>]
"""

import os

# Pin BLAS/LAPACK threads to 1 per process.  The shell fan-out
# (run_intrinsic_dim.sh) launches many concurrent cells; without this each
# process spawns its own multi-threaded BLAS pool for the eigvalsh calls
# inside dim_over_time, over-subscribing the machine's cores far more than
# the actual compute warrants.
os.environ.setdefault('OMP_NUM_THREADS',    '1')
os.environ.setdefault('MKL_NUM_THREADS',    '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import sys
import argparse
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from atlas import load_atlas_masks
from constants import ROI_NAMES, get_bids_root
from io_g03 import load_g03_unfiltered
from io_g04 import load_g04_band


# ── Dimensionality estimators (same as intrinsic_dimensionality.py) ──────────

def participation_ratio(X_t):
    """Participation ratio from z-scored (n_trials, n_sources) slice."""
    C   = np.cov(X_t, rowvar=False)
    lam = np.maximum(np.linalg.eigvalsh(C), 0.0)
    s1  = lam.sum()
    if s1 < 1e-30:
        return 1.0
    return float(s1 ** 2 / (lam ** 2).sum())


def n_pcs_for_var(X_t, var_threshold=0.90):
    """Number of PCs needed to explain >= var_threshold of total variance."""
    C   = np.cov(X_t, rowvar=False)
    lam = np.sort(np.maximum(np.linalg.eigvalsh(C), 0.0))[::-1]
    s   = lam.sum()
    if s < 1e-30:
        return 1
    cum  = np.cumsum(lam) / s
    hits = np.where(cum >= var_threshold)[0]
    return int(hits[0] + 1) if hits.size > 0 else len(lam)


def dim_over_time(data, var_threshold=0.90):
    """
    data : (n_trials, n_times, n_sources)
    Returns pr (n_times,) and npcs (n_times,).
    """
    n_trials, n_times, n_sources = data.shape
    pr   = np.zeros(n_times)
    npcs = np.zeros(n_times, dtype=int)
    for t in range(n_times):
        X_t = data[:, t, :]
        mu  = X_t.mean(axis=0, keepdims=True)
        sd  = X_t.std(axis=0,  keepdims=True)
        sd[sd < 1e-10] = 1.0
        X_z      = (X_t - mu) / sd
        pr[t]    = participation_ratio(X_z)
        npcs[t]  = n_pcs_for_var(X_z, var_threshold)
    return pr, npcs


# ── Output path ──────────────────────────────────────────────────────────────

def output_path(bids_root, subjID, band, lockType, voxRes, outdir=None):
    subName = f'sub-{subjID:02d}'
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        base_dir = outdir
    else:
        base_dir = os.path.join(bids_root, 'derivatives', subName,
                                 'sourceRecon', 'intrinsicDim')
        os.makedirs(base_dir, exist_ok=True)
    fname = (f'{subName}_task-mgs_intrinsicDim_{band}_{lockType}_{voxRes}.npz')
    return os.path.join(base_dir, fname)


# ── Main computation ─────────────────────────────────────────────────────────

def run_cell(subjID, lockType, band, voxRes, bids_root, rois_all,
             var_threshold, outdir=None):

    out_path = output_path(bids_root, subjID, band, lockType, voxRes, outdir)
    if os.path.exists(out_path):
        print(f'SKIP (exists): {out_path}')
        return

    need_whole = 'whole' in rois_all
    rois_cache = [r for r in rois_all if r != 'whole']

    arrays = {}

    # ── Whole-grid path (only when 'whole' ROI is requested) ─────────────────
    # Requires the full G03 (and G04 for non-broadband) HDF5 file.  Skipped
    # entirely when 'whole' is not in rois_all, since every other ROI is
    # served by the per-ROI cache fast path below.
    if need_whole:
        try:
            g03 = load_g03_unfiltered(subjID, lockType, voxRes, bids_root)
        except FileNotFoundError as e:
            print(f'sub-{subjID:02d} G03 missing: {e}')
            if not rois_cache:
                return
            g03 = None

        if g03 is not None:
            inside_pos = g03['inside_pos']
            g03_data   = g03['data']      # (n_trials, n_times, n_sources)

            # Atlas bounds guard
            atlas_masks  = load_atlas_masks(voxRes, bids_root)
            n_atlas_grid = len(next(iter(atlas_masks.values())))
            valid_mask   = (inside_pos >= 1) & (inside_pos <= n_atlas_grid)
            if not valid_mask.all():
                n_bad = (~valid_mask).sum()
                print(f'  sub-{subjID:02d}: {n_bad} source column(s) exceed atlas '
                      f'grid ({n_atlas_grid}), dropping.')
                valid_cols = np.where(valid_mask)[0]
                g03_data   = g03_data[:, :, valid_cols]
            else:
                valid_mask = np.ones(g03_data.shape[2], dtype=bool)  # sentinel

            src_data_whole    = None
            time_vector_whole = None

            if band == 'broadband':
                src_data_whole    = g03_data
                time_vector_whole = g03['time_vector']
            else:
                try:
                    g04 = load_g04_band(subjID, lockType, band, voxRes, bids_root,
                                         want_phase=False)
                except (FileNotFoundError, ValueError) as e:
                    print(f'sub-{subjID:02d} {band}: {e}')
                    if not rois_cache:
                        return
                    g04 = None

                if g04 is not None:
                    amp = g04['amp']
                    if not valid_mask.all():
                        valid_cols = np.where(valid_mask)[0]
                        amp = amp[:, :, valid_cols]
                    src_data_whole    = amp
                    time_vector_whole = g04['time_vector']

            if src_data_whole is not None and src_data_whole.shape[2] > 0:
                pr, npcs = dim_over_time(src_data_whole, var_threshold)
                arrays['pr_whole']          = pr
                arrays['npcs_whole']        = npcs.astype(np.int32)
                arrays['time_vector_whole'] = time_vector_whole

    # ── ROI-cache fast path (non-'whole' ROIs) ────────────────────────────────
    # Loads small per-ROI .npz files written by precompute_roi_splits.py
    # instead of the full whole-grid HDF5 -- avoids 8-10 GB reads when only
    # a handful of ROI sources are needed.
    for roi in rois_cache:
        try:
            if band == 'broadband':
                roi_data    = load_g03_unfiltered(subjID, lockType, voxRes,
                                                   bids_root, roi=roi)
                data_roi    = roi_data['data']
                time_vector = roi_data['time_vector']
            else:
                roi_data    = load_g04_band(subjID, lockType, band, voxRes,
                                             bids_root, want_phase=False, roi=roi)
                data_roi    = roi_data['amp']
                time_vector = roi_data['time_vector']
        except (FileNotFoundError, ValueError) as e:
            print(f'  sub-{subjID:02d} {band} roi={roi}: {e}')
            continue

        if data_roi.shape[2] == 0:
            print(f'  sub-{subjID:02d} {band} {roi}: empty ROI, skipping')
            continue

        pr, npcs = dim_over_time(data_roi, var_threshold)
        arrays[f'pr_{roi}']          = pr
        arrays[f'npcs_{roi}']        = npcs.astype(np.int32)
        arrays[f'time_vector_{roi}'] = time_vector

    if not arrays:
        print(f'sub-{subjID:02d} {band}: no ROI data to save')
        return

    arrays['subjID']        = np.array([subjID])
    arrays['var_threshold'] = np.array([var_threshold])
    np.savez(out_path, **arrays)
    print(f'Saved: {out_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Intrinsic-dimensionality cell: one subject x lockType x band.')
    parser.add_argument('subjID',   type=int)
    parser.add_argument('lockType', choices=['stim', 'resp'])
    parser.add_argument('band')
    parser.add_argument('--voxRes',        default='8mm')
    parser.add_argument('--rois',          nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--var_threshold', type=float, default=0.90)
    parser.add_argument('--outdir',        default=None,
                        help='Override per-subject output dir')
    args = parser.parse_args()

    bids_root = get_bids_root()

    rois_all = list(args.rois)
    if 'whole' not in rois_all:
        rois_all.append('whole')

    print(f'intrinsic_dim_cell | sub-{args.subjID:02d} | {args.lockType} | '
          f'{args.band} | {args.voxRes} | rois={rois_all}')

    run_cell(args.subjID, args.lockType, args.band, args.voxRes,
             bids_root, rois_all, args.var_threshold, args.outdir)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
