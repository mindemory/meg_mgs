#!/usr/bin/env python3
"""
precompute_roi_splits.py

Splits each subject's massive whole-grid G03 (broadband) and G04 (band
amplitude/phase) .mat files into small per-ROI (visual/parietal/frontal)
.npz caches, saved alongside the original raw files. The original
whole-grid files are left untouched -- analyses that genuinely need
whole-brain data (e.g. plot_timeseries.py, intrinsic_dim_cell.py) keep
reading them directly.

Analyses that only need one ROI (e.g. run_glue_cell.py's default
--rois visual parietal frontal) can then load io_g03.load_g03_unfiltered(...,
roi='visual') / io_g04.load_g04_band(..., roi='visual') and get a tiny file
instead of paying the cost of loading+dereferencing the full 8-10GB grid.

Output files:
    derivatives/sub-{XX}/sourceRecon/
        sub-{XX}_task-mgs_sourceSpaceData_{res}_{lockType}_roi-{roi}.npz
    derivatives/sub-{XX}/sourceRecon/freqSpace/
        sub-{XX}_task-mgs_{band}_allTargets_{res}_{lockType}_roi-{roi}.npz

Parallelism:
    One joblib worker per (subjID, lockType) unit (processes, not threads).
    This is memory-bound, not just IO-bound -- each worker loads an 8-10GB
    whole-grid array at a time -- so n_jobs defaults conservatively low,
    unlike plot_timeseries.py's n_jobs=8 default.

Usage:
    python precompute_roi_splits.py [--subjects 1 2 ...] [--lockTypes stim resp]
                                     [--voxRes 8mm] [--rois visual parietal frontal]
                                     [--bands theta alpha ...] [--n_jobs 4] [--force]
"""

import os
import sys
import argparse
import tempfile
import traceback
from pathlib import Path
from multiprocessing import cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from joblib import Parallel, delayed

from atlas import load_atlas_masks, roi_local_indices
from constants import AMP_ONLY_BANDS, AMP_PHASE_BANDS, ROI_NAMES, SUBJECT_LIST, get_bids_root
from io_g03 import g03_roi_cache_path, load_g03_unfiltered
from io_g04 import g04_roi_cache_path, load_g04_band


def _atomic_savez(fpath, **arrays):
    """
    Write arrays to fpath via a same-directory temp file + os.replace(),
    so a concurrent reader never sees a half-written cache file (mirrors
    the race-condition fix in constants.py's _copy_and_open).
    """
    target_dir = os.path.dirname(fpath)
    fd, tmp = tempfile.mkstemp(prefix='.tmp_', suffix='.npz', dir=target_dir)
    os.close(fd)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, fpath)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _filter_inside_pos(inside_pos, atlas_masks):
    """
    Drop source columns whose 1-based inside_pos exceeds the atlas grid --
    same guard plot_timeseries.py/intrinsic_dim_cell.py apply before calling
    roi_local_indices (a small number of subjects have out-of-bounds
    columns). Returns (filtered_inside_pos, valid_cols) where valid_cols
    indexes into the ORIGINAL (unfiltered) source axis -- callers must slice
    any other per-source array (e.g. G04's amp/phase) with valid_cols too,
    since they share this exact column space.
    """
    n_grid = len(next(iter(atlas_masks.values())))
    valid_mask = (inside_pos >= 1) & (inside_pos <= n_grid)
    valid_cols = np.where(valid_mask)[0]
    return inside_pos[valid_cols], valid_cols


def _save_g03_roi_splits(subjID, lockType, voxRes, bids_root, atlas_masks, g03, rois, force):
    """Save any missing (or --force) G03 per-ROI caches from an already-loaded g03 dict."""
    for roi in rois:
        out_path = g03_roi_cache_path(subjID, lockType, voxRes, bids_root, roi)
        if not force and os.path.exists(out_path):
            continue
        idx = roi_local_indices(atlas_masks, g03['inside_pos'], roi)
        _atomic_savez(
            out_path,
            data=g03['data'][:, :, idx],
            time_vector=g03['time_vector'],
            target_labels=g03['target_labels'],
            trialinfo_col2=g03['trialinfo_col2'],
            inside_pos=g03['inside_pos'][idx],
            fsample=g03['fsample'],
        )
        print(f'  sub-{subjID:02d} {lockType} G03 roi={roi}: saved '
              f'({idx.size} sources)', flush=True)


def _save_g04_roi_splits(subjID, lockType, band, voxRes, bids_root, atlas_masks,
                          inside_pos_full, valid_cols, rois, force):
    want_phase = band in AMP_PHASE_BANDS
    out_paths = {roi: g04_roi_cache_path(subjID, lockType, band, voxRes, bids_root, roi)
                 for roi in rois}
    if not force and all(os.path.exists(p) for p in out_paths.values()):
        print(f'  sub-{subjID:02d} {lockType} {band}: SKIP (all ROI caches exist)', flush=True)
        return

    try:
        g04 = load_g04_band(subjID, lockType, band, voxRes, bids_root, want_phase=want_phase)
    except (FileNotFoundError, ValueError) as e:
        print(f'  sub-{subjID:02d} {lockType} {band}: {e}', flush=True)
        return

    # G04's source axis is in the same (unfiltered) column space as G03's inside_pos --
    # drop the same out-of-bounds columns before computing ROI indices (see
    # _filter_inside_pos).
    if valid_cols.size != g04['amp'].shape[2]:
        g04['amp'] = g04['amp'][:, :, valid_cols]
        if g04['phase'] is not None:
            g04['phase'] = g04['phase'][:, :, valid_cols]

    for roi in rois:
        out_path = out_paths[roi]
        if not force and os.path.exists(out_path):
            continue
        idx = roi_local_indices(atlas_masks, inside_pos_full, roi)
        arrays = dict(
            amp=g04['amp'][:, :, idx],
            time_vector=g04['time_vector'],
            target_labels=g04['target_labels'],
            trialinfo_col2=g04['trialinfo_col2'],
            actualRate=g04['actualRate'],
            freq_range=np.array(g04['freq_range']),
            inside_pos=inside_pos_full[idx],
        )
        if g04['phase'] is not None:
            arrays['phase'] = g04['phase'][:, :, idx]
        _atomic_savez(out_path, **arrays)
        print(f'  sub-{subjID:02d} {lockType} {band} roi={roi}: saved '
              f'({idx.size} sources)', flush=True)
    del g04


def _all_exist(paths):
    return all(os.path.exists(p) for p in paths)


def process_unit(subjID, lockType, voxRes, bids_root, rois, bands, force):
    """One (subjID, lockType) unit: G03 + all requested G04 bands, all ROIs."""
    print(f'sub-{subjID:02d} {lockType}: starting', flush=True)

    g03_out_paths = [g03_roi_cache_path(subjID, lockType, voxRes, bids_root, roi)
                      for roi in rois]
    g04_out_paths = [g04_roi_cache_path(subjID, lockType, band, voxRes, bids_root, roi)
                      for band in bands for roi in rois]

    if not force and _all_exist(g03_out_paths) and _all_exist(g04_out_paths):
        print(f'sub-{subjID:02d} {lockType}: SKIP (all ROI caches exist)', flush=True)
        return

    try:
        atlas_masks = load_atlas_masks(voxRes, bids_root)

        # G03 is loaded at most once per unit -- its inside_pos (full-grid indices)
        # is also needed for G04's ROI slicing, since G04's per-target structs don't
        # carry inside_pos themselves (see io_g04.py docstring).
        g03 = load_g03_unfiltered(subjID, lockType, voxRes, bids_root)
        inside_pos_full, valid_cols = _filter_inside_pos(g03['inside_pos'], atlas_masks)
        if valid_cols.size != g03['data'].shape[2]:
            n_bad = g03['data'].shape[2] - valid_cols.size
            print(f'  sub-{subjID:02d} {lockType}: {n_bad} source column(s) exceed atlas '
                  f'grid, dropping.', flush=True)
            g03['data'] = g03['data'][:, :, valid_cols]
            g03['inside_pos'] = inside_pos_full
        _save_g03_roi_splits(subjID, lockType, voxRes, bids_root, atlas_masks, g03, rois, force)
        del g03

        for band in bands:
            _save_g04_roi_splits(subjID, lockType, band, voxRes, bids_root, atlas_masks,
                                  inside_pos_full, valid_cols, rois, force)
    except (FileNotFoundError, OSError) as e:
        # A bad/missing raw file for this subject (e.g. corrupt/truncated .mat)
        # shouldn't abort the whole precompute batch -- skip this unit and continue.
        print(f'sub-{subjID:02d} {lockType}: FAILED, skipping unit: {e}', flush=True)
        return
    print(f'sub-{subjID:02d} {lockType}: done', flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Precompute per-ROI caches for G03/G04 whole-grid source data.')
    parser.add_argument('--subjects', nargs='+', type=int, default=SUBJECT_LIST)
    parser.add_argument('--lockTypes', nargs='+', default=['stim', 'resp'])
    parser.add_argument('--voxRes', default='8mm')
    parser.add_argument('--rois', nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--bands', nargs='+', default=list(AMP_ONLY_BANDS))
    parser.add_argument('--n_jobs', type=int, default=min(4, max(1, cpu_count() - 1)),
                         help='Parallel (subjID, lockType) workers. Default: min(4, ncpu-1) '
                              '-- kept low since each worker loads an 8-10GB array at once.')
    parser.add_argument('--force', action='store_true',
                         help='Rebuild caches even if they already exist.')
    args = parser.parse_args()

    bids_root = get_bids_root()
    n_jobs = max(1, args.n_jobs)
    print(f'precompute_roi_splits | subjects={args.subjects} | lockTypes={args.lockTypes} | '
          f'voxRes={args.voxRes} | rois={args.rois} | bands={args.bands} | n_jobs={n_jobs} | '
          f'force={args.force}')

    units = [(subjID, lockType) for subjID in args.subjects for lockType in args.lockTypes]

    Parallel(n_jobs=n_jobs, prefer='processes', verbose=5)(
        delayed(process_unit)(subjID, lockType, args.voxRes, bids_root,
                               list(args.rois), list(args.bands), args.force)
        for subjID, lockType in units
    )

    print('\nDone.')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
