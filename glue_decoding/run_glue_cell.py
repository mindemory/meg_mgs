#!/usr/bin/env python3
"""
run_glue_cell.py

CLI entry point for glue_decoding: runs ALL (ROI x condition x band)
combinations for ONE (subject, lockType) unit. Meant to be launched as one
background job per subject x lockType by run_glue_decoding.sh (42 jobs total
across 21 subjects x 2 lock types), which also exports
OMP/MKL/OPENBLAS_NUM_THREADS=1 in the shell -- this script sets them again
here (belt-and-suspenders) BEFORE importing numpy/sklearn, since those
libraries read the env vars for BLAS threading at import time, not at call
time. This one-process-per-subject-per-lockType job is the ONLY level of
parallelism glue_decoding uses; everything inside is single-threaded
(run_tgm is always called with n_jobs=1 -- see svr_tgm.py).

Usage:
    python run_glue_cell.py <subjID> <lockType> [--voxRes 8mm]
                             [--rois visual parietal frontal]
                             [--conditions unfiltered ampOnly ampPhase]
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import pickle
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from align import attach_behav, g04_orig_row_index, load_behav, verify_alignment
from constants import AMP_ONLY_BANDS, AMP_PHASE_BANDS, ROI_NAMES, get_bids_root
from features import build_features
from io_g03 import load_g03_unfiltered
from io_g04 import load_g04_band
from svr_tgm import run_tgm


def _output_path(bids_root, subjID, condition, band, lockType, voxRes):
    subName = f'sub-{subjID:02d}'
    out_dir = os.path.join(bids_root, 'derivatives', subName, 'sourceRecon', 'decodingGlue')
    os.makedirs(out_dir, exist_ok=True)
    fname = f'{subName}_task-mgs_glueTGM_{condition}_{band}_{lockType}_{voxRes}.pkl'
    return os.path.join(out_dir, fname)


def _behav_for_rows(orig_row_idx, behav, aligned):
    if not aligned or behav is None:
        n = len(orig_row_idx) if orig_row_idx is not None else 0
        return {'i_sacc_err': np.full(n, np.nan), 'i_sacc_angle': np.full(n, np.nan)}
    return attach_behav(orig_row_idx, behav)


def _decode_and_save(out_path, condition, band, lockType, voxRes, subjID,
                      data_by_roi, target_labels, time_vector, behav_rows, behav_aligned,
                      freq_range):
    output = {
        'time_vector': time_vector,
        'target_labels': target_labels,
        'i_sacc_err': behav_rows['i_sacc_err'],
        'i_sacc_angle': behav_rows['i_sacc_angle'],
        'behav_aligned': behav_aligned,
        'condition': condition,
        'band': band,
        'freq_range': freq_range,
        'lockType': lockType,
        'voxRes': voxRes,
        'subjID': subjID,
    }
    for roi_name, roi_data in data_by_roi.items():
        print(f'    TGM: {roi_name} ({roi_data.shape[2]} features, {roi_data.shape[0]} trials) ...')
        output[f'pred_angles_deg_{roi_name}'] = run_tgm(roi_data, target_labels, n_jobs=1)

    with open(out_path, 'wb') as fh:
        pickle.dump(output, fh)
    print(f'  Saved: {out_path}')


def _g03_for_roi(subjID, lockType, voxRes, bids_root, roi_name, g03_whole):
    """
    Returns (data, trialinfo_col2, time_vector) for one ROI.

    For 'whole' this uses the already-loaded whole-grid g03_whole (unchanged
    behaviour). For any other ROI it loads the small precomputed per-ROI
    cache directly (io_g03.load_g03_unfiltered's roi= fast path) instead of
    touching the whole-grid array at all.
    """
    if roi_name == 'whole':
        return g03_whole['data'], g03_whole['trialinfo_col2'], g03_whole['time_vector']
    g03_roi = load_g03_unfiltered(subjID, lockType, voxRes, bids_root, roi=roi_name)
    return g03_roi['data'], g03_roi['trialinfo_col2'], g03_roi['time_vector']


def run_unfiltered(subjID, lockType, voxRes, bids_root, rois, behav, g03_whole=None):
    band, condition = 'broadband', 'unfiltered'
    out_path = _output_path(bids_root, subjID, condition, band, lockType, voxRes)
    if os.path.exists(out_path):
        print(f'SKIP (exists): {out_path}')
        return

    data_by_roi = {}
    trialinfo_col2 = time_vector = None
    for roi_name in rois:
        data, trialinfo_col2, time_vector = _g03_for_roi(
            subjID, lockType, voxRes, bids_root, roi_name, g03_whole)
        data_by_roi[roi_name] = build_features('unfiltered', data)

    aligned = behav is not None and verify_alignment(trialinfo_col2, behav['tarlocCode'])
    if behav is not None and not aligned:
        print(f'  WARNING: behav alignment check FAILED for sub-{subjID:02d} {lockType} '
              f'(unfiltered) -- proceeding without behavioral attachment.')
    # Unfiltered rows are already in original sourcedataCombined order (no target regrouping).
    orig_row_idx = np.arange(next(iter(data_by_roi.values())).shape[0])
    behav_rows = _behav_for_rows(orig_row_idx, behav, aligned)

    target_labels = trialinfo_col2.astype(np.int64)
    _decode_and_save(out_path, condition, band, lockType, voxRes, subjID,
                      data_by_roi, target_labels, time_vector,
                      behav_rows, aligned, freq_range=None)


def _g04_for_roi(subjID, lockType, band, voxRes, bids_root, roi_name, want_phase, g04_whole):
    """
    Returns (amp, phase, target_labels, time_vector, freq_range, actualRate) for one ROI.

    For 'whole' this uses the already-loaded whole-grid g04_whole. For any
    other ROI it loads the small precomputed per-ROI cache directly instead
    of touching the whole-grid array at all.
    """
    if roi_name == 'whole':
        return (g04_whole['amp'], g04_whole['phase'], g04_whole['target_labels'],
                g04_whole['time_vector'], g04_whole['freq_range'], g04_whole['actualRate'])
    g04_roi = load_g04_band(subjID, lockType, band, voxRes, bids_root,
                             want_phase=want_phase, roi=roi_name)
    return (g04_roi['amp'], g04_roi['phase'], g04_roi['target_labels'], g04_roi['time_vector'],
            g04_roi['freq_range'], g04_roi['actualRate'])


def run_g04_condition(subjID, lockType, voxRes, bids_root, rois, behav,
                       condition, bands, g03_trialinfo_col2):
    want_phase = condition == 'ampPhase'
    for band in bands:
        out_path = _output_path(bids_root, subjID, condition, band, lockType, voxRes)
        if os.path.exists(out_path):
            print(f'SKIP (exists): {out_path}')
            continue

        g04_whole = None
        if 'whole' in rois:
            try:
                g04_whole = load_g04_band(subjID, lockType, band, voxRes, bids_root,
                                           want_phase=want_phase)
            except (FileNotFoundError, ValueError) as e:
                print(f'  SKIP {condition}/{band}: {e}')
                continue

        data_by_roi = {}
        target_labels = time_vector = freq_range = None
        try:
            for roi_name in rois:
                amp_roi, phase_roi, target_labels, time_vector, freq_range, _ = _g04_for_roi(
                    subjID, lockType, band, voxRes, bids_root, roi_name, want_phase, g04_whole)
                data_by_roi[roi_name] = build_features(condition, amp_roi, phase_roi)
        except (FileNotFoundError, ValueError) as e:
            print(f'  SKIP {condition}/{band}: {e}')
            continue

        orig_row_idx = g04_orig_row_index(g03_trialinfo_col2)
        aligned = (behav is not None
                   and orig_row_idx.shape[0] == target_labels.shape[0]
                   and verify_alignment(g03_trialinfo_col2, behav['tarlocCode']))
        if behav is not None and not aligned:
            print(f'  WARNING: behav alignment check FAILED for sub-{subjID:02d} {lockType} '
                  f'{condition}/{band} -- proceeding without behavioral attachment.')
        behav_rows = _behav_for_rows(orig_row_idx, behav, aligned)

        _decode_and_save(out_path, condition, band, lockType, voxRes, subjID,
                          data_by_roi, target_labels, time_vector,
                          behav_rows, aligned, freq_range=freq_range)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('subjID', type=int)
    parser.add_argument('lockType', choices=['stim', 'resp'])
    parser.add_argument('--voxRes', default='8mm')
    parser.add_argument('--rois', nargs='+', default=list(ROI_NAMES))
    parser.add_argument('--conditions', nargs='+', default=['unfiltered', 'ampOnly', 'ampPhase'])
    args = parser.parse_args()

    bids_root = get_bids_root()
    print(f'glue_decoding | sub-{args.subjID:02d} | {args.lockType} | {args.voxRes} | '
          f'conditions={args.conditions} | rois={args.rois}')

    behav = load_behav(args.subjID, bids_root)
    if behav is None:
        print(f'  NOTE: no behavioral file found for sub-{args.subjID:02d} -- '
              f'i_sacc_err/i_sacc_angle will be NaN, behav_aligned=False everywhere.')

    # Only load the whole-grid G03 file when 'whole' is actually requested -- for
    # ROI-only runs (the default: visual/parietal/frontal), each condition loads
    # the small precomputed per-ROI caches directly instead (see io_g03.py's
    # roi= fast path), skipping the 8-10GB whole-grid array entirely.
    need_whole = 'whole' in args.rois
    g03_whole = load_g03_unfiltered(args.subjID, args.lockType, args.voxRes, bids_root) \
        if need_whole else None

    # trialinfo_col2 (per-trial metadata, identical across ROI caches -- only the
    # source axis differs) is needed for G04's row alignment regardless of which
    # ROIs are requested; grab it from whichever G03 load is cheapest.
    g03_trialinfo_col2 = (g03_whole['trialinfo_col2'] if need_whole else
                           load_g03_unfiltered(args.subjID, args.lockType, args.voxRes,
                                                bids_root, roi=args.rois[0])['trialinfo_col2'])

    if 'unfiltered' in args.conditions:
        run_unfiltered(args.subjID, args.lockType, args.voxRes, bids_root,
                        args.rois, behav, g03_whole=g03_whole)

    if 'ampOnly' in args.conditions:
        run_g04_condition(args.subjID, args.lockType, args.voxRes, bids_root,
                           args.rois, behav, 'ampOnly', AMP_ONLY_BANDS, g03_trialinfo_col2)

    if 'ampPhase' in args.conditions:
        run_g04_condition(args.subjID, args.lockType, args.voxRes, bids_root,
                           args.rois, behav, 'ampPhase', AMP_PHASE_BANDS, g03_trialinfo_col2)

    print(f'Done | sub-{args.subjID:02d} | {args.lockType}')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
