"""
io_g03.py

Loads G03_SourceLocalizationBroadband.m's output UNCHANGED -- no bandpass
filter, no Hilbert transform, no baseline correction, no downsampling. This
is the "unfiltered" condition: raw broadband source-space voltage per trial,
timepoint, and source.
"""

import os

import numpy as np

from constants import open_h5, resolution_tag
from h5_ft import deref_cell_trials, deref_first_time, get_trialinfo


def g03_roi_cache_path(subjID, lockType, voxRes, bids_root, roi):
    """Path to the precomputed per-ROI cache built by precompute_roi_splits.py."""
    subName = f'sub-{subjID:02d}'
    res = resolution_tag(voxRes)
    return os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon',
        f'{subName}_task-mgs_sourceSpaceData_{res}_{lockType}_roi-{roi}.npz')


def _load_g03_roi_cache(subjID, lockType, voxRes, bids_root, roi):
    """
    Fast path: load a small pre-sliced per-ROI cache instead of the full
    whole-grid file. Built by precompute_roi_splits.py -- raises with a
    pointer to that script if the cache hasn't been built yet.
    """
    fpath = g03_roi_cache_path(subjID, lockType, voxRes, bids_root, roi)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f'G03 ROI cache not found: {fpath}. Run '
            f'`python precompute_roi_splits.py` for sub-{subjID:02d} first.')
    with np.load(fpath) as npz:
        return {
            'data': npz['data'],
            'time_vector': npz['time_vector'],
            'target_labels': npz['target_labels'],
            'trialinfo_col2': npz['trialinfo_col2'],
            'inside_pos': npz['inside_pos'],
            'fsample': float(npz['fsample']),
        }


def load_g03_unfiltered(subjID, lockType, voxRes, bids_root, roi=None):
    """
    Loads derivatives/sub-{XX}/sourceRecon/sub-{XX}_task-mgs_sourceSpaceData_{res}_{lockType}.mat.
    Note this INCLUDES lockType in the filename, unlike the stale path builder
    in megScripts/temporalGeneralizationDecoding.py (which predates G03's
    lockType-parametrized rewrite).

    roi: None or 'whole' loads the full whole-grid file (default, unchanged
    behaviour). Any other ROI name (e.g. 'visual') loads the small
    precomputed per-ROI cache from precompute_roi_splits.py instead of the
    whole-grid file -- much cheaper when the caller only needs that ROI.

    Returns:
        data:           (n_trials, n_times, n_sources) float32, raw voltage
        time_vector:    (n_times,) float64
        target_labels:  (n_trials,) int64, trialinfo column 2 (1-10)
        trialinfo_col2: same as target_labels (kept as its own key for align.py)
        inside_pos:     (n_sources,) int64, MATLAB 1-based full-template-grid indices
                         (ROI-sliced, same order as `data`, if roi is given)
        fsample:        float
    """
    if roi is not None and roi != 'whole':
        return _load_g03_roi_cache(subjID, lockType, voxRes, bids_root, roi)

    subName = f'sub-{subjID:02d}'
    res = resolution_tag(voxRes)
    fpath = os.path.join(bids_root, 'derivatives', subName, 'sourceRecon',
                          f'{subName}_task-mgs_sourceSpaceData_{res}_{lockType}.mat')
    if not os.path.exists(fpath):
        raise FileNotFoundError(f'G03 output not found: {fpath}')

    f = open_h5(fpath, os.path.basename(fpath))
    try:
        sg = f['sourcedataCombined']
        data = deref_cell_trials(f, sg)
        time_vector = deref_first_time(f, sg)
        trialinfo = get_trialinfo(sg)  # (n_trials, 5)
        fsample = float(np.array(sg['fsample']).flatten()[0])
        inside_pos = np.array(f['inside_pos']).flatten().astype(np.int64)
    finally:
        f.close()

    target_labels = trialinfo[:, 1].astype(np.int64)
    assert data.shape[0] == target_labels.shape[0], (
        f'{fpath}: n_trials mismatch between data ({data.shape[0]}) and trialinfo '
        f'({target_labels.shape[0]})')
    assert data.shape[1] == time_vector.shape[0], (
        f'{fpath}: n_times mismatch between data ({data.shape[1]}) and time_vector '
        f'({time_vector.shape[0]})')

    return {
        'data': data,
        'time_vector': time_vector,
        'target_labels': target_labels,
        'trialinfo_col2': trialinfo[:, 1],
        'inside_pos': inside_pos,
        'fsample': fsample,
    }
