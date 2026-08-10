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


def load_g03_unfiltered(subjID, lockType, voxRes, bids_root):
    """
    Loads derivatives/sub-{XX}/sourceRecon/sub-{XX}_task-mgs_sourceSpaceData_{res}_{lockType}.mat.
    Note this INCLUDES lockType in the filename, unlike the stale path builder
    in megScripts/temporalGeneralizationDecoding.py (which predates G03's
    lockType-parametrized rewrite).

    Returns:
        data:           (n_trials, n_times, n_sources) float32, raw voltage
        time_vector:    (n_times,) float64
        target_labels:  (n_trials,) int64, trialinfo column 2 (1-10)
        trialinfo_col2: same as target_labels (kept as its own key for align.py)
        inside_pos:     (n_sources,) int64, MATLAB 1-based full-template-grid indices
        fsample:        float
    """
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
