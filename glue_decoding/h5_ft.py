"""
h5_ft.py

Small shared helpers for dereferencing FieldTrip-style structs saved by
MATLAB '-v7.3' (HDF5) into h5py, matching the dereferencing pattern already
used in megScripts/temporalGeneralizationDecoding.py's load_source_space_data()
(cell arrays of trials/time are stored as arrays of HDF5 object references
that must be dereferenced individually).
"""

import h5py
import numpy as np


def deref_cell_trials(f, struct_group):
    """
    struct_group['trial'] is a MATLAB cell array (1 x n_trials), each cell a
    (times x sources) matrix -- stored in HDF5 as an (n_trials, 1) array of
    object references. Returns (n_trials, times, sources) float32, or an
    empty (0,0,0) array if there are no trials.
    """
    td = struct_group['trial']
    n = td.shape[0]
    if n == 0:
        return np.zeros((0, 0, 0), dtype=np.float32)
    trials = [np.array(f[td[i, 0]]) for i in range(n)]
    return np.stack(trials, axis=0).astype(np.float32)


def deref_first_time(f, struct_group):
    """First trial's time vector, representative of all trials in this struct
    (epochs are uniform-length after G03/G04's resampling)."""
    td = struct_group['time']
    return np.array(f[td[0, 0]]).flatten()


def get_trialinfo(struct_group):
    """trialinfo stored as (n_cols, n_trials) in HDF5 (MATLAB column-major
    transpose) -> (n_trials, n_cols)."""
    return np.array(struct_group['trialinfo']).T


def get_scalar(struct_group, key):
    return float(np.array(struct_group[key]).flatten()[0])


def iter_target_pairs(f, amp_cell, phase_cell, target_locations):
    """
    Yields (target, amp_struct_or_None, phase_struct_or_None) for each target
    in target_locations, in order. amp_cell/phase_cell are the top-level
    ampDataByTarget/phaseDataByTarget h5py datasets of object references
    (phase_cell may be None if phase wasn't requested/saved). A None struct
    means that target had no trials (G04 leaves that cell empty rather than
    a FieldTrip struct).
    """
    amp_refs = np.asarray(amp_cell[()]).flatten()
    phase_refs = np.asarray(phase_cell[()]).flatten() if phase_cell is not None else None

    for i, target in enumerate(target_locations):
        amp_obj = f[amp_refs[i]]
        if not isinstance(amp_obj, h5py.Group) or 'trial' not in amp_obj:
            amp_obj = None

        phase_obj = None
        if phase_refs is not None and amp_obj is not None:
            candidate = f[phase_refs[i]]
            if isinstance(candidate, h5py.Group) and 'trial' in candidate:
                phase_obj = candidate

        yield target, amp_obj, phase_obj
