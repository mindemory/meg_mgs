"""
io_g04.py

Loads G04_BandAmplitudePhaseInSource.m's per-band output UNCHANGED -- no
re-filtering, no re-computing the Hilbert transform. G04 already did that;
this module only concatenates its 10 per-target-location cells
(ampDataByTarget / phaseDataByTarget) into one (trials, times, sources)
matrix each, in target order 1..10 (empty targets skipped), which is exactly
the row order align.g04_orig_row_index() reconstructs independently from
the corresponding G03 file's trialinfo.

Note: G04's per-target FieldTrip structs do NOT carry inside_pos (it's a
sibling top-level variable in G03's output, not a field of the
sourcedataCombined struct G04 operates on) -- callers must reuse the
inside_pos loaded from the matching G03 file for this same
subject/lockType/voxRes; G04's source columns are the same columns, same
order, just band-filtered.
"""

import os

import numpy as np

from constants import open_h5, resolution_tag
from h5_ft import deref_cell_trials, deref_first_time, get_scalar, get_trialinfo, iter_target_pairs


def g04_roi_cache_path(subjID, lockType, band, voxRes, bids_root, roi):
    """Path to the precomputed per-ROI cache built by precompute_roi_splits.py."""
    subName = f'sub-{subjID:02d}'
    res = resolution_tag(voxRes)
    return os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'freqSpace',
        f'{subName}_task-mgs_{band}_allTargets_{res}_{lockType}_roi-{roi}.npz')


def _load_g04_roi_cache(subjID, lockType, band, voxRes, bids_root, roi, want_phase):
    """
    Fast path: load a small pre-sliced per-ROI cache instead of the full
    whole-grid file. Built by precompute_roi_splits.py -- raises with a
    pointer to that script if the cache hasn't been built yet.
    """
    fpath = g04_roi_cache_path(subjID, lockType, band, voxRes, bids_root, roi)
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f'G04 ROI cache not found: {fpath}. Run '
            f'`python precompute_roi_splits.py` for sub-{subjID:02d} first.')
    with np.load(fpath) as npz:
        if want_phase and 'phase' not in npz:
            raise ValueError(f"'{band}' has no saved phase data (cache: {fpath})")
        return {
            'amp': npz['amp'],
            'phase': npz['phase'] if (want_phase and 'phase' in npz) else None,
            'time_vector': npz['time_vector'],
            'target_labels': npz['target_labels'],
            'trialinfo_col2': npz['trialinfo_col2'],
            'actualRate': float(npz['actualRate']),
            'freq_range': tuple(npz['freq_range'].tolist()),
            # TEMPLATE-GRID index of each ROI source, in column order (written by
            # precompute_roi_splits._save_g04_roi_splits as inside_pos_full[idx]).
            # This is what makes source identity comparable ACROSS subjects: column
            # k of subject A and column k of subject B are only the same anatomical
            # location if these agree. None for caches built before it was saved.
            'inside_pos': npz['inside_pos'] if 'inside_pos' in npz else None,
        }


def load_g04_band(subjID, lockType, band, voxRes, bids_root, want_phase, roi=None):
    """
    Loads derivatives/sub-{XX}/sourceRecon/freqSpace/
        sub-{XX}_task-mgs_{band}_allTargets_{res}_{lockType}.mat

    roi: None or 'whole' loads the full whole-grid file (default, unchanged
    behaviour). Any other ROI name (e.g. 'visual') loads the small
    precomputed per-ROI cache from precompute_roi_splits.py instead of the
    whole-grid file -- much cheaper when the caller only needs that ROI.

    Returns:
        amp:            (n_trials, n_times, n_sources) float32
        phase:          (n_trials, n_times, n_sources) float32, or None if not requested
        time_vector:    (n_times,) float64
        target_labels:  (n_trials,) int64
        trialinfo_col2: same as target_labels
        actualRate:     float, G04's shared post-Hilbert storage rate
        freq_range:     (f_min, f_max) as saved in the file (NOT a hardcoded
                         Python dict -- G04's actual band edges differ
                         slightly from megScripts' old FREQUENCY_BANDS dict)

    Raises ValueError if want_phase=True but this band has no saved phase
    (lowgamma/highgamma).
    """
    if roi is not None and roi != 'whole':
        return _load_g04_roi_cache(subjID, lockType, band, voxRes, bids_root, roi, want_phase)

    subName = f'sub-{subjID:02d}'
    res = resolution_tag(voxRes)
    fpath = os.path.join(bids_root, 'derivatives', subName, 'sourceRecon', 'freqSpace',
                          f'{subName}_task-mgs_{band}_allTargets_{res}_{lockType}.mat')
    if not os.path.exists(fpath):
        raise FileNotFoundError(f'G04 output not found: {fpath}')

    f = open_h5(fpath, os.path.basename(fpath))
    try:
        if want_phase and 'phaseDataByTarget' not in f:
            raise ValueError(f"'{band}' has no saved phase data (file: {fpath})")

        target_locations = np.array(f['target_locations']).flatten().astype(np.int64).tolist()
        phase_cell = f['phaseDataByTarget'] if want_phase else None

        amp_chunks, phase_chunks, trialinfo_chunks = [], [], []
        time_vector, n_sources, actual_rate = None, None, None

        for target, amp_struct, phase_struct in iter_target_pairs(
                f, f['ampDataByTarget'], phase_cell, target_locations):
            if amp_struct is None:
                continue
            amp_trials = deref_cell_trials(f, amp_struct)
            if amp_trials.shape[0] == 0:
                continue

            if time_vector is None:
                time_vector = deref_first_time(f, amp_struct)
                n_sources = amp_trials.shape[2]
                actual_rate = get_scalar(amp_struct, 'fsample')
            else:
                assert amp_trials.shape[2] == n_sources, (
                    f'{fpath}: source count mismatch at target {target} '
                    f'({amp_trials.shape[2]} vs {n_sources})')
                assert amp_trials.shape[1] == time_vector.shape[0], (
                    f'{fpath}: time axis length mismatch at target {target}')

            amp_chunks.append(amp_trials)
            trialinfo_chunks.append(get_trialinfo(amp_struct))

            if want_phase:
                if phase_struct is None:
                    raise ValueError(f'{fpath}: missing phase struct for target {target}')
                phase_trials = deref_cell_trials(f, phase_struct)
                assert phase_trials.shape == amp_trials.shape, (
                    f'{fpath}: amplitude/phase shape mismatch at target {target}')
                phase_chunks.append(phase_trials)

        freq_range = tuple(np.array(f['freq_range']).flatten().tolist())
    finally:
        f.close()

    if not amp_chunks:
        raise ValueError(f'{fpath}: no trials found for any target location')

    amp = np.concatenate(amp_chunks, axis=0).astype(np.float32)
    phase = np.concatenate(phase_chunks, axis=0).astype(np.float32) if (want_phase and phase_chunks) else None
    trialinfo = np.concatenate(trialinfo_chunks, axis=0)
    target_labels = trialinfo[:, 1].astype(np.int64)

    return {
        'amp': amp,
        'phase': phase,
        'time_vector': time_vector,
        'target_labels': target_labels,
        'trialinfo_col2': trialinfo[:, 1],
        'actualRate': actual_rate,
        'freq_range': freq_range,
    }
