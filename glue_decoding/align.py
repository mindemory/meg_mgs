"""
align.py

Behavioral (eyetracking) alignment for glue_decoding.

sourcedataCombined.trialinfo has 5 columns (tarloc, tarlocCode, x, y, delayFlag)
-- no trial/run ID. ii_sess_forSource (built once from the STIM-locked
NaN-removal pattern by S02B_organizeBehavForSource.m) is positionally aligned
to a subject's sourcedataCombined ONLY if that alignment is verified -- it is
guaranteed by construction for lockType='stim' (S02B self-checks this via a
tarlocCode sum-of-differences check), but NOT guaranteed for lockType='resp',
since G03 runs an independent per-lockType NaN removal. We verify rather than
assume, for both lock types, and skip behavioral attachment (rather than
silently mis-join) if verification fails.

G04's per-target regrouping (ampDataByTarget{target} = trials where
trialinfo(:,2)==target, for target=1..10, concatenated in order) is fully
deterministic given the same G03 trialinfo, so we can recompute the exact same
per-target selection in Python and recover each G04 output row's original row
index into sourcedataCombined (and therefore into ii_sess_forSource, once
verify_alignment passes).
"""

import os

import numpy as np

from constants import open_h5

# Which behavioural stream the analyses read. Must match S02B's BEHAV_DIR,
# since that is the script that writes the _forSource file read below.
#
# 'eyetracking_old' is the stream produced WITHOUT run_iipreproc's step-14
# post-hoc calibration. That step warps trial gaze toward the KNOWN target and
# i_sacc_err is then measured against that same target, so on trials where the
# subject went straight to the target and stayed, the error collapses toward 0
# by construction rather than being measured. Those trials then fall under
# I_SACC_ERR_THRESH and are dropped as "missing saccade" -- and they are the
# most accurate trials, so the best performance bin loses exactly what belongs
# in it. Measured across all 21 subjects (inspect_behaviour.py --compare_calib):
# the calibration creates 620 such trials, cutting usable trials 4637 -> 4016;
# reading this stream instead recovers 621 (+15%). No subject has more usable
# trials with the calibration than without, so it is applied uniformly.
BEHAV_DIR = 'eyetracking_old'


def load_behav(subjID, bids_root, behav_dir=BEHAV_DIR):
    """
    Loads derivatives/sub-{XX}/{behav_dir}/sub-{XX}_task-mgs-iisess_forSource.mat
    -> {'tarlocCode': (n,), 'i_sacc_err': (n,), 'i_sacc_angle': (n,)}.
    Returns None if the file doesn't exist (e.g. behavioral data not organized
    for this subject yet -- S02B must have been run with the matching BEHAV_DIR).
    """
    subName = f'sub-{subjID:02d}'
    behav_path = os.path.join(bids_root, 'derivatives', subName, behav_dir,
                               f'{subName}_task-mgs-iisess_forSource.mat')
    if not os.path.exists(behav_path):
        return None

    f = open_h5(behav_path, os.path.basename(behav_path))
    try:
        grp = f['ii_sess_forSource']
        tarlocCode = np.array(grp['tarlocCode']).flatten()
        i_sacc_err = np.array(grp['i_sacc_err']).flatten()
        i_sacc_raw = np.array(grp['i_sacc_raw'])
    finally:
        f.close()

    # Saccade angle from (x,y), matching megScripts/temporalGeneralizationDecoding.py's main().
    # MATLAB stores i_sacc_raw as (n_trials, 2); HDF5/h5py transposes to (2, n_trials).
    # Use i_sacc_err (already correctly flattened) to know n_trials, then orient raw.
    n_trials = len(i_sacc_err)
    raw = i_sacc_raw
    print(f'  [align] sub-{subjID:02d}: i_sacc_raw.shape={raw.shape}, n_trials={n_trials}',
          flush=True)
    if raw.ndim == 1:
        # Edge case: single trial stored as flat [x, y] by MATLAB squeeze
        raw = raw.reshape(2, 1)
    elif raw.shape[1] == 2 and raw.shape[0] == n_trials:
        # (n_trials, 2) -- not transposed (e.g. loaded without HDF5 flip)
        raw = raw.T
    # else: already (2, n_trials) -- use as-is
    # Sanity check
    if raw.shape != (2, n_trials):
        print(f'  [align] WARNING sub-{subjID:02d}: i_sacc_raw shape mismatch '
              f'after reshape: got {raw.shape}, expected (2, {n_trials}). '
              f'i_sacc_angle will be NaN for this subject.', flush=True)
        i_sacc_angle = np.full(n_trials, np.nan)
    else:
        x_c = raw[0]
        y_c = raw[1]
        ref_angle = np.arctan2(0, 5)
        i_sacc_angle = (np.degrees(np.arctan2(y_c, x_c) - ref_angle) + 360) % 360

    return {'tarlocCode': tarlocCode, 'i_sacc_err': i_sacc_err, 'i_sacc_angle': i_sacc_angle}



def verify_alignment(g03_trialinfo_col2, behav_tarlocCode):
    """
    Mirrors S02B_organizeBehavForSource.m's own self-check:
        sum(sourcedataCombined.trialinfo(:,2) - ii_sess_forSource.tarlocCode) == 0
    Requires equal length AND zero (nan-omitted) sum of differences. Returns
    False (never raises) so callers can gate behavioral attachment cleanly.
    """
    a = np.asarray(g03_trialinfo_col2).flatten()
    b = np.asarray(behav_tarlocCode).flatten()
    if a.shape[0] != b.shape[0] or a.shape[0] == 0:
        return False
    return bool(np.nansum(a - b) == 0)


def g04_orig_row_index(g03_trialinfo_col2, target_locations=range(1, 11)):
    """
    Recomputes G04_BandAmplitudePhaseInSource.m's per-target trial selection
    (`valid_trials = find(sourcedataCombined.trialinfo(:,2) == target)` for
    target = 1..10, concatenated in that order) against the SAME G03 trialinfo
    G04 used. Returns a 0-based index array: row k of this array is the
    original sourcedataCombined row that ends up as row k of the concatenated
    G04 amplitude/phase output. Targets with no trials contribute nothing
    (matches G04's `continue` on empty target groups).
    """
    col2 = np.asarray(g03_trialinfo_col2).flatten()
    idx_per_target = [np.where(col2 == target)[0] for target in target_locations]
    idx_per_target = [idx for idx in idx_per_target if idx.size > 0]
    if not idx_per_target:
        return np.array([], dtype=np.int64)
    return np.concatenate(idx_per_target).astype(np.int64)


def attach_behav(orig_row_idx, behav):
    """
    Slices behav's per-trial fields at orig_row_idx (see g04_orig_row_index /
    identity for unfiltered), producing arrays row-aligned with the decoding
    matrix. `orig_row_idx` values must be valid row indices into `behav`'s
    arrays (i.e. verify_alignment must have already passed for this
    subject/lockType).
    """
    orig_row_idx = np.asarray(orig_row_idx).astype(np.int64)
    return {
        'i_sacc_err': behav['i_sacc_err'][orig_row_idx],
        'i_sacc_angle': behav['i_sacc_angle'][orig_row_idx],
    }
