"""
constants.py

Shared constants and host-aware environment helpers for glue_decoding.

ANGLE_MAPPING is intentionally duplicated here rather than imported from
megScripts/temporalGeneralizationDecoding.py — megScripts has no __init__.py
and isn't set up as an importable package, and plotSourceDecodingWithBehavResults.py
already duplicates this same dict, so this matches the established repo convention.
"""

import os
import random
import socket
import tempfile
import time
from shutil import copyfile

import h5py
import numpy as np

# Retries for transient NFS read failures (soft-mount short reads / stale
# filehandles under concurrent load) surfacing as h5py "file signature not
# found" -- seen on vader when 8 parallel workers hit the NFS mount at once,
# even for direct locking=False reads of an otherwise-valid file.
_OPEN_RETRIES = 5
_OPEN_BACKOFF_S = 0.5

# Target-location label (1-10) -> degrees. Used everywhere angle targets are needed.
ANGLE_MAPPING = {1: 0, 2: 25, 3: 50, 4: 130, 5: 155,
                 6: 180, 7: 205, 8: 230, 9: 310, 10: 335}

# All 21 subjects processed by G01-G04 (matches SUBJ_LIST in run_G0*_vader.sh).
SUBJECT_LIST = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]

ROI_NAMES = ('visual', 'parietal', 'frontal')

# Per-hemisphere ROI splits (see atlas.py's MASK_KEYS) -- deliberately kept
# OUT of ROI_NAMES so nothing that iterates the default ROI set (precompute
# caches, plot_timeseries.py, etc.) starts requiring these caches to exist.
# Pass e.g. `--rois visual_left visual_right` explicitly (to
# precompute_roi_splits.py first, then any consumer) to opt in.
HEMI_ROI_NAMES = ('visual_left', 'visual_right', 'parietal_left', 'parietal_right',
                   'frontal_left', 'frontal_right')

# G04's canonical band table (amplitude for all 5; phase only for theta/alpha/beta).
# Band edges are read from each G04 .mat file itself at load time (ground truth) --
# this list is only used to enumerate which bands/conditions to run.
AMP_ONLY_BANDS = ('theta', 'alpha', 'beta', 'lowgamma', 'highgamma')
AMP_PHASE_BANDS = ('theta', 'alpha', 'beta')

# ── Category-grouping schemes ────────────────────────────────────────────────
#
# Coarser-than-10-location groupings for separability tests
# (linear_decoding_categories_cell.py, representational_distance_ts_cell.py's
# --schemes). Motivation: the decoders actually used elsewhere in this repo
# (svr_tgm.py, decoding_ts_cell.py) do continuous circular regression --
# sin/cos of angle -- and never treat the 10 locations as discrete unordered
# categories, so it was unclear whether raw location-level responses are even
# linearly separable as such, or whether that framing (also GLUE's
# ONE_VERSUS_REST assumption in manifold_capacity.py) is simply too
# fine-grained for a smooth circular code. These let both scripts test
# separability at several granularities before assuming 10-way is the right
# unit of analysis.
#
# Angles from ANGLE_MAPPING: {1:0, 2:25, 3:50, 4:130, 5:155, 6:180, 7:205,
#                              8:230, 9:310, 10:335}
CATEGORY_SCHEMES = {
    2: {   # left vs right hemifield (cos(angle) sign)
        'name': 'left_right',
        'groups': {
            'right': (1, 2, 3, 9, 10),    # angles in (-90, 90):  0, 25, 50, 310, 335
            'left':  (4, 5, 6, 7, 8),     # angles in (90, 270):  130, 155, 180, 205, 230
        },
    },
    4: {   # four quadrants -- excludes the two axis-aligned locations (0 deg, 180 deg),
           # which sit exactly on the quadrant boundary and don't belong to any one quadrant
        'name': 'quadrant4',
        'groups': {
            'Q1_upper_right': (2, 3),     # 25, 50
            'Q2_upper_left':  (4, 5),     # 130, 155
            'Q3_lower_left':  (7, 8),     # 205, 230
            'Q4_lower_right': (9, 10),    # 310, 335
        },
    },
    6: {   # four quadrants + the two axis locations as their own singleton categories
           # (same quadrant groups as scheme 4, but nothing is excluded)
        'name': 'quadrant6',
        'groups': {
            'Q1_upper_right': (2, 3),
            'Q2_upper_left':  (4, 5),
            'Q3_lower_left':  (7, 8),
            'Q4_lower_right': (9, 10),
            'right_axis':     (1,),       # 0 deg
            'left_axis':      (6,),       # 180 deg
        },
    },
    10: {  # every raw location its own category (no grouping) -- current
           # manifold_capacity.py / decoding_ts_cell.py granularity
        'name': 'location10',
        'groups': {str(loc): (loc,) for loc in sorted(ANGLE_MAPPING)},
    },
}


def category_labels_for_scheme(target_labels, scheme):
    """
    Maps raw 1-10 target_labels (any int array) to this scheme's group-name
    labels.

    Returns (group_labels, keep_mask):
      keep_mask   : (n_trials,) bool -- False for trials whose raw location
                    isn't assigned to any group in this scheme (e.g. scheme=4
                    drops locations 1 and 6).
      group_labels: (keep_mask.sum(),) string array, group_labels[i]
                    corresponds to target_labels[keep_mask][i].
    """
    if scheme not in CATEGORY_SCHEMES:
        raise ValueError(f'Unknown scheme {scheme!r}, expected one of {sorted(CATEGORY_SCHEMES)}')
    groups = CATEGORY_SCHEMES[scheme]['groups']
    loc_to_group = {loc: name for name, locs in groups.items() for loc in locs}

    target_labels = np.asarray(target_labels).astype(int)
    keep_mask = np.array([loc in loc_to_group for loc in target_labels])
    group_labels = np.array([loc_to_group[loc] for loc in target_labels[keep_mask]])
    return group_labels, keep_mask


def balance_categories(group_labels, points_per_category, seed=0):
    """
    Randomly subsamples (without replacement, fixed seed) each category down
    to exactly points_per_category points, so every category -- and every
    scheme, if called with the same points_per_category -- contributes the
    same amount of data. This is what makes cross-scheme (2 vs 4 vs 6 vs 10
    category) comparisons apples-to-apples rather than confounded by
    different per-category sample sizes.

    Returns a bool mask (same length as group_labels) selecting the kept
    points. Raises ValueError if any category has fewer than
    points_per_category points (caller should catch this and skip/log --
    a genuinely too-small category, not a bug).
    """
    rng = np.random.default_rng(seed)
    keep = np.zeros(group_labels.shape[0], dtype=bool)
    for g in np.unique(group_labels):
        idx = np.where(group_labels == g)[0]
        if idx.size < points_per_category:
            raise ValueError(
                f'category {g!r} has only {idx.size} points, need >= {points_per_category}')
        chosen = rng.choice(idx, size=points_per_category, replace=False)
        keep[chosen] = True
    return keep


def resolution_tag(voxRes):
    """'8mm' -> 8 (int), matching the numeric resolution tag G02/G03/G04 use in filenames."""
    return int(voxRes[:-2]) if voxRes.endswith('mm') else int(voxRes)


def glue_fits_csv_path(bids_root, subjID, lockType, voxRes, outdir=None):
    """
    Path for manifold_capacity.py's per-subject glue-capacity results CSV
    (derivatives/sub-XX/sourceRecon/glueFits/, matching run_glue_cell.py's
    decodingGlue / decoding_ts_cell.py's decodingTS per-subject layout).

    Shared here (rather than living in manifold_capacity.py itself) so
    aggregate_glue_capacity.py can build the exact same path without
    importing manifold_capacity.py, which would trigger its module-level
    `glue` package import check even though aggregation only needs pandas.
    """
    subName = f'sub-{subjID:02d}'
    out_dir = outdir if outdir else os.path.join(
        bids_root, 'derivatives', subName, 'sourceRecon', 'glueFits')
    fname = f'{subName}_task-mgs_glueFits_{lockType}_{voxRes}.csv'
    return os.path.join(out_dir, fname)


def get_bids_root():
    """Host-aware BIDS root, matching megScripts/temporalGeneralizationDecoding.py."""
    h = socket.gethostname()
    if h == 'zod':
        return '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    if h == 'vader':
        return '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    return '/scratch/mdd9787/meg_prf_greene/MEG_HPC'


def _copy_and_open(fpath, tmp_dir, tmp_name):
    """
    Copy fpath to a process/call-unique path under tmp_dir and open it with
    h5py. Using a unique name (rather than just tmp_name, which collides
    whenever two parallel workers load the same file basename) avoids one
    worker's copyfile()/os.remove() truncating or deleting another's
    in-flight copy, which otherwise surfaces as h5py's
    "file signature not found" on the corrupted/half-written copy.
    Cleanup happens in `finally` so a failed h5py.File() open doesn't leak
    the temp copy.
    """
    fd, tmp = tempfile.mkstemp(prefix=f'{os.getpid()}_', suffix=f'_{tmp_name}', dir=tmp_dir)
    os.close(fd)
    try:
        copyfile(fpath, tmp)
        return h5py.File(tmp, 'r')
    finally:
        os.remove(tmp)


def _open_h5_once(fpath, tmp_name):
    """One attempt at the host-aware copy/locking strategy, no retries."""
    h = socket.gethostname()
    if h == 'zod':
        return _copy_and_open(fpath, '/Users/mrugank/Desktop', tmp_name)
    if h == 'vader':
        try:
            return h5py.File(fpath, 'r', locking=False)
        except Exception:
            return _copy_and_open(fpath, '/tmp', tmp_name)
    return h5py.File(fpath, 'r')


def open_h5(fpath, tmp_name):
    """
    Open an h5py file with the same host-aware copy/locking strategy as
    megScripts/temporalGeneralizationDecoding.py's open_h5(): on zod/vader,
    fall back to copying to a local scratch path if direct/locked access fails.

    Retries with jittered backoff on OSError, since both the direct
    locking=False path and the copy fallback can transiently fail under
    concurrent NFS load (short reads / stale filehandles), not just when the
    file is genuinely missing or corrupt.
    """
    last_err = None
    for attempt in range(_OPEN_RETRIES):
        try:
            return _open_h5_once(fpath, tmp_name)
        except OSError as e:
            last_err = e
            if attempt < _OPEN_RETRIES - 1:
                time.sleep(_OPEN_BACKOFF_S * (attempt + 1) + random.uniform(0, 0.5))
    raise OSError(
        f'Failed to open {fpath!r} after {_OPEN_RETRIES} attempts (last error: {last_err})'
    ) from last_err
