"""
constants.py

Shared constants and host-aware environment helpers for glue_decoding.

ANGLE_MAPPING is intentionally duplicated here rather than imported from
megScripts/temporalGeneralizationDecoding.py — megScripts has no __init__.py
and isn't set up as an importable package, and plotSourceDecodingWithBehavResults.py
already duplicates this same dict, so this matches the established repo convention.
"""

import os
import socket
import tempfile
from shutil import copyfile

import h5py

# Target-location label (1-10) -> degrees. Used everywhere angle targets are needed.
ANGLE_MAPPING = {1: 0, 2: 25, 3: 50, 4: 130, 5: 155,
                 6: 180, 7: 205, 8: 230, 9: 310, 10: 335}

# All 21 subjects processed by G01-G04 (matches SUBJ_LIST in run_G0*_vader.sh).
SUBJECT_LIST = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]

ROI_NAMES = ('visual', 'parietal', 'frontal')

# G04's canonical band table (amplitude for all 5; phase only for theta/alpha/beta).
# Band edges are read from each G04 .mat file itself at load time (ground truth) --
# this list is only used to enumerate which bands/conditions to run.
AMP_ONLY_BANDS = ('theta', 'alpha', 'beta', 'lowgamma', 'highgamma')
AMP_PHASE_BANDS = ('theta', 'alpha', 'beta')


def resolution_tag(voxRes):
    """'8mm' -> 8 (int), matching the numeric resolution tag G02/G03/G04 use in filenames."""
    return int(voxRes[:-2]) if voxRes.endswith('mm') else int(voxRes)


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


def open_h5(fpath, tmp_name):
    """
    Open an h5py file with the same host-aware copy/locking strategy as
    megScripts/temporalGeneralizationDecoding.py's open_h5(): on zod/vader,
    fall back to copying to a local scratch path if direct/locked access fails.
    """
    h = socket.gethostname()
    if h == 'zod':
        return _copy_and_open(fpath, '/Users/mrugank/Desktop', tmp_name)
    if h == 'vader':
        try:
            return h5py.File(fpath, 'r', locking=False)
        except Exception:
            return _copy_and_open(fpath, '/tmp', tmp_name)
    return h5py.File(fpath, 'r')
