"""
atlas.py

Loads the Wang-atlas-derived visual/parietal/frontal ROI masks saved by
G02_WangAtlasParcellation.m (derivatives/atlas/rois_{voxRes}.mat) and slices
them down to a given subject's LOCAL source columns via that subject's own
inside_pos.
"""

import os

import numpy as np
from scipy.io import loadmat

from constants import open_h5

MASK_KEYS = {'visual': 'visual_points', 'parietal': 'parietal_points', 'frontal': 'frontal_points'}


def _load_mat_flexible(fpath):
    """
    Load a .mat file, trying scipy.io.loadmat first (MATLAB v7 and earlier)
    and falling back to h5py (v7.3 / HDF5-based). G02_WangAtlasParcellation.m
    saves with '-v7.3', which scipy.io.loadmat cannot read (raises
    NotImplementedError) -- megScripts/temporalGeneralizationDecoding.py's
    roi()/loadmat(atlas_fpath) call would hit exactly this if ever run
    against the current atlas file; we avoid that here.
    """
    try:
        return 'scipy', loadmat(fpath)
    except NotImplementedError:
        return 'h5py', open_h5(fpath, os.path.basename(fpath))


def load_atlas_masks(voxRes, bids_root):
    """
    Returns {'visual': bool[nGrid], 'parietal': bool[nGrid], 'frontal': bool[nGrid]},
    FULL TEMPLATE-GRID length exactly as G02_WangAtlasParcellation.m saved them --
    NOT yet sliced by any subject's inside_pos (see roi_local_indices below).
    """
    atlas_fpath = os.path.join(bids_root, 'derivatives', 'atlas', f'rois_{voxRes}.mat')
    kind, handle = _load_mat_flexible(atlas_fpath)

    masks = {}
    try:
        if kind == 'scipy':
            for roi_name, key in MASK_KEYS.items():
                masks[roi_name] = np.asarray(handle[key]).flatten().astype(bool)
        else:
            for roi_name, key in MASK_KEYS.items():
                masks[roi_name] = np.array(handle[key]).flatten().astype(bool)
    finally:
        if kind == 'h5py':
            handle.close()
    return masks


def roi_local_indices(atlas_masks, inside_pos, roi_name):
    """
    Map a subject's LOCAL source columns (0..n_sources-1, as ordered in that
    subject's sourcedataCombined.trial / G04 amplitude-phase arrays) to the
    requested ROI, using the subject's own inside_pos.

    inside_pos (saved alongside the subject's G03 output) holds MATLAB
    1-based indices into the FULL TEMPLATE GRID -- the same indexing space
    atlas_masks[...] uses (see G02_WangAtlasParcellation.m's documented
    contract: consumers must do `parcel_id_this_subject = parcel_id(inside_pos)`).
    We replicate that here for the boolean group masks, converting to 0-based
    for numpy:

        mask_local = atlas_masks[roi_name][inside_pos - 1]

    This deliberately differs from megScripts/temporalGeneralizationDecoding.py's
    roi() helper, which indexes atlas_masks directly with no inside_pos
    re-slicing at all -- that is only correct if a subject's inside_pos happens
    to equal the identity map 1..nGrid_inside, which is not guaranteed.
    """
    inside_pos = np.asarray(inside_pos).astype(np.int64).flatten()
    mask_local = atlas_masks[roi_name][inside_pos - 1]
    return np.where(mask_local)[0]
