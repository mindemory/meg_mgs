#!/usr/bin/env python3
"""
Script to examine ROI files for both surface (gii) and volume (nii) formats.
"""

import os
import numpy as np
import nibabel as nib
from scipy import stats
import matplotlib.pyplot as plt
from glob import glob

def load_gifti_roi(file_path):
    """Load a Gifti ROI file and return the data."""
    try:
        gii = nib.load(file_path)
        data = gii.darrays[0].data
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_nifti_roi(file_path):
    """Load a NIfTI ROI file and return the data."""
    try:
        nii = nib.load(file_path)
        data = nii.get_fdata()
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def examine_roi_structure(data, roi_name, data_type):
    """Examine the structure of an ROI."""
    print(f"\n{roi_name} ({data_type}):")
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Min value: {np.min(data):.4f}")
    print(f"  Max value: {np.max(data):.4f}")
    print(f"  Mean value: {np.mean(data):.4f}")
    print(f"  Std value: {np.std(data):.4f}")
    
    # Count non-zero elements
    non_zero = np.count_nonzero(data)
    total = data.size
    print(f"  Non-zero elements: {non_zero:,} / {total:,} ({100*non_zero/total:.2f}%)")
    
    # For binary ROIs, show unique values
    unique_vals = np.unique(data)
    print(f"  Unique values: {unique_vals}")
    
    return {
        'shape': data.shape,
        'dtype': data.dtype,
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'std': np.std(data),
        'non_zero': non_zero,
        'total': total,
        'unique_vals': unique_vals
    }

def compare_roi_formats(gii_path, nii_path, roi_name):
    """Compare the same ROI in both gii and nii formats."""
    print(f"\n{'='*60}")
    print(f"Comparing {roi_name} in both formats")
    print(f"{'='*60}")
    
    # Load both formats
    gii_data = load_gifti_roi(gii_path)
    nii_data = load_nifti_roi(nii_path)
    
    if gii_data is None or nii_data is None:
        print("Could not load one or both files")
        return None
    
    # Examine structure
    gii_info = examine_roi_structure(gii_data, f"{roi_name} (Gifti)", "Surface")
    nii_info = examine_roi_structure(nii_data, f"{roi_name} (NIfTI)", "Volume")
    
    # Compare sizes
    print(f"\nSize comparison:")
    print(f"  Gifti vertices: {gii_data.shape[0]:,}")
    print(f"  NIfTI voxels: {nii_data.shape[0]*nii_data.shape[1]*nii_data.shape[2]:,}")
    print(f"  NIfTI dimensions: {nii_data.shape}")
    
    # Check if they represent the same ROI
    gii_nonzero = np.count_nonzero(gii_data)
    nii_nonzero = np.count_nonzero(nii_data)
    
    print(f"\nROI size comparison:")
    print(f"  Gifti non-zero vertices: {gii_nonzero:,}")
    print(f"  NIfTI non-zero voxels: {nii_nonzero:,}")
    print(f"  Ratio (Gifti/NIfTI): {gii_nonzero/nii_nonzero:.4f}")
    
    return {
        'gii_info': gii_info,
        'nii_info': nii_info,
        'gii_data': gii_data,
        'nii_data': nii_data
    }

def main():
    """Main function to examine ROI files."""
    print("=" * 80)
    print("ROI Examination Script")
    print("=" * 80)
    
    # Define paths
    gii_roi_dir = "/System/Volumes/Data/d/DATD/datd/wmRotate_ZD/data_fmri/sub-02/ROIs_gii"
    nii_roi_dir = "/System/Volumes/Data/d/DATD/datd/wmRotate_ZD/data_fmri/sub-02/ROIs_nii"
    
    # Get list of ROI files
    gii_files = glob(os.path.join(gii_roi_dir, "*.func.gii"))
    nii_files = glob(os.path.join(nii_roi_dir, "*.nii.gz"))
    
    print(f"\nFound {len(gii_files)} Gifti ROI files")
    print(f"Found {len(nii_files)} NIfTI ROI files")
    
    # List all available ROIs
    print(f"\nAvailable ROIs (Gifti):")
    for i, file in enumerate(sorted(gii_files)):
        roi_name = os.path.basename(file).replace('.func.gii', '')
        print(f"  {i+1:2d}. {roi_name}")
    
    print(f"\nAvailable ROIs (NIfTI):")
    for i, file in enumerate(sorted(nii_files)):
        roi_name = os.path.basename(file).replace('.nii.gz', '')
        print(f"  {i+1:2d}. {roi_name}")
    
    # Examine a few key ROIs in detail
    key_rois = ['V1.L', 'V1.R', 'IPS.L', 'IPS.R']
    
    print(f"\n{'='*80}")
    print("DETAILED EXAMINATION OF KEY ROIs")
    print(f"{'='*80}")
    
    roi_comparisons = {}
    
    for roi_name in key_rois:
        gii_path = os.path.join(gii_roi_dir, f"{roi_name}.func.gii")
        nii_path = os.path.join(nii_roi_dir, f"{roi_name}.nii.gz")
        
        if os.path.exists(gii_path) and os.path.exists(nii_path):
            comparison = compare_roi_formats(gii_path, nii_path, roi_name)
            if comparison:
                roi_comparisons[roi_name] = comparison
        else:
            print(f"\nMissing files for {roi_name}:")
            print(f"  Gifti: {os.path.exists(gii_path)}")
            print(f"  NIfTI: {os.path.exists(nii_path)}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    if roi_comparisons:
        print(f"\nExamined {len(roi_comparisons)} ROIs:")
        for roi_name, comparison in roi_comparisons.items():
            gii_info = comparison['gii_info']
            nii_info = comparison['nii_info']
            
            print(f"\n{roi_name}:")
            print(f"  Surface vertices: {gii_info['non_zero']:,} / {gii_info['total']:,}")
            print(f"  Volume voxels: {nii_info['non_zero']:,} / {nii_info['total']:,}")
            print(f"  Surface coverage: {100*gii_info['non_zero']/gii_info['total']:.2f}%")
            print(f"  Volume coverage: {100*nii_info['non_zero']/nii_info['total']:.2f}%")
    
    print(f"\n{'='*80}")
    print("ROI examination complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
