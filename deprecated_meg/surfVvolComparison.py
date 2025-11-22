#!/usr/bin/env python3
"""
Surface vs Volume Comparison for fMRIPrep Outputs
Compares surface-based (Gifti) and volume-based (NIfTI) functional data
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

def load_gifti_data(filepath):
    """Load Gifti data and return the 2D array (vertices x time)"""
    print(f"Loading: {filepath}")
    img = nib.load(filepath)
    
    # For functional data, we expect multiple time points
    if len(img.darrays) > 1:
        # Multiple time points - concatenate them
        data_arrays = [darray.data for darray in img.darrays]
        data = np.column_stack(data_arrays)  # Shape: (vertices, timepoints)
    else:
        # Single time point - this might be a single volume
        data = img.darrays[0].data
        if data.ndim == 1:
            # If 1D, we need to check if this is actually time series data
            print(f"  WARNING: Single 1D array detected. This might be a single time point.")
            print(f"  This could indicate the data is not time series but a single volume.")
            # For now, treat as single time point
            data = data.reshape(-1, 1)  # Reshape to (vertices, 1)
    
    print(f"  Final shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Value range: [{np.min(data):.2f}, {np.max(data):.2f}]")
    return data

def load_nifti_data(filepath):
    """Load NIfTI data and return the 4D array"""
    print(f"Loading: {filepath}")
    img = nib.load(filepath)
    data = img.get_fdata()
    
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Value range: [{np.min(data):.2f}, {np.max(data):.2f}]")
    
    return data

def compute_temporal_snr(time_series):
    """
    Compute temporal SNR (tSNR) for each vertex/voxel
    tSNR = mean(time_series) / std(time_series)
    """
    tsnr = []
    for i in range(time_series.shape[0]):
        if np.std(time_series[i]) > 0:  # Avoid division by zero
            tsnr_val = np.mean(time_series[i]) / np.std(time_series[i])
            tsnr.append(tsnr_val)
        else:
            tsnr.append(np.nan)
    
    return np.array(tsnr)

def load_roi_mask(filepath):
    """Load ROI mask and return the data"""
    print(f"Loading ROI: {filepath}")
    if filepath.endswith('.func.gii'):
        img = nib.load(filepath)
        data = img.darrays[0].data
    else:  # .nii.gz
        img = nib.load(filepath)
        data = img.get_fdata()
    
    print(f"  Shape: {data.shape}")
    print(f"  Non-zero elements: {np.count_nonzero(data):,}")
    return data

def compute_percent_signal_change(time_series):
    """Compute percent signal change for time series"""
    # PSC = ((signal - baseline) / baseline) * 100
    # Use mean as baseline
    baseline = np.mean(time_series, axis=1, keepdims=True)
    psc = ((time_series - baseline) / baseline) * 100
    return psc

def extract_roi_data(volume_data, roi_masks, left_hemi_data, right_hemi_data):
    """Extract data for each ROI from both surface and volume, combining left and right hemispheres"""
    roi_results = {}
    
    # Group ROIs by base name (V1, V2, V3, V3AB, sPCS, iPCS)
    roi_groups = {}
    for roi_name, (surface_mask, volume_mask) in roi_masks.items():
        base_name = roi_name.split('.')[0]  # Extract V1, V2, V3, V3AB, sPCS, iPCS
        if base_name not in roi_groups:
            roi_groups[base_name] = {}
        roi_groups[base_name][roi_name] = (surface_mask, volume_mask)
    
    # Process each group (combining left and right)
    for base_name, hemispheres in roi_groups.items():
        print(f"\nProcessing {base_name} (combining L and R)...")
        
        # Get left and right hemisphere data
        left_roi_name = f"{base_name}.L"
        right_roi_name = f"{base_name}.R"
        
        if left_roi_name in hemispheres and right_roi_name in hemispheres:
            left_surface_mask, left_volume_mask = hemispheres[left_roi_name]
            right_surface_mask, right_volume_mask = hemispheres[right_roi_name]
            
            # Surface data extraction - handle left/right hemispheres separately
            if left_surface_mask.ndim == 1 and right_surface_mask.ndim == 1:
                # Left hemisphere ROI
                left_surface_roi_data = left_hemi_data[left_surface_mask > 0, :]
                # Right hemisphere ROI  
                right_surface_roi_data = right_hemi_data[right_surface_mask > 0, :]
                # Combine surface data
                surface_roi_data = np.vstack([left_surface_roi_data, right_surface_roi_data])
            else:
                print(f"  WARNING: Unexpected surface mask shapes")
                continue
                
            # Volume data extraction - combine left and right
            if left_volume_mask.ndim == 3 and right_volume_mask.ndim == 3:
                # Get coordinates of non-zero voxels for both hemispheres
                left_coords = np.where(left_volume_mask > 0)
                right_coords = np.where(right_volume_mask > 0)
                
                if len(left_coords[0]) > 0 and len(right_coords[0]) > 0:
                    # Extract time series for left hemisphere voxels
                    left_volume_roi_data = volume_data[left_coords[0], left_coords[1], left_coords[2], :]
                    # Extract time series for right hemisphere voxels
                    right_volume_roi_data = volume_data[right_coords[0], right_coords[1], right_coords[2], :]
                    # Combine volume data
                    volume_roi_data = np.vstack([left_volume_roi_data, right_volume_roi_data])
                else:
                    print(f"  WARNING: No non-zero voxels in volume masks for {base_name}")
                    continue
            else:
                print(f"  WARNING: Unexpected volume mask shapes")
                continue
            
            print(f"  Surface ROI shape: {surface_roi_data.shape}")
            print(f"  Volume ROI shape: {volume_roi_data.shape}")
            
            # Compute percent signal change
            # surface_psc = compute_percent_signal_change(surface_roi_data)
            # volume_psc = compute_percent_signal_change(volume_roi_data)
            surface_psc = surface_roi_data
            volume_psc = volume_roi_data
            
    
            
            roi_results[base_name] = {
                'surface_data': surface_roi_data,
                'volume_data': volume_roi_data,
                'surface_psc': surface_psc,
                'volume_psc': volume_psc,
                'n_surface_vertices': surface_roi_data.shape[0],
                'n_volume_voxels': volume_roi_data.shape[0]
            }
            
            print(f"  Surface vertices: {surface_roi_data.shape[0]:,}")
            print(f"  Volume voxels: {volume_roi_data.shape[0]:,}")
        else:
            print(f"  WARNING: Missing hemisphere data for {base_name}")
            print(f"    Left: {left_roi_name in hemispheres}")
            print(f"    Right: {right_roi_name in hemispheres}")
    
    return roi_results

def plot_tsnr_histograms(roi_results):
    """Plot tSNR histograms for each ROI - 2 rows: volume (top) and surface (bottom)"""
    roi_names = list(roi_results.keys())
    n_rois = len(roi_names)
    
    # Create subplots - 2 rows: volume (top) and surface (bottom)
    fig, axes = plt.subplots(2, n_rois, figsize=(3*n_rois, 8))
    if n_rois == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle('Temporal SNR (tSNR) Distribution by ROI (Combined Across All Runs)', fontsize=16, fontweight='bold')
    
    for i, roi_name in enumerate(roi_names):
        results = roi_results[roi_name]
        
        # Use pre-computed tSNR values
        surface_tsnr_clean = results['surface_tsnr']
        volume_tsnr_clean = results['volume_tsnr']
        
        # Volume tSNR histogram (top row)
        ax_volume = axes[0, i]
        ax_volume.hist(volume_tsnr_clean, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax_volume.axvline(np.mean(volume_tsnr_clean), color='red', linestyle='--',
                         label=f'Mean: {np.mean(volume_tsnr_clean):.2f}')
        ax_volume.set_xlabel('tSNR')
        ax_volume.set_ylabel('Frequency')
        ax_volume.set_title(f'{roi_name} - Volume\nn={len(volume_tsnr_clean):,} voxels')
        ax_volume.legend()
        ax_volume.grid(True, alpha=0.3)
        ax_volume.set_xlim(0, 200)
        
        # Surface tSNR histogram (bottom row)
        ax_surface = axes[1, i]
        ax_surface.hist(surface_tsnr_clean, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax_surface.axvline(np.mean(surface_tsnr_clean), color='red', linestyle='--',
                          label=f'Mean: {np.mean(surface_tsnr_clean):.2f}')
        ax_surface.set_xlabel('tSNR')
        ax_surface.set_ylabel('Frequency')
        ax_surface.set_title(f'{roi_name} - Surface\nn={len(surface_tsnr_clean):,} vertices')
        ax_surface.legend()
        ax_surface.grid(True, alpha=0.3)
        ax_surface.set_xlim(0, 200)
        
        # Print summary statistics
        print(f"\n{roi_name} tSNR Summary (combined across all runs):")
        print(f"  Surface: mean={np.mean(surface_tsnr_clean):.2f}, std={np.std(surface_tsnr_clean):.2f}, "
              f"min={np.min(surface_tsnr_clean):.2f}, max={np.max(surface_tsnr_clean):.2f}")
        print(f"  Volume:  mean={np.mean(volume_tsnr_clean):.2f}, std={np.std(volume_tsnr_clean):.2f}, "
              f"min={np.min(volume_tsnr_clean):.2f}, max={np.max(volume_tsnr_clean):.2f}")
    
    plt.tight_layout()
    plt.show()

def analyze_single_run(run_num, base_path, roi_base_path, roi_names, subName, surf2vol=False):
    """Analyze a single run and return ROI results"""
    print(f"\n{'='*60}")
    print(f"ANALYZING RUN {run_num:02d}")
    print(f"{'='*60}")
    
    # File paths for this run
    if subName == 'ZID0704':
        task_name = 'wmrotate'
    else:
        task_name = 'TASK'
    
    left_surface_path = os.path.join(base_path, f"sub-{subName}_ses-01_task-{task_name}_run-{run_num:02d}_hemi-L_space-fsnative_bold.func.gii")
    right_surface_path = os.path.join(base_path, f"sub-{subName}_ses-01_task-{task_name}_run-{run_num:02d}_hemi-R_space-fsnative_bold.func.gii")
    
    # Volume data path - choose based on surf2vol flag
    if surf2vol:
        volume_path = os.path.join(roi_base_path, f"ses-01/surf2vol_ses-01_run-{run_num:02d}.nii.gz")
    else:
        volume_path = os.path.join(base_path, f"sub-{subName}_ses-01_task-{task_name}_run-{run_num:02d}_space-T1w_desc-preproc_bold.nii.gz")
    
    # Check if files exist
    if not all(os.path.exists(p) for p in [left_surface_path, right_surface_path, volume_path]):
        print(f"WARNING: Missing files for run {run_num:02d}")
        return None
    
    # Load surface data
    print("\n1. Loading surface data...")
    left_hemi_data = load_gifti_data(left_surface_path)
    right_hemi_data = load_gifti_data(right_surface_path)
    
    # Load volume data
    print("\n2. Loading volume data...")
    volume_data = load_nifti_data(volume_path)
    
    # Load ROI masks
    print("\n3. Loading ROI masks...")
    roi_masks = {}
    
    for roi_name in roi_names:
        # Surface ROI (Gifti)
        surface_roi_path = os.path.join(roi_base_path, "ROIs_gii", f"{roi_name}.func.gii")
        # Volume ROI (NIfTI)
        volume_roi_path = os.path.join(roi_base_path, "ROIs_nii", f"{roi_name}.nii.gz")
        
        if os.path.exists(surface_roi_path) and os.path.exists(volume_roi_path):
            surface_mask = load_roi_mask(surface_roi_path)
            volume_mask = load_roi_mask(volume_roi_path)
            roi_masks[roi_name] = (surface_mask, volume_mask)
    
    print(f"✓ Loaded {len(roi_masks)} ROI masks")
    
    # Extract ROI data and compute PSC
    print("\n4. Extracting ROI data and computing PSC...")
    roi_results = extract_roi_data(volume_data, roi_masks, left_hemi_data, right_hemi_data)
    
    return roi_results

def combine_roi_results(all_run_results):
    """Combine ROI results from all runs"""
    if not all_run_results:
        return {}
    
    # Get ROI names from first successful run
    roi_names = list(all_run_results[list(all_run_results.keys())[0]].keys())
    combined_results = {}
    
    for roi_name in roi_names:
        # Collect all surface and volume data across runs
        surface_data_all = []
        volume_data_all = []
        
        for run_num, run_results in all_run_results.items():
            if roi_name in run_results:
                surface_data_all.append(run_results[roi_name]['surface_data'])
                volume_data_all.append(run_results[roi_name]['volume_data'])
        
        if surface_data_all and volume_data_all:
            # Combine data across runs
            combined_surface_data = np.vstack(surface_data_all)
            combined_volume_data = np.vstack(volume_data_all)
            
            # Compute tSNR for combined data
            surface_tsnr = compute_temporal_snr(combined_surface_data)
            volume_tsnr = compute_temporal_snr(combined_volume_data)
            
            # Remove NaN values
            surface_tsnr_clean = surface_tsnr[~np.isnan(surface_tsnr)]
            volume_tsnr_clean = volume_tsnr[~np.isnan(volume_tsnr)]
            
            combined_results[roi_name] = {
                'surface_data': combined_surface_data,
                'volume_data': combined_volume_data,
                'surface_tsnr': surface_tsnr_clean,
                'volume_tsnr': volume_tsnr_clean,
                'n_surface_vertices': combined_surface_data.shape[0],
                'n_volume_voxels': combined_volume_data.shape[0]
            }
    
    return combined_results

def main():
    subName = 'ZID0704'
    surf2vol = False  # Set to True to use surf2vol volume data
    
    if subName == 'MRD1219':
        subID = 2
    elif subName == 'ZID0704':
        subID = 1
    elif subName == 'MIG0702':
        subID = 7
    elif subName == 'WIC0326':
        subID = 8
    elif subName == 'MOL1020':
        subID = 10
    elif subName == 'JEC0805':
        subID = 13
    # File paths for MRD1219
    base_path = f"/System/Volumes/Data/d/DATD/datd/wmRotate_ZD/wmRotate/fMRIPrep24.0.1/derivatives/sub-{subName}/ses-01/func"
    roi_base_path = f"/System/Volumes/Data/d/DATD/datd/wmRotate_ZD/data_fmri/sub-{subID:02d}"
    
    # ROI names
    roi_names = ['V1.L', 'V1.R', 'V2.L', 'V2.R', 'V3.L', 'V3.R', 'V3AB.L', 'V3AB.R', 'iPCS.L', 'iPCS.R', 'sPCS.L', 'sPCS.R']
    
    print("="*80)
    print(f"Multi-run Surface vs Volume Comparison - {subName} All Runs")
    print(f"Volume data source: {'surf2vol' if surf2vol else 'fMRIPrep'}")
    print("="*80)
    
    # Analyze all runs
    all_run_results = {}
    successful_runs = 0
    
    for run_num in range(1, 3):  # runs 1-18
        run_results = analyze_single_run(run_num, base_path, roi_base_path, roi_names, subName, surf2vol)
        if run_results is not None:
            all_run_results[run_num] = run_results
            successful_runs += 1
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE: {successful_runs}/{len(range(1, 4))} runs successfully analyzed")
    print(f"{'='*80}")
    
    if successful_runs > 0:
        # Combine results from all runs
        print("\n5. Combining results across all runs...")
        combined_roi_results = combine_roi_results(all_run_results)
        
        # Compute tSNR for each ROI and create histograms
        print("\n6. Computing tSNR and creating histograms...")
        plot_tsnr_histograms(combined_roi_results)
    else:
        print("ERROR: No runs were successfully analyzed!")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()