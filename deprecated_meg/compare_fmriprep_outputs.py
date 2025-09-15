#!/usr/bin/env python3
"""
Compare fMRIPrep outputs with and without flag for MOL1020
Plots correlation as a function of time for 1000 randomly selected surface vertices
Uses fsnative_bold.func.gii files (surface-based functional data)
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
import glob

def load_gifti_data(filepath):
    """Load Gifti data and return the 2D array (vertices x time)"""
    print(f"Loading: {filepath}")
    img = nib.load(filepath)
    
    # Debug: Check the structure of the Gifti file
    print(f"  Number of data arrays: {len(img.darrays)}")
    for i, darray in enumerate(img.darrays):
        print(f"  Array {i}: shape={darray.data.shape}, intent={darray.intent}")
    
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

def select_random_vertices(data, n_vertices=1000):
    """Select random vertices from the surface (excluding background)"""
    # For surface data, we have shape (n_vertices, n_timepoints)
    n_total_vertices = data.shape[0]
    
    print(f"Total surface vertices: {n_total_vertices}")
    
    if n_vertices > n_total_vertices:
        n_vertices = n_total_vertices
        print(f"Reduced to {n_vertices} vertices (all available vertices)")
    
    # Randomly select vertices
    random_indices = np.random.choice(n_total_vertices, size=n_vertices, replace=False)
    
    return random_indices

def extract_time_series(data, vertex_indices):
    """Extract time series for selected vertices"""
    time_series = data[vertex_indices, :]  # Shape: (n_vertices, n_timepoints)
    return time_series

def compute_correlations(ts1, ts2):
    """Compute correlations between corresponding time series"""
    correlations = []
    for i in range(ts1.shape[0]):
        if np.std(ts1[i]) > 0 and np.std(ts2[i]) > 0:  # Avoid constant time series
            corr, _ = pearsonr(ts1[i], ts2[i])
            correlations.append(corr)
        else:
            correlations.append(np.nan)
    
    return np.array(correlations)

def compute_temporal_snr(time_series):
    """
    Compute temporal SNR (tSNR) for each vertex
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

def check_nifti_properties(nii_path, label):
    """Check NIfTI file properties and return shape info"""
    if not os.path.exists(nii_path):
        print(f"  {label}: File not found: {nii_path}")
        return None
    
    try:
        img = nib.load(nii_path)
        shape = img.shape
        n_voxels = np.prod(shape[:-1])  # Total voxels (excluding time dimension)
        n_timepoints = shape[-1]
        
        print(f"  {label}:")
        print(f"    Shape: {shape}")
        print(f"    Total voxels: {n_voxels:,}")
        print(f"    Time points: {n_timepoints}")
        print(f"    Data type: {img.get_data_dtype()}")
        
        return {
            'shape': shape,
            'n_voxels': n_voxels,
            'n_timepoints': n_timepoints,
            'dtype': img.get_data_dtype()
        }
    except Exception as e:
        print(f"  {label}: Error loading file - {e}")
        return None

def plot_combined_analysis(correlations, time_points, ts1, ts2, subject_id, save_path=None):
    """Plot combined correlation analysis and sample time series"""
    fig = plt.figure(figsize=(20, 12))
    
    # Create a grid layout: 3 rows, 5 columns
    # Top row: 3 correlation plots
    # Bottom 2 rows: 10 time series plots (2 rows of 5)
    
    # 1. Histogram of correlations (top left)
    ax1 = plt.subplot(3, 5, (1, 2))
    valid_corrs = correlations[~np.isnan(correlations)]
    ax1.hist(valid_corrs, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(valid_corrs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(valid_corrs):.3f}')
    ax1.set_xlabel('Correlation Coefficient')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Vertex Correlations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Correlation vs tSNR scatter plot (top middle)
    ax2 = plt.subplot(3, 5, (3, 4))
    
    # Compute tSNR for both datasets
    tsnr1 = compute_temporal_snr(ts1)
    tsnr2 = compute_temporal_snr(ts2)
    
    # Use average tSNR for plotting
    avg_tsnr = (tsnr1 + tsnr2) / 2
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(correlations) | np.isnan(avg_tsnr))
    valid_corrs_tsnr = correlations[valid_mask]
    valid_tsnr = avg_tsnr[valid_mask]
    
    # Create scatter plot
    scatter = ax2.scatter(valid_tsnr, valid_corrs_tsnr, alpha=0.6, s=10, c=valid_corrs_tsnr, 
                         cmap='viridis', vmin=-1, vmax=1)
    ax2.set_xlabel('Temporal SNR')
    ax2.set_ylabel('Correlation Coefficient')
    ax2.set_title('Correlation vs Temporal SNR')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Correlation')
    
    # 3. Correlation vs vertex index (top right)
    ax3 = plt.subplot(3, 5, 5)
    ax3.scatter(range(len(correlations)), correlations, alpha=0.6, s=10)
    ax3.axhline(np.mean(valid_corrs), color='red', linestyle='--', 
                label=f'Mean: {np.mean(valid_corrs):.3f}')
    ax3.set_xlabel('Vertex Index')
    ax3.set_ylabel('Correlation Coefficient')
    ax3.set_title('Correlation by Vertex Index')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4-13. Sample time series (bottom 2 rows: 10 subplots)
    # Select vertices based on correlation quality
    selected_indices = select_vertices_by_correlation(correlations, n_good=3, n_moderate=3, n_bad=4)
    
    for i, vertex_idx in enumerate(selected_indices):
        if i < 5:  # First row of time series (row 2, columns 1-5)
            ax = plt.subplot(3, 5, 6 + i)  # Positions 6-10
        else:  # Second row of time series (row 3, columns 1-5)
            ax = plt.subplot(3, 5, 11 + (i - 5))  # Positions 11-15
        
        # Plot both time series
        ax.plot(time_points, ts1[vertex_idx], 'b-', alpha=0.7, linewidth=1.5, label='With Flag')
        ax.plot(time_points, ts2[vertex_idx], 'r-', alpha=0.7, linewidth=1.5, label='Without Flag')
        
        # Add correlation as title with color coding
        corr_val = correlations[vertex_idx]
        if corr_val > 0.8:
            title_color = 'green'
            quality = 'Good'
        elif corr_val > 0.5:
            title_color = 'orange'
            quality = 'Moderate'
        else:
            title_color = 'red'
            quality = 'Bad'
            
        ax.set_title(f'Vertex {vertex_idx}\n{quality} (r={corr_val:.3f})', 
                    fontsize=10, color=title_color, fontweight='bold')
        ax.set_xlabel('Time Point', fontsize=8)
        ax.set_ylabel('BOLD Signal', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Add legend only to first subplot
        if i == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle(f'Subject {subject_id}: fMRIPrep Output Comparison', 
                fontsize=16, y=0.95)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to: {save_path}")
    
    plt.show()

def select_vertices_by_correlation(correlations, n_good=3, n_moderate=3, n_bad=4):
    """Select vertices based on correlation quality"""
    valid_indices = ~np.isnan(correlations)
    valid_corrs = correlations[valid_indices]
    valid_vertex_indices = np.where(valid_indices)[0]
    
    # Define correlation thresholds
    good_threshold = 0.8
    moderate_threshold = 0.5
    
    # Categorize vertices
    good_mask = valid_corrs > good_threshold
    moderate_mask = (valid_corrs > moderate_threshold) & (valid_corrs <= good_threshold)
    bad_mask = valid_corrs <= moderate_threshold
    
    good_indices = valid_vertex_indices[good_mask]
    moderate_indices = valid_vertex_indices[moderate_mask]
    bad_indices = valid_vertex_indices[bad_mask]
    
    print(f"Correlation categories:")
    print(f"  Good (>0.8): {len(good_indices)} vertices")
    print(f"  Moderate (0.5-0.8): {len(moderate_indices)} vertices")
    print(f"  Bad (<0.5): {len(bad_indices)} vertices")
    
    # Select vertices from each category
    selected = []
    
    # Good correlations
    if len(good_indices) >= n_good:
        selected.extend(np.random.choice(good_indices, size=n_good, replace=False))
    else:
        selected.extend(good_indices)
    
    # Moderate correlations
    if len(moderate_indices) >= n_moderate:
        selected.extend(np.random.choice(moderate_indices, size=n_moderate, replace=False))
    else:
        selected.extend(moderate_indices)
    
    # Bad correlations
    if len(bad_indices) >= n_bad:
        selected.extend(np.random.choice(bad_indices, size=n_bad, replace=False))
    else:
        selected.extend(bad_indices)
    
    # If we don't have enough, fill with random valid vertices
    while len(selected) < (n_good + n_moderate + n_bad):
        remaining = valid_vertex_indices[~np.isin(valid_vertex_indices, selected)]
        if len(remaining) > 0:
            selected.append(np.random.choice(remaining))
        else:
            break
    
    return selected[:n_good + n_moderate + n_bad]

def compare_datasets_robust(data1, data2, method='common_vertices', n_vertices=1000):
    """
    Compare two datasets using different methods to handle vertex count mismatches
    
    Parameters:
    - data1, data2: 2D arrays (vertices x timepoints)
    - method: 'common_vertices', 'random_common', 'all_common'
    - n_vertices: number of vertices to sample (for random methods)
    """
    print(f"\nUsing comparison method: {method}")
    
    if method == 'common_vertices':
        # Use only the first N vertices where N is the minimum
        min_vertices = min(data1.shape[0], data2.shape[0])
        data1_comp = data1[:min_vertices, :]
        data2_comp = data2[:min_vertices, :]
        print(f"  Using first {min_vertices} vertices from both datasets")
        
    elif method == 'random_common':
        # Randomly sample from the common vertex range
        min_vertices = min(data1.shape[0], data2.shape[0])
        if n_vertices > min_vertices:
            n_vertices = min_vertices
        random_indices = np.random.choice(min_vertices, size=n_vertices, replace=False)
        data1_comp = data1[random_indices, :]
        data2_comp = data2[random_indices, :]
        print(f"  Randomly sampled {n_vertices} vertices from common range")
        
    elif method == 'all_common':
        # Use all common vertices (no sampling)
        min_vertices = min(data1.shape[0], data2.shape[0])
        data1_comp = data1[:min_vertices, :]
        data2_comp = data2[:min_vertices, :]
        print(f"  Using all {min_vertices} common vertices")
        
    return data1_comp, data2_comp

def main():
    # File paths for WIC0326 - using fsaverage6 space for consistent vertex counts
    base_path = "/datd/preproc_hpc_legacy/MgFinalTesting/Data/WIC0326"
    
    # With flag (mark2.2a_NewRecon) - Left hemisphere in fsaverage6 space
    with_flag_path = os.path.join(base_path, "mark2.2a_NewRecon/func/sub-WIC0326_ses-01_task-TASK_run-01_hemi-L_space-fsaverage6_bold.func.gii")
    
    # Without flag (mark2.2d_fMRIPrep24-IgJcbn) - Left hemisphere in fsaverage6 space
    without_flag_path = os.path.join(base_path, "mark2.2d_fMRIPrep24-IgJcbn/func/sub-WIC0326_ses-pRF_task-TASK_run-01_hemi-L_space-fsaverage6_bold.func.gii")
    
    # NIfTI file paths for voxel comparison
    with_flag_nii_path = os.path.join(base_path, "mark2.2a_NewRecon/func/sub-WIC0326_ses-01_task-TASK_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    without_flag_nii_path = os.path.join(base_path, "mark2.2d_fMRIPrep24-IgJcbn/func/sub-WIC0326_ses-pRF_task-TASK_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    
    print("=" * 60)
    print("fMRIPrep Output Comparison - WIC0326 (fsaverage6 Space)")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists(with_flag_path):
        print(f"ERROR: File not found: {with_flag_path}")
        return
    
    if not os.path.exists(without_flag_path):
        print(f"ERROR: File not found: {without_flag_path}")
        return
    
    # Check NIfTI files for voxel comparison
    print("\n0. Checking NIfTI files for voxel differences...")
    nii_with_flag = check_nifti_properties(with_flag_nii_path, "With Flag (NIfTI)")
    nii_without_flag = check_nifti_properties(without_flag_nii_path, "Without Flag (NIfTI)")
    
    if nii_with_flag and nii_without_flag:
        print(f"\n  Voxel Comparison:")
        print(f"    With flag voxels: {nii_with_flag['n_voxels']:,}")
        print(f"    Without flag voxels: {nii_without_flag['n_voxels']:,}")
        print(f"    Difference: {nii_with_flag['n_voxels'] - nii_without_flag['n_voxels']:,}")
        
        if nii_with_flag['n_voxels'] == nii_without_flag['n_voxels']:
            print("    ✓ Same number of voxels")
        else:
            print("    ⚠ Different number of voxels!")
            
        print(f"\n  Time Points Comparison:")
        print(f"    With flag time points: {nii_with_flag['n_timepoints']}")
        print(f"    Without flag time points: {nii_without_flag['n_timepoints']}")
        
        if nii_with_flag['n_timepoints'] == nii_without_flag['n_timepoints']:
            print("    ✓ Same number of time points")
        else:
            print("    ⚠ Different number of time points!")
    else:
        print("  ⚠ Could not compare NIfTI files - some files missing or corrupted")
    
    # Load data
    print("\n1. Loading data...")
    data_with_flag = load_gifti_data(with_flag_path)
    data_without_flag = load_gifti_data(without_flag_path)
    
    # Use robust comparison method
    print("\n2. Comparing datasets...")
    comparison_method = 'all_common'  # Use all common vertices instead of random sampling
    
    data1_comp, data2_comp = compare_datasets_robust(data_with_flag, data_without_flag, 
                                                   method=comparison_method)
    
    print(f"✓ Comparison datasets shape: {data1_comp.shape}")
    
    # Extract time series (data is already in the right format)
    print("\n3. Preparing time series...")
    global ts1, ts2  # Make global for plotting
    ts1 = data1_comp
    ts2 = data2_comp
    print(f"✓ Time series shape: {ts1.shape}")
    
    # Compute correlations
    print("\n4. Computing correlations...")
    correlations = compute_correlations(ts1, ts2)
    valid_correlations = correlations[~np.isnan(correlations)]
    
    print(f"✓ Computed {len(valid_correlations)} valid correlations")
    print(f"  Mean correlation: {np.mean(valid_correlations):.4f}")
    print(f"  Std correlation: {np.std(valid_correlations):.4f}")
    print(f"  Min correlation: {np.min(valid_correlations):.4f}")
    print(f"  Max correlation: {np.max(valid_correlations):.4f}")
    
    # Compute temporal SNR
    print("\n5. Computing temporal SNR...")
    tsnr1 = compute_temporal_snr(ts1)
    tsnr2 = compute_temporal_snr(ts2)
    avg_tsnr = (tsnr1 + tsnr2) / 2
    valid_tsnr = avg_tsnr[~np.isnan(avg_tsnr)]
    
    print(f"✓ Computed temporal SNR for {len(valid_tsnr)} vertices")
    print(f"  Mean tSNR: {np.mean(valid_tsnr):.4f}")
    print(f"  Std tSNR: {np.std(valid_tsnr):.4f}")
    print(f"  Min tSNR: {np.min(valid_tsnr):.4f}")
    print(f"  Max tSNR: {np.max(valid_tsnr):.4f}")
    
    # Compute correlation between correlation and tSNR
    valid_mask = ~(np.isnan(correlations) | np.isnan(avg_tsnr))
    corr_tsnr_corr, _ = pearsonr(correlations[valid_mask], avg_tsnr[valid_mask])
    print(f"  Correlation between correlation and tSNR: {corr_tsnr_corr:.4f}")
    
    # Create time points for plotting
    time_points = np.arange(ts1.shape[1])
    
    # Plot results
    print("\n6. Creating combined plot...")
    
    # Combined correlation analysis and sample time series plot
    plot_combined_analysis(correlations, time_points, ts1, ts2, subject_id="WIC0326",
                          save_path="wic0326_fmriprep_combined_analysis.png")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
