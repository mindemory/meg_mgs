#!/usr/bin/env python3
"""
Script to load and plot source space decoding results across subjects.
Loads behavioral data, filters trials, and plots 4-quantile results with subject averaging.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import socket
import h5py
from shutil import copyfile
from scipy.stats import circmean, circstd, ttest_1samp, gaussian_kde, pearsonr
from sklearn.linear_model import TheilSenRegressor

def load_subject_results(bidsRoot, subjID, voxRes='10mm'):
    """Load results for a single subject"""
    subName = f'sub-{subjID:02d}'
    results_file = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', 
                               'betaDecodingVC', f'{subName}_task-mgs_betaSVR_{voxRes}_withBehav.pkl')
    
    if os.path.exists(results_file):
        print(f"Loading results for {subName}")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        print(f"Results file not found for {subName}: {results_file}")
        return None

def main():
    # Define parameters
    subject_list = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    voxRes = '8mm'
    
    # Set bids root based on hostname
    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    print("Loading source space decoding results...")
    print(f"BIDS root: {bidsRoot}")
    print(f"Voxel resolution: {voxRes}")
    print(f"Subject list: {subject_list}")
    
    # Load results for all subjects
    all_results = []
    valid_subjects = []
    
    for subjID in subject_list:
        results = load_subject_results(bidsRoot, subjID, voxRes)
        if results is not None:
            all_results.append(results)
            valid_subjects.append(subjID)
            print(f"Loaded results for subject {subjID}")
        else:
            print(f"No results found for subject {subjID}")
    
    if not all_results:
        print("No valid results found for any subject")
        return
    
    print(f"Successfully loaded {len(all_results)} subjects: {valid_subjects}")
    
    # Get time vector from first subject (should be same for all)
    time_vector = all_results[0]['time_vector']
    
    # Initialize aggregate arrays
    n_timepoints = time_vector.shape[0]
    n_subjects = len(all_results)
    
    # Define grid for KDE evaluation
    time_vec_flat = time_vector.flatten()
    time_min = max(time_vec_flat.min() - 0.1, -0.5)  # Pad slightly but respect -0.5 limit
    time_max = min(time_vec_flat.max() + 0.1, 1.7)   # Pad slightly but respect 1.7 limit
    
    # Create evaluation grid
    n_time_grid = 25
    n_error_grid = 25
    time_grid = np.linspace(time_min, time_max, n_time_grid)
    error_grid = np.linspace(-180, 180, n_error_grid)
    Time_grid, Error_grid = np.meshgrid(time_grid, error_grid)
    
    # Arrays to store KDE density estimates for each subject
    visual_densities = []
    frontal_densities = []
    visual_control_densities = []
    frontal_control_densities = []
    
    # Prepare grid positions for KDE evaluation (same for all subjects)
    positions = np.vstack([Time_grid.ravel(), Error_grid.ravel()])
    
    # Define angle mapping (in degrees)
    angle_mapping = {
        1: 0,    # 0 degrees
        2: 25,   # 25 degrees  
        3: 50,   # 50 degrees
        4: 130,  # 130 degrees
        5: 155,  # 155 degrees
        6: 180,  # 180 degrees
        7: 205,  # 205 degrees
        8: 230,  # 230 degrees
        9: 310,  # 310 degrees
        10: 335  # 335 degrees
    }
    
    # Process each subject to compute KDE
    for s_idx, results in enumerate(all_results):
        print(f"Processing subject {valid_subjects[s_idx]}...")
        
        # Extract data for this subject
        angular_errors_visual = results['angular_errors_visual']  # Shape: (n_trials, n_timepoints)
        angular_errors_frontal = results['angular_errors_frontal']  # Shape: (n_trials, n_timepoints)
        angular_errors_visual_control = results['angular_errors_visual_control']  # Shape: (n_trials, n_timepoints)
        angular_errors_frontal_control = results['angular_errors_frontal_control']  # Shape: (n_trials, n_timepoints)
        # pred_angles_deg_visual = results['pred_angles_deg_visual']  # Shape: (n_trials, n_timepoints)
        # pred_angles_deg_frontal = results['pred_angles_deg_frontal']  # Shape: (n_trials, n_timepoints)
        # target_labels = results['target_labels']  # Shape: (n_trials,)
        # target_labels_angle = np.array([angle_mapping[label] for label in target_labels])

        # # Compute angular errors (broadcast target_labels to match shape)
        # target_labels_angle_2d = target_labels_angle[:, np.newaxis]  # Reshape to (n_trials, 1) for broadcasting
        # angular_errors_visual = pred_angles_deg_visual - target_labels_angle_2d
        # angular_errors_frontal = pred_angles_deg_frontal - target_labels_angle_2d
        
        # Get time vector (should be same for all subjects)
        time_vec = time_vector.flatten()  # Shape: (n_timepoints,)
        
        # Create arrays of (time, error) pairs for all trials
        # For visual region
        time_vals_visual = np.tile(time_vec, (angular_errors_visual.shape[0], 1))  # Repeat time for each trial
        error_vals_visual = angular_errors_visual  # Errors for each trial and timepoint
        
        # Flatten to get all (time, error) pairs for this subject
        time_flat_visual = time_vals_visual.flatten()
        error_flat_visual = error_vals_visual.flatten()
        
        # Create KDE for this subject's visual data
        kde_visual_subj = gaussian_kde(np.vstack([time_flat_visual, error_flat_visual]))
        # Evaluate KDE on grid
        density_visual_subj = kde_visual_subj(positions).reshape(Time_grid.shape)
        visual_densities.append(density_visual_subj)
        
        # For frontal region
        time_vals_frontal = np.tile(time_vec, (angular_errors_frontal.shape[0], 1))
        error_vals_frontal = angular_errors_frontal
        
        # Flatten to get all (time, error) pairs for this subject
        time_flat_frontal = time_vals_frontal.flatten()
        error_flat_frontal = error_vals_frontal.flatten()
        
        # Create KDE for this subject's frontal data
        kde_frontal_subj = gaussian_kde(np.vstack([time_flat_frontal, error_flat_frontal]))
        # Evaluate KDE on grid
        density_frontal_subj = kde_frontal_subj(positions).reshape(Time_grid.shape)
        frontal_densities.append(density_frontal_subj)
        
        # For control data - visual region
        time_vals_visual_control = np.tile(time_vec, (angular_errors_visual_control.shape[0], 1))
        error_vals_visual_control = angular_errors_visual_control
        
        # Flatten to get all (time, error) pairs for this subject's control data
        time_flat_visual_control = time_vals_visual_control.flatten()
        error_flat_visual_control = error_vals_visual_control.flatten()
        
        # Create KDE for this subject's visual control data
        kde_visual_control_subj = gaussian_kde(np.vstack([time_flat_visual_control, error_flat_visual_control]))
        # Evaluate KDE on grid
        density_visual_control_subj = kde_visual_control_subj(positions).reshape(Time_grid.shape)
        visual_control_densities.append(density_visual_control_subj)
        
        # For control data - frontal region
        time_vals_frontal_control = np.tile(time_vec, (angular_errors_frontal_control.shape[0], 1))
        error_vals_frontal_control = angular_errors_frontal_control
        
        # Flatten to get all (time, error) pairs for this subject's control data
        time_flat_frontal_control = time_vals_frontal_control.flatten()
        error_flat_frontal_control = error_vals_frontal_control.flatten()
        
        # Create KDE for this subject's frontal control data
        kde_frontal_control_subj = gaussian_kde(np.vstack([time_flat_frontal_control, error_flat_frontal_control]))
        # Evaluate KDE on grid
        density_frontal_control_subj = kde_frontal_control_subj(positions).reshape(Time_grid.shape)
        frontal_control_densities.append(density_frontal_control_subj)
    
    # Average KDE densities across subjects
    print("Averaging KDE densities across subjects...")
    visual_densities = np.array(visual_densities)  # Shape: (n_subjects, n_error_grid, n_time_grid)
    frontal_densities = np.array(frontal_densities)
    visual_control_densities = np.array(visual_control_densities)
    frontal_control_densities = np.array(frontal_control_densities)
    
    density_visual = np.mean(visual_densities, axis=0)
    density_frontal = np.mean(frontal_densities, axis=0)
    density_visual_control = np.mean(visual_control_densities, axis=0)
    density_frontal_control = np.mean(frontal_control_densities, axis=0)
    
    # Compute difference: actual - control
    print("Computing difference (actual - control)...")
    density_visual_diff = density_visual - density_visual_control
    density_frontal_diff = density_frontal - density_frontal_control
    
    # Create figure with 2x2 subplots (2 rows, 2 columns: visual and frontal)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot Visual region difference (actual - control) as heatmap
    vmax_diff = max(np.abs(density_visual_diff).max(), np.abs(density_frontal_diff).max())
    im1 = axes[0,0].imshow(density_visual_diff, aspect='auto', origin='lower', 
                          extent=[time_grid[0], time_grid[-1], error_grid[0], error_grid[-1]],
                          cmap='RdBu_r', interpolation='bilinear', vmin=-vmax_diff, vmax=vmax_diff)
    axes[0,0].set_title('Visual Region (Actual - Control)')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Angular Error (°)')
    axes[0,0].axvline(x=0, color='white', linestyle='--', alpha=0.8, linewidth=1.5)
    axes[0,0].axhline(y=0, color='white', linestyle='--', alpha=0.8, linewidth=1.5)
    axes[0,0].set_xlim(-0.5, 1.7)
    axes[0,0].set_ylim(-180, 180)
    plt.colorbar(im1, ax=axes[0,0], label='Density Difference')
    
    # Plot Frontal region difference (actual - control) as heatmap
    im2 = axes[0,1].imshow(density_frontal_diff, aspect='auto', origin='lower',
                          extent=[time_grid[0], time_grid[-1], error_grid[0], error_grid[-1]],
                          cmap='RdBu_r', interpolation='bilinear', vmin=-vmax_diff, vmax=vmax_diff)
    axes[0,1].set_title('Frontal Region (Actual - Control)')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Angular Error (°)')
    axes[0,1].axvline(x=0, color='white', linestyle='--', alpha=0.8, linewidth=1.5)
    axes[0,1].axhline(y=0, color='white', linestyle='--', alpha=0.8, linewidth=1.5)
    axes[0,1].set_xlim(-0.5, 1.7)
    axes[0,1].set_ylim(-180, 180)
    plt.colorbar(im2, ax=axes[0,1], label='Density Difference')
    
    # ============================================================================
    # CORRELATION ANALYSIS: Angular Error vs i_sacc_err at each time point
    # ============================================================================
    print("\n" + "="*70)
    print("Starting correlation analysis: Angular Error vs i_sacc_err...")
    print("="*70)
    
    # Initialize arrays to store correlations for each subject at each time point
    visual_correlations = np.zeros((n_subjects, n_timepoints))
    frontal_correlations = np.zeros((n_subjects, n_timepoints))
    
    # Process each subject
    for s_idx, results in enumerate(all_results):
        subjID = valid_subjects[s_idx]
        print(f"Computing correlations for subject {subjID}...")
        
        # Load behavioral data (i_sacc_err)
        i_sacc_err = results['i_sacc_err']
        i_sacc_angle = results['i_sacc_angle']
        print(i_sacc_angle)
        target_labels = results['target_labels']
        target_labels_angle = np.array([angle_mapping[label] for label in target_labels])
        
        # Extract angular errors for this subject
        angular_errors_visual = results['angular_errors_visual']  # Shape: (n_trials, n_timepoints)
        angular_errors_frontal = results['angular_errors_frontal']  # Shape: (n_trials, n_timepoints)
        
        # Filter out trials with very small i_sacc_err (optional, can adjust threshold)
        errThresh = 1e-5
        valid_trials = i_sacc_err > errThresh
        angular_errors_visual = angular_errors_visual[valid_trials, :]
        angular_errors_frontal = angular_errors_frontal[valid_trials, :]
        # i_sacc_err_filtered = i_sacc_err[valid_trials]
        i_sacc_angle_filtered = i_sacc_angle[valid_trials]
        i_sacc_err_filtered = (target_labels_angle[valid_trials] - i_sacc_angle_filtered) % 360
        
        # Compute correlation at each time point
        for t_idx in range(n_timepoints):
            # Use absolute angular error for correlation
            abs_errors_visual = np.abs(angular_errors_visual[:, t_idx])
            abs_errors_frontal = np.abs(angular_errors_frontal[:, t_idx])
            
            # Compute Pearson correlation
            if len(abs_errors_visual) > 2:  # Need at least 3 points for correlation
                r_visual, _ = pearsonr(abs_errors_visual, i_sacc_err_filtered)
                r_frontal, _ = pearsonr(abs_errors_frontal, i_sacc_err_filtered)
                visual_correlations[s_idx, t_idx] = r_visual
                frontal_correlations[s_idx, t_idx] = r_frontal
    
    # Average correlations across subjects
    mean_corr_visual = np.mean(visual_correlations, axis=0)
    mean_corr_frontal = np.mean(frontal_correlations, axis=0)
    
    # Compute SEM for error bars
    sem_corr_visual = np.std(visual_correlations, axis=0) / np.sqrt(n_subjects)
    sem_corr_frontal = np.std(frontal_correlations, axis=0) / np.sqrt(n_subjects)
    
    # Get time vector for plotting
    time_vec_flat = time_vector.flatten()
    
    # Plot correlations in bottom row
    axes[1,0].plot(time_vec_flat, mean_corr_visual, 'b-', linewidth=2, label='Visual')
    axes[1,0].fill_between(time_vec_flat, 
                          mean_corr_visual - sem_corr_visual,
                          mean_corr_visual + sem_corr_visual,
                          color='blue', alpha=0.3)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Correlation (r)')
    axes[1,0].set_title('Visual Region - Correlation: |Angular Error| vs i_sacc_err')
    axes[1,0].set_xlim(-0.5, 1.7)
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend()
    
    axes[1,1].plot(time_vec_flat, mean_corr_frontal, 'g-', linewidth=2, label='Frontal')
    axes[1,1].fill_between(time_vec_flat,
                          mean_corr_frontal - sem_corr_frontal,
                          mean_corr_frontal + sem_corr_frontal,
                          color='green', alpha=0.3)
    axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Correlation (r)')
    axes[1,1].set_title('Frontal Region - Correlation: |Angular Error| vs i_sacc_err')
    axes[1,1].set_xlim(-0.5, 1.7)
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend()
    
    plt.suptitle(f'SVR Decoding Results - All Subjects (n={n_subjects})', fontsize=16)
    plt.tight_layout()

    # Save figure 1 as SVG
    # output_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs03')
    # os.makedirs(output_dir, exist_ok=True)
    # save_path = os.path.join(output_dir, f'svr_decoding_results_{voxRes}.svg')
    # fig.savefig(save_path, format='svg', bbox_inches='tight')
    # print(f"Figure 1 saved as SVG to {save_path}")
    
    plt.show()
    
    

if __name__ == '__main__':
    main()