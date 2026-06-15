#!/usr/bin/env python3
"""
Script to load and plot source space decoding results across subjects.
Plots a 2x2 grid:
- Row 1: Scatter plots of decoding error vs time (Visual and Frontal regions).
- Row 2: Bar plots of mean decoding error during delay (0.5 to 1.5s) binned into 4 quantiles based on memory error (i_sacc_err).
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import socket
from scipy.stats import circmean, ttest_1samp

def load_subject_results(bidsRoot, subjID, voxRes='8mm', freq_band='alpha'):
    """Load SVR decoding results for a single subject"""
    subName = f'sub-{subjID:02d}'
    results_file = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', 
                               'decodingVC', f'{subName}_task-mgs_SVR_{freq_band}_{voxRes}_withBehav.pkl')
    
    # Fallback to legacy betaDecodingVC path if beta is requested but new path doesn't exist
    if not os.path.exists(results_file) and freq_band == 'beta':
        results_file = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', 
                                   'betaDecodingVC', f'{subName}_task-mgs_betaSVR_{voxRes}_withBehav.pkl')
    
    if os.path.exists(results_file):
        print(f"Loading results for {subName} ({freq_band})")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        print(f"Results file not found for {subName}: {results_file}")
        return None

def main(freq_band='alpha', voxRes='8mm'):
    # Subject list
    subject_list = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    
    # Set bids root based on hostname
    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    print("Loading source space decoding results...")
    print(f"BIDS root: {bidsRoot}")
    print(f"Voxel resolution: {voxRes}")
    print(f"Frequency band: {freq_band}")
    print(f"Subject list: {subject_list}")
    
    # Load results for all subjects
    all_results = []
    valid_subjects = []
    
    for subjID in subject_list:
        results = load_subject_results(bidsRoot, subjID, voxRes, freq_band=freq_band)
        if results is not None:
            all_results.append(results)
            valid_subjects.append(subjID)
    
    if not all_results:
        print("No valid results found for any subject")
        return
    
    n_subjects = len(all_results)
    print(f"Successfully loaded {n_subjects} subjects: {valid_subjects}")
    
    # Get time vector (should be same for all subjects)
    time_vector = all_results[0]['time_vector'].flatten()
    n_timepoints = len(time_vector)
    
    # Initialize aggregate arrays for top row
    visual_mean_all = np.empty((n_subjects, n_timepoints))
    frontal_mean_all = np.empty((n_subjects, n_timepoints))
    
    # Initialize aggregate arrays for bottom row (quantile analysis)
    n_quantiles = 4
    visual_mean_quantiles = np.empty((n_quantiles, n_subjects, n_timepoints))
    frontal_mean_quantiles = np.empty((n_quantiles, n_subjects, n_timepoints))
    
    # Process each subject
    for s_idx, results in enumerate(all_results):
        # Extract signed angular errors (trials, timepoints)
        angular_errors_visual = results['angular_errors_visual']
        angular_errors_frontal = results['angular_errors_frontal']
        i_sacc_err = results['i_sacc_err']
        
        # Compute circular mean across trials for each time point (top row)
        visual_mean_all[s_idx, :] = circmean(angular_errors_visual, axis=0, high=180, low=-180)
        frontal_mean_all[s_idx, :] = circmean(angular_errors_frontal, axis=0, high=180, low=-180)
        
        # Filter trials with saccade error above threshold for quantile analysis
        errThresh = 0.001
        valid_trials = i_sacc_err > errThresh
        
        sub_errors_visual = angular_errors_visual[valid_trials, :]
        sub_errors_frontal = angular_errors_frontal[valid_trials, :]
        sub_sacc_err = i_sacc_err[valid_trials]
        
        if len(sub_sacc_err) >= n_quantiles:
            # Compute quantiles for this subject
            quantile_thresholds = np.percentile(sub_sacc_err, [25, 50, 75])
            
            # Assign trials to quantiles (0 = Q1 (best), 3 = Q4 (worst))
            quantile_labels = np.zeros(len(sub_sacc_err), dtype=int)
            quantile_labels[sub_sacc_err <= quantile_thresholds[0]] = 0  # Q1
            quantile_labels[(sub_sacc_err > quantile_thresholds[0]) & (sub_sacc_err <= quantile_thresholds[1])] = 1  # Q2
            quantile_labels[(sub_sacc_err > quantile_thresholds[1]) & (sub_sacc_err <= quantile_thresholds[2])] = 2  # Q3
            quantile_labels[sub_sacc_err > quantile_thresholds[2]] = 3  # Q4
            
            # Compute circular mean for each quantile and time point
            for q in range(n_quantiles):
                q_mask = quantile_labels == q
                if np.sum(q_mask) > 0:
                    visual_mean_quantiles[q, s_idx, :] = circmean(sub_errors_visual[q_mask, :], axis=0, high=180, low=-180)
                    frontal_mean_quantiles[q, s_idx, :] = circmean(sub_errors_frontal[q_mask, :], axis=0, high=180, low=-180)
                else:
                    visual_mean_quantiles[q, s_idx, :] = np.nan
                    frontal_mean_quantiles[q, s_idx, :] = np.nan
        else:
            visual_mean_quantiles[:, s_idx, :] = np.nan
            frontal_mean_quantiles[:, s_idx, :] = np.nan

    # Compute group averages across subjects (top row)
    visual_mean = np.abs(circmean(visual_mean_all, axis=0, high=180, low=-180))
    frontal_mean = np.abs(circmean(frontal_mean_all, axis=0, high=180, low=-180))
    
    # Bottom row window calculation: delay period (0.5 to 1.5s)
    delay_mask = (time_vector >= 0.2) & (time_vector <= 1.7)
    
    # Calculate mean over time for each subject and quantile during delay
    visual_subject_means = np.zeros((n_quantiles, n_subjects))
    frontal_subject_means = np.zeros((n_quantiles, n_subjects))
    
    for q in range(n_quantiles):
        for s_idx in range(n_subjects):
            v_q_sub = visual_mean_quantiles[q, s_idx, delay_mask]
            f_q_sub = frontal_mean_quantiles[q, s_idx, delay_mask]
            
            # Remove nans if any
            v_q_sub = v_q_sub[~np.isnan(v_q_sub)]
            f_q_sub = f_q_sub[~np.isnan(f_q_sub)]
            
            if len(v_q_sub) > 0:
                visual_subject_means[q, s_idx] = np.abs(circmean(v_q_sub, high=180, low=-180))
            else:
                visual_subject_means[q, s_idx] = np.nan
                
            if len(f_q_sub) > 0:
                frontal_subject_means[q, s_idx] = np.abs(circmean(f_q_sub, high=180, low=-180))
            else:
                frontal_subject_means[q, s_idx] = np.nan

    # Compute mean and SEM across subjects for bar plotting
    visual_mean_window = np.nanmean(visual_subject_means, axis=1)
    frontal_mean_window = np.nanmean(frontal_subject_means, axis=1)
    
    if n_subjects > 1:
        visual_sem_window = np.nanstd(visual_subject_means, axis=1) / np.sqrt(np.sum(~np.isnan(visual_subject_means), axis=1))
        frontal_sem_window = np.nanstd(frontal_subject_means, axis=1) / np.sqrt(np.sum(~np.isnan(frontal_subject_means), axis=1))
    else:
        visual_sem_window = np.zeros(n_quantiles)
        frontal_sem_window = np.zeros(n_quantiles)
        
    # Statistical testing: one-sample t-test against 90 degrees (chance)
    visual_p_values = np.ones(n_quantiles)
    frontal_p_values = np.ones(n_quantiles)
    
    if n_subjects > 1:
        for q in range(n_quantiles):
            v_sub_data = visual_subject_means[q, :]
            v_sub_data = v_sub_data[~np.isnan(v_sub_data)]
            if len(v_sub_data) > 1:
                _, p_val = ttest_1samp(v_sub_data, 90)
                visual_p_values[q] = p_val
                
            f_sub_data = frontal_subject_means[q, :]
            f_sub_data = f_sub_data[~np.isnan(f_sub_data)]
            if len(f_sub_data) > 1:
                _, p_val = ttest_1samp(f_sub_data, 90)
                frontal_p_values[q] = p_val

    # Grayscale palette for quantile bars (darkest=best, lightest=worst)
    bar_colors = ['#333333', '#666666', '#999999', '#cccccc']
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # ------------------------------------------------------------------------
    # Top Row: Scatter plots of decoding error vs time (B and C)
    # ------------------------------------------------------------------------
    # Left subplot (0,0): Visual Cortex (B) - orange dots
    axes[0,0].scatter(time_vector, visual_mean, c='#E8602C', s=60, edgecolors='none')
    axes[0,0].set_title('Visual Cortex', fontsize=14, fontstyle='italic', fontweight='bold')
    axes[0,0].set_xlabel('Time (s)', fontsize=12)
    axes[0,0].set_ylabel('Decoding Error (°)', fontsize=12)
    axes[0,0].axvline(x=0.0,  color='black', linestyle='--', linewidth=1.2)
    axes[0,0].axvline(x=0.2,  color='black', linestyle='--', linewidth=1.2)
    axes[0,0].set_xlim(-0.5, 1.7)
    axes[0,0].set_ylim(0, 180)
    axes[0,0].set_yticks([0, 60, 120, 180])
    axes[0,0].grid(False)
    
    # Right subplot (0,1): Prefrontal Cortex (C) - purple dots
    axes[0,1].scatter(time_vector, frontal_mean, c='#7B3FA0', s=60, edgecolors='none')
    axes[0,1].set_title('Prefrontal Cortex', fontsize=14, fontstyle='italic', fontweight='bold')
    axes[0,1].set_xlabel('Time (s)', fontsize=12)
    axes[0,1].set_ylabel('Decoding Error (°)', fontsize=12)
    axes[0,1].axvline(x=0.0,  color='black', linestyle='--', linewidth=1.2)
    axes[0,1].axvline(x=0.2,  color='black', linestyle='--', linewidth=1.2)
    axes[0,1].set_xlim(-0.5, 1.7)
    axes[0,1].set_ylim(0, 180)
    axes[0,1].set_yticks([0, 60, 120, 180])
    axes[0,1].grid(False)
    
    # ------------------------------------------------------------------------
    # Bottom Row: Bar plots of mean decoding error (0.5 to 1.5s Delay)
    # ------------------------------------------------------------------------
    # Left subplot (1,0): Visual Cortex - Mean Error (Delay) (D) - solid chance line
    axes[1,0].bar(range(n_quantiles), visual_mean_window, yerr=visual_sem_window,
                  color=bar_colors, capsize=5, error_kw={'linewidth': 1.5, 'capthick': 1.5},
                  edgecolor='none')
    axes[1,0].axhline(y=90, color='black', linestyle='-', linewidth=1.2)  # solid line (D)
    
    # Add significance markers (* for p < 0.05)
    for q in range(n_quantiles):
        if visual_p_values[q] < 0.05:
            y_pos = visual_mean_window[q] + visual_sem_window[q]
            axes[1,0].text(q, y_pos + 2, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
            
    axes[1,0].set_xticks(range(n_quantiles))
    axes[1,0].set_xticklabels(['Best', '', '', 'Worst'], fontsize=11)
    axes[1,0].set_title('Visual Cortex', fontsize=14, fontstyle='italic', fontweight='bold')
    axes[1,0].set_ylabel('Decoding Error (°)', fontsize=12)
    axes[1,0].set_xlabel('Memory Performance', fontsize=11)
    # Auto-scale y-axis: always show 90° chance line, pad around data range
    v_min = max(0, np.nanmin(visual_mean_window - visual_sem_window) - 10)
    v_max = min(180, max(95, np.nanmax(visual_mean_window + visual_sem_window) + 10))
    axes[1,0].set_ylim(20, 120)
    axes[1,0].grid(False)
    
    # Right subplot (1,1): Prefrontal Cortex - Mean Error (Delay) (E) - dashed chance line
    axes[1,1].bar(range(n_quantiles), frontal_mean_window, yerr=frontal_sem_window,
                  color=bar_colors, capsize=5, error_kw={'linewidth': 1.5, 'capthick': 1.5},
                  edgecolor='none')
    axes[1,1].axhline(y=90, color='gray', linestyle='--', linewidth=1.2)  # dashed line (E)
    
    # Add significance markers (* for p < 0.05)
    for q in range(n_quantiles):
        if frontal_p_values[q] < 0.05:
            y_pos = frontal_mean_window[q] + frontal_sem_window[q]
            axes[1,1].text(q, y_pos + 2, '*', ha='center', va='bottom', fontsize=20, fontweight='bold')
            
    axes[1,1].set_xticks(range(n_quantiles))
    axes[1,1].set_xticklabels(['Best', '', '', 'Worst'], fontsize=11)
    axes[1,1].set_title('Prefrontal Cortex', fontsize=14, fontstyle='italic', fontweight='bold')
    axes[1,1].set_ylabel('Decoding Error (°)', fontsize=12)
    axes[1,1].set_xlabel('Memory Performance', fontsize=11)
    # Auto-scale y-axis: always show 90° chance line, pad around data range
    f_min = max(0, np.nanmin(frontal_mean_window - frontal_sem_window) - 10)
    f_max = min(180, max(95, np.nanmax(frontal_mean_window + frontal_sem_window) + 10))
    axes[1,1].set_ylim(20, 120)
    axes[1,1].grid(False)
    
    # Title and layout
    plt.suptitle(f'SVR Decoding Results by i_sacc_err Quantile - All Subjects (n={n_subjects})', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save figures
    output_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs03')
    os.makedirs(output_dir, exist_ok=True)
    
    save_path_svg = os.path.join(output_dir, f'{freq_band}_svr_decoding.svg')
    fig.savefig(save_path_svg, format='svg', bbox_inches='tight')
    print(f"Figure saved as SVG to {save_path_svg}")
    
    save_path_png = os.path.join(output_dir, f'{freq_band}_svr_decoding.png')
    fig.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f"Figure saved as PNG to {save_path_png}")
    
    plt.close(fig)

if __name__ == '__main__':
    import sys
    freq_band = 'alpha'
    voxRes = '8mm'
    if len(sys.argv) > 1:
        freq_band = sys.argv[1]
    if len(sys.argv) > 2:
        voxRes = sys.argv[2]
        
    print(f"Running SVR plotting for band: {freq_band}, voxel resolution: {voxRes}")
    main(freq_band=freq_band, voxRes=voxRes)