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
from scipy.stats import circmean, circstd, ttest_1samp, pearsonr
from sklearn.linear_model import TheilSenRegressor

def load_subject_results(bidsRoot, subjID, voxRes='10mm', freq_band='theta'):
    """Load results for a single subject"""
    subName = f'sub-{subjID:02d}'
    results_file = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', 
                               'decodingVC', f'{subName}_task-mgs_SVR_{freq_band}_{voxRes}_withBehav.pkl')
    
    if os.path.exists(results_file):
        print(f"Loading results for {subName}")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        print(f"Results file not found for {subName}: {results_file}")
        return None

def main(freq_band='theta'):
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
            print(f"Loaded results for subject {subjID}")
        else:
            print(f"No results found for subject {subjID}")
    
    if not all_results:
        print("No valid results found for any subject")
        return
    
    print(f"Successfully loaded {len(all_results)} subjects: {valid_subjects}")
    
    # Get time vector from first subject (should be same for all)
    time_vector = all_results[0]['time_vector']
    
    # Get frequency band info from first subject if available
    freq_range_str = ""
    if 'freq_range' in all_results[0]:
        f_min, f_max = all_results[0]['freq_range']
        freq_range_str = f" ({f_min}-{f_max} Hz)"
    elif 'freq_band' in all_results[0]:
        freq_range_str = f" ({all_results[0]['freq_band']} band)"
    
    # Initialize aggregate arrays
    n_timepoints = time_vector.shape[0]
    n_subjects = len(all_results)
    
    # Arrays to store mean angular errors for each subject
    visual_mean_all = np.empty((n_subjects, n_timepoints))
    visual_control_mean_all = np.empty((n_subjects, n_timepoints))
    frontal_mean_all = np.empty((n_subjects, n_timepoints))
    frontal_control_mean_all = np.empty((n_subjects, n_timepoints))
    parietal_mean_all = np.empty((n_subjects, n_timepoints))
    parietal_control_mean_all = np.empty((n_subjects, n_timepoints))
    
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
    
    # Process each subject
    for s_idx, results in enumerate(all_results):
        print(f"Processing subject {valid_subjects[s_idx]}...")
        
        # target_labels = results['target_labels']
        # target_angles_deg = np.array([angle_mapping[t] for t in target_labels])
        # target_angles_deg_2d = target_angles_deg[:, np.newaxis]
        
        # pred_angles_visual = results['pred_angles_deg_visual']
        # pred_angles_visual_control = results['pred_angles_deg_visual_control']
        # pred_angles_frontal = results['pred_angles_deg_frontal']
        # pred_angles_frontal_control = results['pred_angles_deg_frontal_control']
        # pred_angles_parietal = results['pred_angles_deg_parietal']
        # pred_angles_parietal_control = results['pred_angles_deg_parietal_control']
        
        # angular_errors_visual = ((pred_angles_visual - target_angles_deg_2d) + 180) % 360 - 180
        # angular_errors_visual_control = ((pred_angles_visual_control - target_angles_deg_2d) + 180) % 360 - 180
        # angular_errors_frontal = ((pred_angles_frontal - target_angles_deg_2d) + 180) % 360 - 180
        # angular_errors_frontal_control = ((pred_angles_frontal_control - target_angles_deg_2d) + 180) % 360 - 180
        # angular_errors_parietal = ((pred_angles_parietal - target_angles_deg_2d) + 180) % 360 - 180
        # angular_errors_parietal_control = ((pred_angles_parietal_control - target_angles_deg_2d) + 180) % 360 - 180
        
        # Extract data for this subject
        angular_errors_visual = results['angular_errors_visual']
        angular_errors_visual_control = results['angular_errors_visual_control']
        angular_errors_frontal = results['angular_errors_frontal']
        angular_errors_frontal_control = results['angular_errors_frontal_control']
        angular_errors_parietal = results['angular_errors_parietal']
        angular_errors_parietal_control = results['angular_errors_parietal_control']
        # print(angular_errors_visual.min(), angular_errors_visual.max())
        
        # Compute circular mean across trials for each time point
        visual_mean_all[s_idx, :] = circmean(angular_errors_visual, axis=0, high=180, low=-180)
        visual_control_mean_all[s_idx, :] = circmean(angular_errors_visual_control, axis=0, high=180, low=-180)
        frontal_mean_all[s_idx, :] = circmean(angular_errors_frontal, axis=0, high=180, low=-180)
        frontal_control_mean_all[s_idx, :] = circmean(angular_errors_frontal_control, axis=0, high=180, low=-180)
        parietal_mean_all[s_idx, :] = circmean(angular_errors_parietal, axis=0, high=180, low=-180)
        parietal_control_mean_all[s_idx, :] = circmean(angular_errors_parietal_control, axis=0, high=180, low=-180)
    
    # Compute aggregate statistics across subjects
    visual_mean = np.abs(circmean(visual_mean_all, axis=0, high=180, low=-180))
    frontal_mean = np.abs(circmean(frontal_mean_all, axis=0, high=180, low=-180))
    parietal_mean = np.abs(circmean(parietal_mean_all, axis=0, high=180, low=-180))
    visual_control_mean = np.abs(circmean(visual_control_mean_all, axis=0, high=180, low=-180))
    frontal_control_mean = np.abs(circmean(frontal_control_mean_all, axis=0, high=180, low=-180))
    parietal_control_mean = np.abs(circmean(parietal_control_mean_all, axis=0, high=180, low=-180))

    # ============================================================================
    # CORRELATION ANALYSIS: i_sacc_angle vs predicted angles
    # ============================================================================
    print("\n" + "="*70)
    print("Computing correlations between i_sacc_angle and predicted angles...")
    print("="*70)
    
    # Initialize arrays to store correlations for each subject and time point
    correlations_visual_all = np.empty((n_subjects, n_timepoints))
    correlations_parietal_all = np.empty((n_subjects, n_timepoints))
    correlations_frontal_all = np.empty((n_subjects, n_timepoints))
    correlations_visual_control_all = np.empty((n_subjects, n_timepoints))
    correlations_parietal_control_all = np.empty((n_subjects, n_timepoints))
    correlations_frontal_control_all = np.empty((n_subjects, n_timepoints))
    # Function to compute circular correlation
    def circular_correlation(angles1, angles2):
        """Compute circular correlation coefficient between two sets of angles (in degrees)"""
        # Convert to radians
        angles1_rad = np.deg2rad(angles1)
        angles2_rad = np.deg2rad(angles2)
        
        # Center the angles
        angles1_centered = angles1_rad - circmean(angles1_rad, high=2*np.pi, low=0)
        angles2_centered = angles2_rad - circmean(angles2_rad, high=2*np.pi, low=0)
        
        # Compute circular correlation
        numerator = np.sum(np.sin(angles1_centered) * np.sin(angles2_centered))
        denom1 = np.sqrt(np.sum(np.sin(angles1_centered)**2))
        denom2 = np.sqrt(np.sum(np.sin(angles2_centered)**2))
        
        if denom1 == 0 or denom2 == 0:
            return 0.0
        
        r = numerator / (denom1 * denom2)
        return r
    
    # Process each subject to compute correlations
    for s_idx, results in enumerate(all_results):
        print(f"Computing correlations for subject {valid_subjects[s_idx]}...")
        
        # Extract data
        i_sacc_angle = results['i_sacc_angle']
        pred_angles_visual = results['pred_angles_deg_visual']
        pred_angles_parietal = results['pred_angles_deg_parietal']
        pred_angles_frontal = results['pred_angles_deg_frontal']
        pred_angles_visual_control = results['pred_angles_deg_visual_control']
        pred_angles_parietal_control = results['pred_angles_deg_parietal_control']
        pred_angles_frontal_control = results['pred_angles_deg_frontal_control']
        
        # Compute correlation at each time point
        for t_idx in range(n_timepoints):
            correlations_visual_all[s_idx, t_idx] = circular_correlation(
                i_sacc_angle, pred_angles_visual[:, t_idx])
            correlations_parietal_all[s_idx, t_idx] = circular_correlation(
                i_sacc_angle, pred_angles_parietal[:, t_idx])
            correlations_frontal_all[s_idx, t_idx] = circular_correlation(
                i_sacc_angle, pred_angles_frontal[:, t_idx])
            correlations_visual_control_all[s_idx, t_idx] = circular_correlation(
                i_sacc_angle, pred_angles_visual_control[:, t_idx])
            correlations_parietal_control_all[s_idx, t_idx] = circular_correlation(
                i_sacc_angle, pred_angles_parietal_control[:, t_idx])
            correlations_frontal_control_all[s_idx, t_idx] = circular_correlation(
                i_sacc_angle, pred_angles_frontal_control[:, t_idx])
            
    # Compute mean correlations across subjects
    correlations_visual_mean = np.mean(correlations_visual_all, axis=0)
    correlations_parietal_mean = np.mean(correlations_parietal_all, axis=0)
    correlations_frontal_mean = np.mean(correlations_frontal_all, axis=0)
    correlations_visual_control_mean = np.mean(correlations_visual_control_all, axis=0)
    correlations_parietal_control_mean = np.mean(correlations_parietal_control_all, axis=0)
    correlations_frontal_control_mean = np.mean(correlations_frontal_control_all, axis=0)
    correlations_visual_sem = np.std(correlations_visual_all, axis=0) / np.sqrt(n_subjects)
    correlations_parietal_sem = np.std(correlations_parietal_all, axis=0) / np.sqrt(n_subjects)
    correlations_frontal_sem = np.std(correlations_frontal_all, axis=0) / np.sqrt(n_subjects)
    correlations_visual_control_sem = np.std(correlations_visual_control_all, axis=0) / np.sqrt(n_subjects)
    correlations_parietal_control_sem = np.std(correlations_parietal_control_all, axis=0) / np.sqrt(n_subjects)
    correlations_frontal_control_sem = np.std(correlations_frontal_control_all, axis=0) / np.sqrt(n_subjects)
    # Create figure with 2x3 subplots (2 rows, 3 columns: visual, parietal, frontal)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot Visual region
    # axes[0].scatter(time_vector.flatten(), visual_control_mean, c='gray', s=20, alpha=0.7, label='Control')
    axes[0,0].scatter(time_vector.flatten(), visual_mean, c='blue', s=75)
    axes[0,0].scatter(time_vector.flatten(), visual_control_mean, c='gray', s=50, alpha=0.3)
    axes[0,0].set_title('Visual Region')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Decoding error')
    axes[0,0].axvline(x=0, color='black', linestyle='--', alpha=0.8)
    axes[0,0].axhline(y=0, color='black', linestyle='--', alpha=0.8)
    axes[0,0].grid(False)
    axes[0,0].set_xlim(-0.5, 1.7)
    # axes[0,0].set_ylim(0, 180)
    
    # Plot Parietal region
    axes[0,1].scatter(time_vector.flatten(), parietal_mean, c='red', s=75)
    axes[0,1].scatter(time_vector.flatten(), parietal_control_mean, c='gray', s=50, alpha=0.3)
    axes[0,1].set_title('Parietal Region')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Decoding error')
    axes[0,1].axvline(x=0, color='black', linestyle='--', alpha=0.8)
    axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.8)
    axes[0,1].grid(False)
    axes[0,1].set_xlim(-0.5, 1.7)
    # axes[0,1].set_ylim(0, 180)
    
    # Plot Frontal region
    # axes[2].scatter(time_vector.flatten(), frontal_control_mean, c='gray', s=20, alpha=0.7, label='Control')
    axes[0,2].scatter(time_vector.flatten(), frontal_mean, c='green', s=75)
    axes[0,2].scatter(time_vector.flatten(), frontal_control_mean, c='gray', s=50, alpha=0.3)
    axes[0,2].set_title('Frontal Region')
    axes[0,2].set_xlabel('Time (s)')
    axes[0,2].set_ylabel('Decoding error')
    axes[0,2].axvline(x=0, color='black', linestyle='--', alpha=0.8)
    axes[0,2].axhline(y=0, color='black', linestyle='--', alpha=0.8)
    axes[0,2].grid(False)
    axes[0,2].set_xlim(-0.5, 1.7)
    # axes[0,2].set_ylim(0, 180)
    
    # ============================================================================
    # QUANTILE ANALYSIS BASED ON i_sacc_err
    # ============================================================================
    print("\n" + "="*70)
    print("Starting quantile analysis based on i_sacc_err...")
    print("="*70)
    
    # Initialize arrays for quantile analysis (4 quantiles × n_subjects × n_timepoints)
    n_quantiles = 4
    visual_mean_quantiles = np.empty((n_quantiles, n_subjects, n_timepoints))
    frontal_mean_quantiles = np.empty((n_quantiles, n_subjects, n_timepoints))
    parietal_mean_quantiles = np.empty((n_quantiles, n_subjects, n_timepoints))
    
    # Process each subject for quantile analysis
    for s_idx, results in enumerate(all_results):
        subjID = valid_subjects[s_idx]
        print(f"Processing quantiles for subject {subjID}...")
        
        # Load behavioral data (i_sacc_err)
        i_sacc_err = results['i_sacc_err']
        # i_sacc_angle = results['i_sacc_angle']
        # pred_angles_visual = results['pred_angles_deg_visual']
        # pred_angles_parietal = results['pred_angles_deg_parietal']
        # pred_angles_frontal = results['pred_angles_deg_frontal']

        # target_labels = results['target_labels'] # This is a numpy array of label numbers (1-10)
        # target_angles_deg = np.array([angle_mapping[t] for t in target_labels])
        
        # # Reshape target_angles_deg to (n_trials, 1) for broadcasting with (n_trials, n_timepoints)
        # target_angles_deg_2d = target_angles_deg[:, np.newaxis]

        # i_sacc_err = (i_sacc_angle - target_angles_deg) % 360
       
        # angular_errors_visual = (pred_angles_visual - target_angles_deg_2d) % 360
        # angular_errors_parietal = (pred_angles_parietal - target_angles_deg_2d) % 360
        # angular_errors_frontal = (pred_angles_frontal - target_angles_deg_2d) % 360
        
        # Extract angular errors for this subject
        angular_errors_visual = results['angular_errors_visual']
        angular_errors_frontal = results['angular_errors_frontal']
        angular_errors_parietal = results['angular_errors_parietal']

        errThresh = 0.001
    
        # Only include trials with i_sacc_err > 0.001 degrees
        angular_errors_visual = angular_errors_visual[i_sacc_err > errThresh, :]
        angular_errors_frontal = angular_errors_frontal[i_sacc_err > errThresh, :]
        angular_errors_parietal = angular_errors_parietal[i_sacc_err > errThresh, :]
        # Only include i_sacc_err > 0.001 degrees
        i_sacc_err = i_sacc_err[i_sacc_err > errThresh]
        
        # Compute quantiles for this subject
        quantile_thresholds = np.percentile(i_sacc_err, [25, 50, 75])
        # print(quantile_thresholds)
        # exit()
        
        # Assign trials to quantiles (1 = best performance, 4 = worst performance)
        quantile_labels = np.zeros(len(i_sacc_err), dtype=int)
        quantile_labels[i_sacc_err <= quantile_thresholds[0]] = 0  # Q1 (best)
        quantile_labels[(i_sacc_err > quantile_thresholds[0]) & (i_sacc_err <= quantile_thresholds[1])] = 1  # Q2
        quantile_labels[(i_sacc_err > quantile_thresholds[1]) & (i_sacc_err <= quantile_thresholds[2])] = 2  # Q3
        quantile_labels[i_sacc_err > quantile_thresholds[2]] = 3  # Q4 (worst)
        
        # Compute circular mean for each quantile and time point
        for q in range(n_quantiles):
            q_mask = quantile_labels == q
            if np.sum(q_mask) > 0:
                visual_mean_quantiles[q, s_idx, :] = circmean(angular_errors_visual[q_mask, :], axis=0, high=180, low=-180)
                frontal_mean_quantiles[q, s_idx, :] = circmean(angular_errors_frontal[q_mask, :], axis=0, high=180, low=-180)
                parietal_mean_quantiles[q, s_idx, :] = circmean(angular_errors_parietal[q_mask, :], axis=0, high=180, low=-180)
    
    # Compute aggregate statistics across subjects for each quantile
    visual_mean_q = np.abs(circmean(visual_mean_quantiles, axis=1, high=180, low=-180))
    frontal_mean_q = np.abs(circmean(frontal_mean_quantiles, axis=1, high=180, low=-180))
    parietal_mean_q = np.abs(circmean(parietal_mean_quantiles, axis=1, high=180, low=-180))
    
    # Define colors for quantiles (best to worst)
    quantile_colors = ['red', 'orange', 'green', 'blue']
    quantile_labels_text = ['Q1 (Best)', 'Q2', 'Q3', 'Q4 (Worst)']
    
    # --- Second row: Mean decoding error for each quantile ---
    # Set time window based on frequency band
    # Theta: 0.25 to 0.5s, Beta/Alpha: 0.25 to 1.5s
    if freq_band == 'theta':
        time_window_start = 0.1
        time_window_end = 0.5
    elif freq_band == 'alpha':
        time_window_start = 0.25
        time_window_end = 1.0
    elif freq_band == 'beta':
        time_window_start = 0.25
        time_window_end = 1.7
    else:  # lowgamma
        time_window_start = 0.25
        time_window_end = 1.5
    
    # Find time indices for the specified time window
    time_vec_flat = time_vector.flatten()
    tidx_mean_start = np.where(time_vec_flat >= time_window_start)[0]
    tidx_mean_end = np.where(time_vec_flat <= time_window_end)[0]
    
    if len(tidx_mean_start) > 0 and len(tidx_mean_end) > 0:
        tidx_mean_start_val = tidx_mean_start[0]
        tidx_mean_end_val = tidx_mean_end[-1]
        
        # Calculate mean over time for each quantile and subject, then aggregate across subjects
        # visual_mean_quantiles has shape (n_quantiles, n_subjects, n_timepoints)
        visual_subject_means = np.zeros((n_quantiles, n_subjects))  # Mean over time for each subject
        frontal_subject_means = np.zeros((n_quantiles, n_subjects))
        parietal_subject_means = np.zeros((n_quantiles, n_subjects))
        
        for q in range(n_quantiles):
            for s_idx in range(n_subjects):
                # Compute circular mean over time window (1D array, no axis needed)
                visual_subject_means[q, s_idx] = np.abs(circmean(visual_mean_quantiles[q, s_idx, tidx_mean_start_val:tidx_mean_end_val + 1], high=180, low=-180))
                frontal_subject_means[q, s_idx] = np.abs(circmean(frontal_mean_quantiles[q, s_idx, tidx_mean_start_val:tidx_mean_end_val + 1], high=180, low=-180))
                parietal_subject_means[q, s_idx] = np.abs(circmean(parietal_mean_quantiles[q, s_idx, tidx_mean_start_val:tidx_mean_end_val + 1], high=180, low=-180))
        
        # Compute mean and SEM across subjects for each quantile
        visual_mean_window = np.mean(visual_subject_means, axis=1)
        visual_sem_window = np.std(visual_subject_means, axis=1) / np.sqrt(n_subjects)
        
        frontal_mean_window = np.mean(frontal_subject_means, axis=1)
        frontal_sem_window = np.std(frontal_subject_means, axis=1) / np.sqrt(n_subjects)
        
        parietal_mean_window = np.mean(parietal_subject_means, axis=1)
        parietal_sem_window = np.std(parietal_subject_means, axis=1) / np.sqrt(n_subjects)
        
        # Statistical testing: one-sample t-test against 90 degrees for each quantile
        visual_p_values = np.zeros(n_quantiles)
        frontal_p_values = np.zeros(n_quantiles)
        parietal_p_values = np.zeros(n_quantiles)
        
        for q in range(n_quantiles):
            # Test if mean is significantly different from 90
            t_stat_visual, visual_p_values[q] = ttest_1samp(visual_subject_means[q, :], 90)
            t_stat_frontal, frontal_p_values[q] = ttest_1samp(frontal_subject_means[q, :], 90)
            t_stat_parietal, parietal_p_values[q] = ttest_1samp(parietal_subject_means[q, :], 90)
        
        # # Plot Visual region mean errors with error bars
        # bars_visual = axes[1,0].bar(range(n_quantiles), visual_mean_window, yerr=visual_sem_window, 
        #              color=quantile_colors, alpha=0.7, capsize=5, error_kw={'linewidth': 2})
        # axes[1,0].axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='Chance Level (~90°)')
        # 
        # # Add significance markers
        # for q in range(n_quantiles):
        #     if visual_p_values[q] < 0.05:
        #         y_pos = visual_mean_window[q] + visual_sem_window[q]
        #         axes[1,0].text(q, y_pos + 5, '*', ha='center', va='bottom', fontsize=14, fontweight='bold')
        # 
        # axes[1,0].set_xticks(range(n_quantiles))
        # axes[1,0].set_xticklabels([f'Q{q+1}' for q in range(n_quantiles)])
        # axes[1,0].set_title('Visual Region - Mean Error (Delay)')
        # axes[1,0].set_ylabel('Mean Decoding Error')
        # axes[1,0].set_ylim(20, 120)
        # axes[1,0].grid(False)
        # 
        # # Plot Parietal region mean errors with error bars
        # bars_parietal = axes[1,1].bar(range(n_quantiles), parietal_mean_window, yerr=parietal_sem_window,
        #              color=quantile_colors, alpha=0.7, capsize=5, error_kw={'linewidth': 2})
        # axes[1,1].axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='Chance Level (~90°)')
        # 
        # # Add significance markers
        # for q in range(n_quantiles):
        #     if parietal_p_values[q] < 0.05:
        #         y_pos = parietal_mean_window[q] + parietal_sem_window[q]
        #         axes[1,1].text(q, y_pos + 5, '*', ha='center', va='bottom', fontsize=14, fontweight='bold')
        # 
        # axes[1,1].set_xticks(range(n_quantiles))
        # axes[1,1].set_xticklabels([f'Q{q+1}' for q in range(n_quantiles)])
        # axes[1,1].set_title('Parietal Region - Mean Error (Delay)')
        # axes[1,1].set_ylabel('Mean Decoding Error')
        # axes[1,1].grid(False)
        # axes[1,1].set_ylim(20, 120)
        # 
        # # Plot Frontal region mean errors with error bars
        # bars_frontal = axes[1,2].bar(range(n_quantiles), frontal_mean_window, yerr=frontal_sem_window,
        #              color=quantile_colors, alpha=0.7, capsize=5, error_kw={'linewidth': 2})
        # axes[1,2].axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='Chance Level (~90°)')
        # 
        # # Add significance markers
        # for q in range(n_quantiles):
        #     if frontal_p_values[q] < 0.05:
        #         y_pos = frontal_mean_window[q] + frontal_sem_window[q]
        #         axes[1,2].text(q, y_pos + 5, '*', ha='center', va='bottom', fontsize=14, fontweight='bold')
        # 
        # axes[1,2].set_xticks(range(n_quantiles))
        # axes[1,2].set_xticklabels([f'Q{q+1}' for q in range(n_quantiles)])
        # axes[1,2].set_title('Frontal Region - Mean Error (Delay)')
        # axes[1,2].set_ylabel('Mean Decoding Error')
        # axes[1,2].grid(False)
        # axes[1,2].set_ylim(20, 120)
    else:
        print(f"Warning: Could not find time indices for 0.25-{time_window_end}s range")
    
    # ============================================================================
    # SECOND ROW: CORRELATION PLOTS
    # ============================================================================
    time_vec_flat = time_vector.flatten()
    
    # Plot Visual region correlation
    axes[1,0].fill_between(time_vec_flat, 
                            correlations_visual_mean - correlations_visual_sem,
                            correlations_visual_mean + correlations_visual_sem,
                            color='blue', alpha=0.3)
    axes[1,0].fill_between(time_vec_flat, 
                            correlations_visual_control_mean - correlations_visual_control_sem,
                            correlations_visual_control_mean + correlations_visual_control_sem,
                            color='gray', alpha=0.15)
    axes[1,0].plot(time_vec_flat, correlations_visual_mean, c='blue', linewidth=2)
    axes[1,0].plot(time_vec_flat, correlations_visual_control_mean, c='gray', linewidth=1, alpha=0.5)
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].axvline(x=0, color='black', linestyle='--', alpha=0.8)

    axes[1,0].set_title('Visual Region')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Corr w/ behav')
    axes[1,0].set_xlim(-0.5, 1.7)
    axes[1,0].set_ylim(-0.125, 0.125)
    axes[1,0].grid(False)

   
    
    # Plot Parietal region correlation
    axes[1,1].fill_between(time_vec_flat,
                            correlations_parietal_mean - correlations_parietal_sem,
                            correlations_parietal_mean + correlations_parietal_sem,
                            color='red', alpha=0.3)
    axes[1,1].fill_between(time_vec_flat, 
                            correlations_parietal_control_mean - correlations_parietal_control_sem,
                            correlations_parietal_control_mean + correlations_parietal_control_sem,
                            color='gray', alpha=0.15)
    axes[1,1].plot(time_vec_flat, correlations_parietal_mean, c='red', linewidth=2)
    axes[1,1].plot(time_vec_flat, correlations_parietal_control_mean, c='gray', linewidth=1, alpha=0.5)
    axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,1].axvline(x=0, color='black', linestyle='--', alpha=0.8)
    axes[1,1].set_title('Parietal Region')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Corr w/ behav')
    axes[1,1].set_xlim(-0.5, 1.7)
    axes[1,1].set_ylim(-0.125, 0.125) 
    axes[1,1].grid(False)
    
    # Plot Frontal region correlation
    axes[1,2].fill_between(time_vec_flat,
                            correlations_frontal_mean - correlations_frontal_sem,
                            correlations_frontal_mean + correlations_frontal_sem,
                            color='green', alpha=0.3)
    axes[1,2].fill_between(time_vec_flat, 
                            correlations_frontal_control_mean - correlations_frontal_control_sem,
                            correlations_frontal_control_mean + correlations_frontal_control_sem,
                            color='gray', alpha=0.15)
    axes[1,2].plot(time_vec_flat, correlations_frontal_mean, c='green', linewidth=2)
    axes[1,2].plot(time_vec_flat, correlations_frontal_control_mean, c='gray', linewidth=1, alpha=0.5)
    axes[1,2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,2].axvline(x=0, color='black', linestyle='--', alpha=0.8)
    axes[1,2].set_title('Frontal Region')
    axes[1,2].set_xlabel('Time (s)')
    axes[1,2].set_ylabel('Corr w/ behav')
    axes[1,2].set_xlim(-0.5, 1.7)
    axes[1,2].set_ylim(-0.125, 0.125)
    axes[1,2].grid(False)
    
    plt.suptitle(freq_band.upper(), fontsize=16)
    plt.tight_layout()

    # Save figure 1 as SVG
    output_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'decodPlots')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'decoding_results_{voxRes}_{freq_band}.svg')
    fig.savefig(save_path, format='svg', bbox_inches='tight')
    save_path = os.path.join(output_dir, f'decoding_results_{voxRes}_{freq_band}.png')
    fig.savefig(save_path, format='png', bbox_inches='tight')
    print(f"Figure 1 saved as SVG to {save_path}")
    
    plt.show()
    
   

if __name__ == '__main__':
    import sys
    freq_band = sys.argv[1] if len(sys.argv) > 1 else 'theta'
    print(f"Plotting results for frequency band: {freq_band}")
    main(freq_band=freq_band)