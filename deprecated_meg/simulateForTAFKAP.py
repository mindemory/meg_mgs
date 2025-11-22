#!/usr/bin/env python3
"""
Simulate voxel response to a stimulus using double gamma HRF.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import math
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut

# Set random seed for reproducibility
np.random.seed(42)

def get_double_gamma_hrf(tr=1.2, duration=32):
    """
    Generate double gamma HRF (Hemodynamic Response Function).
    
    Parameters:
    -----------
    tr : float
        TR duration in seconds (default: 1.2)
    duration : float
        Duration of HRF in seconds (default: 32)
        
    Returns:
    --------
    tuple
        (hrf, time_points) where hrf is the HRF function and time_points are time values
    """
    # Time points
    time_points = np.arange(0, duration, tr)
    
    # Double gamma parameters (from Glover, 1999)
    a1, a2 = 6, 16  # Shape parameters
    b1, b2 = 1, 1   # Scale parameters
    c = 1/6         # Ratio parameter
    
    # First gamma (positive)
    gamma1 = (time_points**(a1-1) * np.exp(-time_points/b1)) / (b1**a1 * math.factorial(a1-1))
    
    # Second gamma (negative)
    gamma2 = (time_points**(a2-1) * np.exp(-time_points/b2)) / (b2**a2 * math.factorial(a2-1))
    
    # Double gamma HRF
    hrf = gamma1 - c * gamma2
    
    # Normalize
    hrf = hrf / np.max(hrf)
    
    return hrf, time_points

def get_voxel_response(stimulus_angle, preferred_angle=30.0, tuning_sd=30.0):
    """
    Get response of a voxel to a stimulus angle.
    This voxel has Gaussian tuning centered on the preferred angle.
    """
    # Calculate angular distance (handle circular nature)
    angular_diff = np.abs(stimulus_angle - preferred_angle)
    # Take the minimum of the two possible distances around the circle
    angular_diff = np.minimum(angular_diff, 360 - angular_diff)
    
    # Calculate Gaussian response
    response = np.exp(-0.5 * (angular_diff / tuning_sd)**2)
    
    return response

def main():
    """Main function to simulate and visualize 360 voxels with different tuning."""
    # Parameters
    tr = 1.0  # TR duration in seconds
    stimulus_interval = 15  # Stimulus every 15 seconds
    n_trials = 100  # Number of trials
    n_voxels = 360  # Number of voxels
    total_duration = n_trials * stimulus_interval  # Total duration in seconds
    n_trs = int(total_duration / tr)  # Number of TRs

    voxSNR = 0.1
    
    # Possible stimulus angles
    possible_angles = [30, 60, 120, 150, 210, 240, 300, 330]
    
    # Randomly select angles for each trial with jitter
    np.random.seed(42)  # For reproducibility
    base_angles = np.random.choice(possible_angles, size=n_trials)
    
    # Add jitter (±7.5 degrees)
    jitter = np.random.uniform(-7.5, 7.5, size=n_trials)
    stimulus_angles = base_angles + jitter
    
    # Handle wrap-around for angles outside 0-360
    stimulus_angles = np.mod(stimulus_angles, 360)
    
    # Create stimulus sequence (impulse every 15 seconds)
    stimulus = np.zeros(n_trs)
    stimulus_trs = np.arange(0, total_duration, stimulus_interval) / tr  # Convert to TR indices
    
    # Create voxel preferred angles (0 to 359 degrees)
    # Voxel index 0 = 0°, voxel index 30 = 30°, etc.
    voxel_preferred_angles = np.arange(n_voxels)
    
    # Initialize response matrix for all voxels
    all_voxel_responses = np.zeros((n_trs, n_voxels))
    
    # Get HRF
    hrf, hrf_time = get_double_gamma_hrf(tr=tr)
    
    # Simulate each voxel
    for voxel_idx in range(n_voxels):
        # Create voxel response sequence
        voxel_responses = np.zeros(n_trs)
        for i, tr_idx in enumerate(stimulus_trs):
            if int(tr_idx) < n_trs:
                stimulus[int(tr_idx)] = 1.0
                # Voxel responds based on its preferred angle
                voxel_responses[int(tr_idx)] = get_voxel_response(
                    stimulus_angles[i], 
                    preferred_angle=voxel_preferred_angles[voxel_idx], 
                    tuning_sd=30.0
                )
        
        # Convolve voxel response sequence with HRF
        response = convolve(voxel_responses, hrf, mode='full')
        
        # Take only the relevant time points
        if len(response) >= n_trs:
            response = response[:n_trs]
        else:
            # Pad with zeros if needed
            padded = np.zeros(n_trs)
            padded[:len(response)] = response
            response = padded
        
        # Add positive Gaussian noise to the response
        noise = np.random.normal(0, voxSNR, size=response.shape)
        response_with_noise = response + noise
        
        # Subtract baseline 
        # response_with_noise = response_with_noise
        # Store response for this voxel
        all_voxel_responses[:, voxel_idx] = response_with_noise
    
    # Create time axis
    time_axis = np.arange(n_trs) * tr
    
    # Epoch data into trials and extract peak values
    print("Epoching data into trials...")
    
    # Find stimulus onset times (every 15 seconds = 15 TRs)
    stimulus_interval_trs = int(15 / tr)  # 15 TRs
    stimulus_onsets = np.arange(0, n_trials * stimulus_interval_trs, stimulus_interval_trs)
    # Only keep onsets that have enough TRs after them
    stimulus_onsets = stimulus_onsets[stimulus_onsets + stimulus_interval_trs <= n_trs]
    
    # Extract trial data (15 TRs per trial, matching stimulus interval)
    trial_length = stimulus_interval_trs
    n_trials_epoched = len(stimulus_onsets)
    
    # Create trial data matrix: (trials, timepoints, voxels)
    trial_data = np.empty((n_trials_epoched, trial_length, n_voxels))
    trial_labels = np.empty(n_trials_epoched)
    
    # valid_trials = 0 
    for trial_idx, onset in enumerate(stimulus_onsets):
        if onset + trial_length <= n_trs and trial_idx < len(stimulus_angles):
            trial_data[trial_idx] = all_voxel_responses[onset:onset + trial_length]
            trial_labels[trial_idx] = stimulus_angles[trial_idx]
            # valid_trials += 1
    
    # Extract peak values from whole epoch (average across all TRs)
    peak_data = trial_data.max(axis=1)  # Max across entire trial
    
    print(f"Extracted {n_trials_epoched} trials with peak data shape: {peak_data.shape}")
    
    # Run SVR angle prediction
    print("Running SVR angle prediction...")
    
    # Convert angles to radians and compute sin/cos
    angles_rad = np.radians(trial_labels)
    sin_targets = np.sin(angles_rad)
    cos_targets = np.cos(angles_rad)
    
    # Initialize SVR models
    svr_sin = SVR(kernel='rbf')
    svr_cos = SVR(kernel='rbf')
    
    # Leave-One-Out Cross-Validation
    cv = LeaveOneOut()
    predicted_angles = np.empty(n_trials)
    
    for train_idx, test_idx in cv.split(peak_data):
        X_train, X_test = peak_data[train_idx], peak_data[test_idx]
        sin_train, cos_train = sin_targets[train_idx], cos_targets[train_idx]
        
        # Fit models
        svr_sin.fit(X_train, sin_train)
        svr_cos.fit(X_train, cos_train)
        
        # Predict sin and cos
        pred_sin = svr_sin.predict(X_test)
        pred_cos = svr_cos.predict(X_test)
        
        # Compute predicted angles using arctangent
        pred_angles_rad = np.arctan2(pred_sin, pred_cos)
        pred_angles_deg = np.degrees(pred_angles_rad)
        
        # Handle angle wrapping (ensure angles are in [0, 360))
        pred_angles_deg = np.mod(pred_angles_deg, 360)
        predicted_angles[test_idx] = pred_angles_deg
    
    # Compute angular errors (only for trials that were actually extracted)
    true_angles = np.mod(trial_labels, 360)
    pred_angles = np.mod(predicted_angles, 360)
    angular_diff = np.abs(pred_angles - true_angles)
    angular_errors = np.minimum(angular_diff, 360 - angular_diff)
    
    print(f"Mean angular error: {np.mean(angular_errors):.2f} degrees")
    print(f"Std angular error: {np.std(angular_errors):.2f} degrees")
    
    # Visualize
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Heatmap of all voxel responses with better colormap
    im = ax1.imshow(all_voxel_responses.T, aspect='auto', cmap='plasma',
                    extent=[0, n_trs, 0, n_voxels], vmin=0, vmax=np.max(all_voxel_responses),
                    origin='lower')  # Set origin to lower to fix y-axis orientation
    ax1.set_xlabel('Time (TRs)')
    ax1.set_ylabel('Voxel Index (Preferred Angle)')
    ax1.set_title('Voxel Response Heatmap (360 voxels, Gaussian tuning, ±7.5° jitter)')
    
    # Add y-axis labels for key angles
    ax1.set_yticks([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 359])
    ax1.set_yticklabels(['0°', '30°', '60°', '90°', '120°', '150°', '180°', '210°', '240°', '270°', '300°', '330°', '359°'])
    
    # Add colorbar with better positioning
    cbar = plt.colorbar(im, ax=ax1, label='BOLD Response', shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    
    # Add horizontal lines to mark angle regions
    for angle in [90, 180, 270]:
        ax1.axhline(angle, color='white', linestyle=':', alpha=0.5, linewidth=0.5)
    
    # 2. Temporal SNR histogram
    # Calculate temporal SNR for each voxel (mean/std)
    temporal_snr = np.zeros(n_voxels)
    for voxel_idx in range(n_voxels):
        voxel_time_series = all_voxel_responses[:, voxel_idx]
        mean_response = np.mean(voxel_time_series)
        std_response = np.std(voxel_time_series)
        if std_response > 0:
            temporal_snr[voxel_idx] = mean_response / std_response
        else:
            temporal_snr[voxel_idx] = 0
    
    ax2.hist(temporal_snr, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Temporal SNR (Mean/Std)')
    ax2.set_ylabel('Number of Voxels')
    ax2.set_title('Distribution of Temporal SNR Across All Voxels')
    # ax2.set_xlim(0, 0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. Single voxel time series (first 100 TRs)
    # Choose voxel corresponding to first trial stimulus angle
    first_trial_angle = stimulus_angles[0]
    # Find voxel with preferred angle closest to first trial angle
    angle_diffs = np.abs(voxel_preferred_angles - first_trial_angle)
    angle_diffs = np.minimum(angle_diffs, 360 - angle_diffs)  # Handle circular distance
    selected_voxel = np.argmin(angle_diffs)
    
    # Plot first 100 TRs
    voxel_time_series = all_voxel_responses[:100, selected_voxel]
    time_subset = time_axis[:100]
    
    ax3.plot(time_subset, voxel_time_series, color='red', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('BOLD Response')
    ax3.set_title(f'Time Series from Voxel {selected_voxel} ({selected_voxel}°) - First Trial: {first_trial_angle:.1f}°')
    ax3.grid(True, alpha=0.3)
    
    # 4. SVR Results: Target vs Predicted Angles
    ax4.scatter(true_angles, pred_angles, alpha=0.6, s=50, color='blue', edgecolor='black', linewidth=0.5)
    
    # Add diagonal line for perfect prediction
    min_angle = min(np.min(true_angles), np.min(pred_angles))
    max_angle = max(np.max(true_angles), np.max(pred_angles))
    ax4.plot([min_angle, max_angle], [min_angle, max_angle], 'r--', linewidth=2, alpha=0.8, label='Perfect Prediction')
    
    ax4.set_xlabel('True Angle (degrees)')
    ax4.set_ylabel('Predicted Angle (degrees)')
    ax4.set_title(f'SVR Angle Prediction Results (Mean Error: {np.mean(angular_errors):.1f}°)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    ax4.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.show()
    
    return stimulus, all_voxel_responses, time_axis

if __name__ == "__main__":
    stimulus, all_voxel_responses, time_axis = main()