#!/usr/bin/env python3
"""
Visualize MEG Event-Related Fields (ERF): Group trials by target location (left/right) and compute averages
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from plotting_funcs import plot_erf_difference_topography

def load_and_process_meg_data():
    """Load MEG data, filter trials, and group by left/right target locations"""
    file_path = "/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-01/meg/sub-01_task-mgs_stimlocked_lineremoved.mat"
    
    print(f"Loading stimulus-locked MEG data: {file_path}")
    data = sio.loadmat(file_path)
    
    # Extract the data
    epoc_data = data['epocStimLocked'][0, 0]
    trials = epoc_data['trial'][0]  # Cell array with trials
    time_data = epoc_data['time'][0]  # Time vectors
    trialinfo = epoc_data['trialinfo']  # Trial information
    
    print(f"Number of trials: {len(trials)}")
    print(f"Trialinfo shape: {trialinfo.shape}")
    
    # Get time vector
    time_vector = time_data[0]
    if time_vector.ndim > 1:
        time_vector = time_vector.flatten()
    print(f"Time vector: {len(time_vector)} points from {time_vector[0]:.3f} to {time_vector[-1]:.3f} seconds")
    
    # Extract target locations from trialinfo column 2 (index 1)
    target_locations = trialinfo[:, 1]  # Column 2 (0-indexed)
    print(f"Target locations found: {np.unique(target_locations)}")
    print(f"Location counts: {np.bincount(target_locations.astype(int))}")
    
    # Filter out trials with location 11
    valid_trials = target_locations != 11
    print(f"Trials before filtering: {len(trials)}")
    print(f"Trials after removing location 11: {np.sum(valid_trials)}")
    
    # Group trials by left/right
    right_locations = [1, 2, 3, 9, 10]  # Right targets
    left_locations = [4, 5, 6, 7, 8]    # Left targets
    
    right_trials = []
    left_trials = []
    
    for i, trial in enumerate(trials):
        if valid_trials[i]:  # Skip location 11
            location = int(target_locations[i])
            if location in right_locations:
                right_trials.append(trial)
            elif location in left_locations:
                left_trials.append(trial)
    
    print(f"Right trials: {len(right_trials)}")
    print(f"Left trials: {len(left_trials)}")
    
    # Convert to arrays and compute averages
    right_trials = np.array(right_trials)  # Shape: (n_right_trials, 157, 2001)
    left_trials = np.array(left_trials)    # Shape: (n_left_trials, 157, 2001)
    
    # Average across trials for each condition
    right_erf = np.nanmean(right_trials, axis=0)  # Shape: (157, 2001)
    left_erf = np.nanmean(left_trials, axis=0)    # Shape: (157, 2001)
    
    print(f"Right ERF shape: {right_erf.shape}")
    print(f"Left ERF shape: {left_erf.shape}")
    
    return right_erf, left_erf, time_vector, right_trials, left_trials

def plot_erf_comparison(right_erf, left_erf, time_vector):
    """Plot ERF comparison between left and right target conditions"""
    
    # Plot all channels for both conditions
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Right ERF
    for ch in range(157):
        axes[0].plot(time_vector, right_erf[ch, :], alpha=0.7, linewidth=0.5, color='blue')
    
    axes[0].set_title('Right Target ERF - All 157 Channels')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude (T)')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Stimulus Onset')
    axes[0].legend()
    
    # Left ERF
    for ch in range(157):
        axes[1].plot(time_vector, left_erf[ch, :], alpha=0.7, linewidth=0.5, color='green')
    
    axes[1].set_title('Left Target ERF - All 157 Channels')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude (T)')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Stimulus Onset')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

def plot_erf_overlay(right_erf, left_erf, time_vector):
    """Plot ERF overlay for direct comparison"""
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Plot all channels for both conditions
    for ch in range(157):
        ax.plot(time_vector, right_erf[ch, :], alpha=0.5, linewidth=0.5, color='blue')
        ax.plot(time_vector, left_erf[ch, :], alpha=0.5, linewidth=0.5, color='green')
    
    ax.set_title('ERF Comparison: Right (Blue) vs Left (Green) Targets - All 157 Channels')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (T)')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='Stimulus Onset')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='blue', alpha=0.7, label='Right Targets'),
                      Line2D([0], [0], color='green', alpha=0.7, label='Left Targets'),
                      Line2D([0], [0], color='red', linestyle='--', alpha=0.8, label='Stimulus Onset')]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.show()



def main():
    """Main function"""
    print("=== MEG Event-Related Fields (ERF) Analysis ===")
    
    # Load and process the data
    right_erf, left_erf, time_vector, right_trials, left_trials = load_and_process_meg_data()
    
    # Plot: 3D topographic plot of power differences only
    print("\nCreating 3D topographic plot of power differences...")
    power_diff, window_times = plot_erf_difference_topography(
        right_erf, left_erf, time_vector, window_duration=0.5)
    
    print(f"\n=== Summary ===")
    print(f"✓ Right targets (locations 1,2,3,9,10): {len(right_trials)} trials")
    print(f"✓ Left targets (locations 4,5,6,7,8): {len(left_trials)} trials")
    print(f"✓ ERF computed: 157 channels × 2001 time points")
    print(f"✓ Time range: {time_vector[0]:.3f} to {time_vector[-1]:.3f} seconds")
    print(f"✓ Stimulus onset at t=0")
    print(f"✓ Power difference analysis: {len(window_times)} time windows (500ms each)")
    print(f"✓ Window times: {window_times}")

if __name__ == "__main__":
    main()
