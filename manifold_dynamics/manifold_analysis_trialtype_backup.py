#!/usr/bin/env python3
"""
Individual trial manifold learning analysis for MEG ERF data using Isomap
- k=10 neighbors only
- Compute manifold for each trial individually
- Average the trajectories in manifold space
- Show clean average trajectory (no std)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
from visualize_erf import load_and_process_meg_data

def extract_temporal_features_single_trial(trial_data, time_vector, window_duration=0.05):
    """
    Extract temporal features from single trial data
    
    Parameters:
    -----------
    trial_data : np.array, shape (157, 2001)
        Single trial MEG data
    time_vector : np.array, shape (2001,)
        Time vector
    window_duration : float
        Duration of each time window in seconds (default 50ms = 0.05s)
        
    Returns:
    --------
    features : np.array, shape (n_windows, 157)
        Feature matrix: each row is a time window, each column is a channel
    window_times : np.array, shape (n_windows,)
        Center time of each window
    """
    
    # Calculate sampling rate and window size
    dt = np.mean(np.diff(time_vector))
    samples_per_window = int(window_duration / dt)
    n_channels, n_timepoints = trial_data.shape
    n_windows = n_timepoints // samples_per_window
    
    # Initialize feature matrix
    features = np.zeros((n_windows, n_channels))
    window_times = np.zeros(n_windows)
    
    for w in range(n_windows):
        start_idx = w * samples_per_window
        end_idx = start_idx + samples_per_window
        
        # Extract window data
        window_data = trial_data[:, start_idx:end_idx]
        
        # Compute RMS power for each channel in this window
        features[w, :] = np.sqrt(np.mean(window_data**2, axis=1))
        
        # Store center time of window
        window_times[w] = time_vector[start_idx + samples_per_window//2]
    
    return features, window_times

def apply_isomap_single_trial(features, n_neighbors=10, n_components=3):
    """
    Apply Isomap to single trial features with NaN handling
    """
    # Check for NaN or inf values
    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check for zero variance channels
    feature_std = np.std(features, axis=0)
    zero_var_channels = feature_std == 0
    if np.any(zero_var_channels):
        features[:, zero_var_channels] += np.random.normal(0, 1e-6, size=(features.shape[0], np.sum(zero_var_channels)))
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Check again after scaling
    if np.any(np.isnan(features_scaled)) or np.any(np.isinf(features_scaled)):
        features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply Isomap
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric='euclidean')
    embedding = isomap.fit_transform(features_scaled)
    
    return embedding, isomap

def plot_clean_averaged_trajectory(all_embeddings, window_times, title="Averaged Trial Trajectory"):
    """
    Plot the clean averaged trajectory across all trials (no std) - 3D version
    
    Parameters:
    -----------
    all_embeddings : list of np.arrays
        List of embeddings for each trial
    window_times : np.array
        Time points for each window
    """
    
    # Convert to numpy array for easier manipulation
    embeddings_array = np.array(all_embeddings)  # Shape: (n_trials, n_windows, 3)
    
    # Average across trials
    mean_embedding = np.mean(embeddings_array, axis=0)  # Shape: (n_windows, 3)
    
    # Sort by time
    time_order = np.argsort(window_times)
    sorted_times = window_times[time_order]
    sorted_mean = mean_embedding[time_order]
    
    fig = plt.figure(figsize=(20, 6))
    
    # Plot 1: 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Create a color map based on time
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_mean)))
    
    # Plot trajectory line
    ax1.plot(sorted_mean[:, 0], sorted_mean[:, 1], sorted_mean[:, 2], 
             color='blue', linewidth=3, alpha=0.8, label='Average trajectory')
    
    # Plot points colored by time
    scatter = ax1.scatter(sorted_mean[:, 0], sorted_mean[:, 1], sorted_mean[:, 2], 
                         c=sorted_times, cmap='viridis', s=50, alpha=0.8)
    
    # Plot start and end points
    ax1.scatter(sorted_mean[0, 0], sorted_mean[0, 1], sorted_mean[0, 2], 
                c='red', s=150, alpha=0.9, marker='o', edgecolors='black', linewidth=2, label='Start')
    ax1.scatter(sorted_mean[-1, 0], sorted_mean[-1, 1], sorted_mean[-1, 2], 
                c='green', s=120, alpha=0.8, marker='s', edgecolors='black', linewidth=2, label='End')
    
    ax1.set_xlabel('Dimension 1', fontsize=12)
    ax1.set_ylabel('Dimension 2', fontsize=12)
    ax1.set_zlabel('Dimension 3', fontsize=12)
    ax1.set_title('3D Average Trajectory', fontsize=14)
    ax1.legend()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
    cbar.set_label('Time (s)', fontsize=10)
    
    # Plot 2: Time evolution of all 3 dimensions
    ax2 = fig.add_subplot(132)
    ax2.plot(sorted_times, sorted_mean[:, 0], 'b-', linewidth=3, label='Dimension 1', alpha=0.9)
    ax2.plot(sorted_times, sorted_mean[:, 1], 'r-', linewidth=3, label='Dimension 2', alpha=0.9)
    ax2.plot(sorted_times, sorted_mean[:, 2], 'g-', linewidth=3, label='Dimension 3', alpha=0.9)
    
    # Add stimulus onset marker
    ax2.axvline(x=0, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Stimulus (t=0)')
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Isomap Dimension Value', fontsize=12)
    ax2.set_title('Temporal Evolution of 3D Manifold', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 2D projections
    ax3 = fig.add_subplot(133)
    
    # Plot different 2D projections
    ax3.plot(sorted_mean[:, 0], sorted_mean[:, 1], 'b-', linewidth=2, alpha=0.7, label='Dim1 vs Dim2')
    ax3.scatter(sorted_mean[0, 0], sorted_mean[0, 1], c='red', s=100, marker='o', label='Start')
    ax3.scatter(sorted_mean[-1, 0], sorted_mean[-1, 1], c='green', s=80, marker='s', label='End')
    
    ax3.set_xlabel('Dimension 1', fontsize=12)
    ax3.set_ylabel('Dimension 2', fontsize=12)
    ax3.set_title('2D Projection (Dim1 vs Dim2)', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{title}\nAveraged across {len(all_embeddings)} trials, 3D Isomap (k=10)', fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    return fig, (ax1, ax2, ax3)

def plot_left_right_comparison(right_embeddings, left_embeddings, window_times, title="Left vs Right Target Comparison"):
    """
    Plot comparison of left vs right target trials in 2 rows
    
    Parameters:
    -----------
    right_embeddings : list
        List of embeddings for right target trials
    left_embeddings : list
        List of embeddings for left target trials
    window_times : np.array
        Time points for each window
    """
    
    fig = plt.figure(figsize=(20, 12))
    
    # Right targets (Row 1)
    if len(right_embeddings) > 0:
        right_array = np.array(right_embeddings)
        right_mean = np.mean(right_array, axis=0)
        
        # Sort by time
        time_order = np.argsort(window_times)
        sorted_times = window_times[time_order]
        sorted_right = right_mean[time_order]
        
        # Row 1: 3D trajectory for Right targets
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        # Remove the line plot and show only dots color-coded by time
        scatter = ax1.scatter(sorted_right[:, 0], sorted_right[:, 1], sorted_right[:, 2], 
                             c=sorted_times, cmap='RdBu_r', s=50, alpha=0.8)
        # Add start and end markers
        ax1.scatter(sorted_right[0, 0], sorted_right[0, 1], sorted_right[0, 2], 
                    c='red', s=100, alpha=0.9, marker='o', edgecolors='black', linewidth=2, label='Start')
        ax1.scatter(sorted_right[-1, 0], sorted_right[-1, 1], sorted_right[-1, 2], 
                    c='green', s=80, alpha=0.8, marker='s', edgecolors='black', linewidth=2, label='End')
        ax1.set_xlabel('Dim1', fontsize=10)
        ax1.set_ylabel('Dim2', fontsize=10)
        ax1.set_zlabel('Dim3', fontsize=10)
        ax1.set_title(f'Right Targets (n={len(right_embeddings)})\n3D Trajectory', fontsize=11)
        
        # Add colorbar for time
        cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.6, aspect=20)
        cbar1.set_label('Time (s)', fontsize=8)
        
        # Row 1: Time evolution for Right targets
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(sorted_times, sorted_right[:, 0], 'b-', linewidth=2, label='Dim1', alpha=0.8)
        ax2.plot(sorted_times, sorted_right[:, 1], 'r-', linewidth=2, label='Dim2', alpha=0.8)
        ax2.plot(sorted_times, sorted_right[:, 2], 'g-', linewidth=2, label='Dim3', alpha=0.8)
        ax2.axvline(x=0, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Stimulus (t=0)')
        ax2.set_xlabel('Time (s)', fontsize=10)
        ax2.set_ylabel('Dimension Value', fontsize=10)
        ax2.set_title(f'Right Targets\nTemporal Evolution', fontsize=11)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Row 1: 2D projection for Right targets
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(sorted_right[:, 0], sorted_right[:, 1], 'b-', linewidth=2, alpha=0.7)
        ax3.scatter(sorted_right[0, 0], sorted_right[0, 1], c='red', s=60, marker='o', label='Start')
        ax3.scatter(sorted_right[-1, 0], sorted_right[-1, 1], c='green', s=50, marker='s', label='End')
        ax3.set_xlabel('Dimension 1', fontsize=10)
        ax3.set_ylabel('Dimension 2', fontsize=10)
        ax3.set_title(f'Right Targets\n2D Projection', fontsize=11)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
    
    # Left targets (Row 2)
    if len(left_embeddings) > 0:
        left_array = np.array(left_embeddings)
        left_mean = np.mean(left_array, axis=0)
        
        # Sort by time
        sorted_left = left_mean[time_order]
        
        # Row 2: 3D trajectory for Left targets
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        # Remove the line plot and show only dots color-coded by time
        scatter2 = ax4.scatter(sorted_left[:, 0], sorted_left[:, 1], sorted_left[:, 2], 
                              c=sorted_times, cmap='RdBu_r', s=50, alpha=0.8)
        # Add start and end markers
        ax4.scatter(sorted_left[0, 0], sorted_left[0, 1], sorted_left[0, 2], 
                    c='red', s=100, alpha=0.9, marker='o', edgecolors='black', linewidth=2, label='Start')
        ax4.scatter(sorted_left[-1, 0], sorted_left[-1, 1], sorted_left[-1, 2], 
                    c='green', s=80, alpha=0.8, marker='s', edgecolors='black', linewidth=2, label='End')
        ax4.set_xlabel('Dim1', fontsize=10)
        ax4.set_ylabel('Dim2', fontsize=10)
        ax4.set_zlabel('Dim3', fontsize=10)
        ax4.set_title(f'Left Targets (n={len(left_embeddings)})\n3D Trajectory', fontsize=11)
        
        # Add colorbar for time
        cbar2 = plt.colorbar(scatter2, ax=ax4, shrink=0.6, aspect=20)
        cbar2.set_label('Time (s)', fontsize=8)
        
        # Row 2: Time evolution for Left targets
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(sorted_times, sorted_left[:, 0], 'b-', linewidth=2, label='Dim1', alpha=0.8)
        ax5.plot(sorted_times, sorted_left[:, 1], 'r-', linewidth=2, label='Dim2', alpha=0.8)
        ax5.plot(sorted_times, sorted_left[:, 2], 'g-', linewidth=2, label='Dim3', alpha=0.8)
        ax5.axvline(x=0, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Stimulus (t=0)')
        ax5.set_xlabel('Time (s)', fontsize=10)
        ax5.set_ylabel('Dimension Value', fontsize=10)
        ax5.set_title(f'Left Targets\nTemporal Evolution', fontsize=11)
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Row 2: 2D projection for Left targets
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(sorted_left[:, 0], sorted_left[:, 1], 'g-', linewidth=2, alpha=0.7)
        ax6.scatter(sorted_left[0, 0], sorted_left[0, 1], c='red', s=60, marker='o', label='Start')
        ax6.scatter(sorted_left[-1, 0], sorted_left[-1, 1], c='green', s=50, marker='s', label='End')
        ax6.set_xlabel('Dimension 1', fontsize=10)
        ax6.set_ylabel('Dimension 2', fontsize=10)
        ax6.set_title(f'Left Targets\n2D Projection', fontsize=11)
        ax6.legend(fontsize=8)
        ax6.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{title}\n3D Isomap (k=10) - Left vs Right Target Comparison', fontsize=18, y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_four_groups_comparison(group_embeddings, window_times, title="Four Target Location Groups Comparison"):
    """
    Plot comparison of four target location groups in 4 rows with 5 subplots each
    
    Parameters:
    -----------
    group_embeddings : dict
        Dictionary with group results: (2,3), (4,5), (7,8), (9,10)
    window_times : np.array
        Time points for each window
    """
    
    fig = plt.figure(figsize=(25, 16))
    
    # Define row configurations
    row_configs = [
        (2, 3),    # Row 1
        (4, 5),    # Row 2  
        (7, 8),    # Row 3
        (9, 10)    # Row 4
    ]
    
    for row_idx, (loc1, loc2) in enumerate(row_configs):
        group_key = f"({loc1},{loc2})"
        
        if group_key in group_embeddings and len(group_embeddings[group_key]) > 0:
            embeddings = group_embeddings[group_key]
            embeddings_array = np.array(embeddings)
            mean_embedding = np.mean(embeddings_array, axis=0)
            
            # Sort by time
            time_order = np.argsort(window_times)
            sorted_times = window_times[time_order]
            sorted_mean = mean_embedding[time_order]
            
            # Row 1: 3D trajectory
            ax1 = fig.add_subplot(4, 5, row_idx*5 + 1, projection='3d')
            # Remove the line plot and show only dots color-coded by time
            scatter = ax1.scatter(sorted_mean[:, 0], sorted_mean[:, 1], sorted_mean[:, 2], 
                                 c=sorted_times, cmap='RdBu_r', s=50, alpha=0.8)
            # Add start and end markers
            ax1.scatter(sorted_mean[0, 0], sorted_mean[0, 1], sorted_mean[0, 2], 
                        c='red', s=100, alpha=0.9, marker='o', edgecolors='black', linewidth=2, label='Start')
            ax1.scatter(sorted_mean[-1, 0], sorted_mean[-1, 1], sorted_mean[-1, 2], 
                        c='green', s=80, alpha=0.8, marker='s', edgecolors='black', linewidth=2, label='End')
            ax1.set_xlabel('Dim1', fontsize=10)
            ax1.set_ylabel('Dim2', fontsize=10)
            ax1.set_zlabel('Dim3', fontsize=10)
            ax1.set_title(f'Locations {loc1},{loc2} (n={len(embeddings)})\n3D Trajectory', fontsize=11)
            
            # Add colorbar for time
            cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6, aspect=20)
            cbar.set_label('Time (s)', fontsize=8)
            
            # Row 2: Time evolution
            ax2 = fig.add_subplot(4, 5, row_idx*5 + 2)
            ax2.plot(sorted_times, sorted_mean[:, 0], 'b-', linewidth=2, label='Dim1', alpha=0.8)
            ax2.plot(sorted_times, sorted_mean[:, 1], 'r-', linewidth=2, label='Dim2', alpha=0.8)
            ax2.plot(sorted_times, sorted_mean[:, 2], 'g-', linewidth=2, label='Dim3', alpha=0.8)
            
            # Add stimulus onset marker
            ax2.axvline(x=0, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Stimulus (t=0)')
            
            ax2.set_xlabel('Time (s)', fontsize=10)
            ax2.set_ylabel('Dimension Value', fontsize=10)
            ax2.set_title(f'Locations {loc1},{loc2}\nTemporal Evolution', fontsize=11)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Row 3: 2D projection Dim1 vs Dim2
            ax3 = fig.add_subplot(4, 5, row_idx*5 + 3)
            scatter2d_12 = ax3.scatter(sorted_mean[:, 0], sorted_mean[:, 1], 
                                      c=sorted_times, cmap='RdBu_r', s=60, alpha=0.8)
            # Add start and end markers
            ax3.scatter(sorted_mean[0, 0], sorted_mean[0, 1], c='red', s=80, marker='o', 
                       edgecolors='black', linewidth=2, label='Start')
            ax3.scatter(sorted_mean[-1, 0], sorted_mean[-1, 1], c='green', s=60, marker='s', 
                       edgecolors='black', linewidth=2, label='End')
            
            ax3.set_xlabel('Dimension 1', fontsize=10)
            ax3.set_ylabel('Dimension 2', fontsize=10)
            ax3.set_title(f'Locations {loc1},{loc2}\nDim1 vs Dim2', fontsize=11)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar for time
            cbar_12 = plt.colorbar(scatter2d_12, ax=ax3, shrink=0.6, aspect=20)
            cbar_12.set_label('Time (s)', fontsize=8)
            
            # Row 4: 2D projection Dim1 vs Dim3
            ax4 = fig.add_subplot(4, 5, row_idx*5 + 4)
            scatter2d_13 = ax4.scatter(sorted_mean[:, 0], sorted_mean[:, 2], 
                                      c=sorted_times, cmap='RdBu_r', s=60, alpha=0.8)
            # Add start and end markers
            ax4.scatter(sorted_mean[0, 0], sorted_mean[0, 2], c='red', s=80, marker='o', 
                       edgecolors='black', linewidth=2, label='Start')
            ax4.scatter(sorted_mean[-1, 0], sorted_mean[-1, 2], c='green', s=60, marker='s', 
                       edgecolors='black', linewidth=2, label='End')
            
            ax4.set_xlabel('Dimension 1', fontsize=10)
            ax4.set_ylabel('Dimension 3', fontsize=10)
            ax4.set_title(f'Locations {loc1},{loc2}\nDim1 vs Dim3', fontsize=11)
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            # Add colorbar for time
            cbar_13 = plt.colorbar(scatter2d_13, ax=ax4, shrink=0.6, aspect=20)
            cbar_13.set_label('Time (s)', fontsize=8)
            
            # Row 5: 2D projection Dim2 vs Dim3
            ax5 = fig.add_subplot(4, 5, row_idx*5 + 5)
            scatter2d_23 = ax5.scatter(sorted_mean[:, 1], sorted_mean[:, 2], 
                                      c=sorted_times, cmap='RdBu_r', s=60, alpha=0.8)
            # Add start and end markers
            ax5.scatter(sorted_mean[0, 1], sorted_mean[0, 2], c='red', s=80, marker='o', 
                       edgecolors='black', linewidth=2, label='Start')
            ax5.scatter(sorted_mean[-1, 1], sorted_mean[-1, 2], c='green', s=60, marker='s', 
                       edgecolors='black', linewidth=2, label='End')
            
            ax5.set_xlabel('Dimension 2', fontsize=10)
            ax5.set_ylabel('Dimension 3', fontsize=10)
            ax5.set_title(f'Locations {loc1},{loc2}\nDim2 vs Dim3', fontsize=11)
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
            
            # Add colorbar for time
            cbar_23 = plt.colorbar(scatter2d_23, ax=ax5, shrink=0.6, aspect=20)
            cbar_23.set_label('Time (s)', fontsize=8)
    
    # Overall title
    fig.suptitle(f'{title}\n3D Isomap (k≤50, adaptive) - Four Target Location Groups', fontsize=18, y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Main function for four target location groups comparison using Isomap"""
    print("=== Four Target Location Groups MEG Isomap Analysis (3D) ===")
    
    # Load the original data file directly to get all trials
    print("\nLoading MEG data...")
    data_file = "/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-06/meg/sub-06_task-mgs_stimlocked_lineremoved.mat"
    data = sio.loadmat(data_file)
    epocStimLocked = data['epocStimLocked']
    
    # Get the data we need
    trialinfo = epocStimLocked['trialinfo'][0, 0]
    trials = epocStimLocked['trial'][0, 0]  # This is the cell array of trials
    time_vector = epocStimLocked['time'][0, 0].flatten()
    
    print(f"Trialinfo shape: {trialinfo.shape}")
    print(f"Number of trials: {len(trials)}")
    print(f"Time vector shape: {time_vector.shape}")
    print(f"Time vector type: {type(time_vector)}")
    if hasattr(time_vector, 'dtype'):
        print(f"Time vector dtype: {time_vector.dtype}")
    
    # Debug: check what we actually have
    print(f"Trials type: {type(trials)}")
    if hasattr(trials, 'shape'):
        print(f"Trials shape: {trials.shape}")
    if hasattr(trials, 'dtype'):
        print(f"Trials dtype: {trials.dtype}")
    
    print(f"Target locations found: {np.unique(trialinfo[:, 1])}")
    
    # Convert time_vector to proper numpy array if it's an object array
    if time_vector.dtype == object:
        try:
            # The time_vector contains arrays, extract the first (and only) element from each
            time_vector = np.array([t[0, 0] if hasattr(t, 'shape') and t.shape == (1, 2001) else t[0] for t in time_vector])
            print(f"Converted time_vector to float array with shape: {time_vector.shape}")
        except Exception as e:
            print(f"Error converting time_vector: {e}")
            print(f"First few elements: {time_vector[:5]}")
            # Try alternative approach - just use the first time vector since they should all be the same
            try:
                time_vector = time_vector[0][0, :]  # Extract the first time vector
                print(f"Using first time vector: {time_vector.shape}")
            except Exception as e2:
                print(f"Alternative conversion also failed: {e2}")
                return
    
    # Check if we need to create a proper time vector based on trial data
    if len(time_vector) != 2001:
        print(f"Warning: Time vector length ({len(time_vector)}) doesn't match expected 2001. Creating proper time vector.")
        # Create a proper time vector from -1.5 to 2.5 seconds with 2001 points
        time_vector = np.linspace(-1.5, 2.5, 2001)
        print(f"Created new time vector: {time_vector.shape} from {time_vector[0]:.3f}s to {time_vector[-1]:.3f}s")
    
    # Define trial type configurations
    trial_type_configs = {
        (2, 3): [],    # Right targets
        (4, 5): [],    # Left targets  
        (7, 8): [],    # Left targets
        (9, 10): []    # Right targets
    }
    
    # Separate trials by target location (excluding location 11)
    print("\nSeparating trials by target location...")
    for i in range(len(trialinfo)):
        target_loc = trialinfo[i, 1]  # Target location is in column 1
        
        # Skip location 11 trials
        if target_loc == 11:
            continue
            
        for (loc1, loc2) in trial_type_configs.keys():
            if target_loc in [loc1, loc2]:
                # Get the trial data (convert from cell to array)
                # Based on the actual structure: trials has shape (1, 300)
                trial_data = trials[0, i]  # This is an object
                
                # Convert to proper numpy array if it's an object
                if hasattr(trial_data, 'dtype') and trial_data.dtype == object:
                    trial_data = np.array(trial_data, dtype=float)
                
                trial_type_configs[(loc1, loc2)].append(trial_data)
                break
    
    # Process each trial type
    print("\nProcessing trials by type...")
    group_embeddings = {}
    window_times = None
    
    for (loc1, loc2), trials_list in trial_type_configs.items():
        if len(trials_list) == 0:
            print(f"  No trials found for locations {loc1},{loc2}")
            continue
            
        print(f"  Processing locations {loc1},{loc2}: {len(trials_list)} trials")
        
        all_embeddings = []
        skipped_trials = 0
        
        for trial_data in trials_list:
            try:
                features, wt = extract_temporal_features_single_trial(trial_data, time_vector, window_duration=0.05)
                
                if window_times is None:
                    window_times = wt
                
                # Debug: print feature shape
                print(f"      Trial features shape: {features.shape}")
                
                # Apply Isomap to this trial
                # Ensure k is less than the number of time windows
                k_value = min(50, len(features) - 1)
                embedding, _ = apply_isomap_single_trial(features, n_neighbors=k_value, n_components=3)
                all_embeddings.append(embedding)
                
            except Exception as e:
                print(f"      Error processing trial: {str(e)}")
                skipped_trials += 1
        
        if len(all_embeddings) > 0:
            group_embeddings[f"({loc1},{loc2})"] = all_embeddings
            print(f"    Successfully processed: {len(all_embeddings)} trials, {skipped_trials} skipped")
    
    # Plot comparison
    print(f"\nPlotting four groups comparison...")
    if window_times is not None:
        print(f"Time windows: {len(window_times)} windows of 50ms each")
        print(f"Time range: {window_times[0]:.3f}s to {window_times[-1]:.3f}s")
    
    if len(group_embeddings) > 0:
        plot_four_groups_comparison(group_embeddings, window_times)
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        for group_key, embeddings in group_embeddings.items():
            print(f"✓ {group_key}: {len(embeddings)} trials")
        if window_times is not None:
            print(f"✓ Time windows: {len(window_times)} (50ms each)")
        print(f"✓ Isomap: 3D embedding (k≤50, adaptive)")
        print(f"✓ Comparing four target location groups")
    else:
        print("ERROR: No trial types could be processed successfully!")

if __name__ == "__main__":
    main()
