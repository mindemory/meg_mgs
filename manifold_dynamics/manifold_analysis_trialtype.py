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
import scipy.signal as signal

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

def select_occipital_sensors(trial_data, channel_positions=None):
    """
    Select occipital/posterior sensors based on their spatial coordinates
    
    Parameters:
    -----------
    trial_data : np.array, shape (n_channels, n_timepoints)
        MEG trial data
    channel_positions : np.array, shape (n_channels, 3), optional
        3D positions of channels (x, y, z coordinates)
        If None, will use a simple posterior selection based on channel indices
        
    Returns:
    --------
    selected_data : np.array, shape (n_selected_channels, n_timepoints)
        Trial data from selected occipital sensors
    selected_indices : np.array
        Indices of selected channels
    """
    n_channels = trial_data.shape[0]
    
    if channel_positions is not None:
        # Use actual channel positions to select posterior sensors
        # Assuming x is anterior-posterior, y is left-right, z is inferior-superior
        # Select channels in posterior regions (negative x values)
        # and relatively low z values (closer to occipital cortex)
        
        x_coords = channel_positions[:, 0]
        z_coords = channel_positions[:, 2]
        
        # Select posterior channels (lower x values) and lower z values
        posterior_mask = x_coords < np.percentile(x_coords, 30)  # Posterior 30%
        occipital_mask = z_coords < np.percentile(z_coords, 60)  # Lower 60% in z
        
        # Combine criteria
        selected_mask = posterior_mask & occipital_mask
        selected_indices = np.where(selected_mask)[0]
        
    else:
        # Fallback: assume MEG sensors are arranged with occipital sensors in certain ranges
        # This is a rough approximation - typically occipital sensors are in the back
        # For a 157-channel system, assume occipital sensors are roughly in these ranges
        if n_channels == 157:
            # Rough approximation for typical MEG sensor arrangements
            # These indices would need to be adjusted based on actual sensor layout
            occipital_ranges = [
                range(120, 140),  # Posterior sensors
                range(140, 157),  # More posterior sensors
                range(100, 120),  # Additional posterior-lateral sensors
            ]
            selected_indices = []
            for r in occipital_ranges:
                selected_indices.extend(list(r))
            selected_indices = np.array(selected_indices)
        else:
            # Generic selection: take posterior third of sensors
            start_idx = int(n_channels * 0.67)
            selected_indices = np.arange(start_idx, n_channels)
    
    # Ensure we have at least some sensors
    if len(selected_indices) == 0:
        print("Warning: No occipital sensors selected, using posterior 20% of sensors")
        start_idx = int(n_channels * 0.8)
        selected_indices = np.arange(start_idx, n_channels)
    
    selected_data = trial_data[selected_indices, :]
    
    print(f"Selected {len(selected_indices)} occipital sensors out of {n_channels} total sensors")
    print(f"Selected sensor indices: {selected_indices[:10]}..." if len(selected_indices) > 10 else f"Selected sensor indices: {selected_indices}")
    
    return selected_data, selected_indices

def apply_bandpass_filter(trial_data, fs=1000, low_freq=1, high_freq=55, order=4):
    """
    Apply a Butterworth bandpass filter to trial data
    
    Parameters:
    -----------
    trial_data : np.array, shape (n_channels, n_timepoints)
        MEG trial data
    fs : float
        Sampling frequency in Hz
    low_freq : float
        Low cutoff frequency in Hz
    high_freq : float
        High cutoff frequency in Hz
    order : int
        Filter order
        
    Returns:
    --------
    filtered_data : np.array, shape (n_channels, n_timepoints)
        Bandpass filtered trial data
    """
    # Design the bandpass filter
    nyquist = fs / 2
    low_normalized = low_freq / nyquist
    high_normalized = high_freq / nyquist
    
    # Create Butterworth bandpass filter
    b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
    
    # Apply filter to each channel
    filtered_data = np.zeros_like(trial_data)
    for ch in range(trial_data.shape[0]):
        filtered_data[ch, :] = signal.filtfilt(b, a, trial_data[ch, :])
    
    return filtered_data

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

def plot_spatial_topography(group_embeddings, window_times, title="Spatial Topography Analysis"):
    """
    Plot individual target locations (1-10) in Isomap space to visualize spatial topography
    
    Parameters:
    -----------
    group_embeddings : dict
        Dictionary containing embeddings for each target location group
    window_times : np.array
        Time windows for temporal analysis
    title : str
        Title for the analysis
    """
    # Target location mapping (code -> angle in degrees)
    target_angles = {
        1: 0, 2: 25, 3: 50, 4: 130, 5: 155,
        6: 180, 7: 205, 8: 230, 9: 310, 10: 335
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Create distinct colors for each target location
    colors = [
        '#FF0000',  # Red - Target 1 (0°)
        '#FF8000',  # Orange - Target 2 (25°)
        '#FFFF00',  # Yellow - Target 3 (50°)
        '#80FF00',  # Lime - Target 4 (130°)
        '#00FF00',  # Green - Target 5 (155°)
        '#00FFFF',  # Cyan - Target 6 (180°)
        '#0080FF',  # Light Blue - Target 7 (205°)
        '#0000FF',  # Blue - Target 8 (230°)
        '#8000FF',  # Purple - Target 9 (310°)
        '#FF00FF'   # Magenta - Target 10 (335°)
    ]
    
    # Extract individual target locations
    individual_embeddings = {}
    for group_key, embeddings in group_embeddings.items():
        # Extract target code from group key (e.g., "Target 2" -> 2)
        target_code = int(group_key.split()[-1])
        
        if target_code not in individual_embeddings:
            individual_embeddings[target_code] = []
        
        # Get the mean trajectory for this target location
        mean_trajectory = np.mean(embeddings, axis=0)  # Average across trials
        
        # Filter to time >= 0 and <= 1.7
        time_mask = (window_times >= 0) & (window_times <= 1.7)
        filtered_trajectory = mean_trajectory[time_mask, :]
        
        individual_embeddings[target_code].append(filtered_trajectory)
    
    # Plot each target location
    for target_code in range(1, 11):
        if target_code in individual_embeddings:
            trajectory = individual_embeddings[target_code][0]
            angle = target_angles[target_code]
            color = colors[target_code-1]
            
            # Dim1 vs Dim2
            axes[0].plot(trajectory[:, 0], trajectory[:, 1], 
                        color=color, linewidth=2.5, alpha=0.9, 
                        label=f'Target {target_code} ({angle}°)')
            # Start marker (lighter version of trajectory color)
            axes[0].scatter(trajectory[0, 0], trajectory[0, 1], 
                           c=color, s=100, marker='o', alpha=1.0, zorder=5, 
                           edgecolors='white', linewidth=2)
            # End marker (darker version of trajectory color)
            axes[0].scatter(trajectory[-1, 0], trajectory[-1, 1], 
                           c=color, s=100, marker='s', alpha=1.0, zorder=5, 
                           edgecolors='black', linewidth=2)
            
            # Dim1 vs Dim3
            axes[1].plot(trajectory[:, 0], trajectory[:, 2], 
                        color=color, linewidth=2.5, alpha=0.9, 
                        label=f'Target {target_code} ({angle}°)')
            # Start marker (lighter version of trajectory color)
            axes[1].scatter(trajectory[0, 0], trajectory[0, 2], 
                           c=color, s=100, marker='o', alpha=1.0, zorder=5, 
                           edgecolors='white', linewidth=2)
            # End marker (darker version of trajectory color)
            axes[1].scatter(trajectory[-1, 0], trajectory[-1, 2], 
                           c=color, s=100, marker='s', alpha=1.0, zorder=5, 
                           edgecolors='black', linewidth=2)
            
            # Dim2 vs Dim3
            axes[2].plot(trajectory[:, 1], trajectory[:, 2], 
                        color=color, linewidth=2.5, alpha=0.9, 
                        label=f'Target {target_code} ({angle}°)')
            # Start marker (lighter version of trajectory color)
            axes[2].scatter(trajectory[0, 1], trajectory[0, 2], 
                           c=color, s=100, marker='o', alpha=1.0, zorder=5, 
                           edgecolors='white', linewidth=2)
            # End marker (darker version of trajectory color)
            axes[2].scatter(trajectory[-1, 1], trajectory[-1, 2], 
                           c=color, s=100, marker='s', alpha=1.0, zorder=5, 
                           edgecolors='black', linewidth=2)
    
    # Customize plots
    axes[0].set_xlabel('Dimension 1', fontsize=12)
    axes[0].set_ylabel('Dimension 2', fontsize=12)
    axes[0].set_title('Dim1 vs Dim2', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    axes[1].set_xlabel('Dimension 1', fontsize=12)
    axes[1].set_ylabel('Dimension 3', fontsize=12)
    axes[1].set_title('Dim1 vs Dim3', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Dimension 2', fontsize=12)
    axes[2].set_ylabel('Dimension 3', fontsize=12)
    axes[2].set_title('Dim2 vs Dim3', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    # Add legend for markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linewidth=0, markersize=12, 
               markeredgecolor='white', markeredgewidth=2, label='Start (0s)'),
        Line2D([0], [0], marker='s', color='gray', linewidth=0, markersize=12, 
               markeredgecolor='black', markeredgewidth=2, label='End (1.7s)')
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()

def plot_simple_spatial_analysis(group_embeddings, window_times):
    """
    Create a plot showing spatial structure at different time points
    3 rows (time points: 0.2s, 0.5s, 1.5s) x 4 columns (Dim1vs2, Dim1vs3, Dim2vs3, Actual Layout)
    """
    # Target location mapping (code -> angle in degrees)
    target_angles = {
        1: 0, 2: 25, 3: 50, 4: 130, 5: 155,
        6: 180, 7: 205, 8: 230, 9: 310, 10: 335
    }
    
    target_colors = [
        '#FF0000', '#FF8000', '#FFFF00', '#80FF00', '#00FF00',
        '#00FFFF', '#0080FF', '#0000FF', '#8000FF', '#FF00FF'
    ]
    
    # Time points to analyze
    time_points = [0.2, 0.5, 1.5]
    
    # Extract positions at different time points for each target
    target_positions_by_time = {}
    
    for group_key, embeddings in group_embeddings.items():
        target_code = int(group_key.split()[-1])
        mean_trajectory = np.mean(embeddings, axis=0)  # Average across trials
        
        # Filter to time >= 0 and <= 1.7
        time_mask = (window_times >= 0) & (window_times <= 1.7)
        filtered_times = window_times[time_mask]
        filtered_trajectory = mean_trajectory[time_mask, :]
        
        # Extract positions at specific time points
        for time_point in time_points:
            if time_point not in target_positions_by_time:
                target_positions_by_time[time_point] = {}
            
            # Find closest time index
            time_idx = np.argmin(np.abs(filtered_times - time_point))
            position = filtered_trajectory[time_idx, :]
            target_positions_by_time[time_point][target_code] = position
    
    # Create the plot: 3 rows x 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Spatial Structure Analysis - Different Time Points', fontsize=16, fontweight='bold')
    
    for row, time_point in enumerate(time_points):
        target_positions = target_positions_by_time[time_point]
        
        # Column 1: Dim1 vs Dim2
        ax = axes[row, 0]
        for target_code in range(1, 11):
            if target_code in target_positions:
                pos = target_positions[target_code]
                angle = target_angles[target_code]
                color = target_colors[target_code-1]
                
                ax.scatter(pos[0], pos[1], c=color, s=150, alpha=0.8, 
                          edgecolors='black', linewidth=1.5)
                ax.annotate(f'{target_code}', (pos[0], pos[1]), 
                           xytext=(2, 2), textcoords='offset points', 
                           fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(f'Dim1 vs Dim2 (t={time_point}s)')
        ax.grid(True, alpha=0.3)
        
        # Column 2: Dim1 vs Dim3
        ax = axes[row, 1]
        for target_code in range(1, 11):
            if target_code in target_positions:
                pos = target_positions[target_code]
                angle = target_angles[target_code]
                color = target_colors[target_code-1]
                
                ax.scatter(pos[0], pos[2], c=color, s=150, alpha=0.8, 
                          edgecolors='black', linewidth=1.5)
                ax.annotate(f'{target_code}', (pos[0], pos[2]), 
                           xytext=(2, 2), textcoords='offset points', 
                           fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 3')
        ax.set_title(f'Dim1 vs Dim3 (t={time_point}s)')
        ax.grid(True, alpha=0.3)
        
        # Column 3: Dim2 vs Dim3
        ax = axes[row, 2]
        for target_code in range(1, 11):
            if target_code in target_positions:
                pos = target_positions[target_code]
                angle = target_angles[target_code]
                color = target_colors[target_code-1]
                
                ax.scatter(pos[1], pos[2], c=color, s=150, alpha=0.8, 
                          edgecolors='black', linewidth=1.5)
                ax.annotate(f'{target_code}', (pos[1], pos[2]), 
                           xytext=(2, 2), textcoords='offset points', 
                           fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Dimension 2')
        ax.set_ylabel('Dimension 3')
        ax.set_title(f'Dim2 vs Dim3 (t={time_point}s)')
        ax.grid(True, alpha=0.3)
        
        # Column 4: Actual target spatial layout (same for all rows)
        ax = axes[row, 3]
        for target_code in range(1, 11):
            angle = target_angles[target_code]
            color = target_colors[target_code-1]
            
            # Convert angle to x,y coordinates on unit circle
            x = np.cos(np.radians(angle))
            y = np.sin(np.radians(angle))
            
            ax.scatter(x, y, c=color, s=150, alpha=0.8, 
                      edgecolors='black', linewidth=1.5)
            ax.annotate(f'{target_code}', (x, y), 
                       xytext=(2, 2), textcoords='offset points', 
                       fontsize=9, fontweight='bold')
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Actual Target Layout')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()

def analyze_spatial_topography_correlation(group_embeddings, window_times, title="Spatial Topography Correlation Analysis"):
    """
    Analyze whether spatial topography is captured in Isomap dimensions
    
    Parameters:
    -----------
    group_embeddings : dict
        Dictionary containing embeddings for each target location group
    window_times : np.array
        Time windows for temporal analysis
    title : str
        Title for the analysis
    """
    # Target location mapping (code -> angle in degrees)
    target_angles = {
        1: 0,    # Right
        2: 25,   # Right-up
        3: 50,   # Right-up
        4: 130,  # Left-up
        5: 155,  # Left-up
        6: 180,  # Left
        7: 205,  # Left-down
        8: 230,  # Left-down
        9: 310,  # Right-down
        10: 335  # Right-down
    }
    
    # Extract individual target locations from the group embeddings
    individual_embeddings = {}
    
    # Map group embeddings back to individual target locations
    for group_key, embeddings in group_embeddings.items():
        # Extract target code from group key (e.g., "Target 2" -> 2)
        target_code = int(group_key.split()[-1])
        
        if target_code not in individual_embeddings:
            individual_embeddings[target_code] = []
        
        # Get the mean trajectory for this target location
        mean_trajectory = np.mean(embeddings, axis=0)  # Average across trials
        
        # Filter to time >= 0 and <= 1.7
        time_mask = (window_times >= 0) & (window_times <= 1.7)
        filtered_times = window_times[time_mask]
        filtered_trajectory = mean_trajectory[time_mask, :]
        
        individual_embeddings[target_code].append(filtered_trajectory)
    
    # Calculate mean position in Isomap space for each target (using end position at 1.7s)
    target_isomap_positions = {}
    target_angles_list = []
    
    for target_code in range(1, 11):  # All 10 targets
        if target_code in individual_embeddings:
            trajectory = individual_embeddings[target_code][0]
            # Use the end position (last time point)
            end_position = trajectory[-1, :]  # [Dim1, Dim2, Dim3]
            target_isomap_positions[target_code] = end_position
            target_angles_list.append(target_angles[target_code])
    
    # Convert to arrays for correlation analysis
    target_codes = sorted(target_isomap_positions.keys())
    angles_deg = np.array([target_angles[code] for code in target_codes])
    
    dim1_positions = np.array([target_isomap_positions[code][0] for code in target_codes])
    dim2_positions = np.array([target_isomap_positions[code][1] for code in target_codes])
    dim3_positions = np.array([target_isomap_positions[code][2] for code in target_codes])
    
    # Calculate correlations with angular position
    from scipy.stats import pearsonr, spearmanr
    
    # Pearson correlation (linear relationship)
    dim1_pearson, dim1_p = pearsonr(angles_deg, dim1_positions)
    dim2_pearson, dim2_p = pearsonr(angles_deg, dim2_positions)
    dim3_pearson, dim3_p = pearsonr(angles_deg, dim3_positions)
    
    # Spearman correlation (monotonic relationship)
    dim1_spearman, dim1_sp = spearmanr(angles_deg, dim1_positions)
    dim2_spearman, dim2_sp = spearmanr(angles_deg, dim2_positions)
    dim3_spearman, dim3_sp = spearmanr(angles_deg, dim3_positions)
    
    # Circular correlation (for angular data)
    def circular_correlation(angles, values):
        # Convert angles to unit vectors
        x = np.cos(np.radians(angles))
        y = np.sin(np.radians(angles))
        
        # Calculate correlation with values
        corr_x = pearsonr(x, values)[0]
        corr_y = pearsonr(y, values)[0]
        
        # Return magnitude of correlation
        return np.sqrt(corr_x**2 + corr_y**2)
    
    dim1_circular = circular_correlation(angles_deg, dim1_positions)
    dim2_circular = circular_correlation(angles_deg, dim2_positions)
    dim3_circular = circular_correlation(angles_deg, dim3_positions)
    
    # Print results
    print(f"\n{'='*80}")
    print("SPATIAL TOPOGRAPHY CORRELATION ANALYSIS (ISOMAP)")
    print('='*80)
    print(f"Analyzing correlation between target angles and Isomap positions...")
    print(f"Target angles: {angles_deg}")
    print(f"Number of targets: {len(target_codes)}")
    
    print(f"\nDimension 1 vs Angular Position:")
    print(f"  Pearson r: {dim1_pearson:.3f} (p={dim1_p:.3f})")
    print(f"  Spearman ρ: {dim1_spearman:.3f} (p={dim1_sp:.3f})")
    print(f"  Circular r: {dim1_circular:.3f}")
    
    print(f"\nDimension 2 vs Angular Position:")
    print(f"  Pearson r: {dim2_pearson:.3f} (p={dim2_p:.3f})")
    print(f"  Spearman ρ: {dim2_spearman:.3f} (p={dim2_sp:.3f})")
    print(f"  Circular r: {dim2_circular:.3f}")
    
    print(f"\nDimension 3 vs Angular Position:")
    print(f"  Pearson r: {dim3_pearson:.3f} (p={dim3_p:.3f})")
    print(f"  Spearman ρ: {dim3_spearman:.3f} (p={dim3_sp:.3f})")
    print(f"  Circular r: {dim3_circular:.3f}")
    
    # Determine which dimension best captures spatial topography
    correlations = [dim1_circular, dim2_circular, dim3_circular]
    best_dim = np.argmax(correlations) + 1
    best_correlation = np.max(correlations)
    
    print(f"\n{'='*50}")
    print("INTERPRETATION")
    print('='*50)
    print(f"Best spatial topography capture: Dimension {best_dim} (r={best_correlation:.3f})")
    
    if best_correlation > 0.7:
        print(f"✓ STRONG spatial topography preserved in Dimension {best_dim}")
    elif best_correlation > 0.5:
        print(f"✓ MODERATE spatial topography preserved in Dimension {best_dim}")
    elif best_correlation > 0.3:
        print(f"✓ WEAK spatial topography preserved in Dimension {best_dim}")
    else:
        print(f"✗ LITTLE spatial topography preserved in any dimension")
    
    # Check if multiple dimensions together capture topography better
    print(f"\nCombined dimension analysis:")
    print(f"Dim1+Dim2: {np.sqrt(dim1_circular**2 + dim2_circular**2):.3f}")
    print(f"Dim1+Dim3: {np.sqrt(dim1_circular**2 + dim3_circular**2):.3f}")
    print(f"Dim2+Dim3: {np.sqrt(dim2_circular**2 + dim3_circular**2):.3f}")
    print(f"Dim1+Dim2+Dim3: {np.sqrt(dim1_circular**2 + dim2_circular**2 + dim3_circular**2):.3f}")
    
    return target_isomap_positions, target_angles

def main():
    """Main function for four target location groups comparison using Isomap"""
    print("=== Four Target Location Groups MEG Isomap Analysis (3D) with Bandpass Filter (1-55Hz) ===")
    
    # Load the original data file directly to get all trials
    print("\nLoading MEG data...")
    data_file = "/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-06/meg/sub-06_task-mgs_stimlocked_lineremoved.mat"
    data = sio.loadmat(data_file)
    epocStimLocked = data['epocStimLocked']
    
    # Get the data we need
    trialinfo = epocStimLocked['trialinfo'][0, 0]
    trials = epocStimLocked['trial'][0, 0]  # This is the cell array of trials
    time_vector = epocStimLocked['time'][0, 0].flatten()
    
    # Try to extract channel positions if available
    channel_positions = None
    try:
        # Check for channel positions in different possible locations
        for field_name in ['chanpos', 'grad']:
            if field_name in epocStimLocked.dtype.names:
                field_data = epocStimLocked[field_name][0, 0]
                if field_name == 'chanpos':
                    if hasattr(field_data, 'shape') and len(field_data.shape) >= 2:
                        channel_positions = np.array(field_data)
                        print(f"Found channel positions: {channel_positions.shape}")
                        break
                elif field_name == 'grad':
                    if hasattr(field_data, 'dtype') and 'chanpos' in field_data.dtype.names:
                        chanpos = field_data['chanpos'][0, 0]
                        if hasattr(chanpos, 'shape') and len(chanpos.shape) >= 2:
                            channel_positions = np.array(chanpos)
                            print(f"Found channel positions in grad: {channel_positions.shape}")
                            break
    except Exception as e:
        print(f"Could not extract channel positions: {e}")
        print("Will use index-based sensor selection")
    
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
    
    # Define trial type configurations - now individual targets like PCA
    trial_type_configs = {
        1: [],    # Target 1
        2: [],    # Target 2
        3: [],    # Target 3
        4: [],    # Target 4
        5: [],    # Target 5
        6: [],    # Target 6
        7: [],    # Target 7
        8: [],    # Target 8
        9: [],    # Target 9
        10: []    # Target 10
    }
    
    # Separate trials by target location (excluding location 11)
    print("\nSeparating trials by target location...")
    for i in range(len(trialinfo)):
        target_loc = trialinfo[i, 1]  # Target location is in column 1
        
        # Skip location 11 trials
        if target_loc == 11:
            continue
            
        # Add trial to individual target location
        if target_loc in trial_type_configs:
            # Get the trial data (convert from cell to array)
            # Based on the actual structure: trials has shape (1, 300)
            trial_data = trials[0, i]  # This is an object
            
            # Convert to proper numpy array if it's an object
            if hasattr(trial_data, 'dtype') and trial_data.dtype == object:
                trial_data = np.array(trial_data, dtype=float)
            
            trial_type_configs[target_loc].append(trial_data)
    
    # Process each trial type
    print("\nProcessing trials by type...")
    group_embeddings = {}
    window_times = None
    
    for target_loc, trials_list in trial_type_configs.items():
        if len(trials_list) == 0:
            print(f"  No trials found for target {target_loc}")
            continue
            
        print(f"  Processing target {target_loc}: {len(trials_list)} trials")
        
        all_embeddings = []
        skipped_trials = 0
        
        for trial_data in trials_list:
            try:
                # Apply bandpass filter (1-55Hz)
                filtered_trial_data = apply_bandpass_filter(trial_data, fs=1000, low_freq=1, high_freq=55)
                
                features, wt = extract_temporal_features_single_trial(filtered_trial_data, time_vector, window_duration=0.05)
                
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
            group_embeddings[f"Target {target_loc}"] = all_embeddings
            print(f"    Successfully processed: {len(all_embeddings)} trials, {skipped_trials} skipped")
    
    # Plot comparison
    print(f"\nPlotting four groups comparison...")
    if window_times is not None:
        print(f"Time windows: {len(window_times)} windows of 50ms each")
        print(f"Time range: {window_times[0]:.3f}s to {window_times[-1]:.3f}s")
    
    if len(group_embeddings) > 0:
        # Add simple spatial structure analysis
        print(f"\nGenerating spatial structure analysis...")
        plot_simple_spatial_analysis(group_embeddings, window_times)
        
        # Add spatial topography correlation analysis
        print(f"\nAnalyzing spatial topography correlation...")
        analyze_spatial_topography_correlation(group_embeddings, window_times, 
                                            title="Spatial Topography Correlation Analysis")
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        for group_key, embeddings in group_embeddings.items():
            print(f"✓ {group_key}: {len(embeddings)} trials")
        if window_times is not None:
            print(f"✓ Time windows: {len(window_times)} (50ms each)")
        print(f"✓ Isomap: 3D embedding (k≤50, adaptive)")
        print(f"✓ Bandpass filter: 1-55Hz")
        print(f"✓ Comparing individual target locations (1-10)")
    else:
        print("ERROR: No trial types could be processed successfully!")

if __name__ == "__main__":
    main()
