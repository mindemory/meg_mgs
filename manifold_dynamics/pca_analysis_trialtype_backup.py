#!/usr/bin/env python3
"""
PCA-based analysis for MEG ERF data using the same structure as Isomap analysis
- k≤50 neighbors (adjusted per trial), 3D embedding
- Separate analysis for different target location conditions
- 4 rows: (2,3), (4,5), (7,8), (9,10)
- 5 subplots per row: 3D trajectory, temporal evolution, Dim1vs2, Dim1vs3, Dim2vs3
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
from visualize_erf import load_and_process_meg_data
import scipy.signal as signal

def apply_lowpass_filter(trial_data, fs=1000, cutoff=55, order=4):
    """
    Apply lowpass filter to trial data
    
    Parameters:
    -----------
    trial_data : np.array, shape (157, 2001)
        Single trial MEG data
    fs : float
        Sampling frequency in Hz (default 1000 Hz)
    cutoff : float
        Cutoff frequency in Hz (default 55 Hz)
    order : int
        Filter order (default 4)
        
    Returns:
    --------
    filtered_data : np.array, shape (157, 2001)
        Lowpass filtered trial data
    """
    # Design Butterworth lowpass filter
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Apply filter to each channel
    filtered_data = np.zeros_like(trial_data)
    for ch in range(trial_data.shape[0]):
        filtered_data[ch, :] = signal.filtfilt(b, a, trial_data[ch, :])
    
    return filtered_data

def apply_theta_bandpass_filter(trial_data, fs=1000, low_freq=4, high_freq=8, order=4):
    """
    Apply theta bandpass filter to trial data
    
    Parameters:
    -----------
    trial_data : np.array, shape (157, 2001)
        Single trial MEG data
    fs : float
        Sampling frequency in Hz (default 1000 Hz)
    low_freq : float
        Lower cutoff frequency in Hz (default 4 Hz)
    high_freq : float
        Upper cutoff frequency in Hz (default 8 Hz)
    order : int
        Filter order (default 4)
        
    Returns:
    --------
    filtered_data : np.array, shape (157, 2001)
        Theta bandpass filtered trial data
    """
    # Design Butterworth bandpass filter
    nyquist = fs / 2
    low_cutoff = low_freq / nyquist
    high_cutoff = high_freq / nyquist
    b, a = signal.butter(order, [low_cutoff, high_cutoff], btype='band', analog=False)
    
    # Apply filter to each channel
    filtered_data = np.zeros_like(trial_data)
    for ch in range(trial_data.shape[0]):
        filtered_data[ch, :] = signal.filtfilt(b, a, trial_data[ch, :])
    
    return filtered_data

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
    # Use a more robust approach for dt calculation
    if len(time_vector) > 1:
        dt = np.abs(time_vector[1] - time_vector[0])
        if dt == 0 or np.isinf(dt) or np.isnan(dt):
            # Fallback: assume 1000 Hz sampling rate
            dt = 0.001
            print(f"Warning: Using fallback dt = {dt}")
    else:
        dt = 0.001  # Fallback: assume 1000 Hz sampling rate
        print(f"Warning: Using fallback dt = {dt}")
    
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

def apply_pca_single_trial(features, n_components=3):
    """
    Apply PCA to single trial features with NaN handling and return explained variance
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
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(features_scaled)
    
    return embedding, pca, pca.explained_variance_ratio_

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
            ax1.set_xlabel('PC1', fontsize=10)
            ax1.set_ylabel('PC2', fontsize=10)
            ax1.set_zlabel('PC3', fontsize=10)
            ax1.set_title(f'Locations {loc1},{loc2} (n={len(embeddings)})\n3D Trajectory', fontsize=11)
            
            # Add colorbar for time
            cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6, aspect=20)
            cbar.set_label('Time (s)', fontsize=8)
            
            # Row 2: Time evolution
            ax2 = fig.add_subplot(4, 5, row_idx*5 + 2)
            ax2.plot(sorted_times, sorted_mean[:, 0], 'b-', linewidth=2, label='PC1', alpha=0.8)
            ax2.plot(sorted_times, sorted_mean[:, 1], 'r-', linewidth=2, label='PC2', alpha=0.8)
            ax2.plot(sorted_times, sorted_mean[:, 2], 'g-', linewidth=2, label='PC3', alpha=0.8)
            
            # Add stimulus onset marker
            ax2.axvline(x=0, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Stimulus (t=0)')
            
            ax2.set_xlabel('Time (s)', fontsize=10)
            ax2.set_ylabel('Principal Component Value', fontsize=10)
            ax2.set_title(f'Locations {loc1},{loc2}\nTemporal Evolution', fontsize=11)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 2D projection plots
            # Dim1 vs Dim2
            ax3 = fig.add_subplot(4, 5, row_idx*5 + 3)
            # Filter data to only include time >= 0 and <= 1.7
            time_mask = (sorted_times >= 0) & (sorted_times <= 1.7)
            filtered_times = sorted_times[time_mask]
            filtered_mean = sorted_mean[time_mask, :]
            
            # 2D projection plots
            # Dim1 vs Dim2
            ax3.plot(filtered_mean[:, 0], filtered_mean[:, 1], linewidth=1, alpha=0.8)
            # Color different time segments
            for i in range(len(filtered_times) - 1):
                if 0 <= filtered_times[i] < 0.2:
                    color = 'red'
                elif 0.2 <= filtered_times[i] < 1.7:
                    color = 'blue'
                else:
                    color = 'gray'
                ax3.plot(filtered_mean[i:i+2, 0], filtered_mean[i:i+2, 1], color=color, linewidth=1, alpha=0.8)
            
            # Start and end markers
            ax3.scatter(filtered_mean[0, 0], filtered_mean[0, 1], c='red', s=80, marker='o', label='Start (0s)', zorder=5)
            ax3.scatter(filtered_mean[-1, 0], filtered_mean[-1, 1], c='blue', s=80, marker='s', label='End (1.7s)', zorder=5)
            ax3.set_xlabel('PC1')
            ax3.set_ylabel('PC2')
            ax3.set_title(f'PC1 vs PC2')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Dim1 vs Dim3
            ax4 = fig.add_subplot(4, 5, row_idx*5 + 4)
            # Filter data to only include time >= 0 and <= 1.7
            time_mask = (sorted_times >= 0) & (sorted_times <= 1.7)
            filtered_times = sorted_times[time_mask]
            filtered_mean = sorted_mean[time_mask, :]
            
            # Dim1 vs Dim3
            ax4.plot(filtered_mean[:, 0], filtered_mean[:, 2], linewidth=1, alpha=0.8)
            # Color different time segments
            for i in range(len(filtered_times) - 1):
                if 0 <= filtered_times[i] < 0.2:
                    color = 'red'
                elif 0.2 <= filtered_times[i] < 1.7:
                    color = 'blue'
                else:
                    color = 'gray'
                ax4.plot(filtered_mean[i:i+2, 0], filtered_mean[i:i+2, 2], color=color, linewidth=1, alpha=0.8)
            
            # Start and end markers
            ax4.scatter(filtered_mean[0, 0], filtered_mean[0, 2], c='red', s=80, marker='o', label='Start (0s)', zorder=5)
            ax4.scatter(filtered_mean[-1, 0], filtered_mean[-1, 2], c='blue', s=80, marker='s', label='End (1.7s)', zorder=5)
            ax4.set_xlabel('PC1')
            ax4.set_ylabel('PC3')
            ax4.set_title(f'PC1 vs PC3')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # Dim2 vs Dim3
            ax5 = fig.add_subplot(4, 5, row_idx*5 + 5)
            # Filter data to only include time >= 0 and <= 1.7
            time_mask = (sorted_times >= 0) & (sorted_times <= 1.7)
            filtered_times = sorted_times[time_mask]
            filtered_mean = sorted_mean[time_mask, :]
            
            # Dim2 vs Dim3
            ax5.plot(filtered_mean[:, 1], filtered_mean[:, 2], linewidth=1, alpha=0.8)
            # Color different time segments
            for i in range(len(filtered_times) - 1):
                if 0 <= filtered_times[i] < 0.2:
                    color = 'red'
                elif 0.2 <= filtered_times[i] < 1.7:
                    color = 'blue'
                else:
                    color = 'gray'
                ax5.plot(filtered_mean[i:i+2, 1], filtered_mean[i:i+2, 2], color=color, linewidth=1, alpha=0.8)
            
            # Start and end markers
            ax5.scatter(filtered_mean[0, 1], filtered_mean[0, 2], c='red', s=80, marker='o', label='Start (0s)', zorder=5)
            ax5.scatter(filtered_mean[-1, 1], filtered_mean[-1, 2], c='blue', s=80, marker='s', label='End (1.7s)', zorder=5)
            ax5.set_xlabel('PC2')
            ax5.set_ylabel('PC3')
            ax5.set_title(f'PC2 vs PC3')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{title}\n3D PCA - Four Target Location Groups', fontsize=18, y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_spatial_topography(group_embeddings, window_times, title="Spatial Topography Analysis"):
    """
    Plot individual target locations (1-10) in PC space to analyze spatial topography
    
    Parameters:
    -----------
    group_embeddings : dict
        Dictionary containing embeddings for each target location group
    window_times : np.array
        Time windows for temporal analysis
    title : str
        Title for the plot
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
    
    # Create figure with 3 subplots for PC1 vs PC2, PC1 vs PC3, PC2 vs PC3
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Colors for different target locations (using a color wheel)
    colors = plt.cm.hsv(np.linspace(0, 1, 10))
    
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
    
    # Plot each target location
    for target_code in range(1, 11):
        if target_code in individual_embeddings:
            # Get the trajectory for this target
            trajectory = individual_embeddings[target_code][0]  # Take first occurrence
            
            # Get the angle for this target
            angle = target_angles[target_code]
            
            # Plot PC1 vs PC2
            axes[0].plot(trajectory[:, 0], trajectory[:, 1], 
                        color=colors[target_code-1], linewidth=2, 
                        label=f'Target {target_code} ({angle}°)')
            axes[0].scatter(trajectory[0, 0], trajectory[0, 1], 
                           color=colors[target_code-1], s=100, marker='o', zorder=5)
            axes[0].scatter(trajectory[-1, 0], trajectory[-1, 1], 
                           color=colors[target_code-1], s=100, marker='s', zorder=5)
            
            # Plot PC1 vs PC3
            axes[1].plot(trajectory[:, 0], trajectory[:, 2], 
                        color=colors[target_code-1], linewidth=2, 
                        label=f'Target {target_code} ({angle}°)')
            axes[1].scatter(trajectory[0, 0], trajectory[0, 2], 
                           color=colors[target_code-1], s=100, marker='o', zorder=5)
            axes[1].scatter(trajectory[-1, 0], trajectory[-1, 2], 
                           color=colors[target_code-1], s=100, marker='s', zorder=5)
            
            # Plot PC2 vs PC3
            axes[2].plot(trajectory[:, 1], trajectory[:, 2], 
                        color=colors[target_code-1], linewidth=2, 
                        label=f'Target {target_code} ({angle}°)')
            axes[2].scatter(trajectory[0, 1], trajectory[0, 2], 
                           color=colors[target_code-1], s=100, marker='o', zorder=5)
            axes[2].scatter(trajectory[-1, 1], trajectory[-1, 2], 
                           color=colors[target_code-1], s=100, marker='s', zorder=5)
    
    # Customize plots
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].set_title('PC1 vs PC2 - Spatial Topography')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC3')
    axes[1].set_title('PC1 vs PC3 - Spatial Topography')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_xlabel('PC2')
    axes[2].set_ylabel('PC3')
    axes[2].set_title('PC2 vs PC3 - Spatial Topography')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_spatial_topography_correlation(group_embeddings, window_times, title="Spatial Topography Correlation Analysis"):
    """
    Analyze whether spatial topography is captured in PCA dimensions
    
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
    
    # Calculate mean position in PC space for each target (using end position at 1.7s)
    target_pc_positions = {}
    target_angles_list = []
    
    for target_code in range(1, 11):
        if target_code in individual_embeddings:
            trajectory = individual_embeddings[target_code][0]
            # Use the end position (last time point)
            end_position = trajectory[-1, :]  # [PC1, PC2, PC3]
            target_pc_positions[target_code] = end_position
            target_angles_list.append(target_angles[target_code])
    
    # Convert to arrays for correlation analysis
    target_codes = sorted(target_pc_positions.keys())
    angles_rad = np.array([np.radians(target_angles[code]) for code in target_codes])
    angles_deg = np.array([target_angles[code] for code in target_codes])
    
    pc1_positions = np.array([target_pc_positions[code][0] for code in target_codes])
    pc2_positions = np.array([target_pc_positions[code][1] for code in target_codes])
    pc3_positions = np.array([target_pc_positions[code][2] for code in target_codes])
    
    # Calculate correlations with angular position
    from scipy.stats import pearsonr, spearmanr
    
    # Pearson correlation (linear relationship)
    pc1_pearson, pc1_p = pearsonr(angles_deg, pc1_positions)
    pc2_pearson, pc2_p = pearsonr(angles_deg, pc2_positions)
    pc3_pearson, pc3_p = pearsonr(angles_deg, pc3_positions)
    
    # Spearman correlation (monotonic relationship)
    pc1_spearman, pc1_sp = spearmanr(angles_deg, pc1_positions)
    pc2_spearman, pc2_sp = spearmanr(angles_deg, pc2_positions)
    pc3_spearman, pc3_sp = spearmanr(angles_deg, pc3_positions)
    
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
    
    pc1_circular = circular_correlation(angles_deg, pc1_positions)
    pc2_circular = circular_correlation(angles_deg, pc2_positions)
    pc3_circular = circular_correlation(angles_deg, pc3_positions)
    
    # Print results
    print(f"\n{'='*80}")
    print("SPATIAL TOPOGRAPHY CORRELATION ANALYSIS")
    print('='*80)
    print(f"Analyzing correlation between target angles and PCA positions...")
    print(f"Target angles: {angles_deg}")
    print(f"Number of targets: {len(target_codes)}")
    
    print(f"\nPC1 vs Angular Position:")
    print(f"  Pearson r: {pc1_pearson:.3f} (p={pc1_p:.3f})")
    print(f"  Spearman ρ: {pc1_spearman:.3f} (p={pc1_sp:.3f})")
    print(f"  Circular r: {pc1_circular:.3f}")
    
    print(f"\nPC2 vs Angular Position:")
    print(f"  Pearson r: {pc2_pearson:.3f} (p={pc2_p:.3f})")
    print(f"  Spearman ρ: {pc2_spearman:.3f} (p={pc2_sp:.3f})")
    print(f"  Circular r: {pc2_circular:.3f}")
    
    print(f"\nPC3 vs Angular Position:")
    print(f"  Pearson r: {pc3_pearson:.3f} (p={pc3_p:.3f})")
    print(f"  Spearman ρ: {pc3_spearman:.3f} (p={pc3_sp:.3f})")
    print(f"  Circular r: {pc3_circular:.3f}")
    
    # Determine which PC best captures spatial topography
    correlations = [pc1_circular, pc2_circular, pc3_circular]
    best_pc = np.argmax(correlations) + 1
    best_correlation = np.max(correlations)
    
    print(f"\n{'='*50}")
    print("INTERPRETATION")
    print('='*50)
    print(f"Best spatial topography capture: PC{best_pc} (r={best_correlation:.3f})")
    
    if best_correlation > 0.7:
        print(f"✓ STRONG spatial topography preserved in PC{best_pc}")
    elif best_correlation > 0.5:
        print(f"✓ MODERATE spatial topography preserved in PC{best_pc}")
    elif best_correlation > 0.3:
        print(f"✓ WEAK spatial topography preserved in PC{best_pc}")
    else:
        print(f"✗ LITTLE spatial topography preserved in any PC")
    
    # Check if multiple PCs together capture topography better
    print(f"\nCombined PC analysis:")
    print(f"PC1+PC2: {np.sqrt(pc1_circular**2 + pc2_circular**2):.3f}")
    print(f"PC1+PC3: {np.sqrt(pc1_circular**2 + pc3_circular**2):.3f}")
    print(f"PC2+PC3: {np.sqrt(pc2_circular**2 + pc3_circular**2):.3f}")
    print(f"PC1+PC2+PC3: {np.sqrt(pc1_circular**2 + pc2_circular**2 + pc3_circular**2):.3f}")
    
    return target_pc_positions, target_angles

def main():
    """Main function for four target location groups comparison using PCA with theta bandpass filtering"""
    print("=== Four Target Location Groups MEG PCA Analysis (3D) with Theta Bandpass Filter (4-8Hz) ===")
    
    # Load the original data file directly to get all trials
    print("\nLoading MEG data...")
    data_file = "/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/meg/sub-12_task-mgs_stimlocked_lineremoved.mat"
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
    
    # Define trial type configurations for individual targets
    trial_type_configs = {
        1: [],    # Target 1 (0°)
        2: [],    # Target 2 (25°)
        3: [],    # Target 3 (50°)
        4: [],    # Target 4 (130°)
        5: [],    # Target 5 (155°)
        6: [],    # Target 6 (180°)
        7: [],    # Target 7 (205°)
        8: [],    # Target 8 (230°)
        9: [],    # Target 9 (310°)
        10: []    # Target 10 (335°)
    }
    
    # Separate trials by target location (excluding location 11)
    print("\nSeparating trials by target location...")
    for i in range(len(trialinfo)):
        target_loc = trialinfo[i, 1]  # Target location is in column 1
        
        # Skip location 11 trials
        if target_loc == 11:
            continue
            
        # Process individual target locations
        if target_loc in trial_type_configs:
            # Get the trial data (convert from cell to array)
            # Based on the actual structure: trials has shape (1, 300)
            trial_data = trials[0, i]  # This is an object
            
            # Convert to proper numpy array if it's an object
            if hasattr(trial_data, 'dtype') and trial_data.dtype == object:
                trial_data = np.array(trial_data, dtype=float)
            
            # Apply theta bandpass filter
            trial_data = apply_theta_bandpass_filter(trial_data, fs=1000, low_freq=4, high_freq=8, order=4)
            
            trial_type_configs[target_loc].append(trial_data)
    
    # Process each trial type
    print("\nProcessing trials by type...")
    group_embeddings = {}
    window_times = None
    
    for target_code, trials_list in trial_type_configs.items():
        if len(trials_list) == 0:
            print(f"  No trials found for target {target_code}")
            continue
            
        print(f"  Processing target {target_code}: {len(trials_list)} trials")
        
        all_embeddings = []
        skipped_trials = 0
        explained_variance = []  # Initialize for this group
        
        for trial_data in trials_list:
            try:
                features, wt = extract_temporal_features_single_trial(trial_data, time_vector, window_duration=0.02)
                
                if window_times is None:
                    window_times = wt
                
                # Debug: print feature shape
                print(f"      Trial features shape: {features.shape}")
                
                # Apply PCA to this trial
                embedding, _, explained_variance_ratio = apply_pca_single_trial(features, n_components=3)
                all_embeddings.append(embedding)
                
                # Store explained variance for this trial
                explained_variance.append(explained_variance_ratio)
                
            except Exception as e:
                print(f"      Error processing trial: {str(e)}")
                skipped_trials += 1
        
        if len(all_embeddings) > 0:
            group_embeddings[f"Target {target_code}"] = all_embeddings
            
            # Calculate average explained variance for this group
            if len(explained_variance) > 0:
                avg_explained_variance = np.mean(explained_variance, axis=0)
                print(f"    Successfully processed: {len(all_embeddings)} trials, {skipped_trials} skipped")
                print(f"    Average explained variance: PC1={avg_explained_variance[0]:.3f}, PC2={avg_explained_variance[1]:.3f}, PC3={avg_explained_variance[2]:.3f}")
                print(f"    Cumulative explained variance: {np.sum(avg_explained_variance):.3f}")
            else:
                print(f"    Successfully processed: {len(all_embeddings)} trials, {skipped_trials} skipped")
    
    # Plot comparison
    print(f"\nPlotting four groups comparison...")
    if window_times is not None:
        print(f"Time windows: {len(window_times)} windows of 50ms each")
        print(f"Time range: {window_times[0]:.3f}s to {window_times[-1]:.3f}s")
    
    if len(group_embeddings) > 0:
        plot_four_groups_comparison(group_embeddings, window_times)
        
        # Add spatial topography analysis
        print(f"\nGenerating spatial topography analysis...")
        plot_spatial_topography(group_embeddings, window_times, 
                              title=f"Spatial Topography Analysis - Four Target Location Groups")
        
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
            print(f"✓ Time windows: {len(window_times)} (20ms each)")
        print(f"✓ PCA: 3D embedding")
        print(f"✓ Theta bandpass filter: 4-8Hz")
        print(f"✓ Comparing individual target locations (1-10)")
    else:
        print("ERROR: No trial types could be processed successfully!")

if __name__ == "__main__":
    main()
