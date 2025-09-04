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
def main():
    """Main function for individual trial manifold analysis"""
    print("=== Individual Trial MEG Manifold Learning (k=10) ===")
    
    # Load ERF data
    print("\nLoading ERF data...")
    right_erf, left_erf, time_vector, right_trials, left_trials = load_and_process_meg_data()
    
    # Combine all trials (no left/right distinction)
    print("\nCombining all trials...")
    all_trials = np.vstack([right_trials, left_trials])  # Shape: (160, 157, 2001)
    print(f"Total trials: {all_trials.shape[0]}")
    
    # Extract temporal features for each individual trial
    print("\nProcessing individual trials...")
    
    all_embeddings = []
    window_times = None
    skipped_trials = 0
    
    # Process all trials
    for i in range(all_trials.shape[0]):
        try:
            trial_data = all_trials[i]  # Shape: (157, 2001)
            features, wt = extract_temporal_features_single_trial(trial_data, time_vector, window_duration=0.05)
            
            if window_times is None:
                window_times = wt
            
            # Apply Isomap to this trial
            embedding, _ = apply_isomap_single_trial(features, n_neighbors=10, n_components=3)
            all_embeddings.append(embedding)
            
            if (i + 1) % 40 == 0:
                print(f"  Processed {i + 1}/{all_trials.shape[0]} trials")
        
        except Exception as e:
            print(f"  Skipping trial {i}: {str(e)}")
            skipped_trials += 1
    
    # Plot averaged trajectory
    print(f"\nComputing averaged trajectory...")
    print(f"Successfully processed: {len(all_embeddings)} trials")
    print(f"Skipped trials: {skipped_trials}")
    print(f"Time windows: {len(window_times)} windows of 50ms each")
    print(f"Time range: {window_times[0]:.3f}s to {window_times[-1]:.3f}s")
    
    if len(all_embeddings) > 0:
        plot_clean_averaged_trajectory(all_embeddings, window_times)
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        print(f"✓ Processed {len(all_embeddings)} individual trials")
        print(f"✓ Skipped trials: {skipped_trials}")
        print(f"✓ Time windows: {len(window_times)} (50ms each)")
        print(f"✓ Isomap: k=10 neighbors, 2D embedding")
        print(f"✓ Averaged trajectories in manifold space")
        print(f"✓ Clean visualization (no standard deviation)")
    else:
        print("ERROR: No trials could be processed successfully!")

if __name__ == "__main__":
    main()
