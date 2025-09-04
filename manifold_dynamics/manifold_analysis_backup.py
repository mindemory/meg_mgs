#!/usr/bin/env python3
"""
Manifold learning analysis for MEG ERF data using Isomap
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.io as sio
from visualize_erf import load_and_process_meg_data

def extract_temporal_features(erf_data, time_vector, window_duration=0.05):
    """
    Extract temporal features from ERF data for manifold learning
    
    Parameters:
    -----------
    erf_data : np.array, shape (157, 2001)
        ERF data (averaged across all trials)
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
    feature_info : dict
        Information about the features
    """
    
    # Calculate sampling rate and window size
    dt = np.mean(np.diff(time_vector))
    samples_per_window = int(window_duration / dt)
    n_channels, n_timepoints = erf_data.shape
    n_windows = n_timepoints // samples_per_window
    
    print(f"Extracting temporal features: {n_windows} windows of {window_duration}s each")
    print(f"Sampling rate: {1/dt:.1f} Hz, samples per window: {samples_per_window}")
    
    # Initialize feature matrix
    features = np.zeros((n_windows, n_channels))
    window_times = np.zeros(n_windows)
    
    for w in range(n_windows):
        start_idx = w * samples_per_window
        end_idx = start_idx + samples_per_window
        
        # Extract window data
        window_data = erf_data[:, start_idx:end_idx]
        
        # Compute RMS power for each channel in this window
        features[w, :] = np.sqrt(np.mean(window_data**2, axis=1))
        
        # Store center time of window
        window_times[w] = time_vector[start_idx + samples_per_window//2]
    
    feature_info = {
        'type': 'temporal_windows',
        'n_windows': n_windows,
        'window_duration': window_duration,
        'description': f'RMS power in {n_windows} time windows ({window_duration*1000:.0f}ms each) for each channel'
    }
    
    print(f"Features shape: {features.shape}")
    print(f"Window times: {window_times[0]:.3f}s to {window_times[-1]:.3f}s")
    
    return features, window_times, feature_info

def apply_isomap(features, labels=None, n_neighbors=10, n_components=2):
    """
    Apply Isomap manifold learning
    
    Parameters:
    -----------
    features : np.array, shape (n_samples, n_features)
        Feature matrix
    labels : np.array, shape (n_samples,), optional
        Condition labels (not used for Isomap, kept for compatibility)
    n_neighbors : int
        Number of neighbors for Isomap
    n_components : int
        Number of manifold dimensions
        
    Returns:
    --------
    embedding : np.array, shape (n_samples, n_components)
        Low-dimensional embedding
    isomap : Isomap object
        Fitted Isomap object
    """
    
    print(f"\nApplying Isomap with {n_neighbors} neighbors, {n_components} components...")
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply Isomap
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components, metric='euclidean')
    embedding = isomap.fit_transform(features_scaled)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Reconstruction error: {isomap.reconstruction_error():.6f}")
    
    return embedding, isomap

def plot_temporal_manifold(embedding, window_times, feature_info, title="Temporal Isomap"):
    """
    Plot the temporal manifold embedding colored by time
    
    Parameters:
    -----------
    embedding : np.array, shape (n_windows, n_components)
        Low-dimensional embedding
    window_times : np.array, shape (n_windows,)
        Time of each window
    feature_info : dict
        Information about features
    title : str
        Plot title
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: 2D embedding colored by time
    scatter = axes[0].scatter(embedding[:, 0], embedding[:, 1], 
                             c=window_times, cmap='viridis', s=100, alpha=0.8)
    
    axes[0].set_xlabel('Isomap Dimension 1')
    axes[0].set_ylabel('Isomap Dimension 2')
    axes[0].set_title(f'{title}\n{feature_info["description"]}')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Time (s)', fontsize=12)
    
    # Add stimulus onset marker
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Stimulus Onset (t=0)')
    axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Time trajectory in manifold space
    # Sort by time to show trajectory
    time_order = np.argsort(window_times)
    sorted_times = window_times[time_order]
    sorted_embedding = embedding[time_order]
    
    # Plot trajectory with arrows
    for i in range(len(sorted_embedding) - 1):
        dx = sorted_embedding[i+1, 0] - sorted_embedding[i, 0]
        dy = sorted_embedding[i+1, 1] - sorted_embedding[i, 1]
        axes[1].arrow(sorted_embedding[i, 0], sorted_embedding[i, 1], 
                     dx, dy, head_width=0.1, head_length=0.1, 
                     fc='blue', ec='blue', alpha=0.6)
    
    # Color-code trajectory points by time
    scatter2 = axes[1].scatter(sorted_embedding[:, 0], sorted_embedding[:, 1], 
                              c=sorted_times, cmap='viridis', s=100, alpha=0.8)
    
    axes[1].set_xlabel('Isomap Dimension 1')
    axes[1].set_ylabel('Isomap Dimension 2')
    axes[1].set_title('Temporal Trajectory in Manifold Space')
    
    # Add colorbar
    cbar2 = plt.colorbar(scatter2, ax=axes[1])
    cbar2.set_label('Time (s)', fontsize=12)
    
    # Add time labels for key points
    for i, t in enumerate(sorted_times):
        if i % 5 == 0:  # Label every 5th point
            axes[1].annotate(f'{t:.2f}s', 
                           (sorted_embedding[i, 0], sorted_embedding[i, 1]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function for temporal manifold analysis"""
    print("=== MEG Temporal Manifold Learning with Isomap ===")
    
    # Load ERF data
    print("\nLoading ERF data...")
    right_erf, left_erf, time_vector, _, _ = load_and_process_meg_data()
    
    # Combine all trials (no left/right distinction)
    print("\nCombining all trials for temporal analysis...")
    all_erf = (right_erf + left_erf) / 2  # Average across conditions
    print(f"Combined ERF shape: {all_erf.shape}")
    
    # Extract temporal features with 50ms windows
    print("\nExtracting temporal features...")
    features, window_times, feature_info = extract_temporal_features(
        all_erf, time_vector, window_duration=0.05)
    
    # Apply Isomap with different neighbor parameters
    n_neighbors_list = [5, 10, 15, 20]
    
    for n_neighbors in n_neighbors_list:
        print(f"\n{'='*60}")
        print(f"Isomap with {n_neighbors} neighbors")
        print('='*60)
        
        # Apply Isomap
        embedding, isomap = apply_isomap(features, np.zeros(len(features)), 
                                       n_neighbors=n_neighbors, n_components=2)
        
        # Plot temporal manifold
        title = f"Temporal Isomap (k={n_neighbors})"
        plot_temporal_manifold(embedding, window_times, feature_info, title)
        
        # Print temporal dynamics info
        print(f"Number of time windows: {len(window_times)}")
        print(f"Time range: {window_times[0]:.3f}s to {window_times[-1]:.3f}s")
        print(f"Reconstruction error: {isomap.reconstruction_error():.6f}")
        
        # Calculate temporal spread
        if len(embedding) > 1:
            # Sort by time to analyze temporal progression
            time_order = np.argsort(window_times)
            sorted_embedding = embedding[time_order]
            
            # Calculate total trajectory length
            trajectory_length = 0
            for i in range(len(sorted_embedding) - 1):
                trajectory_length += np.linalg.norm(
                    sorted_embedding[i+1] - sorted_embedding[i])
            
            print(f"Total trajectory length: {trajectory_length:.4f}")
            print(f"Average step size: {trajectory_length/(len(sorted_embedding)-1):.4f}")

if __name__ == "__main__":
    main()
