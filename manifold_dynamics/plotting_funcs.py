#!/usr/bin/env python3
"""
Plotting functions for MEG ERF analysis - 3D topographic plots using sensor positions
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from scipy.interpolate import griddata

def load_sensor_positions():
    """Load sensor positions from the MEG data file"""
    file_path = "/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-01/meg/sub-01_task-mgs_stimlocked_lineremoved.mat"
    
    print(f"Loading sensor positions from: {file_path}")
    data = sio.loadmat(file_path)
    
    # Extract sensor positions
    epoc_data = data['epocStimLocked'][0, 0]
    grad = epoc_data['grad'][0, 0]
    
    print(f"Grad structure fields: {grad.dtype.names}")
    
    # Extract channel positions
    chanpos = grad['chanpos']
    print(f"Channel positions shape: {chanpos.shape}")
    print(f"First few channel positions:\n{chanpos[:5]}")
    
    return chanpos

def compute_erf_power_windows(erf_data, time_vector, window_duration=0.5):
    """
    Compute power in time windows for ERF data
    
    Parameters:
    -----------
    erf_data : np.array, shape (157, 2001)
        ERF data for one condition
    time_vector : np.array, shape (2001,)
        Time vector in seconds
    window_duration : float
        Duration of each time window in seconds (default 500ms = 0.5s)
    
    Returns:
    --------
    power_windows : np.array, shape (157, n_windows)
        Power for each channel in each time window
    window_times : np.array, shape (n_windows,)
        Center time of each window
    """
    
    # Calculate sampling rate
    dt = np.mean(np.diff(time_vector))
    samples_per_window = int(window_duration / dt)
    
    print(f"Sampling rate: {1/dt:.1f} Hz")
    print(f"Window duration: {window_duration}s = {samples_per_window} samples")
    
    n_channels, n_timepoints = erf_data.shape
    n_windows = n_timepoints // samples_per_window
    
    print(f"Creating {n_windows} windows of {window_duration}s each")
    
    # Initialize output arrays
    power_windows = np.zeros((n_channels, n_windows))
    window_times = np.zeros(n_windows)
    
    for w in range(n_windows):
        start_idx = w * samples_per_window
        end_idx = start_idx + samples_per_window
        
        # Extract window data
        window_data = erf_data[:, start_idx:end_idx]
        
        # Compute power (RMS)
        power_windows[:, w] = np.sqrt(np.mean(window_data**2, axis=1))
        
        # Store center time of window
        window_times[w] = time_vector[start_idx + samples_per_window//2]
    
    return power_windows, window_times

def plot_3d_topography(power_data, chanpos, title="ERF Power", window_time=None):
    """
    Create 3D topographic plot of power data
    
    Parameters:
    -----------
    power_data : np.array, shape (157,)
        Power values for each channel
    chanpos : np.array, shape (157, 3)
        3D positions of channels
    title : str
        Plot title
    window_time : float or None
        Time of the window being plotted
    """
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract x, y, z coordinates
    x = chanpos[:, 0]
    y = chanpos[:, 1] 
    z = chanpos[:, 2]
    
    # Create scatter plot with power values as colors
    scatter = ax.scatter(x, y, z, c=power_data, cmap='viridis', s=100, alpha=0.9)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Power (T)', fontsize=12)
    
    # Remove grid and axis lines for cleaner view
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    
    if window_time is not None:
        title += f" at t={window_time:.2f}s"
    ax.set_title(title, fontsize=14)
    
    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    return fig, ax

def plot_erf_power_sequence(right_erf, left_erf, time_vector, window_duration=0.5):
    """
    Plot sequence of 3D topographic maps for ERF power in time windows
    
    Parameters:
    -----------
    right_erf : np.array, shape (157, 2001)
        Right target ERF data
    left_erf : np.array, shape (157, 2001)  
        Left target ERF data
    time_vector : np.array, shape (2001,)
        Time vector
    window_duration : float
        Duration of each window in seconds
    """
    
    # Load sensor positions
    chanpos = load_sensor_positions()
    
    # Compute power windows for both conditions
    print("\nComputing power windows for right targets...")
    right_power, window_times = compute_erf_power_windows(right_erf, time_vector, window_duration)
    
    print("Computing power windows for left targets...")
    left_power, _ = compute_erf_power_windows(left_erf, time_vector, window_duration)
    
    n_windows = len(window_times)
    print(f"\nCreating 3D plots for {n_windows} time windows...")
    
    # Create subplots for comparison
    n_cols = min(4, n_windows)  # Max 4 columns
    n_rows = int(np.ceil(n_windows / n_cols))
    
    # Plot right targets
    fig_right = plt.figure(figsize=(5*n_cols, 4*n_rows))
    fig_right.suptitle('Right Targets ERF Power - 3D Topography', fontsize=16)
    
    for w in range(n_windows):
        ax = fig_right.add_subplot(n_rows, n_cols, w+1, projection='3d')
        
        x, y, z = chanpos[:, 0], chanpos[:, 1], chanpos[:, 2]
        power = right_power[:, w]
        
        scatter = ax.scatter(x, y, z, c=power, cmap='viridis', s=100, alpha=0.9)
        ax.set_title(f't={window_times[w]:.2f}s', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Remove grid and axis lines for cleaner view
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        
        # Set consistent scale across all subplots
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x, mid_y, mid_z = (x.max()+x.min())*0.5, (y.max()+y.min())*0.5, (z.max()+z.min())*0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()
    
    # Plot left targets
    fig_left = plt.figure(figsize=(5*n_cols, 4*n_rows))
    fig_left.suptitle('Left Targets ERF Power - 3D Topography', fontsize=16)
    
    for w in range(n_windows):
        ax = fig_left.add_subplot(n_rows, n_cols, w+1, projection='3d')
        
        x, y, z = chanpos[:, 0], chanpos[:, 1], chanpos[:, 2]
        power = left_power[:, w]
        
        scatter = ax.scatter(x, y, z, c=power, cmap='viridis', s=100, alpha=0.9)
        ax.set_title(f't={window_times[w]:.2f}s', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y') 
        ax.set_zlabel('Z')
        
        # Remove grid and axis lines for cleaner view
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        
        # Set consistent scale
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()
    
    return right_power, left_power, window_times

def plot_erf_difference_topography(right_erf, left_erf, time_vector, window_duration=0.5):
    """
    Plot 3D topography of the difference between right and left ERF power
    """
    
    # Load sensor positions
    chanpos = load_sensor_positions()
    
    # Compute power windows
    right_power, window_times = compute_erf_power_windows(right_erf, time_vector, window_duration)
    left_power, _ = compute_erf_power_windows(left_erf, time_vector, window_duration)
    
    # Compute difference (right - left)
    power_diff = right_power - left_power
    
    n_windows = len(window_times)
    n_cols = min(4, n_windows)
    n_rows = int(np.ceil(n_windows / n_cols))
    
    fig = plt.figure(figsize=(5*n_cols, 4*n_rows))
    fig.suptitle('ERF Power Difference (Right - Left) - 3D Topography', fontsize=16)
    
    for w in range(n_windows):
        ax = fig.add_subplot(n_rows, n_cols, w+1, projection='3d')
        
        x, y, z = chanpos[:, 0], chanpos[:, 1], chanpos[:, 2]
        power = power_diff[:, w]
        
        scatter = ax.scatter(x, y, z, c=power, cmap='RdBu_r', s=100, alpha=0.9)
        ax.set_title(f't={window_times[w]:.2f}s', fontsize=12)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # Remove grid and axis lines for cleaner view
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
        ax.xaxis.pane.set_alpha(0)
        ax.yaxis.pane.set_alpha(0)
        ax.zaxis.pane.set_alpha(0)
        
        # Set consistent scale
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x, mid_y, mid_z = (x.max()+x.min())*0.5, (y.max()+y.min())*0.5, (z.max()+z.min())*0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.show()
    
    return power_diff, window_times
