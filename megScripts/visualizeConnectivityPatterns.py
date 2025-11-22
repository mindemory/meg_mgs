#!/usr/bin/env python3
"""
Script to load and plot connectivity results from all subjects.
Loads coherence and imaginary coherence data, computes averages across subjects, and plots results.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import socket
from shutil import copyfile
import h5py
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not available. Install with: pip install nibabel")
from scipy.spatial.distance import cdist

def load_connectivity_results(bidsRoot, subjects, taskName='mgs', voxRes='10mm', 
                              seed='left_frontal', target='left', metric='imcoh'):
    """
    Load seeded connectivity results for all subjects
    
    Parameters:
    -----------
    bidsRoot : str
        Root directory for BIDS data
    subjects : list
        List of subject IDs
    taskName : str
        Task name (default: 'mgs')
    voxRes : str
        Voxel resolution (default: '10mm')
    seed : str
        Seed region (default: 'left_frontal')
    target : str
        Target condition (default: 'left')
    metric : str
        Connectivity metric (default: 'imcoh')
    
    Returns:
    --------
    dict : Dictionary containing all loaded results
    """
    
    print(f"Loading seeded connectivity results for {len(subjects)} subjects...")
    print(f"  Seed: {seed}, Target: {target}, Metric: {metric}")
    
    # Initialize storage for all results
    all_results = {
        'subjects': subjects,
        'data': [],  # List to store arrays from each subject
        'time_vector': None,
        'loaded_subjects': [],
        'seed': seed,
        'target': target,
        'metric': metric
    }
    
    for subjID in subjects:
        print(f"Loading subject {subjID:02d}...")
        
        # Construct file path for seeded connectivity
        outputDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', 'connectivity')
        outputFile = os.path.join(outputDir, f'sub-{subjID:02d}_task-{taskName}_seededConnectivity_{voxRes}_{seed}_{target}_{metric}.pkl')
        
        if os.path.exists(outputFile):
            try:
                with open(outputFile, 'rb') as f:
                    # Load the single numpy array
                    data_array = pickle.load(f)
                
                print(f"  Loaded array with shape: {data_array.shape}")
                
                # Store time vector (should be the same for all subjects)
                # Assuming time range from -1.0 to 2.0 seconds
                if all_results['time_vector'] is None:
                    n_timepoints = data_array.shape[1]  # Second dimension is time
                    all_results['time_vector'] = np.linspace(-1.0, 2.0, n_timepoints)
                    print(f"  Time vector: {all_results['time_vector'][0]:.2f}s to {all_results['time_vector'][-1]:.2f}s ({n_timepoints} time points)")
                
                time_vector = all_results['time_vector']
                
                # Define baseline period: -0.5 to 0.0 seconds
                baseline_start = -1.0
                baseline_end = 0
                baseline_mask = (time_vector >= baseline_start) & (time_vector <= baseline_end)
                baseline_indices = np.where(baseline_mask)[0]
                
                print(f"  Baseline correction: {baseline_start}s to {baseline_end}s ({len(baseline_indices)} time points)")
                
                # Apply baseline correction to each row (voxel/connection)
                # Shape is (n_voxels, n_timepoints)
                corrected_data = np.zeros_like(data_array)
                for i in range(data_array.shape[0]):
                    if len(baseline_indices) > 0:
                        baseline_mean = np.nanmean(data_array[i, baseline_indices])
                        if baseline_mean != 0 and not np.isnan(baseline_mean):
                            corrected_data[i, :] = data_array[i, :] / baseline_mean - 1
                        else:
                            corrected_data[i, :] = data_array[i, :]
                    else:
                        corrected_data[i, :] = data_array[i, :]
                
                all_results['data'].append(corrected_data)
                all_results['loaded_subjects'].append(subjID)
                print(f"  Successfully loaded and baseline-corrected subject {subjID:02d}")
                
            except Exception as e:
                print(f"  Error loading subject {subjID:02d}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  File not found for subject {subjID:02d}: {outputFile}")
    
    print(f"Successfully loaded {len(all_results['loaded_subjects'])} subjects")
    return all_results

def load_atlas_rois(bidsRoot, voxRes='10mm'):
    """
    Load atlas data and extract ROI indices
    
    Parameters:
    -----------
    bidsRoot : str
        Root directory for BIDS data
    voxRes : str
        Voxel resolution (default: '10mm')
    
    Returns:
    --------
    dict : Dictionary containing ROI indices
    """
    
    atlas_fpath = os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat')
    
    if not os.path.exists(atlas_fpath):
        raise FileNotFoundError(f"Atlas file not found: {atlas_fpath}")
    
    atlas_data = loadmat(atlas_fpath)
    
    # Define ROI indices
    visual_points = np.array(atlas_data['visual_points']).flatten()
    left_visual_points = np.array(atlas_data['left_visual_points']).flatten()
    right_visual_points = np.array(atlas_data['right_visual_points']).flatten()
    left_frontal_points = np.array(atlas_data['left_frontal_points']).flatten()
    right_frontal_points = np.array(atlas_data['right_frontal_points']).flatten()
    left_parietal_points = np.array(atlas_data['left_parietal_points']).flatten()
    right_parietal_points = np.array(atlas_data['right_parietal_points']).flatten()
    
    roi_indices = {
        'left_frontal': np.where(left_frontal_points == 1)[0],
        'right_frontal': np.where(right_frontal_points == 1)[0],
        'left_visual': np.where(left_visual_points == 1)[0],
        'right_visual': np.where(right_visual_points == 1)[0],
        'left_parietal': np.where(left_parietal_points == 1)[0],
        'right_parietal': np.where(right_parietal_points == 1)[0],
        'visual': np.where(visual_points == 1)[0]  # Combined visual
    }
    
    print(f"Loaded atlas ROIs:")
    for roi_name, indices in roi_indices.items():
        print(f"  {roi_name}: {len(indices)} voxels")
    
    return roi_indices

def extract_roi_connectivity(all_results, roi_indices):
    """
    Extract ROI-averaged connectivity from loaded data
    
    Parameters:
    -----------
    all_results : dict
        Results from load_connectivity_results
    roi_indices : dict
        ROI indices from load_atlas_rois
    
    Returns:
    --------
    dict : Dictionary containing ROI-averaged connectivity for each subject
    """
    
    print("Extracting ROI-averaged connectivity...")
    
    roi_connectivity = {
        'left_frontal': [],
        'right_frontal': [],
        'left_visual': [],
        'right_visual': [],
        'left_parietal': [],
        'right_parietal': []
    }
    
    for subject_data in all_results['data']:
        # subject_data shape: (n_voxels, n_timepoints)
        # Average across voxels within each ROI
        for roi_name in ['left_frontal', 'right_frontal', 'left_visual', 'right_visual', 'left_parietal', 'right_parietal']:
            if roi_name in roi_indices:
                roi_idx = roi_indices[roi_name]
                # Check if indices are within bounds
                valid_idx = roi_idx[roi_idx < subject_data.shape[0]]
                if len(valid_idx) > 0:
                    roi_mean = np.nanmean(subject_data[valid_idx, :], axis=0)
                    roi_connectivity[roi_name].append(roi_mean)
                else:
                    print(f"  Warning: No valid indices for {roi_name}")
                    roi_connectivity[roi_name].append(np.full(subject_data.shape[1], np.nan))
    
    return roi_connectivity

def compute_roi_averages(roi_connectivity):
    """
    Compute average connectivity across subjects for each ROI
    
    Parameters:
    -----------
    roi_connectivity : dict
        ROI connectivity from extract_roi_connectivity
    
    Returns:
    --------
    dict : Dictionary containing mean and SEM for each ROI
    """
    
    print("Computing averages across subjects...")
    
    averaged = {}
    
    for roi_name, subject_data_list in roi_connectivity.items():
        if len(subject_data_list) > 0:
            # Stack subjects: (n_subjects, n_timepoints)
            stacked = np.stack(subject_data_list, axis=0)
            averaged[roi_name] = {
                'mean': np.nanmean(stacked, axis=0),
                'sem': np.nanstd(stacked, axis=0) / np.sqrt(len(subject_data_list))
            }
            print(f"  {roi_name}: {len(subject_data_list)} subjects")
    
    return averaged

def plot_roi_connectivity_patterns(averaged_roi_connectivity_left, averaged_roi_connectivity_right, 
                                   time_vector, seed_name='left_frontal'):
    """
    Plot connectivity patterns between ROI pairs as a function of time
    Shows both left and right targets for comparison
    
    Parameters:
    -----------
    averaged_roi_connectivity_left : dict
        Averaged ROI connectivity from compute_roi_averages for left targets
    averaged_roi_connectivity_right : dict
        Averaged ROI connectivity from compute_roi_averages for right targets
    time_vector : array
        Time vector for x-axis
    seed_name : str
        Name of the seed region (for title)
    """
    
    print("Plotting ROI connectivity patterns...")
    
    # Define ROI pairs to plot
    # Since this is seeded connectivity, we plot connectivity TO each ROI
    roi_pairs = [
        ('left_frontal', 'Left Frontal'),
        ('right_frontal', 'Right Frontal'),
        ('left_visual', 'Left Visual'),
        ('right_visual', 'Right Visual'),
        ('left_parietal', 'Left Parietal'),
        ('right_parietal', 'Right Parietal')
    ]
    
    # Create figure with subplots (3 rows, 2 columns for 6 ROIs)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(f'Connectivity from {seed_name.replace("_", " ").title()} Seed to Target ROIs', 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    
    for i, (roi_key, roi_label) in enumerate(roi_pairs):
        ax = axes_flat[i]
        
        # Plot left targets (blue)
        if roi_key in averaged_roi_connectivity_left:
            mean_vals_left = averaged_roi_connectivity_left[roi_key]['mean']
            sem_vals_left = averaged_roi_connectivity_left[roi_key]['sem']
            
            ax.plot(time_vector, mean_vals_left, linewidth=2.5, 
                   label='Left Targets', color='blue')
            ax.fill_between(time_vector, mean_vals_left - sem_vals_left, mean_vals_left + sem_vals_left,
                           alpha=0.3, color='blue')
        
        # Plot right targets (orange)
        if roi_key in averaged_roi_connectivity_right:
            mean_vals_right = averaged_roi_connectivity_right[roi_key]['mean']
            sem_vals_right = averaged_roi_connectivity_right[roi_key]['sem']
            
            ax.plot(time_vector, mean_vals_right, linewidth=2.5, 
                   label='Right Targets', color='orange')
            ax.fill_between(time_vector, mean_vals_right - sem_vals_right, mean_vals_right + sem_vals_right,
                           alpha=0.3, color='orange')
        
        # Add reference lines
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Stimulus Onset')
        ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, linewidth=1, label='Delay Start')
        ax.axvline(x=1.7, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Response')
        
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Relative Connectivity', fontsize=12)
        ax.set_title(f'{seed_name.replace("_", " ").title()} → {roi_label}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(time_vector[0], time_vector[-1])
    
    plt.tight_layout()
    plt.show()

def load_sourcemodel(sourcemodel_path):
    """
    Load sourcemodel from MATLAB file
    
    Parameters:
    -----------
    sourcemodel_path : str
        Path to sourcemodel MATLAB file
    
    Returns:
    --------
    pos : array
        Positions of inside vertices (n_inside, 3)
    inside : array
        Boolean array indicating which vertices are inside
    """
    print(f"Loading sourcemodel from {sourcemodel_path}...")
    
    # Handle h5py for MATLAB v7.3 files or loadmat for older versions
    if socket.gethostname() == 'zod':
        # Use temporary copy approach for h5py files
        sourcemodel_temp_path = os.path.join('/Users/mrugank/Desktop', 'sourcemodel_temp.mat')
        if os.path.exists(sourcemodel_path):
            copyfile(sourcemodel_path, sourcemodel_temp_path)
            try:
                sourcemodel_data = h5py.File(sourcemodel_temp_path, 'r')
                sourcemodel = sourcemodel_data['sourcemodel']
                pos_ref = sourcemodel['pos']
                inside_ref = sourcemodel['inside']
                pos = np.array(sourcemodel_data[pos_ref]).T  # Transpose to (n_vertices, 3)
                inside = np.array(sourcemodel_data[inside_ref]).flatten()
                sourcemodel_data.close()
                os.remove(sourcemodel_temp_path)
            except:
                # Try loadmat if h5py fails
                sourcemodel_data = loadmat(sourcemodel_path)
                sourcemodel = sourcemodel_data['sourcemodel']
                pos = sourcemodel['pos'][0, 0]
                inside = sourcemodel['inside'][0, 0].flatten()
                if os.path.exists(sourcemodel_temp_path):
                    os.remove(sourcemodel_temp_path)
        else:
            raise FileNotFoundError(f"Sourcemodel file not found: {sourcemodel_path}")
    else:
        # Try h5py first, then loadmat
        try:
            sourcemodel_data = h5py.File(sourcemodel_path, 'r')
            sourcemodel = sourcemodel_data['sourcemodel']
            pos_ref = sourcemodel['pos']
            inside_ref = sourcemodel['inside']
            pos = np.array(sourcemodel_data[pos_ref]).T  # Transpose to (n_vertices, 3)
            inside = np.array(sourcemodel_data[inside_ref]).flatten()
            sourcemodel_data.close()
        except:
            sourcemodel_data = loadmat(sourcemodel_path)
            sourcemodel = sourcemodel_data['sourcemodel']
            pos = sourcemodel['pos'][0, 0]
            inside = sourcemodel['inside'][0, 0].flatten()
    
    # Get positions of inside vertices
    inside_pos = pos[inside == 1]
    
    print(f"  Loaded {len(pos)} total vertices, {len(inside_pos)} inside vertices")
    
    return inside_pos, inside

def compute_delay_average_connectivity(all_results, time_window=(0.5, 1.5)):
    """
    Compute average connectivity across subjects for delay period
    
    Parameters:
    -----------
    all_results : dict
        Results from load_connectivity_results
    time_window : tuple
        Time window (start, end) in seconds (default: (0.5, 1.5) for delay period)
    
    Returns:
    --------
    avg_connectivity : array
        Average connectivity across subjects and time window (n_voxels,)
    """
    print(f"Computing average connectivity for delay period ({time_window[0]}-{time_window[1]}s)...")
    
    time_vector = all_results['time_vector']
    
    # Find time indices for delay period
    time_mask = (time_vector >= time_window[0]) & (time_vector <= time_window[1])
    time_indices = np.where(time_mask)[0]
    
    if len(time_indices) == 0:
        raise ValueError(f"No time points found in window {time_window[0]}-{time_window[1]}s")
    
    print(f"  Found {len(time_indices)} time points in delay window")
    
    # Average across time and subjects
    all_delay_data = []
    for subject_data in all_results['data']:
        # subject_data shape: (n_voxels, n_timepoints)
        delay_data = subject_data[:, time_indices]  # (n_voxels, n_delay_timepoints)
        delay_mean = np.nanmean(delay_data, axis=1)  # Average across time: (n_voxels,)
        all_delay_data.append(delay_mean)
    
    # Stack and average across subjects
    stacked = np.stack(all_delay_data, axis=0)  # (n_subjects, n_voxels)
    avg_connectivity = np.nanmean(stacked, axis=0)  # (n_voxels,)
    
    print(f"  Computed average connectivity for {len(avg_connectivity)} voxels across {len(all_delay_data)} subjects")
    
    return avg_connectivity

def load_cortical_surface(cortex_surf_path):
    """
    Load cortical surface from GIFTI file
    
    Parameters:
    -----------
    cortex_surf_path : str
        Path to cortex_surf.gii file
    
    Returns:
    --------
    vertices : array
        Surface vertices (n_vertices, 3)
    faces : array
        Surface faces/triangles (n_faces, 3)
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel is required to load GIFTI files. Install with: pip install nibabel")
    
    print(f"Loading cortical surface from {cortex_surf_path}...")
    
    try:
        gii = nib.load(cortex_surf_path)
        
        # Find vertices and faces arrays
        # GIFTI files can have different structures, so we need to identify them
        vertices = None
        faces = None
        
        for i, darray in enumerate(gii.darrays):
            intent = darray.intent
            data = darray.data
            
            # NIFTI_INTENT_POINTSET (1008) or NIFTI_INTENT_TRIANGLE (1009)
            if intent == 1008:  # Points (vertices)
                vertices = data
                print(f"  Found vertices in array {i}: shape {vertices.shape}")
            elif intent == 1009:  # Triangles (faces)
                faces = data
                print(f"  Found faces in array {i}: shape {faces.shape}")
        
        # Fallback: if intent codes don't work, try by shape
        if vertices is None or faces is None:
            for i, darray in enumerate(gii.darrays):
                data = darray.data
                if len(data.shape) == 2 and data.shape[1] == 3:
                    if vertices is None:
                        vertices = data
                        print(f"  Using array {i} as vertices (shape {vertices.shape})")
                    elif faces is None and data.shape[0] > vertices.shape[0]:
                        # Faces typically have more rows than vertices
                        faces = data
                        print(f"  Using array {i} as faces (shape {faces.shape})")
        
        if vertices is None or faces is None:
            raise ValueError("Could not identify vertices and faces in GIFTI file")
        
        print(f"  Successfully loaded {len(vertices)} vertices and {len(faces)} faces")
        return vertices, faces
    except Exception as e:
        print(f"Error loading GIFTI file: {e}")
        import traceback
        traceback.print_exc()
        raise

def map_connectivity_to_surface(connectivity_data, source_pos, surface_vertices, k=5):
    """
    Map connectivity data from source positions to surface vertices using nearest neighbor interpolation
    
    Parameters:
    -----------
    connectivity_data : array
        Connectivity values at source positions (n_sources,)
    source_pos : array
        Source positions (n_sources, 3)
    surface_vertices : array
        Surface vertex positions (n_vertices, 3)
    k : int
        Number of nearest neighbors to use for interpolation (default: 5)
    
    Returns:
    --------
    surface_connectivity : array
        Connectivity values mapped to surface vertices (n_vertices,)
    """
    print(f"Mapping connectivity data to {len(surface_vertices)} surface vertices...")
    
    # Compute distances between surface vertices and source positions
    distances = cdist(surface_vertices, source_pos)  # (n_vertices, n_sources)
    
    # For each surface vertex, find k nearest sources and use inverse distance weighting
    surface_connectivity = np.zeros(len(surface_vertices))
    
    for i in range(len(surface_vertices)):
        # Get k nearest neighbors
        nearest_indices = np.argsort(distances[i])[:k]
        nearest_distances = distances[i, nearest_indices]
        
        # Avoid division by zero
        nearest_distances = np.maximum(nearest_distances, 1e-10)
        
        # Inverse distance weighting
        weights = 1.0 / nearest_distances
        weights = weights / np.sum(weights)
        
        # Weighted average
        surface_connectivity[i] = np.sum(weights * connectivity_data[nearest_indices])
    
    print(f"  Mapped connectivity to surface vertices")
    return surface_connectivity

def plot_whole_brain_connectivity_surface(avg_connectivity_left, avg_connectivity_right, 
                                          inside_pos, surface_vertices, surface_faces,
                                          seed_name='left_frontal', 
                                          time_window=(0.5, 1.5)):
    """
    Create 3D surface plot of whole-brain connectivity during delay period
    
    Parameters:
    -----------
    avg_connectivity_left : array
        Average connectivity for left targets (n_voxels,)
    avg_connectivity_right : array
        Average connectivity for right targets (n_voxels,)
    inside_pos : array
        Positions of inside vertices (n_inside, 3)
    surface_vertices : array
        Surface vertex positions (n_vertices, 3)
    surface_faces : array
        Surface face indices (n_faces, 3)
    seed_name : str
        Name of the seed region (for title)
    time_window : tuple
        Time window used for averaging (for title)
    """
    print("Creating 3D surface visualization of whole-brain connectivity...")
    
    # Filter to left hemisphere only
    left_vertices, left_faces, left_vertex_indices = filter_left_hemisphere(surface_vertices, surface_faces)
    
    # Map connectivity data to full surface first, then filter to left hemisphere
    surface_conn_left_full = map_connectivity_to_surface(avg_connectivity_left, inside_pos, surface_vertices)
    surface_conn_right_full = map_connectivity_to_surface(avg_connectivity_right, inside_pos, surface_vertices)
    
    # Filter connectivity values to left hemisphere
    surface_conn_left = surface_conn_left_full[left_vertex_indices]
    surface_conn_right = surface_conn_right_full[left_vertex_indices]
    
    # Create figure with two subplots (left and right targets)
    fig = plt.figure(figsize=(18, 8))
    
    # Left targets
    ax1 = fig.add_subplot(121, projection='3d')
    plot_surface_mesh(ax1, left_vertices, left_faces, surface_conn_left, 
                     f'Left Targets\n{seed_name.replace("_", " ").title()} Seed, {time_window[0]}-{time_window[1]}s',
                     elev=0, azim=210)  # Left hemisphere view
    
    # Right targets
    ax2 = fig.add_subplot(122, projection='3d')
    plot_surface_mesh(ax2, left_vertices, left_faces, surface_conn_right,
                     f'Right Targets\n{seed_name.replace("_", " ").title()} Seed, {time_window[0]}-{time_window[1]}s',
                     elev=0, azim=210)  # Left hemisphere view
    
    plt.tight_layout()
    plt.show()
    
    print("3D surface visualization completed!")

def plot_whole_brain_connectivity_multiple_windows(connectivity_data_left, connectivity_data_right,
                                                    inside_pos, surface_vertices, surface_faces,
                                                    seed_name='left_frontal', 
                                                    time_windows=[(-0.5, 0.0), (0.0, 0.5), (0.8, 1.5)],
                                                    save_path=None):
    """
    Create 3D surface plot of whole-brain connectivity for multiple time windows
    Shows 3 rows with 4 subplots each: Left hemisphere (left targets), Left hemisphere (right targets),
    Right hemisphere (left targets), Right hemisphere (right targets)
    
    Parameters:
    -----------
    connectivity_data_left : dict
        Dictionary with time window keys and connectivity arrays for left targets
    connectivity_data_right : dict
        Dictionary with time window keys and connectivity arrays for right targets
    inside_pos : array
        Positions of inside vertices (n_inside, 3)
    surface_vertices : array
        Surface vertex positions (n_vertices, 3)
    surface_faces : array
        Surface face indices (n_faces, 3)
    seed_name : str
        Name of the seed region (for title)
    time_windows : list of tuples
        List of time windows to plot (3 windows)
    """
    print("Creating 3D surface visualization for multiple time windows...")
    
    # Filter to both hemispheres (do this once for all plots)
    left_vertices, left_faces, left_vertex_indices = filter_left_hemisphere(surface_vertices, surface_faces)
    right_vertices, right_faces, right_vertex_indices = filter_right_hemisphere(surface_vertices, surface_faces)
    
    # Create figure with 3 rows x 4 columns (larger size for bigger brains)
    fig = plt.figure(figsize=(20, 15))
    
    # Add figure title with seed name at the top
    seed_label = seed_name.replace("_", " ").title()
    fig.suptitle(f'{seed_label} Seed', fontsize=16, fontweight='bold', y=0.98)
    
    # Use gridspec for better control over spacing
    from matplotlib import gridspec
    # Create grid with 3 rows and 4 columns, leave minimal space at bottom for small colorbar
    # Leave more space on left for time period labels
    gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[1, 1, 1, 1], 
                           height_ratios=[1, 1, 1],
                           wspace=0.05, hspace=0.15, left=0.08, right=0.95, top=0.92, bottom=0.08)
    
    gap_between_pairs = 0.08  # Larger gap between the two pairs
    gap_within_pair = 0.005   # Very small gap within each pair (reduced)
    
    # Loop through each time window (row)
    for row_idx, time_window in enumerate(time_windows):
        window_key = f"{time_window[0]}_{time_window[1]}"
        
        # Map connectivity to full surface first
        surface_conn_left_full = map_connectivity_to_surface(
            connectivity_data_left[window_key], inside_pos, surface_vertices)
        surface_conn_right_full = map_connectivity_to_surface(
            connectivity_data_right[window_key], inside_pos, surface_vertices)
        
        # Filter to each hemisphere
        surface_conn_left_left_hem = surface_conn_left_full[left_vertex_indices]
        surface_conn_right_left_hem = surface_conn_right_full[left_vertex_indices]
        surface_conn_left_right_hem = surface_conn_left_full[right_vertex_indices]
        surface_conn_right_right_hem = surface_conn_right_full[right_vertex_indices]
        
        # Create simplified window label
        if time_window[0] < 0:
            window_label = "Fixation"
        elif time_window[0] == 0:
            window_label = "Stimulus"
        else:
            window_label = "Delay"
        
        # Subplot 1: Left hemisphere, Left targets
        ax1 = fig.add_subplot(gs[row_idx, 0], projection='3d')
        plot_surface_mesh(ax1, left_vertices, left_faces, 
                         surface_conn_left_left_hem,
                         '',  # No individual title
                         elev=30, azim=220, show_colorbar=False)  # Left hemisphere view
        
        # Subplot 2: Right hemisphere, Left targets
        ax2 = fig.add_subplot(gs[row_idx, 1], projection='3d')
        plot_surface_mesh(ax2, right_vertices, right_faces,
                         surface_conn_left_right_hem,
                         '',  # No individual title
                         elev=30, azim=-40, show_colorbar=False)  # Right hemisphere view
        
        # Subplot 3: Left hemisphere, Right targets
        ax3 = fig.add_subplot(gs[row_idx, 2], projection='3d')
        plot_surface_mesh(ax3, left_vertices, left_faces,
                         surface_conn_right_left_hem,
                         '',  # No individual title
                         elev=30, azim=220, show_colorbar=False)  # Left hemisphere view
        
        # Subplot 4: Right hemisphere, Right targets
        ax4 = fig.add_subplot(gs[row_idx, 3], projection='3d')
        plot_surface_mesh(ax4, right_vertices, right_faces,
                         surface_conn_right_right_hem,
                         '',  # No individual title
                         elev=30, azim=-40, show_colorbar=False)  # Right hemisphere view
        
        # Manually adjust spacing: make pairs closer, add space between pairs
        # Get positions
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        pos3 = ax3.get_position()
        pos4 = ax4.get_position()
        
        # Position subplot 1
        ax1.set_position([pos1.x0, pos1.y0, pos1.width, pos1.height])
        
        # Position subplot 2 (close to subplot 1 - Left targets pair)
        ax2.set_position([pos1.x1 + gap_within_pair, pos2.y0, pos2.width, pos2.height])
        
        # Position subplot 3 (with larger gap from subplot 2 - Right targets pair starts)
        ax3.set_position([pos2.x1 + gap_between_pairs, pos3.y0, pos3.width, pos3.height])
        
        # Position subplot 4 (close to subplot 3 - Right targets pair)
        ax4.set_position([pos3.x1 + gap_within_pair, pos4.y0, pos4.width, pos4.height])
        
        # Add titles above each pair (only for top row)
        if row_idx == 0:
            # Title for Left Targets pair (above subplots 1-2)
            pos_pair1 = ax1.get_position()
            fig.text((pos_pair1.x0 + pos2.x1) / 2, pos_pair1.y1 + 0.02, 
                     'Left Targets',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            # Title for Right Targets pair (above subplots 3-4)
            pos_pair2 = ax3.get_position()
            fig.text((pos_pair2.x0 + pos4.x1) / 2, pos_pair2.y1 + 0.02,
                     'Right Targets',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add time period label on the left side of the row (only once per row)
        pos_row = ax1.get_position()
        fig.text(pos_row.x0 - 0.02, (pos_row.y0 + pos_row.y1) / 2,
                 window_label,
                 ha='right', va='center', fontsize=12, fontweight='bold', rotation=90)
    
    # Add single colorbar at the bottom of the figure
    # Get colormap
    try:
        cmap = plt.colormaps['RdBu_r']
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap('RdBu_r')
    
    # Create small colorbar at the bottom
    vmin, vmax = -0.15, 0.15
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation='horizontal', 
                        pad=0.02, shrink=0.6, aspect=40, fraction=0.03)
    cbar.set_label('Relative Connectivity', fontsize=10, labelpad=5)
    cbar.ax.tick_params(labelsize=8)
    
    # Save figure if save_path is provided
    if save_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Save as high-resolution PNG
        fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    print("3D surface visualization completed!")

def filter_left_hemisphere(vertices, faces):
    """
    Filter surface to only include left hemisphere vertices and faces
    
    Parameters:
    -----------
    vertices : array
        Vertex positions (n_vertices, 3)
    faces : array
        Face indices (n_faces, 3)
    
    Returns:
    --------
    left_vertices : array
        Left hemisphere vertex positions
    left_faces : array
        Left hemisphere face indices (remapped to new vertex indices)
    vertex_map : dict
        Mapping from old vertex indices to new vertex indices
    """
    # Left hemisphere typically has X < 0 (or X < midpoint)
    # Use median X as threshold to be robust
    x_threshold = np.median(vertices[:, 0])
    left_hemisphere_mask = vertices[:, 0] < x_threshold
    
    # Get left hemisphere vertices
    left_vertex_indices = np.where(left_hemisphere_mask)[0]
    left_vertices = vertices[left_vertex_indices]
    
    # Create mapping from old indices to new indices
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(left_vertex_indices)}
    
    # Filter faces: only keep faces where all vertices are in left hemisphere
    left_faces = []
    for face in faces:
        if all(v_idx in vertex_map for v_idx in face):
            # Remap face indices to new vertex indices
            remapped_face = [vertex_map[v_idx] for v_idx in face]
            left_faces.append(remapped_face)
    
    left_faces = np.array(left_faces)
    
    print(f"  Filtered to left hemisphere: {len(left_vertices)} vertices, {len(left_faces)} faces")
    
    return left_vertices, left_faces, left_vertex_indices

def filter_right_hemisphere(vertices, faces):
    """
    Filter surface to only include right hemisphere vertices and faces
    
    Parameters:
    -----------
    vertices : array
        Vertex positions (n_vertices, 3)
    faces : array
        Face indices (n_faces, 3)
    
    Returns:
    --------
    right_vertices : array
        Right hemisphere vertex positions
    right_faces : array
        Right hemisphere face indices (remapped to new vertex indices)
    vertex_map : dict
        Mapping from old vertex indices to new vertex indices
    """
    # Right hemisphere typically has X > 0 (or X > midpoint)
    # Use median X as threshold to be robust
    x_threshold = np.median(vertices[:, 0])
    right_hemisphere_mask = vertices[:, 0] >= x_threshold
    
    # Get right hemisphere vertices
    right_vertex_indices = np.where(right_hemisphere_mask)[0]
    right_vertices = vertices[right_vertex_indices]
    
    # Create mapping from old indices to new indices
    vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(right_vertex_indices)}
    
    # Filter faces: only keep faces where all vertices are in right hemisphere
    right_faces = []
    for face in faces:
        if all(v_idx in vertex_map for v_idx in face):
            # Remap face indices to new vertex indices
            remapped_face = [vertex_map[v_idx] for v_idx in face]
            right_faces.append(remapped_face)
    
    right_faces = np.array(right_faces)
    
    print(f"  Filtered to right hemisphere: {len(right_vertices)} vertices, {len(right_faces)} faces")
    
    return right_vertices, right_faces, right_vertex_indices

def plot_surface_mesh(ax, vertices, faces, values, title, elev=0, azim=210, show_colorbar=False):
    """
    Plot a 3D surface mesh with color-coded values
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        Axis to plot on
    vertices : array
        Vertex positions (n_vertices, 3)
    faces : array
        Face indices (n_faces, 3)
    values : array
        Values to color-code (n_vertices,)
    title : str
        Plot title
    elev : float
        Elevation angle for view (default: 0)
    azim : float
        Azimuth angle for view (default: 210 for left hemisphere, 30 for right hemisphere)
    show_colorbar : bool
        Whether to show colorbar for this subplot (default: False)
    """
    # Use fixed color scale from -0.1 to 0.1
    vmin, vmax = -0.15, 0.15
    norm_values = (values - vmin) / (vmax - vmin + 1e-10)
    norm_values = np.clip(norm_values, 0, 1)
    
    # Get colormap
    try:
        cmap = plt.colormaps['RdBu_r']
    except (AttributeError, KeyError):
        # Fallback for older matplotlib versions
        cmap = plt.cm.get_cmap('RdBu_r')
    colors = cmap(norm_values)
    
    # Create Poly3DCollection for surface
    triangles = []
    triangle_colors = []
    
    for face in faces:
        # Get triangle vertices
        tri_verts = vertices[face]
        triangles.append(tri_verts)
        
        # Average color of triangle vertices
        tri_color = np.mean(colors[face], axis=0)
        triangle_colors.append(tri_color)
    
    collection = Poly3DCollection(triangles, facecolors=triangle_colors, 
                                  edgecolors='black', alpha=1.0, linewidths=0.1)
    ax.add_collection3d(collection)
    
    # Set axis limits with equal aspect ratio to prevent stretching
    # Calculate center and range for each dimension
    x_center = (vertices[:, 0].min() + vertices[:, 0].max()) / 2
    y_center = (vertices[:, 1].min() + vertices[:, 1].max()) / 2
    z_center = (vertices[:, 2].min() + vertices[:, 2].max()) / 2
    
    x_range = vertices[:, 0].max() - vertices[:, 0].min()
    y_range = vertices[:, 1].max() - vertices[:, 1].min()
    z_range = vertices[:, 2].max() - vertices[:, 2].min()
    
    # Use the maximum range to set symmetric limits (prevents stretching)
    max_range = max(x_range, y_range, z_range)
    
    # Set symmetric limits around the center
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Turn off axes, labels, ticks, and grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.grid(False)
    
    # Turn off axis panes (the background planes)
    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.xaxis.pane.set_alpha(0)
    ax.yaxis.pane.set_alpha(0)
    ax.zaxis.pane.set_alpha(0)
    
    # Turn off axis lines (spines/edges)
    ax.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    # Alternative method: set linewidth to 0
    ax.xaxis.line.set_linewidth(0)
    ax.yaxis.line.set_linewidth(0)
    ax.zaxis.line.set_linewidth(0)
    # Also turn off tick lines
    ax.xaxis._axinfo['tick']['inward_factor'] = 0
    ax.xaxis._axinfo['tick']['outward_factor'] = 0
    ax.yaxis._axinfo['tick']['inward_factor'] = 0
    ax.yaxis._axinfo['tick']['outward_factor'] = 0
    ax.zaxis._axinfo['tick']['inward_factor'] = 0
    ax.zaxis._axinfo['tick']['outward_factor'] = 0
    
    # Create colorbar only if requested
    if show_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Relative Connectivity', rotation=270, labelpad=20)
    
    # Only set title if provided
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Set view angle
    ax.view_init(elev=elev, azim=azim)

def plot_whole_brain_connectivity_3d(avg_connectivity_left, avg_connectivity_right, 
                                     inside_pos, seed_name='left_frontal', 
                                     time_window=(0.5, 1.5), use_surface=False,
                                     surface_vertices=None, surface_faces=None,
                                     save_path=None):
    """
    Create 3D visualization of whole-brain connectivity during delay period
    Can use either scatter plot or surface mesh
    
    Parameters:
    -----------
    avg_connectivity_left : array
        Average connectivity for left targets (n_voxels,)
    avg_connectivity_right : array
        Average connectivity for right targets (n_voxels,)
    inside_pos : array
        Positions of inside vertices (n_inside, 3)
    seed_name : str
        Name of the seed region (for title)
    time_window : tuple
        Time window used for averaging (for title)
    use_surface : bool
        If True, use surface mesh; if False, use scatter plot
    surface_vertices : array, optional
        Surface vertex positions (n_vertices, 3)
    surface_faces : array, optional
        Surface face indices (n_faces, 3)
    """
    if use_surface and surface_vertices is not None and surface_faces is not None:
        plot_whole_brain_connectivity_surface(avg_connectivity_left, avg_connectivity_right,
                                              inside_pos, surface_vertices, surface_faces,
                                              seed_name, time_window)
    else:
        # Fallback to scatter plot
        print("Creating 3D scatter plot visualization...")
        
        # Check that connectivity arrays match number of inside vertices
        n_voxels = len(avg_connectivity_left)
        if n_voxels != len(inside_pos):
            print(f"Warning: Connectivity data has {n_voxels} voxels but sourcemodel has {len(inside_pos)} inside vertices")
            n_plot = min(n_voxels, len(inside_pos))
            inside_pos = inside_pos[:n_plot]
            avg_connectivity_left = avg_connectivity_left[:n_plot]
            avg_connectivity_right = avg_connectivity_right[:n_plot]
        
        # Filter to left hemisphere only
        x_threshold = np.median(inside_pos[:, 0])
        left_hemisphere_mask = inside_pos[:, 0] < x_threshold
        inside_pos_left = inside_pos[left_hemisphere_mask]
        avg_connectivity_left_filtered = avg_connectivity_left[left_hemisphere_mask]
        avg_connectivity_right_filtered = avg_connectivity_right[left_hemisphere_mask]
        
        print(f"  Filtered to left hemisphere: {len(inside_pos_left)} voxels")
        
        # Create figure with two subplots (left and right targets)
        fig = plt.figure(figsize=(16, 7))
        
        # Use fixed color scale from -0.1 to 0.1
        vmin, vmax = -0.15, 0.15
        
        # Left targets
        ax1 = fig.add_subplot(121, projection='3d')
        scatter1 = ax1.scatter(inside_pos_left[:, 0], inside_pos_left[:, 1], inside_pos_left[:, 2], 
                              c=avg_connectivity_left_filtered, cmap='RdBu_r', alpha=1.0, s=20,
                              vmin=vmin, vmax=vmax)
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1)
        cbar1.set_label('Relative Connectivity', rotation=270, labelpad=20)
        ax1.set_title(f'Left Targets\n{seed_name.replace("_", " ").title()} Seed, Delay {time_window[0]}-{time_window[1]}s', 
                      fontsize=12, fontweight='bold')
        # Turn off axes, labels, ticks, and grid
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])
        ax1.set_xlabel('')
        ax1.set_ylabel('')
        ax1.set_zlabel('')
        ax1.grid(False)
        # Turn off axis panes (the background planes)
        ax1.xaxis.pane.set_visible(False)
        ax1.yaxis.pane.set_visible(False)
        ax1.zaxis.pane.set_visible(False)
        ax1.xaxis.pane.set_edgecolor('none')
        ax1.yaxis.pane.set_edgecolor('none')
        ax1.zaxis.pane.set_edgecolor('none')
        ax1.xaxis.pane.set_alpha(0)
        ax1.yaxis.pane.set_alpha(0)
        ax1.zaxis.pane.set_alpha(0)
        
        # Right targets
        ax2 = fig.add_subplot(122, projection='3d')
        scatter2 = ax2.scatter(inside_pos_left[:, 0], inside_pos_left[:, 1], inside_pos_left[:, 2], 
                              c=avg_connectivity_right_filtered, cmap='RdBu_r', alpha=1.0, s=20,
                              vmin=vmin, vmax=vmax)
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.1)
        cbar2.set_label('Relative Connectivity', rotation=270, labelpad=20)
        ax2.set_title(f'Right Targets\n{seed_name.replace("_", " ").title()} Seed, Delay {time_window[0]}-{time_window[1]}s', 
                      fontsize=12, fontweight='bold')
        # Turn off axes, labels, ticks, and grid
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_zticks([])
        ax2.set_xlabel('')
        ax2.set_ylabel('')
        ax2.set_zlabel('')
        ax2.grid(False)
        # Turn off axis panes (the background planes)
        ax2.xaxis.pane.set_visible(False)
        ax2.yaxis.pane.set_visible(False)
        ax2.zaxis.pane.set_visible(False)
        ax2.xaxis.pane.set_edgecolor('none')
        ax2.yaxis.pane.set_edgecolor('none')
        ax2.zaxis.pane.set_edgecolor('none')
        ax2.xaxis.pane.set_alpha(0)
        ax2.yaxis.pane.set_alpha(0)
        ax2.zaxis.pane.set_alpha(0)
        
        # plt.tight_layout()
        
        # Save figure if save_path is provided
        if save_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Save as high-resolution PNG
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
        
        print("3D scatter plot visualization completed!")

def main():
    """Main function to load and visualize ROI connectivity patterns"""
    
    # Define parameters
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    taskName = 'mgs'
    voxRes = '10mm'
    seed = 'right_frontal'
    # target = 'right'
    metric = 'imcoh'
    
    # Set bidsRoot based on hostname
    import socket
    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    print("="*60)
    print("ROI CONNECTIVITY PATTERNS VISUALIZATION")
    print("="*60)
    print(f"Subjects: {subjects}")
    print(f"Task: {taskName}")
    print(f"Voxel resolution: {voxRes}")
    print(f"Seed: {seed}, Metric: {metric}")
    print(f"BIDS root: {bidsRoot}")
    print("="*60)
    
    # Load connectivity results for left targets
    print("\n" + "="*60)
    print("Loading LEFT target data...")
    print("="*60)
    all_results_left = load_connectivity_results(bidsRoot, subjects, taskName, voxRes, 
                                                seed=seed, target='left', metric=metric)
    
    # Load connectivity results for right targets
    print("\n" + "="*60)
    print("Loading RIGHT target data...")
    print("="*60)
    all_results_right = load_connectivity_results(bidsRoot, subjects, taskName, voxRes, 
                                                 seed=seed, target='right', metric=metric)
    
    if len(all_results_left['loaded_subjects']) == 0 or len(all_results_right['loaded_subjects']) == 0:
        print("No connectivity results found!")
        return
    
    # Load atlas ROIs
    roi_indices = load_atlas_rois(bidsRoot, voxRes)
    
    # Extract ROI-averaged connectivity for left targets
    roi_connectivity_left = extract_roi_connectivity(all_results_left, roi_indices)
    
    # Extract ROI-averaged connectivity for right targets
    roi_connectivity_right = extract_roi_connectivity(all_results_right, roi_indices)
    
    # Compute averages across subjects
    averaged_roi_connectivity_left = compute_roi_averages(roi_connectivity_left)
    averaged_roi_connectivity_right = compute_roi_averages(roi_connectivity_right)
    
    # Plot results (use time vector from left results, should be the same)
    # plot_roi_connectivity_patterns(averaged_roi_connectivity_left, averaged_roi_connectivity_right,
    #                                all_results_left['time_vector'], seed_name=seed)
    
    # Whole-brain 3D visualization for multiple time windows
    print("\n" + "="*60)
    print("Creating whole-brain 3D connectivity visualization...")
    print("="*60)
    
    # Define time windows: pre-stimulus, early response, delay
    time_windows = [
        (-0.5, 0.0),  # Pre-stimulus
        (0.0, 0.5),   # Early response
        (0.8, 1.5)     # Late delay
    ]
    
    # Compute connectivity for each time window
    connectivity_data_left = {}
    connectivity_data_right = {}
    
    for time_window in time_windows:
        window_key = f"{time_window[0]}_{time_window[1]}"
        print(f"\nComputing connectivity for time window {time_window[0]}-{time_window[1]}s...")
        connectivity_data_left[window_key] = compute_delay_average_connectivity(all_results_left, time_window=time_window)
        connectivity_data_right[window_key] = compute_delay_average_connectivity(all_results_right, time_window=time_window)
    
    # Load sourcemodel
    if socket.gethostname() == 'zod':
        sourcemodel_path = '/System/Volumes/Data/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/standard_sourcemodel3d10mm.mat'
        # Try different surface resolutions (higher resolution = better visualization but slower)
        cortex_surf_paths = [
            '/System/Volumes/Data/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/cortex_8196.surf.gii',  # Medium-high res
            # '/System/Volumes/Data/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/cortex_20484.surf.gii',  # High res
            # '/System/Volumes/Data/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/cortex_5124.surf.gii',  # Medium res
        ]
    else:
        # Adjust paths for HPC if needed
        sourcemodel_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip/template/sourcemodel/standard_sourcemodel3d10mm.mat'
        cortex_surf_paths = [
            '/scratch/mdd9787/meg_prf_greene/fieldtrip/template/sourcemodel/cortex_8196.surf.gii',
            # '/scratch/mdd9787/meg_prf_greene/fieldtrip/template/sourcemodel/cortex_20484.surf.gii',
            # '/scratch/mdd9787/meg_prf_greene/fieldtrip/template/sourcemodel/cortex_5124.surf.gii',
        ]
    
    if os.path.exists(sourcemodel_path):
        inside_pos, inside = load_sourcemodel(sourcemodel_path)
        
        # Try to load cortical surface for better visualization
        surface_vertices = None
        surface_faces = None
        use_surface = False
        cortex_surf_path = None
        
        # Try different surface file paths
        for surf_path in cortex_surf_paths:
            if os.path.exists(surf_path):
                cortex_surf_path = surf_path
                break
        
        if cortex_surf_path and HAS_NIBABEL:
            try:
                surface_vertices, surface_faces = load_cortical_surface(cortex_surf_path)
                use_surface = True
                print(f"Using cortical surface mesh for visualization ({cortex_surf_path})")
            except Exception as e:
                print(f"Warning: Could not load cortical surface: {e}")
                print("Falling back to scatter plot visualization")
        else:
            if not cortex_surf_path:
                print(f"Warning: Cortical surface file not found. Tried:")
                for surf_path in cortex_surf_paths:
                    print(f"  - {surf_path}")
            if not HAS_NIBABEL:
                print("Warning: nibabel not available. Install with: pip install nibabel")
            print("Using scatter plot visualization")
        
        if use_surface and surface_vertices is not None and surface_faces is not None:
            # Use multi-window surface visualization
            save_path = os.path.join(bidsRoot, 'derivatives', 'figures', 'connectivity', f'{seed}seeds_{metric}_{voxRes}_surface.png')
            plot_whole_brain_connectivity_multiple_windows(
                connectivity_data_left, connectivity_data_right,
                inside_pos, surface_vertices, surface_faces,
                seed_name=seed, time_windows=time_windows,
                save_path=save_path)
        else:
            # Fallback: plot first time window only with scatter plot
            first_window = time_windows[0]
            first_key = f"{first_window[0]}_{first_window[1]}"
            save_path = os.path.join(bidsRoot, 'derivatives', 'figures', 'connectivity', f'{seed}seeds_{metric}_{voxRes}_scatter.png')
            plot_whole_brain_connectivity_3d(
                connectivity_data_left[first_key], connectivity_data_right[first_key],
                inside_pos, seed_name=seed, time_window=first_window,
                use_surface=False, 
                surface_vertices=None,
                surface_faces=None,
                save_path=save_path)
    else:
        print(f"Warning: Sourcemodel file not found at {sourcemodel_path}")
        print("Skipping 3D visualization. Please check the path.")
    
    print("Analysis completed!")

if __name__ == '__main__':
    main()