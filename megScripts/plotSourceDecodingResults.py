#!/usr/bin/env python3
"""
Script to load and plot source space decoding results across subjects.
Loads pickle files from all subjects, computes averages, and plots time courses.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_subject_results(bidsRoot, subjID, voxRes='8mm'):
    """Load results for a single subject"""
    subName = f'sub-{subjID:02d}'
    results_file = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', 
                               'betaDecodingVC', f'{subName}_task-mgs_betaSVR_{voxRes}_withBehav.pkl')
    
    if os.path.exists(results_file):
        print(f"Loading results for {subName}")
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        return results
    else:
        print(f"Results file not found for {subName}: {results_file}")
        return None

def load_all_subjects(bidsRoot, subject_list, voxRes='8mm'):
    """Load results for all subjects"""
    all_results = {
        'visual': {'errors': [], 'time': []},
        'parietal': {'errors': [], 'time': []},
        'frontal': {'errors': [], 'time': []}
    }
    
    valid_subjects = []
    
    for subjID in subject_list:
        results = load_subject_results(bidsRoot, subjID, voxRes)
        if results is not None:
            valid_subjects.append(subjID)
            
            # Store results for each region
            all_results['visual']['errors'].append(results['cv_angle_errors_visual'])
            all_results['visual']['time'].append(results['time_vector_svr_visual'])
            
            all_results['parietal']['errors'].append(results['cv_angle_errors_parietal'])
            all_results['parietal']['time'].append(results['time_vector_svr_parietal'])
            
            all_results['frontal']['errors'].append(results['cv_angle_errors_frontal'])
            all_results['frontal']['time'].append(results['time_vector_svr_frontal'])
    
    print(f"Loaded results for {len(valid_subjects)} subjects: {valid_subjects}")
    return all_results, valid_subjects

def compute_averages(all_results):
    """Compute mean and SEM across subjects for each region"""
    averages = {}
    
    for region in ['visual', 'parietal', 'frontal']:
        # Stack all subject data
        errors_array = np.array(all_results[region]['errors'])  # Shape: (n_subjects, n_timepoints)
        time_array = np.array(all_results[region]['time'])      # Shape: (n_subjects, n_timepoints, 1)
        
        # Compute mean and SEM
        mean_errors = np.mean(errors_array, axis=0)
        sem_errors = np.std(errors_array, axis=0) / np.sqrt(errors_array.shape[0])
        
        # Use time vector from first subject (should be the same for all)
        time_vector = time_array[0, :, 0]
        
        averages[region] = {
            'mean': mean_errors,
            'sem': sem_errors,
            'time': time_vector,
            'n_subjects': errors_array.shape[0]
        }
    
    return averages

def plot_time_courses(averages, title="SVR Angle Prediction Results (Source Space)"):
    """Plot time courses with error bars in 3 subplots"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = {'visual': 'red', 'parietal': 'green', 'frontal': 'blue'}
    regions = ['visual', 'parietal', 'frontal']
    
    for i, region in enumerate(regions):
        ax = axes[i]
        data = averages[region]
        
        # Plot mean line
        ax.plot(data['time'], data['mean'], 
                color=colors[region], linewidth=2, 
                label=f"{region.capitalize()} (n={data['n_subjects']})")
        
        # Add error bars (SEM)
        ax.fill_between(data['time'], 
                        data['mean'] - data['sem'], 
                        data['mean'] + data['sem'],
                        color=colors[region], alpha=0.2)
        
        # Add horizontal line at chance level
        ax.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='Chance Level (~90°)')
        
        # Formatting
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angular Error (°)')
        ax.set_title(f'{region.capitalize()} Region')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set consistent y-axis limits
        ax.set_ylim(80, 100)
        # Set x-axis limits to 1.7 seconds
        ax.set_xlim(-1.5, 1.7)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    # Define parameters
    subject_list = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    voxRes = '8mm'
    
    # Set bids root based on hostname
    import socket
    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    print("Loading source space decoding results...")
    print(f"BIDS root: {bidsRoot}")
    print(f"Voxel resolution: {voxRes}")
    print(f"Subject list: {subject_list}")
    
    # Load all subject results
    all_results, valid_subjects = load_all_subjects(bidsRoot, subject_list, voxRes)
    
    if len(valid_subjects) == 0:
        print("No valid results found!")
        return
    
    print(f"\nSuccessfully loaded {len(valid_subjects)} subjects")
    
    # Compute averages
    print("Computing averages across subjects...")
    averages = compute_averages(all_results)
    
    # Plot results
    print("Plotting time courses...")
    plot_time_courses(averages, title="SVR Angle Prediction Results (Source Space)")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    for region in ['visual', 'parietal', 'frontal']:
        data = averages[region]
        min_error = np.min(data['mean'])
        min_time = data['time'][np.argmin(data['mean'])]
        print(f"{region.capitalize()}: Min error = {min_error:.1f}° at t = {min_time:.2f}s")

if __name__ == '__main__':
    main()
