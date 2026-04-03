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

def load_connectivity_results(bidsRoot, subjects, taskName='mgs', voxRes='10mm'):
    """
    Load connectivity results for all subjects
    
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
    
    Returns:
    --------
    dict : Dictionary containing all loaded results
    """
    
    print(f"Loading connectivity results for {len(subjects)} subjects...")
    
    # Initialize storage for all results
    all_results = {
        'subjects': subjects,
        'coh_results': {},
        'imcoh_results': {},
        'dpli_results': {},
        'wpli_results': {},
        'time_vector': None,
        'loaded_subjects': []
    }
    
    for subjID in subjects:
        print(f"Loading subject {subjID:02d}...")
        
        # Construct file path variants
        dir_variants = [
            os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'connectivity_{voxRes}'),
            os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', 'connectivity')
        ]
        
        outputFile = None
        for outDir in dir_variants:
            f_legacy = f'sub-{subjID:02d}_task-{taskName}_connectivity_{voxRes}.pkl'
            f_seeded = f'sub-{subjID:02d}_task-{taskName}_seededConnectivity_{voxRes}.pkl'
            
            p1 = os.path.join(outDir, f_legacy)
            p2 = os.path.join(outDir, f_seeded)
            
            if os.path.exists(p1):
                outputFile = p1; break
            if os.path.exists(p2):
                outputFile = p2; break
        
        if outputFile:
            try:
                with open(outputFile, 'rb') as f:
                    # Load the dictionaries
                    results_coh = pickle.load(f)
                    results_imcoh = pickle.load(f)
                    results_plv = pickle.load(f)
                    results_dpli = pickle.load(f)
                    try:
                        results_wpli = pickle.load(f)
                    except EOFError:
                        results_wpli = None # Legacy 4-dict file
                
                # Store time vector (should be the same for all subjects)
                if all_results['time_vector'] is None:
                    # Create time vector based on the data length
                    all_results['time_vector'] = np.linspace(-1.0, 2.0, len(list(results_coh.values())[0]))
                
                time_vector = all_results['time_vector']
                
                # Define baseline period: -0.75 to 0.0 seconds
                baseline_start = -0.6
                baseline_end = 0.0
                baseline_mask = (time_vector >= baseline_start) & (time_vector <= baseline_end)
                baseline_indices = np.where(baseline_mask)[0]
                
                print(f"  Baseline correction: {baseline_start}s to {baseline_end}s ({len(baseline_indices)} time points)")
                
                # Apply baseline correction to coherence results
                for key, value in results_coh.items():
                    if key not in all_results['coh_results']:
                        all_results['coh_results'][key] = []
                    
                    # Baseline correction: subtract mean of baseline period
                    if len(baseline_indices) > 0:
                        baseline_mean = np.nanmean(value[baseline_indices])
                        corrected_value = value / baseline_mean - 1
                    else:
                        corrected_value = value
                    
                    all_results['coh_results'][key].append(corrected_value)
                
                # Apply baseline correction to imaginary coherence results
                for key, value in results_imcoh.items():
                    if key not in all_results['imcoh_results']:
                        all_results['imcoh_results'][key] = []
                    
                    # Baseline correction: subtract mean of baseline period
                    if len(baseline_indices) > 0:
                        baseline_mean = np.nanmean(value[baseline_indices])
                        corrected_value = value / baseline_mean - 1
                    else:
                        corrected_value = value
                    
                    all_results['imcoh_results'][key].append(corrected_value)
                
                # Store and baseline-subtract dPLI results
                for key, value in results_dpli.items():
                    if key not in all_results['dpli_results']:
                        all_results['dpli_results'][key] = []
                    
                    if len(baseline_indices) > 0:
                        baseline_mean = np.nanmean(value[baseline_indices])
                        corrected_value = value - baseline_mean
                    else:
                        corrected_value = value
                        
                    all_results['dpli_results'][key].append(corrected_value)
                
                # Store and baseline-subtract wPLI results (if available)
                if results_wpli:
                    for key, value in results_wpli.items():
                        if key not in all_results['wpli_results']:
                            all_results['wpli_results'][key] = []
                        
                        if len(baseline_indices) > 0:
                            # Use division correction for magnitude synchronization (wPLI)
                            baseline_mean = np.nanmean(value[baseline_indices])
                            corrected_value = value / (baseline_mean + 1e-10) - 1
                        else:
                            corrected_value = value
                        
                        all_results['wpli_results'][key].append(corrected_value)
                
                all_results['loaded_subjects'].append(subjID)
                print(f"  Successfully loaded and baseline-corrected subject {subjID:02d}")
                
            except Exception as e:
                print(f"  Error loading subject {subjID:02d}: {e}")
        else:
            print(f"  File not found for subject {subjID:02d}: {outputFile}")
    
    print(f"Successfully loaded {len(all_results['loaded_subjects'])} subjects")
    return all_results

def compute_averages(all_results):
    """
    Compute average coherence and imaginary coherence across subjects
    
    Parameters:
    -----------
    all_results : dict
        Results from load_connectivity_results
    
    Returns:
    --------
    dict : Dictionary containing averaged results
    """
    
    print("Computing averages across subjects...")
    
    averaged_results = {
        'coh_means': {},
        'coh_stds': {},
        'imcoh_means': {},
        'imcoh_stds': {},
        'dpli_means': {},
        'dpli_stds': {},
        'time_vector': all_results['time_vector'],
        'n_subjects': len(all_results['loaded_subjects'])
    }
    
    # Compute averages for coherence
    for key, subject_data in all_results['coh_results'].items():
        if len(subject_data) > 0:
            # Stack all subjects' data
            stacked_data = np.stack(subject_data, axis=0)  # Shape: (n_subjects, n_timepoints)
            averaged_results['coh_means'][key] = np.mean(stacked_data, axis=0)
            averaged_results['coh_stds'][key] = np.nanstd(stacked_data, axis=0) / np.sqrt(len(subject_data))  # SEM
    
    # Compute averages for imaginary coherence
    for key, subject_data in all_results['imcoh_results'].items():
        if len(subject_data) > 0:
            # Stack all subjects' data
            stacked_data = np.stack(subject_data, axis=0)  # Shape: (n_subjects, n_timepoints)
            averaged_results['imcoh_means'][key] = np.mean(stacked_data, axis=0)
            averaged_results['imcoh_stds'][key] = np.nanstd(stacked_data, axis=0) / np.sqrt(len(subject_data))  # SEM
    
    # Compute averages for dPLI
    for key, subject_data in all_results['dpli_results'].items():
        if len(subject_data) > 0:
            stacked_data = np.stack(subject_data, axis=0)
            averaged_results['dpli_means'][key] = np.mean(stacked_data, axis=0)
            averaged_results['dpli_stds'][key] = np.nanstd(stacked_data, axis=0) / np.sqrt(len(subject_data))
    
    return averaged_results

def plot_connectivity_results(averaged_results, save_path=None):
    """
    Plot contrast analysis with individual subplots for each connectivity pair
    
    Parameters:
    -----------
    averaged_results : dict
        Results from compute_averages
    save_path : str, optional
        Path to save the plot
    """
    
    print("Creating individual subplot contrast analysis...")
    
    time_vector = averaged_results['time_vector']
    
    # Define connectivity pairs for contrast analysis
    contrast_pairs = {
        'lV→lF': ('left_lV2lF', 'right_lV2lF'),
        'lV→rF': ('left_lV2rF', 'right_lV2rF'),
        'rV→lF': ('left_rV2lF', 'right_rV2lF'),
        'rV→rF': ('left_rV2rF', 'right_rV2rF'),
        'V→lF': ('left_V2lF', 'right_V2lF'),
        'V→rF': ('left_V2rF', 'right_V2rF')
    }
    
    # Create figure with 3 rows and 2 columns for 6 connectivity pairs
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    fig.suptitle(f'Imaginary Coherence Contrast Analysis: (Left - Right) / All Targets (n={averaged_results["n_subjects"]} subjects)', fontsize=16)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot each connectivity pair in its own subplot
    for i, (pair_name, (left_key, right_key)) in enumerate(contrast_pairs.items()):
        ax = axes_flat[i]
        
        # Imaginary coherence contrast
        imcoh_left_key = f'{left_key}_imcoh'
        imcoh_right_key = f'{right_key}_imcoh'
        # Extract connectivity pair from left_key (e.g., 'left_lV2lF' -> 'lV2lF')
        connectivity_pair = left_key.replace('left_', '').replace('right_', '')
        imcoh_all_key = f'all_{connectivity_pair}_imcoh'
        
        if (imcoh_left_key in averaged_results['imcoh_means'] and 
            imcoh_right_key in averaged_results['imcoh_means'] and 
            imcoh_all_key in averaged_results['imcoh_means']):
            
            left_vals = averaged_results['imcoh_means'][imcoh_left_key]
            right_vals = averaged_results['imcoh_means'][imcoh_right_key]
            all_vals = averaged_results['imcoh_means'][imcoh_all_key]
            left_sem = averaged_results['imcoh_stds'][imcoh_left_key]
            right_sem = averaged_results['imcoh_stds'][imcoh_right_key]
            all_sem = averaged_results['imcoh_stds'][imcoh_all_key]
            
            # Compute contrast: (left - right) / all
            contrast = (left_vals - right_vals) / (all_vals + 1e-10)  # Add small value to avoid division by zero
            
            # Compute SEM for contrast using error propagation
            # For f = (a-b)/c, the error is approximately:
            # df = sqrt((1/c * da)^2 + (-1/c * db)^2 + (-(a-b)/c^2 * dc)^2)
            contrast_sem = np.sqrt((1/(all_vals + 1e-10) * left_sem)**2 + 
                                  (-1/(all_vals + 1e-10) * right_sem)**2 + 
                                  (-(left_vals - right_vals)/(all_vals + 1e-10)**2 * all_sem)**2)
            
            ax.plot(time_vector, contrast, color='red', linewidth=2.5)
            ax.fill_between(time_vector, contrast - contrast_sem, contrast + contrast_sem, 
                           color='red', alpha=0.2)
        
        # Add reference lines and formatting
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=1.7, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Relative Imaginary Coherence')
        ax.set_title(f'{pair_name}')
        ax.set_ylim(-0.22, 0.22)
    
    plt.tight_layout()
    
    if save_path:
        # Save the individual subplot contrast analysis
        base_path = save_path.replace('.png', '')
        contrast_path = f"{base_path}_contrast_individual.png"
        fig.savefig(contrast_path, dpi=300, bbox_inches='tight')
        print(f"Individual subplot contrast analysis saved to {contrast_path}")
    
    plt.show()

def plot_simple_connectivity(averaged_results, bidsRoot, voxRes):
    """
    Plot simple connectivity results for left targets only
    
    Parameters:
    -----------
    averaged_results : dict
        Results from compute_averages
    save_path : str, optional
        Path to save the plot
    """
    
    print("Creating simple connectivity plots for left targets...")
    
    time_vector = averaged_results['time_vector']
    
    # Define connectivity pairs for both left and right targets
    connectivity_pairs = {
        'Left Visual → Left Frontal': ('left_lV2lF_imcoh', 'right_lV2lF_imcoh'),
        'Left Visual → Right Frontal': ('left_lV2rF_imcoh', 'right_lV2rF_imcoh'), 
        'Right Visual → Left Frontal': ('left_rV2lF_imcoh', 'right_rV2lF_imcoh'),
        'Right Visual → Right Frontal': ('left_rV2rF_imcoh', 'right_rV2rF_imcoh'),
        'Visual → Left Frontal': ('left_V2lF_imcoh', 'right_V2lF_imcoh'),
        'Visual → Right Frontal': ('left_V2rF_imcoh', 'right_V2rF_imcoh')
    }
    
    # Create figure with 2 rows and 3 columns for 6 connectivity pairs
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Imaginary Coherence: Left vs Right Targets (n={averaged_results["n_subjects"]} subjects)', fontsize=16)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Plot each connectivity pair in its own subplot
    for i, (pair_name, (left_key, right_key)) in enumerate(connectivity_pairs.items()):
        ax = axes_flat[i]
        
        # Plot left targets
        if left_key in averaged_results['imcoh_means']:
            left_mean = averaged_results['imcoh_means'][left_key]
            left_sem = averaged_results['imcoh_stds'][left_key]
            
            ax.plot(time_vector, left_mean, color='blue', linewidth=2.5, label='Left Targets')
            ax.fill_between(time_vector, left_mean - left_sem, left_mean + left_sem, 
                           color='blue', alpha=0.2)
        
        # Plot right targets
        if right_key in averaged_results['imcoh_means']:
            right_mean = averaged_results['imcoh_means'][right_key]
            right_sem = averaged_results['imcoh_stds'][right_key]
            
            ax.plot(time_vector, right_mean, color='red', linewidth=2.5, label='Right Targets')
            ax.fill_between(time_vector, right_mean - right_sem, right_mean + right_sem, 
                           color='red', alpha=0.2)
        
        # Add reference lines and formatting
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        ax.axhline(y=1, color='gray', linestyle='-', alpha=0.7, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=1.7, color='green', linestyle='--', alpha=0.7, linewidth=1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Imaginary Coherence')
        ax.set_title(f'{pair_name}')
        # ax.set_ylim(0.8, 1.2)
        ax.set_ylim(-0.2, 0.2)
        ax.set_xlim(-0.5, 1.7)
        ax.grid(False)
        ax.legend()
    
    plt.tight_layout()
    
    # Save as both SVG and PNG
    output_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
    os.makedirs(output_dir, exist_ok=True)
    
    save_path_svg = os.path.join(output_dir, f'connectivity_simple_{voxRes}.svg')
    save_path_png = os.path.join(output_dir, f'connectivity_simple_{voxRes}.png')
    
    fig.savefig(save_path_svg, format='svg', bbox_inches='tight')
    fig.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Figures saved to {output_dir} (SVG/PNG)")
    
    plt.show()

def plot_dpli_connectivity(averaged_results, bidsRoot, voxRes):
    """Plot directed Phase Lag Index results"""
    print("Creating dPLI directionality plots...")
    time_vector = averaged_results['time_vector']
    
    connectivity_pairs = {
        'Left Visual → Left Frontal': ('left_lV2lF_dpli', 'right_lV2lF_dpli'),
        'Left Visual → Right Frontal': ('left_lV2rF_dpli', 'right_rV2rF_dpli'), 
        'Right Visual → Left Frontal': ('left_rV2lF_dpli', 'right_rV2lF_dpli'),
        'Right Visual → Right Frontal': ('left_rV2rF_dpli', 'right_rV2rF_dpli'),
        'Visual → Left Frontal': ('left_V2lF_dpli', 'right_V2lF_dpli'),
        'Visual → Right Frontal': ('left_V2rF_dpli', 'right_V2rF_dpli')
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Directed Phase Lag Index: Left vs Right Targets (n={averaged_results["n_subjects"]} subjects)', fontsize=16)
    axes_flat = axes.flatten()
    
    for i, (pair_name, (left_key, right_key)) in enumerate(connectivity_pairs.items()):
        ax = axes_flat[i]
        
        # Plot left targets (Blue)
        if left_key in averaged_results['dpli_means']:
            l_mean, l_sem = averaged_results['dpli_means'][left_key], averaged_results['dpli_stds'][left_key]
            ax.plot(time_vector, l_mean, color='blue', linewidth=2.5, label='Left Targets')
            ax.fill_between(time_vector, l_mean - l_sem, l_mean + l_sem, color='blue', alpha=0.2)
        
        # Plot right targets (Red)
        if right_key in averaged_results['dpli_means']:
            r_mean, r_sem = averaged_results['dpli_means'][right_key], averaged_results['dpli_stds'][right_key]
            ax.plot(time_vector, r_mean, color='red', linewidth=2.5, label='Right Targets')
            ax.fill_between(time_vector, r_mean - r_sem, r_mean + r_sem, color='red', alpha=0.2)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7) # Cue
        ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7) # Delay
        ax.axvline(x=1.7, color='green', linestyle='--', alpha=0.7) # Response
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('$\Delta$dPLI (Baseline Subtracted)')
        ax.set_title(f'{pair_name}')
        ax.set_xlim(-0.5, 1.7)
        ax.legend()
    
    plt.tight_layout()
    output_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
    os.makedirs(output_dir, exist_ok=True)
    
    save_path_svg = os.path.join(output_dir, f'connectivity_dpli_{voxRes}.svg')
    save_path_png = os.path.join(output_dir, f'connectivity_dpli_{voxRes}.png')
    
    fig.savefig(save_path_svg, format='svg', bbox_inches='tight')
    fig.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    
    print(f"dPLI Figures saved to {output_dir} (SVG/PNG)")
    plt.show()

def plot_wpli_connectivity(all_results, voxRes, bidsRoot):
    """Plots Grand Average wPLI results"""
    if not all_results['wpli_results']:
        print("No wPLI results found to plot.")
        return
        
    print("Creating wPLI magnitude plots...")
    averaged_results = {
        'wpli_means': {},
        'wpli_sems': {}
    }
    
    for key, values in all_results['wpli_results'].items():
        data_stack = np.vstack(values)
        averaged_results['wpli_means'][key] = np.nanmean(data_stack, axis=0)
        averaged_results['wpli_sems'][key] = np.nanstd(data_stack, axis=0) / np.sqrt(len(values))
    
    time_vector = np.linspace(-0.6, 1.9, 2048)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
    fig.suptitle(f'8mm Beta Connectivity: Grand Average wPLI (n={len(all_results["loaded_subjects"])})')
    
    pairs = [
        ('all_lV2lF_wpli', 'all_lV2rF_wpli', 'all_V2lF_wpli'),
        ('all_rV2lF_wpli', 'all_rV2rF_wpli', 'all_V2rF_wpli')
    ]
    
    titles = [
        ['Left Vis -> Left Frontal', 'Left Vis -> Right Frontal', 'Combined Vis -> Left Frontal'],
        ['Right Vis -> Left Frontal', 'Right Vis -> Right Frontal', 'Combined Vis -> Right Frontal']
    ]
    
    for row in range(2):
        for col in range(3):
            key = pairs[row][col]
            ax = axes[row, col]
            
            mean = averaged_results['wpli_means'][key]
            sem = averaged_results['wpli_sems'][key]
            
            ax.plot(time_vector, mean, color='purple', linewidth=2)
            ax.fill_between(time_vector, mean - sem, mean + sem, color='purple', alpha=0.2)
            
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.7)
            ax.axvline(x=1.7, color='green', linestyle='--', alpha=0.7)
            
            ax.set_title(titles[row][col])
            ax.set_xlim(-0.5, 1.8)
            if col == 0: ax.set_ylabel('$\Delta$wPLI (Ratio Change)')
            if row == 1: ax.set_xlabel('Time (s)')

    plt.tight_layout()
    output_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
    os.makedirs(output_dir, exist_ok=True)
    
    save_path_svg = os.path.join(output_dir, f'connectivity_wpli_{voxRes}.svg')
    save_path_png = os.path.join(output_dir, f'connectivity_wpli_{voxRes}.png')
    
    fig.savefig(save_path_svg, format='svg', bbox_inches='tight')
    fig.savefig(save_path_png, format='png', dpi=300, bbox_inches='tight')
    
    print(f"wPLI Figures saved to {output_dir} (SVG/PNG)")
    plt.show()

def main():
    """Main function"""
    
    # Define parameters
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    taskName = 'mgs'
    voxRes = '8mm'
    
    # Set bidsRoot based on hostname/filesystem
    import socket
    hostname = socket.gethostname()
    if hostname == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    elif hostname == 'vader':
        bidsRoot = '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        # Check if direct /d/DATD works first
        if os.path.exists('/d/DATD/datd/MEG_MGS/MEG_BIDS'):
            bidsRoot = '/d/DATD/datd/MEG_MGS/MEG_BIDS'
        else:
            bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    print("="*60)
    print("CONNECTIVITY RESULTS ANALYSIS")
    print("="*60)
    print(f"Subjects: {subjects}")
    print(f"Task: {taskName}")
    print(f"Voxel resolution: {voxRes}")
    print(f"BIDS root: {bidsRoot}")
    print(f"Checking Path Exists (sub-01 test): {os.path.exists(os.path.join(bidsRoot, 'derivatives', 'sub-01', 'sourceRecon', 'connectivity_10mm', 'sub-01_task-mgs_seededConnectivity_10mm.pkl'))}")
    print("="*60)
    
    # Load results
    all_results = load_connectivity_results(bidsRoot, subjects, taskName, voxRes)
    
    if len(all_results['loaded_subjects']) == 0:
        print("No connectivity results found!")
        return
    
    # Compute averages
    averaged_results = compute_averages(all_results)
    
    # Create output directory for plots
    output_dir = os.path.join(bidsRoot, 'derivatives', 'connectivity_plots')
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot results
    # plot_save_path = os.path.join(output_dir, f'connectivity_results_{voxRes}.png')
    
    # Create both types of plots
    plot_simple_connectivity(averaged_results, bidsRoot, voxRes)
    if 'dpli_means' in averaged_results and len(averaged_results['dpli_means']) > 0:
        plot_dpli_connectivity(averaged_results, bidsRoot, voxRes)
    
    print("Analysis completed!")

if __name__ == '__main__':
    main()
