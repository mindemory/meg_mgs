import os, h5py, socket, gc
import numpy as np
from shutil import copyfile
import pickle
from scipy.io import loadmat
import time

# Make sure to run conda activate megAnalyses before running this script

def load_source_space_data(subjID, bidsRoot, taskName, voxRes):
    """Load and concatenate source space data for all targets"""
    subName = 'sub-%02d' % subjID
    print(f'Loading source space data for {subName}')
    
    # File paths
    derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName)
    sourceReconRoot = os.path.join(derivativesRoot, 'sourceRecon')
    freqSpaceRoot = os.path.join(sourceReconRoot, 'freqSpace')
    freqSpace_fpath = os.path.join(freqSpaceRoot, f'{subName}_task-{taskName}_complexbeta_allTargets_{voxRes[:-2]}.mat')
    
    # Load data with temporary copy approach
    if socket.gethostname() == 'zod':
        freqSpaceTempPath = os.path.join('/Users/mrugank/Desktop', f'{subName}_task-{taskName}_complexbeta_allTargets_{voxRes[:-2]}.mat')
        copyfile(freqSpace_fpath, freqSpaceTempPath)
        freqSpace_data = h5py.File(freqSpaceTempPath, 'r')
        os.remove(freqSpaceTempPath)
    else:
        freqSpace_data = h5py.File(freqSpace_fpath, 'r')
    
    # Get sourceDataByTarget
    source_data = np.array(freqSpace_data['sourceDataByTarget'])
    
    # Extract all trials from all targets
    all_trials = []
    target_labels = []
    
    for target_idx in range(10):
        target_data = source_data[0, target_idx]
        target_group = freqSpace_data[target_data]
        trial_dataset = target_group['trial']
        
        # Extract time vector from first target
        if target_idx == 0:
            time_data = target_group['time']
            # Get the actual time values - resolve the first time reference
            first_time_ref = time_data[0, 0]
            time_vector = np.array(freqSpace_data[first_time_ref])
        print(f"Target {target_idx + 1} trials: {trial_dataset.shape}")
        for trial_idx in range(trial_dataset.shape[0]):
            trial_ref = trial_dataset[trial_idx, 0] 
            trial_data = freqSpace_data[trial_ref]
            trial_array = np.array(trial_data)
            all_trials.append(trial_array)
            target_labels.append(target_idx + 1)
    
    # Stack all trials (trials × time × sources)
    data_matrix = np.stack(all_trials, axis=0)
    
    # Keep complex data for connectivity analysis
    data_matrix = data_matrix['real'] + 1j * data_matrix['imag']

    # Remove center of head bias by subtracting mean across all sources at each time point
    # This removes the common mode signal without affecting phase relationships
    # mean_across_sources = np.mean(data_matrix, axis=0, keepdims=True)  # Shape: (time, 1, sources0
    # data_matrix = data_matrix - mean_across_sources 
    
    print(f"Data loaded: {data_matrix.shape} (trials × time × sources)")
    print(f"Data type: {data_matrix.dtype}")
    print(f"Time range: {time_vector[0,0]:.2f}s to {time_vector[-1,0]:.2f}s")

    target_labels = np.array(target_labels)
    
    return data_matrix, target_labels, time_vector

def compute_connectivity_measures(seed_indices, target_indices, data_matrix, time_vector):
    """
    Compute multiple connectivity measures between seed sources and target sources for each time point with 100ms window
    Returns: coherence, imaginary coherence, PLV, and PLI time series
    """
    
    
    n_trials, n_timepoints, n_sources = data_matrix.shape
    n_seeds = len(seed_indices)
    n_targets = len(target_indices)
    
    print(f"Computing connectivity measures for {n_seeds} seed sources and {n_targets} target sources...")
    print(f"Data shape: {data_matrix.shape}")
    print(f"Computing for each time point with ±100ms window across entire trial")
    
    # Initialize connectivity time series: (n_timepoints,)
    coherence_timeseries = np.empty(n_timepoints)
    imcoherence_timeseries = np.empty(n_timepoints)
    plv_timeseries = np.empty(n_timepoints)
    pli_timeseries = np.empty(n_timepoints)
    
    # Compute sampling frequency
    sfreq = 1 / np.mean(np.diff(time_vector.flatten()))
    window_samples = int(0.1 * sfreq)  # 100ms window in samples
    
    print(f"Sampling frequency: {sfreq:.1f} Hz")
    print(f"Window samples: {window_samples}")
    
    # Process each time point
    for t_idx in range(n_timepoints):
        if t_idx % 10 == 0:  # Progress update
            print(f"Processing time point {t_idx+1}/{n_timepoints}")
        
        # Define time window around current time point
        start_idx = max(0, t_idx - window_samples)
        end_idx = min(n_timepoints, t_idx + window_samples + 1)
        
        # Extract data for this time window
        time_window_data = data_matrix[:, start_idx:end_idx, :]
        # window_n_timepoints = time_window_data.shape[1]
        
        # Get seed and target data for this time window
        seed_data = time_window_data[:, :, seed_indices].transpose(2, 0, 1)  # (n_seeds, n_trials, window_timepoints)
        target_data = time_window_data[:, :, target_indices]  # (n_trials, window_timepoints, n_targets)
        
        # Compute cross-spectral density for all seed-target pairs
        cross_spectrum = np.mean(seed_data[:, :, :, np.newaxis] * np.conj(target_data[np.newaxis, :, :, :]), axis=1)
        # Shape: (n_seeds, window_timepoints, n_targets)
        
        # Compute power spectra
        seed_power = np.mean(seed_data * np.conj(seed_data), axis=1)  # (n_seeds, window_timepoints)
        target_power = np.mean(target_data * np.conj(target_data), axis=0)  # (window_timepoints, n_targets)
        
        # Compute coherence magnitude for all pairs
        coherence_mag = np.abs(cross_spectrum)**2 / (seed_power[:, :, np.newaxis] * target_power[np.newaxis, :, :] + 1e-10)
        
        # Average coherence across time window and all seed-target pairs
        coherence_timeseries[t_idx] = np.mean(coherence_mag)  # Average across all seeds, targets, and time window
        
        # Imaginary coherence calculation
        normalized_cross_spectrum = cross_spectrum / np.sqrt(seed_power[:, :, np.newaxis] * target_power[np.newaxis, :, :] + 1e-10)
        imcoherence_timeseries[t_idx] = np.mean(np.abs(np.imag(normalized_cross_spectrum)))  # Average across all seeds, targets, and time window
        
        # Phase Locking Value (PLV) calculation
        # Extract phases from complex data
        # seed_phases = np.angle(seed_data)  # (n_seeds, n_trials, window_timepoints)
        # target_phases = np.angle(target_data)  # (n_trials, window_timepoints, n_targets)
        
        # Compute phase differences for all seed-target pairs
        # phase_diff = seed_phases[:, :, :, np.newaxis] - target_phases[np.newaxis, :, :, :]  # (n_seeds, n_trials, window_timepoints, n_targets)
        
        # Compute PLV as |mean(exp(i*phase_diff))|
        # plv_complex = np.mean(np.exp(1j * phase_diff), axis=1)  # Average across trials: (n_seeds, window_timepoints, n_targets)
        # plv_mag = np.abs(plv_complex)  # (n_seeds, window_timepoints, n_targets)
        
        # # Average PLV across time window and all seed-target pairs
        # plv_timeseries[t_idx] = np.mean(plv_mag)
        
        # # Phase Lag Index (PLI) calculation
        # # PLI = |mean(sign(imag(exp(i*phase_diff))))|
        # pli_sign = np.sign(np.imag(np.exp(1j * phase_diff)))  # (n_seeds, n_trials, window_timepoints, n_targets)
        # pli_mean = np.mean(pli_sign, axis=1)  # Average across trials: (n_seeds, window_timepoints, n_targets)
        # pli_abs = np.abs(pli_mean)  # (n_seeds, window_timepoints, n_targets)
        
        # # Average PLI across time window and all seed-target pairs
        # pli_timeseries[t_idx] = np.mean(pli_abs)
    
    return coherence_timeseries, imcoherence_timeseries, plv_timeseries, pli_timeseries

def main(subjID, voxRes):
    """Main function for source space connectivity analysis"""
    # Take into account default voxRes if not provided
    if voxRes is None:
        voxRes = '10mm'

    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    taskName = 'mgs'

    outputDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', 'connectivity')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    outputFile = os.path.join(outputDir, f'sub-{subjID:02d}_task-{taskName}_connectivity_{voxRes}.pkl')

    if not os.path.exists(outputFile):

        data_matrix, target_labels, time_vector = load_source_space_data(subjID, bidsRoot, taskName, voxRes)


        # Load atlas data
        atlas_fpath = os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat')
        atlas_data = loadmat(atlas_fpath)

        # Define ROI indices
        visual_points = np.array(atlas_data['visual_points']).flatten()
        left_visual_points = np.array(atlas_data['left_visual_points']).flatten()
        right_visual_points = np.array(atlas_data['right_visual_points']).flatten()
        # left_parietal_points = np.array(atlas_data['left_parietal_points']).flatten()
        # right_parietal_points = np.array(atlas_data['right_parietal_points']).flatten()
        left_frontal_points = np.array(atlas_data['left_frontal_points']).flatten()
        right_frontal_points = np.array(atlas_data['right_frontal_points']).flatten()
        visual_indices = np.where(visual_points == 1)[0]
        left_visual_indices = np.where(left_visual_points == 1)[0]
        right_visual_indices = np.where(right_visual_points == 1)[0]
        # left_parietal_indices = np.where(left_parietal_points == 1)[0]
        # right_parietal_indices = np.where(right_parietal_points == 1)[0]
        left_frontal_indices = np.where(left_frontal_points == 1)[0]
        right_frontal_indices = np.where(right_frontal_points == 1)[0]

        print('Loading source space data for connectivity analysis...')

        # Filter data for left and right targets
        left_data = data_matrix[np.isin(target_labels, [4, 5, 6, 7, 8]), :, :]
        right_data = data_matrix[np.isin(target_labels, [1, 2, 3, 9, 10]), :, :]

        print(f"Data shape: {data_matrix.shape}")
        print(f"Time points: {data_matrix.shape[1]}")
        print(f"Number of trials: {data_matrix.shape[0]}")
        
        # Compute connectivity measures for left targets
        start_time = time.time()
        print("Computing connectivity measures for left targets...")
        left_lV2lF_coh, left_lV2lF_imcoh, left_lV2lF_plv, left_lV2lF_pli = compute_connectivity_measures(left_visual_indices, left_frontal_indices, left_data, time_vector)
        left_lV2rF_coh, left_lV2rF_imcoh, left_lV2rF_plv, left_lV2rF_pli = compute_connectivity_measures(left_visual_indices, right_frontal_indices, left_data, time_vector)
        left_rV2lF_coh, left_rV2lF_imcoh, left_rV2lF_plv, left_rV2lF_pli = compute_connectivity_measures(right_visual_indices, left_frontal_indices, left_data, time_vector)
        left_rV2rF_coh, left_rV2rF_imcoh, left_rV2rF_plv, left_rV2rF_pli = compute_connectivity_measures(right_visual_indices, right_frontal_indices, left_data, time_vector)
        left_V2lF_coh, left_V2lF_imcoh, left_V2lF_plv, left_V2lF_pli = compute_connectivity_measures(visual_indices, left_frontal_indices, left_data, time_vector)
        left_V2rF_coh, left_V2rF_imcoh, left_V2rF_plv, left_V2rF_pli = compute_connectivity_measures(visual_indices, right_frontal_indices, left_data, time_vector)

        # Compute connectivity measures for right targets
        print("Computing connectivity measures for right targets...")
        right_lV2lF_coh, right_lV2lF_imcoh, right_lV2lF_plv, right_lV2lF_pli = compute_connectivity_measures(left_visual_indices, left_frontal_indices, right_data, time_vector)
        right_lV2rF_coh, right_lV2rF_imcoh, right_lV2rF_plv, right_lV2rF_pli = compute_connectivity_measures(left_visual_indices, right_frontal_indices, right_data, time_vector)
        right_rV2lF_coh, right_rV2lF_imcoh, right_rV2lF_plv, right_rV2lF_pli = compute_connectivity_measures(right_visual_indices, left_frontal_indices, right_data, time_vector)
        right_rV2rF_coh, right_rV2rF_imcoh, right_rV2rF_plv, right_rV2rF_pli = compute_connectivity_measures(right_visual_indices, right_frontal_indices, right_data, time_vector)
        right_V2lF_coh, right_V2lF_imcoh, right_V2lF_plv, right_V2lF_pli = compute_connectivity_measures(visual_indices, left_frontal_indices, right_data, time_vector)
        right_V2rF_coh, right_V2rF_imcoh, right_V2rF_plv, right_V2rF_pli = compute_connectivity_measures(visual_indices, right_frontal_indices, right_data, time_vector)

        print("Computing connectivity measures for all targets...")
        all_lV2lF_coh, all_lV2lF_imcoh, all_lV2lF_plv, all_lV2lF_pli = compute_connectivity_measures(left_visual_indices, left_frontal_indices, data_matrix, time_vector)
        all_lV2rF_coh, all_lV2rF_imcoh, all_lV2rF_plv, all_lV2rF_pli = compute_connectivity_measures(left_visual_indices, right_frontal_indices, data_matrix, time_vector)
        all_rV2lF_coh, all_rV2lF_imcoh, all_rV2lF_plv, all_rV2lF_pli = compute_connectivity_measures(right_visual_indices, left_frontal_indices, data_matrix, time_vector)
        all_rV2rF_coh, all_rV2rF_imcoh, all_rV2rF_plv, all_rV2rF_pli = compute_connectivity_measures(right_visual_indices, right_frontal_indices, data_matrix, time_vector)
        all_V2lF_coh, all_V2lF_imcoh, all_V2lF_plv, all_V2lF_pli = compute_connectivity_measures(visual_indices, left_frontal_indices, data_matrix, time_vector)
        all_V2rF_coh, all_V2rF_imcoh, all_V2rF_plv, all_V2rF_pli = compute_connectivity_measures(visual_indices, right_frontal_indices, data_matrix, time_vector)
        
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        # Save the results
        results_coh = {
            'left_lV2lF_coh': left_lV2lF_coh,
            'left_lV2rF_coh': left_lV2rF_coh,
            'left_rV2lF_coh': left_rV2lF_coh,
            'left_rV2rF_coh': left_rV2rF_coh,
            'left_V2lF_coh': left_V2lF_coh,
            'left_V2rF_coh': left_V2rF_coh,
            'right_lV2lF_coh': right_lV2lF_coh,
            'right_lV2rF_coh': right_lV2rF_coh,
            'right_rV2lF_coh': right_rV2lF_coh,
            'right_rV2rF_coh': right_rV2rF_coh,
            'right_V2lF_coh': right_V2lF_coh,
            'right_V2rF_coh': right_V2rF_coh,
            'all_lV2lF_coh': all_lV2lF_coh,
            'all_lV2rF_coh': all_lV2rF_coh,
            'all_rV2lF_coh': all_rV2lF_coh,
            'all_rV2rF_coh': all_rV2rF_coh,
            'all_V2lF_coh': all_V2lF_coh,
            'all_V2rF_coh': all_V2rF_coh,
        }
        results_imcoh = {
            'left_lV2lF_imcoh': left_lV2lF_imcoh,
            'left_lV2rF_imcoh': left_lV2rF_imcoh,
            'left_rV2lF_imcoh': left_rV2lF_imcoh,
            'left_rV2rF_imcoh': left_rV2rF_imcoh,
            'left_V2lF_imcoh': left_V2lF_imcoh,
            'left_V2rF_imcoh': left_V2rF_imcoh,
            'right_lV2lF_imcoh': right_lV2lF_imcoh,
            'right_lV2rF_imcoh': right_lV2rF_imcoh,
            'right_rV2lF_imcoh': right_rV2lF_imcoh,
            'right_rV2rF_imcoh': right_rV2rF_imcoh,
            'right_V2lF_imcoh': right_V2lF_imcoh,
            'right_V2rF_imcoh': right_V2rF_imcoh,
            'all_lV2lF_imcoh': all_lV2lF_imcoh,
            'all_lV2rF_imcoh': all_lV2rF_imcoh,
            'all_rV2lF_imcoh': all_rV2lF_imcoh,
            'all_rV2rF_imcoh': all_rV2rF_imcoh,
            'all_V2lF_imcoh': all_V2lF_imcoh,
            'all_V2rF_imcoh': all_V2rF_imcoh,
        }
        results_plv = {
            'left_lV2lF_plv': left_lV2lF_plv,
            'left_lV2rF_plv': left_lV2rF_plv,
            'left_rV2lF_plv': left_rV2lF_plv,
            'left_rV2rF_plv': left_rV2rF_plv,
            'left_V2lF_plv': left_V2lF_plv,
            'left_V2rF_plv': left_V2rF_plv,
            'right_lV2lF_plv': right_lV2lF_plv,
            'right_lV2rF_plv': right_lV2rF_plv,
            'right_rV2lF_plv': right_rV2lF_plv,
            'right_rV2rF_plv': right_rV2rF_plv,
            'right_V2lF_plv': right_V2lF_plv,
            'right_V2rF_plv': right_V2rF_plv,
            'all_lV2lF_plv': all_lV2lF_plv,
            'all_lV2rF_plv': all_lV2rF_plv,
            'all_rV2lF_plv': all_rV2lF_plv,
            'all_rV2rF_plv': all_rV2rF_plv,
            'all_V2lF_plv': all_V2lF_plv,
            'all_V2rF_plv': all_V2rF_plv,
        }
        results_pli = {
            'left_lV2lF_pli': left_lV2lF_pli,
            'left_lV2rF_pli': left_lV2rF_pli,
            'left_rV2lF_pli': left_rV2lF_pli,
            'left_rV2rF_pli': left_rV2rF_pli,
            'left_V2lF_pli': left_V2lF_pli,
            'left_V2rF_pli': left_V2rF_pli,
            'right_lV2lF_pli': right_lV2lF_pli,
            'right_lV2rF_pli': right_lV2rF_pli,
            'right_rV2lF_pli': right_rV2lF_pli,
            'right_rV2rF_pli': right_rV2rF_pli,
            'right_V2lF_pli': right_V2lF_pli,
            'right_V2rF_pli': right_V2rF_pli,
            'all_lV2lF_pli': all_lV2lF_pli,
            'all_lV2rF_pli': all_lV2rF_pli,
            'all_rV2lF_pli': all_rV2lF_pli,
            'all_rV2rF_pli': all_rV2rF_pli,
            'all_V2lF_pli': all_V2lF_pli,
            'all_V2rF_pli': all_V2rF_pli,
        }
        
        with open(outputFile, 'wb') as f:
            pickle.dump(results_coh, f)
            pickle.dump(results_imcoh, f)
            pickle.dump(results_plv, f)
            pickle.dump(results_pli, f)
        print(f"Results saved to {outputFile}")

    
    print("Connectivity analysis completed!")
    print(f"Connectivity time series length: {len(left_lV2lF_coh)} time points")
    
    
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inSourceSpaceConnectivity.py <subjID> [voxRes]")
        print("Example: python inSourceSpaceConnectivity.py 1 10mm")
        sys.exit(1)
    
    subjID = int(sys.argv[1])
    voxRes = sys.argv[2] if len(sys.argv) > 2 else '10mm'
    
    print(f"Running source space connectivity for subject {subjID} with voxel resolution {voxRes}")
    main(subjID, voxRes)
