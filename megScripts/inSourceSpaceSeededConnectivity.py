import os, h5py, socket, gc
import numpy as np
from shutil import copyfile
import pickle
from scipy.io import loadmat
import time
import smtplib

NOTIFY_EMAIL = 'mrugank.dake@nyu.edu'

def send_completion_email(subjID, voxRes, connectivityType, success=True, error_msg=None):
    try:
        hostname = socket.gethostname()
        if success:
            subject = f'[SeededConnectivity] sub-{subjID:02d} {voxRes} {connectivityType} DONE on {hostname}'
            body = f'Seeded Connectivity analysis complete!\n\nSubject: {subjID:02d}\nResolution: {voxRes}\nMetric: {connectivityType}'
        else:
            subject = f'[SeededConnectivity] sub-{subjID:02d} {voxRes} {connectivityType} FAILED on {hostname}'
            body = f'Script failed with error:\n{error_msg}'

        msg = f'Subject: {subject}\n\n{body}'
        with smtplib.SMTP('localhost') as s:
            s.sendmail(NOTIFY_EMAIL, NOTIFY_EMAIL, msg)
        print(f'  Notification email sent to {NOTIFY_EMAIL}')
    except Exception as e:
        print(f'  (Email notification skipped: {e})')

# Make sure to run conda activate megAnalyses before running this script

def load_source_space_data(subjID, bidsRoot, taskName, voxRes, targetLoc, freqBand):
    """Load and concatenate source space data for all targets"""
    subName = 'sub-%02d' % subjID
    print(f'Loading source space data for {subName}')
    
    # File paths
    derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName)
    sourceReconRoot = os.path.join(derivativesRoot, 'sourceRecon')
    freqSpaceRoot = os.path.join(sourceReconRoot, 'freqSpace')
    freqSpace_fpath = os.path.join(freqSpaceRoot, f'{subName}_task-{taskName}_complex{freqBand}_allTargets_{voxRes[:-2]}.mat')
    
    # Load data with temporary copy approach
    if socket.gethostname() == 'zod':
        freqSpaceTempPath = os.path.join('/Users/mrugank/Desktop', f'{subName}_task-{taskName}_complex{freqBand}_allTargets_{voxRes[:-2]}.mat')
        copyfile(freqSpace_fpath, freqSpaceTempPath)
        freqSpace_data = h5py.File(freqSpaceTempPath, 'r')
        os.remove(freqSpaceTempPath)
    else:
        freqSpace_data = h5py.File(freqSpace_fpath, 'r')
    
    # Get sourceDataByTarget
    source_data = np.array(freqSpace_data['sourceDataByTarget'])
    
    # Extract all trials from all targets
    all_trials = []
    
    if targetLoc == 'left':
        target_indices = [4, 5, 6, 7, 8]
    elif targetLoc == 'right':
        target_indices = [1, 2, 3, 9, 10]

    for target_idx in target_indices:
        target_data = source_data[0, target_idx - 1]
        target_group = freqSpace_data[target_data]
        trial_dataset = target_group['trial']
        
        # Extract time vector from first target
        if target_idx == target_indices[0]:
            time_data = target_group['time']
            # Get the actual time values - resolve the first time reference
            first_time_ref = time_data[0, 0]
            time_vector = np.array(freqSpace_data[first_time_ref])
        print(f"Target {target_idx} trials: {trial_dataset.shape}")
        for trial_idx in range(trial_dataset.shape[0]):
            trial_ref = trial_dataset[trial_idx, 0] 
            trial_data = freqSpace_data[trial_ref]
            trial_array = np.array(trial_data)
            all_trials.append(trial_array)
    
    # Stack all trials (trials × time × sources)
    data_matrix = np.stack(all_trials, axis=0)
    
    # Keep complex data for connectivity analysis
    data_matrix = data_matrix['real'] + 1j * data_matrix['imag']
    
    print(f"Data loaded: {data_matrix.shape} (trials × time × sources)")
    print(f"Data type: {data_matrix.dtype}")
    print(f"Time range: {time_vector[0,0]:.2f}s to {time_vector[-1,0]:.2f}s")

    
    return data_matrix, time_vector

def compute_connectivity_measures(seed_indices, data_matrix, time_vector, connectivityType, freqBand, batch_size=500):
    """
    Compute connectivity measures between seed sources and all other sources (whole brain) for each time point
    Processes sources in batches to reduce memory usage.
    Returns: coherence and imaginary coherence time series of shape (n_sources, n_timepoints)
             Each row represents connectivity from seed sources to that target source, averaged across seeds
    """
    
    
    _, n_timepoints, n_sources = data_matrix.shape
    n_seeds = len(seed_indices)
    
    print(f"Computing connectivity measures for {n_seeds} seed sources to {n_sources} target sources (whole brain)...")
    print(f"Data shape: {data_matrix.shape}")
    print(f"Computing for each time point with ±333ms window across entire trial")
    print(f"Processing sources in batches of {batch_size}")
    
    # Initialize connectivity time series: (n_sources, n_timepoints)
    connectivity_timeseries = np.empty((n_sources, n_timepoints))
    
    # Compute sampling frequency
    sfreq = 1 / np.mean(np.diff(time_vector.flatten()))

    if freqBand == 'theta':
        window_size = 1.25 # 1250ms window in seconds
    elif freqBand == 'alpha':
        window_size = 0.77 # 770ms window in seconds
    elif freqBand == 'beta':
        window_size = 0.40 # 400ms window in seconds
    elif freqBand == 'lowgamma':
        window_size = 0.20 # 200ms window in seconds

    window_half_samples = int((window_size / 2) * sfreq)  # half window in samples (for centered window)
    
    print(f"Sampling frequency: {sfreq:.1f} Hz")
    print(f"Window size: {window_size} seconds")
    print(f"Window half samples: {window_half_samples} (total window: {2 * window_half_samples} samples)")
    
    # Calculate number of batches
    n_batches = int(np.ceil(n_sources / batch_size))
    
    # Process each time point
    for t_idx in range(n_timepoints):
        if t_idx % 10 == 0:  # Progress update
            print(f"Processing time point {t_idx+1}/{n_timepoints}")
        
        # Define time window around current time point (centered window of window_size seconds)
        start_idx = max(0, t_idx - window_half_samples)
        end_idx = min(n_timepoints, t_idx + window_half_samples + 1)
        
        # Extract data for this time window
        time_window_data = data_matrix[:, start_idx:end_idx, :]
        
        # Get seed data for this time window (same for all batches)
        seed_data = time_window_data[:, :, seed_indices].transpose(2, 0, 1)  # (n_seeds, n_trials, window_timepoints)
        seed_power = np.mean(seed_data * np.conj(seed_data), axis=1)  # (n_seeds, window_timepoints)
        
        # Process sources in batches
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, n_sources)
            batch_indices = np.arange(batch_start, batch_end)
            
            # Get target data for this batch
            target_data_batch = time_window_data[:, :, batch_indices]  # (n_trials, window_timepoints, batch_size)
            
            # Compute cross-spectral density for all seed-target pairs in this batch
            cross_spectrum_batch = np.mean(seed_data[:, :, :, np.newaxis] * np.conj(target_data_batch[np.newaxis, :, :, :]), axis=1)
            # Shape: (n_seeds, window_timepoints, batch_size)
            
            # Compute power spectra for this batch
            target_power_batch = np.mean(target_data_batch * np.conj(target_data_batch), axis=0)  # (window_timepoints, batch_size)
            
            if connectivityType == 'coh':
                connectivity_mag_batch = np.abs(cross_spectrum_batch)**2 / (seed_power[:, :, np.newaxis] * target_power_batch[np.newaxis, :, :] + 1e-10)
                # Shape: (n_seeds, window_timepoints, batch_size)
            elif connectivityType == 'imcoh':
                connectivity_mag_batch = np.abs(np.imag(cross_spectrum_batch)) / np.sqrt(seed_power[:, :, np.newaxis] * target_power_batch[np.newaxis, :, :] + 1e-10)
                # Shape: (n_seeds, window_timepoints, batch_size)
            elif connectivityType == 'dpli':
                # Directed Phase Lag Index: measures directional flow (Seed leads Target > 0.5)
                # Requires calculating Heaviside phase angles per trial before averaging
                cross_spectrum_trials = seed_data[:, :, :, np.newaxis] * np.conj(target_data_batch[np.newaxis, :, :, :])
                connectivity_mag_batch = np.mean(np.heaviside(np.imag(cross_spectrum_trials), 0.5), axis=1)
                # Shape: (n_seeds, window_timepoints, batch_size)
            # Compute connectivity magnitude for all pairs in this batch
            connectivity_timeseries[batch_indices, t_idx] = np.mean(connectivity_mag_batch, axis=(0, 1))  # Result: (batch_size,)
    
    return connectivity_timeseries

def main(subjID, voxRes, seedROI, targetLoc, connectivityType, freqBand):
    """Main function for source space connectivity analysis"""
    # Take into account default voxRes if not provided
    if voxRes is None:
        voxRes = '10mm'
    if seedROI is None:
        seedROI = 'left_visual'
    if targetLoc is None:
        targetLoc = 'left'
    if connectivityType is None:
        connectivityType = 'coh'
    if freqBand is None:
        freqBand = 'theta'
    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    taskName = 'mgs'

    outputDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', 'connectivity')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    outputFile = os.path.join(outputDir, f'sub-{subjID:02d}_task-{taskName}_seededConnectivity_{voxRes}_{seedROI}_{targetLoc}_{connectivityType}_{freqBand}.pkl')

    if not os.path.exists(outputFile):
        print('Loading source space data for connectivity analysis...')
        data_matrix, time_vector = load_source_space_data(subjID, bidsRoot, taskName, voxRes, targetLoc, freqBand)

        # Load atlas data
        atlas_fpath = os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat')
        atlas_data = loadmat(atlas_fpath)

        # Define ROI indices
        left_visual_points = np.array(atlas_data['left_visual_points']).flatten()
        right_visual_points = np.array(atlas_data['right_visual_points']).flatten()
        left_frontal_points = np.array(atlas_data['left_frontal_points']).flatten()
        right_frontal_points = np.array(atlas_data['right_frontal_points']).flatten()
        left_visual_indices = np.where(left_visual_points == 1)[0]
        right_visual_indices = np.where(right_visual_points == 1)[0]
        left_frontal_indices = np.where(left_frontal_points == 1)[0]
        right_frontal_indices = np.where(right_frontal_points == 1)[0]

        print(f"Data shape: {data_matrix.shape}")
        print(f"Time points: {data_matrix.shape[1]}")
        print(f"Number of trials: {data_matrix.shape[0]}")
        
        # Compute connectivity measures for left targets (visual seeds to whole brain)
        start_time = time.time()
        print(f"Computing connectivity measures for {targetLoc} targets...")
        if seedROI == 'left_visual':
            connectivity_measure = compute_connectivity_measures(left_visual_indices, data_matrix, time_vector, connectivityType, freqBand)
        elif seedROI == 'right_visual':
            connectivity_measure = compute_connectivity_measures(right_visual_indices, data_matrix, time_vector, connectivityType, freqBand)
        elif seedROI == 'left_frontal':
            connectivity_measure = compute_connectivity_measures(left_frontal_indices, data_matrix, time_vector, connectivityType, freqBand)
        elif seedROI == 'right_frontal':
            connectivity_measure = compute_connectivity_measures(right_frontal_indices, data_matrix, time_vector, connectivityType, freqBand)
        else:
            print("Invalid seed ROI")
        end_time = time.time()
        print(f"Connectivity analysis completed in {end_time - start_time:.2f} seconds")

        # Save connectivity measure
        with open(outputFile, 'wb') as f:
            pickle.dump(connectivity_measure, f)
        print(f"Connectivity measure saved to {outputFile}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inSourceSpaceSeededConnectivity.py <subjID> [voxRes] [seedROI] [targetLoc] [coh|imcoh|dpli]")
        print("Example: python inSourceSpaceSeededConnectivity.py 1 10mm left_visual left dpli")
        sys.exit(1)
    
    subjID = int(sys.argv[1])
    voxRes = sys.argv[2] if len(sys.argv) > 2 else '10mm'
    seedROI = sys.argv[3] if len(sys.argv) > 3 else 'left_visual'
    targetLoc = sys.argv[4] if len(sys.argv) > 4 else 'left'
    connectivityType = sys.argv[5] if len(sys.argv) > 5 else 'coh'
    freqBand = sys.argv[6] if len(sys.argv) > 6 else 'theta'
    
    print(f"Running source space seeded connectivity for subject {subjID} with voxel resolution {voxRes}")
    print(f"Seed ROI: {seedROI}, Target location: {targetLoc}, Connectivity type: {connectivityType}")
    main(subjID, voxRes, seedROI, targetLoc, connectivityType, freqBand)
