import os, h5py, socket, gc
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneOut
from scipy.io import loadmat, savemat
from scipy.stats import circmean, circstd
from multiprocessing import Pool, cpu_count
import functools


# Make sure to run conda activate megAnalyses before running this script

# Define frequency band ranges (in Hz)
FREQUENCY_BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (18.0, 25.0),
    'lowgamma': (25.0, 50.0)
}

def convert_xy_to_angle(x, y, ref_x=5, ref_y=0):
    """
    Convert (x, y) coordinates to angles in degrees.
    
    Parameters:
    -----------
    x, y : array-like
        x and y coordinates
    ref_x, ref_y : float
        Reference point that should correspond to 0 degrees (default: (5, 0))
    
    Returns:
    --------
    angles : array-like
        Angles in degrees, with (ref_x, ref_y) as 0° and counter-clockwise to 360°
    """
    # Calculate angle of reference point
    ref_angle = np.arctan2(ref_y, ref_x)
    
    # Calculate angles for all points
    angles_rad = np.arctan2(y, x)
    
    # Subtract reference angle to make reference point 0
    angles_rad = angles_rad - ref_angle
    
    # Convert to degrees
    angles_deg = np.degrees(angles_rad)
    
    # Normalize to 0-360 range
    angles_deg = (angles_deg + 360) % 360
    
    return angles_deg

def load_source_space_data(subjID, bidsRoot, taskName, voxRes, freq_band='beta'):
    """Load and concatenate source space data for all targets
    
    Parameters:
    -----------
    subjID : int
        Subject ID
    bidsRoot : str
        BIDS root directory
    taskName : str
        Task name
    voxRes : str
        Voxel resolution (e.g., '10mm')
    freq_band : str
        Frequency band name ('theta', 'alpha', 'beta', 'lowgamma')
        Default: 'beta'
    """
    subName = 'sub-%02d' % subjID
    print(f'Loading source space data for {subName}')
    
    # Validate frequency band
    if freq_band not in FREQUENCY_BANDS:
        raise ValueError(f"Invalid frequency band '{freq_band}'. Must be one of: {list(FREQUENCY_BANDS.keys())}")
    
    f_min, f_max = FREQUENCY_BANDS[freq_band]
    print(f'Using frequency band: {freq_band} ({f_min}-{f_max} Hz)')
    
    # File paths - new path structure
    derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName)
    sourceReconRoot = os.path.join(derivativesRoot, 'sourceRecon')
    
    # Extract surface resolution from voxRes (e.g., '10mm' -> 10)
    surface_resolution = int(voxRes[:-2])
    source_data_fpath = os.path.join(sourceReconRoot, f'{subName}_task-{taskName}_sourceSpaceData_{surface_resolution}.mat')
    
    # Load data with temporary copy approach
    if socket.gethostname() == 'zod':
        source_data_temp_path = os.path.join('/Users/mrugank/Desktop', f'{subName}_task-{taskName}_sourceSpaceData_{surface_resolution}.mat')
        copyfile(source_data_fpath, source_data_temp_path)
        source_data = h5py.File(source_data_temp_path, 'r')
        os.remove(source_data_temp_path)
    else:
        source_data = h5py.File(source_data_fpath, 'r')
    
    # Load sourcedataCombined structure
    sourcedata_group = source_data['sourcedataCombined']
    
    # Get time vector and target labels
    time_data = sourcedata_group['time']
    time_vector = np.array(source_data[time_data[0, 0]]).flatten()
    
    trialinfo = np.array(sourcedata_group['trialinfo']).T  # Transpose to (154, 5)
    target_labels = trialinfo[:, 1]  # Column 2 (target labels)
    
    # Load trial data
    trial_data = sourcedata_group['trial']
    all_trials = []
    
    for trial_idx in range(trial_data.shape[0]):
        trial_ref = trial_data[trial_idx, 0]
        trial_array = np.array(source_data[trial_ref])
        all_trials.append(trial_array)
    
    data_matrix = np.stack(all_trials, axis=0)  # Shape: (trials, time, sources)
    
    # Compute time-varying power using bandpass filter + Hilbert transform (much faster)
    print(f"Computing time-varying power in {f_min}-{f_max} Hz range ({freq_band} band) using bandpass filter...")
    from scipy import signal
    
    dt = np.mean(np.diff(time_vector))
    
    n_trials, n_timepoints, n_sources = data_matrix.shape
    power_matrix = np.zeros((n_trials, n_timepoints, n_sources))
    
    # Design bandpass filter
    nyquist = 1 / (2 * dt)
    low = f_min / nyquist
    high = f_max / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    
    print(f"Processing {n_trials} trials and {n_sources} sources...")
    
    # Process all trials and sources at once using vectorized operations
    for trial_idx in range(n_trials):
        if trial_idx % (n_trials // 10) == 0:
            print(f"Processing trial {trial_idx}/{n_trials}")
        
        # Get all sources for this trial at once: (time, sources)
        trial_data = data_matrix[trial_idx, :, :]  # Shape: (time, sources)
        
        # Apply bandpass filter to all sources at once
        filtered_data = signal.filtfilt(b, a, trial_data, axis=0)  # Filter along time axis
        
        # Compute instantaneous power using Hilbert transform
        analytic_signal = signal.hilbert(filtered_data, axis=0)
        instantaneous_power = np.abs(analytic_signal) ** 2
        
        # Store results
        power_matrix[trial_idx, :, :] = instantaneous_power
    
    data_matrix = power_matrix

    # Baseline correction
    mean_allTrials = data_matrix.mean(axis=0)
    data_matrix = data_matrix / mean_allTrials[np.newaxis, :, :] - 1

    # Downsample to 25ms resolution with 100ms averaging window (50ms before + 50ms after)
    target_dt = 0.025  # 25ms
    window_half_size = 0.050  # 50ms on each side
    downsample_factor = int(target_dt / dt)
    if downsample_factor < 1:
        downsample_factor = 1
    
    # Number of samples to include on each side of the center point
    half_window_samples = int(window_half_size / dt)
    
    n_trials, n_timepoints, n_sources = data_matrix.shape
    n_downsampled = n_timepoints // downsample_factor
    data_matrix_downsampled = np.zeros((n_trials, n_downsampled, n_sources))
    
    for i in range(n_downsampled):
        # Center time point for this 25ms window
        center_idx = i * downsample_factor
        
        # Calculate window: 50ms before to 50ms after
        start_idx = max(0, center_idx - half_window_samples)
        end_idx = min(n_timepoints, center_idx + half_window_samples + 1)
        
        # Average over the 100ms window
        data_matrix_downsampled[:, i, :] = np.mean(data_matrix[:, start_idx:end_idx, :], axis=1)
    
    time_vector_downsampled = np.zeros((n_downsampled, 1))
    for i in range(n_downsampled):
        center_idx = i * downsample_factor
        time_vector_downsampled[i, 0] = time_vector[center_idx]
    
    data_matrix = data_matrix_downsampled
    time_vector = time_vector_downsampled
    target_labels = np.array(target_labels)
    
    return data_matrix, target_labels, time_vector

def plot_source_model_3d_power(volumetric_fpath, data_matrix, time_vector, time_window=(0.8, 1.5)):
    """Create 3D scatter plot of source model vertices colored by power in time window"""
    # Load volumetric data with temporary copy approach
    volumetricTempPath = os.path.join('/Users/mrugank/Desktop', 'volumetric_temp.mat')
    if socket.gethostname() == 'zod':
        copyfile(volumetric_fpath, volumetricTempPath)
        volumetric_data = h5py.File(volumetricTempPath, 'r')
        os.remove(volumetricTempPath)
    else:
        volumetric_data = h5py.File(volumetric_fpath, 'r')
    sourcemodel = volumetric_data['sourcemodel']
    
    # Get positions and inside vertices
    pos = np.array(sourcemodel['pos']).T  # Transpose to get (n_vertices, 3)
    inside = np.array(sourcemodel['inside']).flatten()
    
    # Find time indices for the specified window
    time_points = time_vector.flatten()
    time_mask = (time_points >= time_window[0]) & (time_points <= time_window[1])
    time_indices = np.where(time_mask)[0]
    
    power_in_window = data_matrix[:, time_indices, :].mean(axis=(0, 1))  # Average across trials and time

    # Create 3D scatter plot of inside vertices colored by power
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    inside_pos = pos[inside == 1]
    scatter = ax.scatter(inside_pos[:, 0], inside_pos[:, 1], inside_pos[:, 2], 
                        c=power_in_window, cmap='RdBu_r', alpha=0.8, s=30)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Baseline-Corrected Power (0.8-1.5s)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.show()
    
    # Clean up
    volumetric_data.close()
    if os.path.exists(volumetricTempPath):
        os.remove(volumetricTempPath)


def process_single_timepoint(args):
    """Process a single time point for parallel execution"""
    t, data_matrix, sin_targets, cos_targets, true_angles_deg = args
    
    # Get data for this time point: (trials, sources)
    X = data_matrix[:, t, :]
    # z-score X
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Initialize SVR models
    svr_sin = SVR(kernel='rbf')
    svr_cos = SVR(kernel='rbf')
    
    pred_angles_deg_t = np.empty(data_matrix.shape[0])
    angular_errors_t = np.empty(data_matrix.shape[0])
    
    for test_idx in range(data_matrix.shape[0]):
        train_idx = np.arange(data_matrix.shape[0]) != test_idx
        X_train, X_test = X[train_idx], X[test_idx]
        sin_train, cos_train = sin_targets[train_idx], cos_targets[train_idx]
        true_angles_test = true_angles_deg[test_idx]
        
        # Fit models
        svr_sin.fit(X_train, sin_train)
        svr_cos.fit(X_train, cos_train)
        
        # Predict sin and cos
        pred_sin = svr_sin.predict(X_test.reshape(1, -1))
        pred_cos = svr_cos.predict(X_test.reshape(1, -1))
        
        # Compute predicted angles 
        pred_angles_deg = np.degrees(np.mod(np.arctan2(pred_sin, pred_cos), 2*np.pi))
        pred_angles_deg_t[test_idx] = pred_angles_deg[0]
        
        # Compute angular error (minimum circular distance)
        angular_error = ((pred_angles_deg - true_angles_test) + 180) % 360 - 180
        
        angular_errors_t[test_idx] = angular_error[0]
    
    return t, pred_angles_deg_t, angular_errors_t


def run_timepoint_svr_angle_prediction(data_matrix, target_labels, control=False, n_processes=None):
    """Run SVR to predict angles using sin/cos regression at each time point"""
    
    # Define angle mapping (in degrees)
    angle_mapping = {
        1: 0,    # 0 degrees
        2: 25,   # 25 degrees  
        3: 50,   # 50 degrees
        4: 130,  # 130 degrees
        5: 155,  # 155 degrees
        6: 180,  # 180 degrees
        7: 205,  # 205 degrees
        8: 230,  # 230 degrees
        9: 310,  # 310 degrees
        10: 335  # 335 degrees
    }
    
    # Shuffle target labels for control analysis
    if control:
        target_labels = np.random.permutation(target_labels)
        print("Running CONTROL analysis with shuffled target labels")
    
    # Convert angles to radians and compute sin/cos
    angles_rad = np.array([np.radians(angle_mapping[t]) for t in target_labels])
    sin_targets = np.sin(angles_rad)
    cos_targets = np.cos(angles_rad)
    true_angles_deg = np.array([angle_mapping[t] for t in target_labels])
    
    print(f"Running SVR angle prediction for {data_matrix.shape[1]} time points...")
    print(f"Target angles (degrees): {[angle_mapping[t] for t in np.unique(target_labels)]}")
    
    # Initialize results arrays
    n_timepoints = data_matrix.shape[1]
    pred_angles_deg = np.empty((data_matrix.shape[0], n_timepoints))
    angular_errors = np.empty((data_matrix.shape[0], n_timepoints))
    
    # Set number of processes (default to 8 CPUs)
    if n_processes is None:
        n_processes = min(8, n_timepoints)
    
    print(f"Using {n_processes} processes for parallel computation")
    
    # Prepare arguments for parallel processing
    args_list = [(t, data_matrix, sin_targets, cos_targets, true_angles_deg) 
                 for t in range(n_timepoints)]
    
    # Process time points in parallel
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_single_timepoint, args_list)
    
    # Collect results
    for t, pred_angles_deg_t, angular_errors_t in results:
        pred_angles_deg[:, t] = pred_angles_deg_t
        angular_errors[:, t] = angular_errors_t
        if t % (n_timepoints // 10) == 0:  # Progress update every 10% of time points
            print(f"Completed time point {t}/{n_timepoints} ({100*t/n_timepoints:.1f}%)")
    
    return pred_angles_deg, angular_errors


def plot_svr_results(cv_angle_errors_visual, cv_angle_errors_parietal, cv_angle_errors_frontal, time_vector, title="SVR Angle Prediction Results"):
    """Plot SVR regression results over time"""
    plt.figure(figsize=(10, 6))
    
    # Plot angular error over time
    plt.plot(time_vector.flatten(), cv_angle_errors_frontal, 'b-', linewidth=1, label='Frontal')
    plt.plot(time_vector.flatten(), cv_angle_errors_parietal, 'g-', linewidth=2, label='Parietal')
    plt.plot(time_vector.flatten(), cv_angle_errors_visual, 'r-', linewidth=3, label='Visual')
    
    
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Error (°)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add horizontal line at chance level (random prediction would give ~90 degrees error on average)
    plt.axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='Chance Level (~90°)')
    
    plt.tight_layout()
    plt.show()
    
def main(subjID, voxRes, freq_band='beta'):
    # Take into account default voxRes if not provided
    if voxRes is None:
        voxRes = '10mm'
    taskName = 'mgs'
    
    # Validate frequency band
    if freq_band not in FREQUENCY_BANDS:
        raise ValueError(f"Invalid frequency band '{freq_band}'. Must be one of: {list(FREQUENCY_BANDS.keys())}")
    
    f_min, f_max = FREQUENCY_BANDS[freq_band]
    print(f"Running analysis for frequency band: {freq_band} ({f_min}-{f_max} Hz)")
    
    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    # Create decoding folder (renamed from betaDecodingVC to be frequency-agnostic)
    decoding_dir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', 'decodingVC')
    if not os.path.exists(decoding_dir):
        os.makedirs(decoding_dir)
    results_file = os.path.join(decoding_dir, f'sub-{subjID:02d}_task-{taskName}_SVR_{freq_band}_{voxRes}_withBehav.pkl')
    # if os.path.exists(results_file):
    #     print('Already run for this subject and frequency band')
    #     return
    

    print('Running SVR Angle Prediction Analysis')
    matlab_file = os.path.join(decoding_dir, f'sub-{subjID:02d}_task-{taskName}_sourceSpaceData_{freq_band}_{voxRes}.mat')
    # if os.path.exists(matlab_file):
    #     matlab_data = loadmat(matlab_file)
    #     visual_data_matrix = matlab_data['visual_data_matrix']
    #     parietal_data_matrix = matlab_data['parietal_data_matrix']
    #     frontal_data_matrix = matlab_data['frontal_data_matrix']
    #     target_labels = matlab_data['target_labels'].flatten()
    #     time_vector = matlab_data['time_vector']
    #     i_sacc_err = matlab_data['i_sacc_err'].flatten()
    # else:
    if True:
        # Load source space data
        data_matrix, target_labels, time_vector = load_source_space_data(subjID, bidsRoot, taskName, voxRes, freq_band=freq_band)

        # Load behavioral data
        print("Loading behavioral data...")
        behav_data_path = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'eyetracking', f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')
        
        # Use h5py for MATLAB v7.3 files
        if socket.gethostname() == 'zod':
            behav_data_temp_path = os.path.join('/Users/mrugank/Desktop', f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')
            copyfile(behav_data_path, behav_data_temp_path)
            behav_data = h5py.File(behav_data_temp_path, 'r')
            os.remove(behav_data_temp_path)
        else:
            behav_data = h5py.File(behav_data_path, 'r')
            
        ii_sess_forSource = behav_data['ii_sess_forSource']
            
        # Extract i_sacc_err (initial saccade error)
        # Direct access to i_sacc_err data
        i_sacc_err = np.array(ii_sess_forSource['i_sacc_err']).flatten()
        i_sacc_raw = np.array(ii_sess_forSource['i_sacc_raw'])#.flatten()
        print(f"i_sacc_raw shape: {i_sacc_raw.shape}")
        
        # Convert i_sacc_raw (x, y) coordinates to angles
        # i_sacc_raw is shape (2, n_trials) where first row is x, second row is y
        if i_sacc_raw.shape[0] == 2:
            x_coords = i_sacc_raw[0, :]
            y_coords = i_sacc_raw[1, :]
        else:
            # If shape is (n_trials, 2), transpose
            x_coords = i_sacc_raw[:, 0]
            y_coords = i_sacc_raw[:, 1]
        print(x_coords.shape, y_coords.shape)
        # Convert to angles with (5, 0) as 0 degrees, counter-clockwise to 360
        i_sacc_angle = np.array([convert_xy_to_angle(x_coords[i], y_coords[i], ref_x=5, ref_y=0) for i in range(len(x_coords))])
        # i_sacc_angle = convert_xy_to_angle(x_coords, y_coords, ref_x=5, ref_y=0)
        print(f"Converted i_sacc_raw to angles. Shape: {i_sacc_angle.shape}")
        print(f"Angle range: {np.nanmin(i_sacc_angle):.2f}° to {np.nanmax(i_sacc_angle):.2f}°")

        # exit()

        
        # Filter out trials with NaN values for i_sacc_err
        valid_trials = ~np.isnan(i_sacc_err)
        n_valid_trials = np.sum(valid_trials)
        print(f"Found {n_valid_trials} valid trials out of {len(i_sacc_err)} total trials")
            
        # Filter data matrix and target labels to keep only valid trials
        data_matrix = data_matrix[valid_trials, :, :]
        target_labels = target_labels[valid_trials]
        i_sacc_err = i_sacc_err[valid_trials]
        i_sacc_angle = i_sacc_angle[valid_trials]
        
        # print(f"Data matrix shape after NaN filtering: {data_matrix.shape}")
        # print(f"Target labels shape after NaN filtering: {target_labels.shape}")
        # print(f"i_sacc_err shape after NaN filtering: {i_sacc_err.shape}")
        
        # # Additional filtering: only keep trials with error > 0.01 degrees
        # error_threshold = 0.01
        # error_trials = i_sacc_err > error_threshold
        # n_error_trials = np.sum(error_trials)
        # print(f"Found {n_error_trials} trials with error > {error_threshold}° out of {len(i_sacc_err)} valid trials")
        # print(f"Error range: [{i_sacc_err.min():.3f}°, {i_sacc_err.max():.3f}°]")
        
        # # Apply error filtering
        # data_matrix = data_matrix[error_trials, :, :]
        # target_labels = target_labels[error_trials]
        # i_sacc_err = i_sacc_err[error_trials]
        
        # print(f"Data matrix shape after error filtering: {data_matrix.shape}")
        # print(f"Target labels shape after error filtering: {target_labels.shape}")
        # print(f"i_sacc_err shape after error filtering: {i_sacc_err.shape}")
        # print(f"Final error range: [{i_sacc_err.min():.3f}°, {i_sacc_err.max():.3f}°]")

        # # Extract regional data for filtered trials
        # visual_data = data_matrix[:, :, visual_indices]
        # parietal_data = data_matrix[:, :, parietal_indices]
        # frontal_data = data_matrix[:, :, frontal_indices]


        # Load the Wang atlas saved
        atlas_fpath = os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat')
        atlas_data = loadmat(atlas_fpath)
        visual_points = np.array(atlas_data['visual_points']).flatten()
        parietal_points = np.array(atlas_data['parietal_points']).flatten()
        frontal_points = np.array(atlas_data['frontal_points']).flatten()
        
        # Get indices for each region within the inside vertices
        visual_indices = np.where(visual_points == 1)[0]
        parietal_indices = np.where(parietal_points == 1)[0]
        frontal_indices = np.where(frontal_points == 1)[0]
        
        visual_data_matrix = data_matrix[:, :, visual_indices]
        parietal_data_matrix = data_matrix[:, :, parietal_indices]
        frontal_data_matrix = data_matrix[:, :, frontal_indices]

         # Save MATLAB-compatible file
        
        matlab_data = {
            'visual_data_matrix': visual_data_matrix,
            'parietal_data_matrix': parietal_data_matrix,
            'frontal_data_matrix': frontal_data_matrix,
            'target_labels': target_labels,
            'time_vector': time_vector,
            'i_sacc_err': i_sacc_err,
            'i_sacc_angle': i_sacc_angle,
            
        }
        savemat(matlab_file, matlab_data)
        print(f"Saved MATLAB-compatible data to: {matlab_file}")

        del data_matrix
        gc.collect()
    
    # Run SVR angle prediction
    print("\n" + "="*50)
    print("Running SVR Angle Prediction Analysis")
    print("="*50)

    print("Visual...")
    pred_angles_deg_visual, angular_errors_visual = run_timepoint_svr_angle_prediction(visual_data_matrix, target_labels, control=False)
    pred_angles_deg_visual_control, angular_errors_visual_control = run_timepoint_svr_angle_prediction(visual_data_matrix, target_labels, control=True)
    print("Parietal...")
    pred_angles_deg_parietal, angular_errors_parietal = run_timepoint_svr_angle_prediction(parietal_data_matrix, target_labels, control=False)
    pred_angles_deg_parietal_control, angular_errors_parietal_control = run_timepoint_svr_angle_prediction(parietal_data_matrix, target_labels, control=True)
    print("Frontal...")
    pred_angles_deg_frontal, angular_errors_frontal = run_timepoint_svr_angle_prediction(frontal_data_matrix, target_labels, control=False)
    pred_angles_deg_frontal_control, angular_errors_frontal_control = run_timepoint_svr_angle_prediction(frontal_data_matrix, target_labels, control=True)
    # Save results
    results = {
        'pred_angles_deg_visual': pred_angles_deg_visual,
        'pred_angles_deg_visual_control': pred_angles_deg_visual_control,
        'pred_angles_deg_parietal': pred_angles_deg_parietal,
        'pred_angles_deg_parietal_control': pred_angles_deg_parietal_control,
        'pred_angles_deg_frontal': pred_angles_deg_frontal,
        'pred_angles_deg_frontal_control': pred_angles_deg_frontal_control,
        'angular_errors_visual': angular_errors_visual,
        'angular_errors_visual_control': angular_errors_visual_control,
        'angular_errors_parietal': angular_errors_parietal,
        'angular_errors_parietal_control': angular_errors_parietal_control,
        'angular_errors_frontal': angular_errors_frontal,
        'angular_errors_frontal_control': angular_errors_frontal_control,
        'target_labels': target_labels,
        'i_sacc_err': i_sacc_err,
        'i_sacc_angle': i_sacc_angle,
        'time_vector': time_vector,
        'freq_band': freq_band,
        'freq_range': FREQUENCY_BANDS[freq_band],
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)

    
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inSourceSpaceDecodingWithBehav.py <subjID> [voxRes] [freq_band]")
        print("Example: python inSourceSpaceDecodingWithBehav.py 1 10mm beta")
        print(f"Available frequency bands: {list(FREQUENCY_BANDS.keys())}")
        print(f"Frequency ranges:")
        for band, (f_min, f_max) in FREQUENCY_BANDS.items():
            print(f"  {band}: {f_min}-{f_max} Hz")
        sys.exit(1)
    
    subjID = int(sys.argv[1])
    voxRes = sys.argv[2] if len(sys.argv) > 2 else '10mm'
    freq_band = sys.argv[3] if len(sys.argv) > 3 else 'beta'
    
    print(f"Running source space decoding for subject {subjID} with voxel resolution {voxRes} and frequency band {freq_band}")
    main(subjID, voxRes, freq_band=freq_band)
