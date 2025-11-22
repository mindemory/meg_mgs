import os, h5py, socket, gc
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut
from scipy.io import loadmat, savemat
from scipy.stats import circmean, circstd


# Make sure to run conda activate megAnalyses before running this script

def load_source_space_data(subjID, bidsRoot, taskName, voxRes):
    """Load and concatenate source space data for all targets"""
    subName = 'sub-%02d' % subjID
    print(f'Loading source space data for {subName}')
    
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
    print("Computing time-varying power in 18-25 Hz range using bandpass filter...")
    from scipy import signal
    
    dt = np.mean(np.diff(time_vector))
    f_min, f_max = 18.0, 25.0  # Hz
    
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

    # Downsample to 50ms resolution
    target_dt = 0.05  # 50ms
    downsample_factor = int(target_dt / dt)
    if downsample_factor < 1:
        downsample_factor = 1
    
    n_trials, n_timepoints, n_sources = data_matrix.shape
    n_downsampled = n_timepoints // downsample_factor
    data_matrix_downsampled = np.zeros((n_trials, n_downsampled, n_sources))
    
    for i in range(n_downsampled):
        start_idx = i * downsample_factor
        end_idx = min((i + 1) * downsample_factor, n_timepoints)
        data_matrix_downsampled[:, i, :] = np.mean(data_matrix[:, start_idx:end_idx, :], axis=1)
    
    time_vector_downsampled = np.zeros((n_downsampled, 1))
    for i in range(n_downsampled):
        start_idx = i * downsample_factor
        time_vector_downsampled[i, 0] = time_vector[start_idx]
    
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


def run_timepoint_svc_angle_prediction(data_matrix, target_labels, control=False):
    """Run SVC to predict angles using 10-class classification at each time point"""
    
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
    
    true_angles_deg = np.array([angle_mapping[t] for t in target_labels])
    
    print(f"Running SVC angle prediction for {data_matrix.shape[1]} time points...")
    print(f"Target angles (degrees): {[angle_mapping[t] for t in np.unique(target_labels)]}")
    
    # Initialize results arrays
    n_timepoints = data_matrix.shape[1]
    angular_errors = np.empty((data_matrix.shape[0], n_timepoints))
    
    # Use Leave-One-Out for regression (more robust for small datasets)
    # cv = LeaveOneOut()
    
    for t in range(n_timepoints):
        if t % (n_timepoints // 10) == 0:  # Progress update every 10% of time points
            print(f"Processing time point {t}/{n_timepoints} ({100*t/n_timepoints:.1f}%)")
        
        # Get data for this time point: (trials, sources)
        X = data_matrix[:, t, :]
        # z-score X
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Initialize SVC model
        svc = SVC(kernel='rbf', decision_function_shape='ovr')
        
        for test_idx in range(data_matrix.shape[0]):
        # for train_idx, test_idx in cv.split(X):
            train_idx = np.arange(data_matrix.shape[0]) != test_idx
            X_train, X_test = X[train_idx], X[test_idx]
            # true_class_test = target_labels[test_idx]
            true_angles_test = true_angles_deg[test_idx]
            
            # Fit model
            svc.fit(X_train, target_labels[train_idx])
            
            # Predict class
            pred_class = svc.predict(X_test.reshape(1, -1))
            
            # Convert predicted class to angle
            pred_angles_deg = np.array([angle_mapping[pred_class[0]]])
            
            # Compute angular error (minimum circular distance)
            angular_error = ((pred_angles_deg - true_angles_test) + 180) % 360 - 180
            
            angular_errors[test_idx, t] = angular_error[0]
        
    return angular_errors


def plot_svc_results(cv_angle_errors_visual, cv_angle_errors_parietal, cv_angle_errors_frontal, time_vector, title="SVC Angle Prediction Results"):
    """Plot SVC classification results over time"""
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
    
def main(subjID, voxRes):
    # Take into account default voxRes if not provided
    if voxRes is None:
        voxRes = '10mm'
    taskName = 'mgs'
    
    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    # Create betaDecodingVC folder
    betaDecodingVC_dir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', 'betaDecodingVC')
    if not os.path.exists(betaDecodingVC_dir):
        os.makedirs(betaDecodingVC_dir)
    results_file = os.path.join(betaDecodingVC_dir, f'sub-{subjID:02d}_task-{taskName}_betaSVC_{voxRes}_withBehav.pkl')
    
    print('Running SVC Angle Prediction Analysis')
    
    # Load existing MATLAB-compatible file
    matlab_file = os.path.join(betaDecodingVC_dir, f'sub-{subjID:02d}_task-{taskName}_sourceSpaceData_{voxRes}.mat')
    print(f"Loading existing data from: {matlab_file}")
    
    matlab_data = loadmat(matlab_file)
    visual_data_matrix = matlab_data['visual_data_matrix']
    parietal_data_matrix = matlab_data['parietal_data_matrix']
    frontal_data_matrix = matlab_data['frontal_data_matrix']
    target_labels = matlab_data['target_labels'].flatten()
    time_vector = matlab_data['time_vector']
    i_sacc_err = matlab_data['i_sacc_err'].flatten()
    
    print(f"Loaded data shapes:")
    print(f"  Visual: {visual_data_matrix.shape}")
    print(f"  Parietal: {parietal_data_matrix.shape}")
    print(f"  Frontal: {frontal_data_matrix.shape}")
    print(f"  Target labels: {target_labels.shape}")
    print(f"  Time vector: {time_vector.shape}")
    print(f"  i_sacc_err: {i_sacc_err.shape}")
    
    # Run SVC angle prediction
    print("\n" + "="*50)
    print("Running SVC Angle Prediction Analysis")
    print("="*50)

    angular_errors_visual = run_timepoint_svc_angle_prediction(visual_data_matrix, target_labels)
    angular_errors_visual_control = run_timepoint_svc_angle_prediction(visual_data_matrix, target_labels, control=True)

    f, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].plot(time_vector.flatten(), circmean(angular_errors_visual, axis=0, high=180, low=-180), 'ro')
    axs[1].plot(time_vector.flatten(), circmean(angular_errors_visual_control, axis=0, high=180, low=-180), 'ro')
    axs[0].set_title('Visual')
    axs[1].set_title('Visual (Control)')
    axs[0].set_xlabel('Time (s)')
    axs[1].set_xlabel('Time (s)')
    axs[0].set_ylabel('Angular Error (°)')
    axs[1].set_ylabel('Angular Error (°)')
    axs[0].axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='Chance Level (~90°)')
    plt.show()
    exit()
    angular_errors_parietal = run_timepoint_svc_angle_prediction(parietal_data_matrix, target_labels)
    angular_errors_frontal = run_timepoint_svc_angle_prediction(frontal_data_matrix, target_labels)

    # Save results
    results = {
        'angular_errors_visual': angular_errors_visual,
        'angular_errors_parietal': angular_errors_parietal,
        'angular_errors_frontal': angular_errors_frontal,
        'time_vector': time_vector,
    }

    # with open(results_file, 'wb') as f:
    #     pickle.dump(results, f)

    quantiles = 4
    f, axs = plt.subplots(1, 3, figsize=(18, 5))
    for q in range(1, quantiles + 1):
        print(f"Processing quantile {q}")

        if q == 1:
            trlIdx = np.where(i_sacc_err <= np.quantile(i_sacc_err, 0.25))[0]
            clr = 'green'
        elif q == 2:
            trlIdx = np.where((i_sacc_err > np.quantile(i_sacc_err, 0.25)) & (i_sacc_err <= np.quantile(i_sacc_err, 0.5)))[0]
            clr = 'magenta'
        elif q == 3:
            trlIdx = np.where((i_sacc_err > np.quantile(i_sacc_err, 0.5)) & (i_sacc_err <= np.quantile(i_sacc_err, 0.75)))[0]
            clr = 'blue'
        elif q == 4:
            trlIdx = np.where(i_sacc_err > np.quantile(i_sacc_err, 0.75))[0]
            clr = 'red'
        print(f"Q{q} trials: {len(trlIdx)}")

        angular_errors_visual_q = angular_errors_visual[trlIdx, :]
        angular_errors_parietal_q = angular_errors_parietal[trlIdx, :]
        angular_errors_frontal_q = angular_errors_frontal[trlIdx, :]
        
        cv_angle_errors_visual_q = circmean(angular_errors_visual_q, axis=0, high=180, low=-180)
        cv_angle_errors_parietal_q = circmean(angular_errors_parietal_q, axis=0, high=180, low=-180)
        cv_angle_errors_frontal_q = circmean(angular_errors_frontal_q, axis=0, high=180, low=-180)

        axs[0].plot(time_vector.flatten(), cv_angle_errors_visual_q, clr, linewidth=3, label=f'Q{q}')
        axs[1].plot(time_vector.flatten(), cv_angle_errors_parietal_q, clr, linewidth=2, label=f'Q{q}')
        axs[2].plot(time_vector.flatten(), cv_angle_errors_frontal_q, clr, linewidth=1, label=f'Q{q}')

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    axs[0].set_title('Visual')
    axs[1].set_title('Parietal')
    axs[2].set_title('Frontal')
    axs[0].set_xlabel('Time (s)')
    axs[1].set_xlabel('Time (s)')
    axs[2].set_xlabel('Time (s)')
    axs[0].set_ylabel('Angular Error (°)')
    axs[1].set_ylabel('Angular Error (°)')
    axs[2].set_ylabel('Angular Error (°)')
    axs[0].axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='Chance Level (~90°)')
    axs[1].axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='Chance Level (~90°)')
    axs[2].axhline(y=90, color='gray', linestyle='--', alpha=0.7, label='Chance Level (~90°)')
    axs[0].axvline(x=0, color='black', linestyle='--', alpha=0.7)
    axs[1].axvline(x=0, color='black', linestyle='--', alpha=0.7)
    axs[2].axvline(x=0, color='black', linestyle='--', alpha=0.7)
    # axs[0].set_ylim(80, 100)
    # axs[1].set_ylim(80, 100)
    # axs[2].set_ylim(80, 100)
    axs[0].set_xlim(-1.5, 1.7)
    axs[1].set_xlim(-1.5, 1.7)
    axs[2].set_xlim(-1.5, 1.7)
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inSourceSpaceClassificationWithBehav.py <subjID> [voxRes]")
        print("Example: python inSourceSpaceClassificationWithBehav.py 1 10mm")
        sys.exit(1)
    
    subjID = int(sys.argv[1])
    voxRes = sys.argv[2] if len(sys.argv) > 2 else '10mm'
    
    print(f"Running source space classification for subject {subjID} with voxel resolution {voxRes}")
    main(subjID, voxRes)
