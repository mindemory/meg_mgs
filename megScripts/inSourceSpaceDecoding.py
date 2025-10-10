import os, h5py, socket, gc
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC, SVR
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score
from scipy.io import loadmat

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
    
    # Compute power from complex data
    complex_data = data_matrix['real'] + 1j * data_matrix['imag']
    data_matrix = np.abs(complex_data) ** 2

    # Baseline correction: compute baseline from -0.75s to 0s (deprecated, maybe not needed)
    # baseline_mask = (time_vector.flatten() >= -0.75) & (time_vector.flatten() <= 0)
    # baseline_indices = np.where(baseline_mask)[0]
    # baseline_power = data_matrix[:, baseline_indices, :].mean(axis=(0, 1))  # Shape: (sources,)
    # data_matrix = data_matrix / baseline_power[np.newaxis, np.newaxis, :] - 1
    mean_allTrials = data_matrix.mean(axis=0)
    data_matrix = data_matrix / mean_allTrials[np.newaxis, :, :] - 1

    # Downsample data to 50ms resolution
    dt = np.mean(np.diff(time_vector.flatten()))
    target_dt = 0.05  # 50ms
    downsample_factor = int(target_dt / dt)
    if downsample_factor < 1:
        downsample_factor = 1
    # Downsample data_matrix by averaging over time windows
    n_trials, n_timepoints, n_sources = data_matrix.shape
    n_downsampled = n_timepoints // downsample_factor
    data_matrix_downsampled = np.zeros((n_trials, n_downsampled, n_sources))
    for i in range(n_downsampled):
        start_idx = i * downsample_factor
        end_idx = min((i + 1) * downsample_factor, n_timepoints)
        data_matrix_downsampled[:, i, :] = np.mean(data_matrix[:, start_idx:end_idx, :], axis=1)
    # Downsample time_vector to match data (take starting time of each bin)
    time_vector_downsampled = np.zeros((n_downsampled, 1))
    for i in range(n_downsampled):
        start_idx = i * downsample_factor
        time_vector_downsampled[i, 0] = time_vector[start_idx, 0]
    
    data_matrix = data_matrix_downsampled
    time_vector = time_vector_downsampled
    # plt.figure()
    # plt.hist(data_matrix.flatten(), bins=100)
    # plt.show()

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
    os.remove(volumetricTempPath)

def run_timepoint_svm_classification(data_matrix, target_labels, time_vector, classifType):
    """Run SVM classification at each time point with different classification schemes"""
    print(f"Running time-point-wise SVM classification ({classifType}) with train/test splits...")

    # Shuffle data_matrix and target_labels
    newIdx = np.random.permutation(len(data_matrix))
    data_matrix = data_matrix[newIdx, :, :]
    target_labels = target_labels[newIdx]
    
    # Create labels based on classification type
    if classifType == 'binary':
        # Binary: left vs right targets
        leftTargets = [4, 5, 6, 7, 8]
        rightTargets = [1, 2, 3, 9, 10]
        class_labels = np.zeros(len(target_labels))
        class_labels[np.isin(target_labels, rightTargets)] = 1
        n_classes = 2
        
    elif classifType == '4way':
        # 4-way: class1=2,3; class2=4,5; class3=7,8; class4=9,10
        # Keep only targets 2,3,4,5,7,8,9,10 (drop 1 and 6)
        keep_targets = [2, 3, 4, 5, 7, 8, 9, 10]
        keep_idx = np.isin(target_labels, keep_targets)
        target_labels = target_labels[keep_idx]
        data_matrix = data_matrix[keep_idx, :, :]
        class_labels = np.zeros(len(target_labels))
        class_labels[np.isin(target_labels, [2, 3])] = 0
        class_labels[np.isin(target_labels, [4, 5])] = 1
        class_labels[np.isin(target_labels, [7, 8])] = 2
        class_labels[np.isin(target_labels, [9, 10])] = 3
        n_classes = 4
        
    elif classifType == '6way':
        # 6-way: class1=1; class2=2,3; class3=4,5; class4=6; class5=7,8; class6=9,10
        class_labels = np.zeros(len(target_labels))
        class_labels[np.isin(target_labels, [1])] = 0
        class_labels[np.isin(target_labels, [2, 3])] = 1
        class_labels[np.isin(target_labels, [4, 5])] = 2
        class_labels[np.isin(target_labels, [6])] = 3
        class_labels[np.isin(target_labels, [7, 8])] = 4
        class_labels[np.isin(target_labels, [9, 10])] = 5
        n_classes = 6
        
    elif classifType == '10way':
        # 10-way: each target is its own class (targets 1-10)
        class_labels = target_labels - 1  # Convert to 0-based indexing
        n_classes = 10
        
    else:
        raise ValueError(f"Unknown classification type: {classifType}")
    
    print(f"Classification type: {classifType}")
    print(f"Number of classes: {n_classes}")
    print(f"Class distribution: {np.bincount(class_labels.astype(int))}")
    
    # Initialize results arrays
    n_timepoints = data_matrix.shape[1]
    cv_accuracy_scores = np.zeros(n_timepoints)
    
    print(f"Running classification for {n_timepoints} time points...")
    
    for t in range(n_timepoints):
        if t % (n_timepoints // 10) == 0:  # Progress update every 10% of time points
            print(f"Processing time point {t}/{n_timepoints} ({t/n_timepoints*100:.1f}%)")
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
        
        # Get data for this time point: (trials, sources)
        X = data_matrix[:, t, :]
        # z-score X
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Initialize SVM classifier
        svm = SVC(kernel='rbf', random_state=42)
        
        # Cross-validation for accuracy
        cv_accuracy = cross_val_score(svm, X, class_labels, cv=cv, scoring='accuracy')
        cv_accuracy_scores[t] = np.mean(cv_accuracy)
        
    
    print("Classification completed!")
    
    return cv_accuracy_scores, time_vector

def run_timepoint_svr_angle_prediction(data_matrix, target_labels, time_vector):
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
    
    # Convert angles to radians and compute sin/cos
    angles_rad = np.array([np.radians(angle_mapping[t]) for t in target_labels])
    sin_targets = np.sin(angles_rad)
    cos_targets = np.cos(angles_rad)
    true_angles_deg = np.array([angle_mapping[t] for t in target_labels])
    
    print(f"Running SVR angle prediction for {data_matrix.shape[1]} time points...")
    print(f"Target angles (degrees): {[angle_mapping[t] for t in np.unique(target_labels)]}")
    
    # Initialize results arrays
    n_timepoints = data_matrix.shape[1]
    cv_angle_errors = np.zeros(n_timepoints)
    
    # Use Leave-One-Out for regression (more robust for small datasets)
    cv = LeaveOneOut()
    
    for t in range(n_timepoints):
        if t % (n_timepoints // 10) == 0:  # Progress update every 10% of time points
            print(f"Processing time point {t}/{n_timepoints} ({100*t/n_timepoints:.1f}%)")
        
        # Get data for this time point: (trials, sources)
        X = data_matrix[:, t, :]
        # z-score X
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
        # Initialize SVR models
        svr_sin = SVR(kernel='rbf')
        svr_cos = SVR(kernel='rbf')
        
        # Cross-validation for angle prediction
        fold_angle_errors = []
        
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            sin_train, cos_train = sin_targets[train_idx], cos_targets[train_idx]
            true_angles_test = true_angles_deg[test_idx]
            
            # Fit models
            svr_sin.fit(X_train, sin_train)
            svr_cos.fit(X_train, cos_train)
            
            # Predict sin and cos
            pred_sin = svr_sin.predict(X_test)
            pred_cos = svr_cos.predict(X_test)
            
            # Compute predicted angles using arctangent
            pred_angles_rad = np.arctan2(pred_sin, pred_cos)
            pred_angles_deg = np.degrees(pred_angles_rad)
            
            # Handle angle wrapping (ensure angles are in [0, 360))
            pred_angles_deg = np.mod(pred_angles_deg, 360)
            true_angles_test = np.mod(true_angles_test, 360)
            
            # Compute angular error (minimum circular distance)
            angular_diff = np.abs(pred_angles_deg - true_angles_test)
            angular_error = np.minimum(angular_diff, 360 - angular_diff)
            
            fold_angle_errors.extend(angular_error)
        
        cv_angle_errors[t] = np.mean(fold_angle_errors)
        
    return cv_angle_errors, time_vector

def plot_classification_results(cv_accuracy_scores, time_vector, classifType, title="SVM Classification Results"):
    """Plot classification results over time"""
    plt.figure(figsize=(12, 6))
    
    # Calculate chance level based on classification type
    if classifType == 'binary':
        chance_level = 0.5
    elif classifType == '4way':
        chance_level = 0.25
    elif classifType == '6way':
        chance_level = 1.0/6
    elif classifType == '10way':
        chance_level = 0.1
    else:
        chance_level = 0.5  # Default fallback
    
    # Plot accuracy
    plt.plot(time_vector.flatten(), cv_accuracy_scores, 'b-', linewidth=2, label='CV Accuracy')
    plt.axhline(y=chance_level, color='r', linestyle='--', alpha=0.7, label=f'Chance ({chance_level:.2f})')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Score')
    plt.title(f"{title} ({classifType})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_svr_results(cv_angle_errors_visual, time_vector_svr_visual, cv_angle_errors_parietal, time_vector_svr_parietal, cv_angle_errors_frontal, time_vector_svr_frontal, title="SVR Angle Prediction Results"):
    """Plot SVR regression results over time"""
    plt.figure(figsize=(10, 6))
    
    # Plot angular error over time
    plt.plot(time_vector_svr_frontal.flatten(), cv_angle_errors_frontal, 'b-', linewidth=1, label='Frontal')
    plt.plot(time_vector_svr_parietal.flatten(), cv_angle_errors_parietal, 'g-', linewidth=2, label='Parietal')
    plt.plot(time_vector_svr_visual.flatten(), cv_angle_errors_visual, 'r-', linewidth=3, label='Visual')
    
    
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
    # classifType = '10way' # valid option: binary, 4way, 6way, 10way
    
    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    # Create betaDecodingVC folder
    betaDecodingVC_dir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', 'betaDecodingVC')
    if not os.path.exists(betaDecodingVC_dir):
        os.makedirs(betaDecodingVC_dir)
    results_file = os.path.join(betaDecodingVC_dir, f'sub-{subjID:02d}_task-{taskName}_betaSVR_{voxRes}.pkl')
    if os.path.exists(results_file):
        print('Loading existing results')
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        cv_angle_errors_visual = results['cv_angle_errors_visual']
        time_vector_svr_visual = results['time_vector_svr_visual']
        cv_angle_errors_parietal = results['cv_angle_errors_parietal']
        time_vector_svr_parietal = results['time_vector_svr_parietal']
        cv_angle_errors_frontal = results['cv_angle_errors_frontal']
        time_vector_svr_frontal = results['time_vector_svr_frontal']
    else:
        print('Running SVR Angle Prediction Analysis')
        
        # Load source space data
        data_matrix, target_labels, time_vector = load_source_space_data(subjID, bidsRoot, taskName, voxRes)

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

        del data_matrix
        gc.collect()
    
        # # Run SVM classification at each time point
        # cv_accuracy_scores, time_vector = run_timepoint_svm_classification(
        #     data_matrix, target_labels, time_vector, classifType)
        
        # # Plot classification results
        # plot_classification_results(cv_accuracy_scores, time_vector, classifType,
        #                           title="Target Classification (Source Space)")
        
        # Run SVR angle prediction
        print("\n" + "="*50)
        print("Running SVR Angle Prediction Analysis")
        print("="*50)

        cv_angle_errors_visual, time_vector_svr_visual = run_timepoint_svr_angle_prediction(
            visual_data_matrix, target_labels, time_vector)

        cv_angle_errors_parietal, time_vector_svr_parietal = run_timepoint_svr_angle_prediction(
            parietal_data_matrix, target_labels, time_vector)

        cv_angle_errors_frontal, time_vector_svr_frontal = run_timepoint_svr_angle_prediction(
            frontal_data_matrix, target_labels, time_vector)

        # Save results
        results = {
            'cv_angle_errors_visual': cv_angle_errors_visual,
            'cv_angle_errors_parietal': cv_angle_errors_parietal,
            'cv_angle_errors_frontal': cv_angle_errors_frontal,
            'time_vector_svr_visual': time_vector_svr_visual,
            'time_vector_svr_parietal': time_vector_svr_parietal,
            'time_vector_svr_frontal': time_vector_svr_frontal,
        }
        
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

    # # Plot SVR results  
    # plot_svr_results(cv_angle_errors_visual, time_vector_svr_visual,
    #                 cv_angle_errors_parietal, time_vector_svr_parietal,
    #                 cv_angle_errors_frontal, time_vector_svr_frontal,
    #                 title="SVR Angle Prediction (Source Space)")
    
    
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inSourceSpaceDecoding.py <subjID> [voxRes]")
        print("Example: python inSourceSpaceDecoding.py 1 10mm")
        sys.exit(1)
    
    subjID = int(sys.argv[1])
    voxRes = sys.argv[2] if len(sys.argv) > 2 else '10mm'
    
    print(f"Running source space decoding for subject {subjID} with voxel resolution {voxRes}")
    main(subjID, voxRes)