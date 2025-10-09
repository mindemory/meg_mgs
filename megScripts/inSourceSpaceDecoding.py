import os, h5py, socket
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score

def load_source_space_data(subjID, bidsRoot, taskName):
    """Load and concatenate source space data for all targets"""
    subName = 'sub-%02d' % subjID
    print(f'Loading source space data for {subName}')
    
    # File paths
    derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName)
    sourceReconRoot = os.path.join(derivativesRoot, 'sourceRecon')
    freqSpaceRoot = os.path.join(sourceReconRoot, 'freqSpace')
    freqSpace_fpath = os.path.join(freqSpaceRoot, f'{subName}_task-{taskName}_complexbeta_allTargets_10.mat')
    
    # Load data with temporary copy approach
    freqSpaceTempPath = os.path.join('/Users/mrugank/Desktop', f'{subName}_task-{taskName}_complexbeta_allTargets_10.mat')
    copyfile(freqSpace_fpath, freqSpaceTempPath)
    freqSpace_data = h5py.File(freqSpaceTempPath, 'r')
    os.remove(freqSpaceTempPath)
    
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
    copyfile(volumetric_fpath, volumetricTempPath)
    volumetric_data = h5py.File(volumetricTempPath, 'r')
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
    cv_f1_scores = np.zeros(n_timepoints)
    
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
        
        # Cross-validation for F1 score (macro-averaged for multi-class)
        cv_f1 = cross_val_score(svm, X, class_labels, cv=cv, scoring='f1_macro')
        cv_f1_scores[t] = np.mean(cv_f1)
    
    print("Classification completed!")
    
    return cv_accuracy_scores, cv_f1_scores, time_vector

def plot_classification_results(cv_accuracy_scores, cv_f1_scores, time_vector, classifType, title="SVM Classification Results"):
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
    
    # Plot both metrics on the same plot
    plt.plot(time_vector.flatten(), cv_accuracy_scores, 'b-', linewidth=2, label='CV Accuracy')
    plt.plot(time_vector.flatten(), cv_f1_scores, 'g-', linewidth=2, label='CV F1 Score')
    plt.axhline(y=chance_level, color='r', linestyle='--', alpha=0.7, label=f'Chance ({chance_level:.2f})')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Score')
    plt.title(f"{title} ({classifType})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def main():
    # Test with subject 1 first
    subjID = 1
    taskName = 'mgs'
    classifType = '10way' # valid option: binary, 4way, 6way, 10way
    
    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    # Load source space data
    data_matrix, target_labels, time_vector = load_source_space_data(subjID, bidsRoot, taskName)

    # Select only posterior sources
    volumetric_fpath = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'sub-{subjID:02d}_task-{taskName}_volumetricSources_10mm.mat')
    volumetricTempPath = os.path.join('/Users/mrugank/Desktop', 'volumetric_temp.mat')
    copyfile(volumetric_fpath, volumetricTempPath)
    volumetric_data = h5py.File(volumetricTempPath, 'r')
    sourcemodel = volumetric_data['sourcemodel']
    # Get positions and inside vertices
    pos = np.array(sourcemodel['pos']).T  # Transpose to get (n_vertices, 3)
    inside = np.array(sourcemodel['inside']).flatten()
    
    # Map to inside vertex indices (data_matrix only has inside vertices)
    inside_indices = np.where(inside == 1)[0]
    inside_pos = pos[inside_indices]
    # Filter inside positions to get posterior ones (y < 0)
    posterior_inside_mask = (inside_pos[:, 1] < -25) & (inside_pos[:, 2] > 0)
    posterior_inside = inside_pos[posterior_inside_mask]

    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(inside_pos[:, 0], inside_pos[:, 1], inside_pos[:, 2], c='b', s=30)
    # ax.scatter(posterior_inside[:, 0], posterior_inside[:, 1], posterior_inside[:, 2], c='r', s=30)
    
    # plt.tight_layout()
    # plt.show()
    # exit()
    
    # Get the indices of posterior sources within the inside vertices
    posterior_inside_indices = np.where(posterior_inside_mask)[0]
    data_matrix = data_matrix[:, :, posterior_inside_indices]

   
    
    print("Data loaded successfully!")
    print(f"Data matrix shape: {data_matrix.shape}")
    print(f"Target labels shape: {target_labels.shape}")
    print(f"Time vector shape: {time_vector.shape}")
    print(f"Number of trials: {len(target_labels)}")
    
    # Run SVM classification at each time point
    cv_accuracy_scores, cv_f1_scores, time_vector = run_timepoint_svm_classification(
        data_matrix, target_labels, time_vector, classifType)
    
    # Plot classification results
    plot_classification_results(cv_accuracy_scores, cv_f1_scores, time_vector, classifType,
                              title="Target Classification (Source Space)")

if __name__ == '__main__':
    main()