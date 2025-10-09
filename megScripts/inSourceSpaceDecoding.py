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

    # Baseline correction: compute baseline from -0.75s to 0s
    # baseline_mask = (time_vector.flatten() >= -0.75) & (time_vector.flatten() <= 0)
    # baseline_indices = np.where(baseline_mask)[0]
    # baseline_power = data_matrix[:, baseline_indices, :].mean(axis=(0, 1))  # Shape: (sources,)
    # data_matrix = data_matrix / baseline_power[np.newaxis, np.newaxis, :] - 1
    mean_allTrials = data_matrix.mean(axis=0)
    data_matrix = data_matrix / mean_allTrials[np.newaxis, :, :] - 1

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

def run_timepoint_svm_classification(data_matrix, target_labels, time_vector, leftTargets, rightTargets):
    """Run SVM classification at each time point for left vs right targets with proper train/test splits"""
    print("Running time-point-wise SVM classification with train/test splits...")
    
    # Create binary labels: 0 for left targets, 1 for right targets
    binary_labels = np.zeros(len(target_labels))
    binary_labels[np.isin(target_labels, rightTargets)] = 1
    
    print(f"Binary labels: {np.bincount(binary_labels.astype(int))}")
    print(f"Left targets: {np.sum(binary_labels == 0)}, Right targets: {np.sum(binary_labels == 1)}")
    
    # Initialize results arrays
    n_timepoints = data_matrix.shape[1]
    cv_accuracy_scores = np.zeros(n_timepoints)
    cv_f1_scores = np.zeros(n_timepoints)
    
    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 
    
    print(f"Running classification for {n_timepoints} time points...")
    
    for t in range(n_timepoints):
        if t % 50 == 0:  # Progress update every 50 time points
            print(f"Processing time point {t}/{n_timepoints} ({t/n_timepoints*100:.1f}%)")
        
        # Get data for this time point: (trials, sources)
        X = data_matrix[:, t, :]
        # X = X.reshape(X.shape[0], -1)
        
        # Initialize SVM classifier
        svm = SVC(kernel='rbf', random_state=42)
        
        # Cross-validation for accuracy
        cv_accuracy = cross_val_score(svm, X, binary_labels, cv=cv, scoring='accuracy')
        cv_accuracy_scores[t] = np.mean(cv_accuracy)
        
        # Cross-validation for F1 score
        cv_f1 = cross_val_score(svm, X, binary_labels, cv=cv, scoring='f1')
        cv_f1_scores[t] = np.mean(cv_f1)
    
    print("Classification completed!")
    
    return cv_accuracy_scores, cv_f1_scores, time_vector

def plot_classification_results(cv_accuracy_scores, cv_f1_scores, time_vector, title="SVM Classification Results"):
    """Plot classification results over time"""
    plt.figure(figsize=(12, 6))
    
    # Plot both metrics on the same plot
    plt.plot(time_vector.flatten(), cv_accuracy_scores, 'b-', linewidth=2, label='CV Accuracy')
    plt.plot(time_vector.flatten(), cv_f1_scores, 'g-', linewidth=2, label='CV F1 Score')
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Chance')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Score')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def main():
    # Test with subject 1 first
    subjID = 1
    taskName = 'mgs'
    
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
    posterior_inside_mask = inside_pos[:, 1] < 0
    posterior_inside = inside_pos[posterior_inside_mask]

    # fig = plt.figure(figsize=(12, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(inside_pos[:, 0], inside_pos[:, 1], inside_pos[:, 2], c='b', s=30)
    # ax.scatter(posterior_inside[:, 0], posterior_inside[:, 1], posterior_inside[:, 2], c='r', s=30)
    
    # plt.tight_layout()
    # plt.show()
    # exit()
    
    print(f"Posterior sources: {len(posterior_inside)}")
    print(f"Posterior inside sources: {len(posterior_inside)}")
    
    # Get the indices of posterior sources within the inside vertices
    posterior_inside_indices = np.where(posterior_inside_mask)[0]
    print(f"Posterior inside indices: {len(posterior_inside_indices)}")
    
    data_matrix = data_matrix[:, :, posterior_inside_indices]

    leftTargets = [4, 5, 6, 7, 8]
    rightTargets = [1, 2, 3, 9, 10]
    
    print("Data loaded successfully!")
    print(f"Data matrix shape: {data_matrix.shape}")
    print(f"Target labels shape: {target_labels.shape}")
    print(f"Time vector shape: {time_vector.shape}")
    print(f"Number of trials: {len(target_labels)}")
    
    # Run SVM classification at each time point
    cv_accuracy_scores, cv_f1_scores, time_vector = run_timepoint_svm_classification(
        data_matrix, target_labels, time_vector, leftTargets, rightTargets)
    
    # Plot classification results
    plot_classification_results(cv_accuracy_scores, cv_f1_scores, time_vector, 
                              title="Left vs Right Target Classification (Source Space)")

if __name__ == '__main__':
    main()