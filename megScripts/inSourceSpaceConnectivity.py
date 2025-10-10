import os, h5py, socket, gc
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
import pickle
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
    
    # Keep complex data for connectivity analysis
    data_matrix = data_matrix['real'] + 1j * data_matrix['imag']

    # Remove center of head bias by subtracting mean across all sources at each time point
    # This removes the common mode signal without affecting phase relationships
    real_data = data_matrix.real
    imag_data = data_matrix.imag
    real_mean = np.mean(real_data, axis=0, keepdims=True)
    print(real_mean.shape)
    exit()
    real_normalized = real_data / real_mean
    data_matrix = real_normalized + 1j * imag_data
    # mean_across_sources = np.mean(data_matrix, axis=2, keepdims=True)  # Shape: (trials, time, 1)
    # data_matrix = data_matrix - mean_across_sources
    
    print(f"Data loaded: {data_matrix.shape} (trials × time × sources)")
    print(f"Data type: {data_matrix.dtype}")
    print(f"Time range: {time_vector[0,0]:.2f}s to {time_vector[-1,0]:.2f}s")

    target_labels = np.array(target_labels)
    
    return data_matrix, target_labels, time_vector

    
def main(subjID, voxRes):
    """Main function for source space connectivity analysis"""
    # Take into account default voxRes if not provided
    if voxRes is None:
        voxRes = '10mm'
    taskName = 'mgs'
    
    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    print('Loading source space data for connectivity analysis...')
    
    # Load source space data
    data_matrix, target_labels, time_vector = load_source_space_data(subjID, bidsRoot, taskName, voxRes)

    sourcmodel_fpath = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-01/sourceRecon/sub-01_task-mgs_volumetricSources_10mm.mat'
    # Use temporary copy approach to avoid file locking issues
    if socket.gethostname() == 'zod':
        sourcemodelTempPath = os.path.join('/Users/mrugank/Desktop', 'sourcemodel_temp.mat')
        copyfile(sourcmodel_fpath, sourcemodelTempPath)
        sourcemodel_data = h5py.File(sourcemodelTempPath, 'r')
        sourcemodel = sourcemodel_data['sourcemodel']
        pos = np.array(sourcemodel['pos']).T
        inside = np.array(sourcemodel['inside']).flatten()
        inside_idx = np.where(inside == 1)[0]
        sourcemodel_data.close()
        os.remove(sourcemodelTempPath)
    else:
        sourcemodel_data = h5py.File(sourcmodel_fpath, 'r')
        sourcemodel = sourcemodel_data['sourcemodel']
        pos = np.array(sourcemodel['pos']).T
        inside = np.array(sourcemodel['inside']).flatten()
        inside_idx = np.where(inside == 1)[0]
        sourcemodel_data.close()


    # Mean power from 0.8s to 1.5s
    left_data = data_matrix[np.isin(target_labels, [4, 5, 6, 7, 8]), :, :]
    right_data = data_matrix[np.isin(target_labels, [1, 2, 3, 9, 10]), :, :]
    # Compute power
    left_data = np.abs(left_data) ** 2
    right_data = np.abs(right_data) ** 2
    # Mean power from 0.8s to 1.5s for left and right
    timeIdx = np.where((time_vector >= 0.8) & (time_vector <= 1.5))[0]
    left_mean = left_data[:, timeIdx, :].mean(axis=(0, 1))
    right_mean = right_data[:, timeIdx, :].mean(axis=(0, 1))
    # left_data = left_data / left_mean[np.newaxis, np.newaxis, :] - 1
    # right_data = right_data / right_mean[np.newaxis, np.newaxis, :] - 1
    # Correlation between left and right
    # correlation = np.corrcoef(left_data.flatten(), right_data.flatten())[0, 1]
    # print(f"Correlation between left and right: {correlation}")

   

    f, axs = plt.subplots(2, 1, figsize=(12, 10), subplot_kw={'projection': '3d'})
    
    # Plot left targets (right hemisphere)
    scatter1 = axs[0].scatter(pos[inside_idx, 0], pos[inside_idx, 1], pos[inside_idx, 2], 
                             c=left_mean, cmap='RdBu_r', s=20, alpha=0.8)
    axs[0].set_title('Left Targets (Right Hemisphere)', fontsize=14)
    axs[0].set_xlabel('X (mm)')
    axs[0].set_ylabel('Y (mm)')
    axs[0].set_zlabel('Z (mm)')
    cbar1 = plt.colorbar(scatter1, ax=axs[0], shrink=0.8)
    cbar1.set_label('Mean Power (0.8-1.5s)', rotation=270, labelpad=20)
    
    # Plot right targets (left hemisphere)
    scatter2 = axs[1].scatter(pos[inside_idx, 0], pos[inside_idx, 1], pos[inside_idx, 2], 
                             c=right_mean, cmap='RdBu_r', s=20, alpha=0.8)
    axs[1].set_title('Right Targets (Left Hemisphere)', fontsize=14)
    axs[1].set_xlabel('X (mm)')
    axs[1].set_ylabel('Y (mm)')
    axs[1].set_zlabel('Z (mm)')
    cbar2 = plt.colorbar(scatter2, ax=axs[1], shrink=0.8)
    cbar2.set_label('Mean Power (0.8-1.5s)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.show()
    exit()

    # Load the Wang atlas
    atlas_fpath = os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat')
    atlas_data = loadmat(atlas_fpath)
    visual_points = np.array(atlas_data['visual_points']).flatten()
    parietal_points = np.array(atlas_data['parietal_points']).flatten()
    frontal_points = np.array(atlas_data['frontal_points']).flatten()
    
    # Get indices for each region within the inside vertices
    visual_indices = np.where(visual_points == 1)[0]
    parietal_indices = np.where(parietal_points == 1)[0]
    frontal_indices = np.where(frontal_points == 1)[0]
    
    print(f"Data shape: {data_matrix.shape}")
    print(f"Visual region: {len(visual_indices)} sources")
    print(f"Parietal region: {len(parietal_indices)} sources")
    print(f"Frontal region: {len(frontal_indices)} sources")
    print(f"Time points: {data_matrix.shape[1]}")
    print(f"Number of trials: {data_matrix.shape[0]}")
    
    # TODO: Add connectivity analysis here
    print("Ready for connectivity analysis implementation...")
    
    
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
