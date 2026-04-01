import os, h5py, socket, gc
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
from shutil import copyfile
import pickle
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, hilbert
from scipy.ndimage import uniform_filter1d
import time
from joblib import Parallel, delayed

def get_compute_profile():
    h = socket.gethostname()
    if h == 'zod': return 'mac', 4
    elif h == 'vader': return 'vader', 48
    else: return 'hpc', 10

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut, highcut, fs, axis=1):
    """Apply zero-phase butterworth filter."""
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data, axis=axis)

def extract_phase(data, fs, lowcut, highcut):
    """Filter and extract phase via Hilbert."""
    filtered = apply_bandpass(data, lowcut, highcut, fs)
    analytic = hilbert(filtered, axis=1)
    return np.angle(analytic)

def load_behavioral_mask(subjID, bidsRoot):
    subName = 'sub-%02d' % subjID
    behav_path = os.path.join(bidsRoot, 'derivatives', subName, 'eyetracking', f'{subName}_task-mgs-iisess_forSource.mat')
    if not os.path.exists(behav_path): return None
    try:
        tempPath = f'/tmp/{subName}_behav_{int(time.time())}.mat'
        copyfile(behav_path, tempPath)
        with h5py.File(tempPath, 'r') as f:
            mask = ~np.isnan(np.array(f['ii_sess_forSource']['i_sacc_err']).flatten())
        if os.path.exists(tempPath): os.remove(tempPath)
        return mask
    except: return None

def load_raw_source_data(subjID, bidsRoot, voxRes, behavioral_mask=None):
    """Load raw source-space data (trials, time, sources)"""
    subName = 'sub-%02d' % subjID
    fpath = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'{subName}_task-mgs_sourceSpaceData_raw_{voxRes}.mat')
    
    tempPath = f'/tmp/{subName}_raw_{int(time.time())}.mat'
    print(f"Staging raw data to {tempPath}...")
    copyfile(fpath, tempPath)
    
    with h5py.File(tempPath, 'r', locking=False) as f:
        group = f['sourcedataCombined']
        time_v = np.array(f[group['time'][0, 0]]).flatten()
        trialinfo = np.array(group['trialinfo']).T
        target_labels = trialinfo[:, 1]
        
        trial_data = group['trial']
        all_trials = []
        for i in range(trial_data.shape[0]):
            all_trials.append(np.array(f[trial_data[i, 0]]))
        
        data_matrix = np.stack(all_trials, axis=0) # (trials, times, sources)
        
    if os.path.exists(tempPath): os.remove(tempPath)
    
    if behavioral_mask is not None:
        valid = behavioral_mask[:data_matrix.shape[0]]
        data_matrix = data_matrix[valid, :, :]
        target_labels = target_labels[valid]
        
    return data_matrix, time_v, target_labels

def _process_target_batch(batch_indices, data_matrix, fs, freq_seed, freq_target, is_cross_freq):
    """Filter target voxels and compute phase/envelope phase."""
    # data_matrix: (trials, times, n_batch_sources)
    if not is_cross_freq:
        # Same-band: Just extract phase
        return extract_phase(data_matrix, fs, freq_seed[0], freq_seed[1])
    else:
        # CFC: Extract Gamma Envelope then Theta Phase of that Envelope
        # 1. Filter Target at High Freq (Gamma)
        target_high = apply_bandpass(data_matrix, freq_target[0], freq_target[1], fs)
        # 2. Extract Envelope
        envelope = np.abs(hilbert(target_high, axis=1))
        # 3. Filter Envelope at Low Freq (Theta)
        return extract_phase(envelope, fs, freq_seed[0], freq_seed[1])

def main(subjID, voxRes, seedROI_name, targetHem, freq_seed_tuple, freq_target_tuple=None):
    """
    Directional CFC/Connectivity Engine.
    freq_seed_tuple: (low, high) e.g. (4, 8)
    freq_target_tuple: (low, high) e.g. (30, 80) for envelope dPLI, or None for same-band.
    """
    is_cross_freq = freq_target_tuple is not None
    h = socket.gethostname()
    bidsRoot = '/d/DATD/datd/MEG_MGS/MEG_BIDS' if h == 'vader' else '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    mask = load_behavioral_mask(subjID, bidsRoot)
    data, time_v, target_labels = load_raw_source_data(subjID, bidsRoot, voxRes, mask)
    fs = 1.0 / np.mean(np.diff(time_v))
    
    atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    seed_pts = np.where(atlas[f'{seedROI_name}_points'].flatten() == 1)[0]
    
    # Process Seed (ROI average)
    seed_raw = np.mean(data[:, :, seed_pts], axis=2) # (trials, times)
    seed_phase = extract_phase(seed_raw, fs, freq_seed_tuple[0], freq_seed_tuple[1]) # (trials, times)
    
    # Process Targets in batches for parallel efficiency
    batch_size = 50
    n_sources = data.shape[2]
    n_batches = int(np.ceil(n_sources / batch_size))
    _, n_cores = get_compute_profile()
    
    print(f"Processing {n_sources} targets in {n_batches} batches...")
    
    target_phases = Parallel(n_jobs=n_cores)(
        delayed(_process_target_batch)(
            np.arange(i*batch_size, min((i+1)*batch_size, n_sources)),
            data[:, :, i*batch_size : min((i+1)*batch_size, n_sources)],
            fs, freq_seed_tuple, freq_target_tuple, is_cross_freq
        )
        for i in range(n_batches)
    )
    all_target_phases = np.concatenate(target_phases, axis=2) # (trials, times, sources)
    
    # Compute dPLI
    # seed_phase: (trials, times) -> (trials, times, 1)
    # all_target_phases: (trials, times, sources)
    phase_diff = seed_phase[:, :, np.newaxis] - all_target_phases
    dpli = np.mean(np.heaviside(np.sin(phase_diff), 0.5), axis=0) # (times, sources)
    
    # Smooth (Optional but recommended for dPLI)
    dt = 1.0/fs
    window_s = 1.25 if freq_seed_tuple[0] < 8 else 0.5
    window_samp = int(window_s / dt)
    dpli_smooth = uniform_filter1d(dpli, size=window_samp, axis=0)
    
    # Save results centrally for Left/Right targets
    for loc in ['left', 'right']:
        targets = [4, 5, 6, 7, 8] if loc == 'left' else [1, 2, 3, 9, 10]
        mask_loc = np.isin(target_labels, targets)
        loc_dpli = np.mean(dpli_smooth[:, :], axis=0) # Need to filter by trials too? 
        # Wait, dPLI is already averaged over trials. 
        # For ROI analysis, we usually want dPLI for specific stimulus conditions.
        
        # Split dPLI by stimulus location for the visualizer
        loc_trials_dpli = np.mean(np.heaviside(np.sin(phase_diff[mask_loc, :, :]), 0.5), axis=0)
        loc_smooth = uniform_filter1d(loc_trials_dpli, size=window_samp, axis=0) - 0.5 # Centered
        
        mode_str = f"{freq_seed_tuple[0]}-{freq_seed_tuple[1]}"
        if is_cross_freq: mode_str += f"_to_{freq_target_tuple[0]}-{freq_target_tuple[1]}_AM"
        
        outDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'connectivity_{voxRes}')
        os.makedirs(outDir, exist_ok=True)
        outF = os.path.join(outDir, f'sub-{subjID:02d}_task-mgs_directionalConn_{voxRes}_{seedROI_name}_{loc}_dpli_{mode_str}.pkl')
        with open(outF, 'wb') as f: pickle.dump(loc_smooth.T, f)
        print(f"Saved: {outF}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 7:
        print("Usage: python inSourceSpacedirectionalConnectivity.py <subID> <res> <seedROI> <lowF_low> <lowF_high> [highF_low] [highF_high]")
        sys.exit(1)
    
    sID = int(sys.argv[1])
    res = sys.argv[2]
    seed = sys.argv[3]
    fL = (float(sys.argv[4]), float(sys.argv[5]))
    fH = (float(sys.argv[6]), float(sys.argv[7])) if len(sys.argv) > 7 else None
    
    main(sID, res, seed, None, fL, fH)
