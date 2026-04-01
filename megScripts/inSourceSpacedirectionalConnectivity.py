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
    # Ensure frequencies are within valid range (0 < Wn < 1)
    # Clip to 95% of Nyquist to avoid edge artifacts
    safe_high = min(highcut, 0.95 * nyq)
    safe_low = max(0.5, lowcut) # Avoid DC / very low freq artifacts
    
    low = safe_low / nyq
    high = safe_high / nyq
    
    if low >= high:
        # Fallback for very narrow/invalid bands (e.g. if lowcut > nyq)
        low = 0.1
        high = 0.9
        
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut, highcut, fs, axis=1):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data, axis=axis)

def extract_phase(data, fs, lowcut, highcut):
    """Filter and extract phase via Hilbert."""
    filtered = apply_bandpass(data, lowcut, highcut, fs)
    return np.angle(hilbert(filtered, axis=1))

def load_behavioral_mask(subjID, bidsRoot):
    subName = 'sub-%02d' % subjID
    behav_path = os.path.join(bidsRoot, 'derivatives', subName, 'eyetracking', f'{subName}_task-mgs-iisess_forSource.mat')
    if not os.path.exists(behav_path): return None
    try:
        h = socket.gethostname()
        tempDir = '/Users/mrugank/Desktop' if h == 'zod' else '/tmp'
        tempPath = os.path.join(tempDir, f'{subName}_behav_cfc_{int(time.time())}.mat')
        copyfile(behav_path, tempPath)
        with h5py.File(tempPath, 'r') as f:
            mask = ~np.isnan(np.array(f['ii_sess_forSource']['i_sacc_err']).flatten())
        if os.path.exists(tempPath): os.remove(tempPath)
        return mask
    except: return None

def load_raw_source_data(subjID, bidsRoot, voxRes, behavioral_mask=None):
    """Robust loader with atomic staging."""
    subName = 'sub-%02d' % subjID
    v_code = voxRes.replace('mm', '')
    fpath = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'{subName}_task-mgs_sourceSpaceData_{v_code}.mat')
    
    h = socket.gethostname()
    tempDir = '/Users/mrugank/Desktop' if h == 'zod' else '/d/DATD/hyper/experiments/Mrugank/meg_mgs'
    # Use a fixed staged name for this sub/res to allow reuse during batch
    tempPath = os.path.join(tempDir, f'{subName}_{voxRes}_STAGED_RAW.mat')
    tempPathPartial = tempPath + ".tmp"
    
    # Check if a finished copy exists
    # If one exists but isn't the right size (e.g. stalled), redo it.
    expected_size = os.path.getsize(fpath)
    if os.path.exists(tempPath) and os.path.getsize(tempPath) < expected_size:
        print(f"Staged file {tempPath} is incomplete. Re-staging...")
        os.remove(tempPath)

    if not os.path.exists(tempPath):
        print(f"Staging raw data to {tempPath} (Atomic 1.7GB, one-time)...")
        if os.path.exists(tempPathPartial): os.remove(tempPathPartial)
        copyfile(fpath, tempPathPartial)
        os.rename(tempPathPartial, tempPath)
    else:
        print(f"Reusing fully-staged data: {tempPath}")
    
    with h5py.File(tempPath, 'r', locking=False) as f:
        group = f['sourcedataCombined'] if 'sourcedataCombined' in f else f['sourcedata']
        time_v = np.array(f[group['time'][0, 0]]).flatten()
        trial_data = group['trial']
        
        ds_ti = group['trialinfo']
        if isinstance(ds_ti[0, 0], h5py.Reference):
            trialinfo = np.array(f[ds_ti[0, 0]])
        else:
            trialinfo = np.array(ds_ti)
        if trialinfo.shape[0] < trialinfo.shape[1]: trialinfo = trialinfo.T
        target_labels = trialinfo[:, 1]
        
        all_trials = []
        for i in range(trial_data.shape[0]):
            all_trials.append(np.array(f[trial_data[i, 0]]))
        data_matrix = np.stack(all_trials, axis=0) 
        
    return data_matrix, time_v, target_labels

def _process_target_batch(data_matrix, fs, freq_seed, freq_target):
    """Extract Envelope-Phase for CFC."""
    target_high = apply_bandpass(data_matrix, freq_target[0], freq_target[1], fs)
    envelope = np.abs(hilbert(target_high, axis=1))
    return extract_phase(envelope, fs, freq_seed[0], freq_seed[1])

def main(subjID, voxRes, seedROI_name, loc_str, f_seed, f_env):
    """Modular CFC Engine (Seed A to Target B)"""
    h = socket.gethostname()
    bidsRoot = '/d/DATD/datd/MEG_MGS/MEG_BIDS' if h == 'vader' else '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    mask = load_behavioral_mask(subjID, bidsRoot)
    data, time_v, target_labels = load_raw_source_data(subjID, bidsRoot, voxRes, mask)
    fs = 1.0 / np.abs(np.mean(np.diff(time_v)))
    
    atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    seed_pts = np.where(atlas[f'{seedROI_name}_points'].flatten() == 1)[0]
    
    # 1. Seed Phase (Theta)
    seed_raw = np.mean(data[:, :, seed_pts], axis=2)
    seed_phase = extract_phase(seed_raw, fs, f_seed[0], f_seed[1])
    
    # 2. Filter Targets by location
    targets_req = [4, 5, 6, 7, 8] if loc_str == 'left' else [1, 2, 3, 9, 10]
    mask_loc = np.isin(target_labels, targets_req)
    if np.sum(mask_loc) == 0: 
        print(f"  [!] No trials for {loc_str}. Skipping.")
        return
        
    data_loc = data[mask_loc, :, :]
    seed_p_loc = seed_phase[mask_loc, :]
    
    # 3. Parallel extraction of target AM phases
    batch_size = 50
    n_sources = data.shape[2]
    n_batches = int(np.ceil(n_sources / batch_size))
    _, n_cores = get_compute_profile()
    
    print(f"Computing CFC for {seedROI_name} ({f_seed}Hz) -> {loc_str} ({f_env}Hz AM burts) | {np.sum(mask_loc)} trials")
    t_env_phases = Parallel(n_jobs=n_cores)(
        delayed(_process_target_batch)(data_loc[:, :, i*batch_size : min((i+1)*batch_size, n_sources)], fs, f_seed, f_env)
        for i in range(n_batches)
    )
    all_t_phases = np.concatenate(t_env_phases, axis=2)
    
    # 4. Final dPLI
    # (trials, times, targets)
    phase_diff = seed_p_loc[:, :, np.newaxis] - all_t_phases
    dpli = np.mean(np.heaviside(np.sin(phase_diff), 0.5), axis=0) - 0.5
    
    # Smooth
    window_samp = int(1.25 / (1.0/fs)) if f_seed[0] < 8 else int(0.5 / (1.0/fs))
    dpli_smooth = uniform_filter1d(dpli, size=window_samp, axis=0)
    
    # Save to CFC_8mm
    outDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'CFC_{voxRes}')
    os.makedirs(outDir, exist_ok=True)
    mode_str = f"phase{int(f_seed[0])}-{int(f_seed[1])}_to_env{int(f_env[0])}-{int(f_env[1])}"
    outF = os.path.join(outDir, f'sub-{subjID:02d}_task-mgs_dCFC_{voxRes}_{seedROI_name}_{loc_str}_dpli_{mode_str}.pkl')
    
    with open(outF, 'wb') as f: pickle.dump(dpli_smooth.T, f)
    print(f"  Saved: {outF}")

if __name__ == "__main__":
    import sys
    sID = int(sys.argv[1]); res = sys.argv[2]
    sROI = sys.argv[3]; tLoc = sys.argv[4]
    fS = (float(sys.argv[5]), float(sys.argv[6]))
    fE = (float(sys.argv[7]), float(sys.argv[8]))
    main(sID, res, sROI, tLoc, fS, fE)
