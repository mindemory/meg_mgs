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

def extract_complex_signal(data, fs, lowcut, highcut):
    """Filter and extract complex analytical signal via Hilbert."""
    filtered = apply_bandpass(data, lowcut, highcut, fs)
    return hilbert(filtered, axis=1)

def load_behavioral_mask(subjID, bidsRoot):
    subName = 'sub-%02d' % subjID
    behav_path = os.path.join(bidsRoot, 'derivatives', subName, 'eyetracking', f'{subName}_task-mgs-iisess_forSource.mat')
    if not os.path.exists(behav_path): return None
    try:
        # Use /tmp for Vader or Desktop for Zod to avoid lock issues
        h = socket.gethostname()
        tempDir = '/Users/mrugank/Desktop' if h == 'zod' else '/tmp'
        tempPath = os.path.join(tempDir, f'{subName}_behav_{int(time.time())}.mat')
        copyfile(behav_path, tempPath)
        with h5py.File(tempPath, 'r') as f:
            mask = ~np.isnan(np.array(f['ii_sess_forSource']['i_sacc_err']).flatten())
        if os.path.exists(tempPath): os.remove(tempPath)
        return mask
    except: return None

def load_raw_source_data(subjID, bidsRoot, voxRes, behavioral_mask=None):
    """Load raw source-space data (chronological session order)"""
    subName = 'sub-%02d' % subjID
    # Filenames on Vader use _8.mat and _10.mat
    v_code = voxRes.replace('mm', '')
    fpath = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'{subName}_task-mgs_sourceSpaceData_{v_code}.mat')
    
    h = socket.gethostname()
    tempDir = '/Users/mrugank/Desktop' if h == 'zod' else '/d/DATD/hyper/experiments/Mrugank/meg_mgs'
    tempPath = os.path.join(tempDir, f'{subName}_raw_seeded_{int(time.time())}.mat')
    print(f"Staging raw data to {tempPath}...")
    copyfile(fpath, tempPath)
    
    with h5py.File(tempPath, 'r', locking=False) as f:
        # Check if it's sourcedataCombined or just sourcedata
        if 'sourcedataCombined' in f:
            group = f['sourcedataCombined']
        else:
            # Fallback for some versions of the source recon
            group = f['sourcedata']
            
        time_v = np.array(f[group['time'][0, 0]]).flatten()
        trial_data = group['trial']
        
        # trialinfo handling (Direct vs Reference)
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
        data_matrix = np.stack(all_trials, axis=0) # (trials, times, sources)
        
    if os.path.exists(tempPath) and 'Desktop' not in tempPath: os.remove(tempPath)
    
    if behavioral_mask is not None:
        valid = behavioral_mask[:data_matrix.shape[0]]
        data_matrix = data_matrix[valid, :, :]
        target_labels = target_labels[valid]
        print(f"  Behavioral Filter: Kept {data_matrix.shape[0]} trials.")
        
    return data_matrix, time_v, target_labels

def _process_target_batch(batch_indices, data_matrix, fs, low, high):
    """Filter target voxels to extract complex analytical signal."""
    return extract_complex_signal(data_matrix, fs, low, high)

def main(subjID, voxRes, seedROI_str, targetLoc_str, connectivityType_str, freqBand):
    """Main engine using Raw Time-Domain data for 100% accurate behavioral mapping."""
    h = socket.gethostname()
    bidsRoot = '/d/DATD/datd/MEG_MGS/MEG_BIDS' if h == 'vader' else '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    seedROIs = seedROI_str.split(',') if seedROI_str else ['left_visual']
    targetLocs = targetLoc_str.split(',') if targetLoc_str else ['left']
    metrics = connectivityType_str.split(',') if connectivityType_str else ['dpli']
    # Pre-emptively check if all target files exist to skip expensive 1.7GB load/filter
    all_exist = True
    out_dir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'connectivity_{voxRes}')
    for seed in seedROIs:
        for loc in targetLocs:
            for metric in metrics:
                outF = os.path.join(out_dir, f'sub-{subjID:02d}_task-mgs_seededConnectivity_{voxRes}_{seed}_{loc}_{metric}_{freqBand}.pkl')
                if not os.path.exists(outF):
                    all_exist = False
                    break
            if not all_exist: break
        if not all_exist: break
    
    # FORCE RE-RUN: Skipping check to ensure new integrated window logic is applied
    # if all_exist:
    #     print(f"\n[SKIP] All {freqBand.upper()} connectivity results already exist for Sub-{subjID:02d}. Exiting cleanly.", flush=True)
    #     return

    # 1. Load Raw Chronological Data
    mask = load_behavioral_mask(subjID, bidsRoot)
    data, time_v, target_labels = load_raw_source_data(subjID, bidsRoot, voxRes, mask)
    fs = 1.0 / np.mean(np.diff(time_v))
    
    band_defs = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 18), 'lowgamma': (30, 55)}
    f_low, f_high = band_defs.get(freqBand, (8, 13))
    
    atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    
    # 2. Extract Complex Phase/Amplitude once per band
    batch_size = 50
    n_sources = data.shape[2]
    n_batches = int(np.ceil(n_sources / batch_size))
    _, n_cores = get_compute_profile()
    
    print(f"Filtering {n_sources} targets for {freqBand.upper()}...")
    complex_results = Parallel(n_jobs=n_cores)(
        delayed(_process_target_batch)(None, data[:, :, i*batch_size : min((i+1)*batch_size, n_sources)], fs, f_low, f_high)
        for i in range(n_batches)
    )
    all_target_complex = np.concatenate(complex_results, axis=2) # (trials, times, sources)
    
    for seedROI in seedROIs:
        print(f"\n--- SEED ROI: {seedROI} ---")
        roi_pts = np.where(atlas[f'{seedROI}_points'].flatten() == 1)[0]
        # Average complex signal of Seed ROI
        seed_complex = np.mean(all_target_complex[:, :, roi_pts], axis=2) # (trials, times)
        
        for loc in targetLocs:
            targets = [4, 5, 6, 7, 8] if loc == 'left' else [1, 2, 3, 9, 10]
            mask_loc = np.isin(target_labels, targets)
            if np.sum(mask_loc) == 0: continue
            
            s_comp_loc = seed_complex[mask_loc, :]
            t_comp_loc = all_target_complex[mask_loc, :, :]
            
            for metric in metrics:
                outF = os.path.join(out_dir, f'sub-{subjID:02d}_task-mgs_seededConnectivity_{voxRes}_{seedROI}_{loc}_{metric}_{freqBand}.pkl')
                
                print(f"  Computing {metric} (Integrated) for {loc} targets ({np.sum(mask_loc)} trials)...")
                
                # 1. Point-wise trial-averaged quantities
                cross_spec = np.mean(s_comp_loc[:, :, np.newaxis] * np.conj(t_comp_loc), axis=0) # (times, sources)
                seed_pow = np.mean(np.abs(s_comp_loc)**2, axis=0) # (times,)
                targ_pow = np.mean(np.abs(t_comp_loc)**2, axis=0) # (times, sources)
                
                # 2. Legacy-style sliding window integration (±100ms)
                # This performs the averaging BEFORE division for maximal stability
                window_samples = int(0.1 * fs) * 2 + 1 
                
                if metric == 'dpli':
                    # For dPLI, we integrate the phase-lead probability across trials and time
                    pt_dpli = np.mean(np.heaviside(np.imag(s_comp_loc[:, :, np.newaxis] * np.conj(t_comp_loc)), 0.5), axis=0)
                    conn = uniform_filter1d(pt_dpli, size=window_samples, axis=0)
                else:
                    # For ImCoh, we integrate Cross-Spec and Power before dividing
                    cs_sm = uniform_filter1d(cross_spec, size=window_samples, axis=0)
                    sp_sm = uniform_filter1d(seed_pow, size=window_samples, axis=0)
                    tp_sm = uniform_filter1d(targ_pow, size=window_samples, axis=0)
                    conn = np.abs(np.imag(cs_sm)) / np.sqrt(sp_sm[:, np.newaxis] * tp_sm + 1e-10)
                
                # Save as (sources, times) to match visualization expectations
                with open(outF, 'wb') as f: pickle.dump(conn.T, f)

if __name__ == "__main__":
    import sys
    sID = int(sys.argv[1]); vRes = sys.argv[2]
    sROI = sys.argv[3]; tLocs = sys.argv[4]
    cTypes = sys.argv[5]; fBand = sys.argv[6]
    main(sID, vRes, sROI, tLocs, cTypes, fBand)
