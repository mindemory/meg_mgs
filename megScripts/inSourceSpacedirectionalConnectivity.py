import os, socket, pickle, shutil
import numpy as np
from scipy.io import loadmat
from scipy.signal import hilbert, butter, filtfilt
from scipy.ndimage import uniform_filter1d
import h5py

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    safe_high = min(highcut, nyq * 0.95)
    low = lowcut / nyq
    high = safe_high / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut, highcut, fs, axis=1):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data, axis=axis)

def extract_phase(data, fs, f_low, f_high):
    filt_data = apply_bandpass(data, f_low, f_high, fs)
    return np.angle(hilbert(filt_data, axis=1))

def load_behavioral_mask(subjID, bidsRoot):
    import pandas as pd
    behavF = os.path.join(bidsRoot, 'derivatives', 'behavioral', f'sub-{subjID:02d}', f'sub-{subjID:02d}_task-mgs_behavior.tsv')
    if not os.path.exists(behavF): return None
    df = pd.read_csv(behavF, sep='\t')
    return (df['i_sacc_err'] == 0).values

def load_raw_source_data(subjID, bidsRoot, voxRes, behavioral_mask=None):
    """Refactored to mirror compute_pac.py loading strategy.
    Uses _sourceSpaceData_{vox}.mat and direct read with locking=False.
    """
    subName = f'sub-{subjID:02d}'
    res_num = voxRes[:-2] # '8mm' -> '8'
    rawPath = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'{subName}_task-mgs_sourceSpaceData_{res_num}.mat')
    
    # Standard loading logic from compute_pac.py
    source_data = None
    try:
        # Try direct read with locking=False first (Vader/Greene standard)
        source_data = h5py.File(rawPath, 'r', locking=False)
        print(f"  [+] Direct load successful: {rawPath}", flush=True)
    except Exception:
        # Fallback to transient staging (Mac Mac local or network mount error)
        tempPath = os.path.join('/Users/mrugank/Desktop' if socket.gethostname() == 'zod' else '/tmp', f'{subName}_STAGING_TEMP_{res_num}.mat')
        print(f"  [!] Direct load failed. Performing transient stage to {tempPath}...", flush=True)
        shutil.copyfile(rawPath, tempPath)
        source_data = h5py.File(tempPath, 'r')
        # On Unix, removing the file while it's open works (descriptor stays valid)
        os.remove(tempPath) 

    group = source_data['sourcedataCombined'] if 'sourcedataCombined' in source_data else source_data['sourcedata']
    time_v = np.array(source_data[group['time'][0, 0]]).flatten()
    trial_data = group['trial']
    
    # Handle nested trialinfo references
    ds_ti = group['trialinfo']
    if isinstance(ds_ti[0, 0], h5py.Reference):
        trialinfo = np.array(source_data[ds_ti[0, 0]])
    else:
        trialinfo = np.array(ds_ti)
    if trialinfo.shape[0] < trialinfo.shape[1]: trialinfo = trialinfo.T
    target_labels = trialinfo[:, 1]
    
    all_trials = []
    for i in range(trial_data.shape[0]):
        all_trials.append(np.array(source_data[trial_data[i, 0]]))
    data_matrix = np.stack(all_trials, axis=0) 
    
    source_data.close()
        
    if behavioral_mask is not None:
        valid = behavioral_mask[:data_matrix.shape[0]]
        data_matrix = data_matrix[valid, :, :]
        target_labels = target_labels[valid]
        print(f"  Applied behavioral filter: {np.sum(valid)} trials kept.", flush=True)
        
    return data_matrix, time_v, target_labels

def main(subjID, voxRes):
    """Modular CFC Engine (Complete Hierarchy)
    Computes Directed Phase-Amplitude Coupling for all ROI permutations.
    """
    import warnings
    warnings.simplefilter("ignore")
    
    h = socket.gethostname()
    bidsRoot = '/d/DATD/datd/MEG_MGS/MEG_BIDS' if 'vader' in h else '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    mask = load_behavioral_mask(subjID, bidsRoot)
    print(f"[*] Sub-{subjID:02d} | Resolving Raw Dataset (The 'PAC Way')...", flush=True)
    data, time_v, target_labels = load_raw_source_data(subjID, bidsRoot, voxRes, mask)
    fs = 1.0 / np.abs(np.mean(np.diff(time_v)))
    
    atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    
    label_map = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 18), 'lowgamma': (30, 55)}
    bands = ['theta', 'alpha', 'beta']
    rois = ['left_frontal', 'right_frontal', 'left_visual', 'right_visual']
    
    outDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'CFC_{voxRes}')
    os.makedirs(outDir, exist_ok=True)
    
    print(f"[*] Sub-{subjID:02d} | Computing Complete 108-Pair Directed Hierarchy...", flush=True)
    total_matrix_pairs = 108
    pair_count = 0
    
    for seed_ROI in rois:
        s_pts = np.where(atlas[f'{seed_ROI}_points'].flatten() == 1)[0]
        s_raw = np.mean(data[:, :, s_pts], axis=2)
        for target_ROI in rois:
            if seed_ROI == target_ROI: continue
            t_pts = np.where(atlas[f'{target_ROI}_points'].flatten() == 1)[0]
            t_raw = np.mean(data[:, :, t_pts], axis=2)
            for f_s_name in bands:
                f_seed = label_map[f_s_name]; s_phase = extract_phase(s_raw, fs, f_seed[0], f_seed[1])
                for f_e_name in bands:
                    f_env = label_map[f_e_name]
                    t_high = apply_bandpass(t_raw, f_env[0], f_env[1], fs)
                    envelope = np.abs(hilbert(t_high, axis=1))
                    t_phase = extract_phase(envelope, fs, f_seed[0], f_seed[1]) 
                    mode_str = f"{f_s_name}_to_{f_e_name}"
                    for t_loc in ['left', 'right']:
                        req = [4, 5, 6, 7, 8] if t_loc == 'left' else [1, 2, 3, 9, 10]
                        mask_loc = np.isin(target_labels, req)
                        if np.sum(mask_loc) == 0: continue
                        s_p = s_phase[mask_loc, :]; t_p = t_phase[mask_loc, :]
                        dpli = np.mean(np.heaviside(np.sin(s_p - t_p), 0.5), axis=0) - 0.5 
                        window_samp = int(1.25 / (1.0/fs)) if f_seed[0] < 8 else int(0.5 / (1.0/fs))
                        dpli_sm = uniform_filter1d(dpli, size=window_samp, axis=0)
                        outF = os.path.join(outDir, f'sub-{subjID:02d}_task-mgs_dCFC_{voxRes}_{seed_ROI}_to_{target_ROI}_{t_loc}_targets_dpli_{mode_str}.pkl')
                        with open(outF, 'wb') as f: pickle.dump(dpli_sm, f) 
                    pair_count += 1
                    if pair_count % 36 == 0:
                        print(f"  [{pair_count}/{total_matrix_pairs}] Progressing Hierarchy...", flush=True)

    print(f"[+] Sub-{subjID:02d} | Successfully complete.", flush=True)

if __name__ == "__main__":
    import sys
    sID = int(sys.argv[1]); res = sys.argv[2]
    main(sID, res)
