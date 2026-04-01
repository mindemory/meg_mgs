import os, h5py, socket, gc
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import numpy as np
from shutil import copyfile
import pickle
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d
import time
import smtplib
from joblib import Parallel, delayed

def get_compute_profile():
    h = socket.gethostname()
    if h == 'zod':
        return 'mac', 4
    elif h == 'vader':
        return 'vader', 48       # Cap at 48 for memory safety
    else:
        return 'hpc', 10

NOTIFY_EMAIL = 'mrugank.dake@nyu.edu'

def send_completion_email(subjID, voxRes, connectivityType, success=True, error_msg=None):
    try:
        hostname = socket.gethostname()
        msg = f"Subject: [Connectivity] sub-{subjID:02d} {voxRes} {connectivityType} {'DONE' if success else 'FAILED'}\n\n"
        msg += f"Subject: {subjID:02d}\nMetric: {connectivityType}\n{'Error: ' + error_msg if error_msg else 'Success!'}"
        with smtplib.SMTP('localhost') as s:
            s.sendmail(NOTIFY_EMAIL, NOTIFY_EMAIL, msg)
        print(f'  Notification email sent to {NOTIFY_EMAIL}')
    except Exception as e:
        print(f'  (Email notification skipped: {e})')

def load_behavioral_mask(subjID, bidsRoot, taskName='mgs'):
    """Load behavioral eye-tracking mask (i_sacc_err) to filter bad trials."""
    subName = 'sub-%02d' % subjID
    behav_path = os.path.join(bidsRoot, 'derivatives', subName, 'eyetracking', f'{subName}_task-{taskName}-iisess_forSource.mat')
    
    if not os.path.exists(behav_path):
        print(f"  [!] Behavioral data not found at {behav_path}. Proceeding without filter.")
        return None
    
    try:
        # Avoid NFS locking issues
        h = socket.gethostname()
        tempDir = '/Users/mrugank/Desktop' if h == 'zod' else '/tmp'
        tempPath = os.path.join(tempDir, f'{subName}_behav_{int(time.time())}.mat')
        copyfile(behav_path, tempPath)
        
        with h5py.File(tempPath, 'r') as f:
            gr = f['ii_sess_forSource']
            # i_sacc_err: NaN indicates a bad trial (no primary saccade detected)
            mask = ~np.isnan(np.array(gr['i_sacc_err']).flatten())
            
        if os.path.exists(tempPath): os.remove(tempPath)
        return mask
    except Exception as e:
        print(f"  [!] Error loading behavioral mask: {e}")
        return None

def load_source_space_data(subjID, bidsRoot, taskName, voxRes, freqBand, behavioral_mask=None):
    """Load all 10 targets from HDF5 into a dictionary of trials, applying behavioral filtering."""
    subName = 'sub-%02d' % subjID
    print(f'Loading source space data for {subName} ({freqBand})')
    
    freqSpace_fpath = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', 'freqSpace', 
                                   f'{subName}_task-{taskName}_complex{freqBand}_allTargets_{voxRes[:-2]}.mat')
    
    if not os.path.exists(freqSpace_fpath):
        raise FileNotFoundError(f"Source file not found: {freqSpace_fpath}")

    h = socket.gethostname()
    tempDir = '/Users/mrugank/Desktop' if h == 'zod' else '/d/DATD/hyper/experiments/Mrugank/meg_mgs'
    tempPath = os.path.join(tempDir, f'{subName}_complex{freqBand}_{int(time.time())}_{np.random.randint(1000)}.mat')
    
    print(f'  Staging {freqBand} to {tempPath}...')
    copyfile(freqSpace_fpath, tempPath)
    
    target_data_dict = {}
    time_vector = None
    total_dropped = 0
    total_kept = 0
    
    with h5py.File(tempPath, 'r', locking=False) as f:
        source_refs = np.array(f['sourceDataByTarget'])
        for target_idx in range(1, 11):
            ref = source_refs[0, target_idx - 1]
            group = f[ref]
            trial_ds = group['trial']
            
            # Extract trialinfo to map local trials to global behavioral indices
            ti_ref = group['trialinfo'][0, 0]
            trial_info = np.array(f[ti_ref])
            
            if time_vector is None:
                time_vector = np.array(f[group['time'][0, 0]])
                
            trials = []
            for tr_idx in range(trial_ds.shape[0]):
                global_idx = int(trial_info[tr_idx, 0]) - 1 # 0-indexed
                
                if behavioral_mask is not None:
                    if global_idx >= len(behavioral_mask) or not behavioral_mask[global_idx]:
                        total_dropped += 1
                        continue
                
                tr_ref = trial_ds[tr_idx, 0]
                arr = np.array(f[tr_ref])
                trials.append(arr['real'] + 1j * arr['imag'])
                total_kept += 1
            
            if trials:
                target_data_dict[target_idx] = np.stack(trials, axis=0)

    if behavioral_mask is not None:
        print(f"  Behavioral Filter: Kept {total_kept}, Dropped {total_dropped} trials.")
        
    if os.path.exists(tempPath): os.remove(tempPath)
    return target_data_dict, time_vector

def _compute_source_batch(batch_idx, batch_size, n_sources, n_timepoints, window_size_samples, 
                          data_matrix_seed, data_matrix_target, seed_indices, connectivityType):
    """Worker function for a batch of target sources (Vectorized-Across-Time)"""
    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, n_sources)
    batch_indices = np.arange(batch_start, batch_end)
    
    target_data = data_matrix_target[:, :, batch_indices]
    seed_data = data_matrix_seed[:, :, seed_indices].transpose(2, 0, 1)
    
    if connectivityType == 'dpli':
        cross_complex = seed_data[:, :, :, np.newaxis] * np.conj(target_data[np.newaxis, :, :, :])
        heaviside_trials = np.heaviside(np.imag(cross_complex), 0.5)
        conn_over_time = np.mean(heaviside_trials, axis=1)
    else:
        cross_spec = np.mean(seed_data[:, :, :, np.newaxis] * np.conj(target_data[np.newaxis, :, :, :]), axis=1)
        seed_pow = np.mean(seed_data * np.conj(seed_data), axis=1)
        target_pow = np.mean(target_data * np.conj(target_data), axis=0)
        
        if connectivityType == 'coh':
            conn_over_time = np.abs(cross_spec)**2 / (seed_pow[:, :, np.newaxis] * target_pow[np.newaxis, :, :] + 1e-10)
        elif connectivityType == 'imcoh':
            conn_over_time = np.abs(np.imag(cross_spec)) / np.sqrt(seed_pow[:, :, np.newaxis] * target_pow[np.newaxis, :, :] + 1e-10)
    
    seed_avg_conn = np.mean(conn_over_time, axis=0)
    batch_final = uniform_filter1d(seed_avg_conn, size=window_size_samples, axis=0, mode='constant', cval=0.0)
    return batch_final.T, batch_indices

def compute_connectivity_measures(seed_indices, data_matrix_seed, data_matrix_target, 
                                  time_vector, connectivityType, freqBand, batch_size=20):
    """Parallelized whole-brain connectivity (Parallel-Over-Sources, Vectorized-Over-Time)"""
    _, n_timepoints, n_sources = data_matrix_target.shape
    n_seeds = len(seed_indices)
    
    dt = np.mean(np.diff(time_vector.flatten()))
    window_sizes = {'theta': 1.25, 'alpha': 0.77, 'beta': 0.40, 'lowgamma': 0.20}
    window_samples = int(window_sizes.get(freqBand, 0.5) / dt)
    n_batches = int(np.ceil(n_sources / batch_size))
    
    _, n_cores = get_compute_profile()
    print(f"Computing {connectivityType} ({freqBand}) for {n_seeds} seeds -> {n_sources} sources.")

    results = Parallel(n_jobs=n_cores, backend='threading', verbose=5)(
        delayed(_compute_source_batch)(
            batch_idx, batch_size, n_sources, n_timepoints, window_samples, 
            data_matrix_seed, data_matrix_target, seed_indices, connectivityType
        )
        for batch_idx in range(n_batches)
    )

    connectivity_timeseries = np.empty((n_sources, n_timepoints))
    for batch_res, indices in results:
        connectivity_timeseries[indices, :] = batch_res
    
    return connectivity_timeseries

def main(subjID, voxRes, seedROI_str, targetLoc_str, connectivityType_str, freqBand_seed, freqBand_target=None):
    """Main loop with multi-seed, multi-location, and behavioral validation"""
    voxRes = voxRes or '10mm'
    freqBand_target = freqBand_target or freqBand_seed
    
    seedROIs = seedROI_str.split(',') if seedROI_str else ['left_visual']
    targetLocs = targetLoc_str.split(',') if targetLoc_str else ['left']
    metrics = connectivityType_str.split(',') if connectivityType_str else ['coh']
    
    import socket
    h = socket.gethostname()
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if h == 'zod' else '/d/DATD/datd/MEG_MGS/MEG_BIDS' if h == 'vader' else '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    behav_mask = load_behavioral_mask(subjID, bidsRoot)
    atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    
    seed_data_all, time_v = load_source_space_data(subjID, bidsRoot, 'mgs', voxRes, freqBand_seed, behavioral_mask=behav_mask)
    if freqBand_target == freqBand_seed:
        target_data_all = seed_data_all
    else:
        target_data_all, _ = load_source_space_data(subjID, bidsRoot, 'mgs', voxRes, freqBand_target, behavioral_mask=behav_mask)
    
    results_summary = []
    for seedROI in seedROIs:
        print(f"\n--- SEED ROI: {seedROI} ---")
        roi_indices = np.where(np.array(atlas[f'{seedROI}_points']).flatten() == 1)[0]
        
        for loc in targetLocs:
            targets = [4, 5, 6, 7, 8] if loc == 'left' else [1, 2, 3, 9, 10]
            data_subset_seed = np.concatenate([seed_data_all[t] for t in targets], axis=0)
            if freqBand_target == freqBand_seed:
                data_subset_target = data_subset_seed
            else:
                data_subset_target = np.concatenate([target_data_all[t] for t in targets], axis=0)
            
            for metric in metrics:
                outDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'connectivity_{voxRes}')
                os.makedirs(outDir, exist_ok=True)
                
                band_str = f"{freqBand_seed}" if freqBand_seed == freqBand_target else f"{freqBand_seed}-{freqBand_target}"
                outF = os.path.join(outDir, f'sub-{subjID:02d}_task-mgs_seededConnectivity_{voxRes}_{seedROI}_{loc}_{metric}_{band_str}.pkl')
                
                if not os.path.exists(outF):
                    print(f"  -> Path: {loc} | {metric} | {band_str}")
                    start_t = time.time()
                    res = compute_connectivity_measures(roi_indices, data_subset_seed, data_subset_target, 
                                                       time_v, metric, freqBand_target)
                    with open(outF, 'wb') as f: pickle.dump(res, f)
                    dur = time.time() - start_t
                    results_summary.append(f"{seedROI}/{loc}/{metric}/{band_str}: Done ({dur:.1f}s)")
                else:
                    results_summary.append(f"{seedROI}/{loc}/{metric}/{band_str}: Skip (exists)")

    summary_txt = f"Subject {subjID:02d} ({freqBand_seed}->{freqBand_target}) Summary:\n" + "\n".join(results_summary)
    print("\n" + "="*60 + "\n" + summary_txt + "\n" + "="*60)

if __name__ == '__main__':
    if len(os.sys.argv) < 2: 
        print("Usage: python inSourceSpaceSeededConnectivity.py <subjID> <voxRes> <seedROI> <targetLocs_CSV> <metrics_CSV> <freqBandSeed> [freqBandTarget]")
        os.sys.exit(1)
    
    sID = int(os.sys.argv[1]); vRes = os.sys.argv[2] if len(os.sys.argv) > 2 else '10mm'
    sROI = os.sys.argv[3] if len(os.sys.argv) > 3 else 'left_visual'; tLocs = os.sys.argv[4] if len(os.sys.argv) > 4 else 'left'
    cTypes = os.sys.argv[5] if len(os.sys.argv) > 5 else 'coh'; fSeed = os.sys.argv[6] if len(os.sys.argv) > 6 else 'theta'
    fTarg = os.sys.argv[7] if len(os.sys.argv) > 7 else fSeed
    
    main(sID, vRes, sROI, tLocs, cTypes, fSeed, fTarg)
