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

def load_source_space_data(subjID, bidsRoot, taskName, voxRes, freqBand):
    """Load all 10 targets from HDF5 into a dictionary of trials."""
    subName = 'sub-%02d' % subjID
    print(f'Loading source space data for {subName} ({freqBand})')
    
    freqSpace_fpath = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', 'freqSpace', f'{subName}_task-{taskName}_complex{freqBand}_allTargets_{voxRes[:-2]}.mat')
    
    h = socket.gethostname()
    tempDir = '/Users/mrugank/Desktop' if h == 'zod' else '/d/DATD/hyper/experiments/Mrugank/meg_mgs'
    
    tempPath = os.path.join(tempDir, f'{subName}_complex{freqBand}_{int(time.time())}.mat')
    print(f'  Staging to {tempPath}...')
    copyfile(freqSpace_fpath, tempPath)
    
    target_data_dict = {}
    time_vector = None
    
    with h5py.File(tempPath, 'r', locking=False) as f:
        source_refs = np.array(f['sourceDataByTarget'])
        for target_idx in range(1, 11):
            ref = source_refs[0, target_idx - 1]
            group = f[ref]
            trial_ds = group['trial']
            
            if time_vector is None:
                time_vector = np.array(f[group['time'][0, 0]])
                
            trials = []
            for tr_idx in range(trial_ds.shape[0]):
                tr_ref = trial_ds[tr_idx, 0]
                tr_data = f[tr_ref]
                trials.append(np.array(tr_data))
            
            # (n_trials, n_times, n_sources)
            arr = np.stack(trials, axis=0)
            target_data_dict[target_idx] = arr['real'] + 1j * arr['imag']
            print(f"  Loaded Target {target_idx}: {target_data_dict[target_idx].shape}")

    if os.path.exists(tempPath): os.remove(tempPath)
    return target_data_dict, time_vector

def _compute_source_batch(batch_idx, batch_size, n_sources, n_timepoints, window_size_samples, data_matrix, seed_indices, connectivityType):
    """Worker function for a batch of target sources (Vectorized-Across-Time)"""
    batch_start = batch_idx * batch_size
    batch_end = min((batch_idx + 1) * batch_size, n_sources)
    batch_indices = np.arange(batch_start, batch_end)
    n_batch_sources = batch_indices.shape[0]
    
    # 1. target_data: (n_trials, n_timepoints, n_batch_sources)
    target_data = data_matrix[:, :, batch_indices]
    
    # 2. seed_data: (seeds, trials, n_timepoints)
    seed_data = data_matrix[:, :, seed_indices].transpose(2, 0, 1)
    
    if connectivityType == 'dpli':
        # (seeds, trials, times, 1) * (1, trials, times, sources) -> Im(cross_trials)
        # We do this in one NumPy broadcasting shot (Uses ~3GB RAM for batch=25)
        cross_complex = seed_data[:, :, :, np.newaxis] * np.conj(target_data[np.newaxis, :, :, :])
        # heaviside_trials: (seeds, trials, times, sources)
        heaviside_trials = np.heaviside(np.imag(cross_complex), 0.5)
        # Average over trials (axis 1)
        # conn_over_time: (seeds, times, sources)
        conn_over_time = np.mean(heaviside_trials, axis=1)
        
    else:
        # Vectorized (Im)Coh math for full trial
        cross_spec = np.mean(seed_data[:, :, :, np.newaxis] * np.conj(target_data[np.newaxis, :, :, :]), axis=1)
        seed_pow = np.mean(seed_data * np.conj(seed_data), axis=1)
        target_pow = np.mean(target_data * np.conj(target_data), axis=0)
        
        if connectivityType == 'coh':
            conn_over_time = np.abs(cross_spec)**2 / (seed_pow[:, :, np.newaxis] * target_pow[np.newaxis, :, :] + 1e-10)
        elif connectivityType == 'imcoh':
            conn_over_time = np.abs(np.imag(cross_spec)) / np.sqrt(seed_pow[:, :, np.newaxis] * target_pow[np.newaxis, :, :] + 1e-10)
    
    # 3. Sliding Window Average across Time (axis 1)
    # Average across seeds (axis 0) first to get (times, sources)
    seed_avg_conn = np.mean(conn_over_time, axis=0)  # (times, sources)
    
    # Apply sliding window mean using uniform_filter1d (Lightning Fast)
    # window_size_samples is the total window width (e.g. 1.25s / dt)
    batch_final = uniform_filter1d(seed_avg_conn, size=window_size_samples, axis=0, mode='constant', cval=0.0)
    
    # Transpose back to (sources, times) as requested by main pipeline
    return batch_final.T, batch_indices

def compute_connectivity_measures(seed_indices, data_matrix, time_vector, connectivityType, freqBand, batch_size=20):
    """Parallelized whole-brain connectivity (Parallel-Over-Sources, Vectorized-Over-Time)"""
    _, n_timepoints, n_sources = data_matrix.shape
    n_seeds = len(seed_indices)
    
    dt = np.mean(np.diff(time_vector.flatten()))
    window_sizes = {'theta': 1.25, 'alpha': 0.77, 'beta': 0.40, 'lowgamma': 0.20}
    window_samples = int(window_sizes.get(freqBand, 0.5) / dt)
    n_batches = int(np.ceil(n_sources / batch_size))
    
    _, n_cores = get_compute_profile()
    print(f"Computing {connectivityType} ({freqBand}) for {n_seeds} seeds -> {n_sources} sources.")
    print(f"Parallelizing across {n_batches} batches on {n_cores} cores.")

    results = Parallel(n_jobs=n_cores, backend='threading', verbose=10)(
        delayed(_compute_source_batch)(
            batch_idx, batch_size, n_sources, n_timepoints, window_samples, 
            data_matrix, seed_indices, connectivityType
        )
        for batch_idx in range(n_batches)
    )

    connectivity_timeseries = np.empty((n_sources, n_timepoints))
    for batch_res, indices in results:
        connectivity_timeseries[indices, :] = batch_res
    
    return connectivity_timeseries

def main(subjID, voxRes, seedROI, targetLoc_str, connectivityType_str, freqBand):
    """Main loop with multi-location and multi-metric support"""
    voxRes = voxRes or '10mm'; seedROI = seedROI or 'left_visual'; freqBand = freqBand or 'theta'
    targetLocs = targetLoc_str.split(',') if targetLoc_str else ['left']
    metrics = connectivityType_str.split(',') if connectivityType_str else ['coh']
    
    h = socket.gethostname()
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if h == 'zod' else '/d/DATD/datd/MEG_MGS/MEG_BIDS' if h == 'vader' else '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    roi_indices = np.where(np.array(atlas[f'{seedROI}_points']).flatten() == 1)[0]
    
    # Load all targets once
    all_target_data, time_v = load_source_space_data(subjID, bidsRoot, 'mgs', voxRes, freqBand)
    
    results_summary = []
    
    for loc in targetLocs:
        # Prepare trials for this location
        targets = [4, 5, 6, 7, 8] if loc == 'left' else [1, 2, 3, 9, 10]
        data_subset = np.concatenate([all_target_data[t] for t in targets], axis=0)
        
        for metric in metrics:
            outDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'connectivity_{voxRes}')
            os.makedirs(outDir, exist_ok=True)
            outF = os.path.join(outDir, f'sub-{subjID:02d}_task-mgs_seededConnectivity_{voxRes}_{seedROI}_{loc}_{metric}_{freqBand}.pkl')
            
            if not os.path.exists(outF):
                print(f"--- Processing: {loc} | {metric} ---")
                start_t = time.time()
                res = compute_connectivity_measures(roi_indices, data_subset, time_v, metric, freqBand)
                with open(outF, 'wb') as f: pickle.dump(res, f)
                dur = time.time() - start_t
                results_summary.append(f"{loc}/{metric}: Done ({dur:.1f}s)")
                print(f"  Completed in {dur:.1f}s")
            else:
                results_summary.append(f"{loc}/{metric}: Skipped (exists)")

    # Send one summary email
    summary_txt = f"Subject {subjID:02d} ({freqBand}) Bulk Run Summary:\n" + "\n".join(results_summary)
    print(summary_txt)
    # send_completion_email(...) could be updated here if needed

if __name__ == '__main__':
    if len(os.sys.argv) < 2: 
        print("Usage: python inSourceSpaceSeededConnectivity.py <subjID> <voxRes> <seedROI> <targetLocs_CSV> <metrics_CSV> <freqBand>")
        os.sys.exit(1)
    
    sID = int(os.sys.argv[1]); vRes = os.sys.argv[2] if len(os.sys.argv) > 2 else '10mm'
    sROI = os.sys.argv[3] if len(os.sys.argv) > 3 else 'left_visual'; tLocs = os.sys.argv[4] if len(os.sys.argv) > 4 else 'left'
    cTypes = os.sys.argv[5] if len(os.sys.argv) > 5 else 'coh'; fBand = os.sys.argv[6] if len(os.sys.argv) > 6 else 'theta'
    
    main(sID, vRes, sROI, tLocs, cTypes, fBand)
