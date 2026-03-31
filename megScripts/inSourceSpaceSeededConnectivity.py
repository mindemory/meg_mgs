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
        return 'vader', 24       # Cap at 24 for memory safety
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

def load_source_space_data(subjID, bidsRoot, taskName, voxRes, targetLoc, freqBand):
    """Load and concatenate source space data for all targets (HDF5 staging)"""
    subName = 'sub-%02d' % subjID
    print(f'Loading source space data for {subName}')
    
    derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName)
    sourceReconRoot = os.path.join(derivativesRoot, 'sourceRecon')
    freqSpaceRoot = os.path.join(sourceReconRoot, 'freqSpace')
    freqSpace_fpath = os.path.join(freqSpaceRoot, f'{subName}_task-{taskName}_complex{freqBand}_allTargets_{voxRes[:-2]}.mat')
    
    hostname = socket.gethostname()
    tempDir = '/Users/mrugank/Desktop' if hostname == 'zod' else '/d/DATD/hyper/experiments/Mrugank/meg_mgs'
    
    freqSpaceTempPath = os.path.join(tempDir, f'{subName}_task-{taskName}_complex{freqBand}_allTargets_{voxRes[:-2]}_{int(time.time())}.mat')
    print(f'  Staging to {freqSpaceTempPath}...')
    copyfile(freqSpace_fpath, freqSpaceTempPath)
    
    with h5py.File(freqSpaceTempPath, 'r', locking=False) as f:
        source_data_refs = np.array(f['sourceDataByTarget'])
        all_trials = []
        time_vector = None
        target_indices = [4, 5, 6, 7, 8] if targetLoc == 'left' else [1, 2, 3, 9, 10]

        for target_idx in target_indices:
            target_ref = source_data_refs[0, target_idx - 1]
            target_group = f[target_ref]
            trial_dataset = target_group['trial']
            
            if time_vector is None:
                time_refs = target_group['time']
                time_vector = np.array(f[time_refs[0, 0]])
                
            for trial_idx in range(trial_dataset.shape[0]):
                trial_data = f[trial_dataset[trial_idx, 0]]
                all_trials.append(np.array(trial_data))
    
    data_matrix = np.stack(all_trials, axis=0)
    data_matrix = data_matrix['real'] + 1j * data_matrix['imag']
    if os.path.exists(freqSpaceTempPath):
        os.remove(freqSpaceTempPath)
        
    print(f"Data matrix loaded: {data_matrix.shape} (trials x times x sources)")
    return data_matrix, time_vector

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

def main(subjID, voxRes, seedROI, targetLoc, connectivityType, freqBand):
    """Main function for source space connectivity analysis"""
    voxRes, seedROI, targetLoc, connectivityType, freqBand = voxRes or '10mm', seedROI or 'left_visual', targetLoc or 'left', connectivityType or 'coh', freqBand or 'theta'
    
    h = socket.gethostname()
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if h == 'zod' else '/d/DATD/datd/MEG_MGS/MEG_BIDS' if h == 'vader' else '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    
    outDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'connectivity_{voxRes}')
    os.makedirs(outDir, exist_ok=True)
    outF = os.path.join(outDir, f'sub-{subjID:02d}_task-mgs_seededConnectivity_{voxRes}_{seedROI}_{targetLoc}_{connectivityType}_{freqBand}.pkl')

    if not os.path.exists(outF):
        data, time_v = load_source_space_data(subjID, bidsRoot, 'mgs', voxRes, targetLoc, freqBand)
        atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
        roi_indices = np.where(np.array(atlas[f'{seedROI}_points']).flatten() == 1)[0]
        
        start_t = time.time()
        res = compute_connectivity_measures(roi_indices, data, time_v, connectivityType, freqBand)
        print(f"Analysis completed in {time.time() - start_t:.2f}s")
        with open(outF, 'wb') as f: pickle.dump(res, f)
        print(f"Saved to {outF}")

if __name__ == '__main__':
    msg = f"Usage: python inSourceSpaceSeededConnectivity.py <subjID> <voxRes> <seedROI> <targetLoc> <coh|imcoh|dpli> <freqBand>"
    if len(os.sys.argv) < 2: print(msg); os.sys.exit(1)
    
    sID = int(os.sys.argv[1]); vRes = os.sys.argv[2] if len(os.sys.argv) > 2 else '10mm'
    sROI = os.sys.argv[3] if len(os.sys.argv) > 3 else 'left_visual'; tLoc = os.sys.argv[4] if len(os.sys.argv) > 4 else 'left'
    cType = os.sys.argv[5] if len(os.sys.argv) > 5 else 'coh'; fBand = os.sys.argv[6] if len(os.sys.argv) > 6 else 'theta'
    
    print(f"--- Connectivity Suite: sub-{sID:02d} | {vRes} | {cType} | {fBand} ---")
    main(sID, vRes, sROI, tLoc, cType, fBand)
