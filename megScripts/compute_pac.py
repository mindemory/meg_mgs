import os, h5py, socket, gc, smtplib, pickle
import numpy as np
from shutil import copyfile
from scipy.io import loadmat
from scipy import signal
from mne.time_frequency import tfr_array_morlet
from joblib import Parallel, delayed
import sys

# Hostname → compute profile mapping
def get_compute_profile():
    h = socket.gethostname()
    if h == 'zod':
        return 'mac', 4, '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    elif h == 'vader':
        return 'vader', 48, '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        return 'hpc', 10, '/scratch/mdd9787/meg_prf_greene/MEG_HPC'

# Define Reference Phase Frequency Bands
FREQUENCY_BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (13.0, 18.0)
}

# Email config for completion notification
NOTIFY_EMAIL = 'mrugank.dake@nyu.edu'

def send_completion_email(subjID, voxRes, status_msg, success=True, error_msg=None):
    try:
        hostname = socket.gethostname()
        subject = f'[PAC-Cross] sub-{subjID:02d} {voxRes} {"DONE" if success else "FAILED"} on {hostname}'
        body = f'Cross-Regional PAC analysis complete!\nSubject: {subjID:02d}\nHost: {hostname}\nStatus: {status_msg}'
        if error_msg: body += f'\n\nError:\n{error_msg}'
        msg = f'Subject: {subject}\n\n{body}'
        with smtplib.SMTP('localhost') as s:
            s.sendmail(NOTIFY_EMAIL, NOTIFY_EMAIL, msg)
    except: pass

def load_and_prepare_functional_rois(subjID, bidsRoot, voxRes):
    subName = 'sub-%02d' % subjID
    pid = os.getpid()
    derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName)
    sourceReconRoot = os.path.join(derivativesRoot, 'sourceRecon')
    surface_resolution = int(voxRes[:-2])
    source_data_fpath = os.path.join(sourceReconRoot, f'{subName}_task-mgs_sourceSpaceData_{surface_resolution}.mat')
    
    temp_dir = '/tmp' if socket.gethostname() != 'zod' else '/Users/mrugank/Desktop'
    source_data_temp_path = os.path.join(temp_dir, f'{subName}_{pid}_source.mat')
    copyfile(source_data_fpath, source_data_temp_path)
    
    with h5py.File(source_data_temp_path, 'r') as f:
        sourcedata_group = f['sourcedataCombined']
        time_vector = np.array(f[sourcedata_group['time'][0, 0]]).flatten()
        dt = np.mean(np.diff(time_vector))
        trialinfo = np.array(sourcedata_group['trialinfo']).T
        target_labels = trialinfo[:, 1]
        trial_data = sourcedata_group['trial']
        all_trials = []
        for i in range(trial_data.shape[0]):
            all_trials.append(np.array(f[trial_data[i, 0]]))
        data_matrix = np.stack(all_trials, axis=0) # (trials, times, sources)

    if os.path.exists(source_data_temp_path): os.remove(source_data_temp_path)

    # Behavior for valid trials
    behav_path = os.path.join(bidsRoot, 'derivatives', subName, 'eyetracking', f'{subName}_task-mgs-iisess_forSource.mat')
    behav_temp = os.path.join(temp_dir, f'{subName}_{pid}_behav.mat')
    copyfile(behav_path, behav_temp)
    with h5py.File(behav_temp, 'r') as f:
        valid_trials = ~np.isnan(np.array(f['ii_sess_forSource']['i_sacc_err']).flatten())
    if os.path.exists(behav_temp): os.remove(behav_temp)

    data_matrix = data_matrix[valid_trials]
    target_labels = target_labels[valid_trials]
    left_tgt = np.isin(target_labels, [4, 5, 6, 7, 8])
    right_tgt = np.isin(target_labels, [1, 2, 3, 9, 10])

    # Atlas extraction
    atlas_data = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    l_vis = np.where(atlas_data['left_visual_points'].flatten() == 1)[0]
    r_vis = np.where(atlas_data['right_visual_points'].flatten() == 1)[0]
    l_fro = np.where(atlas_data['left_frontal_points'].flatten() == 1)[0]
    r_fro = np.where(atlas_data['right_frontal_points'].flatten() == 1)[0]

    # Means (Trials, Times)
    V_L = data_matrix[:, :, l_vis].mean(axis=2)
    V_R = data_matrix[:, :, r_vis].mean(axis=2)
    F_L = data_matrix[:, :, l_fro].mean(axis=2)
    F_R = data_matrix[:, :, r_fro].mean(axis=2)

    # Remap to functional streams
    n_trials, n_times = V_L.shape
    func_rois = {
        'Ipsi-Vis': np.zeros((n_trials, n_times)),
        'Contra-Vis': np.zeros((n_trials, n_times)),
        'Ipsi-Fro': np.zeros((n_trials, n_times)),
        'Contra-Fro': np.zeros((n_trials, n_times))
    }
    for i in range(n_trials):
        if left_tgt[i]:
            func_rois['Ipsi-Vis'][i], func_rois['Contra-Vis'][i] = V_L[i], V_R[i]
            func_rois['Ipsi-Fro'][i], func_rois['Contra-Fro'][i] = F_L[i], F_R[i]
        else:
            func_rois['Ipsi-Vis'][i], func_rois['Contra-Vis'][i] = V_R[i], V_L[i]
            func_rois['Ipsi-Fro'][i], func_rois['Contra-Fro'][i] = F_R[i], F_L[i]
            
    return func_rois, dt, time_vector

def get_padded_tfr_cross(phase_data, amp_data, dt, f_phase, f_amp_freqs, time_vector, t_start, t_end, n_cycles=7):
    sfreq = 1/dt
    nyq = sfreq/2
    b, a = signal.butter(4, [f_phase[0]/nyq, f_phase[1]/nyq], btype='band')
    filt_p = signal.filtfilt(b, a, phase_data, axis=1)
    
    t_indices = np.where((time_vector >= t_start) & (time_vector <= t_end))[0]
    all_epochs = []
    cushion = 0.75
    for tr in range(phase_data.shape[0]):
        tgt_span = -filt_p[tr, t_indices[0]:t_indices[-1]+1]
        peaks, _ = signal.find_peaks(tgt_span, distance=int(sfreq/f_phase[1]))
        for p in peaks:
            g_idx = p + t_indices[0]
            s_idx = g_idx - int(cushion*sfreq)
            e_idx = g_idx + int(cushion*sfreq) + 1
            if s_idx >= 0 and e_idx <= phase_data.shape[1]:
                all_epochs.append(amp_data[tr, s_idx:e_idx])
                
    if not all_epochs: return None
    
    epochs = np.stack(all_epochs)[:, np.newaxis, :]
    tfr = tfr_array_morlet(epochs, sfreq, f_amp_freqs, n_cycles=n_cycles, output='power', n_jobs=1)
    tfr_avg = np.mean(tfr, axis=0)[0] 
    crop_s = int((cushion - 0.25) * sfreq)
    crop_e = int((cushion + 0.25) * sfreq)
    tfr_cropped = tfr_avg[:, crop_s:crop_e]
    
    baseline = np.mean(tfr_cropped, axis=1, keepdims=True)
    return (tfr_cropped - baseline) / (baseline + 1e-10)

def main(subjID, voxRes='8mm'):
    profile, n_cores, bidsRoot = get_compute_profile()
    
    def _compute_single_pair(func_rois, intervals, dt, amp_freqs, time_vector, band_name, f_low, p_name, a_target_name):
        pair_res = {}
        for epoch, t_start, t_end in intervals:
            res = get_padded_tfr_cross(func_rois[p_name], func_rois[a_target_name], dt, f_low, 
                                       amp_freqs, time_vector, t_start, t_end)
            pair_res[epoch] = res
        return band_name, p_name, a_target_name, pair_res

    try:
        func_rois, dt, time_vector = load_and_prepare_functional_rois(subjID, bidsRoot, voxRes)
        intervals = [('Baseline', -0.5, 0.0), ('Stimulus', 0.0, 0.2), ('Delay', 0.5, 1.5)]
        amp_freqs = np.arange(21, 57, 2)
        roi_names = list(func_rois.keys())
        
        # Parallelize across Bands and Interaction Pairs
        tasks = []
        for band_name, f_low in FREQUENCY_BANDS.items():
            for p_name in roi_names:
                for a_name in roi_names:
                    tasks.append((band_name, f_low, p_name, a_name))

        print(f"[*] Processing 48 PAC Tasks (3 bands x 16 pairs) in parallel (n_jobs={n_cores})...")
        results_list = Parallel(n_jobs=n_cores)(
            delayed(_compute_single_pair)(func_rois, intervals, dt, amp_freqs, time_vector, b, f, p, a) 
            for b, f, p, a in tasks
        )

        # Reconstruct result dictionary
        results = {band: {epoch: {} for epoch, _, _ in intervals} for band in FREQUENCY_BANDS}
        for b_name, p_name, a_name, pair_res in results_list:
            pair_key = f"{p_name}_to_{a_name}"
            for epoch, res in pair_res.items():
                results[b_name][epoch][pair_key] = res

        output_dir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', 'pac_data')
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f'sub-{subjID:02d}_CrossRegional_PAC_{voxRes}.pkl')
        with open(out_file, 'wb') as f:
            pickle.dump({'data': results, 'amp_freqs': amp_freqs, 'subjID': subjID, 'voxRes': voxRes}, f)
        print(f"  Saved PAC results: {out_file}")
        send_completion_email(subjID, voxRes, 'All-to-all cross-regional suite complete', success=True)
    except Exception as e:
        import traceback
        send_completion_email(subjID, voxRes, 'FAILED', success=False, error_msg=traceback.format_exc())
        raise

if __name__ == '__main__':
    if len(sys.argv) < 2: sys.exit(1)
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else '8mm')
