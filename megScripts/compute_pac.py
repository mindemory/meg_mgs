import os, h5py, socket, gc, smtplib, pickle
import numpy as np
from shutil import copyfile
from scipy.io import loadmat
from scipy import signal
from mne.time_frequency import tfr_array_morlet
from joblib import Parallel, delayed
import sys

# Compute profile mapping
def get_compute_profile():
    h = socket.gethostname()
    if h == 'zod': return 'mac', 4, '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    elif h == 'vader': return 'vader', 48, '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else: return 'hpc', 10, '/scratch/mdd9787/meg_prf_greene/MEG_HPC'

# Define Frequency Bands
PHASE_BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (13.0, 18.0)
}
AMP_BANDS = {
    'high_beta': (21.0, 30.0),
    'low_gamma': (31.0, 55.0)
}

# Email config
NOTIFY_EMAIL = 'mrugank.dake@nyu.edu'
def send_completion_email(subjID, voxRes, status_msg, success=True, error_msg=None):
    try:
        hostname = socket.gethostname()
        subject = f'[PAC-Quant] sub-{subjID:02d} {voxRes} {"DONE" if success else "FAILED"} on {hostname}'
        body = f'PAC Quantification complete!\nSubject: {subjID:02d}\nHost: {hostname}\nStatus: {status_msg}'
        if error_msg: body += f'\n\nError:\n{error_msg}'
        msg = f'Subject: {subject}\n\n{body}'
        with smtplib.SMTP('localhost') as s: s.sendmail(NOTIFY_EMAIL, NOTIFY_EMAIL, msg)
    except: pass

def load_and_prepare_functional_rois(subjID, bidsRoot, voxRes):
    subName = 'sub-%02d' % subjID
    pid = os.getpid()
    source_data_fpath = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'{subName}_task-mgs_sourceSpaceData_{int(voxRes[:-2])}.mat')
    temp_dir = '/tmp' if socket.gethostname() != 'zod' else '/Users/mrugank/Desktop'
    source_temp = os.path.join(temp_dir, f'{subName}_{pid}_source.mat')
    copyfile(source_data_fpath, source_temp)
    
    with h5py.File(source_temp, 'r') as f:
        sourcedata_group = f['sourcedataCombined']
        time_vector = np.array(f[sourcedata_group['time'][0, 0]]).flatten()
        dt = np.mean(np.diff(time_vector))
        target_labels = np.array(sourcedata_group['trialinfo']).T[:, 1]
        trial_data = sourcedata_group['trial']
        all_trials = [np.array(f[trial_data[i, 0]]) for i in range(trial_data.shape[0])]
        data_matrix = np.stack(all_trials, axis=0)
    if os.path.exists(source_temp): os.remove(source_temp)

    behav_path = os.path.join(bidsRoot, 'derivatives', subName, 'eyetracking', f'{subName}_task-mgs-iisess_forSource.mat')
    behav_temp = os.path.join(temp_dir, f'{subName}_{pid}_behav.mat')
    copyfile(behav_path, behav_temp)
    with h5py.File(behav_temp, 'r') as f:
        valid_trials = ~np.isnan(np.array(f['ii_sess_forSource']['i_sacc_err']).flatten())
    if os.path.exists(behav_temp): os.remove(behav_temp)

    data_matrix = data_matrix[valid_trials]
    target_labels = target_labels[valid_trials]
    left_tgt = np.isin(target_labels, [4, 5, 6, 7, 8])

    atlas_data = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    l_vis, r_vis = [np.where(atlas_data[f'{s}_visual_points'].flatten() == 1)[0] for s in ['left', 'right']]
    l_fro, r_fro = [np.where(atlas_data[f'{s}_frontal_points'].flatten() == 1)[0] for s in ['left', 'right']]

    V_L, V_R = [data_matrix[:, :, p].mean(axis=2) for p in [l_vis, r_vis]]
    F_L, F_R = [data_matrix[:, :, p].mean(axis=2) for p in [l_fro, r_fro]]

    n_trials, n_times = V_L.shape
    func_rois = {k: np.zeros((n_trials, n_times)) for k in ['Ipsi-Vis', 'Contra-Vis', 'Ipsi-Fro', 'Contra-Fro']}
    for i in range(n_trials):
        if left_tgt[i]:
            func_rois['Ipsi-Vis'][i], func_rois['Contra-Vis'][i] = V_L[i], V_R[i]
            func_rois['Ipsi-Fro'][i], func_rois['Contra-Fro'][i] = F_L[i], F_R[i]
        else:
            func_rois['Ipsi-Vis'][i], func_rois['Contra-Vis'][i] = V_R[i], V_L[i]
            func_rois['Ipsi-Fro'][i], func_rois['Contra-Fro'][i] = F_R[i], F_L[i]
    return func_rois, dt, time_vector

def compute_tort_mi(phase, amplitude, n_bins=18):
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(phase, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    bin_sum = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (bin_idx == i)
        if np.any(mask): bin_sum[i] = np.mean(amplitude[mask])
    if np.sum(bin_sum) == 0: return 0
    p = bin_sum / np.sum(bin_sum)
    h = -np.sum(p * np.log(p + 1e-12))
    return (np.log(n_bins) - h) / np.log(n_bins)

def get_pac_metrics_cross(phase_data, amp_data, dt, f_phase, f_amp_range, time_vector, t_start, t_end, n_cycles=7):
    sfreq = 1/dt
    nyq = sfreq/2
    # 1. Filter for Trough detection
    b, a = signal.butter(4, [f_phase[0]/nyq, f_phase[1]/nyq], btype='band')
    filt_p = signal.filtfilt(b, a, phase_data, axis=1)
    # 2. Extract Amplitude for TFR
    f_amp_freqs = np.arange(f_amp_range[0], f_amp_range[1] + 1, 2)
    t_indices = np.where((time_vector >= t_start) & (time_vector <= t_end))[0]
    all_epochs = []
    cushion = 0.75
    for tr in range(phase_data.shape[0]):
        tgt_span = -filt_p[tr, t_indices[0]:t_indices[-1]+1]
        peaks, _ = signal.find_peaks(tgt_span, distance=int(sfreq/f_phase[1]))
        for p in peaks:
            g_idx = p + t_indices[0]
            s_idx, e_idx = g_idx - int(cushion*sfreq), g_idx + int(cushion*sfreq) + 1
            if s_idx >= 0 and e_idx <= phase_data.shape[1]: all_epochs.append(amp_data[tr, s_idx:e_idx])
    
    tfr_norm = None
    if all_epochs:
        epochs = np.stack(all_epochs)[:, np.newaxis, :]
        tfr = tfr_array_morlet(epochs, sfreq, f_amp_freqs, n_cycles=n_cycles, output='power', n_jobs=1, zero_mean=False)
        tfr_avg = np.mean(tfr, axis=0)[0] 
        crop_s, crop_e = int((cushion-0.25)*sfreq), int((cushion+0.25)*sfreq)
        tfr_cropped = tfr_avg[:, crop_s:crop_e]
        baseline = np.mean(tfr_cropped, axis=1, keepdims=True)
        tfr_norm = (tfr_cropped - baseline) / (baseline + 1e-10)

    # 3. Scalar MI (Traditional Tort method)
    b_mi, a_mi = signal.butter(4, [f_amp_range[0]/nyq, f_amp_range[1]/nyq], btype='band')
    filt_a_mi = np.abs(signal.hilbert(signal.filtfilt(b_mi, a_mi, amp_data, axis=1), axis=1))
    angle_p_mi = np.angle(signal.hilbert(filt_p, axis=1))
    
    # Use same time window for MI as TFR troughs (or the whole interval)
    idx_mi = t_indices
    mi_val = compute_tort_mi(angle_p_mi[:, idx_mi].flatten(), filt_a_mi[:, idx_mi].flatten())
    
    return {'TFR': tfr_norm, 'MI': mi_val, 'amp_freqs': f_amp_freqs}

def main(subjID, voxRes='8mm'):
    profile, n_cores, bidsRoot = get_compute_profile()
    
    def _compute_single_task(func_rois, intervals, dt, time_vector, b_p_name, f_p, b_a_name, f_a, p_name, a_name):
        return b_p_name, b_a_name, p_name, a_name, {e: get_pac_metrics_cross(func_rois[p_name], func_rois[a_name], dt, f_p, f_a, time_vector, ts, te) for e, ts, te in intervals}

    try:
        func_rois, dt, time_vector = load_and_prepare_functional_rois(subjID, bidsRoot, voxRes)
        intervals = [('Baseline', -0.5, 0.0), ('Stimulus', 0.0, 0.2), ('Delay', 0.5, 1.5)]
        roi_names = list(func_rois.keys())
        
        tasks = []
        for b_p_name, f_p in PHASE_BANDS.items():
            for b_a_name, f_a in AMP_BANDS.items():
                for p_name in roi_names:
                    for a_name in roi_names:
                        tasks.append((b_p_name, f_p, b_a_name, f_a, p_name, a_name))

        print(f"[*] Processing 96 Tasks (3-phase x 2-amp x 16-pairs) in parallel (n_jobs={n_cores})...")
        results_list = Parallel(n_jobs=n_cores)(delayed(_compute_single_task)(func_rois, intervals, dt, time_vector, *t) for t in tasks)

        data = {}
        for b_p, b_a, p, a, res_dict in results_list:
            if b_p not in data: data[b_p] = {}
            if b_a not in data[b_p]: data[b_p][b_a] = {}
            data[b_p][b_a][f"{p}_to_{a}"] = res_dict

        output_dir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', 'pac_data')
        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f'sub-{subjID:02d}_CrossRegional_PAC_Quantified_{voxRes}.pkl')
        with open(out_file, 'wb') as f:
            pickle.dump({'data': data, 'subjID': subjID, 'voxRes': voxRes}, f)
        print(f"  Saved PAC Quantification: {out_file}")
        send_completion_email(subjID, voxRes, 'Quantification suite complete', success=True)
    except Exception:
        import traceback
        send_completion_email(subjID, voxRes, 'FAILED', success=False, error_msg=traceback.format_exc())
        raise

if __name__ == '__main__':
    if len(sys.argv) < 2: sys.exit(1)
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else '8mm')
