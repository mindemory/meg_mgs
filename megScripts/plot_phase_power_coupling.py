import os, h5py, socket, gc, smtplib
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from mne.time_frequency import tfr_array_morlet
from joblib import Parallel, delayed
import sys

# Hostname → compute profile mapping
def get_compute_profile():
    h = socket.gethostname()
    if h == 'zod':
        return 'mac', 4          # Local Mac M1
    elif h == 'vader':
        return 'vader', 48       # Vader local cluster (50 cores)
    else:
        return 'hpc', 10         # Greene HPC or other

def is_local():
    """True on zod (Mac) or vader — both have direct DATD access."""
    return socket.gethostname() in ('zod', 'vader')

# Define Reference Phase Frequency Bands
FREQUENCY_BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (18.0, 25.0)
}

# Email config for completion notification
NOTIFY_EMAIL = 'mrugank.dake@nyu.edu'


def send_completion_email(subjID, voxRes, figures_dir, success=True, error_msg=None):
    """Send a simple email notification when the script finishes."""
    try:
        hostname = socket.gethostname()
        if success:
            subject = f'[PAC] sub-{subjID:02d} {voxRes} DONE on {hostname}'
            body = (f'Phase-Power Coupling analysis complete!\n\n'
                    f'Subject: {subjID:02d}\nResolution: {voxRes}\n'
                    f'Host: {hostname}\n'
                    f'Figures saved to: {figures_dir}')
        else:
            subject = f'[PAC] sub-{subjID:02d} {voxRes} FAILED on {hostname}'
            body = f'Script failed with error:\n{error_msg}'

        msg = f'Subject: {subject}\n\n{body}'
        with smtplib.SMTP('localhost') as s:
            s.sendmail(NOTIFY_EMAIL, NOTIFY_EMAIL, msg)
        print(f'  Notification email sent to {NOTIFY_EMAIL}')
    except Exception as e:
        print(f'  (Email notification skipped: {e})')


def load_and_prepare_data(subjID, bidsRoot, taskName, voxRes):
    subName = 'sub-%02d' % subjID
    print(f'Loading source space data for {subName}')

    derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName)
    sourceReconRoot = os.path.join(derivativesRoot, 'sourceRecon')
    surface_resolution = int(voxRes[:-2])
    source_data_fpath = os.path.join(sourceReconRoot, f'{subName}_task-{taskName}_sourceSpaceData_{surface_resolution}.mat')

    VADER_TEMP = '/d/DATD/home/mrugank/Documents'
    
    if socket.gethostname() == 'zod':
        # Mac: copy to Desktop to avoid network mount issues
        source_data_temp_path = os.path.join('/Users/mrugank/Desktop', f'{subName}_task-{taskName}_sourceSpaceData_raw_{surface_resolution}.mat')
        copyfile(source_data_fpath, source_data_temp_path)
        source_data = h5py.File(source_data_temp_path, 'r')
        os.remove(source_data_temp_path)
    elif socket.gethostname() == 'vader':
        # Vader: copy to local home Documents to avoid network mount issues
        source_data_temp_path = os.path.join(VADER_TEMP, f'{subName}_task-{taskName}_sourceSpaceData_raw_{surface_resolution}.mat')
        copyfile(source_data_fpath, source_data_temp_path)
        source_data = h5py.File(source_data_temp_path, 'r')
        os.remove(source_data_temp_path)
    else:
        # HPC: read directly
        source_data = h5py.File(source_data_fpath, 'r')

    sourcedata_group = source_data['sourcedataCombined']

    time_data = sourcedata_group['time']
    time_vector = np.array(source_data[time_data[0, 0]]).flatten()
    dt = np.mean(np.diff(time_vector))

    trialinfo = np.array(sourcedata_group['trialinfo']).T
    target_labels = trialinfo[:, 1]

    trial_data = sourcedata_group['trial']
    all_trials = []
    for trial_idx in range(trial_data.shape[0]):
        trial_ref = trial_data[trial_idx, 0]
        all_trials.append(np.array(source_data[trial_ref]))

    data_matrix = np.stack(all_trials, axis=0)  # (trials, times, sources)

    print("Filtering valid trials using behavioral data...")
    behav_data_path = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'eyetracking',
                                   f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')

    if socket.gethostname() == 'zod':
        # Mac: copy to Desktop to avoid network mount issues
        behav_data_temp_path = os.path.join('/Users/mrugank/Desktop', f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')
        copyfile(behav_data_path, behav_data_temp_path)
        behav_data = h5py.File(behav_data_temp_path, 'r')
        os.remove(behav_data_temp_path)
    elif socket.gethostname() == 'vader':
        # Vader: copy to local home Documents to avoid network mount issues
        behav_data_temp_path = os.path.join(VADER_TEMP, f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')
        copyfile(behav_data_path, behav_data_temp_path)
        behav_data = h5py.File(behav_data_temp_path, 'r')
        os.remove(behav_data_temp_path)
    else:
        # HPC: read directly
        behav_data = h5py.File(behav_data_path, 'r')

    ii_sess_forSource = behav_data['ii_sess_forSource']
    i_sacc_err = np.array(ii_sess_forSource['i_sacc_err']).flatten()

    valid_trials = ~np.isnan(i_sacc_err)
    data_matrix = data_matrix[valid_trials, :, :]
    target_labels = target_labels[valid_trials]

    print("Extracting Wang Atlas ROIs (Left/Right Visual & Frontal)...")
    atlas_fpath = os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat')
    atlas_data = loadmat(atlas_fpath)

    l_vis = np.where(np.array(atlas_data['left_visual_points']).flatten() == 1)[0]
    r_vis = np.where(np.array(atlas_data['right_visual_points']).flatten() == 1)[0]
    l_front = np.where(np.array(atlas_data['left_frontal_points']).flatten() == 1)[0]
    r_front = np.where(np.array(atlas_data['right_frontal_points']).flatten() == 1)[0]

    left_visual_data = data_matrix[:, :, l_vis]
    right_visual_data = data_matrix[:, :, r_vis]
    left_frontal_data = data_matrix[:, :, l_front]
    right_frontal_data = data_matrix[:, :, r_front]

    left_tgt_mask = np.isin(target_labels, [4, 5, 6, 7, 8])
    right_tgt_mask = np.isin(target_labels, [1, 2, 3, 9, 10])

    roi_dict = {
        'Visual': {'left_roi': left_visual_data, 'right_roi': right_visual_data},
        'Frontal': {'left_roi': left_frontal_data, 'right_roi': right_frontal_data}
    }

    del data_matrix
    source_data.close()
    behav_data.close()
    gc.collect()

    return roi_dict, left_tgt_mask, right_tgt_mask, dt, time_vector


def find_roi_troughs(roi_data, dt, f_min, f_max, time_vector, t_start, t_end):
    """Finds troughs within the specified time window after bandpassing the ROI-average signal."""
    n_trials, _, _ = roi_data.shape
    regional_signal = roi_data.mean(axis=2)  # (trials, times)

    nyq = 1 / (2 * dt)
    b, a = signal.butter(4, [f_min / nyq, f_max / nyq], btype='band')
    filtered_signal = signal.filtfilt(b, a, regional_signal, axis=1)

    t_indices = np.where((time_vector >= t_start) & (time_vector <= t_end))[0]
    if len(t_indices) == 0:
        return []

    all_troughs = []
    for tr_idx in range(n_trials):
        target_span = -filtered_signal[tr_idx, t_indices[0]:t_indices[-1] + 1]
        peaks, _ = signal.find_peaks(target_span)
        for p in peaks:
            global_idx = p + t_indices[0]
            all_troughs.append((tr_idx, global_idx))

    return all_troughs


def extract_epochs(roi_data, troughs, dt, epoch_window=(-1.0, 1.0)):
    """Extracts 2-second epochs centered around each trough. Shape: (epochs, channels, times)"""
    sfreq = 1 / dt
    offset_samples = int(epoch_window[1] * sfreq)
    n_times = 2 * offset_samples + 1

    epochs = []
    for tr_idx, t_idx in troughs:
        start_idx = t_idx - offset_samples
        end_idx = t_idx + offset_samples + 1
        if start_idx >= 0 and end_idx <= roi_data.shape[1]:
            epoch = roi_data[tr_idx, start_idx:end_idx, :].T  # (sources, times)
            epochs.append(epoch)

    if len(epochs) == 0:
        return np.zeros((0, roi_data.shape[2], n_times))

    return np.stack(epochs, axis=0)


def _compute_single_epoch_tfr(epoch, sfreq, freqs, dyn_cycles):
    """Helper for joblib: computes TFR for a single epoch (1, sources, times)."""
    return tfr_array_morlet(epoch[np.newaxis], sfreq, freqs, n_cycles=dyn_cycles, output='power', n_jobs=1)[0]


def get_condition_tfr(roi_data, tgt_mask, dt, freqs, f_min, f_max, time_vector, t_start, t_end, w=7):
    """Compute trough-locked TFR for a given condition."""
    condition_data = roi_data[tgt_mask]

    troughs = find_roi_troughs(condition_data, dt, f_min, f_max, time_vector, t_start, t_end)
    if len(troughs) == 0:
        return None

    epochs = extract_epochs(condition_data, troughs, dt, epoch_window=(-1.0, 1.0))
    if epochs.shape[0] == 0:
        return None

    sfreq = 1 / dt
    dyn_cycles = np.clip(freqs / 2.0, 3, w)

    compute_env, n_cores = get_compute_profile()

    if compute_env == 'hpc':
        # Greene HPC: single MNE call across all epochs (ample RAM)
        tfr_power = tfr_array_morlet(epochs, sfreq, freqs, n_cycles=dyn_cycles, output='power', n_jobs=n_cores)
        mean_tfr = tfr_power.mean(axis=0)
    else:
        # Mac or Vader: joblib dispatches one epoch per worker (RAM-safe)
        print(f'    Computing TFR for {epochs.shape[0]} epochs on {n_cores} cores ({compute_env})...')
        results = Parallel(n_jobs=n_cores)(
            delayed(_compute_single_epoch_tfr)(epochs[i], sfreq, freqs, dyn_cycles)
            for i in range(epochs.shape[0])
        )
        mean_tfr = np.mean(np.stack(results, axis=0), axis=0)

    # Average across sources
    regional_tfr = mean_tfr.mean(axis=0)  # (freqs, times)

    # Fold-change normalization relative to mean across time
    baseline = regional_tfr.mean(axis=1, keepdims=True)
    normalized_tfr = (regional_tfr - baseline) / (baseline + 1e-10)

    return normalized_tfr


def process_lateralized_region(region_name, roi_data_dict, left_tgt_mask, right_tgt_mask,
                                dt, freqs, figures_dir, subjID, time_vector):
    print(f"\nProcessing Event-Related lateralized region: {region_name}")

    left_roi = roi_data_dict['left_roi']
    right_roi = roi_data_dict['right_roi']

    intervals = [
        ('Stimulus\n[-0.5s to 0.5s]', -0.5, 0.5),
        ('Delay\n[0.5s to 1.5s]', 0.5, 1.5)
    ]

    bands_list = list(FREQUENCY_BANDS.items())

    # 3 rows (bands) x 4 cols (Stim Ipsi, Stim Contra, Delay Ipsi, Delay Contra)
    fig, axes = plt.subplots(3, 4, figsize=(22, 14))
    fig.suptitle(
        f'{region_name} ROI — Trough-Locked TFR PAC | Subject {subjID:02d}\n'
        f'Columns: [Stimulus Ipsi | Stimulus Contra | Delay Ipsi | Delay Contra]',
        fontsize=14
    )

    col_titles = ['Stimulus — Ipsi', 'Stimulus — Contra', 'Delay — Ipsi', 'Delay — Contra']
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=11, fontweight='bold')

    im = None
    for row_idx, (band_name, (f_min, f_max)) in enumerate(bands_list):
        print(f"  Phase tracking: {band_name.capitalize()} ({f_min}-{f_max} Hz)")

        # Compute all 4 condition TFRs for both intervals
        tfr_cache = {}
        for interval_name, t_start, t_end in intervals:
            key = interval_name.split('\n')[0]
            tfr_cache[key] = {
                'LL': get_condition_tfr(left_roi,  left_tgt_mask,  dt, freqs, f_min, f_max, time_vector, t_start, t_end),
                'RR': get_condition_tfr(right_roi, right_tgt_mask, dt, freqs, f_min, f_max, time_vector, t_start, t_end),
                'LR': get_condition_tfr(left_roi,  right_tgt_mask, dt, freqs, f_min, f_max, time_vector, t_start, t_end),
                'RL': get_condition_tfr(right_roi, left_tgt_mask,  dt, freqs, f_min, f_max, time_vector, t_start, t_end),
            }

        # Build the 4-column layout per row
        # col 0: Stim Ipsi, col 1: Stim Contra, col 2: Delay Ipsi, col 3: Delay Contra
        col_data = []
        for interval_name, _, _ in intervals:
            key = interval_name.split('\n')[0]
            c = tfr_cache[key]
            if any(v is None for v in c.values()):
                col_data.extend([None, None])
            else:
                ipsi  = (c['LL'] + c['RR']) / 2.0
                contra = (c['LR'] + c['RL']) / 2.0
                col_data.extend([ipsi, contra])

        # Symmetric colormap scale across Ipsi+Contra for this row
        valid_mats = [m for m in col_data if m is not None]
        if valid_mats:
            row_vmax = max(np.abs(m).max() for m in valid_mats)
        else:
            row_vmax = 1.0

        col_labels = ['Ipsi', 'Contra', 'Ipsi', 'Contra']
        for ci, mat in enumerate(col_data):
            ax = axes[row_idx, ci]
            if mat is None:
                ax.text(0.5, 0.5, 'Insufficient\nData', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)
                ax.set_title(f'{band_name.capitalize()} — {col_labels[ci]}')
                continue

            im = ax.imshow(mat, aspect='auto', origin='lower',
                           extent=[-1.0, 1.0, freqs[0], freqs[-1]],
                           cmap='RdBu_r', interpolation='bilinear',
                           vmin=-row_vmax, vmax=row_vmax)
            ax.set_xlim([-0.5, 0.5])
            ax.axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=1.2)

            # Row label on left-most column
            if ci == 0:
                ax.set_ylabel(f'{band_name.capitalize()}\nPower Freq (Hz)', fontsize=9)
            else:
                ax.set_yticklabels([])

            # X-axis label only on bottom row
            if row_idx == len(bands_list) - 1:
                ax.set_xlabel('Time from Trough (s)', fontsize=8)
            else:
                ax.set_xticklabels([])

            ax.set_yticks(freqs[::4])

    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cbar_ax, label='Power Fold Change (baseline-normalized)')

    plot_fname = os.path.join(figures_dir, f'sub-{subjID:02d}_{region_name}_TFR_IpsiContra.png')
    fig.subplots_adjust(left=0.07, right=0.91, top=0.88, bottom=0.07, wspace=0.08, hspace=0.25)
    fig.savefig(plot_fname, dpi=300)
    plt.close(fig)
    print(f"  Saved: {plot_fname}")


def main(subjID, voxRes='10mm'):
    h = socket.gethostname()
    if h == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    elif h == 'vader':
        bidsRoot = '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'

    figures_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'phase_power_coupling')
    os.makedirs(figures_dir, exist_ok=True)

    try:
        roi_dict, left_tgt_mask, right_tgt_mask, dt, time_vector = load_and_prepare_data(
            subjID, bidsRoot, 'mgs', voxRes)

        freqs = np.arange(4, 51, 2)

        process_lateralized_region('Visual', roi_dict['Visual'], left_tgt_mask, right_tgt_mask,
                                   dt, freqs, figures_dir, subjID, time_vector)
        process_lateralized_region('Frontal', roi_dict['Frontal'], left_tgt_mask, right_tgt_mask,
                                   dt, freqs, figures_dir, subjID, time_vector)

        print("\nDone! All analysis and plotting complete.")
        send_completion_email(subjID, voxRes, figures_dir, success=True)

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"\nScript failed:\n{err}")
        send_completion_email(subjID, voxRes, figures_dir, success=False, error_msg=err)
        raise


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python plot_phase_power_coupling.py <subjID> [voxRes]")
        sys.exit(1)

    subjID = int(sys.argv[1])
    voxRes = sys.argv[2] if len(sys.argv) > 2 else '10mm'

    main(subjID, voxRes)
