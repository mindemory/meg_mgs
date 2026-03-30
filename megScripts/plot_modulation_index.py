"""
plot_modulation_index.py

Phase-Amplitude Coupling Comodulogram using the Mean Vector Length (MVL)
as described in Canolty et al. (2006, Science).

  Composite signal: z(t) = A_high(t) * exp(i * phi_low(t))
  MVL = |mean(z)| / mean(A_high)   [normalized]

  X-axis : Phase frequency (f_phase, 2–30 Hz)
  Y-axis : Amplitude frequency (f_amp, 4–50 Hz)
  Color  : MVL strength

Layout per region (Visual / Frontal):
  2 rows (Stimulus interval | Delay interval)
  x
  2 cols (Ipsilateral | Contralateral)

Reference:
  Canolty et al. (2006). High gamma power is phase-locked to theta oscillations
  in human neocortex. Science 313:1626-1628.
"""

import os, h5py, socket, gc, smtplib
import numpy as np
from shutil import copyfile
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import signal
from joblib import Parallel, delayed
import sys

# ── Frequency axes ────────────────────────────────────────────────────────────
PHASE_FREQS  = np.arange(2,  32, 2, dtype=float)   # 2–30 Hz (phase providing)
AMP_FREQS    = np.arange(4,  52, 2, dtype=float)   # 4–50 Hz (amplitude)
N_PHASE_BINS = 18                                   # kept for legacy; not used by MVL
FILTER_ORDER = 4                                    # Butterworth order

# ── Intervals ─────────────────────────────────────────────────────────────────
INTERVALS = [
    ('Stimulus [-0.5s to 0.5s]', -0.5, 0.5),
    ('Delay    [0.5s  to 1.5s]',  0.5, 1.5),
]

# ── Email ─────────────────────────────────────────────────────────────────────
NOTIFY_EMAIL = 'mrugank.dake@nyu.edu'


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_compute_profile():
    h = socket.gethostname()
    if h == 'zod':    return 'mac',   4
    if h == 'vader':  return 'vader', 48
    return 'hpc', 10


def bandpass(data, f_lo, f_hi, sfreq, order=FILTER_ORDER):
    """Zero-phase Butterworth bandpass along last axis."""
    nyq = sfreq / 2.0
    b, a = signal.butter(order, [f_lo / nyq, f_hi / nyq], btype='band')
    return signal.filtfilt(b, a, data, axis=-1)


def mean_vector_length(phase, amplitude):
    """
    Canolty et al. (2006) normalized Mean Vector Length.
    z(t) = A_high(t) * exp(i * phi_low(t))
    MVL  = |mean(z)| / mean(A_high)
    Returns scalar in [0, 1].
    """
    if len(phase) == 0:
        return 0.0
    z = amplitude * np.exp(1j * phase)
    return float(np.abs(np.mean(z)) / (np.mean(amplitude) + 1e-12))


def compute_mi_pair(roi_signal, dt, t_idx, f_phase, f_amp):
    """
    Compute single-trial-averaged MI for one (f_phase, f_amp) pair.
    roi_signal : (trials, times)  — ROI-averaged broadband signal
    t_idx      : boolean mask of time samples within the interval
    Returns scalar MI.
    """
    sfreq = 1.0 / dt
    half_bw = max(1.0, f_phase * 0.5)  # ±50 % bandwidth for narrow bands

    # Phase band
    lo_p = max(0.5, f_phase - half_bw)
    hi_p = f_phase + half_bw
    # Amplitude band: use ±10 % BW
    lo_a = f_amp * 0.9
    hi_a = f_amp * 1.1

    try:
        filt_phase = bandpass(roi_signal, lo_p, hi_p, sfreq)
        filt_amp   = bandpass(roi_signal, lo_a, hi_a, sfreq)
    except Exception:
        return 0.0

    trial_mis = []
    for tr in range(roi_signal.shape[0]):
        ph  = np.angle(signal.hilbert(filt_phase[tr, t_idx]))
        amp = np.abs(  signal.hilbert(filt_amp  [tr, t_idx]))
        trial_mis.append(mean_vector_length(ph, amp))

    return float(np.mean(trial_mis))


def compute_comodulogram(roi_data, tgt_mask, dt, time_vector, t_start, t_end):
    """
    Full MI comodulogram for one condition.
    Returns array of shape (len(AMP_FREQS), len(PHASE_FREQS)).
    """
    data = roi_data[tgt_mask]               # (trials, times, sources)
    roi_signal = data.mean(axis=2)          # (trials, times)
    t_idx = (time_vector >= t_start) & (time_vector <= t_end)

    _, n_cores = get_compute_profile()
    pairs = [(fp, fa) for fp in PHASE_FREQS for fa in AMP_FREQS]

    results = Parallel(n_jobs=n_cores)(
        delayed(compute_mi_pair)(roi_signal, dt, t_idx, fp, fa)
        for fp, fa in pairs
    )

    mi_matrix = np.zeros((len(AMP_FREQS), len(PHASE_FREQS)))
    for idx, (fp, fa) in enumerate(pairs):
        pi = np.searchsorted(PHASE_FREQS, fp)
        ai = np.searchsorted(AMP_FREQS,   fa)
        mi_matrix[ai, pi] = results[idx]

    return mi_matrix


# ══════════════════════════════════════════════════════════════════════════════
# Data loading  (same copy-hack pattern as plot_phase_power_coupling.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare_data(subjID, bidsRoot, taskName, voxRes):
    subName = 'sub-%02d' % subjID
    print(f'Loading source space data for {subName}')

    surface_resolution = int(voxRes[:-2])
    source_data_fpath = os.path.join(
        bidsRoot, 'derivatives', subName, 'sourceRecon',
        f'{subName}_task-{taskName}_sourceSpaceData_{surface_resolution}.mat')

    def open_h5(fpath, fname_tmp):
        h = socket.gethostname()
        if h == 'zod':
            tmp = os.path.join('/Users/mrugank/Desktop', fname_tmp)
        elif h == 'vader':
            try:
                return h5py.File(fpath, 'r', locking=False)
            except Exception:
                tmp = os.path.join('/tmp', fname_tmp)
        else:
            return h5py.File(fpath, 'r')
        copyfile(fpath, tmp)
        f = h5py.File(tmp, 'r')
        os.remove(tmp)
        return f

    source_data = open_h5(
        source_data_fpath,
        f'{subName}_task-{taskName}_sourceSpaceData_raw_{surface_resolution}.mat')

    sg = source_data['sourcedataCombined']
    time_vector = np.array(source_data[sg['time'][0, 0]]).flatten()
    dt = np.mean(np.diff(time_vector))
    target_labels = np.array(sg['trialinfo']).T[:, 1]

    all_trials = []
    td = sg['trial']
    for i in range(td.shape[0]):
        all_trials.append(np.array(source_data[td[i, 0]]))
    data_matrix = np.stack(all_trials, axis=0)   # (trials, times, sources)

    # Behavioral filtering
    behav_path = os.path.join(
        bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'eyetracking',
        f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')
    behav_data = open_h5(
        behav_path,
        f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')
    i_sacc_err = np.array(behav_data['ii_sess_forSource']['i_sacc_err']).flatten()
    valid = ~np.isnan(i_sacc_err)
    data_matrix   = data_matrix[valid]
    target_labels = target_labels[valid]

    # Atlas ROIs
    atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    def roi_idx(key): return np.where(np.array(atlas[key]).flatten() == 1)[0]

    l_vis   = roi_idx('left_visual_points')
    r_vis   = roi_idx('right_visual_points')
    l_front = roi_idx('left_frontal_points')
    r_front = roi_idx('right_frontal_points')

    left_tgt  = np.isin(target_labels, [4, 5, 6, 7, 8])
    right_tgt = np.isin(target_labels, [1, 2, 3, 9, 10])

    roi_dict = {
        'Visual':  {'left_roi': data_matrix[:, :, l_vis],
                    'right_roi': data_matrix[:, :, r_vis]},
        'Frontal': {'left_roi': data_matrix[:, :, l_front],
                    'right_roi': data_matrix[:, :, r_front]},
    }

    source_data.close()
    behav_data.close()
    gc.collect()
    return roi_dict, left_tgt, right_tgt, dt, time_vector


# ══════════════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════════════

def process_region(region_name, roi_dict, left_tgt, right_tgt,
                   dt, time_vector, figures_dir, subjID):
    print(f'\n── {region_name} ROI ──')
    left_roi  = roi_dict['left_roi']
    right_roi = roi_dict['right_roi']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f'{region_name} ROI — MVL Comodulogram (Canolty et al. 2006)\n'
        f'Subject {subjID:02d}',
        fontsize=14)

    im = None
    for row, (iname, t0, t1) in enumerate(INTERVALS):
        print(f'  Interval: {iname.strip()}')
        # Compute 4 condition MI matrices
        print('    LL...'); mi_LL = compute_comodulogram(left_roi,  left_tgt,  dt, time_vector, t0, t1)
        print('    RR...'); mi_RR = compute_comodulogram(right_roi, right_tgt, dt, time_vector, t0, t1)
        print('    LR...'); mi_LR = compute_comodulogram(left_roi,  right_tgt, dt, time_vector, t0, t1)
        print('    RL...'); mi_RL = compute_comodulogram(right_roi, left_tgt,  dt, time_vector, t0, t1)

        ipsi  = (mi_LL + mi_RR) / 2.0
        contra = (mi_LR + mi_RL) / 2.0

        for col, (label, mat) in enumerate([('Ipsilateral', ipsi), ('Contralateral', contra)]):
            ax = axes[row, col]
            im = ax.imshow(mat, aspect='auto', origin='lower',
                           extent=[PHASE_FREQS[0], PHASE_FREQS[-1],
                                   AMP_FREQS[0],   AMP_FREQS[-1]],
                           cmap='RdBu_r', interpolation='bilinear')
            ax.set_title(f'{iname.strip()} — {label}', fontsize=10)
            if row == 1:
                ax.set_xlabel('Phase Frequency (Hz)')
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel('Amplitude Frequency (Hz)')
            else:
                ax.set_yticklabels([])
            fig.colorbar(im, ax=ax, label='MVL', shrink=0.8)

    fname = os.path.join(figures_dir, f'sub-{subjID:02d}_{region_name}_MVL_Comodulogram.png')
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f'  Saved: {fname}')


def send_email(subjID, voxRes, success=True, msg=''):
    try:
        h = socket.gethostname()
        subj = f'[MI] sub-{subjID:02d} {"DONE" if success else "FAILED"} on {h}'
        body = msg if not success else f'MI comodulogram complete. Subject {subjID:02d} | {voxRes} | {h}'
        with smtplib.SMTP('localhost') as s:
            s.sendmail(NOTIFY_EMAIL, NOTIFY_EMAIL, f'Subject: {subj}\n\n{body}')
    except Exception as e:
        print(f'  (Email skipped: {e})')


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(subjID, voxRes='10mm'):
    h = socket.gethostname()
    if h == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    elif h == 'vader':
        bidsRoot = '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'

    figures_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'modulation_index')
    os.makedirs(figures_dir, exist_ok=True)

    try:
        roi_dict, left_tgt, right_tgt, dt, time_vector = \
            load_and_prepare_data(subjID, bidsRoot, 'mgs', voxRes)

        for region in ['Visual', 'Frontal']:
            process_region(region, roi_dict[region], left_tgt, right_tgt,
                           dt, time_vector, figures_dir, subjID)

        print('\nDone!')
        send_email(subjID, voxRes, success=True)

    except Exception:
        import traceback
        err = traceback.format_exc()
        print(err)
        send_email(subjID, voxRes, success=False, msg=err)
        raise


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python plot_modulation_index.py <subjID> [voxRes]')
        sys.exit(1)
    subjID = int(sys.argv[1])
    voxRes = sys.argv[2] if len(sys.argv) > 2 else '10mm'
    main(subjID, voxRes)
