"""
temporalGeneralizationDecoding.py

Temporal Generalization Matrix (TGM) decoding of memory target angles.
Based on King & Dehaene (2014, TICS): train SVR at each time point, test on ALL
other time points, yielding a (n_train_t x n_test_t) cross-temporal accuracy matrix.

  - SVR with sin/cos regression for circular angle prediction
  - Leave-One-Out cross-validation on the trial dimension
  - Parallelized over train time points via joblib
  - Host-aware core count (zod=4, vader=48, hpc=10)
  - Email notification on completion or failure

Usage:
    python temporalGeneralizationDecoding.py <subjID> [voxRes] [freq_band]

Example:
    python temporalGeneralizationDecoding.py 8 8mm beta &

Reference:
    King & Dehaene (2014). Characterizing dynamics of mental representations
    with temporal generalization. Trends Cogn Sci 18:203-210.
"""

import os, h5py, socket, gc, smtplib, pickle
import numpy as np
from shutil import copyfile
from scipy.io import loadmat, savemat
from scipy import signal
from scipy.stats import circmean
from sklearn.svm import SVR
from joblib import Parallel, delayed
import sys

# ── Frequency bands ───────────────────────────────────────────────────────────
FREQUENCY_BANDS = {
    'theta':    (4.0,  8.0),
    'alpha':    (8.0,  12.0),
    'beta':     (18.0, 25.0),
    'lowgamma': (25.0, 50.0),
}

# Angle mapping: target label → degrees
ANGLE_MAPPING = {1: 0, 2: 25, 3: 50, 4: 130, 5: 155,
                 6: 180, 7: 205, 8: 230, 9: 310, 10: 335}

NOTIFY_EMAIL = 'mrugank.dake@nyu.edu'


# ══════════════════════════════════════════════════════════════════════════════
# Environment
# ══════════════════════════════════════════════════════════════════════════════

def get_compute_profile():
    h = socket.gethostname()
    if h == 'zod':    return 'mac',   4
    if h == 'vader':  return 'vader', 48
    return 'hpc', 10


def get_bids_root():
    h = socket.gethostname()
    if h == 'zod':    return '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    if h == 'vader':  return '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    return '/scratch/mdd9787/meg_prf_greene/MEG_HPC'


def send_email(subjID, voxRes, freq_band, success=True, msg=''):
    try:
        h = socket.gethostname()
        subj = f'[TGM] sub-{subjID:02d}/{freq_band} {"DONE" if success else "FAILED"} on {h}'
        body = (msg if not success else
                f'TGM decoding complete.\nSub={subjID:02d} | {freq_band} | {voxRes} | {h}')
        with smtplib.SMTP('localhost') as s:
            s.sendmail(NOTIFY_EMAIL, NOTIFY_EMAIL, f'Subject: {subj}\n\n{body}')
    except Exception as e:
        print(f'  (Email skipped: {e})')


# ══════════════════════════════════════════════════════════════════════════════
# Data loading (mirrors inSourceSpaceDecodingWithBehav.py)
# ══════════════════════════════════════════════════════════════════════════════

def open_h5(fpath, tmp_name):
    """Open h5py file with host-aware copy/locking strategy."""
    h = socket.gethostname()
    if h == 'zod':
        tmp = os.path.join('/Users/mrugank/Desktop', tmp_name)
        copyfile(fpath, tmp)
        f = h5py.File(tmp, 'r')
        os.remove(tmp)
        return f
    if h == 'vader':
        try:
            return h5py.File(fpath, 'r', locking=False)
        except Exception:
            tmp = os.path.join('/tmp', tmp_name)
            copyfile(fpath, tmp)
            f = h5py.File(tmp, 'r')
            os.remove(tmp)
            return f
    return h5py.File(fpath, 'r')


def load_source_space_data(subjID, bidsRoot, taskName, voxRes, freq_band='beta'):
    """Load, bandpass, baseline-correct, and downsample source data."""
    subName  = f'sub-{subjID:02d}'
    f_min, f_max = FREQUENCY_BANDS[freq_band]
    surface_resolution = int(voxRes[:-2])

    src_fpath = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon',
                             f'{subName}_task-{taskName}_sourceSpaceData_{surface_resolution}.mat')
    print(f'Loading {src_fpath} ...')

    src = open_h5(src_fpath, f'{subName}_task-{taskName}_sourceSpaceData_{surface_resolution}.mat')
    sg = src['sourcedataCombined']

    time_vector   = np.array(src[sg['time'][0, 0]]).flatten()
    target_labels = np.array(sg['trialinfo']).T[:, 1]

    all_trials = []
    td = sg['trial']
    for i in range(td.shape[0]):
        all_trials.append(np.array(src[td[i, 0]]))
    data_matrix = np.stack(all_trials, axis=0)   # (trials, times, sources)
    src.close()

    # Bandpass + Hilbert power
    dt     = float(np.mean(np.diff(time_vector)))
    nyquist = 0.5 / dt
    b, a   = signal.butter(4, [f_min / nyquist, f_max / nyquist], btype='band')

    n_trials, n_times, n_sources = data_matrix.shape
    power = np.zeros_like(data_matrix)
    for tr in range(n_trials):
        if tr % max(1, n_trials // 10) == 0:
            print(f'  Power: trial {tr}/{n_trials}')
        filt  = signal.filtfilt(b, a, data_matrix[tr], axis=0)  # (times, sources)
        power[tr] = np.abs(signal.hilbert(filt, axis=0)) ** 2

    # Baseline correction
    mean_power = power.mean(axis=0)
    power = power / (mean_power[np.newaxis] + 1e-12) - 1

    # Downsample: 25ms steps, 100ms averaging window
    target_dt        = 0.025
    half_win         = 0.050
    ds_factor        = max(1, int(target_dt / dt))
    half_win_samples = int(half_win / dt)
    n_ds             = n_times // ds_factor

    power_ds = np.zeros((n_trials, n_ds, n_sources))
    times_ds = np.zeros(n_ds)
    for i in range(n_ds):
        c = i * ds_factor
        s = max(0, c - half_win_samples)
        e = min(n_times, c + half_win_samples + 1)
        power_ds[:, i, :] = power[:, s:e, :].mean(axis=1)
        times_ds[i] = time_vector[c]

    del data_matrix, power
    gc.collect()
    return power_ds, target_labels, times_ds


# ══════════════════════════════════════════════════════════════════════════════
# TGM computation
# ══════════════════════════════════════════════════════════════════════════════

def _process_train_t(train_t, data_matrix, sin_targets, cos_targets):
    """
    Train SVR (LOO) at train_t, test on ALL test time points.
    Returns (train_t, pred_angles_deg) where pred_angles_deg is (n_trials, n_test_t).
    """
    n_trials, n_test_t, n_sources = data_matrix.shape

    X_train_all = data_matrix[:, train_t, :]         # (trials, sources)
    mu  = X_train_all.mean(axis=0)
    sd  = X_train_all.std(axis=0) + 1e-10
    X_z = (X_train_all - mu) / sd

    pred_angles = np.zeros((n_trials, n_test_t))

    svr_sin = SVR(kernel='rbf')
    svr_cos = SVR(kernel='rbf')

    for left_out in range(n_trials):
        train_mask = np.ones(n_trials, dtype=bool)
        train_mask[left_out] = False

        svr_sin.fit(X_z[train_mask], sin_targets[train_mask])
        svr_cos.fit(X_z[train_mask], cos_targets[train_mask])

        for test_t in range(n_test_t):
            x_test    = (data_matrix[left_out, test_t, :] - mu) / sd
            pred_sin  = svr_sin.predict(x_test.reshape(1, -1))[0]
            pred_cos  = svr_cos.predict(x_test.reshape(1, -1))[0]
            pred_angles[left_out, test_t] = np.degrees(
                np.mod(np.arctan2(pred_sin, pred_cos), 2 * np.pi))

    return train_t, pred_angles   # (n_trials, n_test_t)


def run_tgm(data_matrix, target_labels, control=False):
    """
    Full Temporal Generalization Matrix.
    Returns pred_angles of shape (n_trials, n_train_t, n_test_t).
    No error computation — do that downstream.
    """
    _, n_cores = get_compute_profile()

    if control:
        target_labels = np.random.permutation(target_labels)
        print('  [CONTROL] labels shuffled')

    angles_rad  = np.array([np.radians(ANGLE_MAPPING[int(t)]) for t in target_labels])
    sin_targets = np.sin(angles_rad)
    cos_targets = np.cos(angles_rad)

    n_trials, n_train_t, _ = data_matrix.shape
    print(f'  TGM: {n_train_t}×{n_train_t} | {n_trials} trials | {n_cores} cores')

    results = Parallel(n_jobs=n_cores, verbose=5)(
        delayed(_process_train_t)(tr_t, data_matrix, sin_targets, cos_targets)
        for tr_t in range(n_train_t)
    )

    # Assemble (n_trials, n_train_t, n_test_t)
    pred_angles = np.zeros((n_trials, n_train_t, n_train_t))
    for tr_t, preds in results:
        pred_angles[:, tr_t, :] = preds

    return pred_angles


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(subjID, voxRes='8mm', freq_band='beta'):
    taskName  = 'mgs'
    bidsRoot  = get_bids_root()
    _, n_cores = get_compute_profile()

    decoding_dir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', 'decodingVC')
    os.makedirs(decoding_dir, exist_ok=True)
    results_file = os.path.join(decoding_dir, f'sub-{subjID:02d}_task-{taskName}_TGM_{freq_band}_{voxRes}.pkl')

    print(f'TGM decoding | sub-{subjID:02d} | {freq_band} | {voxRes} | {socket.gethostname()} | {n_cores} cores')

    try:
        # ── Load data ──────────────────────────────────────────────────────────
        data_matrix, target_labels, time_vector = load_source_space_data(
            subjID, bidsRoot, taskName, voxRes, freq_band=freq_band)

        # ── Behavioral data ────────────────────────────────────────────────────
        behav_path = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'eyetracking',
                                  f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')
        behav = open_h5(behav_path, f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')
        i_sacc_err   = np.array(behav['ii_sess_forSource']['i_sacc_err']).flatten()
        i_sacc_raw   = np.array(behav['ii_sess_forSource']['i_sacc_raw'])
        behav.close()

        # Saccade angle from (x,y)
        x_c = i_sacc_raw[0] if i_sacc_raw.shape[0] == 2 else i_sacc_raw[:, 0]
        y_c = i_sacc_raw[1] if i_sacc_raw.shape[0] == 2 else i_sacc_raw[:, 1]
        ref_angle = np.arctan2(0, 5)
        i_sacc_angle = (np.degrees(np.arctan2(y_c, x_c) - ref_angle) + 360) % 360

        # Filter NaN trials
        valid = ~np.isnan(i_sacc_err)
        data_matrix   = data_matrix[valid]
        target_labels = target_labels[valid]
        i_sacc_err    = i_sacc_err[valid]
        i_sacc_angle  = i_sacc_angle[valid]

        # ── Atlas ROIs ─────────────────────────────────────────────────────────
        atlas_fpath = os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat')
        atlas = loadmat(atlas_fpath)
        def roi(key): return np.where(np.array(atlas[key]).flatten() == 1)[0]

        regions = {
            'visual':   data_matrix[:, :, roi('visual_points')],
            'parietal': data_matrix[:, :, roi('parietal_points')],
            'frontal':  data_matrix[:, :, roi('frontal_points')],
        }
        del data_matrix
        gc.collect()

        # ── TGM per region ─────────────────────────────────────────────────────
        tgm_results = {}
        for region_name, roi_data in regions.items():
            print(f'\n── {region_name.capitalize()} ROI ──')
            print('  Real...')
            pred_real    = run_tgm(roi_data, target_labels, control=False)
            print('  Control...')
            pred_control = run_tgm(roi_data, target_labels, control=True)
            # (n_trials, n_train_t, n_test_t) — mirrors inSourceSpaceDecodingWithBehav
            tgm_results[f'pred_angles_deg_{region_name}']         = pred_real
            tgm_results[f'pred_angles_deg_{region_name}_control'] = pred_control

        # ── Save ───────────────────────────────────────────────────────────────
        output = {
            **tgm_results,
            'time_vector':   time_vector,
            'target_labels': target_labels,
            'freq_band':     freq_band,
            'freq_range':    FREQUENCY_BANDS[freq_band],
            'voxRes':        voxRes,
            'i_sacc_err':    i_sacc_err,
            'i_sacc_angle':  i_sacc_angle,
        }
        with open(results_file, 'wb') as f:
            pickle.dump(output, f)
        print(f'\nSaved: {results_file}')
        send_email(subjID, voxRes, freq_band, success=True)

    except Exception:
        import traceback
        err = traceback.format_exc()
        print(err)
        send_email(subjID, voxRes, freq_band, success=False, msg=err)
        raise


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python temporalGeneralizationDecoding.py <subjID> [voxRes] [freq_band]')
        print(f'Bands: {list(FREQUENCY_BANDS.keys())}')
        sys.exit(1)
    subjID    = int(sys.argv[1])
    voxRes    = sys.argv[2] if len(sys.argv) > 2 else '8mm'
    freq_band = sys.argv[3] if len(sys.argv) > 3 else 'beta'
    main(subjID, voxRes, freq_band)
