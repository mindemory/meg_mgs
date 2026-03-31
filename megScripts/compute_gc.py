import os, h5py, socket, sys, smtplib, gc, pickle
import numpy as np
from shutil import copyfile
from scipy.io import loadmat
import itertools, warnings
from joblib import Parallel, delayed

try:
    from mne_connectivity import spectral_connectivity_epochs
except ImportError:
    pass

# Email config for completion notification
NOTIFY_EMAIL = 'mrugank.dake@nyu.edu'

def get_compute_profile():
    h = socket.gethostname()
    if h == 'zod':
        return 'mac', 4
    elif h == 'vader':
        return 'vader', 48
    else:
        return 'hpc', 10

def is_local():
    return socket.gethostname() in ('zod', 'vader')

def send_completion_email(subjID, voxRes, figures_dir, success=True, error_msg=None):
    try:
        hostname = socket.gethostname()
        if success:
            subject = f'[GC] sub-{subjID:02d} {voxRes} DONE on {hostname}'
            body = (f'Granger Causality analysis complete!\n\nSubject: {subjID:02d}\nResolution: {voxRes}')
        else:
            subject = f'[GC] sub-{subjID:02d} {voxRes} FAILED on {hostname}'
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

    if socket.gethostname() == 'zod':
        source_data_temp_path = os.path.join('/Users/mrugank/Desktop', f'{subName}_task-{taskName}_sourceSpaceData_raw_{surface_resolution}.mat')
        copyfile(source_data_fpath, source_data_temp_path)
        source_data = h5py.File(source_data_temp_path, 'r')
        os.remove(source_data_temp_path)
    elif socket.gethostname() == 'vader':
        try:
            source_data = h5py.File(source_data_fpath, 'r', locking=False)
        except Exception:
            source_data_temp_path = os.path.join('/tmp', f'{subName}_task-{taskName}_sourceSpaceData_raw_{surface_resolution}.mat')
            copyfile(source_data_fpath, source_data_temp_path)
            source_data = h5py.File(source_data_temp_path, 'r')
            os.remove(source_data_temp_path)
    else:
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

    data_matrix = np.stack(all_trials, axis=0)

    print("Filtering valid trials using behavioral data...")
    behav_data_path = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'eyetracking',
                                   f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')

    if socket.gethostname() == 'zod':
        behav_data_temp_path = os.path.join('/Users/mrugank/Desktop', f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')
        copyfile(behav_data_path, behav_data_temp_path)
        behav_data = h5py.File(behav_data_temp_path, 'r')
        os.remove(behav_data_temp_path)
    elif socket.gethostname() == 'vader':
        try:
            behav_data = h5py.File(behav_data_path, 'r', locking=False)
        except Exception:
            behav_data_temp_path = os.path.join('/tmp', f'sub-{subjID:02d}_task-{taskName}-iisess_forSource.mat')
            copyfile(behav_data_path, behav_data_temp_path)
            behav_data = h5py.File(behav_data_temp_path, 'r')
            os.remove(behav_data_temp_path)
    else:
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

def _compute_chunk(roi1, roi2, pair_indices, sfreq, fmin, fmax):
    warnings.filterwarnings('ignore') # Ignore internal MNE warnings in workers
    
    gc_1to2 = []
    gc_2to1 = []
    freqs = None
    
    for i, j in pair_indices:
        d_seed = roi1[:, :, i]
        d_tgt  = roi2[:, :, j]
        
        # Standardize & regularize per dipole
        s1_std = np.std(d_seed, axis=1, keepdims=True); s1_std[s1_std==0] = 1.0
        s2_std = np.std(d_tgt, axis=1, keepdims=True); s2_std[s2_std==0] = 1.0
        d_seed = (d_seed - np.mean(d_seed, axis=1, keepdims=True)) / s1_std + np.random.normal(0, 1e-3, d_seed.shape)
        d_tgt = (d_tgt - np.mean(d_tgt, axis=1, keepdims=True)) / s2_std + np.random.normal(0, 1e-3, d_tgt.shape)
        
        data = np.stack([d_seed, d_tgt], axis=1) # (epochs, 2, times)
        try:
            con = spectral_connectivity_epochs(
                data, method='gc', sfreq=sfreq, fmin=fmin, fmax=fmax,
                mode='multitaper', indices=([[0], [1]], [[1], [0]]), n_jobs=1, verbose=False
            )
            vals = con.get_data() # (2, n_freqs) -> 0: seed->tgt, 1: tgt->seed
            gc_1to2.append(vals[0])  
            gc_2to1.append(vals[1])  
            if freqs is None:
                freqs = con.freqs
        except Exception as e:
            # We skip this pair if it's too unstable (rank info suppressed)
            pass
            
    if not gc_1to2:
        return None, None, None, 0
    
    # Pre-sum locally to minimize payload sent back to main process
    s_1to2 = np.sum(np.stack(gc_1to2), axis=0)
    s_2to1 = np.sum(np.stack(gc_2to1), axis=0)
    return s_1to2, s_2to1, freqs, len(gc_1to2)


def get_bivariate_gc_for_roi_pair(roi1, roi2, sfreq, fmin, fmax, n_jobs=48):
    n_dipoles1 = roi1.shape[2]
    n_dipoles2 = roi2.shape[2]
    
    # Generate all pairwise combinations
    all_pairs = list(itertools.product(range(n_dipoles1), range(n_dipoles2)))
    chunks = np.array_split(all_pairs, n_jobs)
    
    # We use backend='loky' to avoid Python multiprocessing passing huge contiguous arrays 
    # if joblib detects threading is insufficient. MNE/scipy releases GIL during lapack calls.
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_compute_chunk)(roi1, roi2, chunk, sfreq, fmin, fmax) 
        for chunk in chunks if len(chunk) > 0
    )
    
    total_1to2 = None
    total_2to1 = None
    total_count = 0
    freqs = None
    
    for sum12, sum21, fqs, count in results:
        if count > 0:
            if total_1to2 is None:
                total_1to2 = sum12
                total_2to1 = sum21
                freqs = fqs
            else:
                total_1to2 += sum12
                total_2to1 += sum21
            total_count += count
            
    if total_count == 0:
        return None, None, None
        
    # Global average across all valid dipole pairs
    return total_1to2 / total_count, total_2to1 / total_count, freqs


def get_condition_gc(l_vis, r_vis, l_front, r_front, tgt_mask, time_vector, dt, t_start, t_end, fmin=5, fmax=50, n_jobs=48):
    sfreq = 1 / dt
    
    # Extract stationary time window FIRST (keeps original dipoles un-pooled)
    t_mask = (time_vector >= t_start) & (time_vector <= t_end)
    
    # Shape: (n_epochs, n_times, n_dipoles)
    epoch_lv = l_vis[tgt_mask][:, t_mask, :]
    epoch_rv = r_vis[tgt_mask][:, t_mask, :]
    epoch_lf = l_front[tgt_mask][:, t_mask, :]
    epoch_rf = r_front[tgt_mask][:, t_mask, :]
    
    if epoch_lv.size == 0 or epoch_lv.shape[0] < 2:
        return None, None
        
    print(f"      (Parallel pairwise GC across {epoch_lf.shape[2]*epoch_lv.shape[2]:,} LF->LV inter-parcel pairs)")
    
    lf_lv, lv_lf, freqs = get_bivariate_gc_for_roi_pair(epoch_lf, epoch_lv, sfreq, fmin, fmax, n_jobs=n_jobs)
    lf_rv, rv_lf, _     = get_bivariate_gc_for_roi_pair(epoch_lf, epoch_rv, sfreq, fmin, fmax, n_jobs=n_jobs)
    rf_lv, lv_rf, _     = get_bivariate_gc_for_roi_pair(epoch_rf, epoch_lv, sfreq, fmin, fmax, n_jobs=n_jobs)
    rf_rv, rv_rf, _     = get_bivariate_gc_for_roi_pair(epoch_rf, epoch_rv, sfreq, fmin, fmax, n_jobs=n_jobs)
    
    res = {
        'lf_lv': lf_lv, 'lf_rv': lf_rv,
        'rf_lv': rf_lv, 'rf_rv': rf_rv,
        'lv_lf': lv_lf, 'lv_rf': lv_rf,
        'rv_lf': rv_lf, 'rv_rf': rv_rf
    }
    
    return res, freqs

def main(subjID, voxRes='10mm'):
    h = socket.gethostname()
    host_name, core_count = get_compute_profile()
    
    if h == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    elif h == 'vader':
        bidsRoot = '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'

    try:
        roi_dict, left_tgt_mask, right_tgt_mask, dt, time_vector = load_and_prepare_data(
            subjID, bidsRoot, 'mgs', voxRes)

        intervals = [
            ('Stimulus', -0.2, 0.5),
            ('Delay', 0.5, 1.5)
        ]
        
        window_freqs = {}
        # Results dictionary: window -> condition_target -> {lf_lv, ...}
        results = {}
        for interval_name, t_start, t_end in intervals:
            print(f"\nProcessing Window: {interval_name} [{t_start}s to {t_end}s]")
            results[interval_name] = {}
            
            # Map combinations to LEFT and RIGHT stimulus targets
            l_vis, r_vis = roi_dict['Visual']['left_roi'], roi_dict['Visual']['right_roi']
            l_front, r_front = roi_dict['Frontal']['left_roi'], roi_dict['Frontal']['right_roi']

            for tgt_name, mask in [('TargetLeft', left_tgt_mask), ('TargetRight', right_tgt_mask)]:
                res_dict, fqs = get_condition_gc(l_vis, r_vis, l_front, r_front, mask, time_vector, dt, t_start, t_end, n_jobs=core_count)
                if fqs is not None and interval_name not in window_freqs:
                    window_freqs[interval_name] = fqs
                results[interval_name][tgt_name] = res_dict
                print(f"  Condition: {tgt_name}")

        # Map to stimulus-relative naming in output
        gc_output = {}
        for interval_name, _, _ in intervals:
            L = results[interval_name]['TargetLeft']
            R = results[interval_name]['TargetRight']
            
            def safe_merge(val_l, val_r):
                if val_l is None and val_r is None: return None
                if val_l is None: return val_r
                if val_r is None: return val_l
                return (val_l + val_r) / 2.0
            
            # Frontal -> Visual
            # IpsiF -> IpsiV: (TargetLeft: LF->LV) + (TargetRight: RF->RV)
            if2iv = safe_merge(L['lf_lv'] if L else None, R['rf_rv'] if R else None)
            # IpsiF -> ContraV: (TargetLeft: LF->RV) + (TargetRight: RF->LV)
            if2cv = safe_merge(L['lf_rv'] if L else None, R['rf_lv'] if R else None)
            # ContraF -> IpsiV: (TargetLeft: RF->LV) + (TargetRight: LF->RV)
            cf2iv = safe_merge(L['rf_lv'] if L else None, R['lf_rv'] if R else None)
            # ContraF -> ContraV: (TargetLeft: RF->RV) + (TargetRight: LF->LV)
            cf2cv = safe_merge(L['rf_rv'] if L else None, R['lf_lv'] if R else None)
            
            # Visual -> Frontal
            # IpsiV -> IpsiF: (TargetLeft: LV->LF) + (TargetRight: RV->RF)
            iv2if = safe_merge(L['lv_lf'] if L else None, R['rv_rf'] if R else None)
            # IpsiV -> ContraF: (TargetLeft: LV->RF) + (TargetRight: RV->LF)
            iv2cf = safe_merge(L['lv_rf'] if L else None, R['rv_lf'] if R else None)
            # ContraV -> IpsiF: (TargetLeft: RV->LF) + (TargetRight: LV->RF)
            cv2if = safe_merge(L['rv_lf'] if L else None, R['lv_rf'] if R else None)
            # ContraV -> ContraF: (TargetLeft: RV->RF) + (TargetRight: LV->LF)
            cv2cf = safe_merge(L['rv_rf'] if L else None, R['lv_lf'] if R else None)
            
            gc_output[interval_name] = {
                'f2v': {'if2iv': if2iv, 'if2cv': if2cv, 'cf2iv': cf2iv, 'cf2cv': cf2cv},
                'v2f': {'iv2if': iv2if, 'iv2cf': iv2cf, 'cv2if': cv2if, 'cv2cf': cv2cf}
            }
            
        # Compile standardized group cache
        gc_cache = {
            'window_freqs': window_freqs,
            'windows': [i[0] for i in intervals],
            'gc_data': gc_output
        }

        output_dir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', 'gc_data')
        os.makedirs(output_dir, exist_ok=True)
        
        out_file = os.path.join(output_dir, f'sub-{subjID:02d}_task-mgs_GC_{voxRes}.pkl')
        with open(out_file, 'wb') as f:
            pickle.dump(gc_cache, f)
            
        print(f"\nDone! Saved GC cache: {out_file}")
        send_completion_email(subjID, voxRes, 'Cached sequentially to .pkl', success=True)

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        print(f"\nScript failed:\n{err}")
        send_completion_email(subjID, voxRes, 'FAILED', success=False, error_msg=err)
        raise

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compute_gc.py <subjID> [voxRes]")
        sys.exit(1)

    subjID = int(sys.argv[1])
    voxRes = sys.argv[2] if len(sys.argv) > 2 else '10mm'

    main(subjID, voxRes)
