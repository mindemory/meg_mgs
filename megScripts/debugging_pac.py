import os, h5py, socket, gc, pickle
import numpy as np
from shutil import copyfile
from scipy.io import loadmat
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

# Compute profile mapping
def get_compute_profile():
    h = socket.gethostname()
    if h == 'zod': return 'mac', 4, '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    elif h == 'vader': return 'vader', 48, '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else: return 'hpc', 10, '/scratch/mdd9787/meg_prf_greene/MEG_HPC'

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
    l_front, r_front = [np.where(atlas_data[f'{s}_frontal_points'].flatten() == 1)[0] for s in ['left', 'right']]
    
    V_L, V_R = [data_matrix[:, :, p].mean(axis=2) for p in [l_vis, r_vis]]
    F_L, F_R = [data_matrix[:, :, p].mean(axis=2) for p in [l_front, r_front]]

    n_trials, n_times = V_L.shape
    roi_names = ['Ipsi-Vis', 'Contra-Vis', 'Ipsi-Frontal', 'Contra-Frontal']
    func_rois = {k: np.zeros((n_trials, n_times)) for k in roi_names}
    for i in range(n_trials):
        if left_tgt[i]:
            func_rois['Ipsi-Vis'][i], func_rois['Contra-Vis'][i] = V_L[i], V_R[i]
            func_rois['Ipsi-Frontal'][i], func_rois['Contra-Frontal'][i] = F_L[i], F_R[i]
        else:
            func_rois['Ipsi-Vis'][i], func_rois['Contra-Vis'][i] = V_R[i], V_L[i]
            func_rois['Ipsi-Frontal'][i], func_rois['Contra-Frontal'][i] = F_R[i], F_L[i]
    return func_rois, dt, time_vector

def get_single_subj_pac(subjID, voxRes):
    profile, n_cores, bidsRoot = get_compute_profile()
    try:
        func_rois, dt, time_vector = load_and_prepare_functional_rois(subjID, bidsRoot, voxRes)
    except Exception as e:
        print(f"[!] Error loading Sub-{subjID}: {e}")
        return None
    
    sfreq = 1/dt
    nyq = sfreq/2
    f_beta, f_gamma = (4.0, 8.0), (13.0, 30.0) 
    b_b, a_b = signal.butter(4, [f_beta[0]/nyq, f_beta[1]/nyq], btype='band')
    b_g, a_g = signal.butter(4, [f_gamma[0]/nyq, f_gamma[1]/nyq], btype='band')
    
    windows = [('Stimulus', 0.0, 0.3), ('Delay', 0.5, 1.5)]
    roi_list = ['Ipsi-Vis', 'Contra-Vis', 'Ipsi-Frontal', 'Contra-Frontal']
    
    up_factor = 10
    sfreq_up = sfreq * up_factor
    f_mid = np.mean(f_beta)
    cushion_up = int(2.0 * (1.0/f_mid) * sfreq_up)
    
    subj_results = {roi: {} for roi in roi_list}
    
    for roi_name in roi_list:
        if roi_name not in func_rois: continue
        roi_data = func_rois[roi_name]
        f_beta_sig = signal.filtfilt(b_b, a_b, roi_data, axis=1)
        f_gamma_sig = signal.filtfilt(b_g, a_g, roi_data, axis=1)
        gamma_pow = np.abs(signal.hilbert(f_gamma_sig, axis=1))**2
        
        n_up = roi_data.shape[1] * up_factor
        f_beta_up = signal.resample(f_beta_sig, n_up, axis=1)
        gamma_pow_up = signal.resample(gamma_pow, n_up, axis=1)
        
        for w_name, w_start, w_end in windows:
            t_win = np.where((time_vector >= w_start) & (time_vector <= w_end))[0]
            snippets_b, snippets_g = [], []
            for tr in range(f_beta_sig.shape[0]):
                w_beta = -f_beta_sig[tr, t_win[0]:t_win[-1]+1]
                peaks, _ = signal.find_peaks(w_beta, distance=int(sfreq/f_beta[1]))
                for p in peaks:
                    g_up = (p + t_win[0]) * up_factor
                    s, e = g_up - cushion_up, g_up + cushion_up + 1
                    if s >= 0 and e <= f_beta_up.shape[1]:
                        snippets_b.append(f_beta_up[tr, s:e])
                        snippets_g.append(gamma_pow_up[tr, s:e])
            
            if snippets_b:
                # Local Normalization: Cycle Modulation Depth (%)
                avg_b_raw, avg_g_raw = np.mean(snippets_b, axis=0), np.mean(snippets_g, axis=0)
                gamma_mod_raw = (avg_g_raw - np.mean(avg_g_raw)) / np.mean(avg_g_raw) * 100
                
                avg_b = signal.resample(avg_b_raw, 50)
                gamma_mod = signal.resample(gamma_mod_raw, 50)
                
                subj_results[roi_name][w_name] = {'beta': avg_b, 'gamma': gamma_mod}
                    
    return subj_results

def debug_pac_grand_average(subj_list, voxRes='8mm'):
    profile, n_cores, bidsRoot = get_compute_profile()
    roi_list = ['Ipsi-Vis', 'Contra-Vis', 'Ipsi-Frontal', 'Contra-Frontal']
    windows = ['Stimulus', 'Delay']
    
    # Collection: {ROI: {Window: {'beta': [], 'gamma': []}}}
    group_data = {roi: {win: {'beta': [], 'gamma': []} for win in windows} for roi in roi_list}
    
    for s in subj_list:
        print(f"[*] Processing Sub-{s:02d}...")
        res = get_single_subj_pac(s, voxRes)
        if res is None: continue
        for roi in roi_list:
            for win in windows:
                if win in res[roi]:
                    group_data[roi][win]['beta'].append(res[roi][win]['beta'])
                    group_data[roi][win]['gamma'].append(res[roi][win]['gamma'])
                    
    # Aggregation (Mean & SEM)
    stats = {roi: {win: {} for win in windows} for roi in roi_list}
    g_min, g_max = 0, 0
    b_min, b_max = 0, 0
    
    for roi in roi_list:
        for win in windows:
            betas, gammas = group_data[roi][win]['beta'], group_data[roi][win]['gamma']
            if not betas: continue
            
            # Gamma Stats
            g_mat = np.stack(gammas)
            g_mean, g_sem = np.mean(g_mat, axis=0), np.std(g_mat, axis=0) / np.sqrt(len(subj_list))
            
            # Beta Stats
            b_mat = np.stack(betas)
            b_mean, b_sem = np.mean(b_mat, axis=0), np.std(b_mat, axis=0) / np.sqrt(len(subj_list))
            
            stats[roi][win] = {'g_mean': g_mean, 'g_sem': g_sem, 'b_mean': b_mean, 'b_sem': b_sem}
            g_min, g_max = min(g_min, np.min(g_mean - g_sem)), max(g_max, np.max(g_mean + g_sem))
            b_min, b_max = min(b_min, np.min(b_mean - b_sem)), max(b_max, np.max(b_mean + b_sem))

    # Plotting
    fig = plt.figure(figsize=(24, 14))
    gs = fig.add_gridspec(4, 4, height_ratios=[3, 1, 3, 1], hspace=0.4, wspace=0.3)
    g_lim = [g_min - abs(g_max-g_min)*0.1, g_max + abs(g_max-g_min)*0.1]
    b_lim = [b_min - abs(b_max-b_min)*0.1, b_max + abs(b_max-b_min)*0.1]
    for r_idx, roi in enumerate(roi_list):
        hemi = 'Ipsi' if 'Ipsi' in roi else 'Contra'
        row_set = 0 if 'Vis' in roi else 1
        for w_idx, win in enumerate(windows):
            if not stats[roi][win]: continue
            col = (0 if hemi == 'Ipsi' else 2) + w_idx
            ax_g = fig.add_subplot(gs[row_set*2, col])
            ax_b = fig.add_subplot(gs[row_set*2+1, col])
            
            s = stats[roi][win]
            p_ax = np.linspace(-2*np.pi, 2*np.pi, len(s['g_mean']))
            
            # Gamma
            ax_g.plot(p_ax, s['g_mean'], color='crimson', linewidth=2.5, marker='.')
            ax_g.fill_between(p_ax, s['g_mean']-s['g_sem'], s['g_mean']+s['g_sem'], color='crimson', alpha=0.2)
            ax_g.axvline(0, color='black', alpha=0.2, linestyle='--')
            ax_g.axhline(0, color='black', alpha=0.3, linestyle='--')
            ax_g.set_ylim(g_lim)
            ax_g.set_title(f'{roi}\n({win})', fontsize=14, fontweight='bold')
            if col in [0, 2]: ax_g.set_ylabel('% Change', fontsize=12)
            ax_g.set_xticks([])
            
            # Beta
            ax_b.plot(p_ax, s['b_mean'], color='royalblue', linewidth=2, marker='.')
            ax_b.fill_between(p_ax, s['b_mean']-s['b_sem'], s['b_mean']+s['b_sem'], color='royalblue', alpha=0.2)
            ax_b.axvline(0, color='black', alpha=0.2, linestyle='--')
            ax_b.set_ylim(b_lim)
            if col in [0, 2]: ax_b.set_ylabel('Theta', fontsize=12)
            ax_b.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi], [r'$-2\pi$', r'$-\pi$', '0', r'$\pi$', r'$2\pi$'])
            for ax in [ax_g, ax_b]: ax.grid(False)

    if len(subj_list) > 5:
        subj_str = f"N{len(subj_list)}"
    else:
        subj_str = f"1-{subj_list[-1]}" if len(subj_list) > 1 else str(subj_list[0])
        
    plt.suptitle(f'Grand Average PAC Diagnostic (Sub-{subj_str}, {voxRes})\nStimulus (0-0.3s) vs. Delay (0.5-1.5s) | Shading = SEM', fontsize=20, y=0.98)
    out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'pac_debugging')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'sub-GA{subj_str}_pac_diagnostic_{voxRes}.png')
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    # Add save as svg
    out_file_svg = os.path.join(out_dir, f'sub-GA{subj_str}_pac_diagnostic_{voxRes}.svg')
    plt.savefig(out_file_svg, bbox_inches='tight')
    print(f"[*] Saved Grand Average PAC Diagnostic: {out_file}")

GA_SUBJ_LIST = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
# GA_SUBJ_LIST = [1, 2, 3]

if __name__ == '__main__':
    # Usage: python3 debugging_pac.py [subj_range, subj_list or 'all'] [voxRes]
    # Examples: all, 1-5, 01,02,03, 8mm
    arg = sys.argv[1] if len(sys.argv) > 1 else 'all'
    res = sys.argv[2] if len(sys.argv) > 2 else '8mm'
    
    if arg == 'all':
        debug_pac_grand_average(GA_SUBJ_LIST, res)
    elif ',' in arg:
        subj_list = [int(s) for s in arg.split(',')]
        debug_pac_grand_average(subj_list, res)
    elif '-' in arg:
        start, end = map(int, arg.split('-'))
        subj_list = list(range(start, end + 1))
        debug_pac_grand_average(subj_list, res)
    else:
        debug_pac_grand_average([int(arg)], res)
