import os, h5py, socket, gc, pickle
import numpy as np
from shutil import copyfile
from scipy.io import loadmat
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from joblib import Parallel, delayed

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
    # Also store raw hemi signals for cross-signal PAC in Fig 2
    raw_rois = {}
    for i in range(n_trials):
        if left_tgt[i]:
            func_rois['Ipsi-Vis'][i], func_rois['Contra-Vis'][i] = V_L[i], V_R[i]
            func_rois['Ipsi-Frontal'][i], func_rois['Contra-Frontal'][i] = F_L[i], F_R[i]
        else:
            func_rois['Ipsi-Vis'][i], func_rois['Contra-Vis'][i] = V_R[i], V_L[i]
            func_rois['Ipsi-Frontal'][i], func_rois['Contra-Frontal'][i] = F_R[i], F_L[i]
    return func_rois, dt, time_vector


def _peak_triggered_avg(phase_sig, amp_pow, sfreq, phase_band, time_vector, windows, up_factor=10):
    """
    Core PAC computation: find troughs of phase_sig (peaks of -phase_sig),
    extract snippets of amp_pow centred on those troughs.
    
    phase_sig  : (n_trials, n_times) - already band-filtered phase signal
    amp_pow    : (n_trials, n_times) - already band-filtered + Hilbert amplitude^2
    Both arrays must share the same trials and time axes.
    
    Returns dict {window_name: {'beta': avg_cycle, 'gamma': gamma_mod}}
    """
    nyq = sfreq / 2
    f_mid = np.mean(phase_band)
    sfreq_up = sfreq * up_factor

    cushion_up = int(2.0 * (1.0 / f_mid) * sfreq_up)

    n_up = phase_sig.shape[1] * up_factor
    phase_up = signal.resample(phase_sig, n_up, axis=1)
    amp_up   = signal.resample(amp_pow,   n_up, axis=1)

    results = {}
    for w_name, w_start, w_end in windows:
        t_win = np.where((time_vector >= w_start) & (time_vector <= w_end))[0]
        snippets_b, snippets_g = [], []
        for tr in range(phase_sig.shape[0]):
            w_beta = -phase_sig[tr, t_win[0]:t_win[-1]+1]
            peaks, _ = signal.find_peaks(w_beta, distance=int(sfreq / phase_band[1]))
            for p in peaks:
                g_up = (p + t_win[0]) * up_factor
                s, e = g_up - cushion_up, g_up + cushion_up + 1
                if s >= 0 and e <= phase_up.shape[1]:
                    snippets_b.append(phase_up[tr, s:e])
                    snippets_g.append(amp_up[tr, s:e])

        if snippets_b:
            avg_b_raw = np.mean(snippets_b, axis=0)
            avg_g_raw = np.mean(snippets_g, axis=0)
            gamma_mod_raw = (avg_g_raw - np.mean(avg_g_raw)) / np.mean(avg_g_raw) * 100
            avg_b   = signal.resample(avg_b_raw,   50)
            gamma_mod = signal.resample(gamma_mod_raw, 50)
            results[w_name] = {'beta': avg_b, 'gamma': gamma_mod}
    return results


def compute_tort_mi(phase, amplitude, n_bins=18):
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(phase, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    
    bin_sums = np.bincount(bin_idx, weights=amplitude, minlength=n_bins)
    bin_counts = np.bincount(bin_idx, minlength=n_bins)
    
    mask = bin_counts > 0
    p = np.zeros(n_bins)
    p[mask] = bin_sums[mask] / bin_counts[mask]
    
    if np.sum(p) == 0: return 0
    p /= np.sum(p)
    h = -np.sum(p[p > 0] * np.log(p[p > 0] + 1e-12))
    return (np.log(n_bins) - h) / np.log(n_bins)


def _compute_lagged_mi(phase_angle, amp_env, sfreq, time_vector, windows, lag_range=(-125, 125), lag_step=5, n_surrogates=200):
    """
    Computes Z-scored MI across a range of time lags.
    """
    lags_ms = np.arange(lag_range[0], lag_range[1] + lag_step, lag_step)
    lags_samples = (lags_ms * sfreq / 1000.0).astype(int)
    
    results = {}
    for w_name, w_start, w_end in windows:
        t_win = np.where((time_vector >= w_start) & (time_vector <= w_end))[0]
        
        mi_vals = []
        for shift in lags_samples:
            # Base indices for the window
            t_indices = t_win
            
            # shifted indices for amplitude
            a_indices = t_indices + shift
            
            # keep only indices where shifted amplitude is within signal bounds
            valid_mask = (a_indices >= 0) & (a_indices < amp_env.shape[1])
            
            p_idx = t_indices[valid_mask]
            a_idx = a_indices[valid_mask]
            
            if len(p_idx) == 0:
                mi_vals.append(np.nan)
                continue
            
            # Extract and flatten across all trials
            p_slice = phase_angle[:, p_idx].flatten()
            a_slice = amp_env[:, a_idx].flatten()
            
            # 1. Observed MI
            observed_mi = compute_tort_mi(p_slice, a_slice)
            
            # 2. Surrogate Distribution
            n_samples = len(a_slice)
            shuffled_mis = np.zeros(n_surrogates)
            # Pre-generate random shifts (excluding 0 to ensure true shift)
            random_shifts = np.random.randint(1, n_samples, size=n_surrogates)
            for i in range(n_surrogates):
                a_shuffled = np.roll(a_slice, random_shifts[i])
                shuffled_mis[i] = compute_tort_mi(p_slice, a_shuffled)
            
            # 3. Z-score
            z_mi = (observed_mi - np.mean(shuffled_mis)) / (np.std(shuffled_mis) + 1e-12)
            mi_vals.append(z_mi)
            
        results[w_name] = {'lags': lags_ms, 'mi': np.array(mi_vals)}
        
    return results


def get_single_subj_pac(subjID, voxRes):
    """
    Figure 1: 
      - PFC rows   → phase of theta (4-8 Hz) from the same PFC ROI
      - Visual rows → phase of alpha (8-12 Hz) from the same Visual ROI
    Amplitude signal = beta (13-30 Hz) from the same ROI in all cases.
    """
    profile, n_cores, bidsRoot = get_compute_profile()
    try:
        func_rois, dt, time_vector = load_and_prepare_functional_rois(subjID, bidsRoot, voxRes)
    except Exception as e:
        print(f"[!] Error loading Sub-{subjID}: {e}")
        return None, None

    sfreq = 1 / dt
    nyq = sfreq / 2

    f_theta = (4.0,  8.0)
    f_alpha = (8.0, 12.0)
    f_beta  = (13.0, 30.0)

    b_theta, a_theta = signal.butter(4, [f_theta[0]/nyq, f_theta[1]/nyq], btype='band')
    b_alpha, a_alpha = signal.butter(4, [f_alpha[0]/nyq, f_alpha[1]/nyq], btype='band')
    b_beta,  a_beta  = signal.butter(4, [f_beta[0]/nyq,  f_beta[1]/nyq],  btype='band')

    windows  = [('Stimulus', 0.0, 0.3), ('Delay', 0.5, 1.5)]
    roi_list = ['Ipsi-Vis', 'Contra-Vis', 'Ipsi-Frontal', 'Contra-Frontal']

    subj_fig1 = {roi: {} for roi in roi_list}

    for roi_name in roi_list:
        if roi_name not in func_rois: continue
        roi_data = func_rois[roi_name]

        # Choose phase band: alpha for visual, theta for frontal
        is_visual = 'Vis' in roi_name
        if is_visual:
            phase_sig = signal.filtfilt(b_alpha, a_alpha, roi_data, axis=1)
            phase_band = f_alpha
        else:
            phase_sig = signal.filtfilt(b_theta, a_theta, roi_data, axis=1)
            phase_band = f_theta

        amp_filt = signal.filtfilt(b_beta, a_beta, roi_data, axis=1)
        amp_pow  = np.abs(signal.hilbert(amp_filt, axis=1))**2

        subj_fig1[roi_name] = _peak_triggered_avg(
            phase_sig, amp_pow, sfreq, phase_band, time_vector, windows
        )

    # ---------------------------------------------------------------
    # Figure 2: Cross-signal PAC
    #   Row 1 (Cross-hemisphere): PFC theta from *contra* hemisphere → Visual beta
    #   Row 2 (Same-hemisphere):  PFC theta from *ipsi*  hemisphere → Visual beta
    #   "Contra PFC → Visual" means Contra-Frontal phase drives Contra-Vis amplitude
    #     and  Ipsi-Frontal phase drives Ipsi-Vis amplitude
    #   "Ipsi PFC → Visual"  means Ipsi-Frontal phase drives Contra-Vis amplitude
    #     and  Contra-Frontal phase drives Ipsi-Vis amplitude
    # ---------------------------------------------------------------
    # We compute four pairs, then average across ipsi/contra to get two final traces.

    # 4 individual PFC→Visual pairs for Figure 2 (no averaging)
    fig2_pairs = [
        ('Contra-Frontal', 'Contra-Vis'),   # pair 1
        ('Contra-Frontal', 'Ipsi-Vis'),     # pair 2
        ('Ipsi-Frontal',   'Contra-Vis'),   # pair 3
        ('Ipsi-Frontal',   'Ipsi-Vis'),     # pair 4
    ]
    fig2_keys = [
        'Contra-PFC→Contra-Vis',
        'Contra-PFC→Ipsi-Vis',
        'Ipsi-PFC→Contra-Vis',
        'Ipsi-PFC→Ipsi-Vis',
    ]
    subj_fig2 = {k: {} for k in fig2_keys}
    subj_fig3 = {k: {} for k in fig2_keys}

    # Pre-filter all ROIs once for reuse
    filtered = {}
    for roi_name in roi_list:
        if roi_name not in func_rois: continue
        d = func_rois[roi_name]
        theta_filt = signal.filtfilt(b_theta, a_theta, d, axis=1)
        beta_filt = signal.filtfilt(b_beta, a_beta, d, axis=1)
        beta_hilb = signal.hilbert(beta_filt, axis=1)
        filtered[roi_name] = {
            'theta':    theta_filt,
            'theta_phase': np.angle(signal.hilbert(theta_filt, axis=1)),
            'beta_phase': np.angle(beta_hilb),
            'beta_pow': np.abs(beta_hilb)**2,
            'beta_env': np.abs(beta_hilb)
        }

    # for key, (pfc_roi, vis_roi) in zip(fig2_keys, fig2_pairs):
    #     if pfc_roi not in filtered or vis_roi not in filtered: continue
    #     # res = _peak_triggered_avg(
    #     #     filtered[pfc_roi]['theta'],
    #     #     filtered[vis_roi]['beta_pow'],
    #     #     sfreq, f_theta, time_vector, windows
    #     # )
    #     # subj_fig2[key] = res
    #     res_lag = _compute_lagged_mi(
    #         filtered[pfc_roi]['theta_phase'],
    #         filtered[vis_roi]['beta_env'],
    #         sfreq, time_vector, windows, lag_range=(-125, 125), lag_step=5
    #     )
    #     subj_fig3[key] = res_lag

    return subj_fig1, subj_fig2, subj_fig3


def _process_subject(s, voxRes):
    print(f"[*] Processing Sub-{s:02d}...")
    res1, res2, res3 = get_single_subj_pac(s, voxRes)
    return s, res1, res2, res3

def _collect_group(subj_list, voxRes):
    """Run all subjects and collect raw per-subject results for both figures."""
    roi_list = ['Ipsi-Vis', 'Contra-Vis', 'Ipsi-Frontal', 'Contra-Frontal']
    windows  = ['Stimulus', 'Delay']
    fig2_rows = ['Cross-Hemi', 'Same-Hemi']

    fig2_keys = [
        'Contra-PFC→Contra-Vis',
        'Contra-PFC→Ipsi-Vis',
        'Ipsi-PFC→Contra-Vis',
        'Ipsi-PFC→Ipsi-Vis',
    ]
    group_fig1 = {roi: {win: {'beta': [], 'gamma': []} for win in windows} for roi in roi_list}
    group_fig2 = {k:   {win: {'beta': [], 'gamma': []} for win in windows} for k in fig2_keys}
    group_fig3 = {k:   {win: {'lags': None, 'mi': []} for win in windows} for k in fig2_keys}

    # Process in batches of 4 to keep memory usage in check
    results = []
    for i in range(0, len(subj_list), 4):
        batch = subj_list[i:i+4]
        print(f"\n[*] Processing batch {i//4 + 1}: {batch}")
        batch_results = [_process_subject(s, voxRes) for s in batch]
        results.extend(batch_results)
        gc.collect() # Force garbage collection after each batch

    for s, res1, res2, res3 in results:
        if res1 is None: continue

        for roi in roi_list:
            for win in windows:
                if win in res1.get(roi, {}):
                    group_fig1[roi][win]['beta'].append(res1[roi][win]['beta'])
                    group_fig1[roi][win]['gamma'].append(res1[roi][win]['gamma'])

        if res2:
            for k in fig2_keys:
                for win in windows:
                    if win in res2.get(k, {}):
                        group_fig2[k][win]['beta'].append(res2[k][win]['beta'])
                        group_fig2[k][win]['gamma'].append(res2[k][win]['gamma'])

        if res3:
            for k in fig2_keys:
                for win in windows:
                    if win in res3.get(k, {}):
                        if group_fig3[k][win]['lags'] is None:
                            group_fig3[k][win]['lags'] = res3[k][win]['lags']
                        group_fig3[k][win]['mi'].append(res3[k][win]['mi'])

    return group_fig1, group_fig2, group_fig3


def _aggregate(group_data):
    """Convert list-of-arrays into mean/sem dicts, returns (stats, g_lim, b_lim)."""
    g_min, g_max, b_min, b_max = 0, 0, 0, 0
    stats = {}
    for key, wins in group_data.items():
        stats[key] = {}
        for win, arrs in wins.items():
            if not arrs['beta']: continue
            g_mat = np.stack(arrs['gamma'])
            b_mat = np.stack(arrs['beta'])
            n = g_mat.shape[0]
            g_mean, g_sem = np.mean(g_mat, 0), np.std(g_mat, 0) / np.sqrt(n)
            b_mean, b_sem = np.mean(b_mat, 0), np.std(b_mat, 0) / np.sqrt(n)
            stats[key][win] = {'g_mean': g_mean, 'g_sem': g_sem, 'b_mean': b_mean, 'b_sem': b_sem}
            g_min = min(g_min, np.min(g_mean - g_sem))
            g_max = max(g_max, np.max(g_mean + g_sem))
            b_min = min(b_min, np.min(b_mean - b_sem))
            b_max = max(b_max, np.max(b_mean + b_sem))
    pad_g = abs(g_max - g_min) * 0.1
    pad_b = abs(b_max - b_min) * 0.1
    g_lim = [g_min - pad_g, g_max + pad_g]
    b_lim = [b_min - pad_b, b_max + pad_b]
    return stats, g_lim, b_lim


def _aggregate_lag(group_data):
    stats = {}
    m_min, m_max = np.inf, -np.inf
    for key, wins in group_data.items():
        stats[key] = {}
        for win, d in wins.items():
            if not d['mi']: continue
            mi_mat = np.stack(d['mi'])
            n = mi_mat.shape[0]
            mi_mean = np.mean(mi_mat, axis=0)
            mi_sem = np.std(mi_mat, axis=0) / np.sqrt(n)
            stats[key][win] = {'lags': d['lags'], 'mi_mean': mi_mean, 'mi_sem': mi_sem}
            m_min = min(m_min, np.min(mi_mean - mi_sem))
            m_max = max(m_max, np.max(mi_mean + mi_sem))
            
    pad = abs(m_max - m_min) * 0.1
    m_lim = [m_min - pad, m_max + pad] if m_min != np.inf else [0, 1]
    return stats, m_lim


def _plot_panel(ax_g, ax_b, s, p_ax, g_lim, b_lim, title,
                phase_label='Phase signal', show_labels=False):
    """Plot one gamma + phase panel pair."""
    ax_g.plot(p_ax, s['g_mean'], color='crimson', linewidth=2.5, marker='.')
    ax_g.fill_between(p_ax, s['g_mean']-s['g_sem'], s['g_mean']+s['g_sem'],
                      color='crimson', alpha=0.2)
    ax_g.axvline(0, color='black', alpha=0.2, linestyle='--')
    ax_g.axhline(0, color='black', alpha=0.3, linestyle='--')
    ax_g.set_ylim(g_lim)
    if title: ax_g.set_title(title, fontsize=12, fontweight='bold')
    if show_labels: ax_g.set_ylabel('Beta amp\n% Change', fontsize=11)
    ax_g.set_xticks([])

    ax_b.plot(p_ax, s['b_mean'], color='royalblue', linewidth=2, marker='.')
    ax_b.fill_between(p_ax, s['b_mean']-s['b_sem'], s['b_mean']+s['b_sem'],
                      color='royalblue', alpha=0.2)
    ax_b.axvline(0, color='black', alpha=0.2, linestyle='--')
    ax_b.set_ylim(b_lim)
    if show_labels: ax_b.set_ylabel(phase_label, fontsize=11)
    ax_b.set_xticks([-2*np.pi, -np.pi, 0, np.pi, 2*np.pi],
                    [r'$-2\pi$', r'$-\pi$', '0', r'$\pi$', r'$2\pi$'])
    for ax in [ax_g, ax_b]: ax.grid(False)


def debug_pac_grand_average(subj_list, voxRes='8mm'):
    profile, n_cores, bidsRoot = get_compute_profile()

    # ── Collect data ──────────────────────────────────────────────
    group_fig1, group_fig2, group_fig3 = _collect_group(subj_list, voxRes)
    stats1, g_lim1, b_lim1 = _aggregate(group_fig1)
    stats2, g_lim2, b_lim2 = _aggregate(group_fig2)
    stats3, m_lim3 = _aggregate_lag(group_fig3)

    out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'pac_debugging')
    os.makedirs(out_dir, exist_ok=True)

    if len(subj_list) > 5:
        subj_str = f"N{len(subj_list)}"
    else:
        subj_str = f"1-{subj_list[-1]}" if len(subj_list) > 1 else str(subj_list[0])

    windows  = ['Stimulus', 'Delay']
    p_ax = np.linspace(-2*np.pi, 2*np.pi, 50)

    # ==============================================================
    # FIGURE 1 — Same-ROI PAC
    #   Row 0: Visual  (col 0=Ipsi-Vis×Stim, 1=Ipsi-Vis×Delay, 2=Contra-Vis×Stim, 3=Contra-Vis×Delay)
    #   Row 1: Frontal (col 0=Ipsi-Front×Stim, 1=Ipsi-Front×Delay, 2=Contra-Front×Stim, 3=Contra-Front×Delay)
    #   Visual uses alpha phase; PFC uses theta phase
    # ==============================================================
    # Define as 2 rows, each row fills all 4 cols
    fig1_rows = [
        # (row_label, phase_label, ipsi_roi, contra_roi)
        ('Visual',  'Alpha (8-12 Hz)', 'Ipsi-Vis',      'Contra-Vis'),
        ('Frontal', 'Theta (4-8 Hz)',  'Ipsi-Frontal',  'Contra-Frontal'),
    ]
    # col order: ipsi×stim, ipsi×delay, contra×stim, contra×delay
    fig1_col_defs = [
        ('Ipsi-Vis',      'Stimulus'), ('Ipsi-Vis',      'Delay'),
        ('Contra-Vis',    'Stimulus'), ('Contra-Vis',    'Delay'),
        ('Ipsi-Frontal',  'Stimulus'), ('Ipsi-Frontal',  'Delay'),
        ('Contra-Frontal','Stimulus'), ('Contra-Frontal','Delay'),
    ]

    fig1 = plt.figure(figsize=(24, 8))
    gs1  = fig1.add_gridspec(2*2, 4, height_ratios=[3, 1]*2, hspace=0.5, wspace=0.3)

    col_headers = ['Ipsi × Stimulus', 'Ipsi × Delay', 'Contra × Stimulus', 'Contra × Delay']
    for ci, h in enumerate(col_headers):
        fig1.text(0.14 + ci*0.21, 0.97, h, ha='center', va='top',
                  fontsize=12, style='italic', color='dimgray')

    for r_idx, (row_label, phase_label, ipsi_roi, contra_roi) in enumerate(fig1_rows):
        # left y-axis label for the row
        fig1.text(0.01, 0.75 - r_idx * 0.48, row_label, va='center', ha='left', rotation=90,
                  fontsize=14, fontweight='bold')
        rois_for_cols = [ipsi_roi, ipsi_roi, contra_roi, contra_roi]
        wins_for_cols = ['Stimulus', 'Delay', 'Stimulus', 'Delay']
        for col, (roi, win) in enumerate(zip(rois_for_cols, wins_for_cols)):
            if win not in stats1.get(roi, {}): continue
            ax_g = fig1.add_subplot(gs1[r_idx*2,   col])
            ax_b = fig1.add_subplot(gs1[r_idx*2+1, col])
            _plot_panel(ax_g, ax_b, stats1[roi][win], p_ax, g_lim1, b_lim1,
                        title=f'{roi}\n{win}' if r_idx == 0 else f'{roi}\n{win}',
                        phase_label=phase_label,
                        show_labels=(col == 0))

    fig1.suptitle(
        f'Figure 1 — Same-ROI PAC (Sub-{subj_str}, {voxRes})\n'
        f'Visual: beta amp locked to alpha phase | Frontal: beta amp locked to theta phase',
        fontsize=16, y=1.02
    )

    for ext in ['png', 'svg']:
        fig1.savefig(
            os.path.join(out_dir, f'sub-GA{subj_str}_pac_fig1_sameROI_{voxRes}.{ext}'),
            dpi=300, bbox_inches='tight'
        )
    plt.close(fig1)
    print(f"[*] Saved Figure 1 → pac_fig1_sameROI_{voxRes}.png/.svg")

    # ==============================================================
    # FIGURE 2 — Cross-signal PAC: PFC theta phase → Visual beta amp
    #   Row 0: Contra-PFC (col 0=→Ipsi-Vis×Stim, 1=→Ipsi-Vis×Delay, 2=→Contra-Vis×Stim, 3=→Contra-Vis×Delay)
    #   Row 1: Ipsi-PFC   (col 0=→Ipsi-Vis×Stim, 1=→Ipsi-Vis×Delay, 2=→Contra-Vis×Stim, 3=→Contra-Vis×Delay)
    # ==============================================================
    fig2_rows = [
        # (row_label, ipsi_key, contra_key)
        ('Contra-PFC θ →', 'Contra-PFC→Ipsi-Vis',  'Contra-PFC→Contra-Vis'),
        ('Ipsi-PFC θ →',   'Ipsi-PFC→Ipsi-Vis',    'Ipsi-PFC→Contra-Vis'),
    ]

    fig2 = plt.figure(figsize=(24, 8))
    gs2  = fig2.add_gridspec(2*2, 4, height_ratios=[3, 1]*2, hspace=0.5, wspace=0.3)

    col_headers2 = ['Ipsi-Vis × Stimulus', 'Ipsi-Vis × Delay',
                    'Contra-Vis × Stimulus', 'Contra-Vis × Delay']
    for ci, h in enumerate(col_headers2):
        fig2.text(0.14 + ci*0.21, 0.97, h, ha='center', va='top',
                  fontsize=12, style='italic', color='dimgray')

    for r_idx, (row_label, ipsi_key, contra_key) in enumerate(fig2_rows):
        fig2.text(0.01, 0.75 - r_idx * 0.48, row_label, va='center', ha='left', rotation=90,
                  fontsize=13, fontweight='bold')
        keys_for_cols = [ipsi_key,  ipsi_key,  contra_key, contra_key]
        wins_for_cols = ['Stimulus', 'Delay', 'Stimulus', 'Delay']
        col_subtitles = [
            f'{ipsi_key.split("→")[1]}\nStimulus',
            f'{ipsi_key.split("→")[1]}\nDelay',
            f'{contra_key.split("→")[1]}\nStimulus',
            f'{contra_key.split("→")[1]}\nDelay',
        ]
        for col, (key, win, subtitle) in enumerate(zip(keys_for_cols, wins_for_cols, col_subtitles)):
            if win not in stats2.get(key, {}): continue
            ax_g = fig2.add_subplot(gs2[r_idx*2,   col])
            ax_b = fig2.add_subplot(gs2[r_idx*2+1, col])
            _plot_panel(ax_g, ax_b, stats2[key][win], p_ax, g_lim2, b_lim2,
                        title=subtitle,
                        phase_label='PFC Theta (4-8 Hz)',
                        show_labels=(col == 0))

    fig2.suptitle(
        f'Figure 2 — Cross-Signal PAC: PFC Theta Phase → Visual Beta Amp (Sub-{subj_str}, {voxRes})',
        fontsize=16, y=1.02
    )

    for ext in ['png', 'svg']:
        fig2.savefig(
            os.path.join(out_dir, f'sub-GA{subj_str}_pac_fig2_crosssignal_{voxRes}.{ext}'),
            dpi=300, bbox_inches='tight'
        )
    # plt.close(fig2)
    # print(f"[*] Saved Figure 2 → pac_fig2_crosssignal_{voxRes}.png/.svg")

    # # ==============================================================
    # # FIGURE 3 — Lagged MI Sweep (Cross-Signal PAC)
    # # ==============================================================
    # fig3 = plt.figure(figsize=(24, 8))
    # gs3  = fig3.add_gridspec(2, 4, hspace=0.4, wspace=0.3)

    # for ci, h in enumerate(col_headers2):
    #     fig3.text(0.14 + ci*0.21, 0.95, h, ha='center', va='top',
    #               fontsize=12, style='italic', color='dimgray')

    # fig3_rows = [
    #     ('Contra-PFC θ →', 'Contra-PFC→Ipsi-Vis',  'Contra-PFC→Contra-Vis'),
    #     ('Ipsi-PFC θ →',   'Ipsi-PFC→Ipsi-Vis',    'Ipsi-PFC→Contra-Vis'),
    # ]

    # # Pre-calculate global min and max for shared ylim
    # z_min = np.inf
    # z_max = -np.inf
    # for key in stats3:
    #     for win in stats3[key]:
    #         s = stats3[key][win]
    #         min_val = np.nanmin(s['mi_mean'] - s['mi_sem'])
    #         max_val = np.nanmax(s['mi_mean'] + s['mi_sem'])
    #         if min_val < z_min: z_min = min_val
    #         if max_val > z_max: z_max = max_val

    # z_range = z_max - z_min
    # z_min -= z_range * 0.1
    # z_max += z_range * 0.1
    # if z_range == 0 or np.isnan(z_range):
    #     z_min, z_max = -2, 5


    # for r_idx, (row_label, ipsi_key, contra_key) in enumerate(fig3_rows):
    #     fig3.text(0.01, 0.75 - r_idx * 0.48, row_label, va='center', ha='left', rotation=90,
    #               fontsize=13, fontweight='bold')
    #     keys_for_cols = [ipsi_key,  ipsi_key,  contra_key, contra_key]
    #     wins_for_cols = ['Stimulus', 'Delay', 'Stimulus', 'Delay']
    #     col_subtitles = [
    #         f'{ipsi_key.split("→")[1]}\nStimulus',
    #         f'{ipsi_key.split("→")[1]}\nDelay',
    #         f'{contra_key.split("→")[1]}\nStimulus',
    #         f'{contra_key.split("→")[1]}\nDelay',
    #     ]
        
    #     for col, (key, win, subtitle) in enumerate(zip(keys_for_cols, wins_for_cols, col_subtitles)):
    #         if win not in stats3.get(key, {}): continue
    #         ax = fig3.add_subplot(gs3[r_idx, col])
            
    #         s = stats3[key][win]
    #         ax.plot(s['lags'], s['mi_mean'], color='purple', linewidth=2.5, marker='.')
    #         ax.fill_between(s['lags'], s['mi_mean']-s['mi_sem'], s['mi_mean']+s['mi_sem'],
    #                         color='purple', alpha=0.2)
            
    #         ax.set_title(subtitle, fontsize=12, fontweight='bold')
    #         ax.set_ylabel('Z-Scored MI' if col == 0 else '', fontsize=11)
    #         ax.set_xlabel('Lag (ms)', fontsize=11)
            
    #         ax.set_ylim(z_min, z_max)
            
    #         ax.axvline(0, color='gray', linestyle='--', alpha=0.7)
    #         ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
    #         ax.grid(False)

    # fig3.suptitle(
    #     f'Figure 3 — Lagged MI Sweep: PFC Theta Phase → Visual Beta Amp (Sub-{subj_str}, {voxRes})\nZ-Scored Modulation Index',
    #     fontsize=16, y=1.02
    # )

    # for ext in ['png', 'svg']:
    #     fig3.savefig(
    #         os.path.join(out_dir, f'sub-GA{subj_str}_pac_fig3_lagsweep_{voxRes}.{ext}'),
    #         dpi=300, bbox_inches='tight'
    #     )
    # plt.close(fig3)
    # print(f"[*] Saved Figure 3 → pac_fig3_lagsweep_{voxRes}.png/.svg")

    print("[*] Done.")


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
