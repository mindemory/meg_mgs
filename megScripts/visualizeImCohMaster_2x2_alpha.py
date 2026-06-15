#!/usr/bin/env python3
"""
Script to visualize alpha-band frontal connectivity collapsed across visual hemifield.
Generates a 2x2 Master Figure matching the style, colors, and conditions of the original beta plot:
  Row 1: Z-scored traces (Contra-Vis vs Ipsi-Vis seeds to Frontal targets)
  Row 2: Cross-Hemisphere and Within-Hemisphere connections

Significance: pointwise 1-sample t-test against zero, uncorrected (p < 0.05).
Difference bar at the top: pointwise paired t-test between the two curves (p < 0.05).
"""

import os
import pickle
import socket
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import ttest_1samp

def get_bids_root():
    h = socket.gethostname()
    if 'vader' in h:
        return '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        return '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'

def main():
    bidsRoot = get_bids_root()
    voxRes = '8mm'
    freqBand = 'alpha'
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    
    outDir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
    os.makedirs(outDir, exist_ok=True)

    # 1. Load ROIs
    atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    lfro_pts = np.where(atlas['left_frontal_points'].flatten() == 1)[0]
    rfro_pts = np.where(atlas['right_frontal_points'].flatten() == 1)[0]

    # Structure to hold raw traces for each subject:
    # keys: 'ipsi_within', 'ipsi_cross', 'contra_within', 'contra_cross'
    subj_data = {
        'contra_within': [], # Contra-Vis -> Contra-Front
        'ipsi_cross': [],     # Ipsi-Vis   -> Contra-Front
        'contra_cross': [],  # Contra-Vis -> Ipsi-Front
        'ipsi_within': []    # Ipsi-Vis   -> Ipsi-Front
    }
    
    time_vector = None
    loaded_subs = []

    print(f"[*] Loading seeded connectivity results ({freqBand.upper()}) for {len(subjects)} subjects...")

    for sub in subjects:
        subName = f'sub-{sub:02d}'
        conn_dir = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'connectivity_{voxRes}')
        
        # We need all 4 visual-to-target-location files to include this subject
        files_exist = True
        sub_files = {}
        for s_roi in ['left_visual', 'right_visual']:
            for t_loc in ['left', 'right']:
                f_name = f'{subName}_task-mgs_seededConnectivity_{voxRes}_{s_roi}_{t_loc}_imcoh_{freqBand}.pkl'
                f_path = os.path.join(conn_dir, f_name)
                if not os.path.exists(f_path):
                    files_exist = False
                    break
                sub_files[(s_roi, t_loc)] = f_path
            if not files_exist:
                break
                
        if not files_exist:
            print(f"  [!] Missing connectivity files for {subName}, skipping.")
            continue
            
        try:
            # Load and process the 4 connectivity matrices
            matrices = {}
            for key, path in sub_files.items():
                with open(path, 'rb') as f:
                    matrices[key] = pickle.load(f) # Shape: (sources, times)
                    
            if time_vector is None:
                n_times = list(matrices.values())[0].shape[1]
                time_vector = np.linspace(-1.5, 2.5, n_times)
                
            # Perform spatial averaging over Frontal ROIs for each of the 4 conditions
            # Left Target trials ('left'):
            # - Left Visual seed (lV) to LF: lV2lF
            left_lV2lF = matrices[('left_visual', 'left')][lfro_pts, :].mean(axis=0)
            # - Left Visual seed (lV) to RF: lV2rF
            left_lV2rF = matrices[('left_visual', 'left')][rfro_pts, :].mean(axis=0)
            # - Right Visual seed (rV) to LF: rV2lF
            left_rV2lF = matrices[('right_visual', 'left')][lfro_pts, :].mean(axis=0)
            # - Right Visual seed (rV) to RF: rV2rF
            left_rV2rF = matrices[('right_visual', 'left')][rfro_pts, :].mean(axis=0)
            
            # Right Target trials ('right'):
            # - Left Visual seed (lV) to LF: lV2lF
            right_lV2lF = matrices[('left_visual', 'right')][lfro_pts, :].mean(axis=0)
            # - Left Visual seed (lV) to RF: lV2rF
            right_lV2rF = matrices[('left_visual', 'right')][rfro_pts, :].mean(axis=0)
            # - Right Visual seed (rV) to LF: rV2lF
            right_rV2lF = matrices[('right_visual', 'right')][lfro_pts, :].mean(axis=0)
            # - Right Visual seed (rV) to RF: rV2rF
            right_rV2rF = matrices[('right_visual', 'right')][rfro_pts, :].mean(axis=0)
            
            # Combine across visual seeds / hemifields to form the 4 functional categories
            # 1. Contra-Visual to Contra-Frontal (contra_within)
            # Left target: rV2rF, Right target: lV2lF
            cw = (left_rV2rF + right_lV2lF) / 2.0
            
            # 2. Ipsi-Visual to Contra-Frontal (ipsi_cross)
            # Left target: lV2rF, Right target: rV2lF
            ic = (left_lV2rF + right_rV2lF) / 2.0
            
            # 3. Contra-Visual to Ipsi-Frontal (contra_cross)
            # Left target: rV2lF, Right target: lV2rF
            cc = (left_rV2lF + right_lV2rF) / 2.0
            
            # 4. Ipsi-Visual to Ipsi-Frontal (ipsi_within)
            # Left target: lV2lF, Right target: rV2rF
            iw = (left_lV2lF + right_rV2rF) / 2.0
            
            subj_data['contra_within'].append(cw)
            subj_data['ipsi_cross'].append(ic)
            subj_data['contra_cross'].append(cc)
            subj_data['ipsi_within'].append(iw)
            
            loaded_subs.append(sub)
            
        except Exception as e:
            print(f"  [!] Error processing {subName}: {e}")
            
    n_subs = len(loaded_subs)
    print(f"[*] Successfully loaded and processed {n_subs} subjects.")
    
    if n_subs == 0:
        print("[!] No subjects loaded. Exiting.")
        return

    # 2. Z-score per subject
    b_mask = (time_vector >= -0.6) & (time_vector <= 0.0)
    b_idxs = np.where(b_mask)[0]
    
    def zscore_traces(traces_list):
        traces = np.stack(traces_list, axis=0) # (n_subs, n_times)
        z_traces = np.zeros_like(traces)
        for i in range(traces.shape[0]):
            b_data = traces[i, b_idxs]
            b_mean = np.nanmean(b_data)
            b_std  = np.nanstd(b_data)
            z_traces[i] = (traces[i] - b_mean) / (b_std + 1e-10)
        return z_traces

    z_cw = zscore_traces(subj_data['contra_within'])
    z_ic = zscore_traces(subj_data['ipsi_cross'])
    z_cc = zscore_traces(subj_data['contra_cross'])
    z_iw = zscore_traces(subj_data['ipsi_within'])

    # Plot Configs (matching 2x2 legacy plots)
    plot_configs = [
        (0, 0, (z_cw, z_ic), "Contralateral Frontal Target", ["#e6550d", "#fec44f"], ["Contra-Visual", "Ipsi-Visual"]),
        (0, 1, (z_cc, z_iw), "Ipsilateral Frontal Target",   ["#1b9e77", "#0571b0"], ["Contra-Visual", "Ipsi-Visual"]),
        (1, 0, (z_ic, z_cc), "Cross-Hemisphere Connections", ["#fec44f", "#1b9e77"], ["Ipsi-Vis to Contra-Front", "Contra-Vis to Ipsi-Front"]),
        (1, 1, (z_cw, z_iw), "Within-Hemisphere Connections", ["#e6550d", "#0571b0"], ["Contra-Vis to Contra-Front", "Ipsi-Vis to Ipsi-Front"])
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    r1_y_lim = (-1.5, 1.5)

    def pointwise_ttest_mask(traces, time_vec, label=""):
        """Pointwise 1-sample t-test vs 0 within [0.0, 1.7]s."""
        _, p_vals = ttest_1samp(traces, 0, axis=0)
        sig_mask = p_vals < 0.05
        
        # Apply time window filter
        t_window = (time_vec >= 0.0) & (time_vec <= 1.7)
        sig_mask = sig_mask & t_window
        
        sig_count = np.sum(sig_mask)
        if sig_count > 0:
            print(f"    [{label}] Significant at {sig_count} timepoints in [0.0, 1.7]s.")
        return sig_mask

    for r, c, (d1, d2), title, colors, labels in plot_configs:
        ax = axes[r, c]
        # Plot traces
        for data, color, label in [(d1, colors[0], labels[0]), (d2, colors[1], labels[1])]:
            m = np.mean(data, axis=0)
            sem = np.std(data, axis=0, ddof=1) / np.sqrt(n_subs)
            ax.plot(time_vector, m, color=color, lw=2.5, label=label)
            ax.fill_between(time_vector, m-sem, m+sem, color=color, alpha=0.15)
            
            # Individual sig (vs 0)
            sig = pointwise_ttest_mask(data, time_vector, label=f"({r},{c})-{label}")
            if np.any(sig):
                y_off = r1_y_lim[0] + 0.1 + (0.2 if label == labels[1] else 0)
                ax.fill_between(time_vector, y_off, y_off+0.1, where=sig, color=color, alpha=0.8, interpolate=True)
        
        # Paired contrast sig (d1 - d2)
        diffs = d1 - d2
        sig_diff = pointwise_ttest_mask(diffs, time_vector, label=f"({r},{c})-diff")
        if np.any(sig_diff):
            ax.fill_between(time_vector, r1_y_lim[1]-0.2, r1_y_lim[1]-0.1, where=sig_diff, color='black', alpha=0.5)

        ax.set_title(title, fontweight='bold')
        ax.set_ylim(r1_y_lim)
        ax.set_ylabel("Baseline Z-score")
        ax.legend(loc='upper right', frameon=False, fontsize=8)

    # Global Decorations
    for ax in axes.flat:
        ax.axhline(0, color='black', lw=1, alpha=0.3, ls='--')
        ax.axvline(0,   color='gray', ls='--', alpha=0.6)
        ax.axvline(0.2, color='gray', ls=':',  alpha=0.4)
        ax.axvline(1.7, color='black', ls='--', alpha=0.6)
        ax.set_xlim(-0.3, 1.7)
        ax.grid(False)

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    
    plt.suptitle(f'IMCOH Connectivity (N={n_subs})',
                 y=0.98, fontsize=18, fontweight='bold')

    out_png = os.path.join(outDir, f'{freqBand}_connectivity_ts_master_imcoh_{voxRes}.png')
    out_svg = os.path.join(outDir, f'{freqBand}_connectivity_ts_master_imcoh_{voxRes}.svg')
    
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.savefig(out_svg, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"[*] Replicated 2x2 master figure saved to:")
    print(f"    - PNG: {out_png}")
    print(f"    - SVG: {out_svg}")
    print("Done!")

if __name__ == '__main__':
    main()
