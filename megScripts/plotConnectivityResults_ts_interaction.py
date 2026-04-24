#!/usr/bin/env python3
"""
Script to generate functional connectivity time-series plots for interaction effects.
Uses a 2x1 grid (Visual Interaction, Frontal Interaction).
Includes Cluster-Based Permutation testing (Alpha=0.1 entry, 10000 perms) for temporal significance.
Aesthetically aligned with the interaction bar plots (no grid, custom ylim).
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t
from mne.stats import permutation_cluster_1samp_test

def load_functional_ts_results(bidsRoot, subjects, taskName='mgs', voxRes='8mm'):
    """
    Load connectivity results and group into functional categories.
    """
    print(f"Loading results for {len(subjects)} subjects...")
    
    all_data = {
        'subjects': subjects,
        'loaded_subjects': [],
        'time_vector': None,
        'raw_metrics': {m: {'ipsi_within': [], 'ipsi_cross': [], 'contra_within': [], 'contra_cross': []} 
                        for m in ['imcoh', 'wpli']}
    }
    
    metrics_order = [('coh', 'ratio'), ('imcoh', 'ratio'), (None, None), ('dpli', 'ratio'), ('wpli', 'ratio')]
    
    for subjID in subjects:
        subjDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'connectivity_{voxRes}')
        outputFile = os.path.join(subjDir, f'sub-{subjID:02d}_task-{taskName}_connectivity_{voxRes}.pkl')
        
        if not os.path.exists(outputFile): continue
            
        try:
            with open(outputFile, 'rb') as f:
                subj_metrics = {}
                for m_name, corr_type in metrics_order:
                    try: data_dict = pickle.load(f)
                    except EOFError: continue
                    if m_name and data_dict: subj_metrics[m_name] = (data_dict, corr_type)
                
                if not subj_metrics: continue
                
                if all_data['time_vector'] is None:
                    first_m = list(subj_metrics.keys())[0]
                    sample = list(subj_metrics[first_m][0].values())[0]
                    all_data['time_vector'] = np.linspace(-1.0, 2.0, len(sample))
                
                t_vec = all_data['time_vector']
                b_mask = (t_vec >= -0.6) & (t_vec <= 0.0)
                b_idxs = np.where(b_mask)[0]
                
                for m_name, (data_dict, corr_type) in subj_metrics.items():
                    if m_name not in all_data['raw_metrics']: continue
                    
                    def get_subj_trace(cond, link):
                        for k in [f'{cond}_{link}_{m_name}', f'{cond}_{link}']:
                            if k in data_dict:
                                val = data_dict[k]
                                if val.ndim > 1: val = np.mean(val.reshape(val.shape[0], -1), axis=1)
                                if len(b_idxs) > 0:
                                    b_mean = np.nanmean(val[b_idxs])
                                    return (val / (b_mean + 1e-10) - 1) if corr_type == 'ratio' else (val - b_mean)
                                return val
                        return None

                    # Mapping matches plotConnectivityResults_bar_interaction.py
                    iw = [get_subj_trace('left', 'lV2lF'), get_subj_trace('right', 'rV2rF')]
                    if all(d is not None for d in iw): all_data['raw_metrics'][m_name]['ipsi_within'].append(np.mean(iw, axis=0))
                    
                    ic = [get_subj_trace('left', 'lV2rF'), get_subj_trace('right', 'rV2lF')]
                    if all(d is not None for d in ic): all_data['raw_metrics'][m_name]['ipsi_cross'].append(np.mean(ic, axis=0))

                    cw = [get_subj_trace('left', 'rV2rF'), get_subj_trace('right', 'lV2lF')]
                    if all(d is not None for d in cw): all_data['raw_metrics'][m_name]['contra_within'].append(np.mean(cw, axis=0))
                    
                    cc = [get_subj_trace('left', 'rV2lF'), get_subj_trace('right', 'rV2rF')]
                    if all(d is not None for d in cc): all_data['raw_metrics'][m_name]['contra_cross'].append(np.mean(cc, axis=0))

                all_data['loaded_subjects'].append(subjID)
        except Exception as e:
            print(f"Error loading {subjID}: {e}")
            
    print(f"Successfully loaded {len(all_data['loaded_subjects'])} subjects")
    return all_data

def plot_interaction_ts(results, bidsRoot, voxRes, metrics=['imcoh']):
    """
    Generate 2x1 grid of interaction time-series plots with cluster-permutation.
    """
    time_vector = results['time_vector']
    n_subs = len(results['loaded_subjects'])
    
    # Statistical Window: Cue onset to end of delay
    mask_win = (time_vector >= 0.1) & (time_vector <= 1.5)
    win_idx = np.where(mask_win)[0]
    
    # Cluster-forming threshold at p=0.1 (two-tailed) for high sensitivity
    cluster_forming_p = 0.1
    cluster_threshold = t.ppf(1 - cluster_forming_p / 2, df=n_subs - 1)
    
    for metric in metrics:
        metric_data = results['raw_metrics'][metric]
        if not any(metric_data.values()): continue
            
        print(f"Generating 2x1 time-series plots + cluster-stats for {metric}...")
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharey=True)
        
        plot_configs = [
            (0, 'Visual', 'contra_cross', 'ipsi_cross', 
             'Contra Vis ↔ Ipsi-Front', 'Ipsi Vis ↔ Cross-Front', 'royalblue', 'crimson'),
            (1, 'Frontal', 'contra_within', 'ipsi_within', 
             'Contra Vis ↔ Cross-Front', 'Ipsi Vis ↔ Ipsi-Front', 'royalblue', 'crimson')
        ]
        
        for ax_idx, seed_type, c1_key, c2_key, c1_name, c2_name, c1_color, c2_color in plot_configs:
            ax = axes[ax_idx]
            
            # Extract Subject-Wise Traces
            v1 = np.stack(metric_data[c1_key])
            v2 = np.stack(metric_data[c2_key])
            
            # Means and SEM
            m1, sem1 = np.mean(v1, axis=0), np.std(v1, axis=0, ddof=1) / np.sqrt(n_subs)
            m2, sem2 = np.mean(v2, axis=0), np.std(v2, axis=0, ddof=1) / np.sqrt(n_subs)
            
            # Simple Cluster-Based Permutation (10000 permutations, p=0.1 entry threshold)
            diffs_win = v2[:, win_idx] - v1[:, win_idx]
            print(f"  Computing Cluster Permutation [{metric}]: {seed_type} Interaction (10000 perms, p_init=0.1)...")
            t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                diffs_win, n_permutations=10000, threshold=cluster_threshold, tail=0, out_type='mask'
            )
            
            # Plot Traces
            ax.plot(time_vector, m1, color=c1_color, lw=2, label=c1_name)
            ax.fill_between(time_vector, m1-sem1, m1+sem1, color=c1_color, alpha=0.15)
            
            ax.plot(time_vector, m2, color=c2_color, lw=2, label=c2_name)
            ax.fill_between(time_vector, m2-sem2, m2+sem2, color=c2_color, alpha=0.15)
            
            # Plot Significance Shading
            y_min, y_max = -0.1, 0.15
            sig_mask_win = np.zeros(len(win_idx), dtype=bool)
            
            n_sig = 0
            for i_c, c_mask in enumerate(clusters):
                if cluster_p_values[i_c] < 0.05:
                    sig_mask_win[c_mask] = True
                    n_sig += 1
            
            if np.any(sig_mask_win):
                print(f"    Found {n_sig} significant clusters for {seed_type}.")
                sig_times = time_vector[win_idx[sig_mask_win]]
                # Plot as a subtle background shading
                ax.fill_between(sig_times, y_min, y_max, color='gray', alpha=0.2, zorder=0, label='p_cluster < 0.05')
            else:
                print(f"    No significant clusters (alpha=0.05) found for {seed_type} (Min p: {np.min(cluster_p_values):.3f} if clusters exist)")
            
            # Standard Decorations
            ax.axhline(0, color='black', lw=1, alpha=0.3, ls='--')
            ax.axvline(0, color='red', ls='--', alpha=0.6, label='Cue')
            ax.axvline(0.2, color='orange', ls='--', alpha=0.6, label='Delay')
            ax.axvline(1.7, color='green', ls='--', alpha=0.6, label='Response')
            
            ax.set_xlim(-0.5, 1.8)
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel(f"Relative {metric.upper()} Change")
            ax.set_title(f"{seed_type} Seed | Interaction Time-Series")
            ax.legend(loc='upper right', frameon=False, fontsize=9)
            ax.grid(False)
            
        plt.xlabel("Time (s)")
        plt.suptitle(f'Functional Hierarchy Time-Series ({metric.upper()}, n={n_subs})', y=0.98, fontsize=14)
        
        # Save
        out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'connectivity_ts_interaction_{metric}_{voxRes}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f'connectivity_ts_interaction_{metric}_{voxRes}.svg'), format='svg', bbox_inches='tight')
        plt.close()
        print(f"Time-series plots saved to {out_dir}")

def main():
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    taskName, voxRes = 'mgs', '8mm'
    import socket
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if socket.gethostname() == 'zod' else '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    results = load_functional_ts_results(bidsRoot, subjects, taskName, voxRes)
    if results['loaded_subjects']:
        plot_interaction_ts(results, bidsRoot, voxRes)
    print("Done!")

if __name__ == '__main__':
    main()
