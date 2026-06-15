#!/usr/bin/env python3
"""
Script to generate functional connectivity bar plots across trial phases.
Plots only the 'Interaction' column (Visual Interaction and Frontal Interaction).
Simplified using standard matplotlib for clean SVG generation.
Features Cohen's d effect sizes and detailed statistical reporting to CSV.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_rel

def load_functional_bar_results(bidsRoot, subjects, taskName='mgs', voxRes='8mm'):
    """
    Robustly load connectivity results and group into functional categories.
    """
    print(f"Loading results for {len(subjects)} subjects...")
    
    # categories to store
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
                
                # Set common time vector
                if all_data['time_vector'] is None:
                    first_m = list(subj_metrics.keys())[0]
                    sample = list(subj_metrics[first_m][0].values())[0]
                    all_data['time_vector'] = np.linspace(-1.0, 2.0, len(sample))
                
                t_vec = all_data['time_vector']
                b_mask = (t_vec >= -0.6) & (t_vec <= 0.0)
                b_idxs = np.where(b_mask)[0]
                
                # Process imcoh into categories
                for m_name, (data_dict, corr_type) in subj_metrics.items():
                    if m_name not in all_data['raw_metrics']: continue
                    
                    def get_subj_trace(cond, link):
                        for k in [f'{cond}_{link}_{m_name}', f'{cond}_{link}']:
                            if k in data_dict:
                                val = data_dict[k]
                                if val.ndim > 1: val = np.mean(val.reshape(val.shape[0], -1), axis=1)
                                # Return raw values for CLI calculation
                                return val
                        return None

                    # Mapping
                    iw = [get_subj_trace('left', 'lV2lF'), get_subj_trace('right', 'rV2rF')]
                    iw = [d for d in iw if d is not None]
                    if iw: all_data['raw_metrics'][m_name]['ipsi_within'].append(np.mean(iw, axis=0))
                    
                    ic = [get_subj_trace('left', 'lV2rF'), get_subj_trace('right', 'rV2lF')]
                    ic = [d for d in ic if d is not None]
                    if ic: all_data['raw_metrics'][m_name]['ipsi_cross'].append(np.mean(ic, axis=0))

                    cw = [get_subj_trace('left', 'rV2rF'), get_subj_trace('right', 'lV2lF')]
                    cw = [d for d in cw if d is not None]
                    if cw: all_data['raw_metrics'][m_name]['contra_within'].append(np.mean(cw, axis=0))
                    
                    cc = [get_subj_trace('left', 'rV2lF'), get_subj_trace('right', 'lV2rF')]
                    cc = [d for d in cc if d is not None]
                    if cc: all_data['raw_metrics'][m_name]['contra_cross'].append(np.mean(cc, axis=0))

                all_data['loaded_subjects'].append(subjID)
        except Exception as e:
            print(f"Error loading {subjID}: {e}")
            
    print(f"Successfully loaded {len(all_data['loaded_subjects'])} subjects")
    return all_data

def plot_functional_bars(results, bidsRoot, voxRes, metrics=['imcoh', 'wpli']):
    """
    Generate the 2x1 grid of bar plots for the Interaction column.
    Uses simplified standard matplotlib instead of seaborn objects
    to ensure the SVGs are simple arrays of paths instead of nested groups.
    """
    windows = {'Stimulus': (0.1, 0.3), 'Delay': (0.6, 1.5)}
    window_names = list(windows.keys())
    t_vec = results['time_vector']
    
    for metric in metrics:
        metric_data = results['raw_metrics'][metric]
        if not any(metric_data.values()): continue
            
        print(f"Generating 2x2 interaction master bar plots + stats for {metric}...")
        plt.close('all')
        
        # 2x2 grid: Top row = Z-scored Bars, Bottom row = Contrast (CLI)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=False)
        
        # Data categories (Raw values)
        raw_cw = np.array(metric_data['contra_within'])
        raw_ic = np.array(metric_data['ipsi_cross'])
        raw_cc = np.array(metric_data['contra_cross'])
        raw_iw = np.array(metric_data['ipsi_within'])

        # Calculate CLI from RAW values (matching ts_hemifield)
        cli_contra = (raw_cw - raw_ic) / (raw_cw + raw_ic + 1e-12)
        cli_ipsi   = (raw_cc - raw_iw) / (raw_cc + raw_iw + 1e-12)

        # Z-scoring function (for the bars)
        def zscore_all(traces, b_idxs):
            z = np.zeros_like(traces)
            for i in range(len(traces)):
                b_data = traces[i, b_idxs]
                bm, bs = np.nanmean(b_data), np.nanstd(b_data)
                z[i] = (traces[i] - bm) / (bs + 1e-10)
            return z

        # Baseline indices for z-scoring
        b_mask = (t_vec >= -0.6) & (t_vec <= 0.0)
        b_idxs = np.where(b_mask)[0]
        
        z_cw = zscore_all(raw_cw, b_idxs)
        z_ic = zscore_all(raw_ic, b_idxs)
        z_cc = zscore_all(raw_cc, b_idxs)
        z_iw = zscore_all(raw_iw, b_idxs)

        plot_configs = [
            # ax_row, ax_col, data_pair (c1, c2), title, colors, ylabel, is_paired_top, labels
            (0, 0, (z_cw, z_ic), "Contralateral Frontal Target", ["#e6550d", "#fec44f"], "Baseline Z-score", True, ['Contra Visual', 'Ipsi Visual']),
            (0, 1, (z_cc, z_iw), "Ipsilateral Frontal Target",   ["#1b9e77", "#0571b0"], "Baseline Z-score", True, ['Contra Visual', 'Ipsi Visual']),
            (1, 0, (z_ic, z_cc), "Cross-Hemisphere Connections", ["#fec44f", "#1b9e77"], "Baseline Z-score", True, ['Ipsi-Vis to Contra-Front', 'Contra-Vis to Ipsi-Front']),
            (1, 1, (z_cw, z_iw), "Within-Hemisphere Connections", ["#e6550d", "#0571b0"], "Baseline Z-score", True, ['Contra-Vis to Contra-Front', 'Ipsi-Vis to Ipsi-Front'])
        ]
         
        stat_summary = []
        
        def get_p_label(p):
            return '***' if p <= 0.001 else '**' if p <= 0.01 else '*' if p <= 0.06 else 'ns'

        def run_permutation_vs_zero(data, n_perms=9999):
            obs_mean = np.mean(data)
            # Flip signs of the data to create null distribution
            signs = np.random.choice([-1, 1], size=(n_perms, len(data)))
            null_dist = np.mean(signs * data, axis=1)
            p = (np.sum(np.abs(null_dist) >= np.abs(obs_mean)) + 1) / (n_perms + 1)
            return p

        for r, c, d_pair, title, colors, ylabel, is_paired, bar_labels in plot_configs:
            ax = axes[r, c]
            x_pos = np.arange(len(window_names))
            width = 0.35
            
            means1, sems1 = [], []
            means2, sems2 = [], []
            
            for w_idx, w_name in enumerate(window_names):
                start, end = windows[w_name]
                mask = (t_vec >= start) & (t_vec <= end)
                
                v1 = np.nanmean(d_pair[0][:, mask], axis=1)
                v2 = np.nanmean(d_pair[1][:, mask], axis=1)
                
                # Individual bar stats
                p1 = run_permutation_vs_zero(v1)
                p2 = run_permutation_vs_zero(v2)
                
                m1, s1, sem1 = np.mean(v1), np.std(v1, ddof=1), np.std(v1, ddof=1)/np.sqrt(len(v1))
                m2, s2, sem2 = np.mean(v2), np.std(v2, ddof=1), np.std(v2, ddof=1)/np.sqrt(len(v2))
                d1 = m1 / (s1 + 1e-10)
                d2 = m2 / (s2 + 1e-10)
                
                # Paired contrast stats
                diffs = v2 - v1
                p_paired_perm = (np.sum(np.abs(np.mean(np.random.choice([-1, 1], size=(9999, len(diffs))) * diffs, axis=1)) >= np.abs(np.mean(diffs))) + 1) / 10000
                d_paired = np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-10)
                
                means1.append(m1); sems1.append(sem1)
                means2.append(m2); sems2.append(sem2)
                
                # Labels
                y1 = m1 + sem1; y2 = m2 + sem2
                ax.text(w_idx - width/2, y1 + 0.05, get_p_label(p1), ha='center', fontsize=8, color='black' if p1 <= 0.06 else 'gray')
                ax.text(w_idx + width/2, y2 + 0.05, get_p_label(p2), ha='center', fontsize=8, color='black' if p2 <= 0.06 else 'gray')
                y_max = max(y1, y2)
                ax.text(w_idx, y_max + 0.25, f"({get_p_label(p_paired_perm)})", ha='center', fontsize=9, color='blue' if p_paired_perm <= 0.06 else '0.5')
                
                stat_summary.append({
                    'Metric': metric, 'Plot': title, 'Window': w_name, 
                    'Mean1': m1, 'SEM1': sem1, 'd1': d1, 'p1_perm': p1,
                    'Mean2': m2, 'SEM2': sem2, 'd2': d2, 'p2_perm': p2,
                    'd_paired': d_paired, 'p_paired_perm': p_paired_perm
                })

            ax.bar(x_pos - width/2, means1, width, yerr=sems1, color=colors[0], alpha=0.8, capsize=5, label=bar_labels[0])
            ax.bar(x_pos + width/2, means2, width, yerr=sems2, color=colors[1], alpha=0.8, capsize=5, label=bar_labels[1])
            ax.axhline(0, color='black', lw=1, linestyle='--')
            ax.legend(loc='upper right', frameon=False, fontsize=8)
            ax.set_ylim(-1.5, 1.5)            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(window_names)
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel(ylabel)
            ax.grid(False)

        if metric == 'imcoh':
            # Removed forced ylim logic since we set it globally now to -0.1, 0.15
            pass

        plt.suptitle(f'Functional Hierarchy Bars ({metric.upper()}, n={len(results["loaded_subjects"])})', y=0.98, fontsize=14)
        sns.despine(fig)
        fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.9, bottom=0.1, left=0.1, right=0.9)
        
        # Save Stats to CSV (Overwriting original Fs04 assets)
        out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
        os.makedirs(out_dir, exist_ok=True)
        stats_file = os.path.join(out_dir, f'connectivity_functional_stats_{metric}_{voxRes}.csv')
        pd.DataFrame(stat_summary).to_csv(stats_file, index=False)
        print(f"Stats report saved to {stats_file}")

        # Save Figures (Overwriting original Fs04 assets)
        fig.savefig(os.path.join(out_dir, f'connectivity_functional_bars_{metric}_{voxRes}.png'), dpi=150)
        fig.savefig(os.path.join(out_dir, f'connectivity_functional_bars_{metric}_{voxRes}.svg'))
        plt.close(fig)

def main():
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    taskName, voxRes = 'mgs', '8mm'
    import socket
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if socket.gethostname() == 'zod' else '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    results = load_functional_bar_results(bidsRoot, subjects, taskName, voxRes)
    if results['loaded_subjects']:
        plot_functional_bars(results, bidsRoot, voxRes, metrics=['imcoh'])
    print("Done! Interaction bar plots finalized.")

if __name__ == '__main__':
    main()
