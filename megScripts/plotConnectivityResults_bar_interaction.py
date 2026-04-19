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
                                if len(b_idxs) > 0:
                                    b_mean = np.nanmean(val[b_idxs])
                                    return (val / (b_mean + 1e-10) - 1) if corr_type == 'ratio' else (val - b_mean)
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
    windows = {'Pre-delay': (-0.5, 0.0), 'Stimulus': (0.1, 0.3), 'Delay': (0.6, 1.5)}
    window_names = list(windows.keys())
    t_vec = results['time_vector']
    
    for metric in metrics:
        metric_data = results['raw_metrics'][metric]
        if not any(metric_data.values()): continue
            
        print(f"Generating 2x1 interaction bar plots + stats for {metric}...")
        
        fig, axes = plt.subplots(2, 1, figsize=(6, 8), sharey=True)
        if not isinstance(axes, np.ndarray): axes = [axes]
        
        plot_configs = [
            # ax_idx, seed_type, c1_key, c2_key, c1_name, c2_name, c1_color, c2_color
            (0, 'Visual', 'contra_cross', 'ipsi_cross', 
             'Contra Vis ↔ Ipsi-Front', 'Ipsi Vis ↔ Cross-Front', 'royalblue', 'crimson'),
            (1, 'Frontal', 'contra_within', 'ipsi_within', 
             'Contra Vis ↔ Cross-Front', 'Ipsi Vis ↔ Ipsi-Front', 'royalblue', 'crimson')
        ]
         
        stat_summary = []
        
        for ax_idx, seed_type, c1_key, c2_key, c1_name, c2_name, c1_color, c2_color in plot_configs:
            ax = axes[ax_idx]
            
            x_pos = np.arange(len(window_names))
            width = 0.35
            
            c1_means = []
            c1_sems = []
            c2_means = []
            c2_sems = []
            
            for w_idx, w_name in enumerate(window_names):
                start, end = windows[w_name]
                mask = (t_vec >= start) & (t_vec <= end)
                
                v1 = []
                v2 = []
                for sj_idx in range(len(results['loaded_subjects'])):
                    if len(metric_data[c1_key]) > sj_idx and len(metric_data[c2_key]) > sj_idx:
                        trace1 = metric_data[c1_key][sj_idx]
                        trace2 = metric_data[c2_key][sj_idx]
                        v1.append(np.nanmean(trace1[mask]))
                        v2.append(np.nanmean(trace2[mask]))
                
                v1 = np.array(v1)
                v2 = np.array(v2)
                
                # Stats calculation
                if len(v1) > 1:
                    # diff is v2 - v1 to match original plotting (crimson - royalblue)
                    diffs = v2 - v1
                    t_stat, p = ttest_rel(v2, v1)
                    cohen_d = np.nanmean(diffs) / np.nanstd(diffs, ddof=1)
                    
                    label = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    
                    stat_summary.append({
                        'Metric': metric, 'Facet': f"{seed_type}_Interaction", 'Window': w_name,
                        'Contrast': f"{c2_name} vs {c1_name}", 
                        'c2_name': c2_name, 'c2_mean': np.mean(v2), 'c2_sem': np.std(v2, ddof=1)/np.sqrt(len(v2)),
                        'c1_name': c1_name, 'c1_mean': np.mean(v1), 'c1_sem': np.std(v1, ddof=1)/np.sqrt(len(v1)),
                        't_stat': t_stat, 'p_val': p, 'Cohen_d': cohen_d
                    })
                    
                    y_max = max(np.mean(v1) + (np.std(v1, ddof=1) / np.sqrt(len(v1))), 
                              np.mean(v2) + (np.std(v2, ddof=1) / np.sqrt(len(v2))))
                    y_min = min(np.mean(v1) - (np.std(v1, ddof=1) / np.sqrt(len(v1))), 
                              np.mean(v2) - (np.std(v2, ddof=1) / np.sqrt(len(v2))))
                    
                    text_y = y_max + 0.015 if y_max >= 0 else y_min - 0.02
                    ax.text(w_idx, text_y, label, ha='center', va='bottom' if y_max >= 0 else 'top', 
                            fontsize=12, fontweight='bold' if p < 0.05 else 'normal',
                            color='black' if p < 0.05 else '0.4')
                
                c1_means.append(np.mean(v1))
                c1_sems.append(np.std(v1, ddof=1)/np.sqrt(len(v1)))
                c2_means.append(np.mean(v2))
                c2_sems.append(np.std(v2, ddof=1)/np.sqrt(len(v2)))
            
            # Bars
            ax.bar(x_pos - width/2, c1_means, width, yerr=c1_sems, label=c1_name, color=c1_color, alpha=0.7, capsize=5, zorder=2)
            ax.bar(x_pos + width/2, c2_means, width, yerr=c2_sems, label=c2_name, color=c2_color, alpha=0.7, capsize=5, zorder=2)
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(window_names)
            ax.set_title(f"{seed_type} Seed | Interaction Component")
            ax.set_ylabel(f"Relative {metric.upper()} Change")
            ax.legend(loc='lower left' if metric == 'wpli' else 'upper right', frameon=False)
            
            ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

        if metric == 'imcoh':
            ylim = axes[0].get_ylim()
            max_abs = max(abs(ylim[0]), abs(ylim[1]))
            axes[0].set_ylim(-max_abs, max_abs)

        plt.suptitle(f'Functional Hierarchy Bars ({metric.upper()}, n={len(results["loaded_subjects"])})', y=0.98, fontsize=14)
        sns.despine(fig)
        plt.tight_layout()
        
        # Save Stats to CSV (Overwriting original Fs04 assets)
        out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
        os.makedirs(out_dir, exist_ok=True)
        stats_file = os.path.join(out_dir, f'connectivity_functional_stats_{metric}_{voxRes}.csv')
        pd.DataFrame(stat_summary).to_csv(stats_file, index=False)
        print(f"Stats report saved to {stats_file}")

        # Save Figures (Overwriting original Fs04 assets)
        plt.savefig(os.path.join(out_dir, f'connectivity_functional_bars_{metric}_{voxRes}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f'connectivity_functional_bars_{metric}_{voxRes}.svg'), format='svg', bbox_inches='tight')
        plt.close()

def main():
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    taskName, voxRes = 'mgs', '8mm'
    import socket
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if socket.gethostname() == 'zod' else '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    results = load_functional_bar_results(bidsRoot, subjects, taskName, voxRes)
    if results['loaded_subjects']:
        plot_functional_bars(results, bidsRoot, voxRes)
    print("Done! Interaction bar plots finalized.")

if __name__ == '__main__':
    main()
