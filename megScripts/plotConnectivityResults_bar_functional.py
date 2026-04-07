#!/usr/bin/env python3
"""
Script to generate functional connectivity bar plots across trial phases.
Uses a 2x3 grid (Visual seeds, Frontal seeds) and includes an Interaction column.
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
from scipy.stats import ttest_1samp, ttest_rel

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
    Generate the 2x3 grid of bar plots for each metric with Cohen's d and CSV reporting.
    """
    windows = {'Pre-delay': (-0.5, 0.0), 'Stimulus': (0.1, 0.3), 'Delay': (0.6, 1.5)}
    t_vec = results['time_vector']
    
    for metric in metrics:
        metric_data = results['raw_metrics'][metric]
        if not any(metric_data.values()): continue
            
        print(f"Generating 2x3 bar plots + stats for {metric}...")
        rows = []
        plot_configs = [
            ('Visual', 'Ipsilateral', 'ipsi_within', 'ipsi_cross'),
            ('Visual', 'Contralateral', 'contra_within', 'contra_cross'),
            ('Visual', 'Interaction', 'contra_cross', 'ipsi_cross'),
            ('Frontal', 'Ipsilateral', 'ipsi_within', 'contra_cross'),
            ('Frontal', 'Contralateral', 'contra_within', 'ipsi_cross'),
            ('Frontal', 'Interaction', 'contra_within', 'ipsi_within')
        ]
        
        # Slicing Table for CSV Export
        stat_summary = []
        
        for sj_idx in range(len(results['loaded_subjects'])):
            for seed_type, side, b_cat, r_cat in plot_configs:
                for cat_key in [b_cat, r_cat]:
                    if len(metric_data[cat_key]) > sj_idx:
                        trace = metric_data[cat_key][sj_idx]
                        for w_name, (start, end) in windows.items():
                            mask = (t_vec >= start) & (t_vec <= end)
                            val = np.nanmean(trace[mask])
                            
                            # Labels
                            if side == 'Interaction':
                                if seed_type == 'Visual':
                                    conn_name = 'Ipsi Vis ↔ Cross-Front' if cat_key == 'ipsi_cross' else 'Contra Vis ↔ Ipsi-Front'
                                else:
                                    conn_name = 'Ipsi Vis ↔ Ipsi-Front' if cat_key == 'ipsi_within' else 'Contra Vis ↔ Cross-Front'
                            else:
                                conn_name = 'Within-Hemi' if cat_key in ['ipsi_within', 'contra_within'] else 'Cross-Hemi'
                                
                            rows.append({
                                'Subject': results['loaded_subjects'][sj_idx], 'Window': w_name, 'Seed Type': seed_type,
                                'Side': side, 'Connection': conn_name, 'Value': val
                            })
        
        if not rows: continue
        df = pd.DataFrame(rows)
        sns.set_theme(style="ticks")
        
        # Hue Order for alignment
        conn_order = ['Within-Hemi', 'Cross-Hemi', 'Ipsi Vis ↔ Cross-Front', 'Contra Vis ↔ Ipsi-Front', 
                      'Ipsi Vis ↔ Ipsi-Front', 'Contra Vis ↔ Cross-Front']
        palette = {
            'Within-Hemi': 'royalblue', 'Cross-Hemi': 'crimson',
            'Ipsi Vis ↔ Cross-Front': 'crimson', 'Contra Vis ↔ Ipsi-Front': 'royalblue',
            'Ipsi Vis ↔ Ipsi-Front': 'crimson', 'Contra Vis ↔ Cross-Front': 'royalblue'
        }
        
        g = sns.catplot(
            data=df, kind="bar", x="Window", y="Value", hue="Connection",
            hue_order=conn_order, row="Seed Type", col="Side", 
            palette=palette, height=5, aspect=1.2, errorbar='se', capsize=.1, alpha=0.7
        )
        
        # Overlay dots
        g.map_dataframe(sns.stripplot, x="Window", y="Value", hue="Connection", hue_order=conn_order,
                        dodge=True, palette={k: '0.3' for k in conn_order}, alpha=0.4, size=4, legend=False)
        
        # Paired Statistics and Reporting
        for (seed_type, side), ax in g.axes_dict.items():
            facet_data = df[(df['Seed Type'] == seed_type) & (df['Side'] == side)]
            current_conns = [c for c in conn_order if c in facet_data['Connection'].unique()]
            if len(current_conns) == 2:
                conn1, conn2 = current_conns[0], current_conns[1]
                for i, w_name in enumerate(['Pre-delay', 'Stimulus', 'Delay']):
                    d1 = facet_data[(facet_data['Window'] == w_name) & (facet_data['Connection'] == conn1)].sort_values('Subject')
                    d2 = facet_data[(facet_data['Window'] == w_name) & (facet_data['Connection'] == conn2)].sort_values('Subject')
                    common_subs = set(d1['Subject']) & set(d2['Subject'])
                    if len(common_subs) > 1:
                        v1 = d1[d1['Subject'].isin(common_subs)]['Value'].values
                        v2 = d2[d2['Subject'].isin(common_subs)]['Value'].values
                        diffs = v1 - v2
                        t_stat, p = ttest_rel(v1, v2)
                        # Cohen's d (paired): mean(diff) / std(diff)
                        cohen_d = np.nanmean(diffs) / np.nanstd(diffs, ddof=1)
                        
                        label = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                        y_max = max(np.mean(v1) + (np.std(v1, ddof=1) / np.sqrt(len(v1))), 
                                  np.mean(v2) + (np.std(v2, ddof=1) / np.sqrt(len(v2))))
                        y_min = min(np.mean(v1) - (np.std(v1, ddof=1) / np.sqrt(len(v1))), 
                                  np.mean(v2) - (np.std(v2, ddof=1) / np.sqrt(len(v2))))
                        text_y = y_max + 0.015 if y_max >= 0 else y_min - 0.02
                        ax.text(i, text_y, label, ha='center', va='bottom' if y_max >= 0 else 'top', 
                                fontsize=12, fontweight='bold' if p < 0.05 else 'normal',
                                color='black' if p < 0.05 else '0.4')
                        
                        stat_summary.append({
                            'Metric': metric, 'Facet': f"{seed_type}_{side}", 'Window': w_name,
                            'Contrast': f"{conn1} vs {conn2}", 't_stat': t_stat, 'p_val': p, 'Cohen_d': cohen_d
                        })
        
        # Save Stats to CSV
        out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
        os.makedirs(out_dir, exist_ok=True)
        stats_file = os.path.join(out_dir, f'connectivity_functional_stats_{metric}_{voxRes}.csv')
        pd.DataFrame(stat_summary).to_csv(stats_file, index=False)
        print(f"Stats report saved to {stats_file}")

        title_m = metric.upper()
        g.fig.suptitle(f'Functional Hierarchy Bars ({title_m}, n={len(results["loaded_subjects"])}) - Paired stats', y=1.05, fontsize=16)
        g.set_axis_labels("", f"Relative {title_m} Change")
        y_lim = (-0.15, 0.15) if metric == 'imcoh' else (-0.1, 0.2)
        for ax in g.axes.flat: ax.set_ylim(y_lim)
        
        sns.despine()
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
    print("Done! Functional bar plots finalized for multiple metrics.")

if __name__ == '__main__':
    main()
