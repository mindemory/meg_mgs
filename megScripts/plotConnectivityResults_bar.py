#!/usr/bin/env python3
"""
Script to generate anatomical connectivity bar plots across trial phases.
Uses a 2x2 grid for anatomical links (lV2lF, lV2rF, rV2lF, rV2rF) and compares Target-Left vs Target-Right.
8mm BIDS standard summary version.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def load_anatomical_bar_results(bidsRoot, subjects, taskName='mgs', voxRes='8mm'):
    """
    Robustly load connectivity results for anatomical link summaries.
    """
    print(f"Loading results for {len(subjects)} subjects...")
    all_data = {
        'subjects': subjects,
        'loaded_subjects': [],
        'time_vector': None,
        'metrics': {m: {} for m in ['imcoh', 'wpli', 'dpli']}
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
                b_idxs = np.where((t_vec >= -0.6) & (t_vec <= 0.0))[0]
                
                for m_name, (data_dict, corr_type) in subj_metrics.items():
                    if m_name not in all_data['metrics']: continue
                    for k, val in data_dict.items():
                        if val.ndim > 1: val = np.mean(val.reshape(val.shape[0], -1), axis=1)
                        if len(b_idxs) > 0:
                            b_mean = np.nanmean(val[b_idxs])
                            corrected = (val / (b_mean + 1e-10) - 1) if corr_type == 'ratio' else (val - b_mean)
                        else: corrected = val
                        
                        if k not in all_data['metrics'][m_name]: all_data['metrics'][m_name][k] = []
                        all_data['metrics'][m_name][k].append(corrected)

                all_data['loaded_subjects'].append(subjID)
        except Exception as e:
            print(f"Error loading {subjID}: {e}")
            
    print(f"Successfully loaded {len(all_data['loaded_subjects'])} subjects")
    return all_data

def plot_anatomical_bars(results, bidsRoot, voxRes):
    """Generates the 2x2 grid of bar plots for anatomical links"""
    windows = {'Pre-delay': (-0.5, 0.0), 'Stimulus': (0.0, 0.5), 'Delay': (0.5, 1.5)}
    t_vec = results['time_vector']
    links = ['lV2lF', 'lV2rF', 'rV2lF', 'rV2rF']
    
    for metric in ['imcoh', 'dpli', 'wpli']:
        metric_data = results['metrics'][metric]
        if not metric_data: continue
        
        print(f"Generating anatomical bar plots for {metric}...")
        rows = []
        for sj_idx in range(len(results['loaded_subjects'])):
            for link in links:
                for cond in ['left', 'right']:
                    # Robust key matching
                    key = None
                    for k in [f'{cond}_{link}_{metric}', f'{cond}_{link}']:
                        if k in metric_data and len(metric_data[k]) > sj_idx: key = k; break
                    
                    if key:
                        trace = metric_data[key][sj_idx]
                        for w_name, (start, end) in windows.items():
                            mask = (t_vec >= start) & (t_vec <= end)
                            val = np.nanmean(trace[mask])
                            rows.append({
                                'Subject': sj_idx, 'Window': w_name, 'Link': link.replace('2', ' → '),
                                'Target': cond.capitalize(), 'Value': val
                            })
        
        # TEST: This confirms the file was indeed the new version
        if not rows: continue
        df = pd.DataFrame(rows)
        sns.set_theme(style="whitegrid")
        title_m = metric.upper() if metric != 'dpli' else 'dPLI'
        
        g = sns.catplot(
            data=df, kind="bar", x="Window", y="Value", hue="Target",
            col="Link", col_wrap=2, palette={'Left': 'royalblue', 'Right': 'crimson'},
            height=5, aspect=1.2, errorbar='se', capsize=.1
        )
        
        g.fig.suptitle(f'Anatomical Connectivity Bars: {title_m} (n={len(results["loaded_subjects"])})', y=1.02, fontsize=16)
        g.set_axis_labels("", f"Relative {title_m} Change")
        g.set_titles("{col_name}")
        
        y_lim = (-0.2, 0.2) if metric != 'dpli' else None
        if y_lim:
            for ax in g.axes.flat: ax.set_ylim(y_lim)

        out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'connectivity_bar_anatomical_{metric}_{voxRes}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def main():
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    taskName, voxRes = 'mgs', '8mm'
    import socket
    hostname = socket.gethostname()
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if hostname == 'zod' else '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    results = load_anatomical_bar_results(bidsRoot, subjects, taskName, voxRes)
    if results['loaded_subjects']:
        plot_anatomical_bars(results, bidsRoot, voxRes)
    print("Done! Anatomical bar plots saved to Fs04 folder.")

if __name__ == '__main__':
    main()
