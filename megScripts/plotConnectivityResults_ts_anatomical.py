#!/usr/bin/env python3
"""
Script to visualize connectivity between fixed anatomical regions.
Top Plot: Right-Frontal Target (comparing Left-Visual vs Right-Visual seeds)
Bottom Plot: Left-Frontal Target (comparing Left-Visual vs Right-Visual seeds)
Collapses across target conditions (left, right).
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

def load_anatomical_ts_results(bidsRoot, subjects, taskName='mgs', voxRes='8mm'):
    """
    Load connectivity results and group into anatomical categories.
    """
    print(f"Loading results for {len(subjects)} subjects...")
    
    all_data = {
        'subjects': subjects,
        'loaded_subjects': [],
        'time_vector': None,
        'raw_metrics': {m: {
            'lV_lF_L': [], 'lV_lF_R': [], 'rV_lF_L': [], 'rV_lF_R': [], # Target: Left Frontal
            'lV_rF_L': [], 'lV_rF_R': [], 'rV_rF_L': [], 'rV_rF_R': []  # Target: Right Frontal
        } for m in ['imcoh', 'wpli']}
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
                                    b_data = val[b_idxs]
                                    b_mean = np.nanmean(b_data)
                                    b_std  = np.nanstd(b_data)
                                    # Z-score baseline: (Data - Mean) / Std
                                    return (val - b_mean) / (b_std + 1e-10)
                                return val
                        return None

                    # Anatomical mappings (Keeping conditions separate)
                    # Left Frontal Target
                    all_data['raw_metrics'][m_name]['lV_lF_L'].append(get_subj_trace('left', 'lV2lF'))
                    all_data['raw_metrics'][m_name]['lV_lF_R'].append(get_subj_trace('right', 'lV2lF'))
                    all_data['raw_metrics'][m_name]['rV_lF_L'].append(get_subj_trace('left', 'rV2lF'))
                    all_data['raw_metrics'][m_name]['rV_lF_R'].append(get_subj_trace('right', 'rV2lF'))

                    # Right Frontal Target
                    all_data['raw_metrics'][m_name]['lV_rF_L'].append(get_subj_trace('left', 'lV2rF'))
                    all_data['raw_metrics'][m_name]['lV_rF_R'].append(get_subj_trace('right', 'lV2rF'))
                    all_data['raw_metrics'][m_name]['rV_rF_L'].append(get_subj_trace('left', 'rV2rF'))
                    all_data['raw_metrics'][m_name]['rV_rF_R'].append(get_subj_trace('right', 'rV2rF'))

                all_data['loaded_subjects'].append(subjID)
        except Exception as e:
            print(f"Error loading {subjID}: {e}")
            
    print(f"Successfully loaded {len(all_data['loaded_subjects'])} subjects")
    return all_data

def plot_anatomical_ts(results, bidsRoot, voxRes, metrics=['imcoh']):
    """
    Generate 2x1 grid of anatomical connectivity:
      Top Plot: Right-Frontal Target
      Bottom Plot: Left-Frontal Target
    """
    time_vector = results['time_vector']
    n_subs = len(results['loaded_subjects'])
    y_min, y_max = -0.15, 0.15

    for metric in metrics:
        metric_data = results['raw_metrics'][metric]
        if not any(metric_data.values()): continue

        print(f"Generating 2x4 anatomical comparison plots for {metric}...")

        def extract_stack(key):
            # Filtering out None entries from subject lists
            valid = [s for s in metric_data[key] if s is not None]
            return np.stack(valid) if valid else None

        fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharey=True)

        plot_configs = [
            (0, 'Right-Frontal', 
             [(extract_stack('lV_rF_L'), 'royalblue', 'LV - Left Target'),
              (extract_stack('lV_rF_R'), 'lightsteelblue', 'LV - Right Target'),
              (extract_stack('rV_rF_L'), 'crimson', 'RV - Left Target'),
              (extract_stack('rV_rF_R'), 'salmon', 'RV - Right Target')]),
            (1, 'Left-Frontal', 
             [(extract_stack('lV_lF_L'), 'royalblue', 'LV - Left Target'),
              (extract_stack('lV_lF_R'), 'lightsteelblue', 'LV - Right Target'),
              (extract_stack('rV_lF_L'), 'crimson', 'RV - Left Target'),
              (extract_stack('rV_lF_R'), 'salmon', 'RV - Right Target')]),
        ]

        def pointwise_ttest_mask(traces, time_vec, label=""):
            """Pointwise 1-sample t-test vs 0."""
            # Use scipy's vectorized ttest_1samp across the time dimension (axis 0)
            t_stats, p_vals = ttest_1samp(traces, 0, axis=0)
            
            sig_mask = p_vals < 0.05
            sig_count = np.sum(sig_mask)
            if sig_count > 0:
                sig_times = time_vec[sig_mask]
                print(f"    [{label}] Significant at {sig_count} timepoints.")
                print(f"      Timepoints (s): {', '.join([f'{t:.3f}' for t in sig_times])}")
            else:
                print(f"    [{label}] No significance found. Min p: {np.nanmin(p_vals):.4f}")
            return sig_mask

        for ax_idx, title, lines in plot_configs:
            ax = axes[ax_idx]
            
            bar_bottom = y_min + 0.002
            bar_height = (y_max - y_min) * 0.025

            for i_line, (subj_traces, color, label) in enumerate(lines):
                if subj_traces is None: continue
                n = subj_traces.shape[0]
                m = np.mean(subj_traces, axis=0)
                sem = np.std(subj_traces, axis=0, ddof=1) / np.sqrt(n)
                
                print(f"  Plotting {title} | {label}: N={n}")
                
                # Plot trace
                ax.plot(time_vector, m, color=color, lw=2.0, label=label)
                ax.fill_between(time_vector, m-sem, m+sem, color=color, alpha=0.1)
                
                # Stats (t-test)
                sig = pointwise_ttest_mask(subj_traces, time_vector, label=f"{title}-{label}")
                if np.any(sig):
                    y_off = bar_bottom + (i_line * bar_height * 1.1)
                    ax.fill_between(time_vector, y_off, y_off + bar_height,
                                    where=sig, color=color, alpha=0.9, zorder=3, interpolate=True)

            # Decorations
            ax.axhline(0, color='black', lw=1, alpha=0.3, ls='--')
            ax.axvline(0,   color='gray', ls='--', alpha=0.6)
            ax.axvline(0.2, color='gray', ls=':',  alpha=0.4)
            ax.axvline(1.7, color='black', ls='--', alpha=0.6)
            ax.set_xlim(-0.5, 1.8)
            ax.set_ylim(-3, 3) # Z-score scale
            ax.set_ylabel(f"Connectivity (Z-score)")
            ax.set_title(title, fontweight='bold', pad=15)
            ax.legend(loc='upper right', frameon=False, fontsize=10, ncol=2)
            ax.grid(False)

        axes[-1].set_xlabel("Time (s)")
        plt.suptitle(f'Rel. {metric.upper()} (Anatomical, n={n_subs})',
                     y=0.98, fontsize=15, fontweight='bold')

        out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'connectivity_ts_anatomical_{metric}_{voxRes}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f'connectivity_ts_anatomical_{metric}_{voxRes}.svg'), format='svg', bbox_inches='tight')
        plt.close()
        print(f"Saved anatomical time-series to {out_dir}")

def main():
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    taskName, voxRes = 'mgs', '8mm'
    import socket
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if socket.gethostname() == 'zod' else '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    results = load_anatomical_ts_results(bidsRoot, subjects, taskName, voxRes)
    if results['loaded_subjects']:
        plot_anatomical_ts(results, bidsRoot, voxRes)
    print("Done!")

if __name__ == '__main__':
    main()
