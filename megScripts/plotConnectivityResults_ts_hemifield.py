#!/usr/bin/env python3
"""
Script to visualize frontal connectivity collapsed across visual hemifield.
Two subplots:
  1. Contra-Frontal connections: average of (ipsi_cross + contra_cross) across subjects
  2. Ipsi-Frontal connections:   average of (ipsi_within + contra_within) across subjects

Significance: pointwise 1-sample t-test against zero, uncorrected (p < 0.05).
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

def plot_hemifield_ts(results, bidsRoot, voxRes, metrics=['imcoh']):
    """
    Generate 2x1 grid collapsing across visual hemifield:
      Row 1: Contra-Frontal (avg of ipsi_cross + contra_cross)
      Row 2: Ipsi-Frontal   (avg of ipsi_within + contra_within)
    Significance: pointwise 1-sample sign-flipping permutation test vs 0, uncorrected.
    """
    time_vector = results['time_vector']
    n_subs = len(results['loaded_subjects'])
    y_min, y_max = -0.1, 0.1

    for metric in metrics:
        metric_data = results['raw_metrics'][metric]
        if not any(metric_data.values()): continue

        print(f"Generating 2x1 hemifield time-series plots for {metric}...")

        # Build per-subject collapsed traces
        # Contra-Frontal: both visual seeds connecting to the contra-lateral frontal
        #   ipsi_cross   = ipsi visual  → contra frontal
        #   contra_within= contra visual → contra frontal
        contra_frontal = []
        for i in range(n_subs):
            traces = []
            if i < len(metric_data['ipsi_cross']):    traces.append(metric_data['ipsi_cross'][i])
            if i < len(metric_data['contra_within']):  traces.append(metric_data['contra_within'][i])
            if traces: contra_frontal.append(np.mean(traces, axis=0))

        # Ipsi-Frontal: both visual seeds connecting to the ipsi-lateral frontal
        #   ipsi_within  = ipsi visual  → ipsi frontal
        #   contra_cross = contra visual → ipsi frontal
        ipsi_frontal = []
        for i in range(n_subs):
            traces = []
            if i < len(metric_data['ipsi_within']):   traces.append(metric_data['ipsi_within'][i])
            if i < len(metric_data['contra_cross']):   traces.append(metric_data['contra_cross'][i])
            if traces: ipsi_frontal.append(np.mean(traces, axis=0))

        contra_frontal = np.stack(contra_frontal)  # (N, T)
        ipsi_frontal   = np.stack(ipsi_frontal)    # (N, T)

        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharey=True)

        plot_configs = [
            (0, 'Contra-Frontal (both visual seeds → contra frontal)', contra_frontal, 'royalblue'),
            (1, 'Ipsi-Frontal (both visual seeds → ipsi frontal)',     ipsi_frontal,   'crimson'),
        ]

        for ax_idx, title, subj_traces, color in plot_configs:
            ax = axes[ax_idx]
            n = subj_traces.shape[0]

            # Mean and SEM
            m   = np.mean(subj_traces, axis=0)
            sem = np.std(subj_traces, axis=0, ddof=1) / np.sqrt(n)

            # Vectorized sign-flipping permutation test vs 0 at each timepoint (1000 perms)
            N = subj_traces.shape[0]
            obs_mean = np.mean(subj_traces, axis=0)
            signs = np.random.choice([-1, 1], size=(1000, N)).astype(np.float32)
            null_means = (signs @ subj_traces.astype(np.float32)) / N  # (1000, T)
            p_vals = (np.sum(np.abs(null_means) >= np.abs(obs_mean), axis=0) + 1) / 1001
            sig_mask = p_vals < 0.05  # raw uncorrected

            # Plot trace + SEM band
            ax.plot(time_vector, m, color=color, lw=2, label=f'Mean ± SEM (n={n})')
            ax.fill_between(time_vector, m - sem, m + sem, color=color, alpha=0.15)

            # Draw significance bars just below x-axis as colored strips
            bar_bottom = y_min + 0.002
            bar_height = (y_max - y_min) * 0.025
            if np.any(sig_mask):
                ax.fill_between(time_vector, bar_bottom, bar_bottom + bar_height,
                                where=sig_mask, color=color, alpha=0.8, zorder=3,
                                label='p < 0.05 (permutation, uncorrected)')

            # Decorations
            ax.axhline(0, color='black', lw=1, alpha=0.3, ls='--')
            ax.axvline(0,   color='red',    ls='--', alpha=0.6, label='Cue')
            ax.axvline(0.2, color='orange', ls='--', alpha=0.6, label='Delay')
            ax.axvline(1.7, color='green',  ls='--', alpha=0.6, label='Response')
            ax.set_xlim(-0.5, 1.8)
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel(f"Relative {metric.upper()} Change")
            ax.set_title(title)
            ax.legend(loc='upper right', frameon=False, fontsize=9)
            ax.grid(False)

        axes[-1].set_xlabel("Time (s)")
        plt.suptitle(f'Frontal Connectivity Collapsed Across Visual Hemifield ({metric.upper()}, n={n_subs})',
                     y=0.98, fontsize=13)

        out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'connectivity_ts_hemifield_{metric}_{voxRes}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f'connectivity_ts_hemifield_{metric}_{voxRes}.svg'), format='svg', bbox_inches='tight')
        plt.close()
        print(f"Saved hemifield time-series to {out_dir}")

def main():
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    taskName, voxRes = 'mgs', '8mm'
    import socket
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if socket.gethostname() == 'zod' else '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    results = load_functional_ts_results(bidsRoot, subjects, taskName, voxRes)
    if results['loaded_subjects']:
        plot_hemifield_ts(results, bidsRoot, voxRes)
    print("Done!")

if __name__ == '__main__':
    main()
