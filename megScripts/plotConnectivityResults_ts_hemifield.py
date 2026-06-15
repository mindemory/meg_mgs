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
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

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
                    
                    def get_subj_trace_raw(cond, link):
                        for k in [f'{cond}_{link}_{m_name}', f'{cond}_{link}']:
                            if k in data_dict:
                                val = data_dict[k]
                                if val.ndim > 1: val = np.mean(val.reshape(val.shape[0], -1), axis=1)
                                # Returning RAW values (no baseline correction)
                                return val
                        return None

                    def avg_available_raw(cond_links):
                        valid = [get_subj_trace_raw(c, l) for c, l in cond_links]
                        valid = [v for v in valid if v is not None]
                        return np.mean(valid, axis=0) if valid else None

                    # Mapping raw values (Contra-Vis/Ipsi-Vis x Contra-Front/Ipsi-Front)
                    # 1. Contra-Visual to Contra-Frontal (contra_within)
                    cw_trace = avg_available_raw([('left', 'rV2rF'), ('right', 'lV2lF')])
                    if cw_trace is not None: all_data['raw_metrics'][m_name]['contra_within'].append(cw_trace)
                    
                    # 2. Ipsi-Visual to Contra-Frontal (ipsi_cross)
                    ic_trace = avg_available_raw([('left', 'lV2rF'), ('right', 'rV2lF')])
                    if ic_trace is not None: all_data['raw_metrics'][m_name]['ipsi_cross'].append(ic_trace)

                    # 3. Contra-Visual to Ipsi-Frontal (contra_cross)
                    cc_trace = avg_available_raw([('left', 'rV2lF'), ('right', 'lV2rF')])
                    if cc_trace is not None: all_data['raw_metrics'][m_name]['contra_cross'].append(cc_trace)

                    # 4. Ipsi-Visual to Ipsi-Frontal (ipsi_within)
                    iw_trace = avg_available_raw([('left', 'lV2lF'), ('right', 'rV2rF')])
                    if iw_trace is not None: all_data['raw_metrics'][m_name]['ipsi_within'].append(iw_trace)

                all_data['loaded_subjects'].append(subjID)
        except Exception as e:
            print(f"Error loading {subjID}: {e}")
            
    print(f"Successfully loaded {len(all_data['loaded_subjects'])} subjects")
    return all_data

def plot_hemifield_ts(results, bidsRoot, voxRes, metrics=['imcoh', 'wpli']):
    """
    Generate 2x2 Master Figure:
      Row 1: Z-scored traces (Contra-Vis vs Ipsi-Vis seeds)
      Row 2: Lateralization Index (CLI)
    """
    time_vector = results['time_vector']
    n_subs = len(results['loaded_subjects'])
    
    # Baseline mask for Z-scoring
    b_mask = (time_vector >= -0.6) & (time_vector <= 0.0)
    b_idxs = np.where(b_mask)[0]

    for metric in metrics:
        metric_data = results['raw_metrics'][metric]
        if not any(metric_data.values()): continue

        print(f"Generating 2x2 Master Figure for {metric}...")

        # 1. Prepare Stacks (Raw)
        cw = np.stack(metric_data['contra_within']) # Contra-Vis -> Contra-Front
        ic = np.stack(metric_data['ipsi_cross'])     # Ipsi-Vis   -> Contra-Front
        cc = np.stack(metric_data['contra_cross'])  # Contra-Vis -> Ipsi-Front
        iw = np.stack(metric_data['ipsi_within'])   # Ipsi-Vis   -> Ipsi-Front

        # 2. Z-scoring function
        def zscore_traces(traces):
            z_traces = np.zeros_like(traces)
            for i in range(traces.shape[0]):
                b_data = traces[i, b_idxs]
                b_mean = np.nanmean(b_data)
                b_std  = np.nanstd(b_data)
                z_traces[i] = (traces[i] - b_mean) / (b_std + 1e-10)
            return z_traces

        # 3. Calculate Derived Metrics
        # 3. Calculate Z-scores
        z_cw = zscore_traces(cw)
        z_ic = zscore_traces(ic)
        z_cc = zscore_traces(cc)
        z_iw = zscore_traces(iw)

        # Plot Configs
        plot_configs = [
            (0, 0, (z_cw, z_ic), "Contralateral Frontal Target", ["#e6550d", "#fec44f"], ["Contra-Visual", "Ipsi-Visual"]),
            (0, 1, (z_cc, z_iw), "Ipsilateral Frontal Target",   ["#1b9e77", "#0571b0"], ["Contra-Visual", "Ipsi-Visual"]),
            (1, 0, (z_ic, z_cc), "Cross-Hemisphere Connections", ["#fec44f", "#1b9e77"], ["Ipsi-Vis to Contra-Front", "Contra-Vis to Ipsi-Front"]),
            (1, 1, (z_cw, z_iw), "Within-Hemisphere Connections", ["#e6550d", "#0571b0"], ["Contra-Vis to Contra-Front", "Ipsi-Vis to Ipsi-Front"])
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
        r1_y_lim = (-1.5, 1.5)

        def pointwise_ttest_mask(traces, time_vec, label=""):
            """Pointwise 1-sample t-test vs 0 within [0, 1.7]s."""
            _, p_vals = ttest_1samp(traces, 0, axis=0)
            sig_mask = p_vals < 0.05
            
            # Apply time window filter
            t_window = (time_vec >= 0.0) & (time_vec <= 1.7)
            sig_mask = sig_mask & t_window
            
            sig_count = np.sum(sig_mask)
            if sig_count > 0:
                print(f"    [{label}] Significant at {sig_count} timepoints in [0, 1.7]s.")
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
            
            # Paired contrast sig
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
        
        plt.suptitle(f'{metric.upper()} Connectivity (N={n_subs})',
                     y=0.98, fontsize=18, fontweight='bold')

        out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'connectivity_ts_master_{metric}_{voxRes}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(out_dir, f'connectivity_ts_master_{metric}_{voxRes}.svg'), format='svg', bbox_inches='tight')
        plt.close()
        print(f"Saved master figure to {out_dir}")

def main():
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    taskName, voxRes = 'mgs', '8mm'
    import socket
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if socket.gethostname() == 'zod' else '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    results = load_functional_ts_results(bidsRoot, subjects, taskName, voxRes)
    if results['loaded_subjects']:
        plot_hemifield_ts(results, bidsRoot, voxRes, metrics=['imcoh'])
    print("Done!")

if __name__ == '__main__':
    main()
