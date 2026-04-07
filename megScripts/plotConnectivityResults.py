#!/usr/bin/env python3
"""
Script to load and plot connectivity results from all subjects.
Loads coherence and imaginary coherence data, computes averages across subjects, and plots results.
Features time-resolved cluster-based permutation testing via MNE in a narrowed window (0.1s-1.5s).
"""

import os 
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, t
from statsmodels.stats.multitest import multipletests
from mne.stats import permutation_cluster_1samp_test

def load_connectivity_results(bidsRoot, subjects, taskName='mgs', voxRes='8mm'):
    """
    Load connectivity results for all subjects with baseline correction.
    """
    print(f"Loading connectivity results for {len(subjects)} subjects...")
    all_results = {
        'subjects': subjects,
        'coh_results': {}, 'imcoh_results': {},
        'dpli_results': {}, 'wpli_results': {},
        'wpli_raw_results': {}, 'time_vector': None,
        'loaded_subjects': []
    }
    
    metrics = [
        ('coh_results', 'ratio'),
        ('imcoh_results', 'ratio'),
        (None, None), # Skip PLV
        ('dpli_results', 'ratio'),
        ('wpli_results', 'ratio')
    ]
    
    for subjID in subjects:
        subjDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'connectivity_{voxRes}')
        outputFile = os.path.join(subjDir, f'sub-{subjID:02d}_task-{taskName}_connectivity_{voxRes}.pkl')
        
        if not os.path.exists(outputFile): continue
            
        try:
            with open(outputFile, 'rb') as f:
                for res_key, corr_type in metrics:
                    try:
                        data_dict = pickle.load(f)
                    except EOFError: continue
                    
                    if res_key is None or data_dict is None: continue
                        
                    if all_results['time_vector'] is None and data_dict:
                        first_val = list(data_dict.values())[0]
                        all_results['time_vector'] = np.linspace(-1.0, 2.0, len(first_val))
                    
                    time_vector = all_results['time_vector']
                    baseline_mask = (time_vector >= -0.6) & (time_vector <= 0.0)
                    baseline_indices = np.where(baseline_mask)[0]
                    
                    for roi_key, value in data_dict.items():
                        if roi_key not in all_results[res_key]: all_results[res_key][roi_key] = []
                        
                        if res_key == 'wpli_results':
                            if roi_key not in all_results['wpli_raw_results']: all_results['wpli_raw_results'][roi_key] = []
                            all_results['wpli_raw_results'][roi_key].append(value if value.ndim == 1 else np.mean(value.reshape(value.shape[0], -1), axis=1))

                        if value.ndim > 1:
                            value = np.mean(value.reshape(value.shape[0], -1), axis=1)
                        
                        if len(baseline_indices) > 0:
                            b_mean = np.nanmean(value[baseline_indices])
                            corrected = value / (b_mean + 1e-10) - 1 if corr_type == 'ratio' else value - b_mean
                        else:
                            corrected = value
                        
                        all_results[res_key][roi_key].append(corrected)
            
            all_results['loaded_subjects'].append(subjID)
            
        except Exception as e:
            print(f"  Error loading subject {subjID:02d}: {e}")
            
    print(f"Successfully loaded {len(all_results['loaded_subjects'])} subjects")
    return all_results

def compute_averages(all_results):
    """
    Compute averages and perform Cluster-Based Permutation Testing in narrowed window.
    """
    print("Computing population statistics with Narrowed Cluster-Based Permutation (0.1s-1.5s)...")
    n_subs = len(all_results['loaded_subjects'])
    if n_subs < 2: return {}
    
    averaged_results = {'time_vector': all_results['time_vector'], 'n_subjects': n_subs}
    category_map = {'coh': 'coh_results', 'imcoh': 'imcoh_results', 'dpli': 'dpli_results', 
                    'wpli': 'wpli_results', 'wpli_raw': 'wpli_raw_results'}
    
    for cat, res_key in category_map.items():
        if res_key not in all_results or not all_results[res_key]: continue
        averaged_results[f'{cat}_means'] = {}
        averaged_results[f'{cat}_stds'] = {}
        for roi_key, subject_data in all_results[res_key].items():
            if len(subject_data) == 0: continue
            stacked = np.stack(subject_data, axis=0)
            averaged_results[f'{cat}_means'][roi_key] = np.mean(stacked, axis=0)
            averaged_results[f'{cat}_stds'][roi_key] = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(n_subs)

    # Statistical Window and Threshold
    time_vector = all_results['time_vector']
    mask_win = (time_vector >= 0.1) & (time_vector <= 1.5)
    win_idx = np.where(mask_win)[0]
    cluster_threshold = t.ppf(1 - 0.05 / 2, df=n_subs - 1)
    
    func_map = {'imcoh': 'imcoh_results', 'wpli': 'wpli_results'}
    for metric, res_key in func_map.items():
        if res_key not in all_results or not all_results[res_key]: continue
        raw = all_results[res_key]
        cat_data = {'ipsi_within': [], 'ipsi_cross': [], 'contra_within': [], 'contra_cross': []}
        
        for sj_idx in range(n_subs):
            def get_data(cond, link):
                for k in [f'{cond}_{link}_{metric}', f'{cond}_{link}']:
                    if k in raw and sj_idx < len(raw[k]): return raw[k][sj_idx]
                return None

            iw = [get_data('left', 'lV2lF'), get_data('right', 'rV2rF')]
            ic = [get_data('left', 'lV2rF'), get_data('right', 'lV2lF')]
            cw = [get_data('left', 'rV2rF'), get_data('right', 'lV2lF')]
            cc = [get_data('left', 'rV2lF'), get_data('right', 'rV2rF')]
            
            for c_list, target in [(iw, 'ipsi_within'), (ic, 'ipsi_cross'), (cw, 'contra_within'), (cc, 'contra_cross')]:
                valid = [d for d in c_list if d is not None]
                if valid: cat_data[target].append(np.mean(valid, axis=0))

        stacked_cats = {}
        for cat, traces in cat_data.items():
            if traces:
                stacked = np.stack(traces, axis=0)
                stacked_cats[cat] = stacked
                averaged_results[f'func_{metric}_{cat}_mean'] = np.mean(stacked, axis=0)
                averaged_results[f'func_{metric}_{cat}_sem'] = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(len(traces))

        # 6 Statistical Facets
        comparisons = [
            ('ipsi_within', 'ipsi_cross'), ('contra_within', 'contra_cross'),
            ('contra_cross', 'ipsi_cross'), ('ipsi_within', 'contra_cross'),
            ('contra_within', 'ipsi_cross'), ('contra_within', 'ipsi_within')
        ]
        
        for c1, c2 in comparisons:
            if c1 in stacked_cats and c2 in stacked_cats:
                v1, v2 = stacked_cats[c1], stacked_cats[c2]
                diffs_full = v1 - v2
                diff_win = diffs_full[:, win_idx]
                
                print(f"  Computing Cluster Permutation [{metric}]: {c1} vs {c2} (Window: 0.1s-1.5s)...")
                t_obs, clusters, cluster_p_values, H0 = permutation_cluster_1samp_test(
                    diff_win, n_permutations=1024, threshold=cluster_threshold, tail=0, out_type='mask'
                )
                
                # Combine significant clusters into a mask mapped back to full time
                sig_mask = np.zeros(v1.shape[1], dtype=bool)
                for i_c, c_mask in enumerate(clusters):
                    if cluster_p_values[i_c] < 0.05:
                        sig_mask[win_idx[c_mask]] = True
                
                averaged_results[f'func_{metric}_{c1}_vs_{c2}_sig'] = sig_mask

    return averaged_results

def add_plot_decorations(ax, title=None, ylabel=None, ylim=None):
    """MEG Standard decorations"""
    ax.axhline(0, color='black', lw=1, alpha=0.5, ls='--')
    ax.axvline(0, color='red', ls='--', alpha=0.6, label='Cue')
    ax.axvline(0.2, color='orange', ls='--', alpha=0.6, label='Delay')
    ax.axvline(1.7, color='green', ls='--', alpha=0.6, label='Response')
    ax.set_xlim(-0.5, 1.8)
    if title: ax.set_title(title, fontsize=12)
    if ylabel: ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(ylim)
    ax.set_xlabel('Time (s)')

def plot_anatomical_connectivity(averaged_results, bidsRoot, voxRes, metric='imcoh'):
    """Plot 2x2 grid for anatomical links"""
    time_vector = averaged_results['time_vector']
    links = ['lV2lF', 'lV2rF', 'rV2lF', 'rV2rF']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    title_m = metric.upper()
    fig.suptitle(f'Anatomical Mapping: {title_m} (n={averaged_results["n_subjects"]})', fontsize=16)
    
    means = averaged_results.get(f'{metric}_means', {})
    stds = averaged_results.get(f'{metric}_stds', {})
    
    for i, link in enumerate(links):
        ax = axes.flatten()[i]
        m, s = means.get(link), stds.get(link)
        if m is not None:
            ax.plot(time_vector, m, color='blue', lw=2.5)
            ax.fill_between(time_vector, m-s, m+s, color='blue', alpha=0.15)
        add_plot_decorations(ax, title=link.replace('2', ' → '), ylabel=f'Rel {title_m}', ylim=(-0.2, 0.2))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f'connectivity_{metric}_{voxRes}.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, f'connectivity_{metric}_{voxRes}.svg'), format='svg')
    plt.close()

def plot_functional_connectivity(averaged_results, bidsRoot, voxRes, metric='imcoh'):
    """Plots 2x3 grid with overlay and temporal significance"""
    if f'func_{metric}_ipsi_within_mean' not in averaged_results: return
    
    print(f"Plotting 2x3 plates for {metric} (High-Power Window)...")
    time_vector = averaged_results['time_vector']
    fig, axes = plt.subplots(2, 3, figsize=(24, 14), sharey=True)
    title_m = metric.upper()
    fig.suptitle(f'Functional Hierarchy: {title_m} Focused Significance (0.1s-1.5s, n={averaged_results["n_subjects"]})', fontsize=16)
    
    styles = {'within': {'color': 'blue', 'label': 'Within-Hemi / Same-Visual'}, 
              'cross': {'color': 'red', 'label': 'Cross-Hemi / Cross-Visual'}}
    
    plot_configs = [
        (0, 0, 'Ipsilateral Visual Seed', 'ipsi_within', 'ipsi_cross', 'Seed side == Target side'),
        (0, 1, 'Contralateral Visual Seed', 'contra_within', 'contra_cross', 'Seed side != Target side'),
        (0, 2, 'Hierarchy: Ipsi-Vis vs Contra-Vis', 'contra_cross', 'ipsi_cross', 'High-Power Contrast'),
        (1, 0, 'Ipsilateral Frontal Seed', 'ipsi_within', 'contra_cross', 'Seed side == Target side'),
        (1, 1, 'Contralateral Frontal Seed', 'contra_within', 'ipsi_cross', 'Seed side != Target side'),
        (1, 2, 'Hierarchy: Within-Hemi Comparative', 'contra_within', 'ipsi_within', 'Within-Hemi Integration')
    ]

    for row, col, title, b_cat, r_cat, desc in plot_configs:
        ax = axes[row, col]
        y_lim = (-0.2, 0.2)
        
        # Plot Traces
        for cat, s_key in zip([b_cat, r_cat], ['within', 'cross']):
            m_key, s_key_str = f'func_{metric}_{cat}_mean', f'func_{metric}_{cat}_sem'
            if m_key in averaged_results:
                m, s = averaged_results[m_key], averaged_results[s_key_str]
                if col == 2:
                    label = 'Ipsi Visual ↔ Cross-Frontal' if cat == 'ipsi_cross' else 'Contra Visual ↔ Ipsi-Frontal'
                    if row == 1: label = 'Ipsi Visual ↔ Ipsi-Frontal' if cat == 'ipsi_within' else 'Contra Visual ↔ Cross-Frontal'
                else:
                    label = styles[s_key]['label']
                ax.plot(time_vector, m, color=styles[s_key]['color'], lw=2.5, label=label)
                ax.fill_between(time_vector, m-s, m+s, color=styles[s_key]['color'], alpha=0.15)
        
        # Plot Significance Bar (use squares for clear cluster mapping)
        sig_key = f'func_{metric}_{b_cat}_vs_{r_cat}_sig'
        if sig_key in averaged_results:
            mask = averaged_results[sig_key]
            if np.any(mask):
                y_pos = y_lim[1] * 0.95
                ax.plot(time_vector[mask], [y_pos] * np.sum(mask), 's', color='black', markersize=2, label='p_cluster < 0.05')
        
        add_plot_decorations(ax, title=f'{title}\n({desc})', ylabel=f'Rel {title_m}' if col == 0 else None, ylim=y_lim)
        if col == 2 or (row == 0 and col == 0): ax.legend(loc='lower center', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'Fs04')
    plt.savefig(os.path.join(out_dir, f'connectivity_functional_{metric}_{voxRes}.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, f'connectivity_functional_{metric}_{voxRes}.svg'), format='svg')
    plt.close()

def main():
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    taskName, voxRes = 'mgs', '8mm'
    import socket
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS' if socket.gethostname() == 'zod' else '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    all_results = load_connectivity_results(bidsRoot, subjects, taskName, voxRes)
    if len(all_results['loaded_subjects']) > 1:
        averaged_results = compute_averages(all_results)
        for m in ['imcoh']:
            plot_anatomical_connectivity(averaged_results, bidsRoot, voxRes, metric=m)
            plot_functional_connectivity(averaged_results, bidsRoot, voxRes, metric=m)
        print("Analysis completed! High-power Focused Traces (0.1-1.5s) saved to Fs04.")

if __name__ == '__main__':
    main()
