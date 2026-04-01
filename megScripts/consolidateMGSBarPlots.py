import os, pickle, socket
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import ttest_1samp

def get_bids_root():
    h = socket.gethostname()
    if 'vader' in h: return '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else: return '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'

def load_pkl(f):
    if os.path.exists(f):
        with open(f, 'rb') as fp: return pickle.load(fp)
    return None

def main():
    bidsRoot = get_bids_root()
    voxRes = '8mm'
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    time_v = np.linspace(-1.0, 2.0, 481)
    
    # Analysis Windows
    windows = {
        'Baseline': (-0.5, 0.0),
        'Stimulus': (0.1, 0.5),
        'Delay': (0.8, 1.5)
    }
    w_idx = {k: np.where((time_v >= v[0]) & (time_v <= v[1]))[0] for k, v in windows.items()}

    # ROI Voxel Mapping
    atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    h_masks = {
        'frontal': {'L': np.where(atlas['left_frontal_points'].flatten() == 1)[0], 'R': np.where(atlas['right_frontal_points'].flatten() == 1)[0]},
        'visual': {'L': np.where(atlas['left_visual_points'].flatten() == 1)[0], 'R': np.where(atlas['right_visual_points'].flatten() == 1)[0]}
    }
    
    hierarchies = [('frontal', 'visual', "Top-Down"), ('visual', 'frontal', "Bottom-Up")]
    mappings = ['i_same', 'i_cross', 'c_same', 'c_cross']
    styles = {
        'i_same': {'c': '#d62728', 'label': 'Ipsi Same'},
        'i_cross': {'c': '#ff7f0e', 'label': 'Ipsi Cross'},
        'c_same': {'c': '#1f77b4', 'label': 'Contra Same'},
        'c_cross': {'c': '#17becf', 'label': 'Contra Cross'}
    }
    phase_bands = ['theta', 'alpha', 'beta']
    power_bands = ['theta', 'alpha', 'beta', 'lowgamma']
    bands_dpli = ['theta', 'alpha', 'beta', 'lowgamma']
    
    outDir = os.path.join(bidsRoot, 'derivatives', 'figures', 'connectivity_bar')
    os.makedirs(outDir, exist_ok=True)

    def plot_statistical_bars(ax, data_struct, title_str):
        # data_struct[mapping][epoch] = [subject_delta_values]
        epochs = ['Stimulus', 'Delay']
        x = np.arange(len(epochs))
        width = 0.18
        
        for i, m in enumerate(mappings):
            means = [np.mean(data_struct[m][e]) if data_struct[m][e] else 0 for e in epochs]
            sems = [np.std(data_struct[m][e])/np.sqrt(len(data_struct[m][e])) if data_struct[m][e] else 0 for e in epochs]
            
            # Plot Bars
            bar_pos = x + (i-1.5)*width
            ax.bar(bar_pos, means, width, yerr=sems, color=styles[m]['c'], label=styles[m]['label'], capsize=3, alpha=0.7)
            
            # Plot Individual Subjects
            for j, e in enumerate(epochs):
                if data_struct[m][e]:
                    subs = data_struct[m][e]
                    jitter = (np.random.rand(len(subs)) - 0.5) * 0.05
                    ax.scatter(np.full(len(subs), bar_pos[j]) + jitter, subs, color=styles[m]['c'], s=5, alpha=0.3, edgecolors='none')
                    
                    # Significance Asterisks (One-sample t-test against 0)
                    t, p = ttest_1samp(subs, 0)
                    if p < 0.05:
                        offset = sems[j] + 0.005 # Above the error bar
                        mark = '**' if p < 0.01 else '*'
                        ax.text(bar_pos[j], max(means[j], 0) + offset, mark, ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_xticks(x); ax.set_xticklabels(epochs)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)
        ax.set_title(title_str, fontsize=10, fontweight='bold')
        ax.set_ylabel('dPLI (Directed Sync)', fontsize=8)

    # 1. CFC Bar Plot Generation
    for h_seed, h_targ, h_name in hierarchies:
        print(f"[*] Statistical Bars CFC: {h_name}")
        fig, axes = plt.subplots(len(power_bands), len(phase_bands), figsize=(18, 24))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        for r, eb in enumerate(power_bands):
            for c, pb in enumerate(phase_bands):
                mode = f"{pb}_to_{eb}"
                # data[mapping][epoch] = list of subject-level Delta-dPLI
                data = {m: {e: [] for e in ['Stimulus', 'Delay']} for m in mappings}
                
                for sub in subjects:
                    subName = f'sub-{sub:02d}'
                    cfcDir = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'CFC_{voxRes}')
                    
                    for m in mappings:
                        f1, f2 = None, None
                        if m == 'i_same':
                            f1 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_left_{h_seed}_to_left_{h_targ}_left_targets_dpli_{mode}.pkl')
                            f2 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_right_{h_seed}_to_right_{h_targ}_right_targets_dpli_{mode}.pkl')
                        elif m == 'i_cross':
                            f1 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_left_{h_seed}_to_right_{h_targ}_left_targets_dpli_{mode}.pkl')
                            f2 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_right_{h_seed}_to_left_{h_targ}_right_targets_dpli_{mode}.pkl')
                        elif m == 'c_same':
                            f1 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_right_{h_seed}_to_right_{h_targ}_left_targets_dpli_{mode}.pkl')
                            f2 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_left_{h_seed}_to_left_{h_targ}_right_targets_dpli_{mode}.pkl')
                        elif m == 'c_cross':
                            f1 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_right_{h_seed}_to_left_{h_targ}_left_targets_dpli_{mode}.pkl')
                            f2 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_left_{h_seed}_to_right_{h_targ}_right_targets_dpli_{mode}.pkl')

                        d1 = load_pkl(f1); d2 = load_pkl(f2)
                        if d1 is not None and d2 is not None:
                            avg_ts = np.mean([d1, d2], axis=0)
                            data[m]['Stimulus'].append(np.mean(avg_ts[w_idx['Stimulus']]))
                            data[m]['Delay'].append(np.mean(avg_ts[w_idx['Delay']]))

                plot_statistical_bars(axes[r, c], data, mode.upper())
                if r == 0 and c == 0: axes[r, c].legend(fontsize=8, loc='upper right')

        saveF = os.path.join(outDir, f'STAT_BAR_Gating_ByPhase_{h_name}.png')
        fig.savefig(saveF, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # 2. Standalone dPLI Bar Plot Generation
    print(f"[*] Statistical Bars dPLI")
    fig_d, axes_d = plt.subplots(len(bands_dpli), 2, figsize=(18, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    for row, band in enumerate(bands_dpli):
        for col, (h_seed, h_targ, h_name) in enumerate(hierarchies):
            data = {m: {e: [] for e in ['Stimulus', 'Delay']} for m in mappings}
            masks_t = h_masks[h_targ]
            for sub in subjects:
                subName = f'sub-{sub:02d}'
                connDir = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'connectivity_{voxRes}')
                f_is_1 = os.path.join(connDir, f'{subName}_task-mgs_seededConnectivity_{voxRes}_left_{h_seed}_left_dpli_{band}.pkl')
                f_is_2 = os.path.join(connDir, f'{subName}_task-mgs_seededConnectivity_{voxRes}_right_{h_seed}_right_dpli_{band}.pkl')
                f_cs_1 = os.path.join(connDir, f'{subName}_task-mgs_seededConnectivity_{voxRes}_right_{h_seed}_left_dpli_{band}.pkl')
                f_cs_2 = os.path.join(connDir, f'{subName}_task-mgs_seededConnectivity_{voxRes}_left_{h_seed}_right_dpli_{band}.pkl')
                d1=load_pkl(f_is_1); d2=load_pkl(f_is_2); d3=load_pkl(f_cs_1); d4=load_pkl(f_cs_2)
                if all([x is not None for x in [d1, d2, d3, d4]]):
                    s_ts = {
                        'i_same': np.mean([np.nanmean(d1[masks_t['L'], :], axis=0)-0.5, np.nanmean(d2[masks_t['R'], :], axis=0)-0.5], axis=0),
                        'i_cross': np.mean([np.nanmean(d1[masks_t['R'], :], axis=0)-0.5, np.nanmean(d2[masks_t['L'], :], axis=0)-0.5], axis=0),
                        'c_same': np.mean([np.nanmean(d3[masks_t['R'], :], axis=0)-0.5, np.nanmean(d4[masks_t['L'], :], axis=0)-0.5], axis=0),
                        'c_cross': np.mean([np.nanmean(d3[masks_t['L'], :], axis=0)-0.5, np.nanmean(d4[masks_t['R'], :], axis=0)-0.5], axis=0),
                    }
                    for m, ts in s_ts.items():
                        data[m]['Stimulus'].append(np.mean(ts[w_idx['Stimulus']]))
                        data[m]['Delay'].append(np.mean(ts[w_idx['Delay']]))

            plot_statistical_bars(axes_d[row, col], data, f"{h_name} | {band.capitalize()}")
            if row == 0 and col == 0: axes_d[row, col].legend(fontsize=8, loc='upper right')

    saveF_d = os.path.join(outDir, 'STAT_BAR_Sync_ByPhase.png')
    fig_d.savefig(saveF_d, dpi=300, bbox_inches='tight')
    plt.close(fig_d)
    print(f"[*] Statistical bar plots saved to {outDir}")

if __name__ == "__main__":
    main()
