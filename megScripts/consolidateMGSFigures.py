import os, pickle, socket
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat

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
    
    # ROI Voxel Mapping
    atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    h_masks = {
        'frontal': {'L': np.where(atlas['left_frontal_points'].flatten() == 1)[0], 'R': np.where(atlas['right_frontal_points'].flatten() == 1)[0]},
        'visual': {'L': np.where(atlas['left_visual_points'].flatten() == 1)[0], 'R': np.where(atlas['right_visual_points'].flatten() == 1)[0]}
    }
    
    # Hierarchies
    hierarchies = [
        ('frontal', 'visual', "Top-Down_(FEF_leads_Visual)"),
        ('visual', 'frontal', "Bottom-Up_(Visual_leads_FEF)")
    ]
    
    styles = {
        'i_same': {'c': '#d62728', 'ls': '-', 'label': 'Ipsi Seed -> Same-Hem'},
        'i_cross': {'c': '#ff7f0e', 'ls': '--', 'label': 'Ipsi Seed -> Cross-Hem'},
        'c_same': {'c': '#1f77b4', 'ls': '-', 'label': 'Contra Seed -> Same-Hem'},
        'c_cross': {'c': '#17becf', 'ls': '--', 'label': 'Contra Seed -> Cross-Hem'}
    }

    phase_bands = ['theta', 'alpha', 'beta']
    power_bands = ['theta', 'alpha', 'beta', 'lowgamma']

    for h_seed, h_targ, h_name in hierarchies:
        print(f"[*] Consolidating {h_name}...")
        fig, axes = plt.subplots(len(power_bands), len(phase_bands), figsize=(18, 24), sharex=True)
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        
        for r, eb in enumerate(power_bands):
            for c, pb in enumerate(phase_bands):
                mode = f"{pb}_to_{eb}"
                all_mode = { 'i_same': [], 'i_cross': [], 'c_same': [], 'c_cross': [] }
                
                for sub in subjects:
                    subName = f'sub-{sub:02d}'
                    cfcDir = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'CFC_{voxRes}')
                    
                    f_is_same_1 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_left_{h_seed}_to_left_{h_targ}_left_targets_dpli_{mode}.pkl')
                    f_is_same_2 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_right_{h_seed}_to_right_{h_targ}_right_targets_dpli_{mode}.pkl')
                    f_is_cross_1 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_left_{h_seed}_to_right_{h_targ}_left_targets_dpli_{mode}.pkl')
                    f_is_cross_2 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_right_{h_seed}_to_left_{h_targ}_right_targets_dpli_{mode}.pkl')
                    
                    f_cs_same_1 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_right_{h_seed}_to_right_{h_targ}_left_targets_dpli_{mode}.pkl')
                    f_cs_same_2 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_left_{h_seed}_to_left_{h_targ}_right_targets_dpli_{mode}.pkl')
                    f_cs_cross_1 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_right_{h_seed}_to_left_{h_targ}_left_targets_dpli_{mode}.pkl')
                    f_cs_cross_2 = os.path.join(cfcDir, f'{subName}_task-mgs_dCFC_{voxRes}_left_{h_seed}_to_right_{h_targ}_right_targets_dpli_{mode}.pkl')

                    for k, f1, f2 in [('i_same', f_is_same_1, f_is_same_2), ('i_cross', f_is_cross_1, f_is_cross_2),
                                      ('c_same', f_cs_same_1, f_cs_same_2), ('c_cross', f_cs_cross_1, f_cs_cross_2)]:
                        d1 = load_pkl(f1); d2 = load_pkl(f2)
                        if d1 is not None and d2 is not None: all_mode[k].append(np.mean([d1, d2], axis=0))

                ax = axes[r, c]
                for key in styles:
                    if all_mode[key]:
                        dat = np.array(all_mode[key])
                        av = np.nanmean(dat, axis=0); se = np.nanstd(dat, axis=0)/np.sqrt(len(dat))
                        ax.plot(time_v, av, color=styles[key]['c'], linestyle=styles[key]['ls'], label=styles[key]['label'])
                        ax.fill_between(time_v, av-se, av+se, color=styles[key]['c'], alpha=0.1)
                
                ax.set_title(f"{mode.replace('_', ' ').upper()}", fontsize=12, fontweight='bold')
                ax.axhline(0, color='black', alpha=0.3); ax.axvline(0, color='red', alpha=0.5); ax.set_xlim(-0.5, 1.5)
                if r == 0 and c == 0: ax.legend(loc='upper right', fontsize=8)

        mFile = os.path.join(bidsRoot, 'derivatives', 'figures', 'connectivity', f'MASTER_GOD_MODE_CFC_12-Mode_{h_name}_8mm.png')
        fig.savefig(mFile, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[*] MASTER GATING SAVED: {mFile}")

    # standalone dPLI
    bands_dpli = ['theta', 'alpha', 'beta', 'lowgamma']
    fig_dpli, axes_dpli = plt.subplots(len(bands_dpli), 2, figsize=(18, 20), sharex=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    for row, band in enumerate(bands_dpli):
        for col, (h_seed, h_targ, h_name) in enumerate(hierarchies):
            all_dpli = { 'i_same': [], 'i_cross': [], 'c_same': [], 'c_cross': [] }
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
                    all_dpli['i_same'].append(np.mean([np.nanmean(d1[masks_t['L'], :], axis=0)-0.5, np.nanmean(d2[masks_t['R'], :], axis=0)-0.5], axis=0))
                    all_dpli['i_cross'].append(np.mean([np.nanmean(d1[masks_t['R'], :], axis=0)-0.5, np.nanmean(d2[masks_t['L'], :], axis=0)-0.5], axis=0))
                    all_dpli['c_same'].append(np.mean([np.nanmean(d3[masks_t['R'], :], axis=0)-0.5, np.nanmean(d4[masks_t['L'], :], axis=0)-0.5], axis=0))
                    all_dpli['c_cross'].append(np.mean([np.nanmean(d3[masks_t['L'], :], axis=0)-0.5, np.nanmean(d4[masks_t['R'], :], axis=0)-0.5], axis=0))
            
            ax_d = axes_dpli[row, col]
            for key in styles:
                if all_dpli[key]:
                    dat = np.array(all_dpli[key])
                    av = np.nanmean(dat, axis=0); se = np.nanstd(dat, axis=0)/np.sqrt(len(dat))
                    ax_d.plot(time_v, av, color=styles[key]['c'], linestyle=styles[key]['ls'], label=styles[key]['label'])
                    ax_d.fill_between(time_v, av-se, av+se, color=styles[key]['c'], alpha=0.1)
            ax_d.set_title(f"{h_name}\n{band.capitalize()} Sync", fontsize=12, fontweight='bold')
            ax_d.axhline(0, color='black', alpha=0.3); ax_d.axvline(0, color='red', alpha=0.5)
            ax_d.set_xlim(-0.5, 1.5)
            if row == 0 and col == 0: ax_d.legend(loc='upper right', fontsize=8)

    dpliF = os.path.join(bidsRoot, 'derivatives', 'figures', 'connectivity', f'MASTER_GOD_MODE_dPLI_Stand-alone_8mm.png')
    fig_dpli.savefig(dpliF, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[*] MASTER SYNC SAVED: {dpliF}")

if __name__ == "__main__":
    main()
