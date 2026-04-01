import os, pickle, h5py, socket
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_bids_root():
    h = socket.gethostname()
    if 'vader' in h: return '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else: return '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'

def plot_cfc_hierarchy_matrix(subjects, voxRes, seed_names, target_names, title_suffix, filename_suffix):
    """
    Generate a 3x3 Grid for a specific ROI-to-ROI relationship.
    Groups into Ipsi (within-hemisphere target stimulus side) and Contra (other).
    """
    bidsRoot = get_bids_root()
    time_v = np.linspace(-1.0, 2.0, 481)
    
    bands = [(4, 8), (8, 13), (13, 18)]
    label_map = {(4, 8): 'theta', (8, 13): 'alpha', (13, 18): 'beta'}
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for r, envB in enumerate(bands):
        for c, phaseB in enumerate(bands):
            ax = axes[r, c]
            s_lab = label_map[phaseB]
            e_lab = label_map[envB]
            mode_str = f"{s_lab}_to_{e_lab}"
            
            ipsi_data = []
            contra_data = []
            
            for sub in subjects:
                subName = f'sub-{sub:02d}'
                resDir = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'CFC_{voxRes}')
                
                # We expect seed_names = [left_S, right_S] and target_names = [left_T, right_T]
                # IPSI: Stimulus and ROI pair on same side
                # (e.g. Stimulus Left, Left_S -> Left_T)
                f_ipsi_ll = os.path.join(resDir, f'{subName}_task-mgs_dCFC_{voxRes}_{seed_names[0]}_to_{target_names[0]}_left_targets_dpli_{mode_str}.pkl')
                f_ipsi_rr = os.path.join(resDir, f'{subName}_task-mgs_dCFC_{voxRes}_{seed_names[1]}_to_{target_names[1]}_right_targets_dpli_{mode_str}.pkl')
                
                # CONTRA: Stimulus on side X, Target ROI on side X, but Seed ROI on side Y?? 
                # Or just Stimulus on opposite side of the pair.
                f_contra_ll = os.path.join(resDir, f'{subName}_task-mgs_dCFC_{voxRes}_{seed_names[0]}_to_{target_names[0]}_right_targets_dpli_{mode_str}.pkl')
                f_contra_rr = os.path.join(resDir, f'{subName}_task-mgs_dCFC_{voxRes}_{seed_names[1]}_to_{target_names[1]}_left_targets_dpli_{mode_str}.pkl')
                
                def load_pkl(f):
                    if os.path.exists(f):
                        with open(f, 'rb') as fp: return pickle.load(fp)
                    return None
                
                d_i1 = load_pkl(f_ipsi_ll); d_i2 = load_pkl(f_ipsi_rr)
                if d_i1 is not None and d_i2 is not None:
                    ipsi_data.append(np.mean([d_i1, d_i2], axis=0))
                
                d_c1 = load_pkl(f_contra_ll); d_c2 = load_pkl(f_contra_rr)
                if d_c1 is not None and d_c2 is not None:
                    contra_data.append(np.mean([d_c1, d_c2], axis=0))
            
            if ipsi_data:
                ave = np.nanmean(ipsi_data, axis=0)
                sem = np.nanstd(ipsi_data, axis=0) / np.sqrt(len(ipsi_data))
                ax.plot(time_v, ave, color='#ff7f0e', label='Ipsi' if (r==0 and c==0) else '')
                ax.fill_between(time_v, ave-sem, ave+sem, color='#ff7f0e', alpha=0.2)
            
            if contra_data:
                ave = np.nanmean(contra_data, axis=0)
                sem = np.nanstd(contra_data, axis=0) / np.sqrt(len(contra_data))
                ax.plot(time_v, ave, color='#1f77b4', label='Contra' if (r==0 and c==0) else '')
                ax.fill_between(time_v, ave-sem, ave+sem, color='#1f77b4', alpha=0.2)
            
            ax.axhline(0, color='black', alpha=0.3, linestyle='--')
            ax.axvline(0, color='red', alpha=0.5, label='Onset' if (r==0 and c==0) else '')
            ax.set_xlim(-0.5, 1.5)
            # ax.set_ylim(-0.06, 0.06)
            
            if r == 0: ax.set_title(f"Phase: {s_lab}", fontsize=10)
            if c == 0: ax.set_ylabel(f"ENV AM: {e_lab}", fontsize=10)

    axes[0, 0].legend(loc='upper right', fontsize=8)
    fig.suptitle(f"Directional CFC Matrix: {title_suffix} ({voxRes})", fontsize=14)
    saveDir = os.path.join(bidsRoot, 'derivatives', 'figures', 'connectivity')
    os.makedirs(saveDir, exist_ok=True)
    outPath = os.path.join(saveDir, f'group_cfc_matrix_{voxRes}_{filename_suffix}.png')
    plt.savefig(outPath, dpi=300)
    print(f"[*] Saved Hierarchy Plot: {outPath}")
    plt.close()

if __name__ == "__main__":
    subs = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    res = '8mm'
    
    # 1. Top-Down (Frontal Phase -> Visual Envelope)
    plot_cfc_hierarchy_matrix(subs, res, 
                              ['left_frontal', 'right_frontal'], 
                              ['left_visual', 'right_visual'],
                              "Top-Down (Frontal Leads Visual)", "TopDown_F_to_V")
    
    # 2. Bottom-Up (Visual Phase -> Frontal Envelope)
    plot_cfc_hierarchy_matrix(subs, res, 
                              ['left_visual', 'right_visual'], 
                              ['left_frontal', 'right_frontal'],
                              "Bottom-Up (Visual Leads Frontal)", "BottomUp_V_to_F")
    
    # 3. Inter-Frontal (Coordination)
    plot_cfc_hierarchy_matrix(subs, res, 
                              ['left_frontal', 'right_frontal'], 
                              ['right_frontal', 'left_frontal'],
                              "Inter-Frontal (Cross-Hemisphere)", "InterFrontal")

    # 4. Inter-Visual (Coordination)
    plot_cfc_hierarchy_matrix(subs, res, 
                              ['left_visual', 'right_visual'], 
                              ['right_visual', 'left_visual'],
                              "Inter-Visual (Cross-Hemisphere)", "InterVisual")
