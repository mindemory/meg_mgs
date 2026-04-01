import os, pickle, h5py, socket
import numpy as np
import matplotlib.pyplot as plt
import glob

def load_atlas_rois(bidsRoot, voxRes):
    atlasF = os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat')
    return loadmat(atlasF)

def get_bids_root():
    h = socket.gethostname()
    if h == 'zod': return '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    elif h == 'vader': return '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    return '/scratch/mdd9787/meg_prf_greene/MEG_HPC'

def plot_cfc_functional_matrix(subjects, voxRes, phaseBands, envBands):
    """
    Generate a 3x3 Grid of Ipsi vs Contra CFC-dPLI.
    """
    bidsRoot = get_bids_root()
    time_v = np.linspace(-0.8, 2.0, 337) # (Default if file missing)
    
    n_rows = len(envBands)   # Envelope AM Target
    n_cols = len(phaseBands) # Phase Seed
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for r, envB in enumerate(envBands):
        for c, phaseB in enumerate(phaseBands):
            ax = axes[r, c]
            mode_str = f"phase{phaseB[0]}-{phaseB[1]}_to_env{envB[0]}-{envB[1]}"
            
            ipsi_data = []
            contra_data = []
            
            for sub in subjects:
                subName = f'sub-{sub:02d}'
                resDir = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'CFC_{voxRes}')
                
                # IPSI Combination
                f_ipsi_ll = os.path.join(resDir, f'{subName}_task-mgs_dCFC_{voxRes}_left_frontal_left_dpli_{mode_str}.pkl')
                f_ipsi_rr = os.path.join(resDir, f'{subName}_task-mgs_dCFC_{voxRes}_right_frontal_right_dpli_{mode_str}.pkl')
                
                # CONTRA Combination
                f_contra_lr = os.path.join(resDir, f'{subName}_task-mgs_dCFC_{voxRes}_left_frontal_right_dpli_{mode_str}.pkl')
                f_contra_rl = os.path.join(resDir, f'{subName}_task-mgs_dCFC_{voxRes}_right_frontal_left_dpli_{mode_str}.pkl')
                
                def load_pkl(f):
                    if os.path.exists(f):
                        with open(f, 'rb') as fp: return pickle.load(fp)
                    return None
                
                d_ll = load_pkl(f_ipsi_ll); d_rr = load_pkl(f_ipsi_rr)
                if d_ll is not None and d_rr is not None:
                    ipsi_data.append(np.mean([d_ll, d_rr], axis=0))
                
                d_lr = load_pkl(f_contra_lr); d_rl = load_pkl(f_contra_rl)
                if d_lr is not None and d_rl is not None:
                    contra_data.append(np.mean([d_lr, d_rl], axis=0))
            
            if ipsi_data:
                ipsi_ave = np.nanmean(ipsi_data, axis=0)
                ipsi_sem = np.nanstd(ipsi_data, axis=0) / np.sqrt(len(ipsi_data))
                ax.plot(time_v, ipsi_ave, color='#ff7f0e', label='Ipsi' if (r==0 and c==0) else '')
                ax.fill_between(time_v, ipsi_ave-ipsi_sem, ipsi_ave+ipsi_sem, color='#ff7f0e', alpha=0.2)
            
            if contra_data:
                contra_ave = np.nanmean(contra_data, axis=0)
                contra_sem = np.nanstd(contra_data, axis=0) / np.sqrt(len(contra_data))
                ax.plot(time_v, contra_ave, color='#1f77b4', label='Contra' if (r==0 and c==0) else '')
                ax.fill_between(time_v, contra_ave-contra_sem, contra_ave+contra_sem, color='#1f77b4', alpha=0.2)
            
            ax.axhline(0, color='black', alpha=0.3, linestyle='--')
            ax.axvline(0, color='red', alpha=0.5, label='Onset' if (r==0 and c==0) else '')
            ax.set_ylim(-0.06, 0.06)
            
            if r == 0: ax.set_title(f"Phase: {phaseB[0]}-{phaseB[1]}Hz", fontsize=10)
            if c == 0: ax.set_ylabel(f"ENV AM: {envB[0]}-{envB[1]}Hz", fontsize=10)

    axes[0, 0].legend(loc='upper right', fontsize=8)
    fig.suptitle(f"Directional CFC CFC Hierarchy Matrix: Frontal Phase leads Visual AM ({voxRes})", fontsize=14)
    saveDir = os.path.join(bidsRoot, 'derivatives', 'figures', 'connectivity')
    os.makedirs(saveDir, exist_ok=True)
    plt.savefig(os.path.join(saveDir, f'group_cfc_matrix_{voxRes}.png'), dpi=300)
    plt.show()

if __name__ == "__main__":
    subs = [1, 2, 3, 4, 5, 6]
    p_bands = [(4, 8), (8, 13), (13, 30)]
    e_bands = [(4, 8), (8, 13), (13, 30)]
    plot_cfc_functional_matrix(subs, '8mm', p_bands, e_bands)
