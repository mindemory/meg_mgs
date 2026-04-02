import os, pickle, socket
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_bids_root():
    h = socket.gethostname()
    if 'vader' in h: return '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else: return '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'

def main():
    bidsRoot = get_bids_root()
    voxRes = '8mm'
    subjects = [1]
    outDir = os.path.join(bidsRoot, 'derivatives', 'figures', 'pac_advanced_cross')
    os.makedirs(outDir, exist_ok=True)

    for sub in subjects:
        subName = f'sub-{sub:02d}'
        f_path = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', 'pac_data', f'{subName}_CrossRegional_PAC_{voxRes}.pkl')
        if not os.path.exists(f_path): continue
        with open(f_path, 'rb') as f: data_all = pickle.load(f)
        
        results = data_all['data']
        a_freqs = data_all['amp_freqs']
        bands = list(results.keys())
        epochs = list(results[bands[0]].keys())
        
        # We'll show Stimulus and Delay side-by-side
        target_epochs = ['Stimulus', 'Delay']
        rois = ['Ipsi-Vis', 'Contra-Vis', 'Ipsi-Fro', 'Contra-Fro']
        
        for band in bands:
            print(f"[*] Visualizing Cross-Regional PAC Matrix: {band.capitalize()}")
            
            # 4 Seeds (Rows) x 8 Columns (4 Targets for Stim, 4 Targets for Delay)
            fig, axes = plt.subplots(4, 8, figsize=(32, 18))
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            
            for i, p_seed in enumerate(rois):
                for col_idx, (epoch, a_target) in enumerate([(e, t) for e in target_epochs for t in rois]):
                    ax = axes[i, col_idx]
                    pair_key = f"{p_seed}_to_{a_target}"
                    t_matrix = results[band][epoch].get(pair_key)
                    
                    if t_matrix is not None:
                        # Use correct time extent for epoch
                        t_ext = [-0.1, 0.1] if epoch == 'Stimulus' else [-0.25, 0.25]
                        im = ax.imshow(t_matrix, origin='lower', aspect='auto', 
                                       extent=[t_ext[0], t_ext[1], a_freqs[0], a_freqs[-1]],
                                       cmap='RdBu_r', vmin=-0.1, vmax=0.1)
                        
                        if i == 0: 
                            ax.set_title(f"{epoch}\nTo: {a_target}", fontsize=11, fontweight='bold')
                        if col_idx == 0: 
                            ax.set_ylabel(f"From: {p_seed}\nFreq (Hz)", fontsize=12, fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    
                    if i == 3: ax.set_xlabel('Time from Trough (s)')
            
            fig.suptitle(f"{subName} - Cross-Regional PAC ({band.capitalize()})\nStimulus (0-0.2s) and Delay (0.5-1.5s)", 
                         fontsize=22, fontweight='bold', y=0.97)
            
            out_name = os.path.join(outDir, f'{subName}_CrossRegional_PAC_{band}.png')
            fig.savefig(out_name, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f"[*] Cross-Regional PAC visualizations saved to {outDir}")

if __name__ == "__main__":
    main()
