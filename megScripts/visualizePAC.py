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
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    outDir = os.path.join(bidsRoot, 'derivatives', 'figures', 'pac_grand_average')
    os.makedirs(outDir, exist_ok=True)

    rois = ['Ipsi-Vis', 'Contra-Vis', 'Ipsi-Fro', 'Contra-Fro']
    target_epochs = ['Stimulus', 'Delay']
    phase_bands = ['theta', 'alpha', 'beta']
    amp_bands = ['high_beta', 'low_gamma']
    
    # grand_acc[phase][amp][epoch][pair] = [matrices]
    grand_acc = {pb: {ab: {e: {p: [] for p in [f"{r1}_to_{r2}" for r1 in rois for r2 in rois]} for e in target_epochs} for ab in amp_bands} for pb in phase_bands}
    
    valid_subs = 0
    print(f"[*] Aggregating PAC TFRs for {len(subjects)} subjects...")
    
    for sub in subjects:
        subName = f'sub-{sub:02d}'
        f_path = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', 'pac_data', f'{subName}_CrossRegional_PAC_Quantified_{voxRes}.pkl')
        if not os.path.exists(f_path): continue
        
        with open(f_path, 'rb') as f: data_all = pickle.load(f)
        results = data_all['data'] # [pb][ab][pair][epoch]
        
        for pb in phase_bands:
            for ab in amp_bands:
                for pk in results[pb][ab].keys():
                    for epoch in target_epochs:
                        res = results[pb][ab][pk].get(epoch)
                        if res and res['TFR'] is not None:
                            grand_acc[pb][ab][epoch][pk].append(res['TFR'])
        valid_subs += 1

    if valid_subs == 0:
        print("[!] No data found. Exiting.")
        return

    # Plotting Grand Average Figures
    for pb in phase_bands:
        for ab in amp_bands:
            print(f"[*] Visualizing Grand Average: {pb.capitalize()} Phase -> {ab.capitalize()} Amplitude")
            fig, axes = plt.subplots(4, 8, figsize=(32, 18))
            plt.subplots_adjust(hspace=0.4, wspace=0.3)
            
            for i, p_seed in enumerate(rois):
                for col_idx, (epoch, a_target) in enumerate([(e, t) for e in target_epochs for t in rois]):
                    ax = axes[i, col_idx]
                    pk = f"{p_seed}_to_{a_target}"
                    matrices = grand_acc[pb][ab][epoch][pk]
                    
                    if matrices:
                        tfr_avg = np.mean(matrices, axis=0)
                        t_ext = [-0.1, 0.1] if epoch == 'Stimulus' else [-0.25, 0.25]
                        f_range = (21, 30) if ab == 'high_beta' else (31, 55)
                        
                        im = ax.imshow(tfr_avg, origin='lower', aspect='auto',
                                       extent=[t_ext[0], t_ext[1], f_range[0], f_range[1]],
                                       cmap='RdBu_r', vmin=-0.05, vmax=0.05)
                        
                        if i == 0: ax.set_title(f"{epoch}\nTo: {a_target}", fontsize=11, fontweight='bold')
                        if col_idx == 0: ax.set_ylabel(f"From: {p_seed}\nFreq (Hz)", fontsize=12, fontweight='bold')
                    else:
                        ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                    if i == 3: ax.set_xlabel('Time from Trough (s)')
            
            fig.suptitle(f"Grand Average PAC (N={valid_subs})\nPhase: {pb.capitalize()} | Amplitude: {ab.capitalize()}\nStimulus (0-0.2s) vs Delay (0.5-1.5s)", 
                         fontsize=24, fontweight='bold', y=0.97)
            
            out_name = os.path.join(outDir, f'GrandAverage_PAC_{pb}_{ab}.png')
            fig.savefig(out_name, dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(f"[*] Grand Average PAC visualizations saved to {outDir}")

if __name__ == "__main__":
    main()
