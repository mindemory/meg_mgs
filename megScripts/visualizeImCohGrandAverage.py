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

def main():
    bidsRoot = get_bids_root()
    voxRes = '8mm'
    freqBand = 'beta'
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    outDir = os.path.join(bidsRoot, 'derivatives', 'figures', 'connectivity_grand_average')
    os.makedirs(outDir, exist_ok=True)

    # 1. Load ROIs
    atlas = loadmat(os.path.join(bidsRoot, 'derivatives', 'atlas', f'rois_{voxRes}.mat'))
    lfro_pts = np.where(atlas['left_frontal_points'].flatten() == 1)[0]
    rfro_pts = np.where(atlas['right_frontal_points'].flatten() == 1)[0]
    
    # Structure: grand_acc[subj][pair_idx][target_loc] = time_series (1D)
    # pair_idx: 0: LV->LF, 1: LV->RF, 2: RV->LF, 3: RV->RF
    grand_acc = {sub: {p: {'left': None, 'right': None} for p in range(4)} for sub in subjects}
    
    time_v = None
    valid_subs = 0

    print(f"[*] Aggregating ImCoh ({freqBand.upper()}) for {len(subjects)} subjects...")
    
    baseline_start, baseline_end = -0.6, 0.0
    
    for sub in subjects:
        subName = f'sub-{sub:02d}'
        conn_dir = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', f'connectivity_{voxRes}')
        
        found_sub_data = False
        for s_roi in ['left_visual', 'right_visual']:
            for t_loc in ['left', 'right']:
                f_name = f'{subName}_task-mgs_seededConnectivity_{voxRes}_{s_roi}_{t_loc}_imcoh_{freqBand}.pkl'
                f_path = os.path.join(conn_dir, f_name)
                
                if os.path.exists(f_path):
                    with open(f_path, 'rb') as f: matrix = pickle.load(f) # (sources, times)
                    if time_v is None: time_v = np.linspace(-0.5, 1.7, matrix.shape[1])
                    
                    # 1. Spatial Average over Frontal ROIs
                    series_lf = matrix[lfro_pts, :].mean(axis=0)
                    series_rf = matrix[rfro_pts, :].mean(axis=0)
                    
                    # 2. Baseline Correction: (Value / BaselineMean) - 1
                    b_mask = (time_v >= baseline_start) & (time_v <= baseline_end)
                    
                    b_mean_lf = np.nanmean(series_lf[b_mask])
                    corr_lf = (series_lf / (b_mean_lf + 1e-10)) - 1
                    
                    b_mean_rf = np.nanmean(series_rf[b_mask])
                    corr_rf = (series_rf / (b_mean_rf + 1e-10)) - 1
                    
                    p_base = 0 if s_roi == 'left_visual' else 2
                    grand_acc[sub][p_base][t_loc] = corr_lf
                    grand_acc[sub][p_base + 1][t_loc] = corr_rf
                    found_sub_data = True
                    
        if found_sub_data: valid_subs += 1

    if valid_subs == 0:
        print("[!] No data found. Exiting.")
        return

    # 2. Grand Average Logic
    results = {p: {'left': [], 'right': []} for p in range(6)}
    for sub in subjects:
        for p in range(4):
            for t_loc in ['left', 'right']:
                val = grand_acc[sub][p][t_loc]
                if val is not None: results[p][t_loc].append(val)
        
        # Combine Visual Seeds
        for t_loc in ['left', 'right']:
            l_seed = grand_acc[sub][0][t_loc]
            r_seed = grand_acc[sub][2][t_loc]
            if l_seed is not None and r_seed is not None:
                results[4][t_loc].append((l_seed + r_seed) / 2.0)
            
            l_seed_rf = grand_acc[sub][1][t_loc]
            r_seed_rf = grand_acc[sub][3][t_loc]
            if l_seed_rf is not None and r_seed_rf is not None:
                results[5][t_loc].append((l_seed_rf + r_seed_rf) / 2.0)

    # 3. Plotting (2x3)
    titles = [
        'Left Visual \u2192 Left Frontal', 'Left Visual \u2192 Right Frontal',
        'Right Visual \u2192 Left Frontal', 'Right Visual \u2192 Right Frontal',
        'Visual \u2192 Left Frontal', 'Visual \u2192 Right Frontal'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.2)
    
    for p in range(6):
        ax = axes[p // 3, p % 3]
        for t_loc, color, label in [('left', 'blue', 'Left Targets'), ('right', 'red', 'Right Targets')]:
            if not results[p][t_loc]: continue
            data = np.stack(results[p][t_loc])
            mean = np.mean(data, axis=0)
            sem = np.std(data, axis=0) / np.sqrt(data.shape[0])
            
            ax.plot(time_v, mean, color=color, label=label, linewidth=2)
            ax.fill_between(time_v, mean - sem, mean + sem, color=color, alpha=0.2)
        
        ax.set_title(titles[p], fontsize=14, fontweight='bold')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='Stim On')
        ax.axvline(0.2, color='orange', linestyle='--', alpha=0.5, label='Stim Off')
        ax.axvline(1.7, color='green', linestyle='--', alpha=0.5, label='Delay End')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlim([-0.5, 1.75])
        ax.set_ylim([-0.22, 0.22])
        if p >= 3: ax.set_xlabel('Time (s)')
        if p % 3 == 0: ax.set_ylabel('Relative Imaginary Coherence')
        ax.legend(loc='upper right', fontsize=8)

    fig.suptitle(f'Relative Imaginary Coherence ({freqBand.upper()}): Left vs Right Targets (n={valid_subs} subjects)', fontsize=20, y=0.98)
    
    outF = os.path.join(outDir, f'ImCoh_GrandAverage_{freqBand}_{voxRes}.png')
    plt.savefig(outF, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Grand Average figure saved: {outF}")

if __name__ == "__main__":
    main()
