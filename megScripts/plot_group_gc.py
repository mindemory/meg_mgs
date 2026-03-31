import os, sys, glob, pickle, socket
import numpy as np
import matplotlib.pyplot as plt

def get_bids_root():
    h = socket.gethostname()
    if h == 'zod':
        return '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    elif h == 'vader':
        return '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        return '/scratch/mdd9787/meg_prf_greene/MEG_HPC'

def plot_gc_spectrum(group_data, window_freqs, figures_dir, n_subjects, voxRes):
    """
    group_data structure:
      data[window][condition]['v2f' / 'f2v'] = mean_spectrum (shape: [len(freqs)])
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False, sharey='row')
    fig.suptitle(f'Group Averaged Granger Causality Spectrum (N={n_subjects} | {voxRes})', fontsize=15, fontweight='bold')
    
    windows = ['Stimulus', 'Delay']
    directions = [('f2v', 'Frontal → Visual'), ('v2f', 'Visual → Frontal')]
    
    # Path mappings for legend
    labels = {
        'f2v': {'if2iv': 'IpsiF → IpsiV', 'if2cv': 'IpsiF → ContraV', 'cf2iv': 'ContraF → IpsiV', 'cf2cv': 'ContraF → ContraV'},
        'v2f': {'iv2if': 'IpsiV → IpsiF', 'iv2cf': 'IpsiV → ContraF', 'cv2if': 'ContraV → IpsiF', 'cv2cf': 'ContraV → ContraF'}
    }
    
    colors = {
        'f2v': {'if2iv': 'darkorange', 'if2cv': 'gold', 'cf2iv': 'teal', 'cf2cv': 'darkcyan'},
        'v2f': {'iv2if': 'darkorange', 'iv2cf': 'gold', 'cv2if': 'teal', 'cv2cf': 'darkcyan'}
    }
    
    for row_idx, window in enumerate(windows):
        freqs = window_freqs[window]
        for col_idx, (dir_key, dir_title) in enumerate(directions):
            ax = axes[row_idx, col_idx]
            
            dir_labels = labels[dir_key]
            dir_colors = colors[dir_key]
            
            for path_key, label in dir_labels.items():
                mean_spectrum = group_data[window][dir_key][path_key]
                if mean_spectrum is None:
                    continue
                
                # Plot the line
                ax.plot(freqs, mean_spectrum, 
                        color=dir_colors[path_key], 
                        linewidth=2.0, 
                        label=label)
            
            # Formatting
            if row_idx == 0:
                ax.set_title(dir_title, fontsize=13, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{window}\nGranger Causality", fontsize=11, fontweight='bold')
            if row_idx == 1:
                ax.set_xlabel("Frequency (Hz)", fontsize=11)
                
            ax.set_xlim([freqs[0], freqs[-1]])
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc='upper right', title="Target Hemifield")
            
    # Add shaded regions for frequency bands to guide the eye
    bands = {'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (18, 25), 'Gamma': (30, 50)}
    for ax in axes.flat:
        for b_name, (fmin, fmax) in bands.items():
            ax.axvspan(fmin, fmax, alpha=0.08, color='gray', zorder=0)
            if ax == axes[0, 0]: # Annotate on first subplot
                y_pos = ax.get_ylim()[1] * 0.95
                ax.text(np.mean([fmin, fmax]), y_pos, b_name, ha='center', va='top', fontsize=8, color='gray', alpha=0.8)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fname = os.path.join(figures_dir, f'group_GC_Spectrum_{voxRes}.png')
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"  Saved Group GC Figure: {fname}")

def main(voxRes='10mm'):
    bidsRoot = get_bids_root()
    figures_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'granger_causality', 'group')
    os.makedirs(figures_dir, exist_ok=True)
    
    search_pattern = os.path.join(bidsRoot, 'derivatives', 'sub-*', 'sourceRecon', 'gc_data', f'*_GC_{voxRes}.pkl')
    all_files = glob.glob(search_pattern)
    
    if not all_files:
        print(f"No GC pre-computed files found for voxRes={voxRes} using pattern:\n{search_pattern}")
        return
        
    n_subs = len(all_files)
    print(f"\nLoading N={n_subs} subjects for group GC...")
    
    window_freqs = {}
    windows = ['Stimulus', 'Delay']
    
    accum = {
        w: {
            'f2v': {k: [] for k in ['if2iv', 'if2cv', 'cf2iv', 'cf2cv']},
            'v2f': {k: [] for k in ['iv2if', 'iv2cf', 'cv2if', 'cv2cf']}
        } for w in windows
    }
    
    for pkl in all_files:
        with open(pkl, 'rb') as f:
            d = pickle.load(f)
            
        if not window_freqs and 'window_freqs' in d:
            window_freqs = d['window_freqs']
            
        for w in windows:
            if w in d['gc_data']:
                for dir_key in ['f2v', 'v2f']:
                    for path_key in d['gc_data'][w][dir_key]:
                        val = d['gc_data'][w][dir_key][path_key]
                        if val is not None:
                            accum[w][dir_key][path_key].append(val)
                            
    # Grand average
    group_data = {
        w: {
            'f2v': {k: None for k in ['if2iv', 'if2cv', 'cf2iv', 'cf2cv']},
            'v2f': {k: None for k in ['iv2if', 'iv2cf', 'cv2if', 'cv2cf']}
        } for w in windows
    }
    
    for w in windows:
        for dir_key in ['f2v', 'v2f']:
            for path_key in accum[w][dir_key]:
                mat_list = accum[w][dir_key][path_key]
                if mat_list:
                    group_data[w][dir_key][path_key] = np.mean(np.stack(mat_list), axis=0)
            
    plot_gc_spectrum(group_data, window_freqs, figures_dir, n_subs, voxRes)
    print(f"\nDone! Group-level GC plotting complete.")

if __name__ == '__main__':
    voxRes = sys.argv[1] if len(sys.argv) > 1 else '10mm'
    main(voxRes)
