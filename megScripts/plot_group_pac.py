import os, sys, glob, pickle, socket
import numpy as np
import matplotlib.pyplot as plt

FREQUENCY_BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (18.0, 25.0)
}
COL_TITLES = ['Stimulus — Ipsi', 'Stimulus — Contra', 'Delay — Ipsi', 'Delay — Contra']

def get_bids_root():
    h = socket.gethostname()
    if h == 'zod':
        return '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    elif h == 'vader':
        return '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        return '/scratch/mdd9787/meg_prf_greene/MEG_HPC'

def plot_group_region(region_name, group_data, freqs, figures_dir, n_subjects):
    """Plot group-averaged PAC figures (TFR Comodulograms & Gamma Traces)
    with extended [-0.4, 0.4] time limits."""
    
    bands_list = list(FREQUENCY_BANDS.items())
    gamma_mask = (freqs >= 30) & (freqs <= 50)
    
    # ── Figure A: TFR Comodulograms  (3 rows × 4 cols) ───────────────────────
    fig_a, axes_a = plt.subplots(3, 4, figsize=(22, 13))
    fig_a.suptitle(
        f'{region_name} ROI — Group Average Trough-Locked TFR (N={n_subjects})\n'
        f'Rows: Theta / Alpha / Beta   |   '
        f'Cols: Stim-Ipsi | Stim-Contra | Delay-Ipsi | Delay-Contra',
        fontsize=13
    )
    for ci, ct in enumerate(COL_TITLES):
        axes_a[0, ci].set_title(ct, fontsize=10, fontweight='bold')

    im_a = None
    for row_idx, (band_name, _) in enumerate(bands_list):
        _, _, col_data = group_data[band_name]
        for ci, mat in enumerate(col_data):
            ax = axes_a[row_idx, ci]
            if mat is None:
                ax.text(0.5, 0.5, 'Insufficient\nData', ha='center', va='center',
                        transform=ax.transAxes, fontsize=9)
                continue
                
            im_a = ax.imshow(mat, aspect='auto', origin='lower',
                             extent=[-1.0, 1.0, freqs[0], freqs[-1]],
                             cmap='RdBu_r', interpolation='bilinear',
                             vmin=-0.05, vmax=0.05)
            # Set absolute fixed limits
            ax.set_xlim([-0.4, 0.4])
            ax.axvline(0, color='black', linestyle='--', alpha=0.8, linewidth=1.2)
            ax.set_yticks(freqs[::4])
            
            if ci == 0:
                ax.set_ylabel(f'{band_name.capitalize()}\nPower Freq (Hz)', fontsize=9)
            else:
                ax.set_yticklabels([])
                
            if row_idx < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time from Trough (s)', fontsize=8)

    if im_a is not None:
        cbar_ax = fig_a.add_axes([0.92, 0.15, 0.015, 0.70])
        fig_a.colorbar(im_a, cax=cbar_ax, label='Power Fold Change (baseline-normalized)')

    fname_a = os.path.join(figures_dir, f'group_{region_name}_TFR_Comodulograms.png')
    fig_a.subplots_adjust(left=0.07, right=0.91, top=0.88, bottom=0.07, wspace=0.07, hspace=0.25)
    fig_a.savefig(fname_a, dpi=300)
    plt.close(fig_a)
    print(f"  Saved Group Figure A: {fname_a}")

    # ── Figure B: Mean 30-50 Hz Gamma Traces  (3 rows × 4 cols) ─────────────
    fig_b, axes_b = plt.subplots(3, 4, figsize=(22, 10), sharey='row')
    fig_b.suptitle(
        f'{region_name} ROI — Group Average 30–50 Hz Power (N={n_subjects})\n'
        f'Rows: Theta / Alpha / Beta phase   |   '
        f'Cols: Stim-Ipsi | Stim-Contra | Delay-Ipsi | Delay-Contra',
        fontsize=13
    )
    for ci, ct in enumerate(COL_TITLES):
        axes_b[0, ci].set_title(ct, fontsize=10, fontweight='bold')

    for row_idx, (band_name, _) in enumerate(bands_list):
        _, _, col_data = group_data[band_name]
        for ci, mat in enumerate(col_data):
            ax = axes_b[row_idx, ci]
            if mat is None:
                ax.axis('off')
                continue

            # X-axis extent matches underlying data (-1.0 to 1.0)
            time_extent = np.linspace(-1.0, 1.0, mat.shape[1])
            trace = mat[gamma_mask, :].mean(axis=0)
            
            # Mask visible traces within bounds
            vm = (time_extent >= -0.4) & (time_extent <= 0.4)

            ax.plot(time_extent[vm], trace[vm], color='steelblue', linewidth=1.5)
            ax.fill_between(time_extent[vm], trace[vm], 0,
                            where=trace[vm] > 0, alpha=0.25, color='steelblue')
            ax.fill_between(time_extent[vm], trace[vm], 0,
                            where=trace[vm] < 0, alpha=0.25, color='crimson')
            ax.axhline(0,  color='k',     linestyle='--', alpha=0.5, linewidth=0.8)
            ax.axvline(0,  color='black', linestyle='--', alpha=0.8, linewidth=1.2)
            
            ax.set_xlim([-0.4, 0.4])

            if ci == 0:
                ax.set_ylabel(f'{band_name.capitalize()}\nMean Power', fontsize=9)
            if row_idx < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Time from Trough (s)', fontsize=8)

    fname_b = os.path.join(figures_dir, f'group_{region_name}_GammaTraces.png')
    fig_b.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.08, wspace=0.10, hspace=0.25)
    fig_b.savefig(fname_b, dpi=300)
    plt.close(fig_b)
    print(f"  Saved Group Figure B: {fname_b}")

def main(voxRes='10mm'):
    bidsRoot = get_bids_root()
    figures_dir = os.path.join(bidsRoot, 'derivatives', 'figures', 'phase_power_coupling', 'group')
    os.makedirs(figures_dir, exist_ok=True)
    
    # ── Discover available cohort files ──────────────────────────────────────
    subj_list = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    all_files = []
    for s in subj_list:
        sub_pattern = os.path.join(bidsRoot, 'derivatives', f'sub-{s:02d}', 'sourceRecon', 'pac_data', f'*_{voxRes}.pkl')
        all_files.extend(glob.glob(sub_pattern))
    
    if not all_files:
        print(f"No PAC pre-computed files found for voxRes={voxRes} using pattern:\n{search_pattern}")
        return
        
    regions = ['Visual', 'Frontal']
    
    for r in regions:
        region_files = [f for f in all_files if f'PAC_{r}_' in f]
        n_subs = len(region_files)
        if n_subs == 0:
            print(f"No files for region {r}. Skipping...")
            continue
            
        print(f"\nLoading N={n_subs} subjects for region: {r}")
        
        # Accumulator: per band -> per condition (4) -> list of 2D matrices
        accum = {b: {c: [] for c in range(4)} for b in FREQUENCY_BANDS}
        freqs = None
        
        for pkl in region_files:
            with open(pkl, 'rb') as f:
                d = pickle.load(f)
            
            if freqs is None:
                freqs = d['freqs']
                
            for band_name in d['band_data'].keys():
                _, _, col_data = d['band_data'][band_name]
                for ci, mat in enumerate(col_data):
                    if mat is not None:
                        accum[band_name][ci].append(mat)
                        
        # ── Grand Average ──────────────────────────────────────────────────
        group_data = {}
        for band_name, (f_min, f_max) in FREQUENCY_BANDS.items():
            mean_col_data = []
            for ci in range(4):
                mat_list = accum[band_name][ci]
                if not mat_list:
                    mean_col_data.append(None)
                else:
                    g_mean = np.mean(np.stack(mat_list, axis=0), axis=0)
                    mean_col_data.append(g_mean)
            group_data[band_name] = (f_min, f_max, mean_col_data)
            
        # ── Plot ──────────────────────────────────────────────────────────
        plot_group_region(r, group_data, freqs, figures_dir, n_subs)
        
    print(f"\nDone! Group-level plotting complete in {figures_dir}")

if __name__ == '__main__':
    voxRes = sys.argv[1] if len(sys.argv) > 1 else '10mm'
    main(voxRes)
