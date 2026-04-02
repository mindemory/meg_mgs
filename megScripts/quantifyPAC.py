import os, pickle, socket
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def get_bids_root():
    h = socket.gethostname()
    if 'vader' in h: return '/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else: return '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'

def calculate_cog(tfr_matrix, time_vec, freq_vec, percentile=75):
    """Calculates Time and Frequency Center of Gravity for the 'red blob'."""
    if tfr_matrix is None: return np.nan, np.nan
    
    # Focus on positive values (the 'red blob')
    pos_data = np.maximum(tfr_matrix, 0)
    thresh = np.percentile(pos_data, percentile)
    mask = pos_data >= thresh
    
    masked_data = pos_data * mask
    total_power = np.sum(masked_data)
    if total_power == 0: return np.nan, np.nan
    
    # Weighted average
    t_cog = np.sum(np.sum(masked_data, axis=0) * time_vec) / total_power
    f_cog = np.sum(np.sum(masked_data, axis=1) * freq_vec) / total_power
    
    return t_cog, f_cog

def main():
    bidsRoot = get_bids_root()
    voxRes = '8mm'
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    outDir = os.path.join(bidsRoot, 'derivatives', 'figures', 'pac_quantification')
    os.makedirs(outDir, exist_ok=True)

    rows = []
    print(f"[*] Quantifying PAC for {len(subjects)} subjects...")

    for sub in subjects:
        subName = f'sub-{sub:02d}'
        f_path = os.path.join(bidsRoot, 'derivatives', subName, 'sourceRecon', 'pac_data', f'{subName}_CrossRegional_PAC_Quantified_{voxRes}.pkl')
        if not os.path.exists(f_path): continue
        
        with open(f_path, 'rb') as f: data_all = pickle.load(f)
        results = data_all['data']
        
        for b_p in results.keys():
            for b_a in results[b_p].keys():
                for pair in results[b_p][b_a].keys():
                    for epoch in results[b_p][b_a][pair].keys():
                        res = results[b_p][b_a][pair][epoch]
                        if res is None: continue
                        
                        mi = res['MI']
                        tfr = res['TFR']
                        amp_f = res.get('amp_freqs', np.arange(21, 57, 2))
                        
                        # CoG calculation (relative time/freq)
                        t_vec = np.linspace(-0.1, 0.1, tfr.shape[1]) if epoch == 'Stimulus' else np.linspace(-0.25, 0.25, tfr.shape[1])
                        t_cog, f_cog = calculate_cog(tfr, t_vec, amp_f)
                        
                        rows.append({
                            'Subject': subName,
                            'PhaseBand': b_p,
                            'AmpBand': b_a,
                            'Pair': pair,
                            'Epoch': epoch,
                            'MI': mi,
                            'T_CoG': t_cog,
                            'F_CoG': f_cog
                        })
    
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outDir, 'PAC_Quantification_Full.csv'), index=False)
    
    # Statistical Analysis (Stimulus vs Delay)
    print("[*] Performing T-tests (Stimulus vs Delay)...")
    stats_rows = []
    for (b_p, b_a, pair), sub_df in df.groupby(['PhaseBand', 'AmpBand', 'Pair']):
        stim_data = sub_df[sub_df['Epoch'] == 'Stimulus'].sort_values('Subject')
        delay_data = sub_df[sub_df['Epoch'] == 'Delay'].sort_values('Subject')
        
        # Ensure subjects match
        common_subs = set(stim_data['Subject']) & set(delay_data['Subject'])
        s_vals = stim_data[stim_data['Subject'].isin(common_subs)]['MI'].values
        d_vals = delay_data[delay_data['Subject'].isin(common_subs)]['MI'].values
        
        if len(s_vals) > 5:
            t_stat, p_val = stats.ttest_rel(d_vals, s_vals) # Delay - Stimulus
            stats_rows.append({
                'Phase': b_p, 'Amp': b_a, 'Pair': pair,
                'T_Stat': t_stat, 'P_Val': p_val, 'N': len(s_vals)
            })
            
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(os.path.join(outDir, 'PAC_Stats_StimVsDelay.csv'), index=False)

    # Visualization: Statistical Heatmap per Phase/Amp band
    rois = ['Ipsi-Vis', 'Contra-Vis', 'Ipsi-Fro', 'Contra-Fro']
    for (b_p, b_a), b_stats in stats_df.groupby(['Phase', 'Amp']):
        matrix = np.zeros((4, 4))
        for _, r in b_stats.iterrows():
            p_seed, a_target = r['Pair'].split('_to_')
            i, j = rois.index(p_seed), rois.index(a_target)
            # Use signed log10 p-value for visualization
            val = -np.log10(r['P_Val']) * np.sign(r['T_Stat'])
            matrix[i, j] = val
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(matrix, annot=True, xticklabels=rois, yticklabels=rois, cmap='RdBu_r', center=0)
        plt.title(f"PAC Strength Change: Delay vs Stimulus\nPhase: {b_p}, Amp: {b_a}\n(Signed -log10 p-value)")
        plt.xlabel("Amplitude Target")
        plt.ylabel("Phase Seed")
        plt.savefig(os.path.join(outDir, f'StatsHeatmap_{b_p}_{b_a}.png'), dpi=300)
        plt.close()

    print(f"[*] PAC Quantification results saved to {outDir}")

if __name__ == "__main__":
    main()
