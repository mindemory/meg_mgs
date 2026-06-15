import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp

def load_data():
    subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32]
    taskName, voxRes = 'mgs', '8mm'
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    
    data_store = {'imcoh': [], 'wpli': []}
    time_vector = None
    
    print("Loading data for 21 subjects...")
    for subjID in subjects:
        subjDir = os.path.join(bidsRoot, 'derivatives', f'sub-{subjID:02d}', 'sourceRecon', f'connectivity_{voxRes}')
        outputFile = os.path.join(subjDir, f'sub-{subjID:02d}_task-{taskName}_connectivity_{voxRes}.pkl')
        if not os.path.exists(outputFile): continue
        
        try:
            with open(outputFile, 'rb') as f:
                _ = pickle.load(f) # coh
                imcoh_dict = pickle.load(f)
                _ = pickle.load(f) # dummy
                _ = pickle.load(f) # dpli
                wpli_dict = pickle.load(f)
                
                for m_name, d_dict in [('imcoh', imcoh_dict), ('wpli', wpli_dict)]:
                    traces = []
                    # Looking for Contra-Vis -> Contra-Front Target
                    # matching ts_hemifield logic
                    for k in [f'left_rV2rF_{m_name}', f'right_lV2lF_{m_name}', f'left_rV2rF', f'right_lV2lF']:
                        if k in d_dict:
                            val = d_dict[k]
                            if val.ndim > 1: val = np.mean(val.reshape(val.shape[0], -1), axis=1)
                            traces.append(val)
                    if traces:
                        data_store[m_name].append(np.mean(traces, axis=0))
                        if time_vector is None:
                            time_vector = np.linspace(-1.0, 2.0, len(traces[0]))
        except EOFError: continue
            
    return data_store, time_vector

def run_permutation_vs_zero(data, n_perms=2999):
    obs_mean = np.mean(data)
    signs = np.random.choice([-1, 1], size=(n_perms, len(data)))
    null_dist = np.mean(signs * data, axis=1)
    p = (np.sum(np.abs(null_dist) >= np.abs(obs_mean)) + 1) / (n_perms + 1)
    return p

def main():
    data_store, time_vector = load_data()
    if not data_store['imcoh']:
        print("Error: No data loaded. Check keys/paths.")
        return

    print(f"Loaded {len(data_store['imcoh'])} subjects.")
    b_mask = (time_vector >= -0.6) & (time_vector <= 0.0)
    b_idxs = np.where(b_mask)[0]

    for metric in ['imcoh', 'wpli']:
        all_traces = np.array(data_store[metric])
        z_traces = np.zeros_like(all_traces)
        for i in range(len(all_traces)):
            bm, bs = np.nanmean(all_traces[i, b_idxs]), np.nanstd(all_traces[i, b_idxs])
            z_traces[i] = (all_traces[i] - bm) / (bs + 1e-10)

        results = []
        for w_size in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
            for start in np.arange(0.3, 1.7 - w_size, 0.05):
                end = start + w_size
                mask = (time_vector >= start) & (time_vector <= end)
                v = np.mean(z_traces[:, mask], axis=1)
                p = run_permutation_vs_zero(v)
                results.append({'start': round(start, 2), 'end': round(end, 2), 'size': w_size, 'p': p, 'mean_z': np.mean(v)})

        df = pd.DataFrame(results)
        df = df.sort_values('p')
        print(f"\nTop Windows for {metric.upper()} (Contra-Front Interaction):")
        print(df.head(10).to_string(index=False))

if __name__ == '__main__':
    main()
