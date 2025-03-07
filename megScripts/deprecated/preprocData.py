import os, mne
import numpy as np
import matplotlib.pyplot as plt
from preprocFuncs import importData
from deprecated.flagger import badChanFixer, icaFixer

# Define the path to the raw data
bidsRoot = os.path.join(os.sep, 'd', 'DATD', 'datd', 'MEG_MGS', 'MEG_BIDS')
derivRoot = os.path.join(os.sep, 'd', 'DATD', 'datd', 'MEG_MGS', 'MEG_BIDS', 'derivatives')
taskName = 'mgs'
subDirs = os.listdir(bidsRoot)
subDirs = [subDir for subDir in subDirs if subDir.startswith('sub-')]

sidx = 12
nRuns = 11
fName_root = f'sub-{subDirs[sidx-1][-2:]}_task-{taskName}_'

bids_path = os.path.join(derivRoot, subDirs[sidx-1], 'meg')
if not os.path.exists(bids_path):
    os.makedirs(bids_path)

epocLong_fName = os.path.join(bids_path, f'{fName_root}long-epo.fif')
epocShort_fName = os.path.join(bids_path, f'{fName_root}short-epo.fif')
epocStimlocked_fName = os.path.join(bids_path, f'{fName_root}stimlocked-epo.fif')
epocResplocked_fName = os.path.join(bids_path, f'{fName_root}resplocked-epo.fif')

if os.path.exists(epocLong_fName) and os.path.exists(epocShort_fName) and os.path.exists(epocStimlocked_fName) and os.path.exists(epocResplocked_fName):
    print('Epochs already exist. Skipping...')
    epochsLong = mne.read_epochs(epocLong_fName)
    epochsShort = mne.read_epochs(epocShort_fName)
    epochsStimlocked = mne.read_epochs(epocStimlocked_fName)
    epochsResplocked = mne.read_epochs(epocResplocked_fName)
else:
    for run in range(1, nRuns+1):
        print(f'Processing subject {sidx} run {run}...')
        # Load the data
        raw, events, metadata = importData(bidsRoot, subDirs[sidx-1], run)

        # Fix bad channels
        raw = badChanFixer(raw, sidx, run)

        # Perform ICA
        raw_ica = raw.copy()
        raw_ica.load_data().filter(l_freq=1, h_freq=55, fir_design='firwin', verbose=False)
        ica = mne.preprocessing.ICA(n_components=0.95, 
                                    random_state=42,
                                    max_iter=800, 
                                    verbose=False).fit(raw_ica, 
                                                        reject=dict(mag=4e-12),
                                                        verbose=False)
        
        # Remove bad ica components
        raw_clean = icaFixer(raw, ica, sidx, run)

        # Epoch the data
        # Long trials
        longIdx = metadata.query('islong == True').index
        longEvents = events[longIdx]
        longMetadata = metadata.loc[longIdx]
        epochsLongTemp = mne.Epochs(
                raw_clean,
                longEvents,
                event_id=2,
                tmin=-1.5,
                tmax=4,
                # baseline=(-1, 0),
                baseline=None,
                preload=True,
                detrend=1,
                reject=dict(mag=4e-12),
                metadata=longMetadata,
                verbose=False,
            )
        # Short trials
        shortIdx = metadata.query('islong == False').index
        shortEvents = events[shortIdx]
        shortMetadata = metadata.loc[shortIdx]
        epochsShortTemp = mne.Epochs(
                raw_clean,
                shortEvents,
                event_id=2,
                tmin=-1.5,
                tmax=2,
                # baseline=(-1, 0),
                baseline=None,
                preload=True,
                detrend=1,
                reject=dict(mag=4e-12),
                metadata=shortMetadata,
                verbose=False,
            )
        # Stimlocked trials
        epochsStimlockedTemp = mne.Epochs(
                raw_clean,
                events,
                event_id=2,
                tmin=-1.5,
                tmax=2,
                # baseline=(-1, 0),
                baseline=None,
                preload=True,
                detrend=1,
                reject=dict(mag=4e-12),
                metadata=metadata,
                verbose=False,
            )
        # Resplocked trials
        epochsResplockedTemp = mne.Epochs(
                raw_clean,
                events,
                event_id=4,
                tmin=-2,
                tmax=3,
                # baseline=(2, 3),
                baseline=None,
                preload=True,
                detrend=1,
                reject=dict(mag=4e-12),
                metadata=metadata,
                verbose=False,
            )

        # Generate epochs
        if run == 1:
            epochsLong = epochsLongTemp
            epochsShort = epochsShortTemp
            epochsStimlocked = epochsStimlockedTemp
            epochsResplocked = epochsResplockedTemp
        else:
            epochsLong = mne.concatenate_epochs([epochsLong, epochsLongTemp])
            epochsShort = mne.concatenate_epochs([epochsShort, epochsShortTemp])
            epochsStimlocked = mne.concatenate_epochs([epochsStimlocked, epochsStimlockedTemp])
            epochsResplocked = mne.concatenate_epochs([epochsResplocked, epochsResplockedTemp])

    # Save the data
    epochsLong.save(epocLong_fName)
    epochsShort.save(epocShort_fName)
    epochsStimlocked.save(epocStimlocked_fName)
    epochsResplocked.save(epocResplocked_fName)



# print()
# # Plot ERP for metadata tarlocCode = (1, 2, 3, 9, 10)
# # Define the conditions
# conditions = {
#     '1': 'tarlocCode == 1',
#     '2': 'tarlocCode == 2',
#     '3': 'tarlocCode == 3',
#     '9': 'tarlocCode == 9',
#     '10': 'tarlocCode == 10'
# }

# # Plot the ERPs
# fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# evokedLong = epochsLong.average()
# evokedLong.plot(axes=ax, show=False)
# # for key, value in conditions.items():
# #     evokedLong = epochsLong[value].average()
# #     evokedLong.plot(axes=ax, show=False, label=key)

# ax.set_title('ERPs for different target locations')
# ax.legend()
# plt.show()

