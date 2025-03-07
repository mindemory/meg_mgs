import os, shutil, mne
import numpy as np
from mne_bids import BIDSPath, print_dir_tree, write_raw_bids
from mne_bids.stats import count_events
from deprecated.hackerFuncs import read_elp


# Define the path to the raw data
rootPath = os.path.join(os.sep, 'd', 'DATD', 'datd', 'MEG_MGS')
bidsRoot = os.path.join(os.sep, 'd', 'DATD', 'datd', 'MEG_MGS', 'MEG_BIDS')
taskName = 'mgs'

subDirs = os.listdir(rootPath)
subDirs = [subDir for subDir in subDirs if subDir.endswith('MGS')]
# print(subDirs)

sIdx = 0
subDir = subDirs[sIdx]
subPath = os.path.join(rootPath, subDir)
subjectID = subDir[:2]

# MEG data is in "squids" folder
megRoot = os.path.join(subPath, 'squids')
recFiles = os.listdir(megRoot)
megFiles = [megFile for megFile in recFiles if megFile.endswith('.sqd') and 'coreg' not in megFile and 'Marker' not in megFile and 'ref' not in megFile]

# Path to elp file
elpFile = [elpFile for elpFile in recFiles if elpFile.endswith('.elp')]
elp = read_elp(os.path.join(megRoot, elpFile[0]))
print(len(elp))


hspFile = [hspFile for hspFile in recFiles if hspFile.endswith('.hsp')]

mrkFiles = [mrkFile for mrkFile in recFiles if mrkFile.endswith('.sqd') and 'Marker' in mrkFile and 'Copy' not in mrkFile]

# Read the marker file and average them
rawMrk = np.array([])
for mrkFile in mrkFiles:
    if len(rawMrk) == 0:
        rawMrk = mne.io.kit.read_mrk(os.path.join(megRoot, mrkFile))
    else:
        rawMrk_temp = mne.io.kit.read_mrk(os.path.join(megRoot, mrkFile))
        rawMrk = (rawMrk + rawMrk_temp) / 2


nruns = len(megFiles)
print(os.path.join(megRoot, elpFile[0]))
for runIdx in range(nruns):
    raw = mne.io.read_raw_kit(os.path.join(megRoot, megFiles[runIdx]), 
                            mrk=rawMrk,
                            elp=elp,
                            hsp=os.path.join(megRoot, hspFile[0]),
                            preload=True,
                            allow_unknown_format=True)
    raw.info['line_freq'] = 60

    # Define the path to the BIDS directory
    bids_path = BIDSPath(subject=subjectID, session=None, task=taskName, run=runIdx + 1, root=bidsRoot)

    # Write the raw data to BIDS
    write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True, format='FIF')


# Eyedata is in eyedata folder
eyeRoot = os.path.join(subPath, 'eyedata')
eyeFiles = os.listdir(eyeRoot)
eyeFiles = [eyeFile for eyeFile in eyeFiles if eyeFile.endswith('MGS.edf')]


nruns