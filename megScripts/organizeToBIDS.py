import os, shutil, mne
import numpy as np
# from mne_bids import BIDSPath, print_dir_tree, write_raw_bids
# from mne_bids.stats import count_events
from hackerFuncs import read_elp


# Define the path to the raw data
rootPath = os.path.join(os.sep, 'd', 'DATD', 'datd', 'MEG_MGS')
bidsRoot = os.path.join(os.sep, 'd', 'DATD', 'datd', 'MEG_MGS', 'MEG_BIDS')
taskName = 'mgs'

subIdx = 32
nRuns = 8
fName_prefix = f'sub-{subIdx:02d}_task-{taskName}_run-'
fName_meg_suffix = '_meg.sqd'
fName_stim_suffix = '_stimulus.mat'
fName_eyetracking_suffix = '_eyetracking.edf'

eyePrefix = f'{subIdx:02d}_' 
eyeSuffix = 'MGS.edf'
megPrefix = 'MEG160ref_MGS_'
megSuffix = '.sqd'
stimPrefix = '32MGS_'
stimSuffix = '.mat'

subRootPath = os.path.join(rootPath, f'{subIdx:02d}MGS')
subBidsRootPath = os.path.join(bidsRoot, f'sub-{subIdx:02d}')

# Make directories anat, eyetracking, meg, and stimfiles inside subBidsRootPath
subBidsDirs = ['anat', 'eyetracking', 'meg', 'stimfiles']
subRootDirs = ['eyedata', 'squids', 'stimfiles']
for subBidsDir in subBidsDirs:
    subBidsDirPath = os.path.join(subBidsRootPath, subBidsDir)
    if not os.path.exists(subBidsDirPath):
        os.makedirs(subBidsDirPath)

# Copy eyetracking files
eyerawPath = os.path.join(subRootPath, 'eyedata')
megrawPath = os.path.join(subRootPath, 'squids')
stimrawPath = os.path.join(subRootPath, 'stimfiles')
for run in range(1, nRuns + 1):
    # Copy eye tracking files
    eyerawFile = f'{eyePrefix}{run:02d}{eyeSuffix}'
    eyerawFilePath = os.path.join(eyerawPath, eyerawFile)
    eyetrackingBidsPath = os.path.join(subBidsRootPath, 'eyetracking', f'{fName_prefix}{run:02d}{fName_eyetracking_suffix}')
    shutil.copy(eyerawFilePath, eyetrackingBidsPath)

    # Copy meg files
    if run != 2:
        megrawFile = f'{megPrefix}{run:02d}{megSuffix}'
        # megrawFile = f'{megPrefix}{run}{megSuffix}'
        megrawFilePath = os.path.join(megrawPath, megrawFile)
        megBidsPath = os.path.join(subBidsRootPath, 'meg', f'{fName_prefix}{run:02d}{fName_meg_suffix}')
        shutil.copy(megrawFilePath, megBidsPath)

    # Copy stim files
    stimrawFile = f'{stimPrefix}{run:02d}{stimSuffix}'
    stimrawFilePath = os.path.join(stimrawPath, stimrawFile)
    stimBidsPath = os.path.join(subBidsRootPath, 'stimfiles', f'{fName_prefix}{run:02d}{fName_stim_suffix}')
    shutil.copy(stimrawFilePath, stimBidsPath)