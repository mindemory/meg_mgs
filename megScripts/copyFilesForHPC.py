import os
from shutil import copyfile

rootDir = '/d/DATD/datd/MEG_MGS'
bidsRoot = os.path.join(rootDir, 'MEG_BIDS', 'derivatives')
hpcRoot = os.path.join(rootDir, 'MEG_HPC', 'derivatives')
if not os.path.exists(hpcRoot):
    os.makedirs(hpcRoot)

# Relevant subjects
subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 
           18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32]
for subjID in subList:
    subjRoot = 'sub-%02d' % subjID
    print('Copying files for ' + subjRoot)
    # Initialize directories
    eyetrackHPCdir = os.path.join(hpcRoot, subjRoot, 'eyetracking')
    megHPCdir = os.path.join(hpcRoot, subjRoot, 'meg')
    if not os.path.exists(eyetrackHPCdir):
        os.makedirs(eyetrackHPCdir)
    if not os.path.exists(megHPCdir):
        os.makedirs(megHPCdir)

    # Copy eyetracking file
    # iiOrigPath = os.path.join(bidsRoot, subjRoot, 'eyetracking', subjRoot + '_task-mgs-iisess.mat')
    # iiHPCpath = os.path.join(eyetrackHPCdir, subjRoot + '_task-mgs-iisess.mat')
    # copyfile(iiOrigPath, iiHPCpath)

    # # Copy Epoched Data
    # epocOrigPath = os.path.join(bidsRoot, subjRoot, 'meg', subjRoot + '_task-mgs_stimlocked.mat')
    # epocHPCpath = os.path.join(megHPCdir, subjRoot + '_task-mgs_stimlocked.mat')
    # copyfile(epocOrigPath, epocHPCpath)

    # # Copy TFR Data
    # tfrOrigPath = os.path.join(bidsRoot, subjRoot, 'meg', subjRoot + '_task-mgs_TFR.mat')
    # tfrHPCpath = os.path.join(megHPCdir, subjRoot + '_task-mgs_TFR.mat')
    # copyfile(tfrOrigPath, tfrHPCpath)

    # Copy Phase Data
    phaseOrigPath = os.path.join(bidsRoot, subjRoot, 'meg', subjRoot + '_task-mgs_TFR_phase.mat')
    phaseHPCpath = os.path.join(megHPCdir, subjRoot + '_task-mgs_TFR_phase.mat')
    copyfile(phaseOrigPath, phaseHPCpath)