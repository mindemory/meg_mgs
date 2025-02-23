import os
import numpy as np
from scipy.io import loadmat
import mne
import mne.io.kit as kit
from hackerFuncs import read_elp, read_hsp, extractEvents
from flagger import badChanFixer
import pandas as pd


def importData(bidsRoot, subdir, run):
    subFiles = os.listdir(os.path.join(bidsRoot, subdir, 'meg'))
    sqdFiles = [f for f in subFiles if f.endswith('.sqd') and 'marker' not in f]
    mrkFiles = [f for f in subFiles if f.endswith('.sqd') and 'marker' in f]
    hspFile = [f for f in subFiles if f.endswith('.hsp')][0]
    elpFile = [f for f in subFiles if f.endswith('.elp')][0]

    # Read mrkfiles
    # Average across all the marker files (might not be ideal for all cases?)
    mrkData = np.zeros((len(mrkFiles), 5, 3))
    for idx, mrkFile in enumerate(mrkFiles):
        mrkPath = os.path.join(bidsRoot, subdir, 'meg', mrkFile)
        mrkData[idx, :, :] = kit.read_mrk(mrkPath)
    mrkData = np.mean(mrkData, axis=0) #* 1e+1
    
    # Read the elp and hsp files
    elpData = read_elp(os.path.join(bidsRoot, subdir, 'meg', elpFile)) #* 1e+1
    hspData = read_hsp(os.path.join(bidsRoot, subdir, 'meg', hspFile)) #* 1e+1

    # Read the sqd file
    sqdPath = os.path.join(bidsRoot, subdir, 'meg', sqdFiles[run-1])


    # Read raw data
    raw = mne.io.read_raw_kit(sqdPath, 
                            mrk=mrkData,
                            elp=elpData,
                            hsp=hspData,
                            allow_unknown_format=True)
    
    # Extract events
    events = extractEvents(raw)

    # Extract trial metadata
    stmFiles = os.listdir(os.path.join(bidsRoot, subdir, 'stimfiles'))
    stmFiles = [f for f in stmFiles if f.endswith('.mat')]
    stmFiles.sort()

    # Load the stim file
    stimPath = os.path.join(bidsRoot, subdir, 'stimfiles', stmFiles[run-1])
    stimData = loadmat(stimPath)
    stimData = stimData['stimulus']
    tarloc = stimData['tarloc'][0][0][0]
    tarlocCode = stimData['tarlocCode'][0][0][0]
    x = stimData['x'][0][0][0]
    y = stimData['y'][0][0][0]

    ## Speical cases
    # If subject 12 and run 8, first trial has some issues
    if 'sub-12' in subdir and run == 8:
        events = events[2:] # Remove the first 2 events
        tarloc = tarloc[1:]
        tarlocCode = tarlocCode[1:]
        x = x[1:]
        y = y[1:]

    # Repeat all the flags for each epoch
    tarloc = np.asarray([t for t in tarloc for _ in range(6)])
    tarlocCode = np.asarray([t for t in tarlocCode for _ in range(6)])
    x = np.asarray([t for t in x for _ in range(6)])
    y = np.asarray([t for t in y for _ in range(6)])

    # Check if trials are long are short
    delayOnset = events[events[:, 2] == 3, 0]
    respOnset = events[events[:, 2] == 4, 0]
    delayDur = respOnset - delayOnset
    longTrials = delayDur > 1200
    longTrials = np.asarray([l for l in longTrials for _ in range(6)])

    # Combine all metadata
    metadata = pd.DataFrame({
        'islong': longTrials,
        'tarloc': tarloc,
        'tarlocCode': tarlocCode,
        'x': x,
        'y': y
    })

    return raw, events, metadata