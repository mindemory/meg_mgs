import os, pickle, h5py, socket
import numpy as np
from scipy.io import loadmat
from decodingFuncs import runMultiClassTemporalOnly
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from shutil import copyfile
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
import gc

def run_subject(subjID, bidsRoot, taskName, classifName, classifType, freqband, powOphase):
    subName = 'sub-%02d' % subjID
    print('Running classification for ' + subName)
    # derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName, 'meg')
    derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName)
    eyeRoot = os.path.join(derivativesRoot, 'eyetracking')
    megRoot = os.path.join(derivativesRoot, 'meg')
    fNameRoot = subName + '_task-' + taskName
    iisess_fpath = os.path.join(eyeRoot, fNameRoot + '-iisess.mat')
    stimLocked_fpath = os.path.join(megRoot, fNameRoot + '_stimlocked_lineremoved.mat')
    stimRoot = os.path.join(bidsRoot, subName, 'stimfiles')
    fNameRoot = subName + '_task-' + taskName
    if powOphase == 'power':
        TFR_fpath = os.path.join(megRoot, fNameRoot + '_TFR_evoked_lineremoved.mat')
    else:
        TFR_fpath = os.path.join(megRoot, fNameRoot + '_TFR_phase.mat')
    classifHolderRoot = os.path.join(megRoot, 'decoding_trialwise')
    if not os.path.exists(classifHolderRoot):
        os.mkdir(classifHolderRoot)


    classifPath = os.path.join(classifHolderRoot, classifName)

    if os.path.exists(classifPath):
        # Load the classifier
        print('Loading existing classifier performance')
        with open(classifPath, 'rb') as f:
            classifData = pickle.load(f)
        trlWiseDecoding = classifData['trlWiseDecoding']
        trlWiseDecodingChance = classifData['trlWiseDecodingChance']
        performance = classifData['performance']
        yLabels = classifData['yLabels']
        times_crop = classifData['times_crop']
        
    else:
        print('Running classifier')
        ii_sess = loadmat(iisess_fpath)
        epocStimLocked = loadmat(stimLocked_fpath)
        # Load stimlocked data and TFR data
        # if subjID == 12 or subjID == 17 : # For subjects that were saved using -v7.3
        if socket.gethostname() == 'zod':
            if powOphase == 'power':
                TFRtempPath = os.path.join('/Users/mrugank/Desktop', fNameRoot + '_TFR_lineremoved.mat')
            else:
                TFRtempPath = os.path.join('/Users/mrugank/Desktop', fNameRoot + '_TFR_phase.mat')
            copyfile(TFR_fpath, TFRtempPath)
            # Load the TFR data using h5py
            TFRmat = h5py.File(TFRtempPath, 'r')
            # Delete the temporary file
            os.remove(TFRtempPath)
        else:
            TFRmat = h5py.File(TFR_fpath, 'r')
        v73 = True
        # else:
        #     TFRmat = loadmat(TFR_fpath)
        #     v73 = False
        classifData = {}
        if powOphase == 'power':
            baseCorr = True
        else:
            baseCorr = False
        # aucMatrix, times_crop = runMultiClassClassification(TFRmat, v73, powOphase, classCats=classifType, freqband=freqband, basecorr=baseCorr)
        trlWiseDecoding, trlWiseDecodingChance, performance, yLabels, times_crop = runMultiClassTemporalOnly(subjID, 
                                                                                            ii_sess, 
                                                                                            epocStimLocked, 
                                                                                            TFRmat, 
                                                                                            v73, 
                                                                                            powOphase, 
                                                                                            classCats=classifType, 
                                                                                            freqband=freqband, 
                                                                                            basecorr=baseCorr)
        classifData['trlWiseDecoding'] = trlWiseDecoding
        classifData['trlWiseDecodingChance'] = trlWiseDecodingChance
        classifData['performance'] = performance
        classifData['yLabels'] = yLabels
        classifData['times_crop'] = times_crop
        # Save the classifier
        with open(classifPath, 'wb') as f:
            pickle.dump(classifData, f)
        del TFRmat, classifData
        gc.collect()
        print()
    # return subjID, aucMatrix, times_crop
    return subjID, trlWiseDecoding, trlWiseDecodingChance, performance, yLabels, times_crop


def main():
    # subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 
    #            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 
               18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32]
    # subList = [1]
    # subList = [1]
    # subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17]
    for freqband in ['theta', 'alpha', 'beta', 'broadband']:
    # freqband = 'beta' # valid options: alpha, beta, broadband
        powOphase = 'power' # valid options: power, phase
        classifType = 'indivTargets' # valid options: hemifield, quadrant, locGroups, indivTargets
        if powOphase == 'power':
            classifName = 'Temporaclassif_' + classifType + '_' + freqband + '_loo.pkl'
        else:
            classifName = 'Temporalclassif_' + classifType + '_' + freqband + '_phase.pkl'
        if freqband == 'broadband':
            ntimePts = 90
        else:
            ntimePts = 125

        trlDecodMat = np.empty((len(subList), 10000, ntimePts))
        trlDecodChanceMat = np.empty((len(subList), 10000, ntimePts))
        perfMat = np.empty((len(subList), 10000))
        yLabelMat = np.empty((len(subList), 10000))

        if socket.gethostname() == 'zod':
            bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
        else:
            bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
        taskName = 'mgs'
        figDir = os.path.join(bidsRoot, 'derivatives', 'group_plots')
        if not os.path.exists(figDir):
            os.mkdir(figDir)

        
        # Run without parallelization
        for sIdx, subjID in enumerate(subList):
            res = run_subject(subjID, bidsRoot, taskName, classifName, classifType, freqband, powOphase)
            trlWiseDecoding, trlWiseDecodingChance, perfThisSub, yLabelsThisSub, times_crop = res[1], res[2], res[3], res[4], res[5]
            # f1Mat[sIdx, :, :], f1ChanceMat[sIdx, :, :], trlWiseDecoding, perfThisSub, yLabelsThisSub, times_crop = res[1], res[2], res[3], res[4], res[5], res[6]
            ntrlsThisSub = trlWiseDecoding.shape[0]
            trlDecodMat[sIdx, :ntrlsThisSub, :] = trlWiseDecoding
            trlDecodChanceMat[sIdx, :ntrlsThisSub, :] = trlWiseDecodingChance
            perfMat[sIdx, :ntrlsThisSub] = perfThisSub
            yLabelMat[sIdx, :ntrlsThisSub] = yLabelsThisSub

if __name__ == '__main__':
    main()