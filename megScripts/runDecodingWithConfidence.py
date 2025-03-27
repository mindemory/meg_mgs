import os, sys, socket, pickle, h5py
import numpy as np
from scipy.io import loadmat
from decodingFuncs import runMCCbyConfidence
from shutil import copyfile
from joblib import Parallel, delayed

def process_subject(subjID):
    freqband = 'theta'
    performanceMetric = 'ierr'
    classifType = 'locGroups' # valid options: hemifield, quadrant, locGroups
    classifName = 'classif_' + classifType + '_' + freqband + '_' + performanceMetric + '.pkl'
    if freqband == 'broadband':
        ntimePts = 90
    else:
        ntimePts = 125

    # for sIdx, subjID in enumerate(subList):
    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    taskName = 'mgs'
    subName = 'sub-%02d' % int(subjID)
    print('Running classification for ' + subName)
    derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName)
    eyeRoot = os.path.join(derivativesRoot, 'eyetracking')
    megRoot = os.path.join(derivativesRoot, 'meg')
    fNameRoot = subName + '_task-' + taskName
    iisess_fpath = os.path.join(eyeRoot, fNameRoot + '-iisess.mat')
    stimLocked_fpath = os.path.join(megRoot, fNameRoot + '_stimlocked.mat')
    TFR_fpath = os.path.join(megRoot, fNameRoot + '_TFR.mat')
    classifHolderRoot = os.path.join(megRoot, 'decodingWithConfidence')
    if not os.path.exists(classifHolderRoot):
        os.mkdir(classifHolderRoot)
    
    classifPath = os.path.join(classifHolderRoot, classifName)

    if os.path.exists(classifPath):
        # Load the classifier
        print('Loading existing classifier performance')
        with open(classifPath, 'rb') as f:
            classifData = pickle.load(f)
        auc_matrix = classifData['auc_matrix']
        accuracy_matrix = classifData['accuracy_matrix']
        f1score_matrix = classifData['f1score_matrix']
        y_guessed = classifData['y_guessed']
        y_guessed_proba = classifData['y_guessed_proba']
        performance_valid = classifData['performance_valid']
        times_crop = classifData['times_crop']
        del classifData
        print()
    else:
        print('Running classifier')
        # Load epocStimLocked and ii_sess data
        ii_sess = loadmat(iisess_fpath)
        epocStimLocked = loadmat(stimLocked_fpath)
        # Load TFR data
        if subjID == 12 or subjID == 17 : # For subjects that were saved using -v7.3
            if socket.gethostname() == 'zod':
                TFRtempPath = os.path.join('/Users/mrugank/Desktop', fNameRoot + '_TFR.mat')
                copyfile(TFR_fpath, TFRtempPath)
                # Load the TFR data using h5p`y
                TFRmat = h5py.File(TFRtempPath, 'r')
                # Delete the temporary file
                os.remove(TFRtempPath)
            else:
                TFRmat = h5py.File(TFR_fpath, 'r')
            v73 = True
        else:
            TFRmat = loadmat(TFR_fpath)
            v73 = False
        classifData = {}
        
        auc_matrix, accuracy_matrix, f1score_matrix, y_guessed, y_guessed_proba, performance_valid, times_crop = runMCCbyConfidence(subjID, 
                                                                                                                                    ii_sess, 
                                                                                                                                    epocStimLocked, 
                                                                                                                                    TFRmat, 
                                                                                                                                    v73=v73, 
                                                                                                                                    metric=performanceMetric, 
                                                                                                                                    classCats=classifType, 
                                                                                                                                    freqband=freqband)
        classifData['auc_matrix'] = auc_matrix
        classifData['accuracy_matrix'] = accuracy_matrix
        classifData['f1score_matrix'] = f1score_matrix
        classifData['y_guessed'] = y_guessed
        classifData['y_guessed_proba'] = y_guessed_proba
        classifData['performance_valid'] = performance_valid
        classifData['times_crop'] = times_crop
        # Save the classifier
        with open(classifPath, 'wb') as f:
            pickle.dump(classifData, f)
        del TFRmat, classifData
        print()
    

def main():
    # subsExcluded = 8, 16, 20, 22, 30
    # 8: Many bad trials
    # 16: Many bad trials
    # 20: Exclude cuz for many runs, gaze was off on bottom left
    # 22: Exclude; gaze data is bad in some locations
    # 30: Exclude bad gaze data
    # subjID = sys.argv[1]
    # Valid SubIDs
    subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 
               18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32]
    for subjID in subList:
        process_subject(subjID)

    # results = Parallel(n_jobs=4, verbose=10)(
    #     delayed(process_subject)(subjID) for subjID in subList
    # )
    # for result in results:
    #     print(result)
    
    
if __name__ == '__main__':
    main()