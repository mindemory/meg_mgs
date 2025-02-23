import os
import numpy as np
from scipy.io import loadmat
from decodingFuncs import runBinaryClassification, runMultiClassClassification
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import h5py
from shutil import copyfile


def main():
    subList = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 
               22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    subList = [1, 2, 3, 4, 5, 6, 7, 9, 10]
    freqband = 'broadband'
    classifType = 'quadrant' # valid options: hemifield, quadrant, locGroups
    classifName = 'classif_' + classifType + '_' + freqband + '.pkl'
    if freqband == 'broadband':
        ntimePts = 90
    else:
        ntimePts = 125
    accMat = np.empty((len(subList), ntimePts, ntimePts))
    f1Mat = np.empty((len(subList), ntimePts, ntimePts))

    for sIdx, subjID in enumerate(subList):
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
        taskName = 'mgs'
        subName = 'sub-%02d' % subjID
        print('Running classification for ' + subName)
        derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName, 'meg')
        stimRoot = os.path.join(bidsRoot, subName, 'stimfiles')
        fNameRoot = subName + '_task-' + taskName
        TFR_fpath = os.path.join(derivativesRoot, fNameRoot + '_TFR.mat')
        classifHolderRoot = os.path.join(derivativesRoot, 'decoding')
        if not os.path.exists(classifHolderRoot):
            os.mkdir(classifHolderRoot)

        
        classifPath = os.path.join(classifHolderRoot, classifName)

        if os.path.exists(classifPath):
            # Load the classifier
            print('Loading existing classifier performance')
            with open(classifPath, 'rb') as f:
                classifData = pickle.load(f)
            accuracyMatrix = classifData['accuracyMatrix']
            f1scoreMatrix = classifData['f1scoreMatrix']
            times_crop = classifData['times_crop']
        else:
            print('Running classifier')
            # Load stimlocked data and TFR data
            if subjID == 12 or subjID == 17 : # For subjects that were saved using -v7.3
                TFRtempPath = os.path.join('/Users/mrugank/Desktop', fNameRoot + '_TFR.mat')
                copyfile(TFR_fpath, TFRtempPath)
                # Load the TFR data using h5py
                TFRmat = h5py.File(TFRtempPath, 'r')
                # Delete the temporary file
                os.remove(TFRtempPath)
                v73 = True
            else:
                TFRmat = loadmat(TFR_fpath)
                v73 = False
            classifData = {}
            if classifType == 'hemifield':
                accuracyMatrix, f1scoreMatrix, times_crop =  runBinaryClassification(TFRmat, v73, freqband=freqband)
            else:
                accuracyMatrix, f1scoreMatrix, times_crop = runMultiClassClassification(TFRmat, v73, classCats=classifType, freqband=freqband)
            classifData['accuracyMatrix'] = accuracyMatrix
            classifData['f1scoreMatrix'] = f1scoreMatrix
            classifData['times_crop'] = times_crop
            # Save the classifier
            with open(classifPath, 'wb') as f:
                pickle.dump(classifData, f)
            del TFRmat, classifData
            print()
        accMat[sIdx, :, :], f1Mat[sIdx, :, :], times_crop = accuracyMatrix, f1scoreMatrix, times_crop
    accuracy_matrix = np.nanmean(accMat[:, :, :], axis=0)
    f1score_matrix = np.nanmean(f1Mat[:, :, :], axis=0)

    # Plot the accuracy matrix
    if freqband == 'broadband':
        time_to_plot = [-0.1, 0, 0.2, 0.5, 1, 1.5]
    else:
        time_to_plot = [-0.5, 0, 0.2, 0.5, 1, 1.5, 2]
    tidx_to_plot = [np.argmin(np.abs(times_crop - t)) for t in time_to_plot]

    # if classifType == 'hemifield':
    #     xLow, yLow = 0.45, 0.55
    # elif classifType == 'quadrant':
    #     xLow, yLow = 0.2, 0.3
    # elif classifType == 'locGroups':
    #     xLow, yLow = 0.144, 0.19
    # Identify the lower and upper bounds for the colorbar
    qtThresh = 0.0005
    xLowAcc = np.quantile(accuracy_matrix, qtThresh)
    yLowAcc = np.quantile(accuracy_matrix, 1 - qtThresh)
    xLowF1 = np.quantile(f1score_matrix, qtThresh)
    yLowF1 = np.quantile(f1score_matrix, 1 - qtThresh)
    # Plot the accuracy matrix
    fig = plt.figure(figsize=(6, 6))
    
    gs = GridSpec(3, 4, figure=fig)  
    # Accuracy matrix plot (0, 0)
    ax1 = fig.add_subplot(gs[:2, :2])  
    im = ax1.imshow(accuracy_matrix, aspect='auto', origin='lower', cmap='RdBu_r', vmin=xLowAcc, vmax=yLowAcc)
    ax1.set_title('Accuracy Matrix')
    ax1.set_xlabel('Test Time')
    ax1.set_ylabel('Train Time')
    ax1.set_xticks(tidx_to_plot)
    ax1.set_xticklabels(time_to_plot)
    ax1.set_yticks(tidx_to_plot)
    ax1.set_yticklabels(time_to_plot)
    ax1.axvline(tidx_to_plot[1], color='k', linestyle='--')
    ax1.axhline(tidx_to_plot[1], color='k', linestyle='--')
    fig.colorbar(im, ax=ax1)

    # F1score matrix plot (1, 0)
    ax2 = fig.add_subplot(gs[:2, 2:])
    im = ax2.imshow(f1score_matrix, aspect='auto', origin='lower', cmap='RdBu_r', vmin=xLowF1, vmax=yLowF1)
    ax2.set_title('F1 Score Matrix')
    ax2.set_xlabel('Test Time')
    ax2.set_ylabel('Train Time')
    ax2.set_xticks(tidx_to_plot)
    ax2.set_xticklabels(time_to_plot)
    ax2.set_yticks(tidx_to_plot)
    ax2.set_yticklabels(time_to_plot)
    ax2.axvline(tidx_to_plot[1], color='k', linestyle='--')
    ax2.axhline(tidx_to_plot[1], color='k', linestyle='--')
    fig.colorbar(im, ax=ax2)

    # Plot mean accuracy along diagonal (0, 1)
    ax3 = fig.add_subplot(gs[2, :2])
    ax3.plot(times_crop, np.diag(accuracy_matrix.T))
    ax3.set_title('Accuracy Diagonal')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Accuracy')
    if classifType == 'hemifield':
        ax3.axhline(0.5, color='k', linestyle='--')
    elif classifType == 'quadrant':
        ax3.axhline(0.25, color='k', linestyle='--')
    elif classifType == 'locGroups':
        ax3.axhline(0.167, color='k', linestyle='--')
    ax3.set_ylim([xLowAcc, yLowAcc])

    # Plot mean f1score along diagonal (1, 1)
    ax4 = fig.add_subplot(gs[2, 2:])
    ax4.plot(times_crop, np.diag(f1score_matrix.T))
    ax4.set_title('F1 Score Diagonal')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('F1 Score')
    if classifType == 'hemifield':
        ax4.axhline(0.5, color='k', linestyle='--')
    elif classifType == 'quadrant':
        ax4.axhline(0.25, color='k', linestyle='--')
    elif classifType == 'locGroups':
        ax4.axhline(0.167, color='k', linestyle='--')
    ax4.set_ylim([xLowF1, yLowF1])

    
    plt.tight_layout()
    plt.suptitle('Decoding Performance: ' + classifType + ' ' + freqband)
    plt.show()

if __name__ == '__main__':
    main()