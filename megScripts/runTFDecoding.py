import os, pickle, h5py, socket
import numpy as np
from scipy.io import loadmat
from decodingFuncs import runMultiClassClassification
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from shutil import copyfile
from joblib import Parallel, delayed
from multiprocessing import Pool, cpu_count
import gc

def run_subject(subjID, bidsRoot, taskName, freqband, classifType):
    subName = 'sub-%02d' % subjID
    print('Running classification for ' + subName)
    
    classifName = 'classif_' + classifType + '_crossfrequency.pkl'
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
        aucMatrix = classifData['auc_matrix']
        times_crop = classifData['times_crop']
        valFreqs = classifData['valFreqs']
        print()
    else:
        print('Running classifier')
        # Load stimlocked data and TFR data
        if subjID == 12 or subjID == 17 : # For subjects that were saved using -v7.3
            if socket.gethostname() == 'zod':
                TFRtempPath = os.path.join('/Users/mrugank/Desktop', fNameRoot + '_TFR.mat')
                copyfile(TFR_fpath, TFRtempPath)
                # Load the TFR data using h5py
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
        aucMatrix, times_crop, valFreqs = runMultiClassClassification(TFRmat, v73, classCats=classifType, freqband=freqband)
        classifData['auc_matrix'] = aucMatrix
        classifData['times_crop'] = times_crop
        classifData['valFreqs'] = valFreqs
        # Save the classifier
        with open(classifPath, 'wb') as f:
            pickle.dump(classifData, f)
        del TFRmat, classifData
        gc.collect()
        print()
    return subjID, aucMatrix, times_crop, valFreqs


def main():
    # subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 
    #            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 
               18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32]
    # subList = [32]
    # subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17]
    freqband = 'crossfrequency'
    classifType = 'quadrant' # valid options: hemifield, quadrant, locGroups
    
    # classifName = 'classif_' + classifType + '_' + freqband + '.pkl'
    if freqband == 'broadband' or freqband == 'crossfrequency':
        ntimePts = 90
    else:
        ntimePts = 125
    nFreqs = 52

    aucMat = np.empty((len(subList), ntimePts, nFreqs))

    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_HPC'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    taskName = 'mgs'
    figDir = os.path.join(bidsRoot, 'derivatives', 'group_plots')
    if not os.path.exists(figDir):
        os.mkdir(figDir)

    # Parallelize the decoding
    # results = Parallel(n_jobs=4)(delayed(run_subject)(subjID, bidsRoot, taskName, freqband, classifType) for subjID in subList)
    # # Extract the results
    # for res in results:
    #     sIdx = subList.index(res[0])
    #     # accMat[sIdx, :, :], f1Mat[sIdx, :, :], aucMat[sIdx, :, :], times_crop = res[1], res[2], res[3], res[4]
    #     aucMat[sIdx, :, :], times_crop, valFreqs  = res[1], res[2], res[3]
    # Run without parallelization
    for sIdx, subjID in enumerate(subList):
        res = run_subject(subjID, bidsRoot, taskName, freqband, classifType)
        # aucMat[sIdx, :, :], times_crop, valFreqs = res[1], res[2], res[3]
        aucMat[sIdx, :, :], times_crop, valFreqs = res[1], res[2], res[3]
    
    auc_matrix = np.nanmean(aucMat, axis=0)

    # Plot the time-frequency AUC matrix
    time_to_plot = [-0.1, 0, 0.2, 0.5, 1, 1.5]
    tidx_to_plot = [np.argmin(np.abs(times_crop - t)) for t in time_to_plot]
    freqs_to_plot = [5, 10, 15, 20, 25, 30, 35]
    fidx_to_plot = [np.argmin(np.abs(valFreqs - f)) for f in freqs_to_plot]

    # Identify the lower and upper bounds for the colorbar
    qtThresh = 0.0005
    colLow = np.quantile(auc_matrix, qtThresh)
    colHigh = np.quantile(auc_matrix, 1 - qtThresh)

    fig = plt.figure(figsize=(15, 10))
    im = plt.imshow(auc_matrix.T, aspect='auto', origin='lower', cmap='RdBu_r', vmin=colLow, vmax=colHigh)
    plt.title('ROC-AUC')
    plt.xlabel('Frequency')
    plt.ylabel('Time')
    # plt.xticks(tidx_to_plot)
    plt.xticks(tidx_to_plot, time_to_plot)
    # plt.yticks(fidx_to_plot)
    # plt.yticks(fidx_to_plot, freqs_to_plot)
    plt.colorbar(im)
    plt.show()
    # plt.axvline(tidx_to_plot[1], color='k', linestyle='--')

    # Plot the accuracy matrix
    # if freqband == 'broadband':
    #     time_to_plot = [-0.1, 0, 0.2, 0.5, 1, 1.5]
    # else:
    #     time_to_plot = [-0.5, 0, 0.2, 0.5, 1, 1.5, 2]
    # tidx_to_plot = [np.argmin(np.abs(times_crop - t)) for t in time_to_plot]

    # # if classifType == 'hemifield':
    # #     xLow, yLow = 0.45, 0.55
    # # elif classifType == 'quadrant':
    # #     xLow, yLow = 0.2, 0.3
    # # elif classifType == 'locGroups':
    # #     xLow, yLow = 0.144, 0.19
    # # Identify the lower and upper bounds for the colorbar
    # qtThresh = 0.0005
    # xLowAcc = np.quantile(accuracy_matrix, qtThresh)
    # yLowAcc = np.quantile(accuracy_matrix, 1 - qtThresh)
    # xLowF1 = np.quantile(f1score_matrix, qtThresh)
    # yLowF1 = np.quantile(f1score_matrix, 1 - qtThresh)
    # xLowAuc = np.quantile(auc_matrix, qtThresh)
    # yLowAuc = np.quantile(auc_matrix, 1 - qtThresh)
    # # Plot the accuracy matrix
    # fig = plt.figure(figsize=(15, 10))
    
    # gs = GridSpec(3, 4, figure=fig)  
    # # Accuracy matrix plot (0, 0)
    # ax1 = fig.add_subplot(gs[:2, :2])  
    # im = ax1.imshow(auc_matrix, aspect='auto', origin='lower', cmap='RdBu_r', vmin=xLowAuc, vmax=yLowAuc)
    # ax1.set_title('ROC-AUC')
    # ax1.set_xlabel('Test Time')
    # ax1.set_ylabel('Train Time')
    # ax1.set_xticks(tidx_to_plot)
    # ax1.set_xticklabels(time_to_plot)
    # ax1.set_yticks(tidx_to_plot)
    # ax1.set_yticklabels(time_to_plot)
    # ax1.axvline(tidx_to_plot[1], color='k', linestyle='--')
    # ax1.axhline(tidx_to_plot[1], color='k', linestyle='--')
    # fig.colorbar(im, ax=ax1)

    # # F1score matrix plot (1, 0)
    # ax2 = fig.add_subplot(gs[:2, 2:])
    # im = ax2.imshow(f1score_matrix, aspect='auto', origin='lower', cmap='RdBu_r', vmin=xLowF1, vmax=yLowF1)
    # ax2.set_title('F1 Score Matrix')
    # ax2.set_xlabel('Test Time')
    # ax2.set_ylabel('Train Time')
    # ax2.set_xticks(tidx_to_plot)
    # ax2.set_xticklabels(time_to_plot)
    # ax2.set_yticks(tidx_to_plot)
    # ax2.set_yticklabels(time_to_plot)
    # ax2.axvline(tidx_to_plot[1], color='k', linestyle='--')
    # ax2.axhline(tidx_to_plot[1], color='k', linestyle='--')
    # fig.colorbar(im, ax=ax2)

    # # Plot mean accuracy along diagonal (0, 1)
    # ax3 = fig.add_subplot(gs[2, :2])
    # ax3.plot(times_crop, np.diag(auc_matrix.T))
    # ax3.set_title('ROC-AUC Diagonal')
    # ax3.set_xlabel('Time')
    # ax3.set_ylabel('ROC-AUC')
    # if classifType == 'hemifield':
    #     ax3.axhline(0.5, color='k', linestyle='--')
    # elif classifType == 'quadrant':
    #     ax3.axhline(0.25, color='k', linestyle='--')
    # elif classifType == 'locGroups':
    #     ax3.axhline(0.167, color='k', linestyle='--')
    # elif classifType == 'indivTargets':
    #     ax3.axhline(0.1, color='k', linestyle='--')
    # ax3.set_ylim([xLowAuc, yLowAuc])

    # # Plot mean f1score along diagonal (1, 1)
    # ax4 = fig.add_subplot(gs[2, 2:])
    # ax4.plot(times_crop, np.diag(f1score_matrix.T))
    # ax4.set_title('F1 Score Diagonal')
    # ax4.set_xlabel('Time')
    # ax4.set_ylabel('F1 Score')
    # if classifType == 'hemifield':
    #     ax4.axhline(0.5, color='k', linestyle='--')
    # elif classifType == 'quadrant':
    #     ax4.axhline(0.25, color='k', linestyle='--')
    # elif classifType == 'locGroups':
    #     ax4.axhline(0.167, color='k', linestyle='--')
    # elif classifType == 'indivTargets':
    #     ax4.axhline(0.1, color='k', linestyle='--')
    # ax4.set_ylim([xLowF1, yLowF1])

    
    # plt.tight_layout()
    # plt.suptitle('Decoding Performance: ' + classifType + ' ' + freqband)
    # figPath = os.path.join(figDir, 'group_' + classifType + '_' + freqband + '.png')
    # plt.savefig(figPath, dpi=300)
    # plt.show()


if __name__ == '__main__':
    main()