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

def run_subject(subjID, bidsRoot, taskName, classifName, classifType, freqband, powOphase):
    subName = 'sub-%02d' % subjID
    print('Running classification for ' + subName)
    derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName, 'meg')
    stimRoot = os.path.join(bidsRoot, subName, 'stimfiles')
    fNameRoot = subName + '_task-' + taskName
    if powOphase == 'power':
        TFR_fpath = os.path.join(derivativesRoot, fNameRoot + '_TFR_evoked_lineremoved.mat')
    else:
        TFR_fpath = os.path.join(derivativesRoot, fNameRoot + '_TFR_phase.mat')
    classifHolderRoot = os.path.join(derivativesRoot, 'decoding_lineremoved')
    if not os.path.exists(classifHolderRoot):
        os.mkdir(classifHolderRoot)


    classifPath = os.path.join(classifHolderRoot, classifName)

    if os.path.exists(classifPath):
        # Load the classifier
        print('Loading existing classifier performance')
        with open(classifPath, 'rb') as f:
            classifData = pickle.load(f)
        # aucMatrix = classifData['auc_matrix']
        f1Matrix = classifData['f1_matrix']
        f1ChanceMatrix = classifData['f1_chance_matrix']
        times_crop = classifData['times_crop']
    else:
        print('Running classifier')
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
        f1Matrix, f1ChanceMatrix, times_crop = runMultiClassClassification(TFRmat, v73, powOphase, classCats=classifType, freqband=freqband, basecorr=baseCorr)
        # classifData['auc_matrix'] = aucMatrix
        classifData['f1_matrix'] = f1Matrix
        classifData['f1_chance_matrix'] = f1ChanceMatrix
        classifData['times_crop'] = times_crop
        # Save the classifier
        with open(classifPath, 'wb') as f:
            pickle.dump(classifData, f)
        del TFRmat, classifData
        gc.collect()
        print()
    # return subjID, aucMatrix, times_crop
    return subjID, f1Matrix, f1ChanceMatrix, times_crop


def main():
    # subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 
    #            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 
               18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32]
    # subList = [1]
    # subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17]

    freqband = 'broadband' # valid options: alpha, beta, broadband
    powOphase = 'power' # valid options: power, phase
    classifType = 'locGroups' # valid options: hemifield, quadrant, locGroups, indivTargets
    if powOphase == 'power':
        classifName = 'classif_' + classifType + '_' + freqband + '.pkl'
    else:
        classifName = 'classif_' + classifType + '_' + freqband + '_phase.pkl'
    if freqband == 'broadband':
        ntimePts = 90
    else:
        ntimePts = 125
    # accMat = np.empty((len(subList), ntimePts, ntimePts))
    f1Mat = np.empty((len(subList), ntimePts, ntimePts))
    f1ChanceMat = np.empty((len(subList), ntimePts, ntimePts))
    # aucMat = np.empty((len(subList), ntimePts, ntimePts))

    if socket.gethostname() == 'zod':
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
    else:
        bidsRoot = '/scratch/mdd9787/meg_prf_greene/MEG_HPC'
    taskName = 'mgs'
    figDir = os.path.join(bidsRoot, 'derivatives', 'group_plots')
    if not os.path.exists(figDir):
        os.mkdir(figDir)

    # Parallelize the decoding
    # results = Parallel(n_jobs=4)(delayed(run_subject)(subjID, bidsRoot, taskName, classifName, classifType, freqband) for subjID in subList)
    # # Extract the results
    # for res in results:
    #     sIdx = subList.index(res[0])
    #     accMat[sIdx, :, :], f1Mat[sIdx, :, :], aucMat[sIdx, :, :], times_crop = res[1], res[2], res[3], res[4]
    # Run without parallelization
    for sIdx, subjID in enumerate(subList):
        res = run_subject(subjID, bidsRoot, taskName, classifName, classifType, freqband, powOphase)
        # aucMat[sIdx, :, :], times_crop = res[1], res[2]#, res[3], res[4]
        f1Mat[sIdx, :, :], f1ChanceMat[sIdx, :, :], times_crop = res[1], res[2], res[3]
    # Average the accuracy and f1score matrices
    # auc_matrix = np.nanmean(aucMat[:, :, :], axis=0)
    f1_matrix = np.nanmean(f1Mat, axis=0)
    f1_chance_matrix = np.nanmean(f1ChanceMat, axis=0)


    # Plot the accuracy matrix
    if freqband == 'broadband':
        time_to_plot = [-0.1, 0, 0.2, 0.5, 1, 1.5]
    else:
        time_to_plot = [-0.5, 0, 0.2, 0.5, 1, 1.5, 2]
    tidx_to_plot = [np.argmin(np.abs(times_crop - t)) for t in time_to_plot]
    # Identify the lower and upper bounds for the colorbar
    qtThresh = 0.0005
    # xLowAuc = np.quantile(auc_matrix, qtThresh)
    # yLowAuc = np.quantile(auc_matrix, 1 - qtThresh)
    xLowAuc = np.quantile(f1_matrix, qtThresh)
    yLowAuc = np.quantile(f1_matrix, 1 - qtThresh)
    # Plot the accuracy matrix
    fig = plt.figure(figsize=(10, 10))
    
    gs = GridSpec(3, 2, figure=fig)  
    # Accuracy matrix plot (0, 0)
    ax1 = fig.add_subplot(gs[:2, :])  
    # im = ax1.imshow(auc_matrix, aspect='auto', origin='lower', cmap='RdBu_r', vmin=xLowAuc, vmax=yLowAuc)
    im = ax1.imshow(f1_matrix, aspect='auto', origin='lower', cmap='RdBu_r', vmin=xLowAuc, vmax=yLowAuc)
    # ax1.set_title('ROC-AUC')
    ax1.set_xlabel('Test Time')
    ax1.set_ylabel('Train Time')
    ax1.set_xticks(tidx_to_plot)
    ax1.set_xticklabels(time_to_plot)
    ax1.set_yticks(tidx_to_plot)
    ax1.set_yticklabels(time_to_plot)
    ax1.axvline(tidx_to_plot[1], color='k', linestyle='--')
    ax1.axhline(tidx_to_plot[1], color='k', linestyle='--')
    cm = fig.colorbar(im, ax=ax1)
    # cm.set_label('ROC-AUC')
    cm.set_label('F1')

    # Plot mean accuracy along diagonal (0, 1)
    ax3 = fig.add_subplot(gs[2, :])
    # ax3.plot(times_crop, np.diag(auc_matrix.T))
    ax3.plot(times_crop, np.diag(f1_matrix.T))
    ax3.plot(times_crop, np.diag(f1_chance_matrix.T), linestyle='--')
    # ax3.set_title('ROC-AUC Diagonal')
    ax3.set_title('F1 Diagonal')
    ax3.set_xlabel('Time')
    # ax3.set_ylabel('ROC-AUC')
    ax3.set_ylabel('F1')
    # ax3.axhline(0.5, color='k', linestyle='--')
    ax3.set_ylim([xLowAuc, yLowAuc])
    
    plt.tight_layout()
    # plt.suptitle('Decoding Performance: ' + classifType + ' ' + freqband)
    if powOphase == 'power':
        figPath = os.path.join(figDir, 'group_' + classifType + '_' + freqband + '_lineremoved.png')
    else:
        figPath = os.path.join(figDir, 'group_' + classifType + '_' + freqband + '_phase.png')
    plt.savefig(figPath, dpi=300)
    # plt.show()


if __name__ == '__main__':
    main()