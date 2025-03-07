import os
import numpy as np
from scipy.io import loadmat
from decodingFuncs import runMultiClassClassificationByPerformance
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pickle
import h5py
from shutil import copyfile


def main():
    # subsExcluded = 8, 16, 20, 22, 30
    # 8: Many bad trials
    # 16: Many bad trials
    # 20: Exclude cuz for many runs, gaze was off on bottom left
    # 22: Exclude; gaze data is bad in some locations
    # 30: Exclude bad gaze data
    subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 
               18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32]
    # subList = [1, 2, 3, 4, 5, 6, 7, 9, 10]
    freqband = 'theta'
    performanceMetric = 'ierr'
    classifType = 'quadrant' # valid options: hemifield, quadrant, locGroups
    classifName = 'classif_' + classifType + '_' + freqband + '_' + performanceMetric + '.pkl'
    if freqband == 'broadband':
        ntimePts = 90
    else:
        ntimePts = 125
    accMat_highPerf = np.empty((len(subList), ntimePts, ntimePts))
    f1Mat_highPerf = np.empty((len(subList), ntimePts, ntimePts))
    accMat_lowPerf = np.empty((len(subList), ntimePts, ntimePts))
    f1Mat_lowPerf = np.empty((len(subList), ntimePts, ntimePts))

    for sIdx, subjID in enumerate(subList):
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS'
        taskName = 'mgs'
        subName = 'sub-%02d' % subjID
        print('Running classification for ' + subName)
        derivativesRoot = os.path.join(bidsRoot, 'derivatives', subName)
        eyeRoot = os.path.join(derivativesRoot, 'eyetracking')
        megRoot = os.path.join(derivativesRoot, 'meg')
        fNameRoot = subName + '_task-' + taskName
        iisess_fpath = os.path.join(eyeRoot, fNameRoot + '-iisess.mat')
        stimLocked_fpath = os.path.join(megRoot, fNameRoot + '_stimlocked.mat')
        TFR_fpath = os.path.join(megRoot, fNameRoot + '_TFR.mat')
        classifHolderRoot = os.path.join(megRoot, 'decodingByPerformance')
        if not os.path.exists(classifHolderRoot):
            os.mkdir(classifHolderRoot)
        
        classifPath = os.path.join(classifHolderRoot, classifName)

        if os.path.exists(classifPath):
            # Load the classifier
            print('Loading existing classifier performance')
            with open(classifPath, 'rb') as f:
                classifData = pickle.load(f)
            accuracy_highPerf = classifData['accuracyMatrix_highPerf']
            f1score_highPerf = classifData['f1scoreMatrix_highPerf']
            accuracy_lowPerf = classifData['accuracyMatrix_lowPerf']
            f1score_lowPerf = classifData['f1scoreMatrix_lowPerf']
            times_crop = classifData['times_crop']
        else:
            print('Running classifier')
            # Load epocStimLocked and ii_sess data
            ii_sess = loadmat(iisess_fpath)
            epocStimLocked = loadmat(stimLocked_fpath)
            # Load TFR data
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
                accuracy_highPerf, f1score_highPerf, accuracy_lowPerf, f1score_lowPerf, times_crop = runMultiClassClassificationByPerformance(subjID, 
                                                                                                                                                ii_sess, 
                                                                                                                                                epocStimLocked, 
                                                                                                                                                TFRmat, 
                                                                                                                                                v73, 
                                                                                                                                                metric=performanceMetric, 
                                                                                                                                                classCats=classifType, 
                                                                                                                                                freqband=freqband)
            classifData['accuracyMatrix_highPerf'] = accuracy_highPerf
            classifData['f1scoreMatrix_highPerf'] = f1score_highPerf
            classifData['accuracyMatrix_lowPerf'] = accuracy_lowPerf
            classifData['f1scoreMatrix_lowPerf'] = f1score_lowPerf
            classifData['times_crop'] = times_crop
            # Save the classifier
            with open(classifPath, 'wb') as f:
                pickle.dump(classifData, f)
            del TFRmat, classifData
            print()
        accMat_highPerf[sIdx, :, :], f1Mat_highPerf[sIdx, :, :], accMat_lowPerf[sIdx, :, :], f1Mat_lowPerf[sIdx, :, :], times_crop = accuracy_highPerf, f1score_highPerf, accuracy_lowPerf, f1score_lowPerf, times_crop
    accuracy_matrix_highPerf = np.nanmean(accMat_highPerf[:, :, :], axis=0)
    f1score_matrix_highPerf = np.nanmean(f1Mat_highPerf[:, :, :], axis=0)
    accuracy_matrix_lowPerf = np.nanmean(accMat_lowPerf[:, :, :], axis=0)
    f1score_matrix_lowPerf = np.nanmean(f1Mat_lowPerf[:, :, :], axis=0)

    if freqband == 'broadband':
        time_to_plot = [-0.1, 0, 0.2, 0.5, 1, 1.5]
    else:
        time_to_plot = [-0.5, 0, 0.2, 0.5, 1, 1.5, 2]
    tidx_to_plot = [np.argmin(np.abs(times_crop - t)) for t in time_to_plot]

    # Identify the lower and upper bounds for the colorbar
    qtThresh = 0.0005
    xLowAccHighperf, xHighAccHighpef = np.quantile(accuracy_matrix_highPerf, qtThresh), np.quantile(accuracy_matrix_highPerf, 1-qtThresh)
    xLowF1Highperf, xHighF1Highperf = np.quantile(f1score_matrix_highPerf, qtThresh), np.quantile(f1score_matrix_highPerf, 1-qtThresh)
    xLowAccLowperf, xHighAccLowperf = np.quantile(accuracy_matrix_lowPerf, qtThresh), np.quantile(accuracy_matrix_lowPerf, 1-qtThresh)
    xLowF1Lowperf, xHighF1Lowperf = np.quantile(f1score_matrix_lowPerf, qtThresh), np.quantile(f1score_matrix_lowPerf, 1-qtThresh)

    metricToPlot = 'F1'
    if metricToPlot == 'accuracy':
        xLow_highPerf, xHigh_highPerf = xLowAccHighperf, xHighAccHighpef
        xLow_lowPerf, xHigh_lowPerf = xLowAccLowperf, xHighAccLowperf
        dataMat_highPerf = accuracy_matrix_highPerf
        dataMat_lowPerf = accuracy_matrix_lowPerf
        rawMat_highPerf = accMat_highPerf
        rawMat_lowPerf = accMat_lowPerf
        cbarLabel = 'Accuracy'
        
    else:
        xLow_highPerf, xHigh_highPerf = xLowF1Highperf, xHighF1Highperf
        xLow_lowPerf, xHigh_lowPerf = xLowF1Lowperf, xHighF1Lowperf
        dataMat_highPerf = f1score_matrix_highPerf
        dataMat_lowPerf = f1score_matrix_lowPerf
        rawMat_highPerf = f1Mat_highPerf
        rawMat_lowPerf = f1Mat_lowPerf
        cbarLabel = 'F1 score'
        lowPerf_label = 'Low error'


    if performanceMetric == 'ierr' or performanceMetric == 'ferr':
        highPerf_label = 'Large error'
        lowPerf_label = 'Low error'
    elif performanceMetric == 'irt' or performanceMetric == 'frt':
        highPerf_label = 'Fast saccade'
        lowPerf_label = 'Slow saccade'
    # Compute diagonal for each subject and compute mean and stderror
    diag_highPerf = np.empty((rawMat_highPerf.shape[0], rawMat_highPerf.shape[1]))
    diag_lowPerf = np.empty((rawMat_lowPerf.shape[0], rawMat_lowPerf.shape[1]))
    for s in range(accMat_highPerf.shape[0]):
        diag_highPerf[s, :] = np.diag(rawMat_highPerf[s, :, :])
        diag_lowPerf[s, :] = np.diag(rawMat_lowPerf[s, :, :])
    meanDiag_highPerf = np.nanmean(diag_highPerf, axis=0)
    stdErrDiag_highPerf = np.nanstd(diag_highPerf, axis=0) / np.sqrt(diag_highPerf.shape[0])
    meanDiag_lowPerf = np.nanmean(diag_lowPerf, axis=0)
    stdErrDiag_lowPerf = np.nanstd(diag_lowPerf, axis=0) / np.sqrt(diag_lowPerf.shape[0])

    # Adding temporarily
    if classifType == 'quadrant':
        xLow_highPerf, xHigh_highPerf = 0.22, 0.28
        xLow_lowPerf, xHigh_lowPerf = 0.22, 0.28
    
    # Plot the performance matrices
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(4, 4, figure=fig)
    # Highperformance matrix plot (0, 0)
    ax1 = fig.add_subplot(gs[:2, :2])
    im = ax1.imshow(dataMat_highPerf, aspect='auto', origin='lower', cmap='RdBu_r', vmin=xLow_highPerf, vmax=xHigh_highPerf)
    ax1.set_title(highPerf_label + ' ' + cbarLabel + ' Matrix')
    ax1.set_xlabel('Test Time')
    ax1.set_ylabel('Train Time')
    ax1.set_xticks(tidx_to_plot)
    ax1.set_xticklabels(time_to_plot)
    ax1.set_yticks(tidx_to_plot)
    ax1.set_yticklabels(time_to_plot)
    ax1.axvline(tidx_to_plot[1], color='k', linestyle='--')
    ax1.axhline(tidx_to_plot[1], color='k', linestyle='--')
    fig.colorbar(im, ax=ax1)

    # Lowperformance matrix plot (1, 0)
    ax2 = fig.add_subplot(gs[:2, 2:])
    im = ax2.imshow(dataMat_lowPerf, aspect='auto', origin='lower', cmap='RdBu_r', vmin=xLow_lowPerf, vmax=xHigh_lowPerf)
    ax2.set_title(lowPerf_label + ' ' + cbarLabel + ' Matrix')
    ax2.set_xlabel('Test Time')
    ax2.set_ylabel('Train Time')
    ax2.set_xticks(tidx_to_plot)
    ax2.set_xticklabels(time_to_plot)
    ax2.set_yticks(tidx_to_plot)
    ax2.set_yticklabels(time_to_plot)
    ax2.axvline(tidx_to_plot[1], color='k', linestyle='--')
    ax2.axhline(tidx_to_plot[1], color='k', linestyle='--')
    fig.colorbar(im, ax=ax2)

    # Plot mean perforamnce along diagonal (0, 1)
    ax3 = fig.add_subplot(gs[2:, :])
    ax3.plot(times_crop, meanDiag_highPerf, label=highPerf_label)
    ax3.fill_between(times_crop, meanDiag_highPerf - stdErrDiag_highPerf, meanDiag_highPerf + stdErrDiag_highPerf, alpha=0.5)
    ax3.plot(times_crop, meanDiag_lowPerf, label=lowPerf_label)
    ax3.fill_between(times_crop, meanDiag_lowPerf - stdErrDiag_lowPerf, meanDiag_lowPerf + stdErrDiag_lowPerf, alpha=0.5)
    ax3.set_title('Diagonal Performance')
    ax3.set_xlabel('Time')
    ax3.set_ylabel(cbarLabel)
    if classifType == 'hemifield':
        ax3.axhline(0.5, color='k', linestyle='--')
    elif classifType == 'quadrant':
        ax3.axhline(0.25, color='k', linestyle='--')
    elif classifType == 'locGroups':
        ax3.axhline(0.167, color='k', linestyle='--')
    # ax3.set_ylim([np.min([xLow_highPerf, xLow_lowPerf]), np.max([xHigh_highPerf, xHigh_lowPerf])])
    ax3.legend()
    plt.tight_layout()
    plt.suptitle('Decoding Performance: ' + classifType + ' ' + freqband)
    plt.show()

    
if __name__ == '__main__':
    main()