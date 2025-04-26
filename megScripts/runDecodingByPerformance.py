import os, pickle, h5py, socket
import numpy as np
from scipy.io import loadmat
from decodingFuncs import runMultiClassClassificationByPerformance
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
    
    if powOphase == 'power':
        TFR_fpath = os.path.join(megRoot, fNameRoot + '_TFR_evoked_lineremoved.mat')
    else:
        TFR_fpath = os.path.join(megRoot, fNameRoot + '_TFR_phase.mat')
    classifHolderRoot = os.path.join(megRoot, 'decodingByPerf_lineremoved')
    if not os.path.exists(classifHolderRoot):
        os.mkdir(classifHolderRoot)

    classifPath = os.path.join(classifHolderRoot, classifName)

    if os.path.exists(classifPath):
        # Load the classifier
        print('Loading existing classifier performance')
        with open(classifPath, 'rb') as f:
            classifData = pickle.load(f)
        f1_matrix_highPerf = classifData['f1_matrix_highPerf']
        f1_chance_matrix_highPerf = classifData['f1_chance_matrix_highPerf']
        f1_matrix_lowPerf = classifData['f1_matrix_lowPerf']
        f1_chance_matrix_lowPerf = classifData['f1_chance_matrix_lowPerf']
        times_crop = classifData['times_crop']
    else:
        print('Running classifier')
        # Load stimlocked data and TFR data
        # if subjID == 12 or subjID == 17 : # For subjects that were saved using -v7.3
        ii_sess = loadmat(iisess_fpath)
        epocStimLocked = loadmat(stimLocked_fpath)
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
        # f1Matrix, f1ChanceMatrix, times_crop = runMultiClassClassification(TFRmat, v73, powOphase, classCats=classifType, freqband=freqband, basecorr=baseCorr)
        f1_matrix_highPerf, f1_chance_matrix_highPerf, f1_matrix_lowPerf, f1_chance_matrix_lowPerf, times_crop = runMultiClassClassificationByPerformance(subjID, 
                                                                                                                                                          ii_sess, 
                                                                                                                                                          epocStimLocked, 
                                                                                                                                                          TFRmat, 
                                                                                                                                                          v73=v73, 
                                                                                                                                                          powOphase=powOphase, 
                                                                                                                                                          metric='ierr', 
                                                                                                                                                          classCats=classifType, 
                                                                                                                                                          freqband=freqband, 
                                                                                                                                                          basecorr=baseCorr)

        classifData['f1_matrix_highPerf'] = f1_matrix_highPerf
        classifData['f1_chance_matrix_highPerf'] = f1_chance_matrix_highPerf
        classifData['f1_matrix_lowPerf'] = f1_matrix_lowPerf
        classifData['f1_chance_matrix_lowPerf'] = f1_chance_matrix_lowPerf        
        classifData['times_crop'] = times_crop
        # Save the classifier
        with open(classifPath, 'wb') as f:
            pickle.dump(classifData, f)
        del TFRmat, classifData
        gc.collect()
        print()
    # return subjID, aucMatrix, times_crop
    # return subjID, f1Matrix, f1ChanceMatrix, times_crop
    return subjID, f1_matrix_highPerf, f1_chance_matrix_highPerf, f1_matrix_lowPerf, f1_chance_matrix_lowPerf, times_crop


def main():
    # subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18, 19, 20, 
    #            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 
               18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32]
    # subList = [1]
    # subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17]

    freqband = 'theta' # valid options: alpha, beta, broadband
    powOphase = 'power' # valid options: power, phase
    classifType = 'quadrant' # valid options: hemifield, quadrant, locGroups, indivTargets
    if powOphase == 'power':
        classifName = 'classif_' + classifType + '_' + freqband + '.pkl'
    else:
        classifName = 'classif_' + classifType + '_' + freqband + '_phase.pkl'
    if freqband == 'broadband':
        ntimePts = 90
    else:
        ntimePts = 125
    # accMat = np.empty((len(subList), ntimePts, ntimePts))
    f1MatHighPerf = np.empty((len(subList), ntimePts, ntimePts))
    f1ChanceMatHighPerf = np.empty((len(subList), ntimePts, ntimePts))
    f1MatLowPerf = np.empty((len(subList), ntimePts, ntimePts))
    f1ChanceMatLowPerf = np.empty((len(subList), ntimePts, ntimePts))


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
        # f1Mat[sIdx, :, :], f1ChanceMat[sIdx, :, :], times_crop = res[1], res[2], res[3]
        f1MatHighPerf[sIdx, :, :], f1ChanceMatHighPerf[sIdx, :, :], f1MatLowPerf[sIdx, :, :], f1ChanceMatLowPerf[sIdx, :, :], times_crop = res[1], res[2], res[3], res[4], res[5]
    # Average the accuracy and f1score matrices
    # auc_matrix = np.nanmean(aucMat[:, :, :], axis=0)
    # f1_matrix = np.nanmean(f1Mat, axis=0)
    # f1_chance_matrix = np.nanmean(f1ChanceMat, axis=0)
    f1_matrix_highPerf = np.nanmean(f1MatHighPerf, axis=0)
    f1_chance_matrix_highPerf = np.nanmean(f1ChanceMatHighPerf, axis=0)
    f1_matrix_lowPerf = np.nanmean(f1MatLowPerf, axis=0)
    f1_chance_matrix_lowPerf = np.nanmean(f1ChanceMatLowPerf, axis=0)


    if freqband == 'broadband':
        time_to_plot = [-0.1, 0, 0.2, 0.5, 1, 1.5]
    else:
        time_to_plot = [-0.5, 0, 0.2, 0.5, 1, 1.5, 2]
    tidx_to_plot = [np.argmin(np.abs(times_crop - t)) for t in time_to_plot]

    # Identify the lower and upper bounds for the colorbar
    qtThresh = 0.0005
    # xLowAccHighperf, xHighAccHighpef = np.quantile(f1_matrix_highPerf, qtThresh), np.quantile(f1_matrix_highPerf, 1-qtThresh)
    xLowF1Highperf, xHighF1Highperf = np.quantile(f1_matrix_highPerf, qtThresh), np.quantile(f1_matrix_highPerf, 1-qtThresh)
    # xLowAccLowperf, xHighAccLowperf = np.quantile(accuracy_matrix_lowPerf, qtThresh), np.quantile(accuracy_matrix_lowPerf, 1-qtThresh)
    xLowF1Lowperf, xHighF1Lowperf = np.quantile(f1_matrix_lowPerf, qtThresh), np.quantile(f1_matrix_lowPerf, 1-qtThresh)

    

    metricToPlot = 'F1'
    
    xLow_highPerf, xHigh_highPerf = xLowF1Highperf, xHighF1Highperf
    xLow_lowPerf, xHigh_lowPerf = xLowF1Lowperf, xHighF1Lowperf
    dataMat_highPerf = f1_matrix_highPerf
    dataMat_lowPerf = f1_matrix_lowPerf
    rawMat_highPerf = f1MatHighPerf
    rawMat_lowPerf = f1MatLowPerf
    cbarLabel = 'F1 score'
    lowPerf_label = 'Low error'
    performanceMetric = 'ierr'


    if performanceMetric == 'ierr' or performanceMetric == 'ferr':
        highPerf_label = 'Large error'
        lowPerf_label = 'Low error'
    elif performanceMetric == 'irt' or performanceMetric == 'frt':
        highPerf_label = 'Fast saccade'
        lowPerf_label = 'Slow saccade'
    # Compute diagonal for each subject and compute mean and stderror
    diag_highPerf = np.empty((rawMat_highPerf.shape[0], rawMat_highPerf.shape[1]))
    diag_lowPerf = np.empty((rawMat_lowPerf.shape[0], rawMat_lowPerf.shape[1]))
    diag_chancehighPerf = np.empty((rawMat_highPerf.shape[0], rawMat_highPerf.shape[1]))
    diag_chancelowPerf = np.empty((rawMat_lowPerf.shape[0], rawMat_lowPerf.shape[1]))
    for s in range(rawMat_highPerf.shape[0]):
        diag_highPerf[s, :] = np.diag(rawMat_highPerf[s, :, :])
        diag_lowPerf[s, :] = np.diag(rawMat_lowPerf[s, :, :])
        diag_chancehighPerf[s, :] = np.diag(f1ChanceMatHighPerf[s, :, :])
        diag_chancelowPerf[s, :] = np.diag(f1ChanceMatLowPerf[s, :, :])


    meanDiag_highPerf = np.nanmean(diag_highPerf, axis=0)
    stdErrDiag_highPerf = np.nanstd(diag_highPerf, axis=0) / np.sqrt(diag_highPerf.shape[0])
    meanDiag_lowPerf = np.nanmean(diag_lowPerf, axis=0)
    stdErrDiag_lowPerf = np.nanstd(diag_lowPerf, axis=0) / np.sqrt(diag_lowPerf.shape[0])

    meanDiagChance_highPerf = np.nanmean(diag_chancehighPerf, axis=0)
    meanDiagChance_lowPerf = np.nanmean(diag_chancelowPerf, axis=0)

    tidx_to_baseline = [np.argmin(np.abs(times_crop - t)) for t in [-0.3, 0]]
    baseline_highPerf = np.nanmean(meanDiag_highPerf[tidx_to_baseline[0]:tidx_to_baseline[1]], axis=0)
    baseline_lowPerf = np.nanmean(meanDiag_lowPerf[tidx_to_baseline[0]:tidx_to_baseline[1]], axis=0)
    baselineChance_highPerf = np.nanmean(meanDiagChance_highPerf[tidx_to_baseline[0]:tidx_to_baseline[1]], axis=0)
    baselineChance_lowPerf = np.nanmean(meanDiagChance_lowPerf[tidx_to_baseline[0]:tidx_to_baseline[1]], axis=0)
    print('Baseline high performance: ' + str(np.mean(baseline_highPerf)))
    print('Baseline low performance: ' + str(np.mean(baseline_lowPerf)))
    print('Baseline chance high performance: ' + str(np.mean(baselineChance_highPerf)))
    print('Baseline chance low performance: ' + str(np.mean(baselineChance_lowPerf)))

    # Adding temporarily
    chanceBaseline = 0.23
    extRange = 0.03
    if classifType == 'quadrant':
        # xLow_highPerf, xHigh_highPerf = 0.23, 0.27
        # xLow_lowPerf, xHigh_lowPerf = 0.23, 0.27
        xLow_highPerf, xHigh_highPerf = chanceBaseline-extRange, chanceBaseline+extRange
        xLow_lowPerf, xHigh_lowPerf = chanceBaseline-extRange, chanceBaseline+extRange
    # Plot the performance matrices
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(4, 4, figure=fig)
    # Highperformance matrix plot (0, 0)
    ax1 = fig.add_subplot(gs[:2, :2])
    im = ax1.imshow(dataMat_highPerf, aspect='equal', origin='lower', cmap='RdBu_r', vmin=xLow_highPerf, vmax=xHigh_highPerf)
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
    im = ax2.imshow(dataMat_lowPerf, aspect='equal', origin='lower', cmap='RdBu_r', vmin=xLow_lowPerf, vmax=xHigh_lowPerf)
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
        # ax3.axhline(0.25, color='k', linestyle='--')
        ax3.axhline(chanceBaseline, color='k', linestyle='--')
    elif classifType == 'locGroups':
        ax3.axhline(0.167, color='k', linestyle='--')
    # ax3.set_ylim([np.min([xLow_highPerf, xLow_lowPerf]), np.max([xHigh_highPerf, xHigh_lowPerf])])
    ax3.legend()
    plt.tight_layout()
    plt.suptitle('Decoding Performance: ' + classifType + ' ' + freqband)
    # plt.show()
    if powOphase == 'power':
        # figPath = os.path.join(figDir, 'groupByPerformance_' + classifType + '_' + freqband + '_lineremoved.png')
        # figPath = os.path.join(figDir, 'groupByPerformance_' + classifType + '_' + freqband + '_lineremoved.svg')
    else:
        figPath = os.path.join(figDir, 'groupByPerformance_' + classifType + '_' + freqband + '_phase.png')
    plt.savefig(figPath, dpi=300)


if __name__ == '__main__':
    main()