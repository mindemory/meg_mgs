import numpy as np
from itertools import product
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, sosfiltfilt
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import circmean

def filter_and_downsample(trial_arr, time_arr, Fs):
    sos = butter(N=6, Wn=15, btype='low', fs=Fs, output='sos')
    downsampleFactor = 2 # 500Hz to 50Hz
    newSamples = int(trial_arr.shape[2] / downsampleFactor) # 2001 -> 200
    filtered_trial_arr = np.full_like(trial_arr, np.nan)
    valid_mask = ~np.isnan(trial_arr)  # Preserve original NaN locations

    for trial_idx in range(trial_arr.shape[0]):
        for channel_idx in range(trial_arr.shape[1]):
            trial_data = trial_arr[trial_idx, channel_idx]
            if np.any(valid_mask[trial_idx, channel_idx]):
                filtered_trial_arr[trial_idx, channel_idx] = sosfiltfilt(
                    sos, 
                    np.where(valid_mask[trial_idx, channel_idx], trial_data, 0)
                )
    # Apply downsampling
    filtered_trial_arr = filtered_trial_arr[:, :, ::downsampleFactor]
    filtered_time_arr = time_arr[::downsampleFactor]

    return filtered_trial_arr, filtered_time_arr

def extract_freqband(freqband, freqs, powspctrm_combined, powOphase='power', basecorr=True):
    # Extract features
    if freqband == 'broadband':
        valFreqs = np.where(((freqs > 2) & (freqs <= 4) )| (freqs >= 30))[0]
        freqs_nonBase = freqs[valFreqs]
        powspctrm_nonbased = powspctrm_combined[:, :, valFreqs, :]
        A_nb, B_nb, C_nb, D_nb = powspctrm_nonbased.shape
        A, B, C, D = powspctrm_combined.shape
        freqs_logg_nonBase = np.log10(freqs_nonBase)
        freqs_logg = np.log10(freqs)
        X_nonbase = np.vstack([np.ones_like(freqs_logg_nonBase), freqs_logg_nonBase]).T
        X = np.vstack([np.ones_like(freqs_logg), freqs_logg]).T
        XtX_inv = np.linalg.inv(X_nonbase.T @ X_nonbase)
        
        Y = powspctrm_nonbased.transpose(0, 1, 3, 2).reshape(-1, C_nb)
        # Batch matrix operations using Einstein summation
        beta = np.einsum('ij,kj->ki', XtX_inv @ X_nonbase.T, Y)  # Shape (A*B*D, 2)
        # Project onto new frequency basis
        broadband_combined = np.einsum('ij,kj->ki', X, beta)     # Shape (A*B*D, C)
        # Reshape back to original dimensions (A, B, D, C) -> (A, B, C, D)
        broadband_combined = broadband_combined.reshape(A, B, D, C).transpose(0, 1, 3, 2)
        powspctrm_combined_ = broadband_combined[:, :, 10, :]

        if basecorr:
            # Subtract average power
            avgMat = np.nanmean(powspctrm_combined_, axis=0)
            avgMat = np.repeat(avgMat[np.newaxis, :, :], powspctrm_combined_.shape[0], axis=0)
            powspctrm_combined_ = ((powspctrm_combined_ / avgMat) - 1) * 100 # Convert to % change
        
    else:    
        if freqband == 'theta':
            freq_idx = np.where((freqs >= 4) & (freqs <= 8))[0]
        elif freqband == 'alpha':
            freq_idx = np.where((freqs >= 8) & (freqs <= 12))[0]
        elif freqband == 'beta':
            freq_idx = np.where((freqs >= 12) & (freqs <= 35))[0]
        
        if powOphase == 'power':
            powspctrm_combined_ = np.nanmean(powspctrm_combined[:, :, freq_idx, :], axis=2)
        elif powOphase == 'phase':
            powspctrm_combined_ = circmean(powspctrm_combined[:, :, freq_idx, :], axis=2, low=-np.pi, high=np.pi, nan_policy='omit')

        if basecorr:
            # Subtract average power
            avgMat = np.nanmean(powspctrm_combined_, axis=0)
            avgMat = np.repeat(avgMat[np.newaxis, :, :], powspctrm_combined_.shape[0], axis=0)
            powspctrm_combined_ = 10**(powspctrm_combined_ / 10) / 10**(avgMat / 10)
    return powspctrm_combined_


def process_train_time(train_time, X, y, n_timepoints, n_trials, n_freqs=None):
    print(f'Processing training time point {train_time}/{n_timepoints}')
    
    n_splits = 10
    random_state = 42
    if n_freqs is not None:
        # auc_row = np.zeros(n_freqs)
        f1_row = np.zeros(n_freqs)
        f1_chance_row = np.zeros(n_freqs)
        for freq_idx in range(n_freqs):
            X_train_freq = X[:, :, freq_idx, train_time].reshape(n_trials, -1)
            skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
            for fold, (train_idx, test_idx) in enumerate(skf.split(X_train_freq, y)):
                sc = StandardScaler()
                X_train = X_train_freq[train_idx, :]
                X_train = sc.fit_transform(X_train)

                # Train SVM
                # clf = CalibratedClassifierCV(SVC(kernel='linear', decision_function_shape='ovo', max_iter=10000), cv=3)
                clf = SVC(kernel='linear', decision_function_shape='ovo', max_iter=10000)
                clf.fit(X_train, y[train_idx])

                
                X_test = X_train_freq[test_idx, :]
                X_test = sc.transform(X_test)
                # auc_row[freq_idx] += roc_auc_score(y[test_idx], clf.predict_proba(X_test), multi_class='ovo', average='macro')
                f1_row += f1_score(y[test_idx], clf.predict(X_test), average='macro')
                f1_chance_row = f1_score(y[test_idx], np.random.permutation(y[test_idx]), average='macro')
        
    else:
        # auc_row = np.zeros(n_timepoints)
        f1_row = np.zeros(n_timepoints)
        f1_chance_row = np.zeros(n_timepoints)
        X_train_time = X[:, :, train_time].reshape(n_trials, -1)
        
        # Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_train_time, y)):
            sc = StandardScaler()
            X_train = X_train_time[train_idx]
            X_train = sc.fit_transform(X_train)
            
            # clf = SVC(kernel='linear', decision_function_shape='ovo', probability=True)
            # clf = CalibratedClassifierCV(SVC(kernel='linear', decision_function_shape='ovo', max_iter=10000), cv=3)
            clf = SVC(kernel='linear', decision_function_shape='ovo', max_iter=10000)
            clf.fit(X_train, y[train_idx])
            
            # Test on all timepoints
            for t_test in range(n_timepoints):
                X_test_time = X[:, :, t_test].reshape(n_trials, -1)
                X_test = X_test_time[test_idx, :]
                X_test = sc.transform(X_test)
                # auc_row[t_test] += roc_auc_score(
                #     y[test_idx], 
                #     clf.predict_proba(X_test[test_idx]), 
                #     multi_class='ovo', 
                #     average='macro'
                # )
                f1_row[t_test] += f1_score(y[test_idx], clf.predict(X_test), average='macro')
                f1_chance_row[t_test] += f1_score(y[test_idx], np.random.permutation(y[test_idx]), average='macro')
    
    # Average across folds
    # auc_row /= skf.n_splits
    f1_row /= skf.n_splits
    f1_chance_row /= skf.n_splits
    # return train_time, auc_row
    return train_time, f1_row, f1_chance_row


def crossTemporalDecoding(X, trlInfo_, classCats, parallel=True):
    # Run classification
    if classCats == 'quadrant':
        y = np.select(
            condlist = [
                (trlInfo_ == 2) | (trlInfo_ == 3),
                (trlInfo_ == 4) | (trlInfo_ == 5),
                (trlInfo_ == 7) | (trlInfo_ == 8),
                (trlInfo_ == 9) | (trlInfo_ == 10)   
            ],
            choicelist = [1, 2, 3, 4],
            default = 0
        )
    elif classCats == 'locGroups':
        y = np.select(
            condlist = [
                (trlInfo_ == 1),
                (trlInfo_ == 2) | (trlInfo_ == 3),
                (trlInfo_ == 4) | (trlInfo_ == 5),
                (trlInfo_ == 6),
                (trlInfo_ == 7) | (trlInfo_ == 8),
                (trlInfo_ == 9) | (trlInfo_ == 10)
            ],
            choicelist = [1, 2, 3, 4, 5, 6],
            default = 0
        )
    elif classCats == 'indivTargets':
        y = np.select(
            condlist = [
                (trlInfo_ == 1),
                (trlInfo_ == 2),
                (trlInfo_ == 3),
                (trlInfo_ == 4),
                (trlInfo_ == 5),
                (trlInfo_ == 6),
                (trlInfo_ == 7),
                (trlInfo_ == 8),
                (trlInfo_ == 9),
                (trlInfo_ == 10)
            ],
            choicelist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            default = 0
        )

    # Remove unlabeled trials (if any)
    valid_trials = y != 0
    X = X[valid_trials]
    y = y[valid_trials]
    y -= y.min() # Make labels start from 0

    n_trials, _, n_timepoints = X.shape
    # auc_matrix = np.zeros((n_timepoints, n_timepoints))  # Train-time x Test-time
    f1_matrix = np.zeros((n_timepoints, n_timepoints))  # Train-time x Test-time
    f1_chance_matrix = np.zeros((n_timepoints, n_timepoints))  # Train-time x Test-time

    if parallel:
        from joblib import Parallel, delayed
        from multiprocessing import cpu_count
        n_jobs = min(cpu_count() - 1, 8)  # Leave one CPU free
        print(f'Parallelizing across {n_jobs} cores')
        results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=1)(
            delayed(process_train_time)(
                train_time, X, y, n_timepoints, n_trials
            ) for train_time in range(n_timepoints)
        )
        
        # Collect results
        # for train_time, auc_row in results:
        #     auc_matrix[train_time, :] = auc_row
        for train_time, f1_row, f1_chance_row in results:
            f1_matrix[train_time, :] = f1_row
            f1_chance_matrix[train_time, :] = f1_chance_row
    else:
        # from sklearn.preprocessing import label_binarize
        for train_time in range(n_timepoints):
            if (train_time / n_timepoints * 100) % 10 == 0:
                print('     Finished training ' + str(round(train_time / n_timepoints * 100)) + '%')

            X_train_time = X[:, :, train_time].reshape(n_trials, -1)

            # Stratified K-Fold Cross-Validation
            skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
            for fold, (train_idx, test_idx) in enumerate(skf.split(X_train_time, y)):
                sc = StandardScaler()
                X_train = X_train_time[train_idx, :]
                X_train = sc.fit_transform(X_train)

                # Train SVM
                # clf = CalibratedClassifierCV(SVC(kernel='linear', decision_function_shape='ovo', max_iter=10000), cv=3)
                clf = SVC(kernel='linear', decision_function_shape='ovo', max_iter=10000)
                clf.fit(X_train, y[train_idx])

                # Predict for all test samples in fold
                for test_time in range(n_timepoints):
                    X_test_time = X[:, :, test_time].reshape(n_trials, -1)
                    X_test = X_test_time[test_idx, :]
                    X_test = sc.transform(X_test)
                    y_pred = clf.predict(X_test)
                    f1_matrix[train_time, test_time] += f1_score(y[test_idx], y_pred, average='macro')
                    f1_chance_matrix[train_time, test_time] += f1_score(y[test_idx], np.random.permutation(y[test_idx]), average='macro')
                    # auc_matrix[train_time, test_time] += roc_auc_score(y[test_idx], clf.predict_proba(X_test[test_idx]), multi_class='ovo', average='macro')
                    # f1_matrix[train_time, test_time] += f1_score(y[test_idx], clf.predict(X_test), average='macro')

        # Avererage acorss folds
        # auc_matrix /= skf.n_splits
        f1_matrix /= skf.n_splits
        f1_chance_matrix /= skf.n_splits

    # return accuracy_matrix, f1score_matrix, auc_matrix
    # return auc_matrix
    return f1_matrix, f1_chance_matrix

def TimeFrequencyDecoding(X, trlInfo_, classCats, crossTemporal=False, parallel=True):
    # Run classification
    if classCats == 'quadrant':
        y = np.select(
            condlist = [
                (trlInfo_ == 2) | (trlInfo_ == 3),
                (trlInfo_ == 4) | (trlInfo_ == 5),
                (trlInfo_ == 7) | (trlInfo_ == 8),
                (trlInfo_ == 9) | (trlInfo_ == 10)   
            ],
            choicelist = [1, 2, 3, 4],
            default = 0
        )
    elif classCats == 'locGroups':
        y = np.select(
            condlist = [
                (trlInfo_ == 1),
                (trlInfo_ == 2) | (trlInfo_ == 3),
                (trlInfo_ == 4) | (trlInfo_ == 5),
                (trlInfo_ == 6),
                (trlInfo_ == 7) | (trlInfo_ == 8),
                (trlInfo_ == 9) | (trlInfo_ == 10)
            ],
            choicelist = [1, 2, 3, 4, 5, 6],
            default = 0
        )
    elif classCats == 'indivTargets':
        y = np.select(
            condlist = [
                (trlInfo_ == 1),
                (trlInfo_ == 2),
                (trlInfo_ == 3),
                (trlInfo_ == 4),
                (trlInfo_ == 5),
                (trlInfo_ == 6),
                (trlInfo_ == 7),
                (trlInfo_ == 8),
                (trlInfo_ == 9),
                (trlInfo_ == 10)
            ],
            choicelist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            default = 0
        )

    # Remove unlabeled trials (if any)
    valid_trials = y != 0
    X = X[valid_trials]
    y = y[valid_trials]
    y -= y.min() # Make labels start from 0

    n_trials, n_channels, n_freqs, n_timepoints = X.shape
    if crossTemporal:
        # auc_matrix = np.zeros((n_timepoints, n_timepoints, n_freqs))  # Train-time x Test-time x Freq
        f1_matrix = np.zeros((n_timepoints, n_timepoints, n_freqs))  # Train-time x Test-time x Freq
        f1_chance_matrix = np.zeros((n_timepoints, n_timepoints, n_freqs))
    else:
        # auc_matrix = np.zeros((n_timepoints, n_freqs))  # Time x Freq
        f1_matrix = np.zeros((n_timepoints, n_freqs))  # Time x Freq
        f1_chance_matrix = np.zeros((n_timepoints, n_freqs))

    if parallel:
        from joblib import Parallel, delayed
        from multiprocessing import cpu_count
        if crossTemporal:
            print('No support for this yet, run without parallelization')
            exit()
        else:
            n_jobs = min(cpu_count() - 1, 8)  # Leave one CPU free
            print(f'Parallelizing across {n_jobs} cores')
            results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=1)(
                delayed(process_train_time)(
                    train_time, X, y, n_timepoints, n_trials, n_freqs
                ) for train_time in range(n_timepoints)
            )

            # Collect results
            # for train_time, auc_row in results:
            #     auc_matrix[train_time, :] = auc_row
            for train_time, f1_row, f1_chance_row in results:
                f1_matrix[train_time, :] = f1_row
                f1_chance_matrix[train_time, :] = f1_chance_row
    else:
        for train_time in range(n_timepoints):
            if (train_time / n_timepoints * 100) % 10 == 0:
                print('     Finished training ' + str(round(train_time / n_timepoints * 100)) + '%')
            for freq_idx in range(n_freqs):
                X_train_freq = X[:, :, freq_idx, train_time].reshape(n_trials, -1)
                skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
                for fold, (train_idx, test_idx) in enumerate(skf.split(X_train_freq, y)):
                    sc = StandardScaler()
                    X_train = X_train_freq[train_idx, :]
                    X_train = sc.fit_transform(X_train)

                    # Train SVM
                    # clf = CalibratedClassifierCV(SVC(kernel='linear', decision_function_shape='ovo', max_iter=10000), cv=3)
                    clf = SVC(kernel='linear', decision_function_shape='ovo', max_iter=10000)
                    clf.fit(X_train, y[train_idx])

                    if crossTemporal == True:
                        # Predict for all test samples in fold
                        for test_time in range(n_timepoints):
                            X_test = X[:, :, freq_idx, test_time].reshape(n_trials, -1)
                            X_test = sc.transform(X_test)
                            # auc_matrix[train_time, test_time, freq_idx] += roc_auc_score(y[test_idx], clf.predict_proba(X_test[test_idx]), multi_class='ovo', average='macro')
                            f1_matrix[train_time, test_time, freq_idx] += f1_score(y[test_idx], clf.predict(X_test), average='macro')
                            f1_chance_matrix[train_time, test_time, freq_idx] += f1_score(y[test_idx], np.random.permutation(y[test_idx]), average='macro')
                    else:
                        X_test = X_train_freq[test_idx, :]
                        X_test = sc.transform(X_test)
                        # auc_matrix[train_time, freq_idx] += roc_auc_score(y[test_idx], clf.predict_proba(X_test), multi_class='ovo', average='macro')
                        f1_matrix[train_time, freq_idx] = f1_score(y[test_idx], clf.predict(X_test), average='macro')
                        f1_chance_matrix[train_time, freq_idx] = f1_score(y[test_idx], np.random.permutation(y[test_idx]), average='macro')

        # Avererage acorss folds
        # auc_matrix /= skf.n_splits
        f1_matrix /= skf.n_splits
        f1_chance_matrix /= skf.n_splits

    # return auc_matrix
    return f1_matrix, f1_chance_matrix

def runMultiClassClassification(TFRmat, v73=False, powOphase='power', classCats='quadrant', freqband='alpha', basecorr=True):
    import gc

    leftDataStr = 'TFRleft_' + powOphase
    rightDataStr = 'TFRright_' + powOphase
    if powOphase == 'power':
        metric = 'powspctrm'
    elif powOphase == 'phase':
        metric = 'phaseangle'

    if v73:
        freqs = np.array(TFRmat[leftDataStr]['freq']).T[0]
        times = np.array(TFRmat[leftDataStr]['time']).T[0]
        trialInfo_left = np.array(TFRmat[leftDataStr]['trialinfo']).T
        powspctrm_left = np.array(TFRmat[leftDataStr][metric]).T
        trialInfo_right = np.array(TFRmat[rightDataStr]['trialinfo']).T
        powspctrm_right = np.array(TFRmat[rightDataStr][metric]).T
    else:
        # ch_label = TFRmat['TFRleft_power'][0][0][0]
        # dimord = TFRmat['TFRleft_power'][0][0][1]
        freqs = TFRmat[leftDataStr][0][0][2][0]
        times = TFRmat[leftDataStr][0][0][3][0]
        trialInfo_left = TFRmat[leftDataStr][0][0][5]
        powspctrm_left = TFRmat[leftDataStr][0][0][7]
        trialInfo_right = TFRmat[rightDataStr][0][0][5]
        powspctrm_right = TFRmat[rightDataStr][0][0][7]
    # Combine powspctrm and trialinfo
    powspctrm_combined = np.concatenate((powspctrm_left, powspctrm_right), axis=0)
    trialInfo_combined = np.concatenate((trialInfo_left, trialInfo_right), axis=0)
    del TFRmat, powspctrm_left, powspctrm_right, trialInfo_left, trialInfo_right
    gc.collect()

    # Time Of Interest
    if freqband == 'broadband' or freqband == 'crossfrequency':
        time_idx = np.where((times >= -0.1) & (times <= 1.7))[0]
    else:
        time_idx = np.where((times >= -0.5) & (times <= 2))[0]
    times_crop = times[time_idx]

    trlInfo_ = trialInfo_combined[:, 1]

    ######################################################## CLASSIFICATION ########################################################
    if freqband == 'crossfrequency':
        # This outputs auc_matrix of either (time x time x freq) or (time x freq) depending on crossTemporal
        freq_idx = np.where((freqs > 2))[0]
        valFreqs = freqs[freq_idx]
        powspctrm_combined = powspctrm_combined[:, :, freq_idx, :]
        avgMat = np.nanmean(powspctrm_combined, axis=0) # Average across trials (chan x freq x time)
        avgMat = np.repeat(avgMat[np.newaxis, :, :, :], powspctrm_combined.shape[0], axis=0)
        powspctrm_combined = 10**(powspctrm_combined / 10) / 10**(avgMat / 10)
        powspctrm_combined_ = powspctrm_combined[:, :, :, time_idx]
        # auc_matrix = TimeFrequencyDecoding(powspctrm_combined_, trlInfo_, classCats, crossTemporal=False, parallel=True)
        f1_matrix, f1_chance_matrix = TimeFrequencyDecoding(powspctrm_combined_, trlInfo_, classCats, crossTemporal=False, parallel=True)
        # Clear memory
        del powspctrm_combined_
        gc.collect()

        # return auc_matrix, times_crop, valFreqs
        return f1_matrix, f1_chance_matrix, times_crop, valFreqs
    else:
        # This outputs auc_matrix of (time x time)
        # Extract features
        powspctrm_combined_ = extract_freqband(freqband, freqs, powspctrm_combined, powOphase, basecorr)
        powspctrm_combined_ = powspctrm_combined_[:, :, time_idx]
        print('Starting decoding')
        # auc_matrix = crossTemporalDecoding(powspctrm_combined_, trlInfo_, classCats, parallel=True)
        f1_matrix, f1_chance_matrix = crossTemporalDecoding(powspctrm_combined_, trlInfo_, classCats, parallel=True)

        # Clear memory
        del powspctrm_combined_
        gc.collect()

        # return auc_matrix, times_crop
        return f1_matrix, f1_chance_matrix, times_crop


