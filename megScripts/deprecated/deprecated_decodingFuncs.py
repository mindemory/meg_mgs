
def crossTemporalDecodingWithConfidence(X, trlInfo_, performance, classCats, cvType='stratifiedCV'):
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
    # y -= y.min() # Make labels start from 0
     
    performance_valid = performance[valid_trials]
    print('We are here')

    n_trials, _, n_timepoints = X.shape
    auc_matrix = np.zeros((n_timepoints, n_timepoints))  # Train-time x Test-time
    accuracy_matrix = np.zeros((n_timepoints, n_timepoints))  # Train-time x Test-time
    f1score_matrix = np.zeros((n_timepoints, n_timepoints))  # Train-time x Test-time
    y_guessed = np.zeros((n_timepoints, n_timepoints, n_trials))  # Train-time x Test-time x n_trials
    y_guessed_proba = np.zeros((n_timepoints, n_timepoints, n_trials, len(np.unique(y)))) # Train-time x Test-time x n_trials x n_classes

    # Stratified K-Fold Cross-Validation
    if cvType == 'stratifiedCV':
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

        for train_time in range(n_timepoints):
            if (train_time / n_timepoints * 100) % 10 == 0:
                print('     Finished training ' + str(round(train_time / n_timepoints * 100)) + '%')

            X_train_time = X[:, :, train_time].reshape(n_trials, -1)

            for fold, (train_idx, test_idx) in enumerate(skf.split(X_train_time, y)):
                X_train = X_train_time[train_idx]

                # Train SVM
                clf = CalibratedClassifierCV(LinearSVC(max_iter=10000))
                clf.fit(X_train, y[train_idx])

                # Predict for all test samples in fold
                for test_time in range(n_timepoints):
                    X_test_time = X[:, :, test_time].reshape(n_trials, -1)
                    y_proba_this = clf.predict_proba(X_test_time[test_idx])
                    y_guessed_proba[train_time, test_time, test_idx, :] = y_proba_this
                    y_guessed[train_time, test_time, test_idx] = clf.predict(X_test_time[test_idx])
                    auc_matrix[train_time, test_time] += roc_auc_score(y[test_idx], y_proba_this,  multi_class='ovo', average='macro')
                    accuracy_matrix[train_time, test_time] += accuracy_score(y[test_idx], y_guessed[train_time, test_time, test_idx])
                    f1score_matrix[train_time, test_time] += f1_score(y[test_idx], y_guessed[train_time, test_time, test_idx], average='macro')

        # Avererage acorss folds
        auc_matrix /= skf.n_splits
        accuracy_matrix /= skf.n_splits
        f1score_matrix /= skf.n_splits
    elif cvType == 'leaveOneOut':
        loo = LeaveOneOut()

        for train_time in range(n_timepoints):
            if (train_time / n_timepoints * 100) % 10 == 0:
                print('     Finished training ' + str(round(train_time / n_timepoints * 100)) + '%')

            X_train_time = X[:, :, train_time].reshape(n_trials, -1)

            for train_trl_idx, test_trl_idx in loo.split(X_train_time, y):
                X_train = X_train_time[train_trl_idx]

                # Train SVM
                clf = CalibratedClassifierCV(LinearSVC(max_iter=10000))
                clf.fit(X_train, y[train_trl_idx])

                # Predict for all test samples in fold
                for test_time in range(n_timepoints):
                    X_test_time = X[:, :, test_time].reshape(n_trials, -1)
                    y_guessed_proba[train_time, test_time, test_trl_idx, :] = clf.predict_proba(X_test_time[test_trl_idx])
                    y_guessed[train_time, test_time, test_trl_idx] = clf.predict(X_test_time[test_trl_idx])

                    # Calculate metrics
                    auc_matrix[train_time, test_time] += roc_auc_score(y[test_trl_idx], y_guessed_proba[train_time, test_time, test_trl_idx, :],  multi_class='ovo', average='macro')
                    accuracy_matrix[train_time, test_time] += accuracy_score(y[test_trl_idx], y_guessed[train_time, test_time, test_trl_idx])
                    f1score_matrix[train_time, test_time] += f1_score(y[test_trl_idx], y_guessed[train_time, test_time, test_trl_idx], average='macro')

        # Avererage acorss folds
        auc_matrix /= n_trials
        accuracy_matrix /= n_trials
        f1score_matrix /= n_trials

    return auc_matrix, accuracy_matrix, f1score_matrix, y_guessed, y_guessed_proba, performance_valid

def temporalDecoding(X, trlInfo_, classCats):
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

    # Remove unlabeled trials (if any)
    valid_trials = y != 0
    X = X[valid_trials]
    y = y[valid_trials]
    y -= y.min() # Make labels start from 0

    n_trials, n_timepoints = X.shape
    accuracy_matrix = np.zeros((n_timepoints, 1))  # Train-time x Test-time
    f1score_matrix = np.zeros((n_timepoints, 1))  # Train-time x Test-time

    # Stratified K-Fold Cross-Validation
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

    for train_time in range(n_timepoints):

        # X_train_time = X[:, :, train_time].reshape(n_trials, -1)
        X_train_time = X[:, train_time].reshape(-1, 1)

        accuracies = []
        f1scores = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_train_time, y)):
            X_train = X_train_time[train_idx]
            X_test = X_train_time[test_idx]

            # Train SVM
            # clf = SVC(kernel='linear', decision_function_shape='ovr')
            clf = CalibratedClassifierCV(LinearSVC(max_iter=10000))
            clf.fit(X_train, y[train_idx])

            # Predict for all test samples in fold
            preds = clf.predict(X_test)

            # Store the results
            accuracies.append(accuracy_score(y[test_idx], preds))
            f1scores.append(f1_score(y[test_idx], preds, average='macro'))
        
        accuracy_matrix[train_time] = np.mean(accuracies)
        f1score_matrix[train_time] = np.mean(f1scores)

    return accuracy_matrix, f1score_matrix

def generate_pc_timeseries(data, time, tinfo, n_groups=2, n_components=2):
    if n_groups == 4:
        groups =  {
            'Q1': np.where((tinfo == 2) | (tinfo == 3))[0],
            'Q2': np.where((tinfo == 4) | (tinfo == 5))[0],
            'Q3': np.where((tinfo == 7) | (tinfo == 8))[0],
            'Q4': np.where((tinfo == 9) | (tinfo == 10))[0],
        }
    elif n_groups == 2:
        groups =  {
            'right': np.where((tinfo == 2) | (tinfo == 3) | (tinfo == 1) | (tinfo == 9) | (tinfo == 10))[0],
            'left': np.where((tinfo == 4) | (tinfo == 5) | (tinfo == 6) | (tinfo == 7) | (tinfo == 8))[0],
        }

    # Average data for each group
    averaged_data = {}
    for grp_name, indices in groups.items():
        averaged_data[grp_name] = np.nanmean(data[indices], axis=0) # (nchans, ntimes)

    # Create matrix D at t=0
    tRelevIdx = np.where(np.isclose(time, 0, atol=1e-3))[0][0]
    D = np.array([avg[:, tRelevIdx] for avg in averaged_data.values()]) # (ngroups, nchans)
    # Demean columns
    D = D - np.mean(D, axis=0)

    # Perform PCA on D
    pca = PCA(n_components=n_components)
    pc_comps = pca.fit_transform(D) # (ngroups, ncomponents)
    # Print explained variance
    print(pca.explained_variance_ratio_)

    # Apply PCA to every time point
    pc_timeseries = np.zeros((len(groups), 2, data.shape[2]))
    for t in range(data.shape[2]):
        D_t = np.array([avg[:, t] for avg in averaged_data.values()])
        D_t = D_t - np.mean(D_t, axis=0)
        pc_timeseries[:, :, t] = pca.transform(D_t)

    return pc_timeseries, groups

def runMultiClassClassificationByPerformance(subjID, ii_sess, epocStimLocked, TFRmat, v73=False, metric='ierr', classCats='quadrant', freqband='alpha', basecorr=True):
    ######################################################## EXTRACTION ########################################################
    # Extract ii_sess data
    if metric == 'ierr':
        performance = ii_sess['ii_sess']['i_sacc_err'][0, 0].T[0]
    elif metric == 'ferr':
        performance = ii_sess['ii_sess']['f_sacc_err'][0, 0].T[0]
    elif metric == 'irt':
        performance = ii_sess['ii_sess']['i_sacc_rt'][0, 0].T[0]
    elif metric == 'frt':
        performance = ii_sess['ii_sess']['f_sacc_rt'][0, 0].T[0]
    else:
        raise ValueError('Invalid performance metric, should be one of ierr, ferr, irt, frt')
    tarlocGaze = ii_sess['ii_sess']['tarlocCode'][0, 0].T[0]
    
    # Account for special cases
    rnum = ii_sess['ii_sess']['r_num'][0, 0].T[0]
    if subjID == 4: # Trial 1 from run10 is missing
        badTrials = [np.where(rnum == 10)[0][0]]
    elif subjID == 5: # Remove run 1 and 9
        badTrials = np.where((rnum == 1) | (rnum == 9))[0]
    elif subjID == 10: # Remove run 2 and 7
        badTrials = np.where((rnum == 2) | (rnum == 7))[0]
    elif subjID == 11: # Remove run 1, 3 and 8
        badTrials = np.where((rnum == 1) | (rnum == 3) | (rnum == 8))[0]
    elif subjID == 12: # Trial 1 from run 8 is missing
        badTrials = [np.where(rnum == 8)[0][0]]
    elif subjID == 13: # Trials 1, 2 from run 2 are missing
        badTrials = np.where(rnum == 2)[0][:2]
    elif subjID == 19: # Remove run 8 and 9
        badTrials = np.where((rnum == 8) | (rnum == 9))[0]
    elif subjID == 23: # Remove run 1
        badTrials = np.where(rnum == 1)[0]
    elif subjID == 25: # Remove run 8
        badTrials = np.where(rnum == 8)[0]
    elif subjID == 31: # Remove run 2; Also remove the last 17 trials from run 4
        badTrials = np.where(rnum == 2)[0]
        badTrials = np.concatenate((badTrials, np.where(rnum == 4)[0][-17:]))
    elif subjID == 32: # Remove run 2
        badTrials = np.where(rnum == 2)[0]
    else:
        badTrials = []
        
    # Remove the bad trials
    if len(badTrials) > 0:
        performance = np.delete(performance, badTrials)
        tarlocGaze = np.delete(tarlocGaze, badTrials)

    # Extract epocStimLocked data
    trialEpoched_info = epocStimLocked['epocStimLocked'][0][0][1]
    tarlocEpoched = trialEpoched_info[:, 1]
    trialEpoched = epocStimLocked['epocStimLocked'][0][0][4][0]
    trialsEpoched_list = [np.array(t) for t in trialEpoched]
    trialEpoched_arr = np.stack(trialsEpoched_list, axis=0)
    # Extract TFR data
    if v73:
        freqs = np.array(TFRmat['TFRleft_power']['freq']).T[0]
        times = np.array(TFRmat['TFRleft_power']['time']).T[0]
        trialInfo_left = np.array(TFRmat['TFRleft_power']['trialinfo']).T
        powspctrm_left = np.array(TFRmat['TFRleft_power']['powspctrm']).T
        trialInfo_right = np.array(TFRmat['TFRright_power']['trialinfo']).T
        powspctrm_right = np.array(TFRmat['TFRright_power']['powspctrm']).T
    else:
        # ch_label = TFRmat['TFRleft_power'][0][0][0]
        # dimord = TFRmat['TFRleft_power'][0][0][1]
        freqs = TFRmat['TFRleft_power'][0][0][2][0]
        times = TFRmat['TFRleft_power'][0][0][3][0]
        trialInfo_left = TFRmat['TFRleft_power'][0][0][5]
        powspctrm_left = TFRmat['TFRleft_power'][0][0][7]
        trialInfo_right = TFRmat['TFRright_power'][0][0][5]
        powspctrm_right = TFRmat['TFRright_power'][0][0][7]
    
    ######################################################## DIVIDING DATA ########################################################
    

    left_trials = []
    right_trials = []
    for ijk in range(trialEpoched_arr.shape[0]):
        if trialEpoched_info[ijk, 1] == 4 or trialEpoched_info[ijk, 1] == 5 or trialEpoched_info[ijk, 1] == 6 or trialEpoched_info[ijk, 1] == 7 or trialEpoched_info[ijk, 1] == 8:
            # Check if there is any nan in the data
            if ~np.isnan(trialEpoched_arr[ijk, :, :]).any():
                left_trials.append(ijk)
        elif trialEpoched_info[ijk, 1] == 1 or trialEpoched_info[ijk, 1] == 2 or trialEpoched_info[ijk, 1] == 3 or trialEpoched_info[ijk, 1] == 9 or trialEpoched_info[ijk, 1] == 10:

            if ~np.isnan(trialEpoched_arr[ijk, :, :]).any():
                right_trials.append(ijk)

    # if len(tarlocGaze) != len(tarlocEpoched):
    #     print('Length of tarlocGaze and tarlocEpoched do not match')
    #     print('Length of tarlocGaze:', len(tarlocGaze))
    #     print('Length of tarlocEpoched:', len(tarlocEpoched))
    #     print()

    # if len(left_trials) != powspctrm_left.shape[0]:
    #     print('Length of left_trials and powspctrm_left do not match')
    #     print('Length of left_trials:', len(left_trials))
    #     print('Length of powspctrm_left:', powspctrm_left.shape[0])
    #     print()
    
    # if len(right_trials) != powspctrm_right.shape[0]:
    #     print('Length of right_trials and powspctrm_right do not match')
    #     print('Length of right_trials:', len(right_trials))
    #     print('Length of powspctrm_right:', powspctrm_right.shape[0])
    #     print()
    
    # # Print number of nan trials
    # print('      Left nan trials: ', np.sum(np.isnan(performance[left_trials])))
    # print('      Right nan trials: ', np.sum(np.isnan(performance[right_trials])))
    # print('      Total trials: ', len(left_trials) + len(right_trials))

    # Check if number of trials in powspctrm and trialInfo match


    # Indices of left and right trials with non-nan performance
    leftValidTrls = np.where(~np.isnan(performance[left_trials]))[0]
    rightValidTrls = np.where(~np.isnan(performance[right_trials]))[0]
    performance_left = performance[left_trials][leftValidTrls]
    performance_right = performance[right_trials][rightValidTrls]
    powspctrm_left = powspctrm_left[leftValidTrls]
    powspctrm_right = powspctrm_right[rightValidTrls]
    trialInfo_left = trialInfo_left[leftValidTrls]
    trialInfo_right = trialInfo_right[rightValidTrls]

    # Combine powspctrm, trialinfo and performance
    powspctrm_combined = np.concatenate((powspctrm_left, powspctrm_right), axis=0)
    trialInfo_combined = np.concatenate((trialInfo_left, trialInfo_right), axis=0)
    performance_combined = np.concatenate((performance_left, performance_right), axis=0)

    # Divide into high v low performance
    pctThresh = 40
    threshold_low, threshold_high = np.percentile(performance_combined, [pctThresh, 100 - pctThresh])
    print('Thresholds:', threshold_low, threshold_high)
    highPerfIdx = np.where(performance_combined > threshold_high)[0]
    lowPerfIdx = np.where(performance_combined <= threshold_low)[0]

    powspctrm_highPerf = powspctrm_combined[highPerfIdx]
    powspctrm_lowPerf = powspctrm_combined[lowPerfIdx]
    trialInfo_highPerf = trialInfo_combined[highPerfIdx]
    trialInfo_lowPerf = trialInfo_combined[lowPerfIdx]

    # Time Of Interest
    if freqband == 'broadband':
        time_idx = np.where((times >= -0.1) & (times <= 1.7))[0]
    else:
        time_idx = np.where((times >= -0.5) & (times <= 2))[0]
    times_crop = times[time_idx]

    # Extract features
    powspctrm_highPerf_ = extract_freqband(freqband, freqs, powspctrm_highPerf, basecorr)
    powspctrm_lowPerf_ = extract_freqband(freqband, freqs, powspctrm_lowPerf, basecorr)
    powspctrm_highPerf_ = powspctrm_highPerf_[:, :, time_idx]
    powspctrm_lowPerf_ = powspctrm_lowPerf_[:, :, time_idx]

    ######################################################## CLASSIFICATION ########################################################
    trlInfo_highPerf_ = trialInfo_highPerf[:, 1]
    trlInfo_lowPerf_ = trialInfo_lowPerf[:, 1]
    print('Training on high performance trials')
    accuracy_matrix_highPerf, f1score_matrix_highPerf = crossTemporalDecoding(powspctrm_highPerf_, trlInfo_highPerf_, classCats)
    print('Training on low performance trials')
    accuracy_matrix_lowPerf, f1score_matrix_lowPerf = crossTemporalDecoding(powspctrm_lowPerf_, trlInfo_lowPerf_, classCats)

    
    # Clear memory
    del TFRmat, powspctrm_highPerf_, powspctrm_lowPerf_, trialInfo_highPerf, trialInfo_lowPerf

    # accuracy_matrix_highPerf, f1score_matrix_highPerf, accuracy_matrix_lowPerf, f1score_matrix_lowPerf = None, None, None, None
    # times_crop = None
    return accuracy_matrix_highPerf, f1score_matrix_highPerf, accuracy_matrix_lowPerf, f1score_matrix_lowPerf, times_crop


def runMCCbyConfidence(subjID, ii_sess, epocStimLocked, TFRmat, v73=False, metric='ierr', classCats='quadrant', freqband='alpha'):
    ######################################################## EXTRACTION ########################################################
    # Extract ii_sess data
    if metric == 'ierr':
        performance = ii_sess['ii_sess']['i_sacc_err'][0, 0].T[0]
    elif metric == 'ferr':
        performance = ii_sess['ii_sess']['f_sacc_err'][0, 0].T[0]
    elif metric == 'irt':
        performance = ii_sess['ii_sess']['i_sacc_rt'][0, 0].T[0]
    elif metric == 'frt':
        performance = ii_sess['ii_sess']['f_sacc_rt'][0, 0].T[0]
    else:
        raise ValueError('Invalid performance metric, should be one of ierr, ferr, irt, frt')
    tarlocGaze = ii_sess['ii_sess']['tarlocCode'][0, 0].T[0]
    
    # Account for special cases
    rnum = ii_sess['ii_sess']['r_num'][0, 0].T[0]
    if subjID == 4: # Trial 1 from run10 is missing
        badTrials = [np.where(rnum == 10)[0][0]]
    elif subjID == 5: # Remove run 1 and 9
        badTrials = np.where((rnum == 1) | (rnum == 9))[0]
    elif subjID == 10: # Remove run 2 and 7
        badTrials = np.where((rnum == 2) | (rnum == 7))[0]
    elif subjID == 11: # Remove run 1, 3 and 8
        badTrials = np.where((rnum == 1) | (rnum == 3) | (rnum == 8))[0]
    elif subjID == 12: # Trial 1 from run 8 is missing
        badTrials = [np.where(rnum == 8)[0][0]]
    elif subjID == 13: # Trials 1, 2 from run 2 are missing
        badTrials = np.where(rnum == 2)[0][:2]
    elif subjID == 19: # Remove run 8 and 9
        badTrials = np.where((rnum == 8) | (rnum == 9))[0]
    elif subjID == 23: # Remove run 1
        badTrials = np.where(rnum == 1)[0]
    elif subjID == 25: # Remove run 8
        badTrials = np.where(rnum == 8)[0]
    elif subjID == 31: # Remove run 2; Also remove the last 17 trials from run 4
        badTrials = np.where(rnum == 2)[0]
        badTrials = np.concatenate((badTrials, np.where(rnum == 4)[0][-17:]))
    elif subjID == 32: # Remove run 2
        badTrials = np.where(rnum == 2)[0]
    else:
        badTrials = []
        
    # Remove the bad trials
    if len(badTrials) > 0:
        performance = np.delete(performance, badTrials)
        tarlocGaze = np.delete(tarlocGaze, badTrials)

    # Extract epocStimLocked data
    trialEpoched_info = epocStimLocked['epocStimLocked'][0][0][1]
    tarlocEpoched = trialEpoched_info[:, 1]
    trialEpoched = epocStimLocked['epocStimLocked'][0][0][4][0]
    trialsEpoched_list = [np.array(t) for t in trialEpoched]
    trialEpoched_arr = np.stack(trialsEpoched_list, axis=0)
    # Extract TFR data
    if v73:
        freqs = np.array(TFRmat['TFRleft_power']['freq']).T[0]
        times = np.array(TFRmat['TFRleft_power']['time']).T[0]
        trialInfo_left = np.array(TFRmat['TFRleft_power']['trialinfo']).T
        powspctrm_left = np.array(TFRmat['TFRleft_power']['powspctrm']).T
        trialInfo_right = np.array(TFRmat['TFRright_power']['trialinfo']).T
        powspctrm_right = np.array(TFRmat['TFRright_power']['powspctrm']).T
    else:
        # ch_label = TFRmat['TFRleft_power'][0][0][0]
        # dimord = TFRmat['TFRleft_power'][0][0][1]
        freqs = TFRmat['TFRleft_power'][0][0][2][0]
        times = TFRmat['TFRleft_power'][0][0][3][0]
        trialInfo_left = TFRmat['TFRleft_power'][0][0][5]
        powspctrm_left = TFRmat['TFRleft_power'][0][0][7]
        trialInfo_right = TFRmat['TFRright_power'][0][0][5]
        powspctrm_right = TFRmat['TFRright_power'][0][0][7]
    
    ######################################################## DIVIDING DATA ########################################################
    left_trials = []
    right_trials = []
    for ijk in range(trialEpoched_arr.shape[0]):
        if trialEpoched_info[ijk, 1] == 4 or trialEpoched_info[ijk, 1] == 5 or trialEpoched_info[ijk, 1] == 6 or trialEpoched_info[ijk, 1] == 7 or trialEpoched_info[ijk, 1] == 8:
            # Check if there is any nan in the data
            if ~np.isnan(trialEpoched_arr[ijk, :, :]).any():
                left_trials.append(ijk)
        elif trialEpoched_info[ijk, 1] == 1 or trialEpoched_info[ijk, 1] == 2 or trialEpoched_info[ijk, 1] == 3 or trialEpoched_info[ijk, 1] == 9 or trialEpoched_info[ijk, 1] == 10:

            if ~np.isnan(trialEpoched_arr[ijk, :, :]).any():
                right_trials.append(ijk)

    # Indices of left and right trials with non-nan performance
    leftValidTrls = np.where(~np.isnan(performance[left_trials]))[0]
    rightValidTrls = np.where(~np.isnan(performance[right_trials]))[0]
    performance_left = performance[left_trials][leftValidTrls]
    performance_right = performance[right_trials][rightValidTrls]
    powspctrm_left = powspctrm_left[leftValidTrls]
    powspctrm_right = powspctrm_right[rightValidTrls]
    trialInfo_left = trialInfo_left[leftValidTrls]
    trialInfo_right = trialInfo_right[rightValidTrls]

    # Combine powspctrm, trialinfo and performance
    powspctrm_combined = np.concatenate((powspctrm_left, powspctrm_right), axis=0)
    trialInfo_combined = np.concatenate((trialInfo_left, trialInfo_right), axis=0)
    performance_combined = np.concatenate((performance_left, performance_right), axis=0)

    # Time Of Interest
    if freqband == 'broadband':
        time_idx = np.where((times >= -0.1) & (times <= 1.7))[0]
    else:
        time_idx = np.where((times >= -0.5) & (times <= 2))[0]
        # time_idx = np.where((times >= -0.2) & (times <= 1.7))[0]
    times_crop = times[time_idx]

    # Extract features
    powspctrm_combined_ = extract_freqband(freqband, freqs, powspctrm_combined, True)
    powspctrm_combined_ = powspctrm_combined_[:, :, time_idx]

    trlInfo_ = trialInfo_combined[:, 1]
    auc_matrix, accuracy_matrix, f1score_matrix, y_guessed, y_guessed_proba, performance_valid = crossTemporalDecodingWithConfidence(powspctrm_combined_, 
                                                                                                                                    trlInfo_, 
                                                                                                                                    performance_combined, 
                                                                                                                                    classCats)

    return auc_matrix, accuracy_matrix, f1score_matrix, y_guessed, y_guessed_proba, performance_valid, times_crop