%% Written by Mrugank (April 7, 2025):
% The code compute time-frequency power for individual subjects over each
% trial using compute_TFR function which itself computes a complete fourier
% representation at each time-point and frequency bin and then computes
% power and returns it in dB scale.

clear; close all; clc;
warning('off', 'all');
%% Initialization
% Using Fieldtrip version 20250318
addpath('/d/DATD/hyper/software/fieldtrip-20250318/'); 
ft_defaults;

% List of subjects to run this on
% subList                   = [ 1,  2,  3,  4,  5,  6,  7,  9, 10, 12, 13, 15, ...
%                              17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
subList                   = [ 10, 12, 13, 15, ...
                             17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
for sIdx                  = 1:length(subList)

    disp(['Running ' num2str(sIdx) ' of ' num2str(length(subList)) ' subjects.'])
    subjID                = subList(sIdx);

    % Initalizing Paths
    bidsRoot              = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
    taskName              = 'mgs';
    derivativesRoot       = [bidsRoot filesep 'derivatives/sub-' num2str(subjID, '%02d') '/meg'];
    subName               = ['sub-' num2str(subjID, '%02d')];
    megRoot               = [bidsRoot filesep subName filesep 'meg'];
    stimRoot              = [bidsRoot filesep subName filesep 'stimfiles'];
    fNameRoot             = [subName '_task-' taskName];
    stimLocked_fpath      = [derivativesRoot filesep fNameRoot '_stimlocked_lineremoved.mat'];
    TFR_fpath             = [derivativesRoot filesep fNameRoot '_TFRfull_evoked_lineremoved.mat'];

    % Compute power only if not already computed
    if ~exist(TFR_fpath, 'file')
        disp('TFR does not exist, creating it.')
        load(stimLocked_fpath);
    
        % Load the data
        epocThis          = epocStimLocked;
    
        % Logical mask to find epochs matching your criteria
        trial_criteria_left ...
                          = (epocThis.trialinfo(:,2) == 4) | ...
                            (epocThis.trialinfo(:,2) == 5) | ...
                            (epocThis.trialinfo(:,2) == 6) | ...
                            (epocThis.trialinfo(:,2) == 7) | ...
                            (epocThis.trialinfo(:,2) == 8);
    
        trial_criteria_right ...
                          = (epocThis.trialinfo(:,2) == 1) | ...
                            (epocThis.trialinfo(:,2) == 2) | ...
                            (epocThis.trialinfo(:,2) == 3) | ...
                            (epocThis.trialinfo(:,2) == 9) | ...
                            (epocThis.trialinfo(:,2) == 10);
        % Logical mask to find epochs without NaNs
        has_no_nans       = cellfun(@(x) ~any(isnan(x(:))), epocThis.trial)';
    
        % Combine both criteria
        valid_trialsLeft  = find(trial_criteria_left & has_no_nans);
        valid_trialsRight = find(trial_criteria_right & has_no_nans);
    
        % Select Left and Right trials separately
        cfg               = [];
        cfg.trials        = valid_trialsLeft;
        epocLeft          = ft_selectdata(cfg, epocThis);
        cfg.trials        = valid_trialsRight;
        epocRight         = ft_selectdata(cfg, epocThis);
    
        % Average reference
        cfg               = [];
        cfg.reref         = 'yes';
        cfg.refchannel    = 'all';
        cfg.refmethod     = 'avg';
        epocLeft          = ft_preprocessing(cfg, epocLeft);
        epocRight         = ft_preprocessing(cfg, epocRight);
        
        % Compute power separately for left and right trials
        TFR_fourier_left            = compute_fullFourier(epocLeft);
        TFR_fourier_right           = compute_fullFourier(epocRight);

        save(TFR_fpath, 'TFR_fourier_left', 'TFR_fourier_right', '-v7.3')
    else
        disp('Loading existing TFR. If this is not desired, delete the existing file.')
        load(TFR_fpath);
    end
end