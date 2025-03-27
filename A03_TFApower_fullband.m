
clear; close all; clc;
warning('off', 'all');
%% Initialization
% p.subjID          = subjID;
% [p]               = initialization(p, 'ecog');

addpath('/d/DATD/hyper/software/fieldtrip-20220104/');
ft_defaults;

% subjID = 13; % change this to run a different subject

% subList                                   = [1 2 3 4 5 6 7 8 9 10 11 12 13 15 ...
%                                              16 17 18 19 20 22 23 24 25 26 27 ...
%                                              28 29 30 31 32];
subList                                   = [1 2 3 4];

for sIdx                                  = 1:length(subList)

    disp(['Running ' num2str(sIdx) ' of ' num2str(length(subList)) ' subjects.'])
    subjID                                = subList(sIdx);

    % Initalizing variables
    bidsRoot                              = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
    taskName                              = 'mgs';


    derivativesRoot                       = [bidsRoot filesep 'derivatives/sub-' num2str(subjID, '%02d') '/meg'];
    subName                               = ['sub-' num2str(subjID, '%02d')];
    megRoot                               = [bidsRoot filesep subName filesep 'meg'];
    stimRoot                              = [bidsRoot filesep subName filesep 'stimfiles'];
    fNameRoot                             = [subName '_task-' taskName];

    % if sIdx == subList(1)
    %     rawdata_path = [derivativesRoot filesep fNameRoot '_run-01_raw.mat'];
    %     load(rawdata_path, 'lay');
    % end
    load('NYUKIT_helmet.mat');

    stimLocked_fpath                      = [derivativesRoot filesep fNameRoot '_stimlocked.mat'];
    TFR_fpath                             = [derivativesRoot filesep fNameRoot '_TFR_highfreq_noiseremoved.mat'];
    if ~exist(TFR_fpath, 'file')
        disp('TFR does not exist, creating it.')
        load(stimLocked_fpath);
    
        % Load the data
        epocThis                          = epocStimLocked;
    
        % Logical mask to find epochs matching your criteria
        trial_criteria_left               = (epocThis.trialinfo(:,2) == 4) | ...
                                            (epocThis.trialinfo(:,2) == 5) | ...
                                            (epocThis.trialinfo(:,2) == 6) | ...
                                            (epocThis.trialinfo(:,2) == 7) | ...
                                            (epocThis.trialinfo(:,2) == 8);
    
        trial_criteria_right              = (epocThis.trialinfo(:,2) == 1) | ...
                                            (epocThis.trialinfo(:,2) == 2) | ...
                                            (epocThis.trialinfo(:,2) == 3) | ...
                                            (epocThis.trialinfo(:,2) == 9) | ...
                                            (epocThis.trialinfo(:,2) == 10);
    
        % Logical mask to find epochs without NaNs
        has_no_nans                       = cellfun(@(x) ~any(isnan(x(:))), epocThis.trial)';
    
        % Combine both criteria
        valid_trialsLeft                  = find(trial_criteria_left & has_no_nans);
        valid_trialsRight                 = find(trial_criteria_right & has_no_nans);
    
        % Select Left trials
        cfg                               = [];
        cfg.trials                        = valid_trialsLeft;
        epocLeft                          = ft_selectdata(cfg, epocThis);
    
        % Select Right trials
        cfg                               = [];
        cfg.trials                        = valid_trialsRight;
        epocRight                         = ft_selectdata(cfg, epocThis);
    
        %%%%% Average reference
        cfg                               = [];
        cfg.reref                         = 'yes';
        cfg.refchannel                    = 'all';
        cfg.refmethod                     = 'avg';
        epocLeft                          = ft_preprocessing(cfg, epocLeft);
        epocRight                         = ft_preprocessing(cfg, epocRight);
    
        powHigh_left                      = computePower_highFreq(epocLeft);
        powHigh_right                     = computePower_highFreq(epocRight);

        save(TFR_fpath, 'powHigh_left', 'powHigh_right', '-v7.3')
    else
        disp('Loading existing TFR. If this is not desired, delete the existing file.')
        load(TFR_fpath);
    end
    
end