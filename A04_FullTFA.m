function A04_FullTFA(subjID, desiredOutput)
clearvars -except subjID desiredOutput; close all; clc;
warning('off', 'all');
%% Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20220104/');
ft_defaults;

% Initalizing variables
bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
taskName = 'mgs';


derivativesRoot = [bidsRoot filesep 'derivatives/sub-' num2str(subjID, '%02d') '/meg'];
subName = ['sub-' num2str(subjID, '%02d')];
megRoot = [bidsRoot filesep subName filesep 'meg'];
stimRoot = [bidsRoot filesep subName filesep 'stimfiles'];
fNameRoot = [subName '_task-' taskName];

% Import layout
rawdata_path = [derivativesRoot filesep fNameRoot '_run-01_raw.mat'];
load(rawdata_path, 'lay');

stimLocked_fpath = [derivativesRoot filesep fNameRoot '_stimlocked.mat'];
if strcmp(desiredOutput, 'lowFreqPhase')
    TFR_fpath = [derivativesRoot filesep fNameRoot '_lowFreqPhase.mat'];
elseif strcmp(desiredOutput, 'highFreqPow')
    TFR_fpath = [derivativesRoot filesep fNameRoot '_highFreqPow.mat'];
end
if ~exist(TFR_fpath, 'file')
    disp('TFR does not exist, creating it.')
    load(stimLocked_fpath);

    % Load the data
    epocThis = epocStimLocked;

    % Logical mask to find epochs matching your criteria
    trial_criteria_left = (epocThis.trialinfo(:,2) == 4) | ...
        (epocThis.trialinfo(:,2) == 5) | ...
        (epocThis.trialinfo(:,2) == 6) | ...
        (epocThis.trialinfo(:,2) == 7) | ...
        (epocThis.trialinfo(:,2) == 8);

    trial_criteria_right = (epocThis.trialinfo(:,2) == 1) | ...
        (epocThis.trialinfo(:,2) == 2) | ...
        (epocThis.trialinfo(:,2) == 3) | ...
        (epocThis.trialinfo(:,2) == 9) | ...
        (epocThis.trialinfo(:,2) == 10);

    % Logical mask to find epochs without NaNs
    has_no_nans = cellfun(@(x) ~any(isnan(x(:))), epocThis.trial)';

    % Combine both criteria
    valid_trialsLeft = find(trial_criteria_left & has_no_nans);
    valid_trialsRight = find(trial_criteria_right & has_no_nans);

    % Select Left trials
    cfg = [];
    cfg.trials = valid_trialsLeft;
    epocLeft = ft_selectdata(cfg, epocThis);

    % Select Right trials
    cfg = [];
    cfg.trials = valid_trialsRight;
    epocRight = ft_selectdata(cfg, epocThis);

    %%%%% Average reference
    cfg = [];
    cfg.reref = 'yes';
    cfg.refchannel = 'all';
    cfg.refmethod = 'avg';
    epocLeft = ft_preprocessing(cfg, epocLeft);
    epocRight = ft_preprocessing(cfg, epocRight);

    TFRleft_power       = compute_FullPow_phase(epocLeft, desiredOutput);
    TFRright_power       = compute_FullPow_phase(epocRight, desiredOutput);

    save(TFR_fpath, 'TFRleft_power', 'TFRright_power', '-v7.3')
else
    disp('Loading existing TFR. If this is not desired, delete the existing file.')
    load(TFR_fpath);
end


end

