clear; close all; clc;
%% Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/'); % 2022 doesn't work well for sourerecon
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);
% Load forward model for Clay
forwardmodelfPath                     = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/anatomy/sub-12_task-mgs_forwardmodel-nativespace-10mm.mat';
load(forwardmodelfPath);

subList                               = [ 1,  2,  3,  4,  5,  6,  7,  9, 10, 12, 13, 15, ...
                                         17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
for sIdx                              = 1:length(subList)
    subjID                            = subList(sIdx);
    disp(['Running ' num2str(subjID, '%02d')])
    subDerivativesRoot                = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-' num2str(subjID, '%02d') ...
                                         '/meg/sub-' num2str(subjID, '%02d') '_task-mgs_'];
    sourceSpacefPath                  = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/allsubs_clayspace/sub-' ...
                                         num2str(subjID, '%02d') '_task-mgs_source-clayspace.mat']; 
    load([subDerivativesRoot 'stimlocked_lineremoved.mat'])
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
    
    % Keep a flag handy for left and right trials
    epocLeft.trialinfo(:,6)           = 1;
    epocRight.trialinfo(:,6)          = 2;
    epocCombined                      = ft_appenddata([], epocLeft, epocRight);
    
    % Compute timelocked data with covariance
    cfg                               = [];
    cfg.covariance                    = 'yes';
    cfg.covariancewindow              = 'all';
    cfg.keeptrials                    = 'no';
    timelockedCombined                = ft_timelockanalysis(cfg, epocCombined);
    
    % Compute source model for this subject    
    cfg                               = [];
    cfg.method                        = 'lcmv';
    cfg.sourcemodel                   = grid;
    cfg.headmodel                     = singleShellHeadModel;
    cfg.grad                          = gradData;
    cfg.keepleadfield                 = 'yes';
    cfg.lcmv.keepfilter               = 'yes';
    cfg.lcmv.fixedori                 = 'yes';
    cfg.lcmv.lambda                   = '5%';  % must be '1%' not 1
    source                            = ft_sourceanalysis(cfg, timelockedCombined);
    
    inside_pos                        = find(source.inside);
    
    
    %% Extract all filters for inside voxels
    W_meg                             = cell2mat(cellfun(@(x) x, source.avg.filter(inside_pos), ...
                                                 'UniformOutput', false));
    
    sourcedata                        = [];
    sourcedata.label                  = cell(numel(inside_pos), 1);
    for i                             = 1:numel(inside_pos)
        sourcedata.label{i}           = sprintf('S_%d', inside_pos(i));
    end
    sourcedata.trial                  = cellfun(@(x) W_meg * x, epocCombined.trial, 'UniformOutput', false);
    sourcedata.time                   = epocCombined.time;
    sourcedata.trialinfo              = epocCombined.trialinfo;
    
    % Low-pass filter
    cfg                               = [];
    cfg.lpfilter                      = 'yes';
    cfg.lpfreq                        = 55;  % Or 70, safely below 75 Hz (Nyquist of 150 Hz)
    sourcedata                        = ft_preprocessing(cfg, sourcedata);
    
    % Downsample
    cfg                               = [];
    cfg.resamplefs                    = 75;
    cfg.detrend                       = 'no';
    sourcedata                        = ft_resampledata(cfg, sourcedata);
    
    % Band-pass filter beta-band activity
    cfg                               = [];
    cfg.bpfilter                      = 'yes';
    cfg.bpfreq                        = [18 25];
    sourcedata                        = ft_preprocessing(cfg, sourcedata);
    hilbert_compute                   = @(x) hilbert(x')'; 
    sourcedata.trial                  = cellfun(hilbert_compute, sourcedata.trial, 'UniformOutput', false);
    
    save(sourceSpacefPath, 'source', 'sourcedata', '-v7.3');
    clearvars epocThis epocStimLocked epocLeft epocRight epocCombined timelockedCombined source sourcedata
end
