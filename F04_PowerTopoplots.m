clear; close all; clc;
warning('off', 'all');
%% Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/'); 
load('NYUKIT_helmet.mat')
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
%%
% freqband                            
subList                             = [ 1,  2,  3,  4,  5,  6,  7,  9, 10, 12, 13, 15, ...
                                       17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
% subList                             = [1 2 3 4 5];

tOnset                              = [-0.5 0.1 0.5 1  ];
tOffset                             = [ 0   0.3 1   1.5];
titleDict                           = {'Fixation', 'StimOn', 'Early Delay', 'Late Delay'};

tic
for sIdx                            = 1:length(subList)

    disp(['Running ' num2str(sIdx) ' of ' num2str(length(subList)) ' subjects.'])
    subjID                          = subList(sIdx);

    % Initalizing variables
    bidsRoot                        = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
    taskName                        = 'mgs';


    derivativesRoot                 = [bidsRoot filesep 'derivatives/sub-' ...
                                       num2str(subjID, '%02d') '/meg'];
    subName                         = ['sub-' num2str(subjID, '%02d')];
    megRoot                         = [bidsRoot filesep subName filesep 'meg'];
    stimRoot                        = [bidsRoot filesep subName filesep 'stimfiles'];
    fNameRoot                       = [subName '_task-' taskName];

    stimLocked_fpath                = [derivativesRoot filesep fNameRoot ...
                                       '_stimlocked_lineremoved.mat'];
    load(stimLocked_fpath);

    % Load the data
    epocThis                        = epocStimLocked;

    % Logical mask to find epochs matching your criteria
    trial_criteria_left             = (epocThis.trialinfo(:,2) == 4) | ...
                                      (epocThis.trialinfo(:,2) == 5) | ...
                                      (epocThis.trialinfo(:,2) == 6) | ...
                                      (epocThis.trialinfo(:,2) == 7) | ...
                                      (epocThis.trialinfo(:,2) == 8);
    trial_criteria_right            = (epocThis.trialinfo(:,2) == 1) | ...
                                      (epocThis.trialinfo(:,2) == 2) | ...
                                      (epocThis.trialinfo(:,2) == 3) | ...
                                      (epocThis.trialinfo(:,2) == 9) | ...
                                      (epocThis.trialinfo(:,2) == 10);
    has_no_nans                     = cellfun(@(x) ~any(isnan(x(:))), epocThis.trial)';
    % Combine both criteria
    valid_trialsLeft                = find(trial_criteria_left & has_no_nans);
    valid_trialsRight               = find(trial_criteria_right & has_no_nans);

    % Select Left & Right trials
    cfg                             = [];
    cfg.trials                      = valid_trialsLeft;
    epocLeft                        = ft_selectdata(cfg, epocThis);
    cfg.trials                      = valid_trialsRight;
    epocRight                       = ft_selectdata(cfg, epocThis);

    %%%%% Average reference
    cfg                             = [];
    cfg.reref                       = 'yes';
    cfg.refchannel                  = 'all';
    cfg.refmethod                   = 'avg';
    epocLeft                        = ft_preprocessing(cfg, epocLeft);
    epocRight                       = ft_preprocessing(cfg, epocRight);
    
  
    % Compute power in alpha band
    % cfg                               = [];
    % cfg.lpfilter                      = 'yes';
    % cfg.lpfreq                        = 50;
    % epocLeft_filt                     = ft_preprocessing(cfg, epocLeft);
    % epocRight_filt                    = ft_preprocessing(cfg, epocRight);
    % 
    % cfg                               = [];
    % cfg.resamplefs                    = 50;
    % epocLeft_filt                     = ft_resampledata(cfg, epocLeft_filt);
    % epocRight_filt                    = ft_resampledata(cfg, epocRight_filt);
    
    cfg                               = [];
    cfg.bpfilter                      = 'yes';
    cfg.bpfreq                        = [5 8];
    epocLeft_filt                     = ft_preprocessing(cfg, epocLeft);
    epocRight_filt                    = ft_preprocessing(cfg, epocRight);
    
    
    for iTrial                        = 1:size(epocLeft_filt.trialinfo, 1)
            epocLeft_filt.trial{iTrial} ...
                                      = abs(hilbert(epocLeft_filt.trial{iTrial})).^2;
    end
    for iTrial                        = 1:size(epocRight_filt.trialinfo, 1)
            epocRight_filt.trial{iTrial} ...
                                      = abs(hilbert(epocRight_filt.trial{iTrial})).^2;
    end
    
    cfg                               = [];
    epocLeft_avg                      = ft_timelockanalysis(cfg, epocLeft_filt);
    epocRight_avg                     = ft_timelockanalysis(cfg, epocRight_filt);
    
    cfg                               = [];
    cfg.parameter                     = 'avg';  % Specify the parameter to operate on
    cfg.operation                     = '(x1-x2)/(x1+x2)';
    epocDiff                          = ft_math(cfg, epocLeft_avg, epocRight_avg);

    if sIdx                           == 1
        TFRdiff                       = epocDiff;
    else
        TFRdiff.avg                   = cat(3, TFRdiff.avg, epocDiff.avg);
    end
end

toc
TFRdiff.avg                         = mean(TFRdiff.avg, 3, 'omitnan');
%%

figure;
for tt                              = 1:length(tOnset)
    subplot(2, 2, tt)
    cfg                             = [];
    cfg.figure                      = 'gcf';
    cfg.layout                      = lay;
    cfg.colormap                    = '*RdBu';
    cfg.xlim                        = [tOnset(tt) tOffset(tt)];
    cfg.zlim                        = [-.04 0.04];
    
    ft_topoplotTFR(cfg, TFRdiff)
end