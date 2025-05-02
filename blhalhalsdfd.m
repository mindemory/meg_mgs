load('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/meg/sub-12_task-mgs_stimlocked_lineremoved.mat')
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

cfg                               = [];
cfg.lpfreq                        = 100;
cfg.lpfilter                      = 'yes';
epocLeft_filtered                 = ft_preprocessing(cfg, epocLeft);
epocRight_filtered                = ft_preprocessing(cfg, epocRight);
cfg                               = [];
cfg.resamplefs                    = 100;
cfg.resamplemethod                = 'resample';
epocLeft_filtered                 = ft_resampledata(cfg, epocLeft_filtered);
epocRight_filtered                = ft_resampledata(cfg, epocRight_filtered);
% 
% %%%%% Average reference >> absolutely not
% cfg                               = [];
% cfg.reref                         = 'yes';
% cfg.refchannel                    = 'all';
% cfg.refmethod                     = 'avg';
% epocLeft                          = ft_preprocessing(cfg, epocLeft);
% epocRight                         = ft_preprocessing(cfg, epocRight);
epocCombined                      = ft_appenddata([], epocLeft, epocRight);
epocCombined_filtered             = ft_appenddata([], epocLeft_filtered, epocRight_filtered);


cfg                 = [];
cfg.previous.bpfilter ...
                    = 'yes';
cfg.bpfreq          = [18 27];
epocLeft_filt       = ft_preprocessing(cfg, epocLeft_filtered);
epocRight_filt      = ft_preprocessing(cfg, epocRight_filtered);

% Project data to source space


hilbert_power = @(x) (abs(hilbert(x.')).^2).'; % Transpose handle for row-wise operation

sourcedataLeft.trial = cellfun(hilbert_power, epocLeft_filt.trial, ...
    'UniformOutput', false);
sourcedataRight.trial = cellfun(hilbert_power, epocRight_filt.trial, ...
    'UniformOutput', false);
save(sourceSpacePower, 'sourcedataLeft','sourcedataRig