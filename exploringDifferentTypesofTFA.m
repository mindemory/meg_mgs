
load('NYUKIT_helmet.mat');


% cfg                               = [];
% cfg.layout                        = lay;
% cfg.avgoverrpt                    = 'yes';
% TFRleft_avg                       = ft_freqdescriptives(cfg, TFRleft_power);
% TFRright_avg                      = ft_freqdescriptives(cfg, TFRright_power);

cfg                               = [];
cfg.layout                        = lay;
cfg.avgoverrpt                    = 'yes';
TFRleft_avg                       = ft_freqdescriptives(cfg, TFR_fourier_left);
TFRright_avg                      = ft_freqdescriptives(cfg, TFR_fourier_right);

cfg                               = [];
cfg.parameter                     = 'powspctrm';  % Specify the parameter to operate on
% cfg.operation                     = '(10^(x1/10) - 10^(x2/10)) / (10^(x1/10) + 10^(x2/10))';
cfg.operation                     = '(x1-x2)/(x1+x2)';
TFRdiff                           = ft_math(cfg, TFRleft_avg, TFRright_avg);

cfg                               = [];
cfg.layout                        = lay;
cfg.xlim                          = [0.5 1.0];
cfg.ylim                          = [8 12];
cfg.zlim                          = [-0.1 0.1];
cfg.colormap                      = '*RdBu';
ft_topoplotTFR(cfg, TFRdiff)

%%
load('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/meg/sub-12_task-mgs_stimlocked.mat')
%%
% Load the data
epocThis                          = epocStimLocked;
epocThis.grad = grad;

cfg = [];
cfg.channel = epocThis.label;%(1:157);
cfg.method = 'triangulation';
cfg.grad = grad;
cfg.feedback = 'yes';
neighbors = ft_prepare_neighbours(cfg, epocThis);
% cfg = [];
% cfg.planarmethod = 'sincos';
% cfg.neighbours = neighbors;
% cfg.feedback = 'yes';
% epocThisPlanar = ft_megplanar(cfg, epocThis);

% cfg = [];
% epocThis = ft_combineplanar(cfg, epocThisPlanar);

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
% cfg                               = [];
% cfg.reref                         = 'yes';
% cfg.refchannel                    = 'all';
% cfg.refmethod                     = 'avg';
% epocLeft                          = ft_preprocessing(cfg, epocLeft);
% epocRight                         = ft_preprocessing(cfg, epocRight);



cfg                               = [];
cfg.lpfilter                      = 'yes';
cfg.lpfreq                        = 50;
epocLeft_filt                     = ft_preprocessing(cfg, epocLeft);
epocRight_filt                    = ft_preprocessing(cfg, epocRight);

cfg                               = [];
cfg.resamplefs                    = 50;
epocLeft_filt                     = ft_resampledata(cfg, epocLeft_filt);
epocRight_filt                    = ft_resampledata(cfg, epocRight_filt);

cfg                               = [];
cfg.bpfilter                      = 'yes';
cfg.bpfreq                        = [8 12];
epocLeft_filt                     = ft_preprocessing(cfg, epocLeft_filt);
epocRight_filt                    = ft_preprocessing(cfg, epocRight_filt);


for iTrial                        = 1:size(epocLeft_filt.trialinfo, 1)
        epocLeft_filt.trial{iTrial} ...
                                  = abs(hilbert(epocLeft_filt.trial{iTrial})).^2;
end
for iTrial                        = 1:size(epocRight_filt.trialinfo, 1)
        epocRight_filt.trial{iTrial} ...
                                  = abs(hilbert(epocRight_filt.trial{iTrial})).^2;
end

cfg = [];
cfg.combinemethod = 'sum';
epocLeft_filt = ft_combineplanar(cfg, epocLeft_filt);
epocRight_filt = ft_combineplanar(cfg, epocRight_filt);

cfg                               = [];
epocLeft_avg                      = ft_timelockanalysis(cfg, epocLeft_filt);
epocRight_avg                     = ft_timelockanalysis(cfg, epocRight_filt);

% cfg = [];
% cfg.planarmethod = 'sincos';
% cfg.channel = epocLeft.grad.label;
% cfg.neighbours = neighbors;
% % cfg.feedback = 'yes';
% epocLeft_avgPlanar = ft_megplanar(cfg, epocLeft_avg);
% epocRight_avgPlanar = ft_megplanar(cfg, epocRight_avg);
% cfg = [];
% cfg.combinemethod = 'sum';
% cfg.combinegrad = 'yes';
% epocLeft_avg = ft_combineplanar(cfg, epocLeft_avgPlanar);
% epocRight_avg = ft_combineplanar(cfg, epocRight_avgPlanar);

cfg                               = [];
cfg.parameter                     = 'avg';  % Specify the parameter to operate on
cfg.operation                     = '(x1-x2)/(x1+x2)';
epocDiff                         = ft_math(cfg, epocLeft_avg, epocRight_avg);


cfg                               = [];
cfg.layout                        = lay;
cfg.xlim                          = [0.5 1.0];
cfg.ylim                          = [8 12];
cfg.zlim                          = [-0.1 0.1];
cfg.colormap                      = '*RdBu';
ft_topoplotTFR(cfg, epocDiff)

%%

% Load the data
epocThis                          = epocStimLocked;
epocThis_filt = epocThis;
% cfg                               = [];
% cfg.lpfilter                      = 'yes';
% cfg.lpfreq                        = 50;
% epocThis_filt                     = ft_preprocessing(cfg, epocThis);
% 
% cfg                               = [];
% cfg.resamplefs                    = 50;
% epocThis_filt                     = ft_resampledata(cfg, epocThis_filt);

cfg                               = [];
cfg.bpfilter                      = 'yes';
cfg.bpfreq                        = [8 12];
epocThis_filt                     = ft_preprocessing(cfg, epocThis_filt);


for iTrial                        = 1:size(epocThis_filt.trialinfo, 1)
        epocThis_filt.trial{iTrial} ...
                                  = abs(hilbert(epocThis_filt.trial{iTrial})).^2;
end


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
epocLeft_filt                     = ft_selectdata(cfg, epocThis_filt);

% Select Right trials
cfg                               = [];
cfg.trials                        = valid_trialsRight;
epocRight_filt                    = ft_selectdata(cfg, epocThis_filt);

cfg                               = [];
epocLeft_avg                      = ft_timelockanalysis(cfg, epocLeft_filt);
epocRight_avg                     = ft_timelockanalysis(cfg, epocRight_filt);

cfg                               = [];
cfg.parameter                     = 'avg';  % Specify the parameter to operate on
cfg.operation                     = '(x1-x2)/(x1+x2)';
epocDiff                         = ft_math(cfg, epocLeft_avg, epocRight_avg);


cfg                               = [];
cfg.layout                        = lay;
cfg.xlim                          = [0.5 1.0];
cfg.ylim                          = [8 12];
cfg.zlim                          = [-0.1 0.1];
cfg.colormap                      = '*RdBu';
ft_topoplotTFR(cfg, epocDiff)

%%

% Load the data
epocThis                          = epocStimLocked;
epocThis_filt = epocThis;
% cfg                               = [];
% cfg.lpfilter                      = 'yes';
% cfg.lpfreq                        = 50;
% epocThis_filt                     = ft_preprocessing(cfg, epocThis);
% 
% cfg                               = [];
% cfg.resamplefs                    = 50;
% epocThis_filt                     = ft_resampledata(cfg, epocThis_filt);

cfg                               = [];
cfg.method                        = 'mtmfft';
cfg.output                        = 'fourier';
cfg.keeptrials                    = 'yes';
cfg.tapsmofrq                     = 1;
cfg.foi                           = 10;
freqThis                          = ft_freqanalysis(cfg, epocThis_filt);

% cfg                               = [];
% cfg.bpfilter                      = 'yes';
% cfg.bpfreq                        = [8 12];
% epocThis_filt                     = ft_preprocessing(cfg, epocThis_filt);
% 
% 
% for iTrial                        = 1:size(epocThis_filt.trialinfo, 1)
%         epocThis_filt.trial{iTrial} ...
%                                   = abs(hilbert(epocThis_filt.trial{iTrial})).^2;
% end


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
freqLeft                          = ft_freqdescriptives(cfg, freqThis);

% Select Right trials
cfg                               = [];
cfg.trials                        = valid_trialsRight;
freqRight                         = ft_freqdescriptives(cfg, freqThis);

% cfg                               = [];
% epocLeft_avg                      = ft_timelockanalysis(cfg, epocLeft_filt);
% epocRight_avg                     = ft_timelockanalysis(cfg, epocRight_filt);

cfg                               = [];
cfg.parameter                     = 'powspctrm';  % Specify the parameter to operate on
cfg.operation                     = '(x1-x2)/(x1+x2)';
freqDiff                          = ft_math(cfg, freqLeft, freqRight);


cfg                               = [];
cfg.layout                        = lay;
cfg.xlim                          = [0.5 1.0];
cfg.ylim                          = [8 12];
cfg.zlim                          = [-0.1 0.1];
cfg.colormap                      = '*RdBu';
ft_topoplotTFR(cfg, freqDiff)



%% Exploring cross spectral densities

epocCombined = ft_appenddata([], epocLeft, epocRight);




cfg                     = [];
cfg.method              = 'mtmfft';
cfg.output              = 'powandcsd';
cfg.tapsmofrq           = 1.5;
cfg.foilim              = [6.0 6.0];
cfg.latency             = [0.0 0.3];
freqLeft                = ft_freqanalysis(cfg, epocLeft);
freqRight               = ft_freqanalysis(cfg, epocRight);
freqCombined            = ft_freqanalysis(cfg, epocCombined);
% cfg                     = [];
% cfg.method              = 'mtmconvol';
% cfg.output              = 'powandcsd';
% cfg.taper               = 'hanning';
% cfg.keeptrials          = 'yes';
% cfg.foi                 = 6:2:14;
% cfg.t_ftimwin           = 5./cfg.foi;
% % cfg.toi                 = linspace(min(epocThis.time{1}), max(epocThis.time{1}), 200);
% cfg.toi                 = 0:0.2:1.5;
% freqThis                = ft_freqanalysis(cfg, epocThis);