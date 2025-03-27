clear; close all; clc;
warning('off', 'all');
%% Initialization
% p.subjID          = subjID;
% [p]               = initialization(p, 'ecog');

% addpath('/d/DATD/hyper/software/fieldtrip-20220104/');
% ft_defaults;

% subjID = 13; % change this to run a different subject

% subList = [1 2 3 4 5 6 7 8 9 10 11 12 13 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32];
subList = [12];

TFRleft = {};
TFRright = {};
% for subjID = subList
for sIdx = 1:length(subList)

    disp(['Running ' num2str(sIdx) ' of ' num2str(length(subList)) ' subjects.'])
    subjID = subList(sIdx);

    % Initalizing variables
    bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
    taskName = 'mgs';


    derivativesRoot = [bidsRoot filesep 'derivatives/sub-' num2str(subjID, '%02d') '/meg'];
    subName = ['sub-' num2str(subjID, '%02d')];
    megRoot = [bidsRoot filesep subName filesep 'meg'];
    stimRoot = [bidsRoot filesep subName filesep 'stimfiles'];
    fNameRoot = [subName '_task-' taskName];

    % if sIdx == subList(1)
    %     rawdata_path = [derivativesRoot filesep fNameRoot '_run-01_raw.mat'];
    %     load(rawdata_path, 'lay');
    % end
    load('NYUKIT_helmet.mat');

    stimLocked_fpath = [derivativesRoot filesep fNameRoot '_stimlocked.mat'];
    % TFR_fpath = [derivativesRoot filesep fNameRoot '_TFR_evoked.mat'];
    % if ~exist(TFR_fpath, 'file')
    % disp('TFR does not exist, creating it.')
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

    TFRleft_phase = compute_phase(epocLeft);
    TFRright_phase = compute_phase(epocRight);

    TFR_power_left = compute_TFRs(epocLeft);
    TFR_power_right = compute_TFRs(epocRight);

    TFR_phase_fpath = [derivativesRoot filesep fNameRoot '_TFR_phase.mat'];
    save(TFR_phase_fpath, 'TFRleft_phase', 'TFRright_phase')
end

%% Visualizing power
cfg                           = [];
cfg.avgoverrpt                = 'yes';  % This averages over the rpttap dimension (trials)
powLeftAvg                    = ft_freqdescriptives(cfg, TFR_power_left);
powRightAvg                   = ft_freqdescriptives(cfg, TFR_power_right);


cfg                           = [];
cfg.parameter                 = 'powspctrm';  % Specify the parameter to operate on
cfg.operation                 = '(10^(x1/10) - 10^(x2/10)) / (10^(x1/10) + 10^(x2/10))';
TFRdiff                       = ft_math(cfg, powLeftAvg, powRightAvg);


cfg                           = [];
cfg.xlim                      = [-0.5 1.7];
cfg.ylim                      = [4 40];
cfg.layout                    = lay;
cfg.colormap                  = '*RdBu';
ft_multiplotTFR(cfg, TFRdiff)

%% Visualizing phase
TFR_ITPC_left                = TFRleft_phase;
TFR_ITPC_left.itpc = abs(squeeze(mean(exp(1i * TFRleft_phase.phaseangle), 1, 'omitnan')));
TFR_ITPC_left = rmfield(TFR_ITPC_left, 'phaseangle');
TFR_ITPC_left.dimord = 'chan_freq_time';
TFR_ITPC_right = TFRright_phase;
TFR_ITPC_right.itpc = abs(squeeze(mean(exp(1i * TFRright_phase.phaseangle), 1, 'omitnan')));
TFR_ITPC_right = rmfield(TFR_ITPC_right, 'phaseangle');
TFR_ITPC_right.dimord = 'chan_freq_time';
cfg = [];
cfg.parameter = 'itpc';
cfg.operation = '(x1-x2)';
TFR_ITPC_diff = ft_math(cfg, TFR_ITPC_left, TFR_ITPC_right);

cfg = [];
cfg.layout    = lay;       % Specify your layout file
cfg.colormap  = '*RdBu';   % Use a diverging colormap for differences
cfg.parameter = 'itpc';    % Specify that we are plotting ITPC values
cfg.xlim = [-0.1 1.7];
cfg.ylim = [4 40];
cfg.zlim = [-0.1 0.1];
ft_multiplotTFR(cfg, TFR_ITPC_diff);


%% Phase-locking value PLV
right_sensors = {'AG001', 'AG002', 'AG007', 'AG008', 'AG020', 'AG022', 'AG023', ...
                 'AG024', 'AG034', 'AG036', 'AG050', 'AG055', 'AG065', 'AG066', ...
                 'AG098', 'AG103'};
left_sensors  = {'AG013', 'AG014', 'AG015', 'AG016', 'AG023', 'AG025', 'AG026', ...
                 'AG027', 'AG028', 'AG041', 'AG042', 'AG043', 'AG059', 'AG060', ...
                 'AG066', 'AG092'};

seed_sensors = right_sensors;
TFRphase_chosen = TFRright_phase;
seed_indices = find(ismember(TFRphase_chosen.label, seed_sensors));
seed_phase_left = TFRphase_chosen.phaseangle(:, seed_indices, :, :);
% seed_phase_right = TFR_phase_right.phaseangle(:, seed_indices, :, :);
seed_phase_mean_left = angle(squeeze(mean(exp(1i * seed_phase_left), 2, 'omitnan'))); % [frequencies x time x trials]
% seed_phase_mean_right = angle(squeeze(mean(exp(1i * seed_phase_right), 2, 'omitnan')));

plv_left = zeros(numel(TFRphase_chosen.label), size(TFRphase_chosen.phaseangle, 3), size(TFRphase_chosen.phaseangle, 4)); % chan * freq * time
for ch = 1:numel(TFRphase_chosen.label)
    currChanPhase = squeeze(TFRphase_chosen.phaseangle(:, ch, :, :));
    phase_diff = currChanPhase - seed_phase_mean_left;
    plv_left(ch, :, :) = abs(squeeze(mean(exp(1i * phase_diff), 1, 'omitnan')));
end

TFR_plv_left = TFRphase_chosen;
TFR_plv_left.powspctrm = plv_left;
TFR_plv_left = rmfield(TFR_plv_left, 'phaseangle');
TFR_plv_left.dimord = 'chan_freq_time';
% cfg = [];
% cfg.layout = lay;
% cfg.xlim = [-0.5 1.7];
% cfg.ylim = [4 40];
% cfg.zlim = [0 0.3];
% cfg.parameter = 'powspctrm';
% cfg.colormap = '*RdBu';
% ft_multiplotTFR(cfg, TFR_plv_left)

figure();
tOnset = [-0.5 0.1 0.5 1];
tOffset = [0 0.3 1 1.5];
titleDict = {'Fixation', 'StimOn', 'Early Delay', 'Late Delay'};
freqband = 'beta';
sgtitle(freqband)
for idx = 1:length(tOnset)
    subplot(2, 2, idx)
    cfg = [];
    cfg.layout = lay;
    cfg.figure = 'gcf';
    cfg.xlim = [tOnset(idx) tOffset(idx)];
    if strcmp(freqband, 'theta')
        cfg.ylim = [5 8];
    elseif strcmp(freqband, 'alpha')
        cfg.ylim = [8 12];
    elseif strcmp(freqband, 'beta')
        cfg.ylim = [14 25];
    end
    cfg.zlim = [0 0.2];
    cfg.parameter = 'powspctrm';
    cfg.colormap = '*RdBu';
    cfg.title = titleDict{idx};
    ft_topoplotTFR(cfg, TFR_plv_left)
end

%% Coherence
TFR_fourier_left            = compute_fullFourier(epocLeft);
TFR_fourier_right           = compute_fullFourier(epocRight);
TFR_fourier_fpath           = [derivativesRoot filesep fNameRoot '_TFR_fourier.mat'];
save(TFR_fourier_fpath, 'TFR_fourier_left', 'TFR_fourier_right', '-v7.3')

%%
TFR_toRun                   = TFR_fourier_right;
seed_sensors                = right_sensors;
file_suffix                 = 'rightTrials_rightSensors';

cohspctcm_seeded            = NaN(length(seed_sensors), length(TFR_toRun.label), ...
                                  length(TFR_toRun.freq), length(TFR_toRun.time));
for seed_chan               = seed_sensors
    disp(['Running connectivity for: ' seed_chan])
    cfg                     = [];
    cfg.channelcmb          = [repmat(seed_chan, length(TFR_toRun.label), 1),...
                                TFR_toRun.label];
    cfg.method              = 'coh';
    connectivity_left       = ft_connectivityanalysis(cfg, TFR_toRun);

    seed_chanIdx            = find(ismember(seed_chan, TFR_toRun.label));
    if seed_chanIdx         == 1
        cohspctcm_seeded(seed_chanIdx, 2:end, :, :) ...
                            = connectivity_left.cohspctrm;
    else
        cohspctcm_seeded(seed_chanIdx, 1:seed_chanIdx-1, :, :) ...
                            = connectivity_left.cohspctrm(1:seed_chanIdx-1, :, :);
        cohspctcm_seeded(seed_chanIdx, seed_chanIdx+1:end, :, :) ...
                            = connectivity_left.cohspctrm(seed_chanIdx:end, :, :);
    end
    clearvars connectivity_left;
end


TFR_coh                     = TFR_toRun;
TFR_coh.powspctrm           = squeeze(mean(cohspctcm_seeded, 1, 'omitnan'));
TFR_coh                     = rmfield(TFR_coh, 'fourierspctrm');
TFR_coh.dimord              = 'chan_freq_time';
TFR_coh_fpath               = [derivativesRoot filesep fNameRoot '_coh_' file_suffix '.mat'];
save(TFR_coh_fpath, 'TFR_coh', '-v7.3');

cfg                         = [];
cfg.layout                  = lay;
cfg.xlim                    = [-0.5 1.7];
cfg.ylim                    = [4 40];
% cfg.zlim                    = [0 0.011];
cfg.zlim                    = [0 0.2];
cfg.parameter               = 'powspctrm';
cfg.colormap                = '*RdBu';
ft_multiplotTFR(cfg, TFR_coh)


freqbands                       = {'theta', 'alpha', 'beta'};
for ifx                         = 1:length(freqbands)
    freqband                    = freqbands{ifx};
    ff = figure('Visible', 'off', 'Renderer','painters');
    tOnset                      = [-0.5 0.1 0.5 1  ];
    tOffset                     = [ 0   0.3 1   1.5];
    titleDict                   = {'Fixation', 'StimOn', 'Early Delay', 'Late Delay'};
    
    sgtitle(freqband)
    for idx                     = 1:length(tOnset)
        subplot(2, 2, idx)
        cfg                     = [];
        cfg.layout              = lay;
        cfg.figure              = 'gcf';
        cfg.xlim                = [tOnset(idx) tOffset(idx)];
        if strcmp(freqband, 'theta')
            cfg.ylim            = [5 8];
        elseif strcmp(freqband, 'alpha')
            cfg.ylim            = [8 12];
        elseif strcmp(freqband, 'beta')
            cfg.ylim            = [14 25];
        end
        cfg.zlim                = [0 0.3];
        cfg.marker              = 'off';
        cfg.parameter           = 'powspctrm';
        cfg.colormap            = '*RdBu';
        cfg.highlight           = 'on'; 
        cfg.highlightchannel    = seed_sensors;
        cfg.highlightsymbol     = '+';
        cfg.highlightcolor      = [0 1 0];
        cfg.hightlightsize      = 40;
        cfg.hightlightfontsize  = 30;
    
        cfg.title = titleDict{idx};
        ft_topoplotTFR(cfg, TFR_coh)
    end
    fPath = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/group_plots/' freqband '_coh_topo_' file_suffix '.svg'];
    saveas(ff, fPath)
end
%%
cfg = [];
cfg.method = 'coh';
cfg.channelcmb = 'all';
phase_connectivity = ft_connectivityanalysis(cfg, TFR_fourier);


cfg = [];
cfg.parameter = 'cohspctrm';
ft_connectivityplot(cfg, phase_connectivity)

cfg = [];
cfg.parameter = 'cohspctrm';
