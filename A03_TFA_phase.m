clear; close all; clc;
warning('off', 'all');
%% Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20220104/');
ft_defaults;

subList = [1 2 3 4 5 6 7 8 9 10 11 12 13 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32];
% subList = [18 19 20 22];

TFRleft = {};
TFRright = {};
load('NYUKIT_helmet.mat');
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
    

    stimLocked_fpath = [derivativesRoot filesep fNameRoot '_stimlocked.mat'];
    TFRphase_fpath = [derivativesRoot filesep fNameRoot '_TFR_phase.mat'];
    if ~exist(TFRphase_fpath, 'file')
        disp('TFR does not exist, creating it.')
        epocStimLocked = load(stimLocked_fpath);
    
        % Load the data
        epocThis = epocStimLocked.epocStimLocked;
    
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
    
        TFRleft_phase       = compute_phase(epocLeft);
        TFRright_phase       = compute_phase(epocRight);

        save(TFRphase_fpath, 'TFRleft_phase', 'TFRright_phase', '-v7.3')
    else
        disp('Loading existing TFR. If this is not desired, delete the existing file.')
        load(TFRphase_fpath);
    end
    cfg = [];
    cfg.avgoverrpt = 'yes';  % This averages over the rpttap dimension (trials)
    TFRleft{sIdx} = ft_freqdescriptives(cfg, TFRleft_phase);
    TFRright{sIdx} = ft_freqdescriptives(cfg, TFRright_phase);
    
end

% cfg = [];
% cfg.layout = lay;
% ft_multiplotER(cfg, rightERP)


cfg = [];
cfg.keepindividual = 'no';
TFRleft_grandavg = ft_freqgrandaverage(cfg, TFRleft{:});
TFRright_grandavg = ft_freqgrandaverage(cfg, TFRright{:});
TFRleft_grandavg.grad = TFRleft_phase.grad;
TFRright_grandavg.grad = TFRright_phase.grad;
% 
% 
% cfg = [];
% cfg.parameter = 'powspctrm';  % Specify the parameter to operate on
% % cfg.operation = 'x1 / x2'; %/(x1+x2)';   % Element-wise subtraction
% cfg.operation = '(10^(x1/10) - 10^(x2/10)) / (10^(x1/10) + 10^(x2/10))';
% % cfg.operation = '(10^(x2/10) ) / (10^(x1/10) + 10^(x2/10))';
% TFRdiff = ft_math(cfg, TFRright_grandavg, TFRleft_grandavg);
% 
% cfg = [];
% % cfg.baseline = [-1.0 0];
% % cfg.baselinetype = 'absolute';
% % cfg.maskparameter = 0;
% cfg.xlim = 0:0.5:2;
% % cfg.xlim = [0 0.5];
% cfg.ylim = [8 12];
% cfg.marker = 'on';
% cfg.layout = lay;
% cfg.colormap = '*RdBu';
% ft_topoplotTFR(cfg, TFRdiff)