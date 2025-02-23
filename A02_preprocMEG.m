% function A02_preprocECoG(subjID)
clear; close all; clc;
warning('off', 'all');
%% Initialization
% p.subjID          = subjID;
% [p]               = initialization(p, 'ecog');

addpath('/d/DATD/hyper/software/fieldtrip-20220104/');
ft_defaults;
addpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs');

subjID = 31; % change this to run a different subject

% Initalizing variables
bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
derivativesRoot = [bidsRoot filesep 'derivatives/sub-' num2str(subjID, '%02d') '/meg'];
if ~exist("derivativesRoot", "dir")
    mkdir(derivativesRoot)
end

taskName = 'mgs';
subName = ['sub-' num2str(subjID, '%02d')];
megRoot = [bidsRoot filesep subName filesep 'meg'];
stimRoot = [bidsRoot filesep subName filesep 'stimfiles'];
fNameRoot = [subName '_task-' taskName];

% List files
MEGfiles = dir([megRoot '/*.sqd']);
MEGfiles = MEGfiles(~contains({MEGfiles.name}, 'marker'));
stimFiles = dir([stimRoot '/*.mat']);

nruns = length(MEGfiles);
if subjID == 5
    runList = [2 3 4 5 6 7 8 10 11 12]; %run1,9 are bad, check notes
elseif subjID == 10
    runList = [1 3 4 5 6 8 9 10]; %run2,7 are bad, check notes
elseif subjID == 11
    runList = [2 5 6 7 9 10]; %run8 does not exist; %run3 is bad
elseif subjID == 19 %run8, 9 are bad, check notes
    runList = [1 2 3 4 5 6 7];
elseif subjID == 23
    runList = 2:9;
elseif subjID == 25
    runList = 1:7;
elseif subjID == 31
    runList = [1 3 4 5 6 7 8]; %run2 does not exist
elseif subjID == 32
    runList = [1 3 4 5 6 7 8];
else
    runList = 1:nruns;
end
for run = runList
% for run = 8:max(runList)
    disp(['We are running ' num2str(run) ' of ' num2str(max(runList))])
    % load stimFile
    stimPth = [stimRoot filesep fNameRoot '_run-' num2str(run, '%02d') '_stimulus.mat'];
    load(stimPth, 'stimulus');
    recPth = [megRoot filesep fNameRoot '_run-' num2str(run, '%02d') '_meg.sqd'];
    
    rawdata_path = [derivativesRoot filesep fNameRoot '_run-' num2str(run, '%02d') '_raw.mat'];
    artifact_path = [derivativesRoot filesep fNameRoot '_run-' num2str(run, '%02d') '_artifact.mat'];
    unmixing_fpath = [derivativesRoot filesep fNameRoot '_run-' num2str(run, '%02d') '_unmixing.mat'];

    
    if exist(rawdata_path, 'file') > 0
        load(rawdata_path);
        if run == runList(1)
            % Create a neighbor structure
            cfg = [];
            cfg.channel = data.label;%(1:157);
            cfg.method = 'triangulation';
            cfg.grad = data.grad;
            cfg.feedback = 'yes';
            neighbors = ft_prepare_neighbours(cfg, data);
        end
    else
        % Load data
        cfg = [];
        cfg.dataset = recPth;
        cfg.hpfilter = 'yes';
        cfg.hpfreq = 0.5;
        data = ft_preprocessing(cfg);
    
        % Select first 157 channels
        cfg = [];
        cfg.channel = data.label(1:157);
        data = ft_selectdata(cfg, data);
    
        if run == runList(1)
            % Prepare layout
            cfg = [];
            cfg.grad = data.grad;
            % cfg.feedback = 'yes';
            lay = ft_prepare_layout(cfg);
    
            % Create a neighbor structure
            cfg = [];
            cfg.channel = data.label;%(1:157);
            cfg.method = 'triangulation';
            cfg.grad = data.grad;
            cfg.feedback = 'yes';
            neighbors = ft_prepare_neighbours(cfg, data);
        end
        % save data with layout
        save(rawdata_path, 'data', 'lay');
    end


    if exist(artifact_path, 'file') > 0
        load(artifact_path)
    else
        cfg = [];
        cfg.viewmode = 'vertical';
        cfg.ylim = [-1e-12 1e-12];
        cfg.blocksize = 50;
        cfg.layout = lay;
        cfg.channel = data.label;%(1:157);
        cfg_art = ft_databrowser(cfg, data);
        save(artifact_path, 'cfg_art'); 
    end
    
    %%%%%%%%%%%%%%% ICA %%%%%%%%%%%%%%%%%% 
    % Remove artifacts from the data
    if size(cfg_art.artfctdef.visual.artifact,1) > 0
        artVec = zeros(1, size(data.trial{1},2));
        for aa = 1: size(cfg_art.artfctdef.visual.artifact,1)
            
            artVec(cfg_art.artfctdef.visual.artifact(aa,1)...
                : cfg_art.artfctdef.visual.artifact(aa,2)) = 1;
        end
        data.trial{1}(:, artVec==1) = NaN;
    end

    % Interpolate the bad channel (channel 46)
    bad_chans = badChannelRemover(subjID, run);
    if ~isempty(bad_chans)
        cfg = [];
        cfg.badchannel = data.label(bad_chans);
        cfg.method = 'weighted';
        cfg.neighbours = neighbors;
        data = ft_channelrepair(cfg, data);
    end

    if exist(unmixing_fpath, "file") > 0
        % ICA was already performed so simply load existing unmixing matrix
        load(unmixing_fpath);
        cfg = [];
        cfg.unmixing = unmixing;
        cfg.topolabel = data.label;%(1:157);
        comp = ft_componentanalysis(cfg, data);
    else
        % As a first step make sure to check the components
        cfg = [];
        cfg.method = 'runica';
        cfg.channel = data.label;%(1:157);
        cfg.numcomponent = 157 - length(bad_chans);
        comp = ft_componentanalysis(cfg, data);
        unmixing = comp.unmixing;
        save(unmixing_fpath, 'unmixing', '-v7.3');
    end

    % cfg = [];
    % cfg.component = 1:30;
    % cfg.layout = lay;
    % cfg.comment = 'yes';
    % ft_topoplotIC(cfg, comp)
    
    % if run == 3
        % cfg = [];
        % cfg.layout = lay;
        % cfg.viewmode = 'component';
        % cfg.blocksize = 50;
        % ft_databrowser(cfg, comp)
    % end

    bad_comps = icaCompRemover(subjID, run);

    cfg = [];
    cfg.component = bad_comps;
    data = ft_rejectcomponent(cfg, comp, data);  

    cfg = [];
    cfg.planarmethod = 'sincos';
    cfg.neighbours = neighbors;
    cfg.feedback = 'yes';
    dataPlanar = ft_megplanar(cfg, data);

    cfg = [];
    data = ft_combineplanar(cfg, dataPlanar);

    %% Read events
    cfg  = [];
    cfg.dataset = recPth;
    cfg.trialdef.prestim = 1.5;
    cfg.trialdef.poststim = 1.5;
    cfg.trialfun = 'readEvents';
    cfg = ft_definetrial(cfg);

    % Add additional 
    % epocData = ft_preprocessing(cfg);
    epocData = ft_redefinetrial(cfg, data);
    if (subjID == 12 & run == 8) || (subjID == 4 & run == 10)
        % These have trial1 missing from triggers
        epocData.trialinfo = [stimulus.tarloc(2:end); ...
                              stimulus.tarlocCode(2:end);
                              stimulus.x(2:end);
                              stimulus.y(2:end)]';
    elseif (subjID == 13 & run == 2)
        % This one has 2 trials missing, assuming the first 2 are missing
        epocData.trialinfo = [stimulus.tarloc(3:end); ...
                              stimulus.tarlocCode(3:end);
                              stimulus.x(3:end);
                              stimulus.y(3:end)]';
    elseif (subjID == 31 & run == 4)
        % Only first 18 trials are present and the remaining are missing
        epocData.trialinfo = [stimulus.tarloc(1:18); ...
                              stimulus.tarlocCode(1:18);
                              stimulus.x(1:18);
                              stimulus.y(1:18)]';
    else
        epocData.trialinfo = [stimulus.tarloc; ...
                              stimulus.tarlocCode;
                              stimulus.x;
                              stimulus.y]';
    end
    if run == runList(1)
        allEpoc = epocData;
    else
        % epocData.hdr.nSamples = epocData.hdr.nSamples + allEpoc.hdr.nSamples;
        epocData.sampleinfo = epocData.sampleinfo + max(allEpoc.sampleinfo, [], 'all');
        cfg = [];
        % cfg.keepsampleinfo = 'no';
        allEpoc = ft_appenddata(cfg, allEpoc, epocData);
        % allEpoc.hdr
    end
    clearvars stimulus;
end

%% Segregate data into long and short delays
% Separate trials into long and short delays
trlSamps = arrayfun(@(x) size(allEpoc.trial{x}, 2), 1:length(allEpoc.trial));
allEpoc.trialinfo(:, 5) = trlSamps > 3000; % 1 if long delay, 0 if short delay

epoch_fpath = [derivativesRoot filesep fNameRoot '_epoched.mat'];
stimLocked_fpath = [derivativesRoot filesep fNameRoot '_stimlocked.mat'];
% shortEpoc_fpath = [derivativesRoot filesep fNameRoot '_short_epoched.mat'];
% longEpoc_fpath = [derivativesRoot filesep fNameRoot '_long_epoched.mat'];

cfg = [];
cfg.trials = find(allEpoc.trialinfo(:, 3) == 0);
cfg.latency = 'minperiod';
epocShort = ft_selectdata(cfg, allEpoc);

cfg = [];
cfg.trials = find(allEpoc.trialinfo(:, 3) == 1);
cfg.latency = 'minperiod';
epocLong = ft_selectdata(cfg, allEpoc);

cfg = [];
cfg.latency = [-1.5 2.5]; % stim-locked
epocStimLocked = ft_selectdata(cfg, allEpoc);

% cfg = [];
% cfg.latency = [1]; % response-locked
% epocRespLocked = ft_selectdata(cfg, allEpoc);

% epoch_fpath = [derivativesRoot filesep fNameRoot '_epoched.fif'];

save(epoch_fpath, 'allEpoc');
save(stimLocked_fpath, 'epocStimLocked')
% save(shortEpoc_fpath, 'epocShort');
% save(longEpoc_fpath, 'epocLong');