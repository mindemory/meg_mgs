function G03_SourceLocalizationBroadband(subjID, lockType, volumetric_resolution)
%G03_SOURCELOCALIZATIONBROADBAND  Broadband LCMV source projection for
%   both stim- and resp-locked epochs, adapted from
%   S02A_ReverseModelMNIVolumetric.m (not modified in place -- that
%   script's existing 120Hz/55Hz stim-locked outputs are left untouched
%   for other ongoing analyses).
%
%   Inputs:
%     subjID                  - subject ID (e.g. 1, 2, 3, ...)
%     lockType                - 'stim' or 'resp'
%     volumetric_resolution   - volumetric grid resolution in mm (default: 8;
%                               this pass only targets 8mm, per project scope)
%
%   Key differences from S02A_ReverseModelMNIVolumetric.m:
%     (1) No HPC/SLURM environment detection -- this runs directly on
%         vader (local paths only).
%     (2) Raises the bandwidth ceiling that blocked high-gamma: sensor
%         lowpass raised from 55Hz to 120Hz, resample rate raised from
%         120Hz to 250Hz (Nyquist 125Hz, clear of the highgamma upper
%         edge at 95Hz with margin for filter roll-off). This produces a
%         BROADBAND output; per-band filtering/downsampling is the job of
%         G04_BandAmplitudePhaseInSource.m, not this script.
%     (3) Accepts a lockType parameter and loads/saves accordingly.
%     (4) Reuses the LCMV spatial filter (W_meg/source/inside_pos) across
%         lock types for a given subject+resolution rather than
%         recomputing the beamformer twice -- see "Shared beamformer
%         filter" below.
%
%   *** Shared beamformer filter across lock types ***
%   The LCMV spatial filter is a property of the forward model + sensor
%   covariance, not of which epoch window is later projected through it.
%   Both stim-locked (A02's fixed [-1.5,2.5]s crop) and resp-locked
%   (G01_ExtractRespLockedEpochs.m's fixed 5.0s-ending-at-trial-end crop)
%   epochs are uniform-length across all trials, so covariance/LCMV
%   estimation is valid from either one. Whichever lockType is run FIRST
%   for a given subject/resolution computes the beamformer filter and
%   caches it to sub-XX_task-mgs_beamformerFilter_{res}.mat; the other
%   lockType then REUSES that cached filter and only performs the
%   per-trial projection step (W_meg * trial, applied per-cell).

if nargin < 1 || isempty(subjID)
    error('Subject ID is required');
end
if nargin < 2 || isempty(lockType)
    error('lockType is required (''stim'' or ''resp'')');
end
if ~ismember(lockType, {'stim', 'resp'})
    error('lockType must be ''stim'' or ''resp'', got ''%s''', lockType);
end
if nargin < 3 || isempty(volumetric_resolution)
    volumetric_resolution = 8; % this pass only targets 8mm
end

restoredefaultpath;
clearvars -except subjID lockType volumetric_resolution;
close all; clc;

%% Setup paths (vader / local only -- no HPC)
fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
project_path   = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';

addpath(fieldtrip_path);
addpath(genpath(project_path));
ft_defaults;
ft_hastoolbox('spm12', 1);

fprintf('=== G03: Broadband Source Localization ===\n');
fprintf('Subject: %d | lockType: %s | resolution: %dmm\n', subjID, lockType, volumetric_resolution);

%% Load Forward Model
forward_model_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', sprintf('sub-%02d_task-mgs_forwardModel.mat', subjID));
if ~exist(forward_model_path, 'file')
    error('Forward model not found at: %s\nPlease run S01_ForwardModelMNI.m first!', forward_model_path);
end
fprintf('Loading forward model from: %s\n', forward_model_path);
load(forward_model_path); % provides singleShellHeadModel, gradData

%% Load Volumetric Source Model
volumetric_source_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', sprintf('sub-%02d_task-mgs_volumetricSources_%dmm.mat', subjID, volumetric_resolution));
if ~exist(volumetric_source_path, 'file')
    error('Volumetric source model not found at: %s\nPlease run S01A_VolSources2SubSpace.m first!', volumetric_source_path);
end
fprintf('Loading volumetric source model from: %s\n', volumetric_source_path);
load(volumetric_source_path); % provides sourcemodel

output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

source_data_path     = fullfile(output_dir, sprintf('sub-%02d_task-mgs_sourceSpaceData_%d_%s.mat', subjID, volumetric_resolution, lockType));
beamformer_fpath      = fullfile(output_dir, sprintf('sub-%02d_task-mgs_beamformerFilter_%d.mat', subjID, volumetric_resolution));

if exist(source_data_path, 'file')
    fprintf('Broadband source space data already exists at: %s\n', source_data_path);
    fprintf('Skipping processing to avoid overwriting existing data.\n');
    return;
end

%% Load epochs for the requested lock type
subDerivativesRoot = sprintf('%s/sub-%02d/meg/sub-%02d_task-mgs_', data_base_path, subjID, subjID);
switch lockType
    case 'stim'
        epoch_path = [subDerivativesRoot 'stimlocked_lineremoved.mat'];
        epochVarName = 'epocStimLocked';
        if ~exist(epoch_path, 'file')
            error('Stimlocked data not found at: %s\nPlease run A02_preprocMEG.m first!', epoch_path);
        end
    case 'resp'
        epoch_path = [subDerivativesRoot 'resplocked_lineremoved.mat'];
        epochVarName = 'epocRespLocked';
        if ~exist(epoch_path, 'file')
            error('Resplocked data not found at: %s\nPlease run G01_ExtractRespLockedEpochs.m first!', epoch_path);
        end
end
fprintf('Loading %s epochs from: %s\n', lockType, epoch_path);
loaded = load(epoch_path, epochVarName);
epocThis = loaded.(epochVarName);
clearvars loaded;
fprintf('Loaded %d trials\n', length(epocThis.trial));

%% Remove trials with NaNs before any filtering
fprintf('Removing trials with NaNs before filtering...\n');
has_no_nans = cellfun(@(x) ~any(isnan(x(:))), epocThis.trial)';
valid_trials_all = find(has_no_nans);
cfg = [];
cfg.trials = valid_trials_all;
epocThis_clean = ft_selectdata(cfg, epocThis);
fprintf('Kept %d valid trials (removed %d trials with NaNs)\n', length(epocThis_clean.trial), length(epocThis.trial) - length(epocThis_clean.trial));
clearvars epocThis has_no_nans;

%% Raised-ceiling lowpass + downsample (broadband substrate for G04's per-band stage)
% 120Hz lowpass (vs. S02A's 55Hz) + 250Hz resample (vs. S02A's 120Hz)
% gives Nyquist=125Hz, comfortably above the highgamma band's 95Hz upper
% edge, while the 120Hz anti-alias lowpass sits below that Nyquist with
% margin for filter roll-off.
cfg = [];
cfg.lpfilter = 'yes';
cfg.lpfreq = 120;
epocThis_filtered = ft_preprocessing(cfg, epocThis_clean);
clearvars epocThis_clean;

cfg = [];
cfg.resamplefs = 250;
cfg.detrend = 'no';
epocThis_resampled = ft_resampledata(cfg, epocThis_filtered);
clearvars epocThis_filtered;
fprintf('Sensor data preprocessed. New sampling rate: %.1f Hz\n', epocThis_resampled.fsample);

%% Left/right trial split (same target-code convention as S02A)
trial_criteria_left = ismember(epocThis_resampled.trialinfo(:,2), [4 5 6 7 8]);
trial_criteria_right = ismember(epocThis_resampled.trialinfo(:,2), [1 2 3 9 10]);

valid_trialsLeft = find(trial_criteria_left);
valid_trialsRight = find(trial_criteria_right);
fprintf('Valid left trials: %d | Valid right trials: %d\n', length(valid_trialsLeft), length(valid_trialsRight));

cfg = [];
cfg.trials = valid_trialsLeft;
epocLeft = ft_selectdata(cfg, epocThis_resampled);

cfg = [];
cfg.trials = valid_trialsRight;
epocRight = ft_selectdata(cfg, epocThis_resampled);

epocCombined = ft_appenddata([], epocLeft, epocRight);

%% Obtain the LCMV spatial filter: reuse cached filter, or compute+cache it
if exist(beamformer_fpath, 'file')
    fprintf('Loading cached beamformer filter from: %s\n', beamformer_fpath);
    load(beamformer_fpath, 'source', 'inside_pos', 'W_meg');
else
    fprintf('No cached filter found -- computing beamformer filter from %s-locked data.\n', lockType);
    cfg = [];
    cfg.covariance = 'yes';
    cfg.covariancewindow = 'all';
    cfg.keeptrials = 'no';
    timelockedCombined = ft_timelockanalysis(cfg, epocCombined);
    fprintf('Computed timelocked data with covariance\n');

    cfg = [];
    cfg.method = 'lcmv';
    cfg.sourcemodel = sourcemodel;
    cfg.headmodel = singleShellHeadModel;
    cfg.grad = gradData;
    cfg.keepleadfield = 'yes';
    cfg.lcmv.keepfilter = 'yes';
    cfg.lcmv.fixedori = 'yes';
    cfg.lcmv.lambda = '5%';
    fprintf('Computing leadfield...\n');
    source = ft_sourceanalysis(cfg, timelockedCombined);

    inside_pos = find(source.inside);
    fprintf('Found %d inside sources\n', length(inside_pos));

    W_meg = cell2mat(cellfun(@(x) x, source.avg.filter(inside_pos), 'UniformOutput', false));
    fprintf('Extracted filters: %d sources x %d sensors\n', size(W_meg, 1), size(W_meg, 2));

    fprintf('Caching beamformer filter to: %s\n', beamformer_fpath);
    save(beamformer_fpath, 'source', 'inside_pos', 'W_meg', 'volumetric_resolution', '-v7.3');
end

%% Project every trial to broadband source space
fprintf('Projecting %s-locked data to volumetric source space...\n', lockType);

sourcedataCombined = [];
sourcedataCombined.label = cell(numel(inside_pos), 1);
for i = 1:numel(inside_pos)
    sourcedataCombined.label{i} = sprintf('V_%d', inside_pos(i));
end

sourcedataCombined.trial = cellfun(@(x) single(W_meg * x), epocCombined.trial, 'UniformOutput', false);
sourcedataCombined.time = epocCombined.time;
sourcedataCombined.trialinfo = epocCombined.trialinfo;
sourcedataCombined.fsample = epocCombined.fsample;

fprintf('Saving broadband source space data to: %s\n', source_data_path);
save(source_data_path, 'sourcedataCombined', 'inside_pos', 'volumetric_resolution', 'lockType', '-v7.3');
fprintf('Done.\n');

end
