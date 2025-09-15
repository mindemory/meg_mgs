function S02_ReverseModelMNI(subjID, surface_resolution)
% MEG Reverse Model - Project Data Back to Source Space
% Load forward model and project epoched data back to source space
% 
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   surface_resolution - Surface resolution (default: 5124)
%
% Example:
%   S02_ReverseModelMNI(1, 5124)

if nargin < 1
    error('Subject ID is required');
end
if nargin < 2
    surface_resolution = 5124; % Default resolution
end

restoredefaultpath;
clearvars -except subjID surface_resolution;
close all; clc;

%% Setup and Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);

fprintf('=== MEG Reverse Model Analysis ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Load Forward Model
% Load the forward model created by S01_ForwardModelMNI.m
forward_model_path = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon/sub-%02d_task-mgs_forwardModel.mat', subjID, subjID);

if ~exist(forward_model_path, 'file')
    error('Forward model not found at: %s\nPlease run S01_ForwardModelMNI.m first!', forward_model_path);
end

fprintf('Loading forward model from: %s\n', forward_model_path);
load(forward_model_path);

% Select the appropriate surface model based on resolution
switch surface_resolution
    case 5124
        sourcemodel = sourcemodel_aligned_5124;
    case 8196
        sourcemodel = sourcemodel_aligned_8196;
    case 20484
        sourcemodel = sourcemodel_aligned_20484;
    otherwise
        error('Unsupported surface resolution: %d', surface_resolution);
end

fprintf('Using surface model with %d vertices\n', size(sourcemodel.pos, 1));

%% Load Epoched Data
% Load the stimlocked, lineremoved data
subDerivativesRoot = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/meg/sub-%02d_task-mgs_', subjID, subjID);

stimlocked_path = [subDerivativesRoot 'stimlocked_lineremoved.mat'];

if ~exist(stimlocked_path, 'file')
    error('Stimlocked data not found at: %s\nPlease run preprocessing first!', stimlocked_path);
end

fprintf('Loading stimlocked data from: %s\n', stimlocked_path);
load(stimlocked_path);

% Use the epoched data
epocThis = epocStimLocked;
fprintf('Loaded %d trials\n', length(epocThis.trial));

%% Preprocessing: Lowpass Filter and Downsample BEFORE Source Projection
fprintf('Applying lowpass filter and downsampling to sensor data...\n');

% First, remove trials with NaNs before any filtering
fprintf('Removing trials with NaNs before filtering...\n');
has_no_nans = cellfun(@(x) ~any(isnan(x(:))), epocThis.trial)';
valid_trials_all = find(has_no_nans);
cfg = [];
cfg.trials = valid_trials_all;
epocThis_clean = ft_selectdata(cfg, epocThis);
clearvars epocThis has_no_nans;

fprintf('Kept %d valid trials (removed %d trials with NaNs)\n', length(epocThis_clean.trial), length(valid_trials_all) - length(epocThis_clean.trial));

% Step 1: Low-pass filter at 55Hz on sensor data
cfg = [];
cfg.lpfilter = 'yes';
cfg.lpfreq = 55; % Lowpass at 55Hz
epocThis_filtered = ft_preprocessing(cfg, epocThis_clean);
clearvars epocThis_clean;

% Step 2: Downsample to 120Hz on sensor data
cfg = [];
cfg.resamplefs = 120; % Downsample to 120Hz
cfg.detrend = 'no'; % Detrending already done in previous steps
epocThis_resampled = ft_resampledata(cfg, epocThis_filtered);
clearvars epocThis_filtered;

fprintf('Sensor data preprocessed. New sampling rate: %.1f Hz\n', epocThis_resampled.fsample);

%% Define Trial Criteria on Preprocessed Data
% Left trials (targets 4,5,6,7,8)
trial_criteria_left = (epocThis_resampled.trialinfo(:,2) == 4) | ...
                     (epocThis_resampled.trialinfo(:,2) == 5) | ...
                     (epocThis_resampled.trialinfo(:,2) == 6) | ...
                     (epocThis_resampled.trialinfo(:,2) == 7) | ...
                     (epocThis_resampled.trialinfo(:,2) == 8);

% Right trials (targets 1,2,3,9,10)
trial_criteria_right = (epocThis_resampled.trialinfo(:,2) == 1) | ...
                      (epocThis_resampled.trialinfo(:,2) == 2) | ...
                      (epocThis_resampled.trialinfo(:,2) == 3) | ...
                      (epocThis_resampled.trialinfo(:,2) == 9) | ...
                      (epocThis_resampled.trialinfo(:,2) == 10);

% Find valid trials (NaNs already removed in preprocessing)
valid_trialsLeft = find(trial_criteria_left);
valid_trialsRight = find(trial_criteria_right);

fprintf('Valid left trials: %d\n', length(valid_trialsLeft));
fprintf('Valid right trials: %d\n', length(valid_trialsRight));

%% Select Left and Right Trials
cfg = [];
cfg.trials = valid_trialsLeft;
epocLeft = ft_selectdata(cfg, epocThis_resampled);

cfg = [];
cfg.trials = valid_trialsRight;
epocRight = ft_selectdata(cfg, epocThis_resampled);

%% Compute Timelocked Data and Leadfield
% Combine left and right trials for covariance estimation
epocCombined = ft_appenddata([], epocLeft, epocRight);

% Compute timelocked data with covariance
cfg = [];
cfg.covariance = 'yes';
cfg.covariancewindow = 'all';
cfg.keeptrials = 'no';
timelockedCombined = ft_timelockanalysis(cfg, epocCombined);

fprintf('Computed timelocked data with covariance\n');

%% Compute Leadfield
% Use LCMV beamformer to compute leadfield
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

% Get inside positions
inside_pos = find(source.inside);
fprintf('Found %d inside sources\n', length(inside_pos));

%% Extract Filters
% Extract all filters for inside voxels
W_meg = cell2mat(cellfun(@(x) x, source.avg.filter(inside_pos), 'UniformOutput', false));

fprintf('Extracted filters: %d sources x %d sensors\n', size(W_meg, 1), size(W_meg, 2));

%% Project Data to Source Space
fprintf('Projecting data to source space...\n');

% Create combined source data structure
sourcedataCombined = [];
sourcedataCombined.label = cell(numel(inside_pos), 1);
for i = 1:numel(inside_pos)
    sourcedataCombined.label{i} = sprintf('S_%d', inside_pos(i));
end

% Project combined trials (includes trialinfo for left/right selection)
sourcedataCombined.trial = cellfun(@(x) W_meg * x, epocCombined.trial, 'UniformOutput', false);
% Convert to single precision for memory efficiency
sourcedataCombined.trial = cellfun(@(x) single(x), sourcedataCombined.trial, 'UniformOutput', false);
sourcedataCombined.time = epocCombined.time;
sourcedataCombined.trialinfo = epocCombined.trialinfo; % Keep trial info for left/right selection

fprintf('Data projected to source space\n');

%% Save Source Space Data
% Save the projected data for further analysis
output_dir = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon', subjID);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

source_data_path = fullfile(output_dir, sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, surface_resolution));

fprintf('Saving source space data to: %s\n', source_data_path);
save(source_data_path, 'sourcedataCombined', 'source', 'inside_pos', 'W_meg', '-v7.3');

fprintf('Source space data saved successfully!\n');

%% Visualization (Optional)
% Quick visualization of the setup
figure('Name', 'Source Space Setup', 'Position', [100, 100, 1200, 400]);

subplot(1, 3, 1);
% Plot source model
ft_plot_mesh(sourcemodel, 'facecolor', 'cortex', 'facealpha', 0.8, 'edgecolor', 'none');
hold on;
ft_plot_sens(gradData, 'facecolor', 'green', 'facealpha', 0.7);
title(sprintf('Source Model (%d vertices)', surface_resolution));
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
axis equal; grid on;

subplot(1, 3, 2);
% Plot inside sources only
inside_sources = sourcemodel;
inside_sources.pos = sourcemodel.pos(inside_pos, :);
ft_plot_mesh(inside_sources, 'facecolor', 'red', 'facealpha', 0.8, 'edgecolor', 'none');
hold on;
ft_plot_sens(gradData, 'facecolor', 'green', 'facealpha', 0.7);
title(sprintf('Inside Sources (%d)', length(inside_pos)));
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
axis equal; grid on;

subplot(1, 3, 3);
% Plot trial distribution
trial_counts = [length(valid_trialsLeft), length(valid_trialsRight)];
bar(trial_counts);
set(gca, 'XTickLabel', {'Left', 'Right'});
ylabel('Number of Trials');
title('Trial Distribution');
grid on;

fprintf('\nReverse model setup complete!\n');
end
