function S02A_ReverseModelMNIVolumetric(subjID, volumetric_resolution)
% MEG Reverse Model - Project Data Back to Volumetric Source Space
% Load forward model and project epoched data back to volumetric source space
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   volumetric_resolution - Volumetric resolution in mm (default: 8)
%
% Example:
%   S02A_ReverseModelMNIVolumetric(1, 8)
%   S02A_ReverseModelMNIVolumetric(1, 10)

if nargin < 1
    error('Subject ID is required');
end
if nargin < 2
    volumetric_resolution = 8; % Default resolution in mm
end

restoredefaultpath;
clearvars -except subjID volumetric_resolution;
close all; clc;

%% Environment Detection
% Detect if running on HPC or local machine
[~, hostname] = system('hostname');
hostname = strtrim(hostname);

% Check for common HPC indicators
is_hpc = contains(hostname, {'login', 'compute', 'node', 'hpc'}) || ...
         exist('/etc/slurm', 'dir') || ...
         ~isempty(getenv('SLURM_JOB_ID')) || ...
         ~isempty(getenv('PBS_JOBID'));

%% Setup paths based on environment
if is_hpc
    % HPC paths
    fieldtrip_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/';
    project_path = '/scratch/mdd9787/meg_prf_greene/megScripts';
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
else
    % Local machine paths
    fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
end

%% Setup and Initialization
addpath(fieldtrip_path);
addpath(genpath(project_path));
ft_defaults;
ft_hastoolbox('spm12', 1);

fprintf('=== MEG Reverse Model Analysis - Volumetric ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Volumetric resolution: %dmm\n', volumetric_resolution);

%% Load Forward Model
% Load the forward model created by S01_ForwardModelMNI.m
forward_model_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', sprintf('sub-%02d_task-mgs_forwardModel.mat', subjID));

if ~exist(forward_model_path, 'file')
    error('Forward model not found at: %s\nPlease run S01_ForwardModelMNI.m first!', forward_model_path);
end

fprintf('Loading forward model from: %s\n', forward_model_path);
load(forward_model_path);

%% Load Volumetric Source Model
% Load the volumetric source model created by S01A_VolSources2SubSpace.m
volumetric_source_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', sprintf('sub-%02d_task-mgs_volumetricSources_%dmm.mat', subjID, volumetric_resolution));

if ~exist(volumetric_source_path, 'file')
    error('Volumetric source model not found at: %s\nPlease run S01A_VolSources2SubSpace.m first!', volumetric_source_path);
end

fprintf('Loading volumetric source model from: %s\n', volumetric_source_path);
load(volumetric_source_path);

% Use the transformed volumetric source model
fprintf('Using volumetric source model with %d sources\n', size(sourcemodel.pos, 1));

output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

source_data_path = fullfile(output_dir, sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, volumetric_resolution));

if exist(source_data_path, 'file')
    fprintf('Source space data already exists at: %s\n', source_data_path);
    fprintf('Skipping processing to avoid overwriting existing data.\n');
    fprintf('To reprocess, delete the existing file first.\n');
    return;
else

    %% Load Epoched Data
    % Load the stimlocked, lineremoved data
    if is_hpc
        subDerivativesRoot = sprintf('/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives/sub-%02d/meg/sub-%02d_task-mgs_', subjID, subjID);
    else
        subDerivativesRoot = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/meg/sub-%02d_task-mgs_', subjID, subjID);
    end

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
    fprintf('Projecting data to volumetric source space...\n');

    % Create combined source data structure
    sourcedataCombined = [];
    sourcedataCombined.label = cell(numel(inside_pos), 1);
    for i = 1:numel(inside_pos)
        sourcedataCombined.label{i} = sprintf('V_%d', inside_pos(i)); % V for volumetric
    end

    % Project combined trials (includes trialinfo for left/right selection)
    sourcedataCombined.trial = cellfun(@(x) W_meg * x, epocCombined.trial, 'UniformOutput', false);
    % Convert to single precision for memory efficiency
    sourcedataCombined.trial = cellfun(@(x) single(x), sourcedataCombined.trial, 'UniformOutput', false);
    sourcedataCombined.time = epocCombined.time;
    sourcedataCombined.trialinfo = epocCombined.trialinfo; % Keep trial info for left/right selection

    fprintf('Data projected to volumetric source space\n');

    %% Save Source Space Data
    % Save the projected data for further analysis
    fprintf('Saving source space data to: %s\n', source_data_path);
    save(source_data_path, 'sourcedataCombined', 'source', 'inside_pos', 'W_meg', 'volumetric_resolution', '-v7.3');

    fprintf('Source space data saved successfully!\n');

    %% Visualization (Optional)
    if ~is_hpc
        % Quick visualization of the setup
        figure('Name', 'Volumetric Source Space Setup', 'Position', [100, 100, 1200, 400]);

        subplot(1, 3, 1);
        % Plot volumetric source model
        plot3(sourcemodel.pos(:, 1), sourcemodel.pos(:, 2), sourcemodel.pos(:, 3), 'r.', 'MarkerSize', 1);
        hold on;
        ft_plot_sens(gradData, 'facecolor', 'green', 'facealpha', 0.7);
        ft_plot_mesh(singleShellHeadModel.bnd, 'facecolor', 'brain', 'facealpha', 0.3);
        title(sprintf('Volumetric Source Model (%dmm)', volumetric_resolution));
        xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
        axis equal; grid on;

        subplot(1, 3, 2);
        % Plot inside sources only
        inside_sources_pos = sourcemodel.pos(inside_pos, :);
        plot3(inside_sources_pos(:, 1), inside_sources_pos(:, 2), inside_sources_pos(:, 3), 'r.', 'MarkerSize', 2);
        hold on;
        ft_plot_sens(gradData, 'facecolor', 'green', 'facealpha', 0.7);
        ft_plot_mesh(singleShellHeadModel.bnd, 'facecolor', 'brain', 'facealpha', 0.3);
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
    end
    fprintf('\nVolumetric reverse model setup complete!\n');
end
end
