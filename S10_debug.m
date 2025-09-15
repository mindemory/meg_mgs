%% MEG Reverse Model Debug - Combined S02 and S03
% Load forward model, project data to source space, and compute beta power
% Combined script with proper variable cleanup
restoredefaultpath;
clear; close all; clc;

%% Setup and Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);

% Configuration parameters
subjID = 12; % Change this to the subject you want to analyze
surface_resolution = 8196; % Use lowest resolution for now

fprintf('=== MEG Reverse Model Debug Analysis ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Load Forward Model
fprintf('Loading forward model...\n');
forward_model_path = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon/sub-%02d_task-mgs_forwardModel.mat', subjID, subjID);

if ~exist(forward_model_path, 'file')
    error('Forward model not found at: %s\nPlease run S01_ForwardModelMNI.m first!', forward_model_path);
end

load(forward_model_path);

% Select the appropriate surface model
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

% Clean up unnecessary variables from forward model
clearvars sourcemodel_aligned_5124 sourcemodel_aligned_8196 sourcemodel_aligned_20484;
clearvars surface_resolutions transformApplied segmentedmri_aligned;

%% Load Epoched Data
fprintf('Loading epoched data...\n');
subDerivativesRoot = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/meg/sub-%02d_task-mgs_', subjID, subjID);

% Determine which run to load based on subject
if subjID == 5 || subjID == 23
    run_num = '02';
else
    run_num = '01';
end

stimlocked_path = [subDerivativesRoot 'stimlocked_lineremoved.mat'];

if ~exist(stimlocked_path, 'file')
    error('Stimlocked data not found at: %s\nPlease run preprocessing first!', stimlocked_path);
end

load(stimlocked_path);

% Use the epoched data
epocThis = epocStimLocked;
fprintf('Loaded %d trials\n', length(epocThis.trial));

% Clean up
clearvars epocStimLocked;

%% Define Trial Criteria
fprintf('Defining trial criteria...\n');

% Left trials (targets 4,5,6,7,8)
trial_criteria_left = (epocThis.trialinfo(:,2) == 4) | ...
                     (epocThis.trialinfo(:,2) == 5) | ...
                     (epocThis.trialinfo(:,2) == 6) | ...
                     (epocThis.trialinfo(:,2) == 7) | ...
                     (epocThis.trialinfo(:,2) == 8);

% Right trials (targets 1,2,3,9,10)
trial_criteria_right = (epocThis.trialinfo(:,2) == 1) | ...
                      (epocThis.trialinfo(:,2) == 2) | ...
                      (epocThis.trialinfo(:,2) == 3) | ...
                      (epocThis.trialinfo(:,2) == 9) | ...
                      (epocThis.trialinfo(:,2) == 10);

% Find trials without NaNs
has_no_nans = cellfun(@(x) ~any(isnan(x(:))), epocThis.trial)';

% Combine criteria
valid_trialsLeft = find(trial_criteria_left & has_no_nans);
valid_trialsRight = find(trial_criteria_right & has_no_nans);

fprintf('Valid left trials: %d\n', length(valid_trialsLeft));
fprintf('Valid right trials: %d\n', length(valid_trialsRight));

% Note: Trial selection now happens after NaN removal and preprocessing

% Clean up
clearvars trial_criteria_left trial_criteria_right;

%% Preprocessing: Lowpass Filter and Downsample BEFORE Source Projection
fprintf('Applying lowpass filter and downsampling to sensor data...\n');

% First, remove trials with NaNs before any filtering
fprintf('Removing trials with NaNs before filtering...\n');
valid_trials_all = find(has_no_nans);
cfg = [];
cfg.trials = valid_trials_all;
epocThis_clean = ft_selectdata(cfg, epocThis);
clearvars epocThis has_no_nans;

fprintf('Kept %d valid trials (removed %d trials with NaNs)\n', length(epocThis_clean.trial), length(valid_trials_all) - length(epocThis_clean.trial));

% Step 1: Low-pass filter at 100Hz on sensor data
cfg = [];
cfg.lpfilter = 'yes';
cfg.lpfreq = 55; % Lowpass at 100Hz
epocThis_filtered = ft_preprocessing(cfg, epocThis_clean);
clearvars epocThis_clean;

% Step 2: Downsample to 200Hz on sensor data
cfg = [];
cfg.resamplefs = 120; % Downsample to 200Hz
cfg.detrend = 'no'; % Detrending already done in previous steps
epocThis_resampled = ft_resampledata(cfg, epocThis_filtered);
clearvars epocThis_filtered;

fprintf('Sensor data preprocessed. New sampling rate: %.1f Hz\n', epocThis_resampled.fsample);

% Now re-define trial criteria on the cleaned data
fprintf('Re-defining trial criteria on cleaned data...\n');
trial_criteria_left = (epocThis_resampled.trialinfo(:,2) == 4) | ...
                      (epocThis_resampled.trialinfo(:,2) == 5) | ...
                      (epocThis_resampled.trialinfo(:,2) == 6) | ...
                      (epocThis_resampled.trialinfo(:,2) == 7) | ...
                      (epocThis_resampled.trialinfo(:,2) == 8);

trial_criteria_right = (epocThis_resampled.trialinfo(:,2) == 1) | ...
                       (epocThis_resampled.trialinfo(:,2) == 2) | ...
                       (epocThis_resampled.trialinfo(:,2) == 3) | ...
                       (epocThis_resampled.trialinfo(:,2) == 9) | ...
                       (epocThis_resampled.trialinfo(:,2) == 10);

valid_trialsLeft = find(trial_criteria_left);
valid_trialsRight = find(trial_criteria_right);

fprintf('Valid left trials: %d\n', length(valid_trialsLeft));
fprintf('Valid right trials: %d\n', length(valid_trialsRight));

% Select left and right trials
cfg = [];
cfg.trials = valid_trialsLeft;
epocLeft = ft_selectdata(cfg, epocThis_resampled);

cfg = [];
cfg.trials = valid_trialsRight;
epocRight = ft_selectdata(cfg, epocThis_resampled);

clearvars epocThis_resampled trial_criteria_left trial_criteria_right valid_trialsLeft valid_trialsRight;

%% Compute Timelocked Data and Leadfield
fprintf('Computing timelocked data...\n');

% Combine left and right trials for covariance estimation
epocCombined = ft_appenddata([], epocLeft, epocRight);

% Compute timelocked data with covariance
cfg = [];
cfg.covariance = 'yes';
cfg.covariancewindow = 'all';
cfg.keeptrials = 'no';
timelockedCombined = ft_timelockanalysis(cfg, epocCombined);

fprintf('Computing leadfield...\n');

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

source = ft_sourceanalysis(cfg, timelockedCombined);

% Get inside positions
inside_pos = find(source.inside);
fprintf('Found %d inside sources\n', length(inside_pos));

% Clean up
clearvars epocCombined timelockedCombined;

%% Extract Filters and Project Data
fprintf('Extracting filters and projecting data...\n');

% Extract all filters for inside voxels
W_meg = cell2mat(cellfun(@(x) x, source.avg.filter(inside_pos), 'UniformOutput', false));

% Create source data structures
sourcedataLeft = [];
sourcedataLeft.label = cell(numel(inside_pos), 1);
for i = 1:numel(inside_pos)
    sourcedataLeft.label{i} = sprintf('S_%d', inside_pos(i));
end

sourcedataRight = sourcedataLeft;

% Project left trials
sourcedataLeft.trial = cellfun(@(x) W_meg * x, epocLeft.trial, 'UniformOutput', false);
sourcedataLeft.time = epocLeft.time;

% Project right trials
sourcedataRight.trial = cellfun(@(x) W_meg * x, epocRight.trial, 'UniformOutput', false);
sourcedataRight.time = epocRight.time;

% Clean up
clearvars epocLeft epocRight W_meg;

%% Beta Band Filtering and Hilbert Transform
fprintf('Applying beta band filter (18-27Hz)...\n');

% Band-pass filter for beta band (18-27Hz)
cfg = [];
cfg.bpfilter = 'yes';
cfg.bpfreq = [18 27];
sourcedataLeft_beta = ft_preprocessing(cfg, sourcedataLeft);
sourcedataRight_beta = ft_preprocessing(cfg, sourcedataRight);

% Clean up
clearvars sourcedataLeft sourcedataRight;

fprintf('Computing Hilbert transform...\n');

% Apply Hilbert transform to get analytic signal
hilbert_compute = @(x) hilbert(x')';
sourcedataLeft_beta.trial = cellfun(hilbert_compute, sourcedataLeft_beta.trial, 'UniformOutput', false);
sourcedataRight_beta.trial = cellfun(hilbert_compute, sourcedataRight_beta.trial, 'UniformOutput', false);

%% Compute Beta Power
fprintf('Computing beta power...\n');

% Compute power (magnitude of analytic signal)
sourceLeftPow = sourcedataLeft_beta;
sourceLeftPow.trial = cellfun(@(x) abs(x), sourceLeftPow.trial, 'UniformOutput', false);

sourceRightPow = sourcedataRight_beta;
sourceRightPow.trial = cellfun(@(x) abs(x), sourceRightPow.trial, 'UniformOutput', false);

% Clean up
clearvars sourcedataLeft_beta sourcedataRight_beta;

%% Average Across Trials
fprintf('Averaging across trials...\n');

% Average across trials
cfg = [];
sourceDataLeft_avg = ft_timelockanalysis(cfg, sourceLeftPow);
sourceDataRight_avg = ft_timelockanalysis(cfg, sourceRightPow);

% Clean up
clearvars sourceLeftPow sourceRightPow;

%% Compute Lateralization Index
fprintf('Computing lateralization index...\n');

% Compute lateralization index: (Left - Right) / (Left + Right)
cfg = [];
cfg.parameter = 'avg';
cfg.operation = '(x1-x2)/(x1+x2)';
sourceDiff = ft_math(cfg, sourceDataLeft_avg, sourceDataRight_avg);

%% Define Time Window of Interest
time_window = [0.8, 1.5];
TOI = find(sourceDiff.time >= time_window(1) & sourceDiff.time <= time_window(2));

fprintf('Time window of interest: %.1f-%.1f s (%d time points)\n', ...
        time_window(1), time_window(2), length(TOI));

%% Create Source Visualization Structure
fprintf('Creating source visualization structure...\n');

% Create source visualization structure
sourceVisualize = source;
sourceVisualize.lateralizedPow = NaN(size(source.inside));

% Compute mean lateralization in time window
mean_lateralization = squeeze(mean(sourceDiff.avg(:, TOI), 2, 'omitnan'));
sourceVisualize.lateralizedPow(source.inside) = mean_lateralization;

%% Interpolate to MRI for Visualization
fprintf('Interpolating to MRI for visualization...\n');

% Load the aligned MRI for interpolation
cfg = [];
cfg.parameter = {'lateralizedPow'};
interp = ft_sourceinterpolate(cfg, sourceVisualize, mri_reslice);

%% Save Results
fprintf('Saving results...\n');

output_dir = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon', subjID);
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

beta_power_path = fullfile(output_dir, sprintf('sub-%02d_task-mgs_betaPower_debug_%d.mat', subjID, surface_resolution));

save(beta_power_path, 'sourceVisualize', 'sourceDiff', 'sourceDataLeft_avg', 'sourceDataRight_avg', ...
     'interp', 'time_window', 'TOI', 'subjID', 'surface_resolution', 'inside_pos', '-v7.3');

fprintf('Results saved to: %s\n', beta_power_path);

%% Visualization
fprintf('Creating visualizations...\n');

% Create comprehensive visualization
figure('Name', 'Beta Power Analysis Results', 'Position', [100, 100, 1600, 1000]);
% Subplot 1: Source space lateralization (3D scatter)
scatter3(sourcemodel.pos(inside_pos, 1), sourcemodel.pos(inside_pos, 2), sourcemodel.pos(inside_pos, 3), ...
         20, mean_lateralization, 'filled');
colorbar;

% Create custom red-blue colormap
colormap("turbo");
caxis([-0.05, 0.05]);
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
title('Source Space Lateralization');
axis equal;
grid on;

%% Time Window Visualization (500ms windows from -0.5 to 1.5s)
fprintf('Creating time window visualizations...\n');

% Define time windows (500ms each, overlapping)
time_start = -0.5;
time_end = 1.5;
window_duration = 0.5; % 500ms
window_step = 0.1; % 100ms step (overlapping windows)

time_windows = [];
window_centers = [];
for t = time_start:window_step:(time_end - window_duration)
    time_windows = [time_windows; t, t + window_duration];
    window_centers = [window_centers; t + window_duration/2];
end

fprintf('Created %d time windows of %.1fs duration\n', size(time_windows, 1), window_duration);

% Create figure for time window visualization
figure('Name', 'Beta Power Lateralization - Time Windows', 'Position', [100, 100, 1600, 1200]);

% Calculate number of subplots needed (aim for roughly square layout)
n_windows = size(time_windows, 1);
n_cols = ceil(sqrt(n_windows));
n_rows = ceil(n_windows / n_cols);

for i = 1:n_windows
    subplot(n_rows, n_cols, i);
    
    % Get time indices for this windowy once?
    window_start_idx = dsearchn(sourceVisualize.time', time_windows(i, 1));
    window_end_idx = dsearchn(sourceVisualize.time', time_windows(i, 2));
    
    % Calculate mean lateralization for this time window using sourceDiff.avg (which has time dimension)
    window_lateralization = mean(sourceDiff.avg(inside_pos, window_start_idx:window_end_idx), 2, 'omitnan');
    
    % Create 3D scatter plot for this time window
    scatter3(sourcemodel.pos(inside_pos, 1), sourcemodel.pos(inside_pos, 2), sourcemodel.pos(inside_pos, 3), ...
             15, window_lateralization, 'filled');
    
    % Set colormap and limits
    colormap("turbo");
    caxis([-0.05, 0.05]);
    
    % Formatting
    title(sprintf('%.1f-%.1fs', time_windows(i, 1), time_windows(i, 2)));
    xlabel('X (mm)');
    ylabel('Y (mm)');
    zlabel('Z (mm)');
    axis equal;
    grid on;
    
    % Set consistent view angle - top view with posterior cortex visible
    view([0, 40]); % Rotated view: frontal down, posterior up
    % Alternative views for better posterior visibility:
    % view([-45, 30]); % Slightly angled to see posterior
    % view([0, 45]); % Angled top view
    
    % Add colorbar for first subplot only
    if i == 1
        colorbar;
    end
end

% Add overall title
sgtitle(sprintf('Subject %02d: Beta Power Lateralization Across Time Windows', subjID));

fprintf('Time window visualization complete!\n');

%% Final Cleanup
clearvars -except sourceVisualize sourceDiff sourceDataLeft_avg sourceDataRight_avg interp time_window TOI subjID surface_resolution inside_pos mean_lateralization;
