function S04_betaPowerInMNI(subjID, surface_resolution)
%% MEG Beta Power Analysis in MNI Space
% Load complex beta data, compute power, lateralization, and visualize in MNI space
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   surface_resolution - Surface resolution (default: 5124)
%
% Example:
%   S04_betaPowerInMNI(1, 5124)

if nargin < 1
    error('Subject ID is required');
end
if nargin < 2
    surface_resolution = 5124; % Default resolution
end

restoredefaultpath;
clearvars -except subjID surface_resolution; % Keep inputs
close all; clc;

%% Setup and Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);

fprintf('=== MEG Beta Power Analysis in MNI Space ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Load Complex Beta Data
% Load the complex beta data created by S03_betaPowerInSource.m
beta_data_path = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon/sub-%02d_task-mgs_complexBeta_allTargets_%d.mat', subjID, subjID, surface_resolution);

if ~exist(beta_data_path, 'file')
    error('Complex beta data not found at: %s\nPlease run S03_betaPowerInSource.m first!', beta_data_path);
end

fprintf('Loading complex beta data from: %s\n', beta_data_path);
load(beta_data_path);

fprintf('Loaded complex beta data for %d targets\n', length(target_locations));

%% Load Forward Model for MNI Space Information
% Load the forward model to get MNI space coordinates and source model
forward_model_path = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon/sub-%02d_task-mgs_forwardModel.mat', subjID, subjID);

if ~exist(forward_model_path, 'file')
    error('Forward model not found at: %s\nPlease run S01_ForwardModelMNI.m first!', forward_model_path);
end

fprintf('Loading forward model from: %s\n', forward_model_path);
load(forward_model_path);

%% Compute Global Average Power Across All Targets
fprintf('Computing global average power across all targets...\n');

% Define target locations (1-10)
target_locations = 1:10;
all_power_data = [];

% Collect power data from all targets
for target = target_locations
    if isempty(sourceDataByTarget{target})
        fprintf('  No data for target %d\n', target);
        continue;
    end
    
    fprintf('  Collecting power data from target %d...\n', target);
    
    % Get the complex beta data for this target
    sourcedataTarget = sourceDataByTarget{target};
    
    % Compute power (magnitude squared of complex signal) for all trials
    powerAll = sourcedataTarget;
    powerAll.trial = cellfun(@(x) abs(x).^2, sourcedataTarget.trial, 'UniformOutput', false);
    
    % Average across trials
    cfg = [];
    cfg.keeptrials = 'no';
    powerAll_avg = ft_timelockanalysis(cfg, powerAll);
    
    % Store power data for this target
    all_power_data = cat(3, all_power_data, powerAll_avg.avg);
end

% Compute global average power across all targets and all locations
fprintf('Computing global average power...\n');
global_avg_power_time = mean(all_power_data, 2, 'omitnan'); % Average across targets for each time point

fprintf('Global average power computed across %d targets\n', size(all_power_data, 3));

%% Process Each Target Location for Relative Power
fprintf('Processing each target location for relative power...\n');

powerDataByTarget = cell(10, 1);

for target = target_locations
    fprintf('Processing target location %d...\n', target);
    
    if isempty(sourceDataByTarget{target})
        fprintf('  No data for target %d\n', target);
        continue;
    end
    
    % Get the complex beta data for this target
    sourcedataTarget = sourceDataByTarget{target};
    
    % Compute power (magnitude squared of complex signal) for all trials
    fprintf('  Computing power for all trials...\n');
    
    % Compute power for all trials
    powerAll = sourcedataTarget;
    powerAll.trial = cellfun(@(x) abs(x).^2, sourcedataTarget.trial, 'UniformOutput', false);
    
    % Average across trials
    cfg = [];
    cfg.keeptrials = 'no';
    powerAll_avg = ft_timelockanalysis(cfg, powerAll);
    
    % Compute relative power: power at each location / global average power
    fprintf('  Computing relative power...\n');
    
    % Compute relative power for each location and time point
    relative_power_all = powerAll_avg.avg ./ global_avg_power_time;
    
    % Define time windows for analysis (-1.5 to 1.7s with 200ms steps)
    time_start = -1.5;
    time_end = 1.7;
    time_step = 0.2; % 200ms
    time_windows = time_start:time_step:(time_end - time_step);
    
    % Compute relative power for each time window
    relative_power_windows = zeros(size(relative_power_all, 1), length(time_windows), 'single');
    
    for tw = 1:length(time_windows)
        window_start = time_windows(tw);
        window_end = window_start + time_step;
        
        % Find time indices for this window
        time_idx = powerAll_avg.time >= window_start & powerAll_avg.time < window_end;
        
        if sum(time_idx) > 0
            % Average relative power across this time window
            relative_power_windows(:, tw) = single(mean(relative_power_all(:, time_idx), 2, 'omitnan'));
        else
            % No data in this window, set to NaN
            relative_power_windows(:, tw) = single(NaN);
        end
    end
    
    % Store relative power data in single precision for efficiency
    powerDataByTarget{target} = struct();
    powerDataByTarget{target}.relative_power_windows = relative_power_windows;
    powerDataByTarget{target}.time_windows = time_windows;
    powerDataByTarget{target}.time_step = time_step;
    
    fprintf('  Target %d processing complete\n', target);
end

fprintf('Relative power analysis complete for all targets\n');

%% Create MNI Space Visualization Structure
fprintf('Creating MNI space visualization structure...\n');

% Use the source model from the forward model (already in MNI space)
sourcemodel_mni = sourcemodel_aligned_5124; % Use the 5124 resolution source model

% Create visualization structure for each target
for target = target_locations
    if isempty(powerDataByTarget{target})
        continue;
    end
    
    fprintf('Creating visualization structure for target %d...\n', target);
    
    % Get relative power data for all time windows
    relative_power_windows = powerDataByTarget{target}.relative_power_windows;
    time_windows = powerDataByTarget{target}.time_windows;
    
    % Create source visualization structure
    sourceVisualize = struct();
    sourceVisualize.pos = sourcemodel_mni.pos;
    sourceVisualize.tri = sourcemodel_mni.tri;
    sourceVisualize.inside = sourcemodel_mni.inside;
    sourceVisualize.unit = sourcemodel_mni.unit;
    sourceVisualize.coordsys = 'mni';
    sourceVisualize.relative_power_windows = relative_power_windows;
    sourceVisualize.time_windows = time_windows;
    
    % Store for this target
    powerDataByTarget{target}.sourceVisualize = sourceVisualize;
end

%% Visualize Results in MNI Space
fprintf('Creating MNI space visualizations...\n');

% Create figure for all targets
figure('Position', [100, 100, 1200, 800]);

% Define subplot layout (2 rows, 5 columns for 10 targets)
n_targets = length(target_locations);
n_cols = 5;
n_rows = ceil(n_targets / n_cols);

for i = 1:n_targets
    target = target_locations(i);
    
    if isempty(powerDataByTarget{target}) || ~isfield(powerDataByTarget{target}, 'sourceVisualize')
        continue;
    end
    
    subplot(n_rows, n_cols, i);
    
    % Get relative power data for a specific time window (0.8-1.0s)
    relative_power_windows = powerDataByTarget{target}.sourceVisualize.relative_power_windows;
    time_windows = powerDataByTarget{target}.sourceVisualize.time_windows;
    
    % Find the time window closest to 0.8-1.0s
    target_window_start = 0.8;
    [~, window_idx] = min(abs(time_windows - target_window_start));
    relative_power_data = relative_power_windows(:, window_idx);
    
    % Create 3D scatter plot
    scatter3(sourcemodel_mni.pos(:,1), sourcemodel_mni.pos(:,2), sourcemodel_mni.pos(:,3), ...
             20, relative_power_data, 'filled');
    
    % Set color limits for consistent scaling (relative power around 1.0)
    caxis([0, 2]);
    colormap('jet');
    colorbar;
    
    % Set view angle (posterior view)
    view(0, 40);
    
    % Labels and title
    title(sprintf('Target %d', target));
    xlabel('X (mm)');
    ylabel('Y (mm)');
    zlabel('Z (mm)');
    
    % Set axis properties
    axis equal;
    axis tight;
    grid on;
end

% Overall title
sgtitle(sprintf('Subject %02d: Beta Relative Power in MNI Space (Targets 1-10)', subjID));

%% Save Results
fprintf('Saving MNI space analysis results...\n');

output_dir = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon', subjID);
mni_data_path = fullfile(output_dir, sprintf('sub-%02d_task-mgs_relativePowerMNI_%d.mat', subjID, surface_resolution));

% Convert to single precision for efficiency
fprintf('Converting data to single precision for efficiency...\n');
for target = target_locations
    if ~isempty(powerDataByTarget{target})
        % Convert time windows to single precision
        powerDataByTarget{target}.time_windows = single(powerDataByTarget{target}.time_windows);
        
        % relative_power_windows is already single precision
    end
end

% Save power and relative power data in single precision
save(mni_data_path, 'powerDataByTarget', 'target_locations', 'subjID', 'surface_resolution', 'sourcemodel_mni', 'global_avg_power_time', '-v7.3');
fprintf('MNI space analysis saved to: %s\n', mni_data_path);

% Print summary
fprintf('Summary of MNI space analysis:\n');
fprintf('  Time range: %.1f to %.1f s\n', time_start, time_end);
fprintf('  Time step: %.1f s (%.0f ms)\n', time_step, time_step*1000);
fprintf('  Number of time windows: %d\n', length(time_windows));
for target = target_locations
    if ~isempty(powerDataByTarget{target}) && isfield(powerDataByTarget{target}, 'sourceVisualize')
        fprintf('  Target %d: Relative power computed for %d time windows\n', target, size(powerDataByTarget{target}.relative_power_windows, 2));
    else
        fprintf('  Target %d: No data\n', target);
    end
end

fprintf('\nMNI space relative power analysis complete!\n');
