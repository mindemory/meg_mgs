function S06_SourceContributionAnalysis(subjID, surface_resolution)
%% Source Contribution Analysis for PCA Components
% Analyze which brain sources contribute most to the first N PCA components at each time point
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   surface_resolution - Surface resolution (default: 5124)
%
% Example:
%   S06_SourceContributionAnalysis(1, 5124, false)

if nargin < 1
    error('Subject ID is required');
end
if nargin < 2
    surface_resolution = 5124; % Default resolution
end

restoredefaultpath;
clearvars -except subjID surface_resolution displayFigure; % Keep inputs
close all; clc;

%% Environment Detection and Path Setup
% Detect if running on HPC or local machine
[~, hostname] = system('hostname');
hostname = strtrim(hostname);

% Check for common HPC indicators
is_hpc = contains(hostname, {'login', 'compute', 'node', 'hpc'}) || ...
    exist('/etc/slurm', 'dir') || ...
    ~isempty(getenv('SLURM_JOB_ID')) || ...
    ~isempty(getenv('PBS_JOBID'));

fprintf('=== MEG Beta Power Analysis in MNI Space ===\n');
fprintf('Environment: %s\n', hostname);
fprintf('Detected HPC: %s\n', string(is_hpc));
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Setup paths based on environment
if is_hpc
    % HPC paths
    fieldtrip_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/';
    project_path = '/scratch/mdd9787/meg_prf_greene/megScripts';
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
    ft_gifti_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/external/gifti';
else
    % Local machine paths
    fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    ft_gifti_path = '/d/DATD/hyper/software/fieldtrip-20250318/external/gifti'; % Add Gifti toolbox for .surf.gii files

end

% Setup and Initialization
addpath(fieldtrip_path);
addpath(ft_gifti_path);
addpath(genpath(project_path));
ft_defaults;
ft_hastoolbox('spm12', 1);

%% Initialize File Paths
% Complex beta data path (input)
beta_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_complexBeta_allTargets_%d.mat', subjID, surface_resolution));

% Forward model path (input)
forward_model_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_forwardModel.mat', subjID));

% Output directory and file paths
output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'source_contribution_analysis');



%% Load Complex Beta Data
% Load the complex beta data created by S03_betaPowerInSource.m
if ~exist(beta_data_path, 'file')
    error('Complex beta data not found at: %s\nPlease run S03_betaPowerInSource.m first!', beta_data_path);
end

fprintf('Loading complex beta data from: %s\n', beta_data_path);
load(beta_data_path);

%% Load Forward Model for MNI Space Information
% Load the forward model to get MNI space coordinates and source model
if ~exist(forward_model_path, 'file')
    error('Forward model not found at: %s\nPlease run S01_ForwardModelMNI.m first!', forward_model_path);
end

fprintf('Loading forward model from: %s\n', forward_model_path);
load(forward_model_path);

%% Define time windows for analysis (-1.0 to 2.0s with 100ms steps w/ moving window of 200ms)
time_start = -1.0;
time_end = 2;
time_step = 0.1; % 100ms
time_windows = time_start:time_step:time_end;

%% Collect All Trial Data Efficiently
fprintf('Collecting power data for all trials and time points...\n');

% First pass: count total trials to preallocate
total_trials = 0;
for target = target_locations
    if ~isempty(sourceDataByTarget{target})
        total_trials = total_trials + length(sourceDataByTarget{target}.trial);
    end
end

% Preallocate 3D arrays: [surface_resolution, n_trials, n_time_windows]
all_trial_data = zeros(surface_resolution, total_trials, length(time_windows), 'single');
all_target_labels = zeros(total_trials, 1);
trial_count = 0;

% Process each target location
for target = target_locations
    fprintf('Processing target %d...\n', target);
    
    if isempty(sourceDataByTarget{target})
        fprintf('  No data for target %d\n', target);
        continue;
    end
    
    % Get the complex beta data for this target
    sourcedataTarget = sourceDataByTarget{target};
    n_trials = length(sourcedataTarget.trial);
    
    % Process each trial
    for trial = 1:n_trials
        trial_count = trial_count + 1;
        
        % Compute power (magnitude squared)
        power_data = abs( sourcedataTarget.trial{trial}).^2;
        
        % Process each time window
        for tw = 1:length(time_windows)
            window_start = time_windows(tw) - time_step;
            window_end = time_windows(tw) + time_step;
            
            % Find time indices for this window
            time_vec = sourcedataTarget.time{1};
            time_idx = time_vec >= window_start & time_vec < window_end;
            
            if sum(time_idx) > 0
                % Average power across this time window
                window_power = mean(power_data(:, time_idx), 2);
                
                % Store data in 3D array
                all_trial_data(:, trial_count, tw) = single(window_power);
            end
        end
        
        % Store target label
        all_target_labels(trial_count) = target;
    end
    
    fprintf('  Target %d: %d trials processed\n', target, n_trials);
end


%% Compute Relative Power Efficiently
fprintf('Computing relative power...\n');

global_avg_power_time = mean(all_trial_data, 2);
% Compute relative power for all trials at once
relative_power_data = all_trial_data  ./ global_avg_power_time;

fprintf('Relative power computation complete\n');

%% Compute Effective Dimensionality as a Function of Time
fprintf('Computing effective dimensionality as a function of time...\n');

% Initialize arrays to store dimensionality results
n_time_windows = length(time_windows);
effective_dimensionality = zeros(n_time_windows, 1);
participation_ratio = zeros(n_time_windows, 1);

% Process each time window
for tw = 1:n_time_windows
    fprintf('Processing time window %d/%d (%.2fs)...\n', tw, n_time_windows, time_windows(tw));
    
    % Get data for this time window: [n_sources × n_trials]
    time_data = relative_power_data(:, :, tw);
    
    % Remove any NaN values
    valid_trials = ~any(isnan(time_data), 1);
    if sum(valid_trials) < 2
        fprintf('  Not enough valid trials for time window %.2fs\n', time_windows(tw));
        effective_dimensionality(tw) = NaN;
        participation_ratio(tw) = NaN;
        continue;
    end
    
    time_data_clean = time_data(:, valid_trials);
    
    % Compute covariance matrix
    cov_matrix = cov(time_data_clean');
    
    % Perform eigendecomposition
    [eigenvectors, eigenvalues] = eig(cov_matrix);
    eigenvalues = diag(eigenvalues);
    
    % Sort eigenvalues in descending order
    [eigenvalues, sort_idx] = sort(eigenvalues, 'descend');
    eigenvectors = eigenvectors(:, sort_idx);
    
    % Remove negative eigenvalues (numerical artifacts)
    positive_eigenvals = eigenvalues(eigenvalues > 0);
    
    if isempty(positive_eigenvals)
        fprintf('  No positive eigenvalues for time window %.2fs\n', time_windows(tw));
        effective_dimensionality(tw) = NaN;
        participation_ratio(tw) = NaN;
        continue;
    end
    
    % Normalize eigenvalues
    normalized_eigenvals = positive_eigenvals / sum(positive_eigenvals);
    
    % Compute participation ratio (effective dimensionality)
    participation_ratio(tw) = 1 / sum(normalized_eigenvals.^2);
    effective_dimensionality(tw) = participation_ratio(tw);
    
    fprintf('  Time %.2fs: Effective dimensionality = %.2f\n', time_windows(tw), effective_dimensionality(tw));
end

% Create dimensionality visualization
fprintf('Creating dimensionality visualization...\n');

figure('Position', [100, 100, 1200, 600]);

% Plot 1: Effective dimensionality over time
subplot(1, 2, 1);
plot(time_windows, effective_dimensionality, 'b-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Effective Dimensionality');
title('Effective Dimensionality Over Time');
grid on;
xline(0, 'k--', 'Stimulus Onset', 'LineWidth', 1);
xline(0.8, 'g--', 'Memory Period', 'LineWidth', 1);

% Plot 2: Participation ratio over time
subplot(1, 2, 2);
plot(time_windows, participation_ratio, 'r-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Participation Ratio');
title('Participation Ratio Over Time');
grid on;
xline(0, 'k--', 'Stimulus Onset', 'LineWidth', 1);
xline(0.8, 'g--', 'Memory Period', 'LineWidth', 1);

sgtitle(sprintf('Subject %02d: Dimensionality Analysis (Surface Resolution %d)', subjID, surface_resolution), 'FontSize', 16);

% Save results
fprintf('Saving dimensionality results...\n');
dimensionality_results = struct();
dimensionality_results.time_windows = time_windows;
dimensionality_results.effective_dimensionality = effective_dimensionality;
dimensionality_results.participation_ratio = participation_ratio;
dimensionality_results.subject_id = subjID;
dimensionality_results.surface_resolution = surface_resolution;

fprintf('Dimensionality analysis complete!\n');

% %% Group Data by Target Location and Plot
% fprintf('Grouping data by target location and plotting...\n');

% % Define time window of interest (0.8-1.5s)
% target_time_start = 0.8;
% target_time_end = 1.5;

% % Define baseline window (-0.5 to 0s)
% baseline_time_start = -0.5;
% baseline_time_end = 0.0;

% % Find time window indices
% time_window_indices = find(time_windows >= target_time_start & time_windows <= target_time_end);
% baseline_window_indices = find(time_windows >= baseline_time_start & time_windows <= baseline_time_end);

% if isempty(time_window_indices)
%     error('No time windows found in the specified range (0.8-1.5s)');
% end

% if isempty(baseline_window_indices)
%     error('No time windows found in the baseline range (-0.5 to 0s)');
% end

% fprintf('Time window indices: %s\n', mat2str(time_window_indices));
% fprintf('Time windows: %s\n', mat2str(time_windows(time_window_indices)));
% fprintf('Baseline window indices: %s\n', mat2str(baseline_window_indices));
% fprintf('Baseline time windows: %s\n', mat2str(time_windows(baseline_window_indices)));

% % Group relative power data by target location
% target_angles = [0 25 50 130 155 180 205 230 310 335];
% powerDataByTarget = cell(10, 1);

% for target = 1:10
%     % Find trials for this target
%     target_trials = find(all_target_labels == target);
    
%     if isempty(target_trials)
%         fprintf('  No trials found for target %d\n', target);
%         powerDataByTarget{target} = [];
%         continue;
%     end
    
%     % Extract relative power data for this target across the time window
%     target_relative_power = relative_power_data(:, target_trials, time_window_indices);
    
%     % Extract baseline data for this target
%     target_baseline_power = relative_power_data(:, target_trials, baseline_window_indices);
    
%     % Compute baseline average across time and trials
%     baseline_avg = mean(mean(target_baseline_power, 3), 2);
    
%     % Subtract baseline from the target time window data
%     target_relative_power_baseline_corrected = target_relative_power - baseline_avg;
    
%     % Store data in the format expected by the visualization code
%     powerDataByTarget{target} = struct();
%     powerDataByTarget{target}.relative_power_windows = target_relative_power_baseline_corrected;
%     powerDataByTarget{target}.time_windows = time_windows(time_window_indices);
%     powerDataByTarget{target}.angle = target_angles(target);
    
%     fprintf('  Target %d: %d trials, angle %.0f°, baseline corrected\n', target, length(target_trials), target_angles(target));
% end


% %% Create Second Visualization: Native Subject Space Surface Dots
% fprintf('Creating native subject space surface dot visualization...\n');
% sourcemodel_subject = sourcemodel_aligned_5124; 
% % Create second figure for native space visualization
% figure('Position', [200, 500, 1500, 1000]);
% target_angles = [0, 25, 50, 130, 155, 180, 205, 230, 310, 335]; % degrees
% n_targets = length(target_locations);
% % Create circular subplot layout for native space
% for i = 1:n_targets
%     target = target_locations(i);

%     if isempty(powerDataByTarget{target})
%         continue;
%     end

%     % Calculate position for circular layout
%     angle_rad = deg2rad(target_angles(i));
%     radius = 0.35; % Distance from center
%     x_pos = 0.5 + radius * cos(angle_rad); % Center around 0.5
%     y_pos = 0.5 + radius * sin(angle_rad); % Center around 0.5
%     subplot_size = 0.12; % Size of each subplot

%     % Create subplot at calculated position
%     subplot('Position', [x_pos - subplot_size/2, y_pos - subplot_size/2, subplot_size, subplot_size]);

%     % Get relative power data for time window 0.8-1.5s
%     relative_power_windows = powerDataByTarget{target}.relative_power_windows;
%     time_windows = powerDataByTarget{target}.time_windows;

%     % Find time windows between 0.8 and 1.5 seconds
%     target_window_start = 0.8;
%     target_window_end = 1.5;
%     window_indices = find(time_windows >= target_window_start & time_windows <= target_window_end);

%     if isempty(window_indices)
%         continue;
%     end

%     % Average relative power across the time window
%     relative_power_data = mean(relative_power_windows(:, window_indices), 2, 'omitnan');

%     % Check data dimensions
%     fprintf('Target %d: sourcemodel_subject.pos size: %s, relative_power_data size: %s\n', ...
%         target, mat2str(size(sourcemodel_subject.pos)), mat2str(size(relative_power_data)));

%     % Create scatter plot in native subject space
%     scatter3(sourcemodel_subject.pos(:,1), sourcemodel_subject.pos(:,2), sourcemodel_subject.pos(:,3), ...
%         20, relative_power_data, 'filled');

%     % Set color limits for consistent scaling
%     caxis([-1 1]);
%     colormap('jet');
%     colorbar;

%     % Set view angle (posterior view)
%     view(0, 40);

%     % Labels and title
%     title(sprintf('Target %d\n%d°\n(Native)', target, target_angles(i)));
%     xlabel('X (mm)');
%     ylabel('Y (mm)');
%     zlabel('Z (mm)');

%     % Set axis properties
%     axis equal;
%     axis tight;
%     grid on;
% end

% % Overall title for native space
% sgtitle(sprintf('Subject %02d: Beta Relative Power in Native Subject Space (Targets 1-10)', subjID));