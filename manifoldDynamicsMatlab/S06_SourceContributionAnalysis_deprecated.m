function S06_SourceContributionAnalysis(subjID, surface_resolution)
% S06_SourceContributionAnalysis - Analyze source contributions to PCA components
%
% This script analyzes which sources contribute most to the first 50 PCA 
% components at each time point, using the existing dimensionality analysis data.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   surface_resolution - Surface resolution (5124 or 8196)
%
% Outputs:
%   - Source contribution analysis results saved as .mat files
%   - Visualization plots saved as .fig and .png files

%% Setup
clearvars -except subjID surface_resolution; close all; clc;

[~, hostname] = system('hostname');
hostname = strtrim(hostname);

% Check for common HPC indicators
is_hpc = contains(hostname, {'login', 'compute', 'node', 'hpc'}) || ...
         exist('/etc/slurm', 'dir') || ...
         ~isempty(getenv('SLURM_JOB_ID')) || ...
         ~isempty(getenv('PBS_JOBID'));

% Check if running on HPC or local
if is_hpc
    % HPC paths
    PROJECT_DIR = '/scratch/mdd9787/meg_prf_greene/MEG_HPC';
    DERIVATIVES_DIR = fullfile(PROJECT_DIR, 'derivatives');
    OUTPUT_DIR = fullfile(DERIVATIVES_DIR, sprintf('sub-%02d', subjID), 'sourceRecon', 'source_contribution_analysis');
    CIRCLE_STATISTICS_DIR = '/scratch/mdd9787/meg_prf_greene/CircStat2012a';
else
    % Local paths
    PROJECT_DIR = '/d/DATD/datd/MEG_MGS/MEG_BIDS';
    DERIVATIVES_DIR = fullfile(PROJECT_DIR, 'derivatives');
    OUTPUT_DIR = fullfile(DERIVATIVES_DIR, sprintf('sub-%02d', subjID), 'sourceRecon', 'source_contribution_analysis');
    CIRCLE_STATISTICS_DIR = '/d/DATD/hyper/toolboxes/CircStat2012a';
end

% Add circular statistics toolbox
addpath(CIRCLE_STATISTICS_DIR);
% Create output directory
if ~exist(OUTPUT_DIR, 'dir')
    mkdir(OUTPUT_DIR);
end

fprintf('=== Source Contribution Analysis ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d\n', surface_resolution);
fprintf('Output directory: %s\n', OUTPUT_DIR);

%% Define Analysis Parameters
fprintf('\n=== Setting Analysis Parameters ===\n');

% Define time points (same as other scripts)
time_points = -0.5:0.1:1.5;
n_timepoints = length(time_points);
n_components = 50; % Number of PCA components to analyze

fprintf('Analysis parameters:\n');
fprintf('  Time points: %d (%.1f to %.1fs)\n', n_timepoints, time_points(1), time_points(end));
fprintf('  Components to analyze: %d\n', n_components);

%% Load Source Data for PCA Computation
fprintf('\n=== Loading Source Data for PCA ===\n');

% Load complex beta data
beta_file = fullfile(DERIVATIVES_DIR, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_complexBeta_allTargets_%d.mat', subjID, surface_resolution));

if ~exist(beta_file, 'file')
    error('Complex beta data not found: %s\nPlease run S03_betaPowerInSource.m first!', beta_file);
end

fprintf('Loading complex beta data from: %s\n', beta_file);
load(beta_file, 'sourceDataByTarget');
fprintf('Loaded complex beta data:\n');
fprintf('  Number of targets: %d\n', length(sourceDataByTarget));

% Count trials per target
total_trials = 0;
for target = 1:length(sourceDataByTarget)
    if ~isempty(sourceDataByTarget{target})
        n_trials = length(sourceDataByTarget{target}.trial);
        fprintf('  Target %d: %d trials\n', target, n_trials);
        total_trials = total_trials + n_trials;
    end
end
fprintf('  Total trials: %d\n', total_trials);

% Load source positions to get number of sources
source_file = fullfile(DERIVATIVES_DIR, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, surface_resolution));

if ~exist(source_file, 'file')
    error('Source space data not found at:\n%s\nPlease run S01_ForwardModelMNI.m first!', source_file);
end

fprintf('Loading source positions...\n');
load(source_file);
% Check what variables are available
if exist('sourceSpaceData', 'var')
    n_sources = length(sourceSpaceData.pos);
    fprintf('  Total sources: %d\n', n_sources);
elseif exist('pos', 'var')
    n_sources = length(pos);
    fprintf('  Total sources: %d\n', n_sources);
else
    % If we can't find source positions, use the number from the beta data
    fprintf('  Source positions not found, using all sources from beta data\n');
    n_sources = 5124; % Default for 5124 resolution
end

% Use all sources (whole brain)
fprintf('  Using all sources: %d (100.0%%)\n', n_sources);

%% Perform Source Contribution Analysis
fprintf('\n=== Performing Source Contribution Analysis ===\n');

% Initialize results
source_contributions = cell(n_timepoints, 1);
top_sources_per_component = cell(n_timepoints, 1);
top_sources_overall = cell(n_timepoints, 1);
component_loadings = cell(n_timepoints, 1);
explained_variance = cell(n_timepoints, 1);

% Compute global average power across all trials and time points for normalization
fprintf('Computing global average power for relative power normalization...\n');
all_power_data = [];
for target = 1:length(sourceDataByTarget)
    if isempty(sourceDataByTarget{target})
        continue;
    end
    n_trials = length(sourceDataByTarget{target}.trial);
    for trial = 1:n_trials
        complex_data = sourceDataByTarget{target}.trial{trial};
        all_complex = complex_data(1:n_sources, :);
        power_data = abs(all_complex).^2;
        all_power_data = [all_power_data, power_data(:)];
    end
end
global_avg_power = mean(all_power_data);

% Compute global baseline from -0.5 to 0.0s
fprintf('Computing global baseline from -0.5-0.0s...\n');
baseline_start = -0.5;
baseline_end = 0.0;

% Initialize baseline data collection
all_baseline_data = [];
baseline_trial_count = 0;

% Collect baseline data from all trials
for target = 1:length(sourceDataByTarget)
    if isempty(sourceDataByTarget{target})
        continue;
    end
    
    % Get time vector
    time_vec = sourceDataByTarget{target}.time{1};
    
    % Find time indices for baseline window
    baseline_idx = time_vec >= baseline_start & time_vec <= baseline_end;
    
    if sum(baseline_idx) == 0
        continue;
    end
    
    % Extract baseline data for this target
    n_trials = length(sourceDataByTarget{target}.trial);
    
    for trial = 1:n_trials
        baseline_trial_count = baseline_trial_count + 1;
        
        % Get complex beta data for this trial
        complex_data = sourceDataByTarget{target}.trial{trial};
        
        % Select all sources
        all_complex = complex_data(1:n_sources, :);
        
        % Use relative power (power / global average power) to remove center bias
        power_data = abs(all_complex).^2;
        relative_power = power_data ./ global_avg_power;
        trial_features = relative_power(:);
        
        % Extract baseline data and compute relative power
        baseline_trial_data = all_complex(:, baseline_idx);
        baseline_power = abs(baseline_trial_data).^2;
        baseline_relative_power = baseline_power ./ global_avg_power;
        baseline_features = mean(baseline_relative_power, 2);
        
        % Store baseline data
        all_baseline_data(baseline_trial_count, :) = baseline_features';
    end
end
% Compute global baseline statistics
global_baseline_mean = mean(all_baseline_data, 1)';
global_baseline_std = std(all_baseline_data, 0, 1)';

% Analyze source contributions at each time point independently
fprintf('\nAnalyzing source contributions at each time point...\n');

for t = 1:n_timepoints
    fprintf('--- Time Point %d/%d: %.1fs ---\n', t, n_timepoints, time_points(t));
    
    % Extract data for this time point
    all_trials_tp = [];
    all_targets_tp = [];
    
    for target = 1:length(sourceDataByTarget)
        if isempty(sourceDataByTarget{target})
            continue;
        end
        
        % Get time vector
        time_vec = sourceDataByTarget{target}.time{1};
        
        % Find time indices for this time point (0.2s window)
        time_window = 0.2; % seconds
        time_start = time_points(t) - time_window/2;
        time_end = time_points(t) + time_window/2;
        
        time_idx = time_vec >= time_start & time_vec <= time_end;
        
        if sum(time_idx) == 0
            continue;
        end
        
        % Extract data for this target
        n_trials = length(sourceDataByTarget{target}.trial);
        
        for trial = 1:n_trials
            % Get complex beta data for this trial
            complex_data = sourceDataByTarget{target}.trial{trial};
            
            % Select all sources
            all_complex = complex_data(1:n_sources, :);
            
            % Extract data for this time window
            trial_data = all_complex(:, time_idx);
            
            % Use relative power (power / global average power) to remove center bias
            power_data = abs(trial_data).^2;
            relative_power = power_data ./ global_avg_power;
            trial_features = relative_power(:);
            
            all_trials_tp = [all_trials_tp, trial_features];
            all_targets_tp = [all_targets_tp, target];
        end
    end
    
    % Apply baseline correction and normalization
    % Ensure dimensions match
    if size(all_trials_tp, 1) ~= size(global_baseline_mean, 1)
        min_dim = min(size(all_trials_tp, 1), size(global_baseline_mean, 1));
        all_trials_tp = all_trials_tp(1:min_dim, :);
        global_baseline_mean = global_baseline_mean(1:min_dim);
    end
    
    all_trials_tp_normalized = (all_trials_tp - global_baseline_mean); 
    all_trials_tp_normalized = zscore(all_trials_tp_normalized, 0, 1);
    
    % Perform PCA on this time point's data
    [coeff, score, latent, ~, explained] = pca(all_trials_tp_normalized');
    
    % Analyze source contributions to first n_components
    source_contrib = zeros(n_sources, n_components);
    top_sources_comp = cell(n_components, 1);
    
    % Since we're using only power, PCA coefficients are directly for sources
    for comp = 1:n_components
        % Get the loading of this component (PCA coefficients)
        comp_loading = coeff(:, comp);
        
        % Use absolute value of loadings as source contributions
        source_loading = abs(comp_loading);
        
        % Store source contributions for this component
        source_contrib(:, comp) = source_loading;
        
        % Find top contributing sources for this component
        [~, source_rank] = sort(source_loading, 'descend');
        top_sources_comp{comp} = source_rank(1:min(100, n_sources)); % Top 100 sources
    end
    
    % Find overall top contributing sources (across all components)
    overall_contrib = sum(source_contrib, 2);
    [~, overall_rank] = sort(overall_contrib, 'descend');
    top_sources_overall{t} = overall_rank(1:min(200, n_sources)); % Top 200 sources overall
    
    % Store results
    source_contributions{t} = source_contrib;
    top_sources_per_component{t} = top_sources_comp;
    component_loadings{t} = coeff(:, 1:n_components);
    explained_variance{t} = explained;
    
    fprintf('  Analyzed %d sources for %d components\n', n_sources, n_components);
    fprintf('  Variance explained by first %d components: %.2f%%\n', n_components, sum(explained(1:n_components)));
end


%% Create Visualizations
fprintf('\n=== Creating Visualizations ===\n');

% Load source positions for visualization
fprintf('Loading source positions for visualization...\n');
% source_data_path = fullfile(OUTPUT_DIR, sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, surface_resolution));
source_data_path = fullfile(DERIVATIVES_DIR, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, surface_resolution));

fprintf('Loading source positions from: %s\n', source_data_path);
source_data = load(source_data_path);
source_pos = source_data.source.pos;
fprintf('  Loaded %d source positions\n', size(source_pos, 1));

% Create visualizations for each time point
fprintf('Creating source contribution visualizations for each time point...\n');

for t = 1:n_timepoints
    fprintf('  Creating visualization for time point %d/21: %.1fs\n', t, time_points(t));
    
    % Calculate average contribution over 50 PC components for each source
    avg_contrib = mean(source_contributions{t}, 2);
    
    % Calculate weighted average contribution (weighted by PC explained variance)
    pc_weights = explained_variance{t}(1:n_components) / 100; % Convert to proportion
    weighted_contrib = sum(source_contributions{t} .* pc_weights', 2);
    
    % Create figure for this time point
    fig = figure('Position', [100, 100, 1400, 600]);
    
    % Plot 1: Average contribution over 50 PC components
    subplot(1, 2, 1);
    scatter3(source_pos(:, 1), source_pos(:, 2), source_pos(:, 3), 50, avg_contrib, 'filled');
    colorbar;
    colormap('hot');
    xlabel('X (mm)');
    ylabel('Y (mm)');
    zlabel('Z (mm)');
    title(sprintf('Average Source Contribution (%.1fs)', time_points(t)));
    axis equal;
    view(45, 30);
    
    % Plot 2: Weighted average contribution over 50 PC components
    subplot(1, 2, 2);
    scatter3(source_pos(:, 1), source_pos(:, 2), source_pos(:, 3), 50, weighted_contrib, 'filled');
    colorbar;
    colormap('hot');
    xlabel('X (mm)');
    ylabel('Y (mm)');
    zlabel('Z (mm)');
    title(sprintf('Weighted Source Contribution (%.1fs)', time_points(t)));
    axis equal;
    view(45, 30);
    
    sgtitle(sprintf('Source Contributions - Subject %d, Time %.1fs (%d vertices)', subjID, time_points(t), surface_resolution));
    
    % Save figure for this time point
    fig_file = fullfile(OUTPUT_DIR, sprintf('sub-%02d_source_contributions_%.1fs_%d.fig', subjID, time_points(t), surface_resolution));
    savefig(fig, fig_file);
    
    % Save PNG for this time point
    png_file = fullfile(OUTPUT_DIR, sprintf('sub-%02d_source_contributions_%.1fs_%d.png', subjID, time_points(t), surface_resolution));
    print(fig, png_file, '-dpng', '-r300');
    
    close(fig); % Close figure to save memory
end

% Create summary figure showing time series of top contributing sources
fig_summary = figure('Position', [100, 100, 1200, 800]);

% Calculate time series of average contributions
avg_contrib_time = zeros(n_timepoints, 1);
weighted_contrib_time = zeros(n_timepoints, 1);
max_contrib_time = zeros(n_timepoints, 1);

for t = 1:n_timepoints
    avg_contrib_time(t) = mean(mean(source_contributions{t}));
    
    % Calculate weighted contribution for this time point
    pc_weights = explained_variance{t}(1:n_components) / 100; % Convert to proportion
    weighted_contrib_t = sum(source_contributions{t} .* pc_weights', 2);
    weighted_contrib_time(t) = mean(weighted_contrib_t);
    
    max_contrib_time(t) = max(mean(source_contributions{t}, 2));
end

% Plot 1: Time series of average contributions
subplot(2, 2, 1);
plot(time_points, avg_contrib_time, 'b-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Average Contribution');
title('Average Source Contribution Over Time');
grid on;

% Plot 2: Time series of weighted contributions
subplot(2, 2, 2);
plot(time_points, weighted_contrib_time, 'r-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Weighted Contribution');
title('Weighted Source Contribution Over Time');
grid on;

% Plot 3: Time series of maximum contributions
subplot(2, 2, 3);
plot(time_points, max_contrib_time, 'g-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Maximum Contribution');
title('Maximum Source Contribution Over Time');
grid on;

% Plot 4: Variance explained over time
subplot(2, 2, 4);
variance_explained = zeros(n_timepoints, 1);
for t = 1:n_timepoints
    if t <= length(explained_variance) && ~isempty(explained_variance{t})
        variance_explained(t) = sum(explained_variance{t}(1:min(10, length(explained_variance{t}))));
    end
end
plot(time_points, variance_explained, 'm-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Variance Explained (%)');
title('Variance Explained by First 10 Components');
grid on;

sgtitle(sprintf('Source Contribution Analysis Summary - Subject %d (%d vertices)', subjID, surface_resolution));

% Save summary figure
fig_file = fullfile(OUTPUT_DIR, sprintf('sub-%02d_source_contribution_summary_%d.fig', subjID, surface_resolution));
savefig(fig_summary, fig_file);
fprintf('Summary figure saved to: %s\n', fig_file);

% Save summary PNG
png_file = fullfile(OUTPUT_DIR, sprintf('sub-%02d_source_contribution_summary_%d.png', subjID, surface_resolution));
print(fig_summary, png_file, '-dpng', '-r300');
fprintf('Summary PNG saved to: %s\n', png_file);

%% Save Results
fprintf('\n=== Saving Results ===\n');

% Create results structure
results = struct();
results.subject_id = subjID;
results.surface_resolution = surface_resolution;
results.time_points = time_points;
results.n_sources = n_sources;
results.n_components = n_components;
results.source_contributions = source_contributions;
results.top_sources_per_component = top_sources_per_component;
results.top_sources_overall = top_sources_overall;
results.component_loadings = component_loadings;
results.explained_variance = explained_variance;
results.global_baseline_mean = global_baseline_mean;
results.global_baseline_std = global_baseline_std;

% Save results
results_file = fullfile(OUTPUT_DIR, sprintf('sub-%02d_source_contribution_analysis_%d.mat', subjID, surface_resolution));
save(results_file, 'results');
fprintf('Results saved to: %s\n', results_file);

%% Summary
fprintf('\n=== Source Contribution Analysis Summary ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d\n', surface_resolution);
fprintf('Time points analyzed: %d\n', n_timepoints);
fprintf('Sources analyzed: %d\n', n_sources);
fprintf('Components analyzed: %d\n', n_components);

fprintf('\nSource contribution analysis completed!\n');

end
