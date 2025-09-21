function S05_DimensionalityAnalysis(subjID, surface_resolution)
% S05_DimensionalityAnalysis - Analyze dimensionality of beta power data
%
% This script performs eigen decomposition on beta power data at each time point
% to estimate the intrinsic dimensionality of the neural data.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, ...)
%   surface_resolution - Surface resolution (5124 or 8196)
%
% Outputs:
%   - Saves results to MAT file
%   - Creates visualization plots
%   - Saves figures as FIG and PNG files
%
% Example:
%   S05_DimensionalityAnalysis(1, 5124);

[~, hostname] = system('hostname');
hostname = strtrim(hostname);

% Check for common HPC indicators
is_hpc = contains(hostname, {'login', 'compute', 'node', 'hpc'}) || ...
         exist('/etc/slurm', 'dir') || ...
         ~isempty(getenv('SLURM_JOB_ID')) || ...
         ~isempty(getenv('PBS_JOBID'));

%% Initialize
fprintf('\n=== MEG Beta Power Dimensionality Analysis ===\n');
fprintf('Environment: %s\n', getenv('HOSTNAME'));

% Check available memory
if ispc
    [~, meminfo] = memory;
    fprintf('Available memory: %.1f GB\n', meminfo.PhysicalMemory.Available / 1e9);
else
    fprintf('Memory check not available on this system\n');
end
fprintf('Detected HPC: %s\n', string(~isempty(getenv('SLURM_JOB_ID'))));
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Set up parameters
baseline_start = -0.5; % seconds (baseline window start)
baseline_end = 0.0;    % seconds (baseline window end)
time_points = -0.5:0.1:1.5; % Time points to analyze
n_time_points = length(time_points);

% Define target angles
target_angles = [0, 25, 50, 130, 155, 180, 205, 230, 310, 335]; % degrees
n_targets = length(target_angles);

% Dimensionality analysis parameters
explained_variance_threshold = 0.95; % Threshold for explained variance (for comparison)
eigenvalue_threshold = 1.0;          % Threshold for eigenvalues (for comparison)

%% Set up paths
if is_hpc
    % HPC paths
    base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
    circstat_path = '/scratch/mdd9787/meg_prf_greene/CircStat2012a/';
else
    % Local paths
    base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    circstat_path = '/d/DATD/hyper/toolboxes/CircStat2012a';
end

% Add circstat toolbox
addpath(circstat_path);

% Set up output directory
output_dir = fullfile(base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'dimensionality_analysis');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load data
fprintf('\n=== Loading Data ===\n');

% Load complex beta data
beta_file = fullfile(base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_complexBeta_allTargets_%d.mat', subjID, surface_resolution));

if ~exist(beta_file, 'file')
    error('Complex beta data not found at:\n%s\nPlease run S03_betaPowerInSource.m first!', beta_file);
end

fprintf('Loading complex beta data...\n');
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

% Load source positions
source_file = fullfile(base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, surface_resolution));

if ~exist(source_file, 'file')
    error('Source space data not found at:\n%s\nPlease run S01_ForwardModelMNI.m first!', source_file);
end

fprintf('Loading source positions...\n');
load(source_file);
% Check what variables are available
if exist('sourceSpaceData', 'var')
    fprintf('  Total sources: %d\n', length(sourceSpaceData.pos));
    all_sources = 1:length(sourceSpaceData.pos);
elseif exist('pos', 'var')
    fprintf('  Total sources: %d\n', length(pos));
    all_sources = 1:length(pos);
else
    % If we can't find source positions, use the number from the beta data
    fprintf('  Source positions not found, using all sources from beta data\n');
    all_sources = 1:5124; % Default for 5124 resolution
end

% Use all sources (whole brain)
fprintf('  Using all sources: %d (100.0%%)\n', length(all_sources));

fprintf('Target angles: %s\n', mat2str(target_angles));
fprintf('Time points: %s\n', mat2str(time_points));

%% Compute global baseline statistics once
fprintf('\n=== Computing Global Baseline ===\n');
fprintf('Computing baseline from -0.5s to 0.0s across all trials...\n');

% Initialize baseline data collection
all_baseline_data = [];
baseline_trial_count = 0;

% Collect baseline data from all trials
for target = 1:n_targets
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
        all_complex = complex_data(all_sources, :);
        
        % Compute power (magnitude squared)
        power_data = abs(all_complex).^2;
        
        % Calculate baseline power (average across baseline window)
        baseline_power = mean(power_data(:, baseline_idx), 2);
        
        % Store baseline data
        all_baseline_data(baseline_trial_count, :) = baseline_power';
    end
end

% Calculate global baseline statistics
if baseline_trial_count > 0
    global_baseline_mean = mean(all_baseline_data, 1);
    global_baseline_std = std(all_baseline_data, 0, 1);
    fprintf('  Computed global baseline from %d trials\n', baseline_trial_count);
    fprintf('  Baseline mean: %.4f ± %.4f (across sources)\n', mean(global_baseline_mean), mean(global_baseline_std));
else
    error('No baseline data found for global baseline calculation');
end

%% Dimensionality analysis at each time point
fprintf('\n=== Dimensionality Analysis ===\n');

% Initialize results
results = struct();
results.time_points = time_points;
results.n_sources = length(all_sources);
results.total_trials = total_trials;
results.explained_variance_threshold = explained_variance_threshold;
results.eigenvalue_threshold = eigenvalue_threshold;

% Storage for each time point
results.eigenvalues = cell(n_time_points, 1);
results.normalized_eigenvalues = cell(n_time_points, 1);
results.explained_variance = cell(n_time_points, 1);
results.cumulative_variance = cell(n_time_points, 1);
results.n_components_95 = zeros(n_time_points, 1);
results.n_components_eigenvalue = zeros(n_time_points, 1);
results.effective_dimensionality = zeros(n_time_points, 1);  % Participation ratio: 1/Σλ̃ᵢ²
results.condition_number = zeros(n_time_points, 1);
results.trial_count = zeros(n_time_points, 1);

fprintf('Analyzing %d time points...\n', n_time_points);

for tp = 1:n_time_points
    current_time = time_points(tp);
    fprintf('\n--- Time Point %d/%d: %.1fs ---\n', tp, n_time_points, current_time);
    
    % Define analysis window (0.2s window centered on time point)
    window_half = 0.1;
    time_start = current_time - window_half;
    time_end = current_time + window_half;
    
    fprintf('  Analysis window: %.1f-%.1fs\n', time_start, time_end);
    
    % Extract data for this time point
    [all_trials, all_targets] = extract_data_for_timepoint(sourceDataByTarget, all_sources, ...
        target_angles, time_start, time_end, global_baseline_mean);
    
    if isempty(all_trials)
        fprintf('  No data available for this time point\n');
        continue;
    end
    
    results.trial_count(tp) = size(all_trials, 1);
    fprintf('  Processed %d trials\n', results.trial_count(tp));
    
    % Estimate memory requirements
    n_trials = size(all_trials, 1);
    n_sources = size(all_trials, 2);
    estimated_memory_gb = (n_trials * n_sources * 8 * 2 + n_sources^2 * 8) / 1e9; % 8 bytes per double
    fprintf('  Estimated memory requirement: %.2f GB\n', estimated_memory_gb);
    
    % Apply z-score normalization across trials
    fprintf('  Applying z-score normalization across trials...\n');
    all_trials_normalized = zscore(all_trials, 0, 1);
    
    % Perform eigen decomposition
    fprintf('  Performing eigen decomposition...\n');
    
    % Compute covariance matrix (memory efficient)
    fprintf('  Computing covariance matrix...\n');
    cov_matrix = cov(all_trials_normalized);
    
    % Clear large data to free memory
    clear all_trials_normalized;
    
    % Eigen decomposition
    fprintf('  Performing eigen decomposition...\n');
    [eigenvectors, eigenvalues] = eig(cov_matrix);
    eigenvalues = diag(eigenvalues);
    
    % Clear covariance matrix to free memory
    clear cov_matrix;
    
    % Sort eigenvalues in descending order
    [eigenvalues, sort_idx] = sort(eigenvalues, 'descend');
    eigenvectors = eigenvectors(:, sort_idx);
    
    % Store results
    results.eigenvalues{tp} = eigenvalues;
    
    % Calculate normalized eigenvalues (λ̃ᵢ = λᵢ / Σⱼ λⱼ)
    total_variance = sum(eigenvalues);
    normalized_eigenvalues = eigenvalues / total_variance;
    results.normalized_eigenvalues{tp} = normalized_eigenvalues;
    
    % Calculate explained variance (same as normalized eigenvalues)
    explained_variance = normalized_eigenvalues;
    results.explained_variance{tp} = explained_variance;
    
    % Calculate cumulative explained variance
    cumulative_variance = cumsum(explained_variance);
    results.cumulative_variance{tp} = cumulative_variance;
    
    % Find number of components for 95% explained variance
    n_components_95 = find(cumulative_variance >= explained_variance_threshold, 1);
    if isempty(n_components_95)
        n_components_95 = length(eigenvalues);
    end
    results.n_components_95(tp) = n_components_95;
    
    % Find number of components above eigenvalue threshold
    n_components_eigenvalue = sum(eigenvalues >= eigenvalue_threshold);
    results.n_components_eigenvalue(tp) = n_components_eigenvalue;
    
    % Effective dimensionality using participation ratio: dim = 1 / Σᵢ λ̃ᵢ²
    participation_ratio = 1 / sum(normalized_eigenvalues.^2);
    results.effective_dimensionality(tp) = participation_ratio;
    
    % Condition number (ratio of largest to smallest eigenvalue)
    if min(eigenvalues) > 0
        results.condition_number(tp) = max(eigenvalues) / min(eigenvalues);
    else
        results.condition_number(tp) = Inf;
    end
    
    fprintf('  Eigenvalues: %.2f to %.2f (range: %.2f)\n', ...
        min(eigenvalues), max(eigenvalues), max(eigenvalues) - min(eigenvalues));
    fprintf('  Components for 95%% variance: %d\n', n_components_95);
    fprintf('  Components above threshold: %d\n', n_components_eigenvalue);
    fprintf('  Effective dimensionality (participation ratio): %.2f\n', results.effective_dimensionality(tp));
    fprintf('  Condition number: %.2e\n', results.condition_number(tp));
end

%% Calculate summary statistics
fprintf('\n=== Summary Statistics ===\n');

% Overall statistics
valid_idx = results.trial_count > 0;
if any(valid_idx)
    mean_dimensionality = mean(results.effective_dimensionality(valid_idx));
    std_dimensionality = std(results.effective_dimensionality(valid_idx));
    min_dimensionality = min(results.effective_dimensionality(valid_idx));
    max_dimensionality = max(results.effective_dimensionality(valid_idx));
    
    mean_condition_number = mean(results.condition_number(valid_idx));
    std_condition_number = std(results.condition_number(valid_idx));
    
    fprintf('Effective dimensionality:\n');
    fprintf('  Mean: %.1f ± %.1f\n', mean_dimensionality, std_dimensionality);
    fprintf('  Range: %d - %d\n', min_dimensionality, max_dimensionality);
    fprintf('Condition number:\n');
    fprintf('  Mean: %.2e ± %.2e\n', mean_condition_number, std_condition_number);
    fprintf('  Range: %.2e - %.2e\n', min(results.condition_number(valid_idx)), max(results.condition_number(valid_idx)));
end

%% Create visualization
fprintf('\n=== Creating Visualization ===\n');

% Create figure
fig = figure('Position', [100, 100, 1400, 1000]);

% Plot 1: Effective dimensionality over time (participation ratio)
subplot(2, 3, 1);
plot(time_points(valid_idx), results.effective_dimensionality(valid_idx), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Time (s)');
ylabel('Effective Dimensionality (Participation Ratio)');
title('Effective Dimensionality Over Time');
grid on;
xline(0, 'r--', 'LineWidth', 1, 'Alpha', 0.7);
xline(0.8, 'g--', 'LineWidth', 1, 'Alpha', 0.7);
legend('Participation Ratio', 'Stimulus Onset', 'Memory Period', 'Location', 'best');

% Plot 2: Condition number over time
subplot(2, 3, 2);
semilogy(time_points(valid_idx), results.condition_number(valid_idx), 'r-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Time (s)');
ylabel('Condition Number (log scale)');
title('Condition Number Over Time');
grid on;
xline(0, 'r--', 'LineWidth', 1, 'Alpha', 0.7);
xline(0.8, 'g--', 'LineWidth', 1, 'Alpha', 0.7);

% Plot 3: Number of trials per time point
subplot(2, 3, 3);
bar(time_points(valid_idx), results.trial_count(valid_idx), 'FaceColor', 'green', 'FaceAlpha', 0.7);
xlabel('Time (s)');
ylabel('Number of Trials');
title('Trial Count Per Time Point');
grid on;
xline(0, 'r--', 'LineWidth', 1, 'Alpha', 0.7);
xline(0.8, 'g--', 'LineWidth', 1, 'Alpha', 0.7);

% Plot 4: Eigenvalue spectrum for best time point
subplot(2, 3, 4);
[~, best_tp] = min(results.effective_dimensionality(valid_idx));
best_tp_idx = find(valid_idx, best_tp);
if ~isempty(best_tp_idx)
    eigenvalues = results.eigenvalues{best_tp_idx};
    semilogy(1:min(50, length(eigenvalues)), eigenvalues(1:min(50, length(eigenvalues))), 'b-o', 'LineWidth', 2, 'MarkerSize', 4);
    xlabel('Component Index');
    ylabel('Eigenvalue (log scale)');
    title(sprintf('Eigenvalue Spectrum (t=%.1fs)', time_points(best_tp_idx)));
    grid on;
end

% Plot 5: Cumulative explained variance for best time point
subplot(2, 3, 5);
if ~isempty(best_tp_idx)
    cumulative_variance = results.cumulative_variance{best_tp_idx};
    plot(1:min(100, length(cumulative_variance)), cumulative_variance(1:min(100, length(cumulative_variance))), 'g-o', 'LineWidth', 2, 'MarkerSize', 4);
    xlabel('Number of Components');
    ylabel('Cumulative Explained Variance');
    title(sprintf('Cumulative Variance (t=%.1fs)', time_points(best_tp_idx)));
    grid on;
    yline(0.95, 'r--', 'LineWidth', 2, 'Alpha', 0.7);
    legend('Cumulative Variance', '95% Threshold', 'Location', 'best');
end

% Plot 6: Summary statistics
subplot(2, 3, 6);
text(0.1, 0.9, sprintf('Subject: %d', subjID), 'FontSize', 12, 'FontWeight', 'bold');
text(0.1, 0.8, sprintf('Surface Resolution: %d', surface_resolution), 'FontSize', 12);
text(0.1, 0.7, sprintf('Total Sources: %d', results.n_sources), 'FontSize', 12);
text(0.1, 0.6, sprintf('Total Trials: %d', results.total_trials), 'FontSize', 12);
text(0.1, 0.5, sprintf('Mean Dimensionality: %.2f ± %.2f', mean_dimensionality, std_dimensionality), 'FontSize', 12, 'Color', 'blue');
text(0.1, 0.4, sprintf('Dimensionality Range: %.2f - %.2f', min_dimensionality, max_dimensionality), 'FontSize', 12, 'Color', 'blue');
text(0.1, 0.3, sprintf('Mean Condition #: %.2e', mean_condition_number), 'FontSize', 12, 'Color', 'red');
text(0.1, 0.2, sprintf('Variance Threshold: %.1f%%', explained_variance_threshold*100), 'FontSize', 12);
text(0.1, 0.1, sprintf('Eigenvalue Threshold: %.1f', eigenvalue_threshold), 'FontSize', 12);
axis off;

sgtitle(sprintf('Beta Power Dimensionality Analysis - Subject %d', subjID), 'FontSize', 16, 'FontWeight', 'bold');

%% Save results
fprintf('\n=== Saving Results ===\n');

% Save results
results_file = fullfile(output_dir, sprintf('sub-%02d_dimensionality_%d.mat', subjID, surface_resolution));
save(results_file, 'results', 'baseline_start', 'baseline_end', 'explained_variance_threshold', 'eigenvalue_threshold');
fprintf('Results saved to: %s\n', results_file);

% Save figure
fig_file = fullfile(output_dir, sprintf('sub-%02d_dimensionality_%d.fig', subjID, surface_resolution));
savefig(fig, fig_file);
fprintf('Figure saved to: %s\n', fig_file);

% Save PNG
png_file = fullfile(output_dir, sprintf('sub-%02d_dimensionality_%d.png', subjID, surface_resolution));
print(fig, png_file, '-dpng', '-r300');
fprintf('PNG saved to: %s\n', png_file);

%% Summary
fprintf('\n=== Dimensionality Analysis Summary ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d\n', surface_resolution);
fprintf('Total sources: %d\n', results.n_sources);
fprintf('Total trials: %d\n', results.total_trials);
fprintf('Time points analyzed: %d\n', sum(valid_idx));
fprintf('Mean effective dimensionality: %.2f ± %.2f\n', mean_dimensionality, std_dimensionality);
fprintf('Dimensionality range: %.2f - %.2f\n', min_dimensionality, max_dimensionality);
fprintf('Mean condition number: %.2e\n', mean_condition_number);

fprintf('\nDimensionality analysis completed!\n');

end

%% Helper Functions

function [all_trials, all_targets] = extract_data_for_timepoint(sourceDataByTarget, all_sources, ...
    target_angles, time_start, time_end, global_baseline_mean)
% Extract data for specific time point

all_trials = [];
all_targets = [];
trial_count = 0;

% Extract data for this time point
for target = 1:length(sourceDataByTarget)
    if isempty(sourceDataByTarget{target})
        continue;
    end
    
    % Get time vector
    time_vec = sourceDataByTarget{target}.time{1};
    
    % Find time indices for analysis window
    time_idx = time_vec >= time_start & time_vec <= time_end;
    
    if sum(time_idx) == 0
        continue;
    end
    
    % Extract data for this target
    n_trials = length(sourceDataByTarget{target}.trial);
    
    for trial = 1:n_trials
        trial_count = trial_count + 1;
        
        % Get complex beta data for this trial
        complex_data = sourceDataByTarget{target}.trial{trial};
        
        % Select all sources
        all_complex = complex_data(all_sources, :);
        
        % Compute power (magnitude squared)
        power_data = abs(all_complex).^2;
        
        % Calculate analysis window power (average across analysis window)
        analysis_power = mean(power_data(:, time_idx), 2);
        
        % Apply global baseline correction: (analysis - global_baseline_mean)
        avg_power = (analysis_power - global_baseline_mean');
        
        % Store data
        all_trials(trial_count, :) = avg_power';
        all_targets(trial_count) = target_angles(target);
    end
end

end
