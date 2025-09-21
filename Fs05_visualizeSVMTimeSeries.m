function Fs05_visualizeSVMTimeSeries()
%% Fs05_visualizeSVMTimeSeries - Visualize SVM decoding time series results
%
% This script loads SVM decoding results from S05_SVM_Decoding_TimeSeries
% and creates visualizations showing decoding error as a function of time.
%
% Plots:
% 1. Individual subject error time series with SEM
% 2. Group-averaged error time series with SEM across subjects
%
% Dependencies:
%   - S05_SVM_Decoding_TimeSeries.m results (must be run first)
%
% Author: Mrugank Dake
% Date: 2025-01-20

restoredefaultpath;
clearvars; close all; clc;

%% Environment Detection and Path Setup
% Detect if running on HPC or local machine
[~, hostname] = system('hostname');
hostname = strtrim(hostname);

% Check for common HPC indicators
is_hpc = contains(hostname, {'login', 'compute', 'node', 'hpc'}) || ...
         exist('/etc/slurm', 'dir') || ...
         ~isempty(getenv('SLURM_JOB_ID')) || ...
         ~isempty(getenv('PBS_JOBID'));

fprintf('=== MEG SVM Time Series Visualization ===\n');
fprintf('Environment: %s\n', hostname);
fprintf('Detected HPC: %s\n', string(is_hpc));

%% Setup paths based on environment
if is_hpc
    % HPC paths
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
    output_path = '/scratch/mdd9787/meg_prf_greene/figures';
    circstat_path = '/scratch/mdd9787/meg_prf_greene/CircStat2012a/';
else
    % Local machine paths
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    output_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/figures';
    circstat_path = '/d/DATD/hyper/toolboxes/CircStat2012a/';
end

% Add CircStat toolbox to path
addpath(circstat_path);

% Create output directory if it doesn't exist
if ~exist(output_path, 'dir')
    mkdir(output_path);
end

%% Define subjects and surface resolutions
subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32];
surface_resolutions = [5124, 8196];

fprintf('Subjects: %s\n', mat2str(subjects));
fprintf('Surface resolutions: %s\n', mat2str(surface_resolutions));

%% Load data for each subject and surface resolution
all_results = struct();
all_results.subjects = subjects;
all_results.surface_resolutions = surface_resolutions;
all_results.time_points = -0.5:0.1:1.5;
all_results.n_timepoints = length(all_results.time_points);

% Initialize storage
n_subjects = length(subjects);
n_resolutions = length(surface_resolutions);
mean_errors = nan(n_subjects, n_resolutions, all_results.n_timepoints);
std_errors = nan(n_subjects, n_resolutions, all_results.n_timepoints);
median_errors = nan(n_subjects, n_resolutions, all_results.n_timepoints);
n_trials = nan(n_subjects, n_resolutions, all_results.n_timepoints);
success = false(n_subjects, n_resolutions, all_results.n_timepoints);

fprintf('\n=== Loading SVM Results ===\n');

for subj_idx = 1:n_subjects
    subjID = subjects(subj_idx);
    fprintf('Loading Subject %d...\n', subjID);
    
    for res_idx = 1:n_resolutions
        surface_resolution = surface_resolutions(res_idx);
        
        % Load summary file
        summary_file = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'svm_decoding', ...
            sprintf('sub-%02d_svm_timeseries_summary_%d.mat', subjID, surface_resolution));
        
        if exist(summary_file, 'file')
            fprintf('  Loading %d resolution... ', surface_resolution);
            load(summary_file);
            
            % Store results
            mean_errors(subj_idx, res_idx, :) = results_summary.mean_errors;
            std_errors(subj_idx, res_idx, :) = results_summary.std_errors;
            median_errors(subj_idx, res_idx, :) = results_summary.median_errors;
            n_trials(subj_idx, res_idx, :) = results_summary.n_trials;
            success(subj_idx, res_idx, :) = results_summary.success;
            
            fprintf('Success\n');
        else
            fprintf('  %d resolution: File not found\n', surface_resolution);
        end
    end
end

%% Calculate group statistics
fprintf('\n=== Calculating Group Statistics ===\n');

% Calculate group means and SEMs using circular statistics
group_mean_errors = nan(n_resolutions, all_results.n_timepoints);
group_std_errors = nan(n_resolutions, all_results.n_timepoints);
group_sem_errors = nan(n_resolutions, all_results.n_timepoints);

for res_idx = 1:n_resolutions
    for t_idx = 1:all_results.n_timepoints
        % Get errors for this time point and resolution across all subjects
        subject_errors = squeeze(mean_errors(:, res_idx, t_idx));
        valid_subjects = ~isnan(subject_errors);
        
        if sum(valid_subjects) > 0
            % Convert to radians for circular statistics
            errors_rad = circ_ang2rad(subject_errors(valid_subjects));
            
            % Calculate circular mean and std
            circ_mean_rad = circ_mean(errors_rad);
            circ_std_rad = circ_std(errors_rad);
            
            % Convert back to degrees (ensure scalar output)
            group_mean_errors(res_idx, t_idx) = circ_rad2ang(circ_mean_rad(1));
            group_std_errors(res_idx, t_idx) = circ_rad2ang(circ_std_rad(1));
            group_sem_errors(res_idx, t_idx) = group_std_errors(res_idx, t_idx) / sqrt(sum(valid_subjects));
        end
    end
end

% Calculate success rates
success_rate = squeeze(nanmean(success, 1)); % Average success rate across subjects

fprintf('Group statistics calculated for %d subjects using circular statistics\n', n_subjects);

%% Create individual subject plots
fprintf('\n=== Creating Individual Subject Plots ===\n');

for res_idx = 1:n_resolutions
    surface_resolution = surface_resolutions(res_idx);
    
    % Create figure for this resolution
    figure('Position', [100, 100, 1200, 800]);
    
    % Plot individual subject time series
    subplot(2, 1, 1);
    hold on;
    
    colors = lines(n_subjects);
    for subj_idx = 1:n_subjects
        subjID = subjects(subj_idx);
        valid_trials = success(subj_idx, res_idx, :);
        valid_trials = squeeze(valid_trials);
        
        if any(valid_trials)
            plot(all_results.time_points, squeeze(mean_errors(subj_idx, res_idx, :)), ...
                'Color', colors(subj_idx, :), 'LineWidth', 1.5, ...
                'DisplayName', sprintf('Subject %d', subjID));
        end
    end
    
    xlabel('Time (s)');
    ylabel('Mean Decoding Error (degrees)');
    title(sprintf('Individual Subject SVM Decoding Errors - %d Sources', surface_resolution));
    legend('Location', 'best', 'NumColumns', 4);
    grid on;
    xlim([-0.6, 1.6]);
    
    % Plot group average with SEM
    subplot(2, 1, 2);
    hold on;
    
    % Plot mean ± SEM
    shadedErrorBar(all_results.time_points, group_mean_errors(res_idx, :), ...
        group_sem_errors(res_idx, :), 'lineprops', {'r-', 'LineWidth', 2});
    
    xlabel('Time (s)');
    ylabel('Mean Decoding Error (degrees)');
    title(sprintf('Group Average SVM Decoding Errors - %d Sources (n=%d subjects)', ...
        surface_resolution, n_subjects));
    grid on;
    xlim([-0.6, 1.6]);
    
    % Add vertical line at stimulus onset
    xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.7);
    
    % Save figure
    fig_file = fullfile(output_path, sprintf('Fs05_svm_timeseries_individual_%d.fig', surface_resolution));
    png_file = fullfile(output_path, sprintf('Fs05_svm_timeseries_individual_%d.png', surface_resolution));
    
    savefig(fig_file);
    print(png_file, '-dpng', '-r300');
    
    fprintf('Saved individual plots for %d resolution\n', surface_resolution);
end

%% Create group comparison plot
fprintf('\n=== Creating Group Comparison Plot ===\n');

figure('Position', [100, 100, 1200, 600]);

% Plot both resolutions on same axes
hold on;

colors = [0.2, 0.4, 0.8; 0.8, 0.2, 0.2]; % Blue and red
for res_idx = 1:n_resolutions
    surface_resolution = surface_resolutions(res_idx);
    
    % Plot mean ± SEM
    shadedErrorBar(all_results.time_points, group_mean_errors(res_idx, :), ...
        group_sem_errors(res_idx, :), 'lineprops', ...
        {'Color', colors(res_idx, :), 'LineWidth', 2});
end

xlabel('Time (s)');
ylabel('Mean Decoding Error (degrees)');
title(sprintf('Group Average SVM Decoding Errors (n=%d subjects)', n_subjects));
legend(sprintf('%d Sources', surface_resolutions(1)), ...
       sprintf('%d Sources', surface_resolutions(2)), ...
       'Location', 'best');
grid on;
xlim([-0.6, 1.6]);

% Add vertical line at stimulus onset
xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.7);

% Save figure
fig_file = fullfile(output_path, 'Fs05_svm_timeseries_group_comparison.fig');
png_file = fullfile(output_path, 'Fs05_svm_timeseries_group_comparison.png');

savefig(fig_file);
print(png_file, '-dpng', '-r300');

fprintf('Saved group comparison plot\n');

%% Create summary statistics table
fprintf('\n=== Summary Statistics ===\n');

% Calculate overall statistics using circular statistics
overall_mean = nan(n_resolutions, 1);
overall_std = nan(n_resolutions, 1);
overall_sem = nan(n_resolutions, 1);

for res_idx = 1:n_resolutions
    % Get all time points for this resolution
    time_errors = group_mean_errors(res_idx, :);
    valid_times = ~isnan(time_errors);
    
        if sum(valid_times) > 0
            % Convert to radians for circular statistics
            errors_rad = circ_ang2rad(time_errors(valid_times));
            
            % Calculate circular mean and std
            circ_mean_rad = circ_mean(errors_rad);
            circ_std_rad = circ_std(errors_rad);
            
            % Convert back to degrees (ensure scalar output)
            overall_mean(res_idx) = circ_rad2ang(circ_mean_rad(1));
            overall_std(res_idx) = circ_rad2ang(circ_std_rad(1));
            overall_sem(res_idx) = overall_std(res_idx) / sqrt(sum(valid_times));
        else
            overall_mean(res_idx) = NaN;
            overall_std(res_idx) = NaN;
            overall_sem(res_idx) = NaN;
        end
end

% Find best and worst time points
[best_error, best_idx] = min(group_mean_errors, [], 2);
[worst_error, worst_idx] = max(group_mean_errors, [], 2);

fprintf('\nSurface Resolution | Overall Mean Error | SEM | Best Time | Best Error | Worst Time | Worst Error\n');
fprintf('-------------------|-------------------|-----|-----------|------------|------------|------------\n');

for res_idx = 1:n_resolutions
    surface_resolution = surface_resolutions(res_idx);
    best_time = all_results.time_points(best_idx(res_idx));
    worst_time = all_results.time_points(worst_idx(res_idx));
    
    fprintf('%17d | %17.2f | %3.2f | %9.1f | %10.2f | %10.1f | %10.2f\n', ...
        surface_resolution, overall_mean(res_idx), overall_sem(res_idx), ...
        best_time, best_error(res_idx), worst_time, worst_error(res_idx));
end

%% Save summary data
fprintf('\n=== Saving Summary Data ===\n');

summary_data = struct();
summary_data.subjects = subjects;
summary_data.surface_resolutions = surface_resolutions;
summary_data.time_points = all_results.time_points;
summary_data.mean_errors = mean_errors;
summary_data.std_errors = std_errors;
summary_data.median_errors = median_errors;
summary_data.n_trials = n_trials;
summary_data.success = success;
summary_data.group_mean_errors = group_mean_errors;
summary_data.group_sem_errors = group_sem_errors;
summary_data.success_rate = success_rate;

summary_file = fullfile(output_path, 'Fs05_svm_timeseries_summary.mat');
save(summary_file, 'summary_data');

fprintf('Summary data saved to: %s\n', summary_file);
fprintf('Figures saved to: %s\n', output_path);

fprintf('\n=== Visualization Complete ===\n');

end

%% Helper function for shaded error bars
function shadedErrorBar(x, y, errBar, varargin)
% SHADEDERRORBAR - Creates a 2D line plot with shaded error bars
%
% Usage:
%   shadedErrorBar(x, y, errBar, 'lineprops', {'r-', 'LineWidth', 2})
%
% Inputs:
%   x - x-axis values
%   y - y-axis values (mean)
%   errBar - error bar values (SEM, std, etc.)
%   varargin - line properties

% Parse inputs
p = inputParser;
addParameter(p, 'lineprops', {'r-', 'LineWidth', 2});
parse(p, varargin{:});

% Create shaded area
fill([x, fliplr(x)], [y + errBar, fliplr(y - errBar)], ...
    p.Results.lineprops{1}(1), 'FaceAlpha', 0.2, 'EdgeColor', 'none');

% Create line
plot(x, y, p.Results.lineprops{:});

end
