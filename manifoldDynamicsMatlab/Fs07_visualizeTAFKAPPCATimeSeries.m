function Fs07_visualizeTAFKAPPCATimeSeries()
%% Fs07_visualizeTAFKAPPCATimeSeries - Visualize TAFKAP PCA time series results
%
% This script loads TAFKAP PCA time series results for all subjects and creates
% visualizations showing individual subject results and group-averaged results.
%
% Inputs: None (loads data from derivatives directory)
%
% Outputs:
%   - Individual subject plots
%   - Group-averaged plots
%   - Summary statistics
%
% Author: Mrugank Dake
% Date: 2025-01-20

%% Set up paths and parameters
fprintf('Starting TAFKAP PCA time series visualization...\n');

% Define subjects and parameters
subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32];
surface_resolution = 8196;
algorithm = 'TAFKAP';

% Set up paths
base_dir = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
output_dir = fullfile(base_dir, 'group_analysis', 'tafkap_pca_timeseries');

% Create output directory
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load data for all subjects
fprintf('Loading TAFKAP PCA time series data for all subjects...\n');

all_subject_data = struct();
successful_subjects = [];
n_subjects = length(subjects);

for i = 1:n_subjects
    subjID = subjects(i);
    fprintf('Loading Subject %d...\n', subjID);
    
    % Construct file path
    subject_dir = fullfile(base_dir, sprintf('sub-%02d', subjID), 'sourceRecon', 'tafkap_pca_decoding');
    summary_file = fullfile(subject_dir, sprintf('sub-%02d_tafkap_pca_timeseries_summary_%d.mat', subjID, surface_resolution));
    
    if exist(summary_file, 'file')
        try
            load(summary_file, 'results');
            all_subject_data(i).subject = subjID;
            all_subject_data(i).time_points = results.time_points;
            all_subject_data(i).mean_errors = results.mean_error;
            all_subject_data(i).std_errors = results.std_error;
            all_subject_data(i).successful = results.success_flags;
            all_subject_data(i).loaded = true;
            successful_subjects = [successful_subjects, subjID];
            fprintf('  Successfully loaded Subject %d\n', subjID);
        catch ME
            fprintf('  Error loading Subject %d: %s\n', subjID, ME.message);
            all_subject_data(i).loaded = false;
        end
    else
        fprintf('  Summary file not found for Subject %d: %s\n', subjID, summary_file);
        all_subject_data(i).loaded = false;
    end
end

fprintf('Successfully loaded %d/%d subjects\n', length(successful_subjects), n_subjects);

if isempty(successful_subjects)
    error('No subject data found!');
end

%% Create individual subject plots
fprintf('Creating individual subject plots...\n');

% Figure 1: Individual subject time series
figure('Position', [100, 100, 1400, 1000]);

% Calculate subplot layout
n_successful = length(successful_subjects);
n_cols = 4;
n_rows = ceil(n_successful / n_cols);

for i = 1:n_successful
    subjID = successful_subjects(i);
    data_idx = find([all_subject_data.subject] == subjID);
    
    subplot(n_rows, n_cols, i);
    
    if all_subject_data(data_idx).loaded
        time_points = all_subject_data(data_idx).time_points;
        mean_errors = all_subject_data(data_idx).mean_errors;
        std_errors = all_subject_data(data_idx).std_errors;
        successful = all_subject_data(data_idx).successful;
        
        % Plot successful time points
        successful_idx = successful;
        plot(time_points(successful_idx), mean_errors(successful_idx), 'b-o', 'LineWidth', 2, 'MarkerSize', 4);
        hold on;
        
        % Add error bars
        plot(time_points(successful_idx), mean_errors(successful_idx) + std_errors(successful_idx), 'b--');
        plot(time_points(successful_idx), mean_errors(successful_idx) - std_errors(successful_idx), 'b--');
        
        % Add stimulus onset line
        xline(0, 'r--', 'Stimulus Onset', 'LineWidth', 1.5);
        
        xlabel('Time (s)');
        ylabel('Mean Error (degrees)');
        title(sprintf('Subject %d', subjID));
        grid on;
        xlim([min(time_points), max(time_points)]);
        
        % Add success rate text
        success_rate = sum(successful) / length(successful) * 100;
        text(0.05, 0.95, sprintf('Success: %.1f%%', success_rate), 'Units', 'normalized', ...
             'VerticalAlignment', 'top', 'FontSize', 8);
    else
        text(0.5, 0.5, 'Data not available', 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', 'middle', 'FontSize', 12);
        title(sprintf('Subject %d', subjID));
    end
end

sgtitle('TAFKAP PCA Time Series - Individual Subjects (8196 vertices)', 'FontSize', 16);

% Save individual subject figure
fig_file = fullfile(output_dir, 'tafkap_pca_individual_subjects.fig');
saveas(gcf, fig_file);
fprintf('Individual subject figure saved to: %s\n', fig_file);

png_file = fullfile(output_dir, 'tafkap_pca_individual_subjects.png');
saveas(gcf, png_file);
fprintf('Individual subject PNG saved to: %s\n', png_file);

%% Create group-averaged plots
fprintf('Creating group-averaged plots...\n');

% Extract common time points (assuming all subjects have the same time points)
first_successful = find([all_subject_data.loaded], 1);
if isempty(first_successful)
    error('No successful subjects found!');
end

common_time_points = all_subject_data(first_successful).time_points;
n_time_points = length(common_time_points);

% Initialize group data
group_mean_errors = zeros(n_time_points, 1);
group_std_errors = zeros(n_time_points, 1);
group_sem_errors = zeros(n_time_points, 1);
group_success_count = zeros(n_time_points, 1);

% Calculate group statistics
for t = 1:n_time_points
    time_errors = [];
    time_stds = [];
    
    for i = 1:length(successful_subjects)
        subjID = successful_subjects(i);
        data_idx = find([all_subject_data.subject] == subjID);
        
        if all_subject_data(data_idx).loaded && all_subject_data(data_idx).successful(t)
            time_errors = [time_errors, all_subject_data(data_idx).mean_errors(t)];
            time_stds = [time_stds, all_subject_data(data_idx).std_errors(t)];
        end
    end
    
    if ~isempty(time_errors)
        group_mean_errors(t) = mean(time_errors);
        group_std_errors(t) = std(time_errors);
        group_sem_errors(t) = std(time_errors) / sqrt(length(time_errors));
        group_success_count(t) = length(time_errors);
    else
        group_mean_errors(t) = NaN;
        group_std_errors(t) = NaN;
        group_sem_errors(t) = NaN;
        group_success_count(t) = 0;
    end
end

% Figure 2: Group-averaged time series
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Mean error over time
subplot(2, 2, 1);
valid_idx = ~isnan(group_mean_errors);
plot(common_time_points(valid_idx), group_mean_errors(valid_idx), 'b-o', 'LineWidth', 3, 'MarkerSize', 6);
hold on;

% Add SEM error bars
plot(common_time_points(valid_idx), group_mean_errors(valid_idx) + group_sem_errors(valid_idx), 'b--', 'LineWidth', 1.5);
plot(common_time_points(valid_idx), group_mean_errors(valid_idx) - group_sem_errors(valid_idx), 'b--', 'LineWidth', 1.5);

% Fill between SEM
time_points_fill = common_time_points(valid_idx);
upper_bound = group_mean_errors(valid_idx) + group_sem_errors(valid_idx);
lower_bound = group_mean_errors(valid_idx) - group_sem_errors(valid_idx);
% Ensure all arrays are row vectors
time_points_fill = time_points_fill(:)';
upper_bound = upper_bound(:)';
lower_bound = lower_bound(:)';
fill([time_points_fill, fliplr(time_points_fill)], ...
     [upper_bound, fliplr(lower_bound)], ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');

xline(0, 'r--', 'Stimulus Onset', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Mean Error (degrees)');
title('Group-Averaged TAFKAP PCA Performance');
grid on;
xlim([min(common_time_points), max(common_time_points)]);

% Plot 2: Standard deviation over time
subplot(2, 2, 2);
plot(common_time_points(valid_idx), group_std_errors(valid_idx), 'r-o', 'LineWidth', 2, 'MarkerSize', 4);
xline(0, 'r--', 'Stimulus Onset', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Error Standard Deviation (degrees)');
title('Group Error Variability');
grid on;
xlim([min(common_time_points), max(common_time_points)]);

% Plot 3: Success rate over time
subplot(2, 2, 3);
success_rate = group_success_count / length(successful_subjects) * 100;
bar(common_time_points, success_rate, 'FaceColor', [0.2, 0.6, 0.2], 'EdgeColor', 'none');
xline(0, 'r--', 'Stimulus Onset', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Success Rate (%)');
title('Analysis Success Rate Over Time');
grid on;
xlim([min(common_time_points), max(common_time_points)]);
ylim([0, 105]);

% Plot 4: Summary statistics
subplot(2, 2, 4);
if any(valid_idx)
    best_time_idx = find(valid_idx & group_mean_errors == min(group_mean_errors(valid_idx)), 1);
    worst_time_idx = find(valid_idx & group_mean_errors == max(group_mean_errors(valid_idx)), 1);
    
    text(0.1, 0.8, sprintf('Subjects: %d', length(successful_subjects)), 'FontSize', 12);
    text(0.1, 0.7, sprintf('Time Points: %d', n_time_points), 'FontSize', 12);
    text(0.1, 0.6, sprintf('Best Performance: %.2f° at %.1fs', group_mean_errors(best_time_idx), common_time_points(best_time_idx)), 'FontSize', 12);
    text(0.1, 0.5, sprintf('Worst Performance: %.2f° at %.1fs', group_mean_errors(worst_time_idx), common_time_points(worst_time_idx)), 'FontSize', 12);
    text(0.1, 0.4, sprintf('Overall Mean: %.2f° ± %.2f°', mean(group_mean_errors(valid_idx)), std(group_mean_errors(valid_idx))), 'FontSize', 12);
    text(0.1, 0.3, sprintf('Mean Success Rate: %.1f%%', mean(success_rate)), 'FontSize', 12);
else
    text(0.1, 0.5, 'No valid data', 'FontSize', 12, 'Color', 'red');
end
axis off;
title('Group Summary Statistics');

sgtitle('TAFKAP PCA Time Series - Group Analysis (8196 vertices)', 'FontSize', 16);

% Save group figure
fig_file = fullfile(output_dir, 'tafkap_pca_group_analysis.fig');
saveas(gcf, fig_file);
fprintf('Group analysis figure saved to: %s\n', fig_file);

png_file = fullfile(output_dir, 'tafkap_pca_group_analysis.png');
saveas(gcf, png_file);
fprintf('Group analysis PNG saved to: %s\n', png_file);

%% Create comparison plot (if we have both resolutions)
fprintf('Creating comparison plots...\n');

% Figure 3: Performance comparison
figure('Position', [100, 100, 1000, 600]);

% Plot mean error over time with confidence intervals
plot(common_time_points(valid_idx), group_mean_errors(valid_idx), 'b-o', 'LineWidth', 3, 'MarkerSize', 6, 'DisplayName', 'TAFKAP PCA (8196 vertices)');
hold on;

% Add confidence intervals
time_points_fill = common_time_points(valid_idx);
upper_bound = group_mean_errors(valid_idx) + group_sem_errors(valid_idx);
lower_bound = group_mean_errors(valid_idx) - group_sem_errors(valid_idx);
% Ensure all arrays are row vectors
time_points_fill = time_points_fill(:)';
upper_bound = upper_bound(:)';
lower_bound = lower_bound(:)';
fill([time_points_fill, fliplr(time_points_fill)], ...
     [upper_bound, fliplr(lower_bound)], ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'DisplayName', 'SEM');

xline(0, 'r--', 'Stimulus Onset', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Mean Error (degrees)');
title('TAFKAP PCA Time Series Performance');
legend('Location', 'best');
grid on;
xlim([min(common_time_points), max(common_time_points)]);

% Save comparison figure
fig_file = fullfile(output_dir, 'tafkap_pca_performance_comparison.fig');
saveas(gcf, fig_file);
fprintf('Performance comparison figure saved to: %s\n', fig_file);

png_file = fullfile(output_dir, 'tafkap_pca_performance_comparison.png');
saveas(gcf, png_file);
fprintf('Performance comparison PNG saved to: %s\n', png_file);

%% Save group results
fprintf('Saving group results...\n');

group_results = struct();
group_results.subjects = successful_subjects;
group_results.time_points = common_time_points;
group_results.group_mean_errors = group_mean_errors;
group_results.group_std_errors = group_std_errors;
group_results.group_sem_errors = group_sem_errors;
group_results.group_success_count = group_success_count;
group_results.success_rate = success_rate;
group_results.algorithm = algorithm;
group_results.surface_resolution = surface_resolution;

results_file = fullfile(output_dir, 'tafkap_pca_group_results.mat');
save(results_file, 'group_results', 'all_subject_data');
fprintf('Group results saved to: %s\n', results_file);

%% Print final summary
fprintf('\n=== TAFKAP PCA Time Series Group Analysis Summary ===\n');
fprintf('Successful subjects: %d/%d\n', length(successful_subjects), n_subjects);
fprintf('Subjects: %s\n', sprintf('%d ', successful_subjects));
fprintf('Time points: %d (%.1fs to %.1fs)\n', n_time_points, min(common_time_points), max(common_time_points));
fprintf('Surface resolution: %d vertices\n', surface_resolution);

if any(valid_idx)
    best_time_idx = find(valid_idx & group_mean_errors == min(group_mean_errors(valid_idx)), 1);
    worst_time_idx = find(valid_idx & group_mean_errors == max(group_mean_errors(valid_idx)), 1);
    
    fprintf('Best performance: %.2f° at %.1fs\n', group_mean_errors(best_time_idx), common_time_points(best_time_idx));
    fprintf('Worst performance: %.2f° at %.1fs\n', group_mean_errors(worst_time_idx), common_time_points(worst_time_idx));
    fprintf('Overall mean error: %.2f° ± %.2f°\n', mean(group_mean_errors(valid_idx)), std(group_mean_errors(valid_idx)));
    fprintf('Mean success rate: %.1f%%\n', mean(success_rate));
end

fprintf('TAFKAP PCA time series visualization completed!\n');

end
