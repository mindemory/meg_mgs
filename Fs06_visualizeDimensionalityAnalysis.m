%% Visualization script for dimensionality analysis results
% This script loads and visualizes dimensionality analysis results across subjects
% Shows individual subject plots and group-averaged plots

clear; close all; clc;

fprintf('=== MEG Beta Power Dimensionality Analysis Visualization ===\n');

%% Set up paths
if ispc
    % HPC paths
    base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
else
    % Local paths
    base_path = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
end

% Add circstat toolbox
circstat_path = '/d/DATD/hyper/toolboxes/CircStat2012a';
if exist(circstat_path, 'dir')
    addpath(circstat_path);
    fprintf('Added CircStat toolbox to path\n');
else
    fprintf('Warning: CircStat toolbox not found at %s\n', circstat_path);
end

%% Define subjects and parameters
subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32];
surface_resolutions = [5124, 8196];
n_subjects = length(subjects);
n_resolutions = length(surface_resolutions);

% Time points
time_points = -0.5:0.1:1.5;
n_time_points = length(time_points);

fprintf('Subjects: %s\n', mat2str(subjects));
fprintf('Surface resolutions: %s\n', mat2str(surface_resolutions));
fprintf('Time points: %s\n', mat2str(time_points));

%% Load data for all subjects and resolutions
fprintf('\n=== Loading Dimensionality Data ===\n');

% Storage for all data
all_dimensionality = zeros(n_subjects, n_resolutions, n_time_points);
all_condition_numbers = zeros(n_subjects, n_resolutions, n_time_points);
all_n_components_95 = zeros(n_subjects, n_resolutions, n_time_points);
all_n_components_eigenvalue = zeros(n_subjects, n_resolutions, n_time_points);

successful_loads = 0;
failed_loads = 0;

for subj_idx = 1:n_subjects
    subjID = subjects(subj_idx);
    
    for res_idx = 1:n_resolutions
        surface_resolution = surface_resolutions(res_idx);
        
        % Construct file path
        result_file = fullfile(base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
            'dimensionality_analysis', sprintf('sub-%02d_dimensionality_%d.mat', subjID, surface_resolution));
        
        fprintf('Loading Subject %d, Resolution %d...', subjID, surface_resolution);
        
        if exist(result_file, 'file')
            try
                load(result_file, 'results');
                
                % Store data
                all_dimensionality(subj_idx, res_idx, :) = results.effective_dimensionality;
                all_condition_numbers(subj_idx, res_idx, :) = results.condition_number;
                all_n_components_95(subj_idx, res_idx, :) = results.n_components_95;
                all_n_components_eigenvalue(subj_idx, res_idx, :) = results.n_components_eigenvalue;
                
                fprintf(' ✓\n');
                successful_loads = successful_loads + 1;
                
            catch ME
                fprintf(' ✗ Error: %s\n', ME.message);
                failed_loads = failed_loads + 1;
            end
        else
            fprintf(' ✗ File not found\n');
            failed_loads = failed_loads + 1;
        end
    end
end

fprintf('\nData loading summary:\n');
fprintf('  Successful loads: %d\n', successful_loads);
fprintf('  Failed loads: %d\n', failed_loads);
fprintf('  Total expected: %d\n', n_subjects * n_resolutions);

%% Calculate group statistics
fprintf('\n=== Calculating Group Statistics ===\n');

% Calculate mean and SEM across subjects for each resolution and time point
group_mean_dimensionality = squeeze(mean(all_dimensionality, 1));  % [resolutions x time_points]
group_sem_dimensionality = squeeze(std(all_dimensionality, 0, 1)) / sqrt(n_subjects);

group_mean_condition = squeeze(mean(all_condition_numbers, 1));
group_sem_condition = squeeze(std(all_condition_numbers, 0, 1)) / sqrt(n_subjects);

group_mean_components_95 = squeeze(mean(all_n_components_95, 1));
group_sem_components_95 = squeeze(std(all_n_components_95, 0, 1)) / sqrt(n_subjects);

%% Create individual subject plots
fprintf('\n=== Creating Individual Subject Plots ===\n');

% Figure 1: Individual subjects
fig1 = figure('Position', [100, 100, 1600, 1200]);
sgtitle('Dimensionality Analysis: Individual Subjects', 'FontSize', 16, 'FontWeight', 'bold');

% Calculate subplot layout
n_cols = 5;
n_rows = ceil(n_subjects / n_cols);

for subj_idx = 1:n_subjects
    subplot(n_rows, n_cols, subj_idx);
    
    % Plot both resolutions for this subject
    plot(time_points, squeeze(all_dimensionality(subj_idx, 1, :)), 'b-o', 'LineWidth', 2, 'MarkerSize', 4);
    hold on;
    plot(time_points, squeeze(all_dimensionality(subj_idx, 2, :)), 'r-s', 'LineWidth', 2, 'MarkerSize', 4);
    
    % Add stimulus onset and memory period markers
    xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.7);
    xline(0.8, 'g--', 'LineWidth', 1, 'Alpha', 0.7);
    
    % Formatting
    xlabel('Time (s)');
    ylabel('Effective Dimensionality');
    title(sprintf('Subject %d', subjects(subj_idx)));
    grid on;
    xlim([-0.6, 1.6]);
    
    if subj_idx == 1
        legend('5124 vertices', '8196 vertices', 'Stimulus Onset', 'Memory Period', 'Location', 'best');
    end
end

% Save individual subject figure
fig1_file = fullfile(base_path, 'dimensionality_analysis', 'individual_subjects_dimensionality.fig');
png1_file = fullfile(base_path, 'dimensionality_analysis', 'individual_subjects_dimensionality.png');

% Create directory if it doesn't exist
if ~exist(fullfile(base_path, 'dimensionality_analysis'), 'dir')
    mkdir(fullfile(base_path, 'dimensionality_analysis'));
end

savefig(fig1, fig1_file);
print(fig1, png1_file, '-dpng', '-r300');
fprintf('Individual subject figure saved to: %s\n', fig1_file);

%% Create group-averaged plots
fprintf('\n=== Creating Group-Averaged Plots ===\n');

fig2 = figure('Position', [200, 200, 1400, 1000]);
sgtitle('Dimensionality Analysis: Group-Averaged Results', 'FontSize', 16, 'FontWeight', 'bold');

% Subplot 1: Effective dimensionality over time
subplot(2, 2, 1);
% Plot 5124 vertices with shaded error
fill([time_points, fliplr(time_points)], ...
     [group_mean_dimensionality(1, :) + group_sem_dimensionality(1, :), ...
      fliplr(group_mean_dimensionality(1, :) - group_sem_dimensionality(1, :))], ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
hold on;
plot(time_points, group_mean_dimensionality(1, :), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);

% Plot 8196 vertices with shaded error
fill([time_points, fliplr(time_points)], ...
     [group_mean_dimensionality(2, :) + group_sem_dimensionality(2, :), ...
      fliplr(group_mean_dimensionality(2, :) - group_sem_dimensionality(2, :))], ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
plot(time_points, group_mean_dimensionality(2, :), 'r-s', 'LineWidth', 2, 'MarkerSize', 6);

xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.7);

xlabel('Time (s)');
ylabel('Effective Dimensionality (Participation Ratio)');
title('Effective Dimensionality Over Time');
legend('5124 vertices', '8196 vertices', 'Stimulus Onset', 'Location', 'best');
grid on;
xlim([-0.6, 1.6]);

% Subplot 2: Condition number over time (log scale)
subplot(2, 2, 2);
semilogy(time_points, group_mean_condition(1, :), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
semilogy(time_points, group_mean_condition(2, :), 'r-s', 'LineWidth', 2, 'MarkerSize', 6);

xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.7);

xlabel('Time (s)');
ylabel('Condition Number (log scale)');
title('Condition Number Over Time');
legend('5124 vertices', '8196 vertices', 'Stimulus Onset', 'Location', 'best');
grid on;
xlim([-0.6, 1.6]);

% Subplot 3: Number of components for 95% variance
subplot(2, 2, 3);
% Plot 5124 vertices with shaded error
fill([time_points, fliplr(time_points)], ...
     [group_mean_components_95(1, :) + group_sem_components_95(1, :), ...
      fliplr(group_mean_components_95(1, :) - group_sem_components_95(1, :))], ...
     'b', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
hold on;
plot(time_points, group_mean_components_95(1, :), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);

% Plot 8196 vertices with shaded error
fill([time_points, fliplr(time_points)], ...
     [group_mean_components_95(2, :) + group_sem_components_95(2, :), ...
      fliplr(group_mean_components_95(2, :) - group_sem_components_95(2, :))], ...
     'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
plot(time_points, group_mean_components_95(2, :), 'r-s', 'LineWidth', 2, 'MarkerSize', 6);

xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.7);

xlabel('Time (s)');
ylabel('Number of Components (95% Variance)');
title('Components for 95% Explained Variance');
legend('5124 vertices', '8196 vertices', 'Stimulus Onset', 'Location', 'best');
grid on;
xlim([-0.6, 1.6]);

% Subplot 4: Summary statistics
subplot(2, 2, 4);
text(0.1, 0.9, sprintf('Group Summary Statistics'), 'FontSize', 14, 'FontWeight', 'bold');
text(0.1, 0.8, sprintf('Subjects: %d', n_subjects), 'FontSize', 12);
text(0.1, 0.7, sprintf('Time points: %d', n_time_points), 'FontSize', 12);
text(0.1, 0.6, sprintf('Surface resolutions: %s', mat2str(surface_resolutions)), 'FontSize', 12);

% Calculate overall statistics
overall_mean_5124 = mean(group_mean_dimensionality(1, :));
overall_mean_8196 = mean(group_mean_dimensionality(2, :));
overall_std_5124 = std(group_mean_dimensionality(1, :));
overall_std_8196 = std(group_mean_dimensionality(2, :));

text(0.1, 0.5, sprintf('5124 vertices:'), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'blue');
text(0.1, 0.45, sprintf('  Mean: %.2f ± %.2f', overall_mean_5124, overall_std_5124), 'FontSize', 12, 'Color', 'blue');
text(0.1, 0.4, sprintf('  Range: %.2f - %.2f', min(group_mean_dimensionality(1, :)), max(group_mean_dimensionality(1, :))), 'FontSize', 12, 'Color', 'blue');

text(0.1, 0.3, sprintf('8196 vertices:'), 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'red');
text(0.1, 0.25, sprintf('  Mean: %.2f ± %.2f', overall_mean_8196, overall_std_8196), 'FontSize', 12, 'Color', 'red');
text(0.1, 0.2, sprintf('  Range: %.2f - %.2f', min(group_mean_dimensionality(2, :)), max(group_mean_dimensionality(2, :))), 'FontSize', 12, 'Color', 'red');

axis off;

% Save group-averaged figure
fig2_file = fullfile(base_path, 'dimensionality_analysis', 'group_averaged_dimensionality.fig');
png2_file = fullfile(base_path, 'dimensionality_analysis', 'group_averaged_dimensionality.png');

savefig(fig2, fig2_file);
print(fig2, png2_file, '-dpng', '-r300');
fprintf('Group-averaged figure saved to: %s\n', fig2_file);

%% Create comparison plot (5124 vs 8196)
fprintf('\n=== Creating Resolution Comparison Plot ===\n');

fig3 = figure('Position', [300, 300, 1200, 800]);

% Plot both resolutions with shaded error regions
% Plot 5124 vertices with shaded error
fill([time_points, fliplr(time_points)], ...
     [group_mean_dimensionality(1, :) + group_sem_dimensionality(1, :), ...
      fliplr(group_mean_dimensionality(1, :) - group_sem_dimensionality(1, :))], ...
     'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
hold on;
plot(time_points, group_mean_dimensionality(1, :), 'b-o', 'LineWidth', 3, 'MarkerSize', 8);

% Plot 8196 vertices with shaded error
fill([time_points, fliplr(time_points)], ...
     [group_mean_dimensionality(2, :) + group_sem_dimensionality(2, :), ...
      fliplr(group_mean_dimensionality(2, :) - group_sem_dimensionality(2, :))], ...
     'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(time_points, group_mean_dimensionality(2, :), 'r-s', 'LineWidth', 3, 'MarkerSize', 8);

% Add stimulus markers
xline(0, 'k--', 'LineWidth', 2, 'Alpha', 0.8);

% Formatting
xlabel('Time (s)', 'FontSize', 14);
ylabel('Effective Dimensionality (Participation Ratio)', 'FontSize', 14);
title('Dimensionality Comparison: 5124 vs 8196 Vertices', 'FontSize', 16, 'FontWeight', 'bold');
legend('5124 vertices', '8196 vertices', 'Stimulus Onset', 'Location', 'best', 'FontSize', 12);
grid on;
xlim([-0.6, 1.6]);
set(gca, 'FontSize', 12);

% Add text annotations
text(-0.4, max(group_mean_dimensionality(:)) * 0.9, sprintf('N = %d subjects', n_subjects), 'FontSize', 12, 'BackgroundColor', 'white');
text(-0.4, max(group_mean_dimensionality(:)) * 0.85, sprintf('Mean 5124: %.2f ± %.2f', overall_mean_5124, overall_std_5124), 'FontSize', 12, 'Color', 'blue', 'BackgroundColor', 'white');
text(-0.4, max(group_mean_dimensionality(:)) * 0.8, sprintf('Mean 8196: %.2f ± %.2f', overall_mean_8196, overall_std_8196), 'FontSize', 12, 'Color', 'red', 'BackgroundColor', 'white');

% Save comparison figure
fig3_file = fullfile(base_path, 'dimensionality_analysis', 'resolution_comparison_dimensionality.fig');
png3_file = fullfile(base_path, 'dimensionality_analysis', 'resolution_comparison_dimensionality.png');

savefig(fig3, fig3_file);
print(fig3, png3_file, '-dpng', '-r300');
fprintf('Resolution comparison figure saved to: %s\n', fig3_file);

%% Print summary statistics
fprintf('\n=== Summary Statistics ===\n');
fprintf('Group-averaged effective dimensionality:\n');
fprintf('  5124 vertices: %.2f ± %.2f (range: %.2f - %.2f)\n', ...
    overall_mean_5124, overall_std_5124, min(group_mean_dimensionality(1, :)), max(group_mean_dimensionality(1, :)));
fprintf('  8196 vertices: %.2f ± %.2f (range: %.2f - %.2f)\n', ...
    overall_mean_8196, overall_std_8196, min(group_mean_dimensionality(2, :)), max(group_mean_dimensionality(2, :)));

% Find peak dimensionality times
[~, peak_idx_5124] = max(group_mean_dimensionality(1, :));
[~, peak_idx_8196] = max(group_mean_dimensionality(2, :));

fprintf('\nPeak dimensionality times:\n');
fprintf('  5124 vertices: %.1fs (dimensionality = %.2f)\n', time_points(peak_idx_5124), group_mean_dimensionality(1, peak_idx_5124));
fprintf('  8196 vertices: %.1fs (dimensionality = %.2f)\n', time_points(peak_idx_8196), group_mean_dimensionality(2, peak_idx_8196));

fprintf('\nDimensionality analysis visualization completed!\n');
