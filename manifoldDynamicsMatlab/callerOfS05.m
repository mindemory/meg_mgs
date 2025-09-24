function callerOfS05(subjID, surface_resolution, algorithm)
%% callerOfS05 - Call S05_TAFKAPwPCA_Decoding for multiple time points
%
% This script calls S05_TAFKAPwPCA_Decoding for each time point from -0.5 to 1.7s
% with a time step of 0.1s, creating a time series analysis.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   surface_resolution - Surface resolution (default: 5124)
%   algorithm - 'TAFKAP' or 'PRINCE' (default: 'TAFKAP')
%
% Outputs:
%   - Saves decoding results for each time point
%   - Creates summary visualization across all time points
%
% Example:
%   callerOfS05(1, 5124, 'TAFKAP');
%
% Author: Mrugank Dake
% Date: 2025-01-20

%% Set Default Parameters
if nargin < 1, subjID = 1; end
if nargin < 2, surface_resolution = 5124; end
if nargin < 3, algorithm = 'TAFKAP'; end

%% Define Time Points
time_points = -0.5:0.1:1.7;
n_time_points = length(time_points);

fprintf('Starting TAFKAP PCA time series analysis for Subject %d, surface resolution %d\n', subjID, surface_resolution);
fprintf('Time points: [%s]\n', sprintf('%.1f ', time_points));
fprintf('Number of time points: %d\n', n_time_points);

%% Initialize Results Storage
all_results = struct();
all_results.time_points = time_points;
all_results.mean_errors = zeros(n_time_points, 1);
all_results.std_errors = zeros(n_time_points, 1);
all_results.successful = false(n_time_points, 1);
all_results.estimates = cell(n_time_points, 1);
all_results.targets = cell(n_time_points, 1);

%% Run Analysis for Each Time Point
for t = 1:n_time_points
    current_time = time_points(t);
    fprintf('\n--- Time Point %d/%d: %.1fs ---\n', t, n_time_points, current_time);
    
    try
        % Call the main TAFKAP PCA decoding script
        S05_TAFKAPwPCA_Decoding(subjID, surface_resolution, algorithm, current_time);
        
        % Load the results to extract performance metrics
        if strcmp(algorithm, 'PRINCE')
            results_file = sprintf('sub-%02d_prince_pca_performance_%d_t%.1f.mat', subjID, surface_resolution, current_time);
        else
            results_file = sprintf('sub-%02d_tafkap_pca_performance_%d_t%.1f.mat', subjID, surface_resolution, current_time);
        end
        
        % Try to find the results file in the expected location
        results_path = fullfile('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives', ...
                               sprintf('sub-%02d', subjID), 'sourceRecon', ...
                               sprintf('%s_pca_decoding', lower(algorithm)), results_file);
        
        if exist(results_path, 'file')
            load(results_path, 'performance');
            all_results.mean_errors(t) = performance.mean_error;
            all_results.std_errors(t) = performance.std_error;
            all_results.successful(t) = true;
            fprintf('  Success: Mean error = %.2f° ± %.2f°\n', performance.mean_error, performance.std_error);
        else
            fprintf('  Warning: Results file not found at %s\n', results_path);
            all_results.successful(t) = false;
        end
        
    catch ME
        fprintf('  Error at time %.1fs: %s\n', current_time, ME.message);
        all_results.successful(t) = false;
    end
end

%% Create Summary Visualization
fprintf('\n=== Creating Summary Visualization ===\n');

% Create figure
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Mean error over time
subplot(2, 2, 1);
successful_idx = all_results.successful;
plot(time_points(successful_idx), all_results.mean_errors(successful_idx), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(time_points(successful_idx), all_results.mean_errors(successful_idx) + all_results.std_errors(successful_idx), 'b--', 'Alpha', 0.5);
plot(time_points(successful_idx), all_results.mean_errors(successful_idx) - all_results.std_errors(successful_idx), 'b--', 'Alpha', 0.5);
xlabel('Time (s)');
ylabel('Mean Error (degrees)');
title(sprintf('TAFKAP PCA Decoding Performance - Subject %d', subjID));
grid on;
xline(0, 'r--', 'Stimulus Onset', 'LineWidth', 1.5);
xlim([min(time_points), max(time_points)]);

% Plot 2: Standard deviation over time
subplot(2, 2, 2);
plot(time_points(successful_idx), all_results.std_errors(successful_idx), 'r-o', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Time (s)');
ylabel('Error Standard Deviation (degrees)');
title('Error Variability Over Time');
grid on;
xline(0, 'r--', 'Stimulus Onset', 'LineWidth', 1.5);
xlim([min(time_points), max(time_points)]);

% Plot 3: Success rate
subplot(2, 2, 3);
bar(time_points, double(all_results.successful), 'FaceColor', [0.2, 0.6, 0.2]);
xlabel('Time (s)');
ylabel('Successful Analysis');
title('Analysis Success Rate');
grid on;
xline(0, 'r--', 'Stimulus Onset', 'LineWidth', 1.5);
xlim([min(time_points), max(time_points)]);
ylim([0, 1.2]);

% Plot 4: Summary statistics
subplot(2, 2, 4);
if any(successful_idx)
    best_time_idx = find(successful_idx & all_results.mean_errors == min(all_results.mean_errors(successful_idx)), 1);
    worst_time_idx = find(successful_idx & all_results.mean_errors == max(all_results.mean_errors(successful_idx)), 1);
    
    text(0.1, 0.8, sprintf('Total Time Points: %d', n_time_points), 'FontSize', 12);
    text(0.1, 0.7, sprintf('Successful: %d (%.1f%%)', sum(successful_idx), 100*sum(successful_idx)/n_time_points), 'FontSize', 12);
    text(0.1, 0.6, sprintf('Best Performance: %.2f° at %.1fs', all_results.mean_errors(best_time_idx), time_points(best_time_idx)), 'FontSize', 12);
    text(0.1, 0.5, sprintf('Worst Performance: %.2f° at %.1fs', all_results.mean_errors(worst_time_idx), time_points(worst_time_idx)), 'FontSize', 12);
    text(0.1, 0.4, sprintf('Overall Mean: %.2f° ± %.2f°', mean(all_results.mean_errors(successful_idx)), std(all_results.mean_errors(successful_idx))), 'FontSize', 12);
else
    text(0.1, 0.5, 'No successful analyses', 'FontSize', 12, 'Color', 'red');
end
axis off;
title('Summary Statistics');

sgtitle(sprintf('TAFKAP PCA Time Series Analysis - Subject %02d (Surface %d)', subjID, surface_resolution));

%% Save Results
fprintf('Saving summary results...\n');

% Create output directory
output_dir = fullfile('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives', ...
                     sprintf('sub-%02d', subjID), 'sourceRecon', ...
                     sprintf('%s_pca_decoding', lower(algorithm)));

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save summary results
summary_file = fullfile(output_dir, sprintf('sub-%02d_%s_pca_timeseries_summary_%d.mat', subjID, lower(algorithm), surface_resolution));
save(summary_file, 'all_results');
fprintf('Summary saved to: %s\n', summary_file);

% Save figure
fig_file = fullfile(output_dir, sprintf('sub-%02d_%s_pca_timeseries_%d.fig', subjID, lower(algorithm), surface_resolution));
saveas(gcf, fig_file);
fprintf('Figure saved to: %s\n', fig_file);

% Save PNG
png_file = fullfile(output_dir, sprintf('sub-%02d_%s_pca_timeseries_%d.png', subjID, lower(algorithm), surface_resolution));
saveas(gcf, png_file);
fprintf('PNG saved to: %s\n', png_file);

%% Print Final Summary
fprintf('\n=== TAFKAP PCA Time Series Summary ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d\n', surface_resolution);
fprintf('Time points analyzed: %d\n', n_time_points);
fprintf('Successful analyses: %d/%d (%.1f%%)\n', sum(successful_idx), n_time_points, 100*sum(successful_idx)/n_time_points);

if any(successful_idx)
    fprintf('Best performance: %.2f° at %.1fs\n', all_results.mean_errors(best_time_idx), time_points(best_time_idx));
    fprintf('Worst performance: %.2f° at %.1fs\n', all_results.mean_errors(worst_time_idx), time_points(worst_time_idx));
    fprintf('Overall mean error: %.2f° ± %.2f°\n', mean(all_results.mean_errors(successful_idx)), std(all_results.mean_errors(successful_idx)));
end

fprintf('TAFKAP PCA time series analysis completed!\n');

end
