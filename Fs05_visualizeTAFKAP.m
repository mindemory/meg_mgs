function Fs05_visualizeTAFKAP(vol_res, algorithm)
% Fs05_visualizeTAFKAP - Visualize TAFKAP decoding results across subjects
%
% This script loads TAFKAP decoding results from HPC and creates visualizations
% showing average decoding error across subjects.
%
% Inputs:
%   vol_res - Volume resolution in mm (e.g., 8, 4, 2, etc.)
%   algorithm - 'TAFKAP' or 'PRINCE' (default: 'TAFKAP')
%
% Outputs:
%   - Creates figures showing decoding performance across subjects
%   - Saves figures as PNG and .fig files
%
% Dependencies:
%   - S05_TAFKAP_Decoding.m results (must be run first on HPC)
%
% Example:
%   Fs05_visualizeTAFKAP(8, 'TAFKAP')
%   Fs05_visualizeTAFKAP(8, 'PRINCE')
%
% Author: Mrugank Dake
% Date: 2025-01-20

if nargin < 1
    vol_res = 8; % Default volume resolution in mm
end
if nargin < 2
    algorithm = 'TAFKAP'; % Default to TAFKAP
end

clearvars -except vol_res algorithm; % Keep inputs
close all; clc;

%% Path Setup - Auto-detect HPC vs Local
fprintf('=== TAFKAP Decoding Results Visualization ===\n');

% Auto-detect environment and set paths
if exist('/scratch/mdd9787', 'dir')
    % HPC environment
    fprintf('Detected HPC environment\n');
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
elseif exist('/d/DATD', 'dir')
    % Local environment (DATD)
    fprintf('Detected local DATD environment\n');
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
else
    % Fallback
    fprintf('Detected local environment (fallback)\n');
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
end

fprintf('Data base path: %s\n', data_base_path);
fprintf('Volume resolution: %d mm\n', vol_res);
fprintf('Algorithm: %s\n', algorithm);

%% Create Figures Directory Structure
figures_base_dir = fullfile(data_base_path, 'figures', 'Fs05');

% Create directories if they don't exist
if ~exist(figures_base_dir, 'dir')
    mkdir(figures_base_dir);
end

%% Define Subject List
subjects = [1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32];

%% Load Results from All Subjects
fprintf('Loading TAFKAP results from all subjects...\n');

all_subjects_error = [];
all_subjects_uncertainty = [];
valid_subjects = [];

for subjID = subjects
    % Set up output directory path
    if strcmp(algorithm, 'PRINCE')
        results_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'prince_decoding');
        results_file = sprintf('sub-%02d_prince_performance_%dmm_avgdelay.mat', subjID, vol_res);
    else
        results_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'tafkap_decoding');
        results_file = sprintf('sub-%02d_tafkap_performance_%dmm_avgdelay.mat', subjID, vol_res);
    end
    
    results_path = fullfile(results_dir, results_file);
    
    if exist(results_path, 'file')
        fprintf('  Loading subject %02d...\n', subjID);
        loaded_perf = load(results_path);
        performance = loaded_perf.performance;
        
        % Store decoding error (uncertainty field may not exist)
        % Ensure column vector for consistent concatenation
        error_vector = performance.decoding_error(:);
        all_subjects_error = [all_subjects_error; error_vector];
        
        if isfield(performance, 'uncertainty')
            uncertainty_vector = performance.uncertainty(:);
            all_subjects_uncertainty = [all_subjects_uncertainty; uncertainty_vector];
        else
            % If uncertainty field doesn't exist, use NaN
            all_subjects_uncertainty = [all_subjects_uncertainty; NaN(size(error_vector))];
        end
        valid_subjects = [valid_subjects, subjID];
        
        fprintf('    Mean error: %.2f° ± %.2f°\n', performance.mean_error, performance.std_error);
    else
        fprintf('  Subject %02d results not found, skipping...\n', subjID);
    end
end

if isempty(valid_subjects)
    error('No valid subject results found!');
end

fprintf('Loaded results from %d subjects: %s\n', length(valid_subjects), mat2str(valid_subjects));

%% Compute Group Statistics
fprintf('Computing group statistics...\n');

% Overall group statistics
group_mean_error = mean(all_subjects_error);
group_std_error = std(all_subjects_error);
group_median_error = median(all_subjects_error);

group_mean_uncertainty = mean(all_subjects_uncertainty, 'omitnan');
group_std_uncertainty = std(all_subjects_uncertainty, 'omitnan');

fprintf('Group Statistics:\n');
fprintf('  Mean decoding error: %.2f° ± %.2f°\n', group_mean_error, group_std_error);
fprintf('  Median decoding error: %.2f°\n', group_median_error);
fprintf('  Mean uncertainty: %.3f ± %.3f\n', group_mean_uncertainty, group_std_uncertainty);

%% Create Visualizations
fprintf('Creating visualizations...\n');

figure('Position', [100, 100, 1400, 1000]);

% Subplot 1: Error distribution histogram
subplot(2, 3, 1);
histogram(all_subjects_error, 30, 'FaceAlpha', 0.7, 'EdgeColor', 'black');
xlabel('Decoding Error (degrees)');
ylabel('Number of Trials');
title('Distribution of Decoding Errors');
grid on;
hold on;
xline(group_mean_error, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mean: %.1f°', group_mean_error));
xline(group_median_error, 'g--', 'LineWidth', 2, 'DisplayName', sprintf('Median: %.1f°', group_median_error));
legend;

% Subplot 2: Uncertainty distribution histogram
subplot(2, 3, 2);
histogram(all_subjects_uncertainty, 30, 'FaceAlpha', 0.7, 'EdgeColor', 'black');
xlabel('Uncertainty');
ylabel('Number of Trials');
title('Distribution of Uncertainty Values');
grid on;
hold on;
xline(group_mean_uncertainty, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Mean: %.3f', group_mean_uncertainty));
legend;

% Subplot 3: Error vs Uncertainty scatter
subplot(2, 3, 3);
scatter(all_subjects_uncertainty, all_subjects_error, 20, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Uncertainty');
ylabel('Decoding Error (degrees)');
title('Error vs Uncertainty');
grid on;

% Subplot 4: Box plot of all errors
subplot(2, 3, 4);
boxplot(all_subjects_error, 'Labels', {'All Trials'});
ylabel('Decoding Error (degrees)');
title('Error Distribution');
grid on;

% Subplot 5: Error distribution by bins
subplot(2, 3, 5);
error_bins = 0:20:180;
histogram(all_subjects_error, error_bins, 'FaceAlpha', 0.7, 'EdgeColor', 'black');
xlabel('Decoding Error (degrees)');
ylabel('Frequency');
title('Error Distribution (20° bins)');
grid on;

% Subplot 6: Summary statistics text
subplot(2, 3, 6);
axis off;
summary_text = {
    sprintf('Algorithm: %s', algorithm);
    sprintf('Volume Resolution: %d mm', vol_res);
    sprintf('Subjects: %d', length(valid_subjects));
    sprintf('Total Trials: %d', length(all_subjects_error));
    '';
    sprintf('Group Performance:');
    sprintf('Mean Error: %.2f° ± %.2f°', group_mean_error, group_std_error);
    sprintf('Median Error: %.2f°', group_median_error);
    sprintf('Mean Uncertainty: %.3f ± %.3f', group_mean_uncertainty, group_std_uncertainty);
    '';
    sprintf('Error Range: %.1f° - %.1f°', min(all_subjects_error), max(all_subjects_error));
    sprintf('Uncertainty Range: %.3f - %.3f', min(all_subjects_uncertainty), max(all_subjects_uncertainty));
};

text(0.1, 0.9, summary_text, 'FontSize', 12, 'VerticalAlignment', 'top');

sgtitle(sprintf('%s Decoding Results - Group Analysis (%dmm)', algorithm, vol_res));

% %% Save Figure
% fig_name = sprintf('group_%s_decoding_results_%dmm', lower(algorithm), vol_res);
% saveas(gcf, fullfile(figures_base_dir, [fig_name, '.fig']));
% saveas(gcf, fullfile(figures_base_dir, [fig_name, '.png']));
% fprintf('Figure saved: %s\n', fig_name);

%% Create Additional Summary Figure
figure('Position', [200, 200, 800, 600]);

% Box plot of errors by subject
subplot(2, 1, 1);
subject_errors = [];
subject_labels = {};
for i = 1:length(valid_subjects)
    trials_per_subject = length(all_subjects_error) / length(valid_subjects);
    start_idx = (i-1) * trials_per_subject + 1;
    end_idx = i * trials_per_subject;
    subject_trials = start_idx:end_idx;
    subject_errors{i} = all_subjects_error(subject_trials);
    subject_labels{i} = sprintf('S%02d', valid_subjects(i));
end

boxplot([subject_errors{:}], 'Labels', subject_labels);
ylabel('Decoding Error (degrees)');
title(sprintf('%s Decoding Error Distribution by Subject', algorithm));
grid on;
xtickangle(45);

% Overall performance summary
subplot(2, 1, 2);
axis off;
performance_text = {
    sprintf('%s DECODING PERFORMANCE SUMMARY', algorithm);
    sprintf('Volume Resolution: %d mm', vol_res);
    sprintf('Time Window: 0.5-1.0 seconds');
    '';
    sprintf('GROUP STATISTICS:');
    sprintf('• Mean Error: %.2f° ± %.2f°', group_mean_error, group_std_error);
    sprintf('• Median Error: %.2f°', group_median_error);
    sprintf('• Mean Uncertainty: %.3f ± %.3f', group_mean_uncertainty, group_std_uncertainty);
    '';
    sprintf('SUBJECTS ANALYZED: %d', length(valid_subjects));
    sprintf('TOTAL TRIALS: %d', length(all_subjects_error));
    '';
    sprintf('PERFORMANCE RANGE:');
    sprintf('• Best Trial: %.1f°', min(all_subjects_error));
    sprintf('• Worst Trial: %.1f°', max(all_subjects_error));
    sprintf('• 25th Percentile: %.1f°', prctile(all_subjects_error, 25));
    sprintf('• 75th Percentile: %.1f°', prctile(all_subjects_error, 75));
};

text(0.05, 0.95, performance_text, 'FontSize', 14, 'VerticalAlignment', 'top', 'FontFamily', 'monospace');

sgtitle(sprintf('%s Decoding Performance Summary', algorithm));

% % Save summary figure
% summary_fig_name = sprintf('group_%s_performance_summary_%dmm', lower(algorithm), vol_res);
% saveas(gcf, fullfile(figures_base_dir, [summary_fig_name, '.fig']));
% saveas(gcf, fullfile(figures_base_dir, [summary_fig_name, '.png']));
% fprintf('Summary figure saved: %s\n', summary_fig_name);

% fprintf('\nTAFKAP visualization complete!\n');
% fprintf('Figures saved in: %s\n', figures_base_dir);

end
