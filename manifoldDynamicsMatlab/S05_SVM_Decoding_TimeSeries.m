function S05_SVM_Decoding_TimeSeries(subjID, surface_resolution)
% S05_SVM_Decoding_TimeSeries - Run SVM decoding across multiple time points
% 
% This function runs SVM decoding for multiple time windows to analyze
% how decoding performance changes over time.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3...)
%   surface_resolution - Surface resolution (e.g., 5124)

%
% Time points: -0.5 to 1.5s in 0.1s intervals (21 time points)
% Each analysis uses a 0.2s window

%% Input validation and setup
if nargin < 2
    surface_resolution = 5124; % Default surface resolution
end

if nargin < 1
    subjID = 1; % Default subject
end

fprintf('Starting SVM decoding time series for Subject %d, surface resolution %d\n', ...
    subjID, surface_resolution);

%% Define time points
time_points = -0.5:0.1:1.5; % -0.5, -0.4, -0.3, ..., 1.5
n_timepoints = length(time_points);

fprintf('Time points: %s\n', mat2str(time_points));
fprintf('Number of time points: %d\n', n_timepoints);

%% Initialize results storage
results_summary = struct();
results_summary.time_points = time_points;
results_summary.mean_errors = nan(n_timepoints, 1);
results_summary.std_errors = nan(n_timepoints, 1);
results_summary.median_errors = nan(n_timepoints, 1);
results_summary.n_trials = nan(n_timepoints, 1);
results_summary.n_sources = nan(n_timepoints, 1);
results_summary.success = false(n_timepoints, 1);

% Arrays for true and predicted angles
results_summary.true_angles = cell(n_timepoints, 1);
results_summary.pred_angles = cell(n_timepoints, 1);

%% Environment Detection and Path Setup
% Detect if running on HPC or local machine
[~, hostname] = system('hostname');
hostname = strtrim(hostname);

% Check if running on HPC
is_hpc = ~isempty(getenv('SLURM_JOB_ID'));

fprintf('\n=== MEG SVM Time Series Decoding Analysis ===\n');
fprintf('Environment: %s\n', hostname);
fprintf('Detected HPC: %s\n', string(is_hpc));
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);

% Set up paths based on environment
if is_hpc
    % HPC paths
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
    project_path = '/scratch/mdd9787/meg_prf_greene';
    circstat_path = '/scratch/mdd9787/meg_prf_greene/toolboxes/CircStat2012a';
else
    % Local paths
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    circstat_path = '/d/DATD/hyper/toolboxes/CircStat2012a';
end

% Add CircStat toolbox
addpath(circstat_path);

% Add project path to MATLAB path
addpath(genpath(project_path));

%% Load data once (outside the loop)
fprintf('\n=== Loading Data ===\n');

% Set up paths
input_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon');

% Load complex beta data
fprintf('Loading complex beta data...\n');
beta_data_path = fullfile(input_dir, sprintf('sub-%02d_task-mgs_complexBeta_allTargets_%d.mat', subjID, surface_resolution));

if ~exist(beta_data_path, 'file')
    error('Complex beta data not found at: %s\nPlease run S03_betaPowerInSource.m first!', beta_data_path);
end

fprintf('Loading complex beta data from: %s\n', beta_data_path);
load(beta_data_path);

fprintf('Loaded complex beta data:\n');
fprintf('  Number of targets: %d\n', length(sourceDataByTarget));
for target = 1:length(sourceDataByTarget)
    if ~isempty(sourceDataByTarget{target})
        fprintf('  Target %d: %d trials\n', target, length(sourceDataByTarget{target}.trial));
    end
end

% Load source positions
fprintf('Loading source positions...\n');
source_data_path = fullfile(input_dir, sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, surface_resolution));

if ~exist(source_data_path, 'file')
    error('Source space data not found at: %s\nPlease run S02_ReverseModelMNI.m first!', source_data_path);
end

fprintf('Loading source positions from: %s\n', source_data_path);
source_data = load(source_data_path);
source_pos = source_data.source.pos;
n_sources = size(source_pos, 1);
% Use all sources (whole brain) to match Python implementation
posterior_sources = 1:n_sources; % Use all sources

fprintf('  Total sources: %d\n', n_sources);
fprintf('  Using all sources: %d (100.0%%)\n', n_sources);

% Define target angles
target_angles = [0, 25, 50, 130, 155, 180, 205, 230, 310, 335]; % degrees
n_targets = length(target_angles);

fprintf('Target angles: %s\n', mat2str(target_angles));

%% Compute PCA on 0.3-0.5s window for dimensionality reduction
fprintf('\n=== Computing PCA for Dimensionality Reduction ===\n');
fprintf('Computing PCA on 0.3-0.5s window...\n');

% Define PCA window (0.3-0.5s window)
pca_window_start = 0.3;
pca_window_end = 0.5;

% Extract data for PCA computation using the same approach as run_svm_decoding_timepoint
fprintf('  Extracting data for PCA computation (%.1f-%.1fs)...\n', pca_window_start, pca_window_end);

% First pass: collect all baseline data for global baseline calculation
all_baseline_data = [];
baseline_trial_count = 0;
baseline_start = -0.5;
baseline_end = 0.0;

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
        
        % Select posterior sources only
        posterior_complex = complex_data(posterior_sources, :);
        
        % Compute power (magnitude squared)
        power_data = abs(posterior_complex).^2;
        
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
else
    error('No baseline data found for global baseline calculation');
end

% Second pass: extract PCA window data and apply global baseline correction
all_trials_pca = [];
all_targets_pca = [];
trial_count = 0;

for target = 1:n_targets
    if isempty(sourceDataByTarget{target})
        continue;
    end
    
    % Get time vector
    time_vec = sourceDataByTarget{target}.time{1};
    
    % Find time indices for PCA window
    time_idx = time_vec >= pca_window_start & time_vec <= pca_window_end;
    
    if sum(time_idx) == 0
        continue;
    end
    
    % Extract trials for this target
    n_trials = length(sourceDataByTarget{target}.trial);
    
    for trial = 1:n_trials
        trial_count = trial_count + 1;
        
        % Get complex beta data for this trial
        complex_data = sourceDataByTarget{target}.trial{trial};
        
        % Select posterior sources only
        posterior_complex = complex_data(posterior_sources, :);
        
        % Compute power (magnitude squared)
        power_data = abs(posterior_complex).^2;
        
        % Calculate analysis window power (average across analysis window)
        analysis_power = mean(power_data(:, time_idx), 2);
        
        % Apply global baseline correction: (analysis - global_baseline_mean) / global_baseline_mean (% change)
        avg_power = (analysis_power - global_baseline_mean') ./ (global_baseline_mean' + eps);
        
        % Store data
        all_trials_pca(trial_count, :) = avg_power';
        all_targets_pca(trial_count) = target_angles(target);
    end
end

if trial_count == 0
    error('No trials found for PCA computation in window %.1f-%.1fs', pca_window_start, pca_window_end);
end

fprintf('  PCA data: %d trials × %d sources\n', size(all_trials_pca, 1), size(all_trials_pca, 2));

% Apply z-score normalization for PCA
fprintf('  Applying z-score normalization for PCA...\n');
all_trials_pca_normalized = zscore(all_trials_pca, 0, 1);

% Compute PCA
fprintf('  Computing PCA...\n');
[coeff, score, latent, ~, explained] = pca(all_trials_pca_normalized);

% Select number of components (50 dimensions as requested)
n_components = 50;
if n_components > size(coeff, 2)
    n_components = size(coeff, 2);
    fprintf('  Warning: Requested %d components, but only %d available. Using %d components.\n', 50, size(coeff, 2), n_components);
end

% Store PCA components for projection
pca_coeff = coeff(:, 1:n_components);
fprintf('  Selected %d PCA components (%.1f%% variance explained)\n', n_components, sum(explained(1:n_components)));

%% Run SVM decoding for each time point
fprintf('\n=== Running SVM Decoding Across Time Points ===\n');

for t_idx = 1:n_timepoints
    current_time = time_points(t_idx);
    
    fprintf('\n--- Time Point %d/%d: %.1fs ---\n', t_idx, n_timepoints, current_time);
    
    try
        % Run SVM decoding for this time point with PCA projection
        [mean_error, std_error, median_error, n_trials, true_angles, pred_angles] = run_svm_decoding_timepoint_pca(...
            sourceDataByTarget, posterior_sources, target_angles, current_time, subjID, surface_resolution, ...
            global_baseline_mean, pca_coeff, n_components);
        
        % Store results
        results_summary.mean_errors(t_idx) = mean_error;
        results_summary.std_errors(t_idx) = std_error;
        results_summary.median_errors(t_idx) = median_error;
        results_summary.n_trials(t_idx) = n_trials;
        results_summary.n_sources(t_idx) = length(posterior_sources);
        results_summary.success(t_idx) = true;
        
        % Store true and predicted angles
        results_summary.true_angles{t_idx} = true_angles;
        results_summary.pred_angles{t_idx} = pred_angles;
        
        fprintf('  Success: Mean error = %.2f° ± %.2f°\n', mean_error, std_error);
        
    catch ME
        fprintf('  Error: %s\n', ME.message);
        results_summary.success(t_idx) = false;
    end
end

%% Create visualizations
fprintf('\n=== Creating Summary Visualization ===\n');

% Create main figure
fig1 = figure('Position', [100, 100, 1200, 800]);

% Plot 1: Mean error over time
subplot(2, 2, 1);
plot(time_points, results_summary.mean_errors, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.7);
xlabel('Time (seconds)');
ylabel('Mean Absolute Error (degrees)');
title('SVM Decoding Performance Over Time (PCA-Reduced)');
grid on;
xlim([-0.6, 1.6]);

% Plot 2: Standard deviation over time
subplot(2, 2, 2);
plot(time_points, results_summary.std_errors, 'r-s', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.7);
xlabel('Time (seconds)');
ylabel('Error Standard Deviation (degrees)');
title('Error Variability Over Time (PCA-Reduced)');
grid on;
xlim([-0.6, 1.6]);

% Plot 3: Number of trials over time
subplot(2, 2, 3);
yyaxis left;
bar(time_points, results_summary.success * 100, 'FaceAlpha', 0.7);
ylabel('Success Rate (%)');
yyaxis right;
plot(time_points, results_summary.n_trials, 'g-s', 'LineWidth', 2, 'MarkerSize', 4);
ylabel('Number of Trials');
xlabel('Time (seconds)');
title('Success Rate and Trial Count');
grid on;
xlim([-0.6, 1.6]);

% Add legend
legend('Success Rate', 'Trial Count', 'Location', 'best');

% Create additional figure for true vs predicted angles
fig2 = figure('Position', [200, 200, 1200, 800]);

% Find best time point for visualization
[~, best_idx] = min(abs(results_summary.mean_errors));
if ~isnan(results_summary.mean_errors(best_idx)) && ~isempty(results_summary.true_angles{best_idx})
    
    % Plot 1: True vs Predicted scatter plot
    subplot(2, 2, 1);
    true_vals = results_summary.true_angles{best_idx};
    pred_vals = results_summary.pred_angles{best_idx};
    scatter(true_vals, pred_vals, 50, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    plot([0, 360], [0, 360], 'r--', 'LineWidth', 2);
    xlabel('True Angle (degrees)');
    ylabel('Predicted Angle (degrees)');
    title(sprintf('True vs Predicted (Best Time: %.1fs)', time_points(best_idx)));
    axis equal;
    xlim([0, 360]);
    ylim([0, 360]);
    grid on;
    
    % Plot 2: Circular scatter plot
    subplot(2, 2, 2);
    true_rad = deg2rad(true_vals);
    pred_rad = deg2rad(pred_vals);
    polarscatter(true_rad, ones(size(true_rad)), 50, 'filled', 'MarkerFaceAlpha', 0.6, 'MarkerFaceColor', 'blue');
    hold on;
    polarscatter(pred_rad, ones(size(pred_rad)), 30, 'filled', 'MarkerFaceAlpha', 0.8, 'MarkerFaceColor', 'red');
    title(sprintf('Circular Plot (Best Time: %.1fs)', time_points(best_idx)));
    legend('True', 'Predicted', 'Location', 'best');
    
    % Plot 3: Error distribution for best time point
    subplot(2, 2, 3);
    errors = circ_dist(deg2rad(pred_vals), deg2rad(true_vals));
    errors_deg = rad2deg(errors);
    histogram(errors_deg, 20, 'FaceAlpha', 0.7);
    xlabel('Circular Error (degrees)');
    ylabel('Frequency');
    title(sprintf('Error Distribution (Best Time: %.1fs)', time_points(best_idx)));
    grid on;
    
    % Plot 4: Time series of correlation between true and predicted
    subplot(2, 2, 4);
    correlations = nan(n_timepoints, 1);
    for t_idx = 1:n_timepoints
        if ~isempty(results_summary.true_angles{t_idx}) && ~isempty(results_summary.pred_angles{t_idx})
            true_vals = results_summary.true_angles{t_idx};
            pred_vals = results_summary.pred_angles{t_idx};
            % Calculate circular correlation
            correlations(t_idx) = circ_corrcc(deg2rad(true_vals), deg2rad(pred_vals));
        end
    end
    plot(time_points, correlations, 'g-o', 'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    xline(0, 'k--', 'LineWidth', 1, 'Alpha', 0.7);
    xlabel('Time (seconds)');
    ylabel('Circular Correlation');
    title('True vs Predicted Correlation Over Time');
    grid on;
    xlim([-0.6, 1.6]);
    ylim([-1, 1]);
    
else
    % If no valid data, show message
    subplot(2, 2, 1);
    text(0.5, 0.5, 'No valid data for visualization', 'HorizontalAlignment', 'center', 'FontSize', 14);
end

%% Save results
fprintf('Saving summary results...\n');
output_dir = fullfile(input_dir, 'svm_decoding');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save summary results
summary_file = fullfile(output_dir, sprintf('sub-%02d_svm_timeseries_summary_%d.mat', subjID, surface_resolution));
save(summary_file, 'results_summary');
fprintf('Summary saved to: %s\n', summary_file);

% Save figures
fig1_file = fullfile(output_dir, sprintf('sub-%02d_svm_timeseries_%d.fig', subjID, surface_resolution));
savefig(fig1, fig1_file);
fprintf('Figure saved to: %s\n', fig1_file);

png1_file = fullfile(output_dir, sprintf('sub-%02d_svm_timeseries_%d.png', subjID, surface_resolution));
print(fig1, png1_file, '-dpng', '-r300');
fprintf('PNG saved to: %s\n', png1_file);

fig2_file = fullfile(output_dir, sprintf('sub-%02d_svm_angles_%d.fig', subjID, surface_resolution));
savefig(fig2, fig2_file);
fprintf('Angles figure saved to: %s\n', fig2_file);

png2_file = fullfile(output_dir, sprintf('sub-%02d_svm_angles_%d.png', subjID, surface_resolution));
print(fig2, png2_file, '-dpng', '-r300');
fprintf('Angles PNG saved to: %s\n', png2_file);

%% Print summary statistics
fprintf('\n=== SVM Time Series Summary (PCA-Reduced) ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d\n', surface_resolution);
fprintf('PCA components: %d (from 0.3-0.5s window)\n', n_components);
fprintf('Time points analyzed: %d\n', n_timepoints);
fprintf('Successful analyses: %d/%d\n', sum(results_summary.success), n_timepoints);

if any(results_summary.success)
    valid_errors = results_summary.mean_errors(results_summary.success);
    fprintf('Best performance: %.2f° at %.1fs\n', min(valid_errors), time_points(valid_errors == min(valid_errors)));
    fprintf('Worst performance: %.2f° at %.1fs\n', max(valid_errors), time_points(valid_errors == max(valid_errors)));
    fprintf('Overall mean error: %.2f° ± %.2f°\n', mean(valid_errors), std(valid_errors));
end

fprintf('\nSVM time series analysis completed!\n');

end

function [mean_error, std_error, median_error, n_trials, true_angles, pred_angles] = run_svm_decoding_timepoint_pca(...
    sourceDataByTarget, posterior_sources, target_angles, input_time, subjID, surface_resolution, ...
    global_baseline_mean, pca_coeff, n_components)
% run_svm_decoding_timepoint_pca - Run SVM decoding for a single time point with PCA projection
% 
% This function performs SVM decoding for a specific time window using
% pre-loaded data and projects it into PCA space before decoding.
%
% Inputs:
%   sourceDataByTarget - Pre-loaded complex beta data
%   posterior_sources - Pre-selected posterior source indices
%   target_angles - Target angle array
%   input_time - Analysis window start time
%   subjID - Subject ID
%   surface_resolution - Surface resolution
%   global_baseline_mean - Global baseline mean for correction
%   pca_coeff - PCA coefficients for projection
%   n_components - Number of PCA components
%
% Outputs:
%   mean_error - Mean decoding error in degrees
%   std_error - Standard deviation of decoding error
%   median_error - Median decoding error
%   n_trials - Number of trials processed
%   true_angles - True angles for each trial
%   pred_angles - Predicted angles for each trial

%% Set up parameters
time_start = input_time;      % seconds (analysis window start)
time_end = input_time + 0.2;  % seconds (analysis window end)

n_targets = length(target_angles);

%% Extract beta power data for this time point
fprintf('  Extracting beta power data (%.1f-%.1fs window)...\n', time_start, time_end);

% Initialize data matrices
all_trials = [];
all_targets = [];
trial_count = 0;

% Extract analysis window data and apply global baseline correction
for target = 1:n_targets
    if isempty(sourceDataByTarget{target})
        continue;
    end
    
    % Get time vector
    time_vec = sourceDataByTarget{target}.time{1};
    
    % Find time indices for analysis window
    time_idx = time_vec >= time_start & time_vec <= time_end;
    
    if sum(time_idx) == 0
        fprintf('    Warning: No time points found for target %d in window %.1f-%.1fs\n', target, time_start, time_end);
        continue;
    end
    
    % Extract trials for this target
    n_trials = length(sourceDataByTarget{target}.trial);
    
    for trial = 1:n_trials
        trial_count = trial_count + 1;
        
        % Get complex beta data for this trial
        complex_data = sourceDataByTarget{target}.trial{trial};
        
        % Select posterior sources only
        posterior_complex = complex_data(posterior_sources, :);
        
        % Compute power (magnitude squared)
        power_data = abs(posterior_complex).^2;
        
        % Calculate analysis window power (average across analysis window)
        analysis_power = mean(power_data(:, time_idx), 2);
        
        % Apply global baseline correction: (analysis - global_baseline_mean) / global_baseline_mean (% change)
        avg_power = (analysis_power - global_baseline_mean') ./ (global_baseline_mean' + eps);
        
        % Store data
        all_trials(trial_count, :) = avg_power';
        all_targets(trial_count) = target_angles(target);
    end
end

if trial_count == 0
    error('No trials found for time window %.1f-%.1fs', time_start, time_end);
end

% Apply z-score normalization across all trials
fprintf('  Applying z-score normalization across all trials...\n');
all_trials = zscore(all_trials, 0, 1); % z-score along columns (sources)

fprintf('  Processed %d trials\n', trial_count);

% Project data into PCA space
fprintf('  Projecting data into PCA space (%d → %d dimensions)...\n', size(all_trials, 2), n_components);
all_trials_pca = all_trials * pca_coeff;

%% Shuffle the trials to mix up target angles
shuffle_idx = randperm(trial_count);
all_trials_pca = all_trials_pca(shuffle_idx, :);
all_targets = all_targets(shuffle_idx);

%% Prepare data for SVM
% Convert angles to radians for sine/cosine computation
angle_rad = (pi/180) * all_targets;

% Compute sine and cosine of target angles
target_sin = sin(angle_rad);
target_cos = cos(angle_rad);

% Add intercept term (column of ones) to the data
data_with_intercept = [all_trials_pca, ones(trial_count, 1)];

%% Set up cross-validation
n_folds = 5;
fold_size = floor(trial_count / n_folds);
run_ind = repmat((1:n_folds)', fold_size, 1);
if length(run_ind) < trial_count
    run_ind = [run_ind; repmat(n_folds, trial_count - length(run_ind), 1)];
end

% Shuffle run assignments
run_ind = run_ind(randperm(length(run_ind)));

%% Run SVM Decoding
fprintf('  Running SVM decoding...\n');

% Initialize results
est = nan(trial_count, 1);

% Use regular for loop for cross-validation
for testrun_idx = 1:n_folds
    % Set up train/test splits
    test_trials = run_ind == testrun_idx;
    train_trials = run_ind ~= testrun_idx;
    
    % Get training and test data
    train_data = data_with_intercept(train_trials, :);
    test_data = data_with_intercept(test_trials, :);
    train_sin = target_sin(train_trials);
    train_cos = target_cos(train_trials);
    
    % Train sine model using SVR
    try
        sin_model = fitrlinear(train_data, train_sin, ...
            'Learner', 'svm', ...
            'Regularization', 'ridge', ...
            'Lambda', 1e-3);
        
        % Train cosine model using SVR
        cos_model = fitrlinear(train_data, train_cos, ...
            'Learner', 'svm', ...
            'Regularization', 'ridge', ...
            'Lambda', 1e-3);
        
        % Predict sine and cosine for test data
        pred_sin = predict(sin_model, test_data);
        pred_cos = predict(cos_model, test_data);
        
        % Convert back to orientation using atan2
        pred_angles = atan2(pred_sin, pred_cos) * (180/pi);
        
        % Ensure angles are in [0, 360] range
        pred_angles = mod(pred_angles, 360);
        
        % Store results
        est(test_trials) = pred_angles;
        
    catch ME
        fprintf('    Error in fold %d: %s\n', testrun_idx, ME.message);
        % Leave estimates as NaN for this fold
    end
end

%% Compute performance metrics
% Ensure vectors are the same length before computing errors
if length(all_targets) ~= length(est)
    min_len = min(length(all_targets), length(est));
    all_targets = all_targets(1:min_len);
    est = est(1:min_len);
end

% Calculate decoding error using circular statistics
% Convert angles to radians for circ_dist
est_rad = circ_ang2rad(est);
target_rad = circ_ang2rad(all_targets');

% Calculate circular distance (in radians)
circular_errors_rad = circ_dist(est_rad, target_rad);
% Convert back to degrees
circular_errors_deg = circ_rad2ang(circular_errors_rad);

% Calculate performance metrics using mean absolute error
mean_error = mean(abs(circular_errors_deg));
std_error = std(abs(circular_errors_deg));
median_error = median(abs(circular_errors_deg));
n_trials = length(all_targets);

% Return true and predicted angles for visualization
true_angles = all_targets';
pred_angles = est;

end