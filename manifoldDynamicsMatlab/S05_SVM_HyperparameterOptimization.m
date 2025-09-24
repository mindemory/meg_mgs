function S05_SVM_HyperparameterOptimization(subjID, surface_resolution)
% S05_SVM_HyperparameterOptimization - Optimize SVM hyperparameters for MEG decoding
% 
% This script performs hyperparameter optimization for SVM decoding using
% a fixed time window (0.8-1.5s) and tests different parameter combinations.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, ...)
%   surface_resolution - Surface resolution (5124 or 8196)
%
% Outputs:
%   Saves results to svm_decoding directory with hyperparameter optimization results

%% Environment Detection and Path Setup
% Detect if running on HPC or local machine
[~, hostname] = system('hostname');
hostname = strtrim(hostname);

% Check for common HPC indicators
is_hpc = contains(hostname, {'login', 'compute', 'node', 'hpc'}) || ...
         exist('/etc/slurm', 'dir') || ...
         ~isempty(getenv('SLURM_JOB_ID')) || ...
         ~isempty(getenv('PBS_JOBID'));

fprintf('=== MEG SVM Hyperparameter Optimization ===\n');
fprintf('Environment: %s\n', hostname);
fprintf('Detected HPC: %s\n', string(is_hpc));
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Setup paths based on environment
if is_hpc
    % HPC paths
    project_path = '/scratch/mdd9787/meg_prf_greene/megScripts';
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
    circstat_path = '/scratch/mdd9787/meg_prf_greene/CircStat2012a/';
else
    % Local machine paths
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    circstat_path = '/d/DATD/hyper/toolboxes/CircStat2012a/';
end

% Add CircStat toolbox to path
addpath(circstat_path);

%% Set up parameters
time_start = 0.8;      % seconds (analysis window start)
time_end = 1.5;        % seconds (analysis window end)
baseline_start = -0.5; % seconds (baseline window start)
baseline_end = 0.0;    % seconds (baseline window end)

% Define target angles
target_angles = [0, 25, 50, 130, 155, 180, 205, 230, 310, 335]; % degrees
n_targets = length(target_angles);

%% Load data
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

fprintf('Target angles: %s\n', mat2str(target_angles));

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
    fprintf('  Baseline mean: %.4f ± %.4f (across sources)\n', mean(global_baseline_mean), mean(global_baseline_std));
else
    error('No baseline data found for global baseline calculation');
end

%% Extract data for hyperparameter optimization
fprintf('\n=== Extracting Data for Hyperparameter Optimization ===\n');
fprintf('Using time window: %.1f-%.1fs\n', time_start, time_end);

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
        avg_power = (analysis_power - global_baseline_mean');% ./ (global_baseline_mean' + eps);
        
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

%% Define hyperparameter search space
fprintf('\n=== Hyperparameter Search Space ===\n');

% Linear regression parameters to test
C_values = [0.001, 0.01, 0.1, 1, 10, 100, 1000];  % Regularization parameter (Lambda = 1/C)

% Create parameter combinations
param_combinations = [];
param_idx = 1;

for c_idx = 1:length(C_values)
    param_combinations(param_idx).C = C_values(c_idx);
    param_combinations(param_idx).Lambda = 1/C_values(c_idx);
    param_idx = param_idx + 1;
end

fprintf('Testing %d parameter combinations:\n', length(param_combinations));
fprintf('  C values: %s\n', mat2str(C_values));
fprintf('  Lambda values (1/C): %s\n', mat2str(1./C_values));
fprintf('  Using linear regression with ridge regularization\n');

%% Cross-validation setup
n_folds = 5;
cv_indices = crossvalind('Kfold', trial_count, n_folds);

%% Run hyperparameter optimization
fprintf('\n=== Running Hyperparameter Optimization ===\n');

% Initialize results storage
results = struct();
results.param_combinations = param_combinations;
results.cv_errors = zeros(length(param_combinations), n_folds);
results.mean_cv_errors = zeros(length(param_combinations), 1);
results.std_cv_errors = zeros(length(param_combinations), 1);
results.best_params = [];
results.best_error = inf;

% Shuffle the trials to mix up target angles
shuffle_idx = randperm(trial_count);
all_trials_shuffled = all_trials(shuffle_idx, :);
all_targets_shuffled = all_targets(shuffle_idx);

for param_idx = 1:length(param_combinations)
    fprintf('\n--- Parameter Combination %d/%d ---\n', param_idx, length(param_combinations));
    fprintf('C: %.3f, Lambda: %.3f\n', ...
        param_combinations(param_idx).C, ...
        param_combinations(param_idx).Lambda);
    
    % Run cross-validation for this parameter combination
    fold_errors = zeros(n_folds, 1);
    
    for fold = 1:n_folds
        % Split data into train/test
        test_idx = (cv_indices == fold);
        train_idx = ~test_idx;
        
        X_train = all_trials_shuffled(train_idx, :);
        y_train = all_targets_shuffled(train_idx);
        X_test = all_trials_shuffled(test_idx, :);
        y_test = all_targets_shuffled(test_idx);
        
        % Convert angles to radians and then to sine/cosine
        y_train_rad = deg2rad(y_train);
        y_test_rad = deg2rad(y_test);
        
        % Create sine and cosine targets
        y_train_sin = sin(y_train_rad);
        y_train_cos = cos(y_train_rad);
        y_test_sin = sin(y_test_rad);
        y_test_cos = cos(y_test_rad);
        
        % Train sine linear regression model
        svm_sin = fitrlinear(X_train, y_train_sin, ...
            'Lambda', 1/param_combinations(param_idx).C, ...
            'Learner', 'svm', ...
            'Regularization', 'ridge');
        
        % Train cosine linear regression model
        svm_cos = fitrlinear(X_train, y_train_cos, ...
            'Lambda', 1/param_combinations(param_idx).C, ...
            'Learner', 'svm', ...
            'Regularization', 'ridge');
        
        % Make predictions
        pred_sin = predict(svm_sin, X_test);
        pred_cos = predict(svm_cos, X_test);
        
        % Convert back to angles
        y_pred_rad = atan2(pred_sin, pred_cos);
        y_pred = rad2deg(y_pred_rad);
        
        % Calculate circular error
        true_rad = deg2rad(y_test(:));
        pred_rad = deg2rad(y_pred(:));
        errors_rad = circ_dist(pred_rad, true_rad);
        errors_deg = rad2deg(errors_rad);
        
        % Calculate mean absolute error
        fold_errors(fold) = mean(abs(errors_deg));
    end
    
    % Store results
    results.cv_errors(param_idx, :) = fold_errors;
    results.mean_cv_errors(param_idx) = mean(fold_errors);
    results.std_cv_errors(param_idx) = std(fold_errors);
    
    fprintf('  CV Error: %.2f° ± %.2f°\n', results.mean_cv_errors(param_idx), results.std_cv_errors(param_idx));
    
    % Update best parameters
    if results.mean_cv_errors(param_idx) < results.best_error
        results.best_error = results.mean_cv_errors(param_idx);
        results.best_params = param_combinations(param_idx);
        fprintf('  *** New best parameters! ***\n');
    end
end

%% Find and display best parameters
fprintf('\n=== Hyperparameter Optimization Results ===\n');
fprintf('Best parameters found:\n');
fprintf('  C: %.3f\n', results.best_params.C);
fprintf('  Lambda: %.3f\n', results.best_params.Lambda);
fprintf('  Best CV Error: %.2f° ± %.2f°\n', results.best_error, results.std_cv_errors(find([results.param_combinations.C] == results.best_params.C)));

%% Create visualization
fprintf('\n=== Creating Visualization ===\n');

% Create figure
fig = figure('Position', [100, 100, 1200, 600]);

% Plot 1: CV error vs parameter combinations
subplot(1, 2, 1);
plot(1:length(param_combinations), results.mean_cv_errors, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(1:length(param_combinations), results.mean_cv_errors + results.std_cv_errors, 'b--', 'LineWidth', 1);
plot(1:length(param_combinations), results.mean_cv_errors - results.std_cv_errors, 'b--', 'LineWidth', 1);
xlabel('Parameter Combination');
ylabel('CV Error (degrees)');
title('Cross-Validation Error vs Parameter Combinations');
grid on;

% Highlight best parameter
[~, best_idx] = min(results.mean_cv_errors);
plot(best_idx, results.mean_cv_errors(best_idx), 'ro', 'MarkerSize', 10, 'LineWidth', 2);

% Plot 2: Error by C value
subplot(1, 2, 2);
plot([param_combinations.C], results.mean_cv_errors, 'g-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot([param_combinations.C], results.mean_cv_errors + results.std_cv_errors, 'g--', 'LineWidth', 1);
plot([param_combinations.C], results.mean_cv_errors - results.std_cv_errors, 'g--', 'LineWidth', 1);
xlabel('C (Regularization Parameter)');
ylabel('CV Error (degrees)');
title('Error vs C (Linear Regression)');
set(gca, 'XScale', 'log');
grid on;

% Highlight best parameter
plot(results.best_params.C, results.best_error, 'ro', 'MarkerSize', 10, 'LineWidth', 2);

sgtitle(sprintf('Linear Regression Hyperparameter Optimization - Subject %d (%.1f-%.1fs)', subjID, time_start, time_end), 'FontSize', 16, 'FontWeight', 'bold');

%% Save results
fprintf('\n=== Saving Results ===\n');

% Create output directory
output_dir = fullfile(input_dir, 'svm_decoding');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Save results
results_file = fullfile(output_dir, sprintf('sub-%02d_svm_hyperopt_%d.mat', subjID, surface_resolution));
save(results_file, 'results', 'time_start', 'time_end', 'baseline_start', 'baseline_end', 'n_folds', 'trial_count');
fprintf('Results saved to: %s\n', results_file);

% Save figure
fig_file = fullfile(output_dir, sprintf('sub-%02d_svm_hyperopt_%d.fig', subjID, surface_resolution));
savefig(fig, fig_file);
fprintf('Figure saved to: %s\n', fig_file);

% Save as PNG
png_file = fullfile(output_dir, sprintf('sub-%02d_svm_hyperopt_%d.png', subjID, surface_resolution));
print(fig, png_file, '-dpng', '-r300');
fprintf('PNG saved to: %s\n', png_file);

%% Final summary
fprintf('\n=== Hyperparameter Optimization Summary ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d\n', surface_resolution);
fprintf('Time window: %.1f-%.1fs\n', time_start, time_end);
fprintf('Number of trials: %d\n', trial_count);
fprintf('Number of sources: %d\n', n_sources);
fprintf('Cross-validation folds: %d\n', n_folds);
fprintf('Parameter combinations tested: %d\n', length(param_combinations));
fprintf('Best CV error: %.2f° ± %.2f°\n', results.best_error, results.std_cv_errors(best_idx));
fprintf('Best parameters:\n');
fprintf('  C: %.3f\n', results.best_params.C);
fprintf('  Lambda: %.3f\n', results.best_params.Lambda);

fprintf('\nSVM hyperparameter optimization completed!\n');

end
