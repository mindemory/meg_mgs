function S05_SVM_Decoding(subjID, surface_resolution, input_time)
% S05_SVM_Decoding - SVM-based orientation decoding for MEG data
% 
% This function implements Support Vector Regression (SVR) for decoding
% orientation from MEG source-localized beta power data, following the
% approach described in the SVM decoder document.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3...)
%   surface_resolution - Surface resolution (e.g., 5124)
%   input_time - Analysis window start time in seconds (e.g., 0.8)
%
% The method uses two separate SVR models to predict sine and cosine of
% orientation, then combines them using atan2 to get the final estimate.

%% Input validation and setup
if nargin < 3
    input_time = 0.8; % Default start time
end

if nargin < 2
    surface_resolution = 5124; % Default surface resolution
end

if nargin < 1
    subjID = 1; % Default subject
end

fprintf('Starting SVM decoding for Subject %d, surface resolution %d, time %.1fs\n', ...
    subjID, surface_resolution, input_time);

% Clear workspace but keep inputs
clearvars -except subjID surface_resolution algorithm input_time;

%% Set up paths and parameters
% Analysis parameters
time_start = input_time;      % seconds (analysis window start)
time_end = input_time + 0.5;  % seconds (analysis window end)
baseline_start = -0.5;        % seconds (baseline window start)
baseline_end = 0.0;           % seconds (baseline window end)

% Paths
if ispc
    base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
else
    base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
end

% Input and output paths
input_dir = fullfile(base_path, sprintf('sub-%02d', subjID), 'sourceRecon');
output_dir = fullfile(base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'svm_decoding');

% Create output directory if it doesn't exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% Load Complex Beta Data
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

%% Define target angles (0-360 degrees)
target_angles = [0, 25, 50, 130, 155, 180, 205, 230, 310, 335]; % degrees
n_targets = length(target_angles);

fprintf('Target angles: %s\n', mat2str(target_angles));

%% Select Posterior 25% of Sources
fprintf('Selecting posterior 25%% of sources...\n');

% Load source positions from the original source space data
source_data_path = fullfile(input_dir, sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, surface_resolution));

if ~exist(source_data_path, 'file')
    error('Source space data not found at: %s\nPlease run S02_ReverseModelMNI.m first!', source_data_path);
end

fprintf('Loading source positions from: %s\n', source_data_path);
source_data = load(source_data_path);
source_pos = source_data.source.pos;
n_sources = size(source_pos, 1);
n_posterior = round(n_sources * 0.25); % 25% of sources

% Select posterior sources (lowest Y coordinates)
[~, posterior_idx] = sort(source_pos(:, 2), 'ascend');
posterior_sources = posterior_idx(1:n_posterior);

fprintf('  Total sources: %d\n', n_sources);
fprintf('  Selected posterior sources: %d (%.1f%%)\n', n_posterior, 100*n_posterior/n_sources);


%% Extract Beta Power Data
% Initialize data matrices
all_trials = [];
all_targets = [];
trial_count = 0;

for target = 1:n_targets
    if isempty(sourceDataByTarget{target})
        continue;
    end
    
    % Get time vector
    time_vec = sourceDataByTarget{target}.time{1};
    
    % Find time indices for baseline and analysis windows
    baseline_idx = time_vec >= baseline_start & time_vec <= baseline_end;
    time_idx = time_vec >= time_start & time_vec <= time_end;
    
    if sum(baseline_idx) == 0
        fprintf('  Warning: No baseline time points found for target %d in window %.1f-%.1fs\n', target, baseline_start, baseline_end);
        continue;
    end
    
    if sum(time_idx) == 0
        fprintf('  Warning: No time points found for target %d in window %.1f-%.1fs\n', target, time_start, time_end);
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
        
        % Calculate baseline power (average across baseline window)
        baseline_power = mean(power_data(:, baseline_idx), 2);
        
        % Calculate analysis window power (average across analysis window)
        analysis_power = mean(power_data(:, time_idx), 2);
        
        % Baseline correct: subtract baseline from analysis window
        avg_power = analysis_power - baseline_power;
        
        % Store data
        all_trials(trial_count, :) = avg_power';
        all_targets(trial_count) = target_angles(target);
    end
    
    fprintf('  Target %d: %d trials processed\n', target, n_trials);
end

fprintf('Total trials processed: %d\n', trial_count);
fprintf('Data matrix size: %d trials × %d sources\n', size(all_trials, 1), size(all_trials, 2));

%% Shuffle the trials to mix up target angles
fprintf('Shuffling trials to randomize target angle sequence...\n');
shuffle_idx = randperm(trial_count);
all_trials = all_trials(shuffle_idx, :);
all_targets = all_targets(shuffle_idx);

%% Z-score the data across trials for each source
fprintf('Z-scoring data across trials for each source...\n');
all_trials = zscore(all_trials, 0, 1); % Z-score across trials (dimension 1)
fprintf('Data z-scored across trials\n');

%% Prepare data for SVM
fprintf('Preparing data for SVM decoding...\n');

% Convert angles to radians for sine/cosine computation
% Convert from 0-360° to 0-2π radians
angle_rad = (pi/180) * all_targets;

% Compute sine and cosine of target angles
target_sin = sin(angle_rad);
target_cos = cos(angle_rad);

% Add intercept term (column of ones) to the data
data_with_intercept = [all_trials, ones(trial_count, 1)];

fprintf('Target angle distribution after shuffling:\n');
unique_targets = unique(all_targets);
for i = 1:length(unique_targets)
    n_trials_for_target = sum(all_targets == unique_targets(i));
    fprintf('  %d°: %d trials\n', unique_targets(i), n_trials_for_target);
end

%% Set up cross-validation
fprintf('Setting up cross-validation...\n');

% Create run indices for leave-one-run-out cross-validation
n_folds = 5;
fold_size = floor(trial_count / n_folds);
run_ind = repmat((1:n_folds)', fold_size, 1);
if length(run_ind) < trial_count
    run_ind = [run_ind; repmat(n_folds, trial_count - length(run_ind), 1)];
end

% Shuffle run assignments
run_ind = run_ind(randperm(length(run_ind)));

fprintf('Cross-validation: %d folds\n', n_folds);
fprintf('Trials per fold: ~%d\n', round(trial_count / n_folds));

%% Run SVM Decoding
fprintf('Running SVM decoding...\n');

% Initialize results
est = nan(trial_count, 1);

% Use regular for loop for cross-validation
for testrun_idx = 1:n_folds
    fprintf('Processing fold %d/%d...\n', testrun_idx, n_folds);
    
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
        % Convert from radians back to degrees
        pred_angles = atan2(pred_sin, pred_cos) * (180/pi);
        
        % Ensure angles are in [0, 360] range
        pred_angles = mod(pred_angles, 360);
        
        % Store results
        est(test_trials) = pred_angles;
        
    catch ME
        fprintf('Error in fold %d: %s\n', testrun_idx, ME.message);
        % Leave estimates as NaN for this fold
    end
end


%% Save Results
fprintf('Saving SVM results...\n');

results = struct();
results.est = est;                    % Decoded estimates
results.target_ang = all_targets;     % Target angles
results.posterior_sources = posterior_sources; % Source indices used
results.n_sources = n_posterior;      % Number of sources
results.time_window = [time_start, time_end]; % Time window used
results.target_angles = target_angles; % Target angles (0-360)
results.method = 'SVM_SVR';           % Method used

% Save results
results_path = fullfile(output_dir, sprintf('sub-%02d_svm_results_%d_t%.1f.mat', subjID, surface_resolution, input_time));
save(results_path, 'results');
fprintf('Results saved to: %s\n', results_path);

%% Compute Decoding Performance
fprintf('Computing decoding performance...\n');

% Ensure vectors are the same length before computing errors
if length(all_targets) ~= length(est)
    min_len = min(length(all_targets), length(est));
    all_targets = all_targets(1:min_len);
    est = est(1:min_len);
end

% Calculate decoding error (circular distance)
decoding_error = abs(est - all_targets');
decoding_error = min(decoding_error, 360 - decoding_error); % Handle circularity

% Calculate performance metrics
mean_error = mean(decoding_error);
std_error = std(decoding_error);
median_error = median(decoding_error);

fprintf('\n=== SVM Decoding Performance ===\n');
fprintf('Mean absolute error: %.2f° ± %.2f°\n', mean_error, std_error);
fprintf('Median absolute error: %.2f°\n', median_error);


%% Create visualization
fprintf('Creating visualization...\n');

% Create figure
fig = figure('Position', [100, 100, 1200, 800]);

% Scatter plot: predicted vs actual
subplot(2, 2, 1);
scatter(all_targets, est, 50, 'filled');
xlabel('Target Angle (degrees)');
ylabel('Predicted Angle (degrees)');
title('SVM: Predicted vs Target Angles');
axis equal;
xlim([0, 360]);
ylim([0, 360]);
grid on;

% Add diagonal line
hold on;
plot([0, 360], [0, 360], 'r--', 'LineWidth', 2);
hold off;

% Error distribution
subplot(2, 2, 2);
histogram(decoding_error, 20);
xlabel('Decoding Error (degrees)');
ylabel('Frequency');
title('Distribution of Decoding Errors');
grid on;

% Error by target angle
subplot(2, 2, 3);
error_by_target = zeros(n_targets, 1);
for i = 1:n_targets
    target_mask = all_targets == target_angles(i);
    if sum(target_mask) > 0
        error_by_target(i) = mean(decoding_error(target_mask));
    end
end
bar(target_angles, error_by_target);
xlabel('Target Angle (degrees)');
ylabel('Mean Decoding Error (degrees)');
title('Decoding Error by Target Angle');
grid on;

% Circular plot
subplot(2, 2, 4);
% Convert to radians for circular plot
target_rad = deg2rad(all_targets);
est_rad = deg2rad(est);
% Plot target angles
polarscatter(target_rad, ones(size(target_rad)), 50, 'b', 'filled');
hold on;
% Plot predicted angles
polarscatter(est_rad, ones(size(est_rad)), 30, 'r', 'filled');
title('Circular Plot: Target (blue) vs Predicted (red)');
legend('Target', 'Predicted', 'Location', 'best');
hold off;

% Save figure
fig_path = fullfile(output_dir, sprintf('sub-%02d_svm_decoding_%d_t%.1f.fig', subjID, surface_resolution, input_time));
savefig(fig, fig_path);
fprintf('Figure saved to: %s\n', fig_path);

% Save PNG
png_path = fullfile(output_dir, sprintf('sub-%02d_svm_decoding_%d_t%.1f.png', subjID, surface_resolution, input_time));
print(fig, png_path, '-dpng', '-r300');
fprintf('PNG saved to: %s\n', png_path);

%% Performance summary
fprintf('\n=== SVM Decoding Summary ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d\n', surface_resolution);
fprintf('Analysis window: %.1f-%.1fs\n', time_start, time_end);
fprintf('Number of sources: %d\n', n_posterior);
fprintf('Number of trials: %d\n', trial_count);
fprintf('Cross-validation folds: %d\n', n_folds);
fprintf('Mean decoding error: %.2f°\n', mean_error);
fprintf('Results saved to: %s\n', results_path);

fprintf('\nSVM decoding completed successfully!\n');

end
