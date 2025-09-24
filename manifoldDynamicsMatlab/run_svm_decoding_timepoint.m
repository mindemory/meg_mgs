function [mean_error, std_error, median_error, n_trials, true_angles, pred_angles] = run_svm_decoding_timepoint(...
    sourceDataByTarget, posterior_sources, target_angles, input_time, subjID, surface_resolution)
% run_svm_decoding_timepoint - Run SVM decoding for a single time point
% 
% This function performs SVM decoding for a specific time window using
% pre-loaded data to avoid repeated data loading.
%
% Inputs:
%   sourceDataByTarget - Pre-loaded complex beta data
%   posterior_sources - Pre-selected posterior source indices

% CircStat toolbox should already be added by the calling script
%   target_angles - Target angle array
%   input_time - Analysis window start time
%   subjID - Subject ID
%   surface_resolution - Surface resolution
%
% Outputs:
%   mean_error - Mean decoding error in degrees
%   std_error - Standard deviation of decoding error
%   median_error - Median decoding error
%   n_trials - Number of trials processed

%% Set up parameters
time_start = input_time;      % seconds (analysis window start)
time_end = input_time + 0.2;  % seconds (analysis window end)
baseline_start = -0.5;        % seconds (baseline window start)
baseline_end = 0.0;           % seconds (baseline window end)

n_targets = length(target_angles);

%% Extract beta power data for this time point
fprintf('  Extracting beta power data (%.1f-%.1fs window)...\n', time_start, time_end);

% Initialize data matrices
all_trials = [];
all_targets = [];
trial_count = 0;

% First pass: collect all baseline data for global baseline calculation
all_baseline_data = [];
baseline_trial_count = 0;

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
else
    error('No baseline data found for global baseline calculation');
end

% Second pass: extract analysis window data and apply global baseline correction
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

%% Shuffle the trials to mix up target angles
shuffle_idx = randperm(trial_count);
all_trials = all_trials(shuffle_idx, :);
all_targets = all_targets(shuffle_idx);

% Note: Global baseline correction already includes z-scoring, so no additional z-scoring needed

%% Prepare data for SVM
% Convert angles to radians for sine/cosine computation
angle_rad = (pi/180) * all_targets;

% Compute sine and cosine of target angles
target_sin = sin(angle_rad);
target_cos = cos(angle_rad);

% Add intercept term (column of ones) to the data
data_with_intercept = [all_trials, ones(trial_count, 1)];

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

% Calculate performance metrics using circular statistics
mean_error = circ_rad2ang(circ_mean(circular_errors_rad));
std_error = circ_rad2ang(circ_std(circular_errors_rad));
median_error = circ_rad2ang(circ_median(circular_errors_rad));
n_trials = length(all_targets);

% Return true and predicted angles for visualization
true_angles = all_targets';
pred_angles = est;

end
