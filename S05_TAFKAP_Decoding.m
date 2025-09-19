function S05_TAFKAP_Decoding(subjID, surface_resolution)
%% S05_TAFKAP_Decoding - Decode stimulus location using TAFKAP
%
% This script loads complex beta data from S03 and runs TAFKAP decoding
% to decode stimulus location from beta power in posterior sources.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   surface_resolution - Surface resolution (default: 5124)
%
% Outputs:
%   - Saves TAFKAP decoding results in derivatives/sourceRecon/tafkap_decoding/
%   - Results include: estimates, uncertainty, likelihoods, hyperparameters
%
% Dependencies:
%   - S03_betaPowerInSource.m (must be run first)
%   - TAFKAP installation at /d/DATD/hyper/experiments/Mrugank/wmJointRepresentation/
%
% Example:
%   S05_TAFKAP_Decoding(1, 5124)
%
% Author: Mrugank Dake
% Date: 2025-01-20

if nargin < 1
    error('Subject ID is required');
end
if nargin < 2
    surface_resolution = 5124; % Default resolution
end

restoredefaultpath;
clearvars -except subjID surface_resolution; % Keep inputs
close all; clc;

%% Environment Detection and Path Setup
% Detect if running on HPC or local machine
[~, hostname] = system('hostname');
hostname = strtrim(hostname);

% Check for common HPC indicators
is_hpc = contains(hostname, {'login', 'compute', 'node', 'hpc'}) || ...
         exist('/etc/slurm', 'dir') || ...
         ~isempty(getenv('SLURM_JOB_ID')) || ...
         ~isempty(getenv('PBS_JOBID'));

fprintf('=== MEG TAFKAP Decoding Analysis ===\n');
fprintf('Environment: %s\n', hostname);
fprintf('Detected HPC: %s\n', string(is_hpc));
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Setup paths based on environment
if is_hpc
    % HPC paths
    fieldtrip_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/';
    project_path = '/scratch/mdd9787/meg_prf_greene/megScripts';
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
    tafkap_path = '/scratch/mdd9787/meg_prf_greene/TAFKAP_mgs/';
else
    % Local machine paths
    fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    tafkap_path = '/d/DATD/hyper/experiments/Mrugank/TAFKAP_mgs/';
end

% Verify paths exist
if ~exist(fieldtrip_path, 'dir')
    error('FieldTrip path not found: %s', fieldtrip_path);
end
if ~exist(project_path, 'dir')
    error('Project path not found: %s', project_path);
end
if ~exist(data_base_path, 'dir')
    error('Data base path not found: %s', data_base_path);
end
if ~exist(tafkap_path, 'dir')
    error('TAFKAP path not found: %s', tafkap_path);
end

%% Setup and Initialization
addpath(fieldtrip_path);
addpath(genpath(project_path));
addpath(genpath(tafkap_path));
ft_defaults;
ft_hastoolbox('spm12', 1);

%% Initialize File Paths
% Complex beta data path (input from S03)
beta_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_complexBeta_allTargets_%d.mat', subjID, surface_resolution));

% Output directory for TAFKAP results
output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'tafkap_decoding');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Check if beta data exists
if ~exist(beta_data_path, 'file')
    error('Complex beta data not found at: %s\nPlease run S03_betaPowerInSource.m first!', beta_data_path);
end

%% Load Complex Beta Data
fprintf('Loading complex beta data from: %s\n', beta_data_path);
load(beta_data_path);

fprintf('Loaded complex beta data:\n');
fprintf('  Number of targets: %d\n', length(sourceDataByTarget));
for target = 1:length(sourceDataByTarget)
    if ~isempty(sourceDataByTarget{target})
        fprintf('  Target %d: %d trials\n', target, length(sourceDataByTarget{target}.trial));
    end
end

%% Define Target Locations and Angles
target_locations = 1:10;
target_angles = [0, 25, 50, 130, 155, 180, 205, 230, 310, 335]; % degrees
n_targets = length(target_locations);

% Use original 0-360° angles directly

fprintf('Target angles (0-360°): %s\n', mat2str(target_angles));

%% Select Posterior 25% of Sources
fprintf('Selecting posterior 25%% of sources...\n');

% Load source positions from the original source space data
source_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, surface_resolution));

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
fprintf('Extracting beta power data (0.8-1.5s window)...\n');

% Time windows for beta power analysis
baseline_start = -0.5; % seconds (before stimulus onset)
baseline_end = 0.0;    % seconds (stimulus onset)
time_start = 0.8;      % seconds (analysis window start)
time_end = 1.5;        % seconds (analysis window end)

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

% Shuffle the trials to mix up target angles
fprintf('Shuffling trials to randomize target angle sequence...\n');
shuffle_idx = randperm(trial_count);
all_trials = all_trials(shuffle_idx, :);
all_targets = all_targets(shuffle_idx);

% Z-score the data across trials for each source
fprintf('Z-scoring data across trials for each source...\n');
all_trials = zscore(all_trials, 0, 1); % Z-score across trials (dimension 1)
fprintf('Data z-scored across trials\n');

fprintf('After shuffling - Target angle distribution:\n');
unique_targets = unique(all_targets);
for i = 1:length(unique_targets)
    n_trials_for_target = sum(all_targets == unique_targets(i));
    fprintf('  Target %.1f°: %d trials\n', unique_targets(i), n_trials_for_target);
end

%% Prepare Data for TAFKAP
fprintf('Preparing data for TAFKAP...\n');


%% Run TAFKAP Decoding
fprintf('Running TAFKAP decoding...\n');

% Set up TAFKAP parameters
p = struct();
p.stimval = all_targets(:); % Stimulus values for each trial (0-360 range) - ensure column vector
% Create proper cross-validation splits
% Use 5-fold cross-validation
n_folds = 5;
fold_size = floor(trial_count / n_folds);
p.runNs = repmat((1:n_folds)', fold_size, 1);
if length(p.runNs) < trial_count
    p.runNs = [p.runNs; repmat(n_folds, trial_count - length(p.runNs), 1)];
end

% For each fold, use 80% for training, 20% for testing
p.train_trials = false(trial_count, 1);
p.test_trials = false(trial_count, 1);

% Use first 4 folds for training, last fold for testing
p.train_trials(p.runNs <= 4) = true;
p.test_trials(p.runNs == 5) = true;
p.dec_type = 'TAFKAP'; % Use TAFKAP algorithm
% p.Nboot = 200; % Set bootstrap iterations to 200
p.nchan = 8; % Number of channels

% Verify TAFKAP parameters
fprintf('TAFKAP parameters: %d trials, %d sources\n', size(all_trials, 1), size(all_trials, 2));
fprintf('Data range: %.3f to %.3f (z-scored)\n', min(all_trials(:)), max(all_trials(:)));
fprintf('Stimulus range: %.1f to %.1f degrees\n', min(p.stimval), max(p.stimval));

% Change to TAFKAP directory so relative paths work
original_dir = pwd;
cd(tafkap_path);

% Run TAFKAP
try
    [est, unc, lf, hypers] = TAFKAP_Decode(all_trials, p);
    fprintf('TAFKAP decoding completed successfully\n');
catch ME
    cd(original_dir); % Make sure we change back even if there's an error
    error('TAFKAP decoding failed: %s', ME.message);
end

% Change back to original directory
cd(original_dir);

%% Save Results
fprintf('Saving TAFKAP results...\n');

results = struct();
results.est = est;           % Decoded estimates
results.unc = unc;           % Uncertainty values
results.lf = lf;             % Likelihood functions
results.hypers = hypers;     % Hyperparameters
results.target_ang = all_targets; % Target angles
results.posterior_sources = posterior_sources; % Source indices used
results.n_sources = n_posterior; % Number of sources
results.time_window = [time_start, time_end]; % Time window used
results.target_angles = target_angles; % Target angles (0-360)

% Save results
results_path = fullfile(output_dir, sprintf('sub-%02d_tafkap_results_%d.mat', subjID, surface_resolution));
save(results_path, 'results');
fprintf('Results saved to: %s\n', results_path);

%% Use TAFKAP Results Directly (0-360°)
fprintf('Using TAFKAP results directly (0-360°)...\n');

% TAFKAP outputs are already in 0-360° range
est_original = est; % TAFKAP outputs 0-360°

% Target angles are already in 0-360° space
targets_original = all_targets; % Already 0-360°

fprintf('TAFKAP estimates (0-360°): %s\n', mat2str(est_original(1:min(5, length(est_original)))));
fprintf('Target angles (0-360°): %s\n', mat2str(targets_original(1:min(5, length(targets_original)))));

%% Compute Decoding Performance
fprintf('Computing decoding performance...\n');

% Ensure vectors are the same length before computing errors
if length(targets_original) ~= length(est_original)
    min_len = min(length(targets_original), length(est_original));
    targets_original = targets_original(1:min_len);
    est_original = est_original(1:min_len);
end

% Calculate decoding error (circular distance)
decoding_error = abs(est_original - targets_original);
decoding_error = min(decoding_error, 360 - decoding_error); % Handle circularity

% Calculate mean absolute error
mean_error = mean(decoding_error);
std_error = std(decoding_error);

% Calculate accuracy within different error thresholds
acc_5deg = mean(decoding_error <= 5);
acc_10deg = mean(decoding_error <= 10);
acc_20deg = mean(decoding_error <= 20);

fprintf('Decoding Performance:\n');
fprintf('  Mean absolute error: %.2f° ± %.2f°\n', mean_error, std_error);
fprintf('  Accuracy within 5°: %.1f%%\n', 100*acc_5deg);
fprintf('  Accuracy within 10°: %.1f%%\n', 100*acc_10deg);
fprintf('  Accuracy within 20°: %.1f%%\n', 100*acc_20deg);

% Save performance metrics
performance = struct();
performance.mean_error = mean_error;
performance.std_error = std_error;
performance.acc_5deg = acc_5deg;
performance.acc_10deg = acc_10deg;
performance.acc_20deg = acc_20deg;
performance.decoding_error = decoding_error;

perf_path = fullfile(output_dir, sprintf('sub-%02d_tafkap_performance_%d.mat', subjID, surface_resolution));
save(perf_path, 'performance');
fprintf('Performance metrics saved to: %s\n', perf_path);

%% Create Visualization
fprintf('Creating visualization...\n');

% Check that target and estimate vectors have the same length
if length(targets_original) ~= length(est_original)
    fprintf('Warning: Mismatch in vector lengths - targets: %d, estimates: %d\n', length(targets_original), length(est_original));
    % Use the minimum length to avoid errors
    min_len = min(length(targets_original), length(est_original));
    targets_original = targets_original(1:min_len);
    est_original = est_original(1:min_len);
    fprintf('Truncated to length %d for plotting\n', min_len);
end

figure('Position', [100, 100, 1200, 800]);

% Subplot 1: Target vs Estimated angles
subplot(2, 2, 1);
scatter(targets_original, est_original, 50, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([0, 360], [0, 360], 'r--', 'LineWidth', 2);
xlabel('Target Angle (degrees)');
ylabel('Estimated Angle (degrees)');
title('Target vs Estimated Angles');
axis equal;
xlim([0, 360]);
ylim([0, 360]);
grid on;

% Subplot 2: Decoding error distribution
subplot(2, 2, 2);
histogram(decoding_error, 20, 'FaceAlpha', 0.7);
xlabel('Decoding Error (degrees)');
ylabel('Number of Trials');
title('Distribution of Decoding Errors');
grid on;

% Subplot 3: Error vs Target angle
subplot(2, 2, 3);
scatter(targets_original, decoding_error, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Target Angle (degrees)');
ylabel('Decoding Error (degrees)');
title('Decoding Error vs Target Angle');
grid on;

% Subplot 4: Sample likelihood functions
subplot(2, 2, 4);
sample_trials = 1:min(5, length(est_original));
for i = 1:length(sample_trials)
    trial_idx = sample_trials(i);
    plot(0:360, lf(trial_idx, :), 'LineWidth', 1.5);
    hold on;
end
xlabel('Angle (degrees)');
ylabel('Likelihood');
title('Sample Likelihood Functions');
grid on;
legend(arrayfun(@(x) sprintf('Trial %d', x), sample_trials, 'UniformOutput', false));

sgtitle(sprintf('TAFKAP Decoding Results - Subject %02d', subjID));

% Save figure
fig_path = fullfile(output_dir, sprintf('sub-%02d_tafkap_results_%d.fig', subjID, surface_resolution));
saveas(gcf, fig_path);
fprintf('Figure saved to: %s\n', fig_path);

fprintf('\nTAFKAP decoding analysis complete!\n');
fprintf('Results saved in: %s\n', output_dir);

end
