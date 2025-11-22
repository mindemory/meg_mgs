function S05_TAFKAP_Decoding(subjID, vol_res, algorithm)
%% S05_TAFKAP_Decoding - Decode stimulus location using TAFKAP or PRINCE
%
% This script loads complex beta data from S03 and runs TAFKAP or PRINCE decoding
% to decode stimulus location from beta power in posterior sources.
% Uses average activity between 0.5-1.0s.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   vol_res - Volume resolution in mm (e.g., 8, 4, 2, etc.)
%   algorithm - 'TAFKAP' or 'PRINCE' (default: 'TAFKAP')
%
% Outputs:
%   - Saves decoding results in derivatives/sourceRecon/tafkap_decoding/ or prince_decoding/
%   - Results include: estimates, uncertainty, likelihoods, hyperparameters
%
% Dependencies:
%   - S03_betaPowerInSource.m (must be run first)
%   - TAFKAP installation at /d/DATD/hyper/experiments/Mrugank/wmJointRepresentation/
%
% Example:
%   S05_TAFKAP_Decoding(1, 8, 'TAFKAP')
%   S05_TAFKAP_Decoding(1, 4, 'PRINCE')
%
% Author: Mrugank Dake
% Date: 2025-01-20

if nargin < 1
    error('Subject ID is required');
end
if nargin < 2
    vol_res = 8; % Default volume resolution in mm
end
if nargin < 3
    algorithm = 'TAFKAP'; % Default to TAFKAP
end

% Validate algorithm choice
if ~ismember(algorithm, {'TAFKAP', 'PRINCE'})
    error('Algorithm must be either ''TAFKAP'' or ''PRINCE''');
end

restoredefaultpath;
clearvars -except subjID vol_res algorithm; % Keep inputs
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
fprintf('Volume resolution: %d mm\n', vol_res);

%% Setup paths based on environment
if is_hpc
    % HPC paths
    project_path = '/scratch/mdd9787/meg_prf_greene/megScripts';
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
    tafkap_path = '/scratch/mdd9787/meg_prf_greene/TAFKAP_mgs/';
else
    % Local machine paths
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    tafkap_path = '/d/DATD/hyper/experiments/Mrugank/TAFKAP_mgs/';
end

% Verify paths exist
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
addpath(genpath(project_path));
addpath(genpath(tafkap_path));

%% Load Data
% Load the data using environment-appropriate path
data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'betaDecodingVC', sprintf('sub-%02d_task-mgs_sourceSpaceData_%dmm.mat', subjID, vol_res));
fprintf('Loading data from: %s\n', data_path);
data = load(data_path);

% Extract the loaded variables
visual_data_matrix = data.visual_data_matrix;  % 153x80x597 (trials x time x sources)
target_labels = data.target_labels;           % 1x153 (target locations 1-10)
time_vector = data.time_vector;               % 80x1 (time points)
i_sacc_err = data.i_sacc_err;               % 1x153 (saccade errors)

fprintf('Loaded data structure:\n');
fprintf('  Visual data: %s (trials x time x sources)\n', mat2str(size(visual_data_matrix)));
fprintf('  Target labels: %s\n', mat2str(size(target_labels)));
fprintf('  Time vector: %s\n', mat2str(size(time_vector)));
fprintf('  Saccade errors: %s\n', mat2str(size(i_sacc_err)));

%% Define Target Locations and Angles
% Define angle mapping (in degrees)
angle_mapping = containers.Map({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, ...
                               {0, 25, 50, 130, 155, 180, 205, 230, 310, 335});

% Convert target labels to angles
target_angles = arrayfun(@(x) angle_mapping(x), target_labels);
n_trials = length(target_angles);
n_timepoints = length(time_vector);
n_sources = size(visual_data_matrix, 3);

fprintf('Target angles (0-360°): %s\n', mat2str(unique(target_angles)));
fprintf('Number of trials: %d\n', n_trials);
fprintf('Number of time points: %d\n', n_timepoints);
fprintf('Number of visual sources: %d\n', n_sources);

%% Extract Average Activity Between 0.5-1.0s
fprintf('Extracting average activity between 0.5-1.0s...\n');

% Find time indices for the analysis window (0.5-1.0s)
time_start = 0.5;
time_end = 1.0;
time_idx = time_vector >= time_start & time_vector <= time_end;

if sum(time_idx) == 0
    error('No time points found in window %.1f-%.1fs', time_start, time_end);
end

fprintf('Found %d time points in window %.1f-%.1fs\n', sum(time_idx), time_start, time_end);
fprintf('Time points: %.3f to %.3f seconds\n', min(time_vector(time_idx)), max(time_vector(time_idx)));

% Extract and average data across the time window
% visual_data_matrix is n_trials x n_timepoints x n_sources
% We want to average across time for each trial and source
averaged_data = squeeze(mean(visual_data_matrix(:, time_idx, :), 2)); % n_trials x n_sources

fprintf('Averaged data shape: %s (trials x sources)\n', mat2str(size(averaged_data)));

% Z-score the data across trials for each source
averaged_data = zscore(averaged_data, 0, 1);
fprintf('Data z-scored across trials\n');

%% Run TAFKAP on Averaged Data
fprintf('Running TAFKAP decoding for averaged data (0.5-1.0s)...\n');

% Set up TAFKAP parameters
p = struct();
p.stimval = target_angles(:); % Stimulus values for each trial (0-360 range)
p.nchan = 8; % Number of channels
p.dec_type = algorithm; % Use specified algorithm
p.Nboot = 5e3; % Set bootstrap iterations
p.DJS_tol = 1e-6; % Set DJS tolerance

% Add PRINCE-specific parameter
if strcmp(algorithm, 'PRINCE')
    p.singletau = true; % PRINCE uses single tau parameter
end

% Create proper cross-validation splits using run-based CV
n_folds = 5;
fold_size = floor(n_trials / n_folds);
run_ind = repmat((1:n_folds)', fold_size, 1);
if length(run_ind) < n_trials
    run_ind = [run_ind; repmat(n_folds, n_trials - length(run_ind), 1)];
end
p.runNs = run_ind;

% Deal with random number generator
rng('shuffle');
nrun = max(run_ind);
fprintf('There are %d runs for cross-validation\n', nrun);
randseed = randi(2^20, nrun, 1); % Generate random seed for each run

% Change to TAFKAP directory so relative paths work
original_dir = pwd;
cd(tafkap_path);

% Run TAFKAP on the specific timepoint
try
    % Initialize parallel pool if not already running
    if isempty(gcp('nocreate'))
        parpool('local', min(8, feature('numcores'))); % Use up to 8 cores
    end
    
    % Initialize results arrays
    est = nan(n_trials, 1);
    unc = nan(n_trials, 1);
    lf = nan(n_trials, 1000); % TAFKAP typically outputs 1000 points for likelihood
    hypers = nan(nrun, 2);
    
    % Pre-allocate results for parfor
    temp_est = nan(n_trials, nrun);
    temp_unc = nan(n_trials, nrun);
    temp_lf = cell(nrun, 1);
    temp_hyper = cell(nrun, 1);
    
    % Use 'parfor' for parallel cross-validation
    parfor testrun_idx = 1:nrun
        % Set up parameters for this run
        thisp = p;
        thisp.test_trials = run_ind == testrun_idx;
        thisp.train_trials = run_ind ~= testrun_idx;
        thisp.randseed = randseed(testrun_idx);
        
        fprintf('Starting testrun #%d of %d\n', testrun_idx, nrun);
        
        [this_est, this_unc, this_lf, this_hypers] = TAFKAP_Decode(averaged_data, thisp);
        fprintf('Completed testrun #%d of %d\n', testrun_idx, nrun);
        
        % Store results (handle variable run sizes)
        test_trials_this_run = run_ind == testrun_idx;
        n_test_trials = sum(test_trials_this_run);
        
        % Create temporary variables for this run
        this_temp_est = nan(n_trials, 1);
        this_temp_unc = nan(n_trials, 1);
        
        % Ensure dimensions match
        if length(this_est) == n_test_trials
            this_temp_est(test_trials_this_run) = this_est(:);
            this_temp_unc(test_trials_this_run) = this_unc(:);
        else
            error('Dimension mismatch in run %d: expected %d test trials, got %d', testrun_idx, n_test_trials, length(this_est));
        end
        
        % Store in pre-allocated arrays
        temp_est(:, testrun_idx) = this_temp_est;
        temp_unc(:, testrun_idx) = this_temp_unc;
        temp_lf{testrun_idx} = this_lf;
        temp_hyper{testrun_idx} = this_hypers;
    end
    
    % Put decoded results into the shape we want
    for testrun_idx = 1:nrun
        test_trials_this_run = run_ind == testrun_idx;
        est(test_trials_this_run) = temp_est(test_trials_this_run, testrun_idx);
        unc(test_trials_this_run) = temp_unc(test_trials_this_run, testrun_idx);
        lf(test_trials_this_run, :) = temp_lf{testrun_idx};
        hypers(testrun_idx, :) = temp_hyper{testrun_idx};
    end
    
    fprintf('TAFKAP decoding completed successfully for averaged data (0.5-1.0s)\n');
catch ME
    cd(original_dir); % Make sure we change back even if there's an error
    error('TAFKAP decoding failed: %s', ME.message);
end

% Change back to original directory
cd(original_dir);

%% Save Results
fprintf('Saving TAFKAP results for averaged data (0.5-1.0s)...\n');

% Set up output directory using environment-appropriate path
if strcmp(algorithm, 'PRINCE')
    output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'prince_decoding');
else
    output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'tafkap_decoding');
end
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Create results structure
results = struct();
results.estimates = est;           % Decoded estimates
results.uncertainty = unc;         % Uncertainty values
results.likelihoods = lf;         % Likelihood functions
results.hyperparameters = hypers; % Hyperparameters
results.target_angles = target_angles; % Target angles
results.time_window = [time_start, time_end]; % Time window used
results.n_trials = n_trials;
results.n_sources = n_sources;
results.subject_id = subjID;
results.volume_resolution = vol_res;
results.algorithm = algorithm;

% Save results
if strcmp(algorithm, 'PRINCE')
    results_path = fullfile(output_dir, sprintf('sub-%02d_prince_results_%dmm_avgdelay.mat', subjID, vol_res));
else
    results_path = fullfile(output_dir, sprintf('sub-%02d_tafkap_results_%dmm_avgdelay.mat', subjID, vol_res));
end
save(results_path, 'results');
fprintf('Results saved to: %s\n', results_path);

%% Compute Decoding Performance
fprintf('Computing decoding performance for averaged data (0.5-1.0s)...\n');

% Calculate decoding error (circular distance)
decoding_error = abs(est - target_angles);
decoding_error = min(decoding_error, 360 - decoding_error); % Handle circularity

% Calculate mean absolute error
mean_error = mean(decoding_error);
std_error = std(decoding_error);

fprintf('Decoding Performance for averaged data (0.5-1.0s):\n');
fprintf('  Mean absolute error: %.2f° ± %.2f°\n', mean_error, std_error);

% Save performance metrics
performance = struct();
performance.decoding_error = decoding_error;
performance.mean_error = mean_error;
performance.std_error = std_error;
performance.time_window = [time_start, time_end];

if strcmp(algorithm, 'PRINCE')
    perf_path = fullfile(output_dir, sprintf('sub-%02d_prince_performance_%dmm_avgdelay.mat', subjID, vol_res));
else
    perf_path = fullfile(output_dir, sprintf('sub-%02d_tafkap_performance_%dmm_avgdelay.mat', subjID, vol_res));
end
save(perf_path, 'performance');
fprintf('Performance metrics saved to: %s\n', perf_path);

%% Create Visualization
fprintf('Creating visualization...\n');

figure('Position', [100, 100, 1200, 800]);

% Subplot 1: Target vs Estimated angles
subplot(2, 2, 1);
scatter(target_angles, est, 50, 'filled', 'MarkerFaceAlpha', 0.6);
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
scatter(target_angles, decoding_error, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Target Angle (degrees)');
ylabel('Decoding Error (degrees)');
title('Decoding Error vs Target Angle');
grid on;

% Subplot 4: Uncertainty vs Target angle
subplot(2, 2, 4);
scatter(target_angles, unc, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Target Angle (degrees)');
ylabel('Uncertainty');
title('Uncertainty vs Target Angle');
grid on;

sgtitle(sprintf('TAFKAP Decoding Results - Subject %02d (0.5-1.0s)', subjID));

% Save figure
fig_path = fullfile(output_dir, sprintf('sub-%02d_tafkap_results_%dmm_avgdelay.fig', subjID, vol_res));
saveas(gcf, fig_path);
fprintf('Figure saved to: %s\n', fig_path);

% Also save as PNG
png_path = fullfile(output_dir, sprintf('sub-%02d_tafkap_results_%dmm_avgdelay.png', subjID, vol_res));
saveas(gcf, png_path);
fprintf('PNG saved to: %s\n', png_path);

fprintf('\nTAFKAP decoding analysis complete for averaged data (0.5-1.0s)!\n');
fprintf('Results saved in: %s\n', output_dir);

end
