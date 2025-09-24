function S05_TAFKAPwPCA_Decoding(subjID, surface_resolution, algorithm, input_time)
%% S05_TAFKAPwPCA_Decoding - Decode stimulus location using TAFKAP or PRINCE with PCA
%
% This script loads complex beta data from S03 and runs TAFKAP or PRINCE decoding
% to decode stimulus location from beta power in posterior sources with PCA dimensionality reduction.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   surface_resolution - Surface resolution (default: 5124)
%   algorithm - 'TAFKAP' or 'PRINCE' (default: 'TAFKAP')
%   input_time - Start time for analysis window (default: 0.8)
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
%   S05_TAFKAPwPCA_Decoding(1, 5124, 'TAFKAP', 0.8)
%   S05_TAFKAPwPCA_Decoding(1, 5124, 'PRINCE', 1.0)
%
% Author: Mrugank Dake
% Date: 2025-01-20

if nargin < 1
    error('Subject ID is required');
end
if nargin < 2
    surface_resolution = 5124; % Default resolution
end
if nargin < 3
    algorithm = 'TAFKAP'; % Default to TAFKAP
end
if nargin < 4
    input_time = 0.8; % Default start time
end

% Validate algorithm choice
if ~ismember(algorithm, {'TAFKAP', 'PRINCE'})
    error('Algorithm must be either ''TAFKAP'' or ''PRINCE''');
end

restoredefaultpath;
clearvars -except subjID surface_resolution algorithm input_time; % Keep inputs
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
if strcmp(algorithm, 'PRINCE')
    output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'prince_decoding');
else
    output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'tafkap_decoding');
end
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
% Time windows for beta power analysis
baseline_start = -0.5; % seconds (before stimulus onset)
baseline_end = 0.0;    % seconds (stimulus onset)
time_start = input_time;      % seconds (analysis window start)
time_end = input_time + 0.2;  % seconds (analysis window end)

fprintf('Extracting beta power data (%.1f-%.1fs window)...\n', time_start, time_end);

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
        
        % Use real and imaginary parts to preserve phase information
        real_data = real(posterior_complex);
        imag_data = imag(posterior_complex);
        
        % Calculate baseline real/imag (average across baseline window)
        baseline_real = mean(real_data(:, baseline_idx), 2);
        baseline_imag = mean(imag_data(:, baseline_idx), 2);
        
        % Calculate analysis window real/imag (average across analysis window)
        analysis_real = mean(real_data(:, time_idx), 2);
        analysis_imag = mean(imag_data(:, time_idx), 2);
        
        % Baseline correct: subtract baseline from analysis window
        avg_real = analysis_real - baseline_real;
        avg_imag = analysis_imag - baseline_imag;
        
        % Combine real and imaginary parts
        avg_complex_features = [avg_real; avg_imag];
        
        % Store data
        all_trials(trial_count, :) = avg_complex_features';
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
fprintf('Using real and imaginary parts (2x sources: %d features)\n', size(all_trials, 2));

% Use original 0-360° angles directly for TAFKAP
fprintf('Using original 0-360° angles for TAFKAP...\n');

fprintf('After shuffling - Target angle distribution:\n');
unique_targets = unique(all_targets);
for i = 1:length(unique_targets)
    n_trials_for_target = sum(all_targets == unique_targets(i));
    fprintf('  Target %.1f°: %d trials\n', unique_targets(i), n_trials_for_target);
end

%% Compute PCA on 0.3-0.5s window for dimensionality reduction
fprintf('\n=== Computing PCA for Dimensionality Reduction ===\n');
fprintf('Computing PCA on 0.3-0.5s window...\n');

% Define PCA window (0.3-0.5s window)
pca_window_start = 0.3;
pca_window_end = 0.5;

% Extract data for PCA computation using the same approach as the main analysis
fprintf('  Extracting data for PCA computation (%.1f-%.1fs)...\n', pca_window_start, pca_window_end);

% Initialize data matrices for PCA
all_trials_pca = [];
all_targets_pca = [];
trial_count_pca = 0;

for target = 1:n_targets
    if isempty(sourceDataByTarget{target})
        continue;
    end
    
    % Get time vector
    time_vec = sourceDataByTarget{target}.time{1};
    
    % Find time indices for baseline and PCA windows
    baseline_idx = time_vec >= baseline_start & time_vec <= baseline_end;
    pca_time_idx = time_vec >= pca_window_start & time_vec <= pca_window_end;
    
    if sum(baseline_idx) == 0
        fprintf('  Warning: No baseline time points found for target %d in window %.1f-%.1fs\n', target, baseline_start, baseline_end);
        continue;
    end
    
    if sum(pca_time_idx) == 0
        fprintf('  Warning: No time points found for target %d in PCA window %.1f-%.1fs\n', target, pca_window_start, pca_window_end);
        continue;
    end
    
    % Extract trials for this target
    n_trials = length(sourceDataByTarget{target}.trial);
    
    for trial = 1:n_trials
        trial_count_pca = trial_count_pca + 1;
        
        % Get complex beta data for this trial
        complex_data = sourceDataByTarget{target}.trial{trial};
        
        % Select posterior sources only
        posterior_complex = complex_data(posterior_sources, :);
        
        % Use real and imaginary parts to preserve phase information
        real_data = real(posterior_complex);
        imag_data = imag(posterior_complex);
        
        % Calculate baseline real/imag (average across baseline window)
        baseline_real = mean(real_data(:, baseline_idx), 2);
        baseline_imag = mean(imag_data(:, baseline_idx), 2);
        
        % Calculate PCA window real/imag (average across PCA window)
        pca_analysis_real = mean(real_data(:, pca_time_idx), 2);
        pca_analysis_imag = mean(imag_data(:, pca_time_idx), 2);
        
        % Baseline correct: subtract baseline from PCA window
        avg_real = pca_analysis_real - baseline_real;
        avg_imag = pca_analysis_imag - baseline_imag;
        
        % Combine real and imaginary parts
        avg_complex_features = [avg_real; avg_imag];
        
        % Store data
        all_trials_pca(trial_count_pca, :) = avg_complex_features';
        all_targets_pca(trial_count_pca) = target_angles(target);
    end
end

if trial_count_pca == 0
    error('No trials found for PCA computation in window %.1f-%.1fs', pca_window_start, pca_window_end);
end

fprintf('  PCA data: %d trials × %d features (real+imag parts)\n', size(all_trials_pca, 1), size(all_trials_pca, 2));

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

%% Apply PCA to the main analysis data
fprintf('\n=== Applying PCA to Main Analysis Data ===\n');
fprintf('Projecting main analysis data into PCA space (%d → %d dimensions)...\n', size(all_trials, 2), n_components);

% Project the main analysis data into PCA space
all_trials_pca_projected = all_trials * pca_coeff;

fprintf('Data projected from %d to %d dimensions (preserving phase information)\n', size(all_trials, 2), n_components);

%% Run TAFKAP Decoding
fprintf('Running TAFKAP decoding...\n');

% Set up TAFKAP parameters
p = struct();
p.stimval = all_targets(:); % Stimulus values for each trial (0-360 range) - ensure column vector
p.nchan = 8; % Number of channels
p.dec_type = algorithm; % Use specified algorithm
p.Nboot = 5e4; % Set bootstrap iterations
p.DJS_tol = 1e-8; % Set DJS tolerance

% Add PRINCE-specific parameter
if strcmp(algorithm, 'PRINCE')
    p.singletau = true; % PRINCE uses single tau parameter
end

% Create proper cross-validation splits using run-based CV
n_folds = 5;
fold_size = floor(trial_count / n_folds);
run_ind = repmat((1:n_folds)', fold_size, 1);
if length(run_ind) < trial_count
    run_ind = [run_ind; repmat(n_folds, trial_count - length(run_ind), 1)];
end
p.runNs = run_ind;

% Initialize results arrays
nrun = max(run_ind);
est = nan(trial_count, 1);
unc = nan(trial_count, 1);
lf = nan(trial_count, 1000); % TAFKAP typically outputs 1000 points for likelihood
hypers = nan(nrun, 2);

% Deal with random number generator
rng('shuffle');
fprintf('There are %d runs for cross-validation\n', nrun);
randseed = randi(2^20, nrun, 1); % Generate random seed for each run

% Verify TAFKAP parameters
fprintf('TAFKAP parameters: %d trials, %d sources (PCA-reduced)\n', size(all_trials_pca_projected, 1), size(all_trials_pca_projected, 2));
fprintf('Data range: %.3f to %.3f (z-scored, PCA-projected)\n', min(all_trials_pca_projected(:)), max(all_trials_pca_projected(:)));
fprintf('Stimulus range: %.1f to %.1f degrees\n', min(p.stimval), max(p.stimval));

% Change to TAFKAP directory so relative paths work
original_dir = pwd;
cd(tafkap_path);

% Run TAFKAP with proper cross-validation
try
    
    % Initialize parallel pool if not already running
    if isempty(gcp('nocreate'))
        parpool('local', min(8, feature('numcores'))); % Use up to 8 cores
    end
    
    % Pre-allocate results for parfor
    temp_est = nan(trial_count, nrun);
    temp_unc = nan(trial_count, nrun);
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
        
        % Debug: Check dimensions before calling TAFKAP
        fprintf('  Data size: %s\n', mat2str(size(all_trials_pca_projected)));
        fprintf('  Test trials: %d\n', sum(thisp.test_trials));
        fprintf('  Train trials: %d\n', sum(thisp.train_trials));
        fprintf('  Stimval length: %d\n', length(thisp.stimval));
        
        [this_est, this_unc, this_lf, this_hypers] = TAFKAP_Decode(all_trials_pca_projected, thisp);
        fprintf('Completed testrun #%d of %d\n', testrun_idx, nrun);
        
        % Store results (handle variable run sizes)
        test_trials_this_run = run_ind == testrun_idx;
        n_test_trials = sum(test_trials_this_run);
        
        % Create temporary variables for this run
        this_temp_est = nan(trial_count, 1);
        this_temp_unc = nan(trial_count, 1);
        
        % Ensure dimensions match
        if length(this_est) == n_test_trials
            this_temp_est(test_trials_this_run) = this_est(:);
            this_temp_unc(test_trials_this_run) = this_unc(:);
        else
            fprintf('  Warning: Dimension mismatch for run %d - expected %d, got %d\n', testrun_idx, n_test_trials, length(this_est));
            % Use only the first n_test_trials or pad with NaN
            if length(this_est) > n_test_trials
                this_temp_est(test_trials_this_run) = this_est(1:n_test_trials);
                this_temp_unc(test_trials_this_run) = this_unc(1:n_test_trials);
            else
                this_temp_est(test_trials_this_run) = [this_est(:); NaN(n_test_trials - length(this_est), 1)];
                this_temp_unc(test_trials_this_run) = [this_unc(:); NaN(n_test_trials - length(this_unc), 1)];
            end
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
    
    fprintf('TAFKAP decoding completed successfully\n');
catch ME
    cd(original_dir); % Make sure we change back even if there's an error
    error('TAFKAP decoding failed: %s', ME.message);
end

% Change back to original directory
cd(original_dir);

% TAFKAP outputs are already in 0-360° range
fprintf('TAFKAP estimates are in 0-360° range...\n');

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
results.pca_components = n_components; % Number of PCA components used
results.pca_variance_explained = sum(explained(1:n_components)); % Variance explained by PCA
results.pca_window = [pca_window_start, pca_window_end]; % PCA computation window
results.feature_type = 'real_imag_parts'; % Type of features used (real+imag instead of power)
results.n_features = size(all_trials, 2); % Total number of features (2x sources)

% Save results
if strcmp(algorithm, 'PRINCE')
    results_path = fullfile(output_dir, sprintf('sub-%02d_prince_pca_results_%d_t%.1f.mat', subjID, surface_resolution, input_time));
else
    results_path = fullfile(output_dir, sprintf('sub-%02d_tafkap_pca_results_%d_t%.1f.mat', subjID, surface_resolution, input_time));
end
save(results_path, 'results');
fprintf('Results saved to: %s\n', results_path);

%% Use TAFKAP Results
fprintf('Using TAFKAP results...\n');

% TAFKAP outputs are in 0-360° range
est_original = est; % 0-360°

% Target angles are in 0-360° space
targets_original = all_targets; % 0-360°

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

fprintf('Decoding Performance:\n');
fprintf('  Mean absolute error: %.2f° ± %.2f°\n', mean_error, std_error);

% Save performance metrics
performance = struct();
performance.mean_error = mean_error;
performance.std_error = std_error;
performance.decoding_error = decoding_error;

if strcmp(algorithm, 'PRINCE')
    perf_path = fullfile(output_dir, sprintf('sub-%02d_prince_pca_performance_%d_t%.1f.mat', subjID, surface_resolution, input_time));
else
    perf_path = fullfile(output_dir, sprintf('sub-%02d_tafkap_pca_performance_%d_t%.1f.mat', subjID, surface_resolution, input_time));
end
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


sgtitle(sprintf('TAFKAP Decoding Results (PCA-Reduced) - Subject %02d', subjID));

% Save figure
fig_path = fullfile(output_dir, sprintf('sub-%02d_tafkap_pca_results_%d_t%.1f.fig', subjID, surface_resolution, input_time));
saveas(gcf, fig_path);
fprintf('Figure saved to: %s\n', fig_path);

% Also save as PNG
png_path = fullfile(output_dir, sprintf('sub-%02d_tafkap_pca_results_%d_t%.1f.png', subjID, surface_resolution, input_time));
saveas(gcf, png_path);
fprintf('PNG saved to: %s\n', png_path);

fprintf('\nTAFKAP decoding analysis with PCA complete!\n');
fprintf('PCA: %d components (%.1f%% variance explained)\n', n_components, sum(explained(1:n_components)));
fprintf('Results saved in: %s\n', output_dir);

end
