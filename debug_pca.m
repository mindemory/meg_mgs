% Debug script to test PCA SVM decoding
clear; close all; clc;

% Load data
subjID = 1;
surface_resolution = 5124;
input_dir = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-01/sourceRecon';
beta_data_path = fullfile(input_dir, sprintf('sub-%02d_task-mgs_complexBeta_allTargets_%d.mat', subjID, surface_resolution));
load(beta_data_path);

% Set up basic parameters
target_angles = [0, 25, 50, 130, 155, 180, 205, 230, 310, 335];
n_targets = length(target_angles);
posterior_sources = 1:5124; % All sources

% Compute global baseline
fprintf('Computing global baseline...\n');
all_baseline_data = [];
baseline_trial_count = 0;
baseline_start = -0.5;
baseline_end = 0.0;

for target = 1:n_targets
    if isempty(sourceDataByTarget{target})
        continue;
    end
    time_vec = sourceDataByTarget{target}.time{1};
    baseline_idx = time_vec >= baseline_start & time_vec <= baseline_end;
    if sum(baseline_idx) == 0
        continue;
    end
    n_trials = length(sourceDataByTarget{target}.trial);
    for trial = 1:n_trials
        baseline_trial_count = baseline_trial_count + 1;
        complex_data = sourceDataByTarget{target}.trial{trial};
        posterior_complex = complex_data(posterior_sources, :);
        power_data = abs(posterior_complex).^2;
        baseline_power = mean(power_data(:, baseline_idx), 2);
        all_baseline_data(baseline_trial_count, :) = baseline_power';
    end
end

global_baseline_mean = mean(all_baseline_data, 1);
fprintf('Baseline computed from %d trials\n', baseline_trial_count);

% Extract data for PCA (0.3-0.5s window)
fprintf('Extracting PCA data...\n');
all_trials_pca = [];
all_targets_pca = [];
trial_count = 0;
pca_window_start = 0.3;
pca_window_end = 0.5;

for target = 1:n_targets
    if isempty(sourceDataByTarget{target})
        continue;
    end
    time_vec = sourceDataByTarget{target}.time{1};
    time_idx = time_vec >= pca_window_start & time_vec <= pca_window_end;
    if sum(time_idx) == 0
        continue;
    end
    n_trials = length(sourceDataByTarget{target}.trial);
    for trial = 1:n_trials
        trial_count = trial_count + 1;
        complex_data = sourceDataByTarget{target}.trial{trial};
        posterior_complex = complex_data(posterior_sources, :);
        power_data = abs(posterior_complex).^2;
        analysis_power = mean(power_data(:, time_idx), 2);
        avg_power = (analysis_power - global_baseline_mean') ./ (global_baseline_mean' + eps);
        all_trials_pca(trial_count, :) = avg_power';
        all_targets_pca(trial_count) = target_angles(target);
    end
end

fprintf('PCA data: %d trials × %d sources\n', size(all_trials_pca, 1), size(all_trials_pca, 2));

% Apply z-score and compute PCA
all_trials_pca_normalized = zscore(all_trials_pca, 0, 1);
[coeff, score, latent, ~, explained] = pca(all_trials_pca_normalized);
n_components = 50;
pca_coeff = coeff(:, 1:n_components);
fprintf('PCA computed: %d components (%.1f%% variance)\n', n_components, sum(explained(1:n_components)));

% Test SVM on PCA data
fprintf('Testing SVM on PCA data...\n');
all_trials_pca_projected = all_trials_pca_normalized * pca_coeff;
fprintf('Projected data: %d trials × %d components\n', size(all_trials_pca_projected, 1), size(all_trials_pca_projected, 2));

% Convert angles
angle_rad = (pi/180) * all_targets_pca;
target_sin = sin(angle_rad);
target_cos = cos(angle_rad);

% Add intercept
data_with_intercept = [all_trials_pca_projected, ones(size(all_trials_pca_projected, 1), 1)];
fprintf('Data with intercept: %d trials × %d features\n', size(data_with_intercept, 1), size(data_with_intercept, 2));

% Test fitrlinear
try
    sin_model = fitrlinear(data_with_intercept, target_sin, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 1e-3);
    fprintf('fitrlinear for sine works!\n');
catch ME
    fprintf('Error in sine model: %s\n', ME.message);
end

try
    cos_model = fitrlinear(data_with_intercept, target_cos, 'Learner', 'svm', 'Regularization', 'ridge', 'Lambda', 1e-3);
    fprintf('fitrlinear for cosine works!\n');
catch ME
    fprintf('Error in cosine model: %s\n', ME.message);
end

fprintf('Debug complete!\n');
