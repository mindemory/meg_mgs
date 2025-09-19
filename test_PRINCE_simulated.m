function test_PRINCE_simulated()
%% Test PRINCE with simulated data
%
% This script creates simulated data and tests PRINCE decoding
% (the original algorithm before TAFKAP) to avoid MEX function issues.
%
% Author: Mrugank Dake
% Date: 2025-01-20

restoredefaultpath;
clearvars;
close all; clc;

%% Add TAFKAP path
tafkap_path = '/d/DATD/hyper/experiments/Mrugank/wmJointRepresentation/';
addpath(genpath(tafkap_path));

fprintf('=== Testing PRINCE with Simulated Data ===\n');

%% Create Simulated Data
fprintf('Creating simulated data...\n');

% Parameters
n_trials = 100;
n_sources = 200;
n_targets = 10;

% Create target angles (0-360°)
target_angles = [0, 25, 50, 130, 155, 180, 205, 230, 310, 335];
fprintf('Target angles: %s\n', mat2str(target_angles));

% Create trial labels (which target each trial belongs to)
trial_targets = randi(n_targets, n_trials, 1);
trial_angles = target_angles(trial_targets)';

fprintf('Number of trials: %d\n', n_trials);
fprintf('Number of sources: %d\n', n_sources);
fprintf('Trial targets: %s\n', mat2str(trial_targets(1:10)));

% Create simulated source data
% Each source has a tuning function around one of the target angles
sources_data = zeros(n_trials, n_sources);

for trial = 1:n_trials
    for source = 1:n_sources
        % Each source is tuned to a random target angle
        source_preferred = target_angles(randi(n_targets));
        
        % Calculate angular distance
        angle_diff = abs(trial_angles(trial) - source_preferred);
        angle_diff = min(angle_diff, 360 - angle_diff); % Handle circularity
        
        % Create tuning function (cosine with some noise)
        tuning_strength = cos(deg2rad(angle_diff))^2;
        noise = 0.3 * randn();
        
        sources_data(trial, source) = tuning_strength + noise;
    end
end

fprintf('Simulated data created: %d trials × %d sources\n', size(sources_data, 1), size(sources_data, 2));

%% Set up PRINCE Parameters
fprintf('Setting up PRINCE parameters...\n');

p = struct();
p.stimval = trial_angles; % Stimulus values for each trial (0-360°)
p.runNs = repmat([1; 2; 3; 4; 5], ceil(n_trials/5), 1); % Create multiple runs
p.runNs = p.runNs(1:n_trials); % Trim to exact number of trials
p.train_trials = true(n_trials, 1); % All trials for training
p.test_trials = true(n_trials, 1);  % All trials for testing

% Use PRINCE instead of TAFKAP
p.algorithm = 'PRINCE';
p.Nboot = 200; % Set bootstrap iterations to 200

% Debug: Check dimensions
fprintf('PRINCE parameters:\n');
fprintf('  stimval length: %d\n', length(p.stimval));
fprintf('  runNs: %s\n', mat2str(unique(p.runNs)));
fprintf('  train_trials length: %d\n', length(p.train_trials));
fprintf('  test_trials length: %d\n', length(p.test_trials));
fprintf('  sources_data size: %s\n', mat2str(size(sources_data)));

%% Run PRINCE Decoding
fprintf('Running PRINCE decoding...\n');

try
    [est, unc, lf, hypers] = TAFKAP_Decode(sources_data, p);
    fprintf('PRINCE decoding completed successfully!\n');
catch ME
    fprintf('PRINCE decoding failed: %s\n', ME.message);
    fprintf('Error location: %s\n', ME.stack(1).name);
    fprintf('Line: %d\n', ME.stack(1).line);
    return;
end

%% Analyze Results
fprintf('Analyzing results...\n');

% Calculate decoding error (circular distance)
decoding_error = abs(est - trial_angles);
decoding_error = min(decoding_error, 360 - decoding_error); % Handle circularity

% Calculate performance metrics
mean_error = mean(decoding_error);
std_error = std(decoding_error);
acc_5deg = mean(decoding_error <= 5);
acc_10deg = mean(decoding_error <= 10);
acc_20deg = mean(decoding_error <= 20);

fprintf('\n=== PRINCE Results ===\n');
fprintf('Mean absolute error: %.2f° ± %.2f°\n', mean_error, std_error);
fprintf('Accuracy within 5°: %.1f%%\n', 100*acc_5deg);
fprintf('Accuracy within 10°: %.1f%%\n', 100*acc_10deg);
fprintf('Accuracy within 20°: %.1f%%\n', 100*acc_20deg);

%% Create Visualization
fprintf('Creating visualization...\n');

figure('Position', [100, 100, 1200, 800]);

% Subplot 1: Target vs Estimated angles
subplot(2, 2, 1);
scatter(trial_angles, est, 50, 'filled', 'MarkerFaceAlpha', 0.6);
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
scatter(trial_angles, decoding_error, 50, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('Target Angle (degrees)');
ylabel('Decoding Error (degrees)');
title('Decoding Error vs Target Angle');
grid on;

% Subplot 4: Sample likelihood functions
subplot(2, 2, 4);
sample_trials = 1:min(5, n_trials);
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

sgtitle('PRINCE Decoding Results (Simulated Data)');

%% Print Sample Results
fprintf('\n=== Sample Results ===\n');
fprintf('Trial | Target | Estimate | Error\n');
fprintf('------|--------|----------|------\n');
for i = 1:min(10, n_trials)
    fprintf('%5d | %6.1f | %8.1f | %5.1f\n', i, trial_angles(i), est(i), decoding_error(i));
end

fprintf('\nPRINCE test completed successfully!\n');

end
