%% MEG Beta Power Analysis in Source Space
% Load source space data and compute beta power with lateralization index
restoredefaultpath;
clear; close all; clc;

%% Setup and Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);

% Configuration parameters
subjID = 1; % Change this to the subject you want to analyze
surface_resolution = 5124; % Should match the resolution used in S02

fprintf('=== MEG Beta Power Analysis in Source Space ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Load Source Space Data
% Load the source space data created by S02_ReverseModelMNI.m
source_data_path = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon/sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, subjID, surface_resolution);

if ~exist(source_data_path, 'file')
    error('Source space data not found at: %s\nPlease run S02_ReverseModelMNI.m first!', source_data_path);
end

fprintf('Loading source space data from: %s\n', source_data_path);
load(source_data_path);

fprintf('Loaded source space data:\n');
fprintf('  Total trials: %d\n', length(sourcedataCombined.trial));
fprintf('  Inside sources: %d\n', length(inside_pos));

%% Select Left and Right Trials from Combined Data
fprintf('Selecting left and right trials from combined data...\n');

% Left trials (targets 4,5,6,7,8)
trial_criteria_left = (sourcedataCombined.trialinfo(:,2) == 4) | ...
                     (sourcedataCombined.trialinfo(:,2) == 5) | ...
                     (sourcedataCombined.trialinfo(:,2) == 6) | ...
                     (sourcedataCombined.trialinfo(:,2) == 7) | ...
                     (sourcedataCombined.trialinfo(:,2) == 8);

% Right trials (targets 1,2,3,9,10)
trial_criteria_right = (sourcedataCombined.trialinfo(:,2) == 1) | ...
                      (sourcedataCombined.trialinfo(:,2) == 2) | ...
                      (sourcedataCombined.trialinfo(:,2) == 3) | ...
                      (sourcedataCombined.trialinfo(:,2) == 9) | ...
                      (sourcedataCombined.trialinfo(:,2) == 10);

% Find valid trials
valid_trialsLeft = find(trial_criteria_left);
valid_trialsRight = find(trial_criteria_right);

fprintf('Valid left trials: %d\n', length(valid_trialsLeft));
fprintf('Valid right trials: %d\n', length(valid_trialsRight));

% Select left and right trials
cfg = [];
cfg.trials = valid_trialsLeft;
sourcedataLeft = ft_selectdata(cfg, sourcedataCombined);

cfg = [];
cfg.trials = valid_trialsRight;
sourcedataRight = ft_selectdata(cfg, sourcedataCombined);

fprintf('Selected trials:\n');
fprintf('  Left trials: %d\n', length(sourcedataLeft.trial));
fprintf('  Right trials: %d\n', length(sourcedataRight.trial));

%% Load Forward Model for Visualization
% Load the forward model for visualization
forward_model_path = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon/sub-%02d_task-mgs_forwardModel.mat', subjID, subjID);
load(forward_model_path);

% Select the appropriate surface model
switch surface_resolution
    case 5124
        sourcemodel = sourcemodel_aligned_5124;
    case 8196
        sourcemodel = sourcemodel_aligned_8196;
    case 20484
        sourcemodel = sourcemodel_aligned_20484;
    otherwise
        error('Unsupported surface resolution: %d', surface_resolution);
end

%% Beta Band Filtering and Hilbert Transform
fprintf('Applying beta band filter (18-27Hz)...\n');

% Band-pass filter for beta band (18-27Hz)
cfg = [];
cfg.bpfilter = 'yes';
cfg.bpfreq = [18 27];
sourcedataLeft_beta = ft_preprocessing(cfg, sourcedataLeft);
sourcedataRight_beta = ft_preprocessing(cfg, sourcedataRight);

fprintf('Computing Hilbert transform...\n');

% Apply Hilbert transform to get analytic signal
hilbert_compute = @(x) hilbert(x')';
sourcedataLeft_beta.trial = cellfun(hilbert_compute, sourcedataLeft_beta.trial, 'UniformOutput', false);
sourcedataRight_beta.trial = cellfun(hilbert_compute, sourcedataRight_beta.trial, 'UniformOutput', false);

fprintf('Beta band processing complete\n');

%% Compute Beta Power
fprintf('Computing beta power...\n');

% Compute power (magnitude of analytic signal)
sourceLeftPow = sourcedataLeft_beta;
sourceLeftPow.trial = cellfun(@(x) abs(x), sourceLeftPow.trial, 'UniformOutput', false);

sourceRightPow = sourcedataRight_beta;
sourceRightPow.trial = cellfun(@(x) abs(x), sourceRightPow.trial, 'UniformOutput', false);

%% Average Across Trials
fprintf('Averaging across trials...\n');

% Average across trials
cfg = [];
sourceDataLeft_avg = ft_timelockanalysis(cfg, sourceLeftPow);
sourceDataRight_avg = ft_timelockanalysis(cfg, sourceRightPow);

%% Compute Lateralization Index
fprintf('Computing lateralization index...\n');

% Compute lateralization index: (Left - Right) / (Left + Right)
cfg = [];
cfg.parameter = 'avg';
cfg.operation = '(x1-x2)/(x1+x2)';
sourceDiff = ft_math(cfg, sourceDataLeft_avg, sourceDataRight_avg);

%% Define Time Window of Interest
% Define time window for analysis (e.g., 0.8-1.5s after stimulus)
time_window = [0.8, 1.5];
TOI = find(sourceDiff.time >= time_window(1) & sourceDiff.time <= time_window(2));

fprintf('Time window of interest: %.1f-%.1f s (%d time points)\n', ...
        time_window(1), time_window(2), length(TOI));

%% Create Source Visualization Structure
fprintf('Creating source visualization structure...\n');

% Create source visualization structure
sourceVisualize = source;
sourceVisualize.lateralizedPow = NaN(size(source.inside));

% Compute mean lateralization in time window
mean_lateralization = squeeze(mean(sourceDiff.avg(:, TOI), 2, 'omitnan'));
sourceVisualize.lateralizedPow(source.inside) = mean_lateralization;

%% Interpolate to MRI for Visualization
fprintf('Interpolating to MRI for visualization...\n');

% Load the aligned MRI for interpolation
cfg = [];
cfg.parameter = {'lateralizedPow'};
interp = ft_sourceinterpolate(cfg, sourceVisualize, mri_aligned);

%% Time Window Visualization (500ms windows from -0.5 to 1.5s)
fprintf('Creating time window visualizations...\n');

% Define time windows (500ms each, overlapping)
time_start = -0.5;
time_end = 1.5;
window_duration = 0.5; % 500ms
window_step = 0.1; % 100ms step (overlapping windows)

time_windows = [];
window_centers = [];
for t = time_start:window_step:(time_end - window_duration)
    time_windows = [time_windows; t, t + window_duration];
    window_centers = [window_centers; t + window_duration/2];
end

fprintf('Created %d time windows of %.1fs duration\n', size(time_windows, 1), window_duration);

% Create figure for time window visualization
figure('Name', 'Beta Power Lateralization - Time Windows', 'Position', [100, 100, 1600, 1200]);

% Calculate number of subplots needed (aim for roughly square layout)
n_windows = size(time_windows, 1);
n_cols = ceil(sqrt(n_windows));
n_rows = ceil(n_windows / n_cols);

for i = 1:n_windows
    subplot(n_rows, n_cols, i);
    
    % Get time indices for this window
    window_start_idx = dsearchn(sourceDiff.time', time_windows(i, 1));
    window_end_idx = dsearchn(sourceDiff.time', time_windows(i, 2));
    
    % Calculate mean lateralization for this time window using sourceDiff.avg
    window_lateralization = mean(sourceDiff.avg(inside_pos, window_start_idx:window_end_idx), 2, 'omitnan');
    
    % Create 3D scatter plot for this time window
    scatter3(sourcemodel.pos(inside_pos, 1), sourcemodel.pos(inside_pos, 2), sourcemodel.pos(inside_pos, 3), ...
             15, window_lateralization, 'filled');
    
    % Set colormap and limits
    colormap("turbo");
    caxis([-0.05, 0.05]);
    
    % Formatting
    title(sprintf('%.1f-%.1fs', time_windows(i, 1), time_windows(i, 2)));
    xlabel('X (mm)');
    ylabel('Y (mm)');
    zlabel('Z (mm)');
    axis equal;
    grid on;
    
    % Set consistent view angle - top view with posterior cortex visible
    view([0, 40]); % Top view with slight downward tilt to better see posterior
    
    % Add colorbar for first subplot only
    if i == 1
        colorbar;
    end
end

% Add overall title
sgtitle(sprintf('Subject %02d: Beta Power Lateralization Across Time Windows', subjID));

fprintf('Time window visualization complete!\n');

%% Save Results
fprintf('Saving results...\n');

output_dir = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon', subjID);
beta_power_path = fullfile(output_dir, sprintf('sub-%02d_task-mgs_betaPower_%d.mat', subjID, surface_resolution));

save(beta_power_path, 'sourceVisualize', 'sourceDiff', 'sourceDataLeft_avg', 'sourceDataRight_avg', ...
     'interp', 'time_window', 'TOI', 'subjID', 'surface_resolution', '-v7.3');

fprintf('Results saved to: %s\n', beta_power_path);

%% Visualization
fprintf('Creating visualizations...\n');

% Create comprehensive visualization
figure('Name', 'Beta Power Analysis Results', 'Position', [100, 100, 1600, 1000]);

% Subplot 1: Time series of lateralization
subplot(2, 3, 1);
plot(sourceDiff.time, mean(sourceDiff.avg, 1, 'omitnan'), 'k-', 'LineWidth', 2);
hold on;
xline(time_window(1), 'r--', 'LineWidth', 2);
xline(time_window(2), 'r--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Mean Lateralization Index');
title('Lateralization Over Time');
grid on;
legend('Lateralization', 'Analysis Window', 'Location', 'best');

% Subplot 2: Distribution of lateralization values
subplot(2, 3, 2);
histogram(mean_lateralization, 50, 'FaceColor', 'blue', 'FaceAlpha', 0.7);
xlabel('Lateralization Index');
ylabel('Number of Sources');
title('Distribution of Lateralization Values');
grid on;

% Subplot 3: Left vs Right power comparison
subplot(2, 3, 3);
left_power = squeeze(mean(sourceDataLeft_avg.avg(:, TOI), 2, 'omitnan'));
right_power = squeeze(mean(sourceDataRight_avg.avg(:, TOI), 2, 'omitnan'));
scatter(left_power, right_power, 20, 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min([left_power; right_power]), max([left_power; right_power])], ...
     [min([left_power; right_power]), max([left_power; right_power])], 'r--', 'LineWidth', 2);
xlabel('Left Power');
ylabel('Right Power');
title('Left vs Right Beta Power');
grid on;
axis equal;

% Subplot 4: Source space lateralization (3D scatter)
subplot(2, 3, 4);
scatter3(sourcemodel.pos(inside_pos, 1), sourcemodel.pos(inside_pos, 2), sourcemodel.pos(inside_pos, 3), ...
         20, mean_lateralization, 'filled');
colorbar;
colormap('RdBu');
caxis([-0.1, 0.1]);
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
title('Source Space Lateralization');
axis equal;
grid on;

% Subplot 5: MRI visualization (orthogonal slices)
subplot(2, 3, [5, 6]);
cfg = [];
cfg.method = 'ortho';
cfg.crosshair = 'yes';
cfg.funparameter = 'lateralizedPow';
cfg.funcolormap = '*RdBu';
cfg.funcolorlim = [-0.1, 0.1];
ft_sourceplot(cfg, interp);
title('Lateralization in MRI Space');

%% Summary Statistics
fprintf('\n=== SUMMARY STATISTICS ===\n');
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);
fprintf('Time window: %.1f-%.1f s\n', time_window(1), time_window(2));
fprintf('Number of sources: %d\n', length(inside_pos));
fprintf('Mean lateralization: %.4f\n', mean(mean_lateralization, 'omitnan'));
fprintf('Std lateralization: %.4f\n', std(mean_lateralization, 'omitnan'));
fprintf('Min lateralization: %.4f\n', min(mean_lateralization, [], 'omitnan'));
fprintf('Max lateralization: %.4f\n', max(mean_lateralization, [], 'omitnan'));

% Count sources with significant lateralization
left_lateralized = sum(mean_lateralization > 0.05);
right_lateralized = sum(mean_lateralization < -0.05);
fprintf('Left-lateralized sources (>0.05): %d\n', left_lateralized);
fprintf('Right-lateralized sources (<-0.05): %d\n', right_lateralized);

%% Additional Analysis: ROI-based Analysis
fprintf('\n=== ROI-BASED ANALYSIS ===\n');

% Define some basic ROIs based on anatomical coordinates
% Left hemisphere (negative X)
left_hemisphere = sourcemodel.pos(inside_pos, 1) < 0;
% Right hemisphere (positive X)
right_hemisphere = sourcemodel.pos(inside_pos, 1) > 0;
% Posterior (negative Y)
posterior = sourcemodel.pos(inside_pos, 2) < 0;
% Anterior (positive Y)
anterior = sourcemodel.pos(inside_pos, 2) > 0;

% Compute mean lateralization for each ROI
left_hem_lat = mean(mean_lateralization(left_hemisphere), 'omitnan');
right_hem_lat = mean(mean_lateralization(right_hemisphere), 'omitnan');
posterior_lat = mean(mean_lateralization(posterior), 'omitnan');
anterior_lat = mean(mean_lateralization(anterior), 'omitnan');

fprintf('Left hemisphere lateralization: %.4f\n', left_hem_lat);
fprintf('Right hemisphere lateralization: %.4f\n', right_hem_lat);
fprintf('Posterior lateralization: %.4f\n', posterior_lat);
fprintf('Anterior lateralization: %.4f\n', anterior_lat);

%% Save Summary
summary_path = fullfile(output_dir, sprintf('sub-%02d_task-mgs_betaPower_summary_%d.txt', subjID, surface_resolution));
fid = fopen(summary_path, 'w');
fprintf(fid, 'Beta Power Analysis Summary\n');
fprintf(fid, '==========================\n');
fprintf(fid, 'Subject: %d\n', subjID);
fprintf(fid, 'Surface resolution: %d vertices\n', surface_resolution);
fprintf(fid, 'Time window: %.1f-%.1f s\n', time_window(1), time_window(2));
fprintf(fid, 'Number of sources: %d\n', length(inside_pos));
fprintf(fid, 'Mean lateralization: %.4f\n', mean(mean_lateralization, 'omitnan'));
fprintf(fid, 'Std lateralization: %.4f\n', std(mean_lateralization, 'omitnan'));
fprintf(fid, 'Min lateralization: %.4f\n', min(mean_lateralization, [], 'omitnan'));
fprintf(fid, 'Max lateralization: %.4f\n', max(mean_lateralization, [], 'omitnan'));
fprintf(fid, 'Left-lateralized sources (>0.05): %d\n', left_lateralized);
fprintf(fid, 'Right-lateralized sources (<-0.05): %d\n', right_lateralized);
fprintf(fid, 'Left hemisphere lateralization: %.4f\n', left_hem_lat);
fprintf(fid, 'Right hemisphere lateralization: %.4f\n', right_hem_lat);
fprintf(fid, 'Posterior lateralization: %.4f\n', posterior_lat);
fprintf(fid, 'Anterior lateralization: %.4f\n', anterior_lat);
fclose(fid);

fprintf('Summary saved to: %s\n', summary_path);
fprintf('\nBeta power analysis complete!\n');
