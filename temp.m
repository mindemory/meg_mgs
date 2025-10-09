% function temp()
%% Temporary script to load complex beta data for subject 01 in 10mm space
% Load complex beta data and concatenate across all target locations

restoredefaultpath;
clearvars;
close all; clc;

%% Setup paths
% Local machine paths
fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';

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

%% Setup and Initialization
addpath(fieldtrip_path);
addpath(genpath(project_path));
ft_defaults;
ft_hastoolbox('spm12', 1);

%% Load Volumetric Sources Data
subjID = 1;

% Load volumetric sources data
volumetric_path = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-01/sourceRecon/sub-01_task-mgs_volumetricSources_10mm.mat';
fprintf('Loading volumetric sources data from: %s\n', volumetric_path);
load(volumetric_path);

% Load complex beta data
beta_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'freqspace', ...
    sprintf('sub-%02d_task-mgs_complexbeta_allTargets_10.mat', subjID));

if ~exist(beta_data_path, 'file')
    error('Complex beta signal data not found at: %s\nPlease run S03_betaPowerInSource.m first!', beta_data_path);
end

fprintf('Loading complex beta signal data from: %s\n', beta_data_path);
load(beta_data_path, 'sourceDataByTarget');

%% Concatenate Data Across All Target Locations
fprintf('\n=== Concatenating Data Across All Target Locations ===\n');

% Initialize combined structure
allTrials = [];
allTrialInfo = [];
trialCount = 0;

% Combine trials from all targets
for target = 1:length(sourceDataByTarget)
    if ~isempty(sourceDataByTarget{target})
        fprintf('Adding %d trials from target %d\n', length(sourceDataByTarget{target}.trial), target);
        
        % Convert complex beta to power (magnitude squared)
        sourceDataByTarget{target}.trial = cellfun(@(x) abs(x).^2, sourceDataByTarget{target}.trial, 'UniformOutput', false);
        
        % Add trials to combined structure
        if isempty(allTrials)
            % First target - initialize structure (keep label, fsample, time from first)
            allTrials = sourceDataByTarget{target};
            allTrialInfo = [sourceDataByTarget{target}.trialinfo, target * ones(size(sourceDataByTarget{target}.trialinfo, 1), 1)];
        else
            % Subsequent targets - append trials and concatenate time
            allTrials.trial = [allTrials.trial, sourceDataByTarget{target}.trial];
            allTrials.trialinfo = [allTrials.trialinfo; sourceDataByTarget{target}.trialinfo];
            allTrials.time = [allTrials.time, sourceDataByTarget{target}.time];
            allTrialInfo = [allTrialInfo; [sourceDataByTarget{target}.trialinfo, target * ones(size(sourceDataByTarget{target}.trialinfo, 1), 1)]];
        end
        
        trialCount = trialCount + length(sourceDataByTarget{target}.trial);
    end
end

% Update trialinfo with target information (add target column)
allTrials.trialinfo = allTrialInfo;

fprintf('Combined %d total trials from all targets\n', trialCount);
fprintf('Trial info structure: [original_columns, target_location]\n');

%% Compute Baseline-Corrected Power
fprintf('Computing baseline-corrected power...\n');

% Define baseline period (-0.5 to 0 seconds)
baseline_start = -0.5;
baseline_end = 0.0;
baseline_idx = allTrials.time{1} >= baseline_start & allTrials.time{1} <= baseline_end;

% Define analysis period (0 to 0.5 seconds)
analysis_start = 0.0;
analysis_end = 0.5;
analysis_idx = allTrials.time{1} >= analysis_start & allTrials.time{1} <= analysis_end;

fprintf('Baseline period: %.1f to %.1f s (%d timepoints)\n', baseline_start, baseline_end, sum(baseline_idx));
fprintf('Analysis period: %.1f to %.1f s (%d timepoints)\n', analysis_start, analysis_end, sum(analysis_idx));

% Compute baseline and analysis power for each source
nSources = size(allTrials.trial{1}, 1);
baseline_power = zeros(nSources, 1);
analysis_power = zeros(nSources, 1);

for source = 1:nSources
    % Compute baseline power
    baseline_data = [];
    for trial = 1:length(allTrials.trial)
        baseline_data = [baseline_data, allTrials.trial{trial}(source, baseline_idx)];
    end
    baseline_power(source) = mean(baseline_data);
    
    % Compute analysis power
    analysis_data = [];
    for trial = 1:length(allTrials.trial)
        analysis_data = [analysis_data, allTrials.trial{trial}(source, analysis_idx)];
    end
    analysis_power(source) = mean(analysis_data);
end

% Compute baseline-corrected power (divide by baseline)
baseline_corrected_power = analysis_power ./ baseline_power - 1;

fprintf('Baseline power computed for %d sources\n', nSources);
fprintf('Baseline power range: %.6f to %.6f\n', min(baseline_power), max(baseline_power));
fprintf('Analysis power range: %.6f to %.6f\n', min(analysis_power), max(analysis_power));
fprintf('Baseline-corrected power range: %.6f to %.6f\n', min(baseline_corrected_power), max(baseline_corrected_power));

%% Create Baseline-Corrected Power Scatter Plot
figure;
scatter3(sourcemodel.pos(sourcemodel.inside, 1), sourcemodel.pos(sourcemodel.inside, 2), sourcemodel.pos(sourcemodel.inside, 3), ...
    30, baseline_corrected_power, 'filled');
colorbar;
title(sprintf('Baseline-Corrected Power (%.1f-%.1fs / %.1f-%.1fs)', analysis_start, analysis_end, baseline_start, baseline_end));
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
axis equal;
grid on;

fprintf('Baseline-corrected scatter plot complete!\n');

%% Divide Trials into Left and Right Groups
fprintf('\n=== Dividing Trials into Left and Right Groups ===\n');

% Define left and right targets (same as S05_DecodeBeta)
leftTargets = [4, 5, 6, 7, 8];
rightTargets = [1, 2, 3, 9, 10];

% Get target labels (last column of trialinfo)
targetLocations = allTrials.trialinfo(:, end);

% Create binary labels: 1 = left, 2 = right
targetLabels = zeros(length(targetLocations), 1);
targetLabels(ismember(targetLocations, leftTargets)) = 1;  % Left
targetLabels(ismember(targetLocations, rightTargets)) = 2; % Right

fprintf('Left targets: %s\n', mat2str(leftTargets));
fprintf('Right targets: %s\n', mat2str(rightTargets));
fprintf('Left trials: %d, Right trials: %d\n', sum(targetLabels == 1), sum(targetLabels == 2));

%% Compute Power for Left and Right Groups Separately
fprintf('Computing power for left and right groups...\n');

% Find time indices for 0 to 0.5 seconds (same as before)
analysis_start = 1.0;
analysis_end = 1.5;
analysis_idx = allTrials.time{1} >= analysis_start & allTrials.time{1} <= analysis_end;

% Compute power for left trials
left_trials = targetLabels == 1;
left_power = zeros(nSources, 1);
for source = 1:nSources
    source_power_data = [];
    for trial = find(left_trials)'
        source_power_data = [source_power_data, allTrials.trial{trial}(source, analysis_idx)];
    end
    left_power(source) = mean(source_power_data);
end

% Compute power for right trials
right_trials = targetLabels == 2;
right_power = zeros(nSources, 1);
for source = 1:nSources
    source_power_data = [];
    for trial = find(right_trials)'
        source_power_data = [source_power_data, allTrials.trial{trial}(source, analysis_idx)];
    end
    right_power(source) = mean(source_power_data);
end

% Apply baseline correction to both groups
left_baseline_corrected = left_power ./ baseline_power - 1;
right_baseline_corrected = right_power ./ baseline_power - 1;

fprintf('Left group power range: %.6f to %.6f\n', min(left_baseline_corrected), max(left_baseline_corrected));
fprintf('Right group power range: %.6f to %.6f\n', min(right_baseline_corrected), max(right_baseline_corrected));

%% Create Side-by-Side Scatter Plots
figure('Position', [100, 100, 1200, 500]);

% Left trials scatter plot
subplot(1, 2, 1);
scatter3(sourcemodel.pos(sourcemodel.inside, 1), sourcemodel.pos(sourcemodel.inside, 2), sourcemodel.pos(sourcemodel.inside, 3), ...
    30, left_baseline_corrected, 'filled');
colorbar;
caxis([-0.7, 0.7]);
title(sprintf('Left Trials (Targets %s)', mat2str(leftTargets)));
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
axis equal;
grid on;

% Right trials scatter plot
subplot(1, 2, 2);
scatter3(sourcemodel.pos(sourcemodel.inside, 1), sourcemodel.pos(sourcemodel.inside, 2), sourcemodel.pos(sourcemodel.inside, 3), ...
    30, right_baseline_corrected, 'filled');
colorbar;
caxis([-0.7, 0.7]);
title(sprintf('Right Trials (Targets %s)', mat2str(rightTargets)));
xlabel('X (mm)');
ylabel('Y (mm)');
zlabel('Z (mm)');
axis equal;
grid on;

sgtitle(sprintf('Baseline-Corrected Power: Left vs Right Trials (%.1f-%.1fs)', analysis_start, analysis_end));

fprintf('Side-by-side scatter plots complete!\n');

% end
%% 