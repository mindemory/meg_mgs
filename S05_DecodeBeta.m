function S05_DecodeBeta(subjID, resolution, use_loocv)
%% MEG Beta Decoding Analysis (Volumetric)
% Load complex beta signal data and perform decoding analysis
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   resolution - Volumetric resolution (5, 8, or 10mm, default: 10)
%   use_loocv - Use Leave-One-Out CV instead of 5-fold CV (default: false)
%
% Example:
%   S05_DecodeBeta(1, 10, false)  % 10mm, 5-fold CV
%   S05_DecodeBeta(1, 10, true)   % 10mm, LOOCV

if nargin < 1
    error('Subject ID is required');
end
if nargin < 2
    resolution = 10; % Default to 10mm
end
if nargin < 3
    use_loocv = false; % Default to 5-fold CV
end

restoredefaultpath;
clearvars -except subjID resolution use_loocv; % Keep inputs
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


%% Setup paths based on environment
if is_hpc
    % HPC paths
    fieldtrip_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/';
    project_path = '/scratch/mdd9787/meg_prf_greene/megScripts';
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
else
    % Local machine paths
    fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
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

%% Setup and Initialization
addpath(fieldtrip_path);
addpath(genpath(project_path));
ft_defaults;
ft_hastoolbox('spm12', 1);

% Load volumetric sources data
volumetric_path = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-01/sourceRecon/sub-01_task-mgs_volumetricSources_10mm.mat';
fprintf('Loading volumetric sources data from: %s\n', volumetric_path);
load(volumetric_path);

%% Load Complex Beta Signal Data (Volumetric)
% Path to the complex beta signal data in freqspace folder
beta_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'freqspace', ...
    sprintf('sub-%02d_task-mgs_complexbeta_allTargets_%d.mat', subjID, resolution));

if ~exist(beta_data_path, 'file')
    error('Complex beta signal data not found at: %s\nPlease run S03_betaPowerInSource.m first!', beta_data_path);
end

fprintf('Loading complex beta signal data from: %s\n', beta_data_path);
load(beta_data_path, 'sourceDataByTarget');

%% Combine All Trials into Common Structure
fprintf('\n=== Combining All Trials ===\n');

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

%% Baseline Correction (like temp.m)
fprintf('\n=== Baseline Correction ===\n');

% Define baseline period (-0.5 to 0 seconds)
baseline_start = -0.5;
baseline_end = 0.0;
baseline_idx = allTrials.time{1} >= baseline_start & allTrials.time{1} <= baseline_end;

fprintf('Baseline period: %.1f to %.1f s (%d timepoints)\n', baseline_start, baseline_end, sum(baseline_idx));

% Compute baseline power for each source across all trials
nSources = size(allTrials.trial{1}, 1);
baseline_power = zeros(nSources, 1);

for source = 1:nSources
    source_baseline_data = [];
    for trial = 1:length(allTrials.trial)
        source_baseline_data = [source_baseline_data, allTrials.trial{trial}(source, baseline_idx)];
    end
    baseline_power(source) = mean(source_baseline_data);
end

fprintf('Baseline power computed for %d sources\n', nSources);
fprintf('Baseline power range: %.6f to %.6f\n', min(baseline_power), max(baseline_power));

% Apply baseline correction to all trials
fprintf('Applying baseline correction...\n');
for trial = 1:length(allTrials.trial)
    for source = 1:nSources
        allTrials.trial{trial}(source, :) = allTrials.trial{trial}(source, :) ./ baseline_power(source) - 1;
    end
end

fprintf('Baseline correction complete!\n');

%% SVM Classification Analysis
fprintf('\n=== Starting SVM Classification ===\n');

% Extract features and labels
nTrials = length(allTrials.trial);
nTimepoints = length(allTrials.time{1});
nSources = length(allTrials.label);

% Get target labels and convert to binary (left vs right)
targetLocations = allTrials.trialinfo(:, end);

% Define left and right targets
leftTargets = [4, 5, 6, 7, 8];
rightTargets = [1, 2, 3, 9, 10];

% Create binary labels: 1 = left, 2 = right
targetLabels = zeros(length(targetLocations), 1);
targetLabels(ismember(targetLocations, leftTargets)) = 1;  % Left
targetLabels(ismember(targetLocations, rightTargets)) = 2; % Right

uniqueTargets = [1, 2]; % Binary classification
nTargets = 2;

fprintf('Binary classification: Left (targets %s) vs Right (targets %s)\n', ...
    mat2str(leftTargets), mat2str(rightTargets));
fprintf('Left trials: %d, Right trials: %d\n', sum(targetLabels == 1), sum(targetLabels == 2));

% Shuffle trials
fprintf('Shuffling trials...\n');
shuffleIdx = randperm(nTrials);
allTrials.trial = allTrials.trial(shuffleIdx);
allTrials.trialinfo = allTrials.trialinfo(shuffleIdx, :);
targetLabels = targetLabels(shuffleIdx);

% Select posterior 1/3rd of sources based on Y-coordinate (posterior = more negative Y)
posteriorFraction = 1/3;
nPosteriorSources = floor(nSources * posteriorFraction);

% Get source positions and sort by Y-coordinate (posterior = more negative)
sourcePositions = sourcemodel.pos(sourcemodel.inside, :);
yCoordinates = sourcePositions(:, 2); % Y-coordinate (anterior-posterior axis)
[sortedY, sortIdx] = sort(yCoordinates, 'ascend'); % Ascending = most posterior first

% Select the most posterior sources
posteriorIndices = sortIdx(1:nPosteriorSources);

fprintf('Source selection: using posterior %d/%d sources\n', nPosteriorSources, nSources);
fprintf('Y-coordinate range: %.2f to %.2f mm\n', min(yCoordinates(posteriorIndices)), max(yCoordinates(posteriorIndices)));

% Filter trials to keep only posterior sources
for trial = 1:nTrials
    allTrials.trial{trial} = allTrials.trial{trial}(posteriorIndices, :);
end

% Update source labels
allTrials.label = allTrials.label(posteriorIndices);
nSources = nPosteriorSources;

fprintf('Using posterior volumetric sources: %d sources\n', nSources);


% Single time point analysis (average over 0.5 to 1.5 seconds)
analysis_start = 0.5;
analysis_end = 1.5;
analysis_idx = allTrials.time{1} >= analysis_start & allTrials.time{1} <= analysis_end;

fprintf('Single time point analysis: averaging over %.1f to %.1f s (%d timepoints)\n', ...
    analysis_start, analysis_end, sum(analysis_idx));

fprintf('Classification setup:\n');
fprintf('  Total trials: %d\n', nTrials);
fprintf('  Sources: %d\n', nSources);
fprintf('  Target locations: %d\n', nTargets);
fprintf('  Analysis window: %.1f to %.1f s\n', analysis_start, analysis_end);

% Single time point classification
fprintf('\nPerforming single time point classification...\n');

% Extract features (average power over analysis window)
features = zeros(nTrials, nSources);
for trial = 1:nTrials
    % Average power over the analysis window
    features(trial, :) = mean(allTrials.trial{trial}(:, analysis_idx), 2)';
end
    
% Prepare data for classification
X = features;
y = targetLabels;

% Cross-validation setup
if use_loocv
    % Leave-One-Out Cross-Validation
    cv = cvpartition(y, 'LeaveOut');
    nFolds = cv.NumTestSets;
else
    % 5-fold Cross-Validation
    cv = cvpartition(y, 'KFold', 5);
    nFolds = 5;
end

% Initialize predictions
predictions = zeros(nTrials, 1);

% Cross-validation loop
for fold = 1:nFolds
    trainIdx = cv.training(fold);
    testIdx = cv.test(fold);
    
    % Train binary SVM
    svmModel = fitcsvm(X(trainIdx, :), y(trainIdx), ...
        'Standardize', true, 'KernelFunction', 'rbf');
    
    % Test SVM
    predictions(testIdx) = predict(svmModel, X(testIdx, :));
end

% Calculate accuracy
accuracy = sum(predictions == y) / length(y);

% Calculate confusion matrix
confusionMatrix = confusionmat(y, predictions);

fprintf('Classification accuracy: %.2f%%\n', accuracy*100);

%% Results Summary
fprintf('\n=== Classification Results ===\n');
fprintf('Classification accuracy: %.2f%%\n', accuracy*100);
fprintf('Analysis window: %.1f to %.1f s\n', analysis_start, analysis_end);
fprintf('Confusion matrix:\n');
disp(confusionMatrix);

% Display confusion matrix details
fprintf('True Left classified as Left: %d\n', confusionMatrix(1,1));
fprintf('True Left classified as Right: %d\n', confusionMatrix(1,2));
fprintf('True Right classified as Left: %d\n', confusionMatrix(2,1));
fprintf('True Right classified as Right: %d\n', confusionMatrix(2,2));

fprintf('Classification analysis complete!\n');

end
