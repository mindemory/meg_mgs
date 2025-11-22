function S02B_organizeBehavForSource(subjID)
% Process Behavioral Data for Source Analysis
% Align behavioral data with MEG trial structure and split into left/right
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%
% Example:
%   S02B_organizeBehavForSource(1)

if nargin < 1
    error('Subject ID is required');
end

restoredefaultpath;
clearvars -except subjID;
close all; clc;

%% Environment Detection
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
    subeyePath = sprintf('/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives/sub-%02d/eyetracking/sub-%02d_task-mgs-iisess.mat', subjID, subjID);
    megPath = sprintf('/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives/sub-%02d/meg/sub-%02d_task-mgs_stimlocked_lineremoved.mat', subjID, subjID);

else
    % Local machine paths
    fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    subeyePath = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/eyetracking/sub-%02d_task-mgs-iisess.mat', subjID, subjID);
    megPath = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/meg/sub-%02d_task-mgs_stimlocked_lineremoved.mat', subjID, subjID);
end

%% Setup and Initialization
addpath(fieldtrip_path);
addpath(genpath(project_path));
ft_defaults;
ft_hastoolbox('spm12', 1);

%% Load Data
% Load behavioral data

load(subeyePath);

% Load MEG data to get trial information for behavioral data alignment
load(megPath, 'epocStimLocked');

%% Account for special cases - remove bad trials from behavioral data
fprintf('Checking for special cases and removing bad trials...\n');
rnum = ii_sess.r_num;

% Initialize bad trials list
badTrials = [];

if subjID == 4 % Trial 1 from run10 is missing
    badTrials = find(rnum == 10, 1);
elseif subjID == 5 % Remove run 1 and 9
    badTrials = find((rnum == 1) | (rnum == 9));
elseif subjID == 10 % Remove run 2 and 7
    badTrials = find((rnum == 2) | (rnum == 7));
elseif subjID == 11 % Remove run 1, 3 and 8
    badTrials = find((rnum == 1) | (rnum == 3) | (rnum == 8));
elseif subjID == 12 % Trial 1 from run 8 is missing
    badTrials = find(rnum == 8, 1);
elseif subjID == 13 % Trials 1, 2 from run 2 are missing
    run2_trials = find(rnum == 2);
    badTrials = run2_trials(1:2);
elseif subjID == 19 % Remove run 8 and 9
    badTrials = find((rnum == 8) | (rnum == 9));
elseif subjID == 23 % Remove run 1
    badTrials = find(rnum == 1);
elseif subjID == 25 % Remove run 8
    badTrials = find(rnum == 8);
elseif subjID == 31 % Remove run 2; Also remove the last 17 trials from run 4
    badTrials = find(rnum == 2);
    run4_trials = find(rnum == 4);
    badTrials = [badTrials; run4_trials(end-16:end)]; % Last 17 trials
elseif subjID == 32 % Remove run 2
    badTrials = find(rnum == 2);
end

% Remove the bad trials from all behavioral variables
if ~isempty(badTrials)
    fprintf('Removing %d bad trials for subject %d\n', length(badTrials), subjID);
    
    % Remove bad trials from all fields in ii_sess except params
    allFields = fieldnames(ii_sess);
    for i = 1:length(allFields)
        fieldName = allFields{i};
        if ~strcmp(fieldName, 'params')
            % Get valid trial indices (excluding bad trials)
            validTrials = setdiff(1:length(ii_sess.(fieldName)), badTrials);
            ii_sess.(fieldName) = ii_sess.(fieldName)(validTrials);
            
            % Check dimensions and convert row vectors to column vectors
            if ~iscell(ii_sess.(fieldName)) && isvector(ii_sess.(fieldName))
                if size(ii_sess.(fieldName), 1) == 1 % Row vector
                    ii_sess.(fieldName) = ii_sess.(fieldName)'; % Convert to column
                end
            end
        end
    end
else
    fprintf('No bad trials to remove for subject %d\n', subjID);
end

% Load source data for verification
fprintf('Loading source data for verification...\n');
sourceDataPath = sprintf('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon/sub-%02d_task-mgs_sourceSpaceData_10.mat', subjID, subjID);
load(sourceDataPath, 'sourcedataCombined');

% Remove trials with NaNs to get valid trial indices
fprintf('Removing trials with NaNs...\n');
has_no_nans = cellfun(@(x) ~any(isnan(x(:))), epocStimLocked.trial)';
valid_trials_all = find(has_no_nans);

% Create epocThis_clean with filtered trials
cfg = [];
cfg.trials = valid_trials_all;
epocThis_clean = ft_selectdata(cfg, epocStimLocked);

%% Apply same trial removal to ii_sess variables
fprintf('Applying same trial removal to ii_sess variables...\n');

% Create ii_sess_forSource with same trial removal logic
ii_sess_forSource = ii_sess;

% Remove trials from all fields in ii_sess except params
allFields = fieldnames(ii_sess);
for i = 1:length(allFields)
    fieldName = allFields{i};
    if ~strcmp(fieldName, 'params')
        ii_sess_forSource.(fieldName) = ii_sess.(fieldName)(valid_trials_all);
    end
end

fprintf('Created ii_sess_forSource with %d valid trials\n', length(valid_trials_all));

%% Define Trial Criteria for Left/Right Split
% Left trials (targets 4,5,6,7,8)
trial_criteria_left = (epocThis_clean.trialinfo(:,2) == 4) | ...
    (epocThis_clean.trialinfo(:,2) == 5) | ...
    (epocThis_clean.trialinfo(:,2) == 6) | ...
    (epocThis_clean.trialinfo(:,2) == 7) | ...
    (epocThis_clean.trialinfo(:,2) == 8);

% Right trials (targets 1,2,3,9,10)
trial_criteria_right = (epocThis_clean.trialinfo(:,2) == 1) | ...
    (epocThis_clean.trialinfo(:,2) == 2) | ...
    (epocThis_clean.trialinfo(:,2) == 3) | ...
    (epocThis_clean.trialinfo(:,2) == 9) | ...
    (epocThis_clean.trialinfo(:,2) == 10);

% Find valid trials (NaNs already removed)
valid_trialsLeft = find(trial_criteria_left);
valid_trialsRight = find(trial_criteria_right);

fprintf('Valid left trials: %d\n', length(valid_trialsLeft));
fprintf('Valid right trials: %d\n', length(valid_trialsRight));

%% Apply same left/right splitting to ii_sess_forSource
fprintf('Splitting ii_sess_forSource into left and right trials...\n');

% Create left and right versions of ii_sess_forSource
ii_sess_left = ii_sess_forSource;
ii_sess_right = ii_sess_forSource;

% Split all fields (except params) into left and right
allFields = fieldnames(ii_sess_forSource);
for i = 1:length(allFields)
    fieldName = allFields{i};
    if ~strcmp(fieldName, 'params')
        ii_sess_left.(fieldName) = ii_sess_forSource.(fieldName)(valid_trialsLeft);
        ii_sess_right.(fieldName) = ii_sess_forSource.(fieldName)(valid_trialsRight);
    end
end

% Append left and right behavioral data
fprintf('Appending left and right behavioral data...\n');
allFields = fieldnames(ii_sess_forSource);
for i = 1:length(allFields)
    fieldName = allFields{i};
    if ~strcmp(fieldName, 'params')
        ii_sess_forSource.(fieldName) = [ii_sess_left.(fieldName); ii_sess_right.(fieldName)];
    end
end

fprintf('Created ii_sess_forSource with left trials first, then right trials\n');

%% Save ii_sess_forSource
fprintf('Saving ii_sess_forSource to new file...\n');

% Create new file path for processed behavioral data
[filepath, filename, ext] = fileparts(subeyePath);
newBehavPath = fullfile(filepath, sprintf('%s_forSource%s', filename, ext));

% Save ii_sess_forSource to new file
save(newBehavPath, 'ii_sess_forSource', '-v7.3');

fprintf('Saved ii_sess_forSource to: %s\n', newBehavPath);

%% Verification
fprintf('Verifying alignment between source data and behavioral data...\n');
fprintf('Sum of differences (should be 0): %d\n', sum(sourcedataCombined.trialinfo(:, 2) - ii_sess_forSource.tarlocCode, 'all', 'omitnan'));

fprintf('Behavioral data processing complete!\n');

end