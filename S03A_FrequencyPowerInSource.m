function S03A_FrequencyPowerInSource(subjID, resolution, frequency_band)
%% MEG Frequency Power Analysis in Source Space
% Load source space data and compute frequency power with lateralization index
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   resolution - Resolution (surface: 5124, 8196, 20484 or volumetric: 8, 10)
%   frequency_band - Frequency band ('theta', 'alpha', 'beta')
%
% Example:
%   S03A_FrequencyPowerInSource(1, 5124, 'beta')
%   S03A_FrequencyPowerInSource(1, 8, 'alpha')
%   S03A_FrequencyPowerInSource(1, 10, 'theta')

if nargin < 1
    error('Subject ID is required');
end
if nargin < 2
    resolution = 5124; % Default resolution
end
if nargin < 3
    frequency_band = 'beta'; % Default frequency band
end

% Validate frequency band
valid_bands = {'theta', 'alpha', 'beta', 'lowgamma'};
if ~ismember(frequency_band, valid_bands)
    error('Invalid frequency band: %s. Must be one of: %s', frequency_band, strjoin(valid_bands, ', '));
end

restoredefaultpath;
clearvars -except subjID resolution frequency_band; % Keep inputs
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

fprintf('=== MEG Frequency Power Analysis in Source Space ===\n');
fprintf('Environment: %s\n', hostname);
fprintf('Detected HPC: %s\n', string(is_hpc));
fprintf('Subject: %d\n', subjID);
fprintf('Resolution: %d\n', resolution);
fprintf('Frequency band: %s\n', frequency_band);

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

%% Define Frequency Bands
frequency_bands = struct();
frequency_bands.theta = [4, 8];    % Theta: 4-8 Hz
frequency_bands.alpha = [8, 13];   % Alpha: 8-13 Hz  
frequency_bands.beta = [18, 30];   % Beta: 18-30 Hz
frequency_bands.lowgamma = [30, 50];   % Low gamma: 30-50 Hz

% Get frequency range for the specified band
freq_range = frequency_bands.(frequency_band);
fprintf('Frequency range: %.1f-%.1f Hz\n', freq_range(1), freq_range(2));

%% Initialize File Paths
% Source data path (input)
source_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, resolution));

% Output directory and file paths
output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'freqSpace');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
frequency_data_path = fullfile(output_dir, sprintf('sub-%02d_task-mgs_complex%s_allTargets_%d.mat', subjID, frequency_band, resolution));

if exist(frequency_data_path, 'file')
    fprintf('Complex %s signal data already exists at: %s\n', frequency_band, frequency_data_path);
    fprintf('Skipping processing to avoid overwriting existing data.\n');
    fprintf('To reprocess, delete the existing file first.\n');
    return;
else
    %% Load Source Space Data
    % Load the source space data created by S02_ReverseModelMNI.m or S02A_ReverseModelMNIVolumetric.m
    if ~exist(source_data_path, 'file')
        error('Source space data not found at: %s\nPlease run S02_ReverseModelMNI.m or S02A_ReverseModelMNIVolumetric.m first!', source_data_path);
    end
    
    fprintf('Loading source space data from: %s\n', source_data_path);
    load(source_data_path);
    
    fprintf('Loaded source space data:\n');
    fprintf('  Total trials: %d\n', length(sourcedataCombined.trial));
    fprintf('  Inside sources: %d\n', length(inside_pos));
        
    %% Process Each Target Location Separately
    fprintf('Processing each target location separately...\n');

    % Define target locations (1-10)
    target_locations = 1:10;
    sourceDataByTarget = cell(10, 1);

    for target = target_locations
        fprintf('Processing target location %d...\n', target);
        
        % Find trials for this specific target
        trial_criteria = sourcedataCombined.trialinfo(:,2) == target;
        valid_trials = find(trial_criteria);
        
        if isempty(valid_trials)
            fprintf('  No trials found for target %d\n', target);
            continue;
        end
        
        fprintf('  Found %d trials for target %d\n', length(valid_trials), target);
        
        % Select trials for this target
        cfg = [];
        cfg.trials = valid_trials;
        sourcedataTarget = ft_selectdata(cfg, sourcedataCombined);
        
        % Apply frequency band filter
        cfg = [];
        cfg.bpfilter = 'yes';
        cfg.bpfreq = freq_range; % Frequency band
        cfg.bpfilttype = 'but';
        cfg.bpfiltord = 4;
        
        sourcedataTarget_freq = ft_preprocessing(cfg, sourcedataTarget);
        
        % Apply Hilbert transform to get analytic signal
        hilbert_compute = @(x) hilbert(x')';
        sourcedataTarget_freq.trial = cellfun(hilbert_compute, sourcedataTarget_freq.trial, 'UniformOutput', false);
        
        % Convert to single precision for memory efficiency
        sourcedataTarget_freq.trial = cellfun(@(x) single(x), sourcedataTarget_freq.trial, 'UniformOutput', false);
        
        % Store the complex frequency signal for this target
        sourceDataByTarget{target} = sourcedataTarget_freq;
        
        % Clear variables for memory efficiency
        clear sourcedataTarget sourcedataTarget_freq;
        
        fprintf('  Target %d processing complete\n', target);
    end
        
    fprintf('%s band processing complete for all targets\n', frequency_band);
        
    %% Save Complex Frequency Signal Results for All Targets
    fprintf('Saving complex %s signal results for all targets...\n', frequency_band);

    % Save all target data together with target information
    save(frequency_data_path, 'sourceDataByTarget', 'target_locations', 'subjID', 'resolution', 'frequency_band', 'freq_range', '-v7.3');
    fprintf('All target data saved to: %s\n', frequency_data_path);

    % Print summary of what was saved
    fprintf('Summary of saved data:\n');
    for target = target_locations
        if ~isempty(sourceDataByTarget{target})
            fprintf('  Target %d: %d trials\n', target, length(sourceDataByTarget{target}.trial));
        else
            fprintf('  Target %d: No data\n', target);
        end
    end
        
    fprintf('\nComplex %s signal analysis complete!\n', frequency_band);
end
