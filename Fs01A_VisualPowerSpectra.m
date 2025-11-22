function Fs01A_VisualPowerSpectra(volRes)
% Fs01A_VisualPowerSpectra - Compute simple power spectra for visual areas
%
% This script loads source space data from S02A and computes power spectra
% for left and right visual areas across different trial types for two
% time periods: Fixation (0-0.2s) and Delay (0.5-1.5s).
%
% Inputs:
%   volRes - Volumetric resolution (5, 8, 10)
%
% Outputs:
%   - Computes power spectra for visual areas without baseline correction
%   - Stores group averages for Fixation and Delay periods
%
% Dependencies:
%   - S02A_ReverseModelMNIVolumetric.m (must be run first)
%
% Example:
%   Fs01A_VisualPowerSpectra(5)
%
% Author: Mrugank Dake
% Date: 2025-01-20

clearvars -except volRes; % Keep inputs
close all; clc;

%% Path Setup - Auto-detect HPC vs Local
fprintf('=== MEG Visual Power Spectra Analysis (Fixation & Delay) ===\n');

% Auto-detect environment and set paths
if exist('/scratch/mdd9787', 'dir')
    % HPC environment
    fprintf('Detected HPC environment\n');
    fieldtrip_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/';
    project_path = '/scratch/mdd9787/meg_prf_greene/megScripts';
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
elseif exist('/d/DATD', 'dir')
    % Local environment (DATD)
    fprintf('Detected local DATD environment\n');
    fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
end

% Verify paths exist
if ~exist(fieldtrip_path, 'dir')
    warning('FieldTrip path not found: %s', fieldtrip_path);
end
if ~exist(project_path, 'dir')
    warning('Project path not found: %s', project_path);
end
if ~exist(data_base_path, 'dir')
    warning('Data base path not found: %s', data_base_path);
end

fprintf('FieldTrip path: %s\n', fieldtrip_path);
fprintf('Project path: %s\n', project_path);
fprintf('Data base path: %s\n', data_base_path);

% Add FieldTrip to path
addpath(fieldtrip_path);
ft_defaults;

% Add project path
addpath(genpath(project_path));

fprintf('Volumetric resolution: %dmm\n', volRes);

subjects = [1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32];
% subjects = [1 2 3 4 5];

% Load ROI information
load(sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/atlas/rois_%dmm.mat', volRes), 'left_visual_points', 'right_visual_points');

left_visual_indices = find(left_visual_points == 1);
right_visual_indices = find(right_visual_points == 1);

% Define time windows
fixation_window = [-1, 0];  % Fixation period: -1 to 0s
delay_window = [0.5, 1.5];   % Delay period: 0.5 to 1.5s

fprintf('Processing %d subjects...\n', length(subjects));
fprintf('Time windows: Fixation [%.1f-%.1f]s, Delay [%.1f-%.1f]s\n', ...
    fixation_window(1), fixation_window(2), delay_window(1), delay_window(2));

% Initialize storage arrays for matched conditions only
% Structure: [subjects x frequencies] for matched conditions (left visual-left trials + right visual-right trials)
matched_fixation = [];
matched_delay = [];

for subjID = subjects
    %% Load Data from S02A
    fprintf('Loading source space data for subject %02d...\n', subjID);

    subj_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
        sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, volRes));
    
    if ~exist(subj_data_path, 'file')
        fprintf('Subject %02d data not found, skipping...\n', subjID);
        continue;
    end
    
    load(subj_data_path, 'sourcedataCombined');

    % Extract basic information
    n_trials = length(sourcedataCombined.trial);
    target_info = sourcedataCombined.trialinfo(:, 2); % 2nd column has target info

    fprintf('  Loaded %d trials\n', n_trials);

    % Left trials: targets 4, 5, 6, 7, 8
    % Right trials: targets 1, 2, 3, 9, 10
    left_trials = target_info == 4 | target_info == 5 | target_info == 6 | target_info == 7 | target_info == 8;
    right_trials = target_info == 1 | target_info == 2 | target_info == 3 | target_info == 9 | target_info == 10;

    fprintf('  Left trials: %d, Right trials: %d\n', sum(left_trials), sum(right_trials));

    % Create FieldTrip data structures for left and right visual areas
    % Left visual data
    left_visual_data = sourcedataCombined;
    cfg = [];
    cfg.channel = left_visual_indices;
    left_visual_data = ft_selectdata(cfg, left_visual_data);

    % Right visual data
    right_visual_data = sourcedataCombined;
    cfg = [];
    cfg.channel = right_visual_indices;
    right_visual_data = ft_selectdata(cfg, right_visual_data);

    % Compute power spectra for matched conditions only
    fprintf('  Computing power spectra for matched conditions...\n');
    
    % Initialize arrays for this subject's matched conditions
    leftVisual_leftTrials_fixation = [];
    leftVisual_leftTrials_delay = [];
    rightVisual_rightTrials_fixation = [];
    rightVisual_rightTrials_delay = [];
    
    % Condition 1: Left Visual - Left Trials
    if sum(left_trials) > 0
        cfg = [];
        cfg.trials = left_trials;
        cfg.latency = fixation_window;
        fixation_data = ft_selectdata(cfg, left_visual_data);
        cfg.latency = delay_window;
        delay_data = ft_selectdata(cfg, left_visual_data);
        
        % Fixation period
        cfg = [];
        cfg.method = 'mtmfft';
        cfg.output = 'pow';
        cfg.foi = 2:50;
        cfg.t_ftimwin = ones(length(cfg.foi), 1) * 2;
        cfg.pad = 'nextpow2';
        cfg.taper = 'hanning';
        fixation_spectrum = ft_freqanalysis(cfg, fixation_data);
        
        % Delay period
        delay_spectrum = ft_freqanalysis(cfg, delay_data);
        
        leftVisual_leftTrials_fixation = squeeze(mean(fixation_spectrum.powspctrm, 1));
        leftVisual_leftTrials_delay = squeeze(mean(delay_spectrum.powspctrm, 1));
    end
    
    % Condition 2: Right Visual - Right Trials
    if sum(right_trials) > 0
        cfg = [];
        cfg.trials = right_trials;
        cfg.latency = fixation_window;
        fixation_data = ft_selectdata(cfg, right_visual_data);
        cfg.latency = delay_window;
        delay_data = ft_selectdata(cfg, right_visual_data);
        
        % Fixation period
        cfg = [];
        cfg.method = 'mtmfft';
        cfg.output = 'pow';
        cfg.foi = 2:50;
        cfg.t_ftimwin = ones(length(cfg.foi), 1) * 2;
        cfg.pad = 'nextpow2';
        cfg.taper = 'hanning';
        fixation_spectrum = ft_freqanalysis(cfg, fixation_data);
        
        % Delay period
        delay_spectrum = ft_freqanalysis(cfg, delay_data);
        
        rightVisual_rightTrials_fixation = squeeze(mean(fixation_spectrum.powspctrm, 1));
        rightVisual_rightTrials_delay = squeeze(mean(delay_spectrum.powspctrm, 1));
    end
    
    % Combine matched conditions for this subject (average left visual-left trials and right visual-right trials)
    if ~isempty(leftVisual_leftTrials_fixation) && ~isempty(rightVisual_rightTrials_fixation)
        % Both conditions available - average them
        matched_fixation_subj = (leftVisual_leftTrials_fixation + rightVisual_rightTrials_fixation) / 2;
        matched_delay_subj = (leftVisual_leftTrials_delay + rightVisual_rightTrials_delay) / 2;
    elseif ~isempty(leftVisual_leftTrials_fixation)
        % Only left visual-left trials available
        matched_fixation_subj = leftVisual_leftTrials_fixation;
        matched_delay_subj = leftVisual_leftTrials_delay;
    elseif ~isempty(rightVisual_rightTrials_fixation)
        % Only right visual-right trials available
        matched_fixation_subj = rightVisual_rightTrials_fixation;
        matched_delay_subj = rightVisual_rightTrials_delay;
    else
        % No matched conditions available - skip this subject
        fprintf('  Warning: No matched conditions available for subject %02d\n', subjID);
        continue;
    end
    
    % Store combined results
    matched_fixation = [matched_fixation; matched_fixation_subj];
    matched_delay = [matched_delay; matched_delay_subj];
    
    fprintf('  Subject %02d completed\n', subjID);
    clearvars sourcedataCombined left_visual_data right_visual_data;
end

fprintf('All subjects processed. Computing group averages...\n');

% Compute group averages for matched conditions
if length(subjects) > 1
    matched_fixation_avg = mean(matched_fixation, 1, 'omitnan') * 1e9;
    matched_delay_avg = mean(matched_delay, 1, 'omitnan') * 1e9;
else
    matched_fixation_avg = matched_fixation * 1e9;
    matched_delay_avg = matched_delay * 1e9;
end


disp(size(matched_fixation_avg));

% Create frequency vector for plotting - use actual frequencies from analysis
% We'll get this from the last spectrum computed
freqs = fixation_spectrum.freq;

fprintf('Visual power spectra analysis completed!\n');
fprintf('Subjects processed: %d\n', length(subjects));

%% Create Plots
fprintf('Creating power spectra plots...\n');

% Create figure directory
figures_dir = fullfile(data_base_path, 'figures', 'Fs01');
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

% Create main figure with single plot
figure('Position', [100, 100, 800, 600], 'Renderer', 'painters');

% Plot: Matched Conditions (Left Visual-Left Trials + Right Visual-Right Trials)
plot(freqs, matched_fixation_avg, 'LineWidth', 2, 'Color', 'blue', 'DisplayName', 'Fixation');
hold on;
plot(freqs, matched_delay_avg, 'LineWidth', 2, 'Color', 'red', 'DisplayName', 'Delay');
title('Matched Conditions: Left Visual-Left Trials + Right Visual-Right Trials', 'FontSize', 14);
xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Power (fT^2/Hz)', 'FontSize', 12);
grid on;
xlim([5, 50]);
legend('Location', 'best');

sgtitle(sprintf('Visual Power Spectra Analysis - %dmm Resolution (Fixation vs Delay)', volRes), 'FontSize', 16);

% Save the main figure
saveas(gcf, fullfile(figures_dir, sprintf('power_spectra_matched_conditions_%dmm.png', volRes)));
saveas(gcf, fullfile(figures_dir, sprintf('power_spectra_matched_conditions_%dmm.fig', volRes)));
saveas(gcf, fullfile(figures_dir, sprintf('power_spectra_matched_conditions_%dmm.svg', volRes)));



end
