function Fs01A_PowerSpectra(volRes)
% Fs01A_PowerSpectra - Compute simple power spectra for visual and frontal areas
%
% This script loads source space data from S02A and computes power spectra
% for left and right visual AND frontal areas across different trial types for two
% time periods: Fixation (0-0.2s) and Delay (0.5-1.5s).
%
% Inputs:
%   volRes - Volumetric resolution (5, 8, 10)
%
% Outputs:
%   - Computes power spectra for visual and frontal areas without baseline correction
%   - Stores group averages for Fixation and Delay periods
%   - Saves output figures as PNG, FIG, and SVG formats
%
% Dependencies:
%   - S02A_ReverseModelMNIVolumetric.m (must be run first)
%
% Example:
%   Fs01A_PowerSpectra(5)
%
% Author: Mrugank Dake
% Date: 2025-01-20

clearvars -except volRes; % Keep inputs
close all; clc;

%% Path Setup - Auto-detect HPC vs Local
fprintf('=== MEG Visual & Frontal Power Spectra Analysis (Fixation & Delay) ===\n');

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
load(sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/atlas/rois_%dmm.mat', volRes), ...
    'left_visual_points', 'right_visual_points', 'left_frontal_points', 'right_frontal_points');

left_visual_indices = find(left_visual_points == 1);
right_visual_indices = find(right_visual_points == 1);
left_frontal_indices = find(left_frontal_points == 1);
right_frontal_indices = find(right_frontal_points == 1);

% Define time windows
fixation_window = [-1, 0];  % Fixation period: -1 to 0s
delay_window = [0.5, 1.5];   % Delay period: 0.5 to 1.5s

fprintf('Processing %d subjects...\n', length(subjects));
fprintf('Time windows: Fixation [%.1f-%.1f]s, Delay [%.1f-%.1f]s\n', ...
    fixation_window(1), fixation_window(2), delay_window(1), delay_window(2));

% Initialize storage arrays for matched conditions only
matched_fixation_vis = [];
matched_delay_vis = [];

matched_fixation_front = [];
matched_delay_front = [];

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
    fprintf('  Computing power spectra for matched conditions...\n');
    
    leftVis_fix = []; leftVis_del = [];
    leftFront_fix = []; leftFront_del = [];
    rightVis_fix = []; rightVis_del = [];
    rightFront_fix = []; rightFront_del = [];
    
    % Get data for LEFT trials
    if sum(left_trials) > 0
        [leftVis_fix, leftVis_del, freqs] = get_roi_spectra(sourcedataCombined, left_visual_indices, left_trials, fixation_window, delay_window);
        [leftFront_fix, leftFront_del, ~] = get_roi_spectra(sourcedataCombined, left_frontal_indices, left_trials, fixation_window, delay_window);
    end
    
    % Get data for RIGHT trials
    if sum(right_trials) > 0
        [rightVis_fix, rightVis_del, freqs] = get_roi_spectra(sourcedataCombined, right_visual_indices, right_trials, fixation_window, delay_window);
        [rightFront_fix, rightFront_del, ~] = get_roi_spectra(sourcedataCombined, right_frontal_indices, right_trials, fixation_window, delay_window);
    end
    
    % Combine matched conditions for this subject (VISUAL)
    if ~isempty(leftVis_fix) && ~isempty(rightVis_fix)
        matched_fixation_vis_subj = (leftVis_fix + rightVis_fix) / 2;
        matched_delay_vis_subj = (leftVis_del + rightVis_del) / 2;
    elseif ~isempty(leftVis_fix)
        matched_fixation_vis_subj = leftVis_fix;
        matched_delay_vis_subj = leftVis_del;
    elseif ~isempty(rightVis_fix)
        matched_fixation_vis_subj = rightVis_fix;
        matched_delay_vis_subj = rightVis_del;
    else
        fprintf('  Warning: No matched conditions available for subject %02d (Visual)\n', subjID);
        continue;
    end
    
    % Combine matched conditions for this subject (FRONTAL)
    if ~isempty(leftFront_fix) && ~isempty(rightFront_fix)
        matched_fixation_front_subj = (leftFront_fix + rightFront_fix) / 2;
        matched_delay_front_subj = (leftFront_del + rightFront_del) / 2;
    elseif ~isempty(leftFront_fix)
        matched_fixation_front_subj = leftFront_fix;
        matched_delay_front_subj = leftFront_del;
    elseif ~isempty(rightFront_fix)
        matched_fixation_front_subj = rightFront_fix;
        matched_delay_front_subj = rightFront_del;
    else
        fprintf('  Warning: No matched conditions available for subject %02d (Frontal)\n', subjID);
        continue;
    end
    
    % Store combined results
    matched_fixation_vis = [matched_fixation_vis; matched_fixation_vis_subj];
    matched_delay_vis = [matched_delay_vis; matched_delay_vis_subj];
    
    matched_fixation_front = [matched_fixation_front; matched_fixation_front_subj];
    matched_delay_front = [matched_delay_front; matched_delay_front_subj];
    
    fprintf('  Subject %02d completed\n', subjID);
    clearvars sourcedataCombined;
end

fprintf('All subjects processed. Computing group averages...\n');

% Compute group averages for matched conditions
if size(matched_fixation_vis, 1) > 1
    matched_fixation_vis_avg = mean(matched_fixation_vis, 1, 'omitnan') * 1e9;
    matched_delay_vis_avg = mean(matched_delay_vis, 1, 'omitnan') * 1e9;
    
    matched_fixation_front_avg = mean(matched_fixation_front, 1, 'omitnan') * 1e9;
    matched_delay_front_avg = mean(matched_delay_front, 1, 'omitnan') * 1e9;
else
    matched_fixation_vis_avg = matched_fixation_vis * 1e9;
    matched_delay_vis_avg = matched_delay_vis * 1e9;
    
    matched_fixation_front_avg = matched_fixation_front * 1e9;
    matched_delay_front_avg = matched_delay_front * 1e9;
end


fprintf('Visual and Frontal power spectra analysis completed!\n');
fprintf('Subjects processed: %d\n', size(matched_fixation_vis, 1));

%% Create Plots
fprintf('Creating power spectra plots...\n');

% Create figure directory
figures_dir = fullfile(data_base_path, 'figures', 'Fs01');
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

% -------------- VISUAL FIGURE --------------
fig_vis = figure('Position', [100, 100, 800, 600], 'Renderer', 'painters');

plot(freqs, matched_fixation_vis_avg, 'LineWidth', 2, 'Color', 'blue', 'DisplayName', 'Fixation');
hold on;
plot(freqs, matched_delay_vis_avg, 'LineWidth', 2, 'Color', 'red', 'DisplayName', 'Delay');
title('Visual Areas Matched Conditions: Left(Vis)-Left(Trial) + Right(Vis)-Right(Trial)', 'FontSize', 14);
xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Power (fT^2/Hz)', 'FontSize', 12);
grid on;
xlim([5, 40]);
legend('Location', 'best');

sgtitle(sprintf('Visual Power Spectra Analysis - %dmm Resolution (Fixation vs Delay)', volRes), 'FontSize', 16);

% Save the main figure for Visual
saveas(fig_vis, fullfile(figures_dir, sprintf('power_spectra_visual_matched_%dmm.png', volRes)));
saveas(fig_vis, fullfile(figures_dir, sprintf('power_spectra_visual_matched_%dmm.fig', volRes)));
saveas(fig_vis, fullfile(figures_dir, sprintf('power_spectra_visual_matched_%dmm.svg', volRes)));

% -------------- FRONTAL FIGURE --------------
fig_front = figure('Position', [200, 200, 800, 600], 'Renderer', 'painters');

plot(freqs, matched_fixation_front_avg, 'LineWidth', 2, 'Color', 'blue', 'DisplayName', 'Fixation');
hold on;
plot(freqs, matched_delay_front_avg, 'LineWidth', 2, 'Color', 'red', 'DisplayName', 'Delay');
title('Frontal Areas Matched Conditions: Left(Frontal)-Left(Trial) + Right(Frontal)-Right(Trial)', 'FontSize', 14);
xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Power (fT^2/Hz)', 'FontSize', 12);
grid on;
xlim([5, 40]);
legend('Location', 'best');

sgtitle(sprintf('Frontal Power Spectra Analysis - %dmm Resolution (Fixation vs Delay)', volRes), 'FontSize', 16);

% Save the main figure for Frontal
saveas(fig_front, fullfile(figures_dir, sprintf('power_spectra_frontal_matched_%dmm.png', volRes)));
saveas(fig_front, fullfile(figures_dir, sprintf('power_spectra_frontal_matched_%dmm.fig', volRes)));
saveas(fig_front, fullfile(figures_dir, sprintf('power_spectra_frontal_matched_%dmm.svg', volRes)));

end

%% SUBFUNCTIONS 
function [fix_pow, del_pow, freqs] = get_roi_spectra(sourcedataCombined, roi_indices, target_trials, fixation_window, delay_window)
    % Select ROI channels
    cfg = [];
    cfg.channel = roi_indices;
    roi_data = ft_selectdata(cfg, sourcedataCombined);
    
    % Ensure that roi_data is a structural fieldtrip output before slicing time/trials
    
    % Select Fixation Window Data
    cfg = [];
    cfg.trials = target_trials;
    cfg.latency = fixation_window;
    fixation_data = ft_selectdata(cfg, roi_data);
    
    % Select Delay Window Data
    cfg.latency = delay_window;
    delay_data = ft_selectdata(cfg, roi_data);
    
    % Compute freq analysis for Fixation
    cfg = [];
    cfg.method = 'mtmfft';
    cfg.output = 'pow';
    cfg.foi = 2:50;
    cfg.t_ftimwin = ones(length(cfg.foi), 1) * 2;
    cfg.pad = 'nextpow2';
    cfg.taper = 'hanning';
    
    fixation_spectrum = ft_freqanalysis(cfg, fixation_data);
    
    % Compute freq analysis for Delay
    delay_spectrum = ft_freqanalysis(cfg, delay_data);
    
    fix_pow = squeeze(mean(fixation_spectrum.powspctrm, 1));
    del_pow = squeeze(mean(delay_spectrum.powspctrm, 1));
    freqs = fixation_spectrum.freq;
end
