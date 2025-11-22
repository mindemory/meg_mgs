function Fs01_exploreTFplots(volRes)
% Fs01_exploreTFplots - Explore time-frequency plots from source space data
%
% This script loads source space data from S02A and creates time-frequency
% visualizations to explore the data structure and content.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   volRes - Volumetric resolution (5, 8, 10)
%
% Outputs:
%   - Creates time-frequency plots and visualizations
%   - Saves figures as PNG and .fig files
%
% Dependencies:
%   - S02A_ReverseModelMNIVolumetric.m (must be run first)
%
% Example:
%   Fs01_exploreTFplots(5)
%
% Author: Mrugank Dake
% Date: 2025-01-20

clearvars -except volRes; % Keep inputs
close all; clc;

%% Path Setup - Auto-detect HPC vs Local
fprintf('=== MEG Time-Frequency Exploration ===\n');

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

%% Create Figures Directory Structure
figures_base_dir = fullfile(data_base_path, 'figures', 'Fs01');


% Create directories if they don't exist
if ~exist(figures_base_dir, 'dir')
    mkdir(figures_base_dir);
end

subjects = [1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32];
% subjects = [1 2];

% save('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/atlas/rois_5mm.mat', "visual_points", "parietal_points", "frontal_points", ...
%     "left_visual_points", "right_visual_points", "left_parietal_points", "right_parietal_points", "left_frontal_points", "right_frontal_points");
load(sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/atlas/rois_%dmm.mat', volRes), 'left_visual_points', 'right_visual_points');

left_visual_indices = find(left_visual_points == 1);
right_visual_indices = find(right_visual_points == 1);

% leftVisual_leftTrials_power = [];
% leftVisual_rightTrials_power = [];
% rightVisual_leftTrials_power = [];
% rightVisual_rightTrials_power = [];

fprintf('Processing %d subjects...\n', length(subjects));

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

    % Create FieldTrip data structures for left and right visual areas (ALL TRIALS)
    % Left visual data (all trials)
    left_visual_data = sourcedataCombined;
    cfg = [];
    cfg.channel = left_visual_indices;
    cfg.trials = left_trials;
    leftVisual_leftTrials_data = ft_selectdata(cfg, left_visual_data);
    cfg.trials = right_trials;
    leftVisual_rightTrials_data = ft_selectdata(cfg, left_visual_data);

    % Right visual data (all trials)
    right_visual_data = sourcedataCombined;
    cfg = [];
    cfg.channel = right_visual_indices;
    cfg.trials = left_trials;
    rightVisual_leftTrials_data = ft_selectdata(cfg, right_visual_data);
    cfg.trials = right_trials;
    rightVisual_rightTrials_data = ft_selectdata(cfg, right_visual_data);

    % Compute TFR using compute_TFRs function for ALL trials
    fprintf('  Computing TFR for left visual area (all trials)...\n');
    leftVisual_leftTrials_data = compute_TFRs(leftVisual_leftTrials_data, 0); % With baseline correction
    leftVisual_rightTrials_data = compute_TFRs(leftVisual_rightTrials_data, 0);

    fprintf('  Computing TFR for right visual area (all trials)...\n');
    rightVisual_leftTrials_data = compute_TFRs(rightVisual_leftTrials_data, 0); % With baseline correction
    rightVisual_rightTrials_data = compute_TFRs(rightVisual_rightTrials_data, 0);

    % Segregate power spectra by trial type
    leftVisual_leftTrials_power_subj = leftVisual_leftTrials_data.powspctrm;
    leftVisual_rightTrials_power_subj = leftVisual_rightTrials_data.powspctrm;
    rightVisual_leftTrials_power_subj = rightVisual_leftTrials_data.powspctrm;
    rightVisual_rightTrials_power_subj = rightVisual_rightTrials_data.powspctrm;

    % Concatenate across subjects
    if subjID == 1
        % Initialize arrays to store power spectra across subjects
        leftVisual_leftTrials_power = leftVisual_leftTrials_power_subj;
        leftVisual_rightTrials_power = leftVisual_rightTrials_power_subj;
        rightVisual_leftTrials_power = rightVisual_leftTrials_power_subj;
        rightVisual_rightTrials_power = rightVisual_rightTrials_power_subj;
    else
        leftVisual_leftTrials_power = cat(4, leftVisual_leftTrials_power, leftVisual_leftTrials_power_subj);
        leftVisual_rightTrials_power = cat(4, leftVisual_rightTrials_power, leftVisual_rightTrials_power_subj);
        rightVisual_leftTrials_power = cat(4, rightVisual_leftTrials_power, rightVisual_leftTrials_power_subj);
        rightVisual_rightTrials_power = cat(4, rightVisual_rightTrials_power, rightVisual_rightTrials_power_subj);
    end
    fprintf('  Subject %02d completed\n', subjID);
    clearvars sourcedataCombined left_visual_data right_visual_data;
end

fprintf('All subjects processed. Computing group averages...\n');

leftVisual_leftTrials_power_dB = 10*log10(leftVisual_leftTrials_power);
leftVisual_rightTrials_power_dB = 10*log10(leftVisual_rightTrials_power);
rightVisual_leftTrials_power_dB = 10*log10(rightVisual_leftTrials_power);
rightVisual_rightTrials_power_dB = 10*log10(rightVisual_rightTrials_power);

baseline_tidx = find(leftVisual_leftTrials_data.time >= -1 & ...
                     leftVisual_leftTrials_data.time < 0);
fixation_tidx = find(leftVisual_leftTrials_data.time >= -0.5 & ...
                     leftVisual_leftTrials_data.time < 0);
delay_tidx = find(leftVisual_leftTrials_data.time >= 1.0 & ...
                  leftVisual_leftTrials_data.time < 1.7);
leftVisual_leftTrials_baseline = repmat(mean(leftVisual_leftTrials_power_dB(:, :, baseline_tidx, :), 3, 'omitnan'), ...
                                        [1 1 length(leftVisual_leftTrials_data.time) 1]);
leftVisual_rightTrials_baseline = repmat(mean(leftVisual_rightTrials_power_dB(:, :, baseline_tidx, :), 3, 'omitnan'), ...
                                        [1 1 length(leftVisual_leftTrials_data.time) 1]);
rightVisual_leftTrials_baseline = repmat(mean(rightVisual_leftTrials_power_dB(:, :, baseline_tidx, :), 3, 'omitnan'), ...
                                        [1 1 length(leftVisual_leftTrials_data.time) 1]);
rightVisual_rightTrials_baseline = repmat(mean(rightVisual_rightTrials_power_dB(:, :, baseline_tidx, :), 3, 'omitnan'), ...
                                        [1 1 length(leftVisual_leftTrials_data.time) 1]);
% Compute group averages (unbasecorr)
leftVisual_leftTrials_TFR_uncorr = leftVisual_leftTrials_data;
leftVisual_leftTrials_TFR_uncorr.powspctrm = squeeze(mean(leftVisual_leftTrials_power, 4, 'omitnan'));
leftVisual_rightTrials_TFR_uncorr = leftVisual_rightTrials_data;
leftVisual_rightTrials_TFR_uncorr.powspctrm = squeeze(mean(leftVisual_rightTrials_power, 4, 'omitnan'));
rightVisual_leftTrials_TFR_uncorr = rightVisual_leftTrials_data;
rightVisual_leftTrials_TFR_uncorr.powspctrm = squeeze(mean(rightVisual_leftTrials_power, 4, 'omitnan'));
rightVisual_rightTrials_TFR_uncorr = rightVisual_rightTrials_data;
rightVisual_rightTrials_TFR_uncorr.powspctrm = squeeze(mean(rightVisual_rightTrials_power, 4, 'omitnan'));

% Compute group averages (unbasecorr)
leftVisual_leftTrials_TFR = leftVisual_leftTrials_data;
leftVisual_leftTrials_TFR.powspctrm = squeeze(mean(leftVisual_leftTrials_power_dB - leftVisual_leftTrials_baseline, 4, 'omitnan'));
leftVisual_rightTrials_TFR = leftVisual_rightTrials_data;
leftVisual_rightTrials_TFR.powspctrm = squeeze(mean(leftVisual_rightTrials_power_dB - leftVisual_rightTrials_baseline, 4, 'omitnan'));
rightVisual_leftTrials_TFR = rightVisual_leftTrials_data;
rightVisual_leftTrials_TFR.powspctrm = squeeze(mean(rightVisual_leftTrials_power_dB - rightVisual_leftTrials_baseline, 4, 'omitnan'));
rightVisual_rightTrials_TFR = rightVisual_rightTrials_data;
rightVisual_rightTrials_TFR.powspctrm = squeeze(mean(rightVisual_rightTrials_power_dB - rightVisual_rightTrials_baseline, 4, 'omitnan'));

% Visualize group averages
figure('Position', [100, 100, 1400, 1000]);

cfg = [];
cfg.figure = 'gcf';
cfg.baseline = 'no';
cfg.colorbar = 'yes';
cfg.colormap = '*RdBu';
cfg.xlim = [-0.5 1.7];
cfg.ylim = [5 40];
% cfg.zlim = [-0.5 0.5];

% Top row: Left Visual Area
subplot(2, 2, 1);
cfg.title = 'Left Visual Area - Left Trials (Group Average)';
ft_singleplotTFR(cfg, leftVisual_leftTrials_TFR);

subplot(2, 2, 2);
cfg.title = 'Left Visual Area - Right Trials (Group Average)';
ft_singleplotTFR(cfg, leftVisual_rightTrials_TFR);

% Bottom row: Right Visual Area
subplot(2, 2, 3);
cfg.title = 'Right Visual Area - Left Trials (Group Average)';
ft_singleplotTFR(cfg, rightVisual_leftTrials_TFR);

subplot(2, 2, 4);
cfg.title = 'Right Visual Area - Right Trials (Group Average)';
ft_singleplotTFR(cfg, rightVisual_rightTrials_TFR);

fprintf('Group TFR analysis completed!\n');
%%
% Create combined ipsilateral, contralateral and difference spectrograms
ipsi_TFR = leftVisual_leftTrials_TFR;
ipsi_TFR.powspctrm = mean(cat(1, leftVisual_leftTrials_TFR.powspctrm, rightVisual_rightTrials_TFR.powspctrm), 1, 'omitnan');
contra_TFR = leftVisual_rightTrials_TFR;
contra_TFR.powspctrm = mean(cat(1, leftVisual_rightTrials_TFR.powspctrm, rightVisual_leftTrials_TFR.powspctrm), 1, 'omitnan');
lateralized_TFR = ipsi_TFR;
lateralized_TFR.powspctrm = (contra_TFR.powspctrm + ipsi_TFR.powspctrm)./2;% ./ (contra_TFR.powspctrm + ipsi_TFR.powspctrm + 1e-3);
ipsi_TFR.label = ipsi_TFR.label(1);
contra_TFR.label = contra_TFR.label(1);
lateralized_TFR.label = lateralized_TFR.label(1);


figure('Position', [100, 100, 2000, 1500], 'Renderer','painters');
cfg = [];
cfg.figure = 'gcf';
cfg.baseline = 'no';
cfg.colorbar = 'yes';
cfg.colormap = '*RdBu';
cfg.xlim = [-0.5 1.7];
cfg.ylim = [5 40];

subplot(2, 2, 1)
cfg.zlim = [-0.5 0.5];
cfg.title = 'Ipsi Power';
ft_singleplotTFR(cfg, ipsi_TFR);

subplot(2, 2, 2)
cfg.zlim = [-0.5 0.5];
cfg.title = 'Contra Power';
ft_singleplotTFR(cfg, contra_TFR);

subplot(2, 2, 3)
cfg.zlim = [-0.4 0.4];
cfg.title = 'Lateralized power';
ft_singleplotTFR(cfg, lateralized_TFR);

subplot(2, 2, 4)
% plot(leftVisual_leftTrials_TFR_uncorr.freq, mean(leftVisual_leftTrials_TFR_uncorr.powspctrm(:, :, fixation_tidx), [1, 3], 'omitnan') * 1e7, ...
%     'LineWidth', 2, 'Color', 'black', 'DisplayName', 'Ipsi Fixation');
spcLeft = mean(leftVisual_leftTrials_TFR_uncorr.powspctrm(:, :, delay_tidx), [1, 3], 'omitnan');
spcRight = mean(leftVisual_rightTrials_TFR_uncorr.powspctrm(:, :, delay_tidx), [1, 3], 'omitnan');
spcSumm = (spcLeft + spcRight) ./ 2;
hold on;
plot(leftVisual_leftTrials_TFR_uncorr.freq, mean(leftVisual_rightTrials_TFR_uncorr.powspctrm(:, :, fixation_tidx), [1, 3], 'omitnan') * 1e7, ...
    'LineWidth', 2, 'Color', 'black', 'DisplayName', 'Fixation');
plot(leftVisual_leftTrials_TFR_uncorr.freq, spcSumm * 1e7, ...
    'LineWidth', 2, 'Color', 'red', 'DisplayName', 'Delay');
% plot(leftVisual_leftTrials_TFR_uncorr.freq, mean(leftVisual_rightTrials_TFR_uncorr.powspctrm(:, :, delay_tidx), [1, 3], 'omitnan') * 1e7, ...
%     'LineWidth', 2, 'Color', 'blue', 'DisplayName', 'Contra Delay');
xlabel('Frequency (Hz)', 'FontSize', 12);
ylabel('Power (fT^2/Hz)', 'FontSize', 12);
% grid on;
xlim([5, 40]);
legend('Location', 'best');


end
