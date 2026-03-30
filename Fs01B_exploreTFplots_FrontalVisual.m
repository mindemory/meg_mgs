function Fs01B_exploreTFplots_FrontalVisual(volRes)
% Fs01B_exploreTFplots_FrontalVisual - Explore time-frequency plots for Visual & Frontal areas
%
% This script loads source space data from S02A and creates time-frequency
% visualizations for both visual and frontal regions.
%
% Inputs:
%   volRes - Volumetric resolution (5, 8, 10)
%
% Outputs:
%   - Creates time-frequency plots and visualizations for visual and frontal areas
%   - Saves figures as PNG, SVG, and .fig files
%
% Author: Mrugank Dake (modified by agent)

clearvars -except volRes; % Keep inputs
close all; clc;

%% Path Setup
fprintf('=== MEG Time-Frequency Exploration (Visual & Frontal) ===\n');

if exist('/scratch/mdd9787', 'dir')
    fieldtrip_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/';
    project_path = '/scratch/mdd9787/meg_prf_greene/megScripts';
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
elseif exist('/d/DATD', 'dir')
    fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
end

addpath(fieldtrip_path);
ft_defaults;
addpath(genpath(project_path));

%% Create Figures Directory Structure
figures_base_dir = fullfile(data_base_path, 'figures', 'Fs01B');
if ~exist(figures_base_dir, 'dir')
    mkdir(figures_base_dir);
end

subjects = [1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32];

load(sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/atlas/rois_%dmm.mat', volRes), ...
    'left_visual_points', 'right_visual_points', 'left_frontal_points', 'right_frontal_points');

left_visual_indices = find(left_visual_points == 1);
right_visual_indices = find(right_visual_points == 1);
left_frontal_indices = find(left_frontal_points == 1);
right_frontal_indices = find(right_frontal_points == 1);

fprintf('Processing %d subjects...\n', length(subjects));

first_subj = 1;

for subjID = subjects
    fprintf('Loading source space data for subject %02d...\n', subjID);
    subj_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
        sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, volRes));
    
    if ~exist(subj_data_path, 'file')
        fprintf('Subject %02d data not found, skipping...\n', subjID);
        continue;
    end
    
    load(subj_data_path, 'sourcedataCombined');
    target_info = sourcedataCombined.trialinfo(:, 2);
    left_trials = target_info == 4 | target_info == 5 | target_info == 6 | target_info == 7 | target_info == 8;
    right_trials = target_info == 1 | target_info == 2 | target_info == 3 | target_info == 9 | target_info == 10;

    % VISUAL SELECTION
    cfg = []; cfg.channel = left_visual_indices; cfg.trials = left_trials;
    lv_lt_data = ft_selectdata(cfg, sourcedataCombined);
    cfg.trials = right_trials;
    lv_rt_data = ft_selectdata(cfg, sourcedataCombined);

    cfg = []; cfg.channel = right_visual_indices; cfg.trials = left_trials;
    rv_lt_data = ft_selectdata(cfg, sourcedataCombined);
    cfg.trials = right_trials;
    rv_rt_data = ft_selectdata(cfg, sourcedataCombined);

    % FRONTAL SELECTION
    cfg = []; cfg.channel = left_frontal_indices; cfg.trials = left_trials;
    lf_lt_data = ft_selectdata(cfg, sourcedataCombined);
    cfg.trials = right_trials;
    lf_rt_data = ft_selectdata(cfg, sourcedataCombined);

    cfg = []; cfg.channel = right_frontal_indices; cfg.trials = left_trials;
    rf_lt_data = ft_selectdata(cfg, sourcedataCombined);
    cfg.trials = right_trials;
    rf_rt_data = ft_selectdata(cfg, sourcedataCombined);

    fprintf('  Computing TFRs...\n');
    lv_lt_data = compute_TFRs(lv_lt_data, 0);
    lv_rt_data = compute_TFRs(lv_rt_data, 0);
    rv_lt_data = compute_TFRs(rv_lt_data, 0);
    rv_rt_data = compute_TFRs(rv_rt_data, 0);

    lf_lt_data = compute_TFRs(lf_lt_data, 0);
    lf_rt_data = compute_TFRs(lf_rt_data, 0);
    rf_lt_data = compute_TFRs(rf_lt_data, 0);
    rf_rt_data = compute_TFRs(rf_rt_data, 0);

    if first_subj
        leftVisual_leftTrials_power = lv_lt_data.powspctrm;
        leftVisual_rightTrials_power = lv_rt_data.powspctrm;
        rightVisual_leftTrials_power = rv_lt_data.powspctrm;
        rightVisual_rightTrials_power = rv_rt_data.powspctrm;
        
        leftFrontal_leftTrials_power = lf_lt_data.powspctrm;
        leftFrontal_rightTrials_power = lf_rt_data.powspctrm;
        rightFrontal_leftTrials_power = rf_lt_data.powspctrm;
        rightFrontal_rightTrials_power = rf_rt_data.powspctrm;
        
        ref_time = lv_lt_data.time;
        ref_freq = lv_lt_data.freq;
        first_subj = 0;
    else
        leftVisual_leftTrials_power = cat(4, leftVisual_leftTrials_power, lv_lt_data.powspctrm);
        leftVisual_rightTrials_power = cat(4, leftVisual_rightTrials_power, lv_rt_data.powspctrm);
        rightVisual_leftTrials_power = cat(4, rightVisual_leftTrials_power, rv_lt_data.powspctrm);
        rightVisual_rightTrials_power = cat(4, rightVisual_rightTrials_power, rv_rt_data.powspctrm);

        leftFrontal_leftTrials_power = cat(4, leftFrontal_leftTrials_power, lf_lt_data.powspctrm);
        leftFrontal_rightTrials_power = cat(4, leftFrontal_rightTrials_power, lf_rt_data.powspctrm);
        rightFrontal_leftTrials_power = cat(4, rightFrontal_leftTrials_power, rf_lt_data.powspctrm);
        rightFrontal_rightTrials_power = cat(4, rightFrontal_rightTrials_power, rf_rt_data.powspctrm);
    end
    clearvars sourcedataCombined;
end

pow_to_dB = @(x) 10*log10(x);

% Convert to dB
lvl_dB = pow_to_dB(leftVisual_leftTrials_power);
lvr_dB = pow_to_dB(leftVisual_rightTrials_power);
rvl_dB = pow_to_dB(rightVisual_leftTrials_power);
rvr_dB = pow_to_dB(rightVisual_rightTrials_power);

lfl_dB = pow_to_dB(leftFrontal_leftTrials_power);
lfr_dB = pow_to_dB(leftFrontal_rightTrials_power);
rfl_dB = pow_to_dB(rightFrontal_leftTrials_power);
rfr_dB = pow_to_dB(rightFrontal_rightTrials_power);

baseline_tidx = find(ref_time >= -1 & ref_time < 0);

% Function to subtract baseline
baseline_corr = @(data_dB) data_dB - repmat(mean(data_dB(:, :, baseline_tidx, :), 3, 'omitnan'), [1 1 length(ref_time) 1]);

lvl_corr = baseline_corr(lvl_dB);
lvr_corr = baseline_corr(lvr_dB);
rvl_corr = baseline_corr(rvl_dB);
rvr_corr = baseline_corr(rvr_dB);

lfl_corr = baseline_corr(lfl_dB);
lfr_corr = baseline_corr(lfr_dB);
rfl_corr = baseline_corr(rfl_dB);
rfr_corr = baseline_corr(rfr_dB);

% Function to prepare TFR structure
create_tfr = @(base_data, pow_data) setfield(base_data, 'powspctrm', squeeze(mean(pow_data, 4, 'omitnan')));

% VISUAL TFRs
lv_lt_TFR = create_tfr(lv_lt_data, lvl_corr);
lv_rt_TFR = create_tfr(lv_rt_data, lvr_corr);
rv_lt_TFR = create_tfr(rv_lt_data, rvl_corr);
rv_rt_TFR = create_tfr(rv_rt_data, rvr_corr);

% FRONTAL TFRs
lf_lt_TFR = create_tfr(lf_lt_data, lfl_corr);
lf_rt_TFR = create_tfr(lf_rt_data, lfr_corr);
rf_lt_TFR = create_tfr(rf_lt_data, rfl_corr);
rf_rt_TFR = create_tfr(rf_rt_data, rfr_corr);

cfg_plot = []; cfg_plot.figure = 'gcf'; cfg_plot.baseline = 'no'; cfg_plot.colorbar = 'yes';
cfg_plot.colormap = '*RdBu'; cfg_plot.xlim = [-0.5 1.7]; cfg_plot.ylim = [5 40];
cfg_plot.interactive = 'no'; % Prevent topoplot callback error on click
cfg_plot.zlim = [-0.4 0.4];

%% VISUAL FIGURE
fig_vis = figure('Position', [100, 100, 1400, 1000]);
subplot(2, 2, 1); cfg_plot.title = 'Left Visual - Left Trials'; ft_singleplotTFR(cfg_plot, lv_lt_TFR);
subplot(2, 2, 2); cfg_plot.title = 'Left Visual - Right Trials'; ft_singleplotTFR(cfg_plot, lv_rt_TFR);
subplot(2, 2, 3); cfg_plot.title = 'Right Visual - Left Trials'; ft_singleplotTFR(cfg_plot, rv_lt_TFR);
subplot(2, 2, 4); cfg_plot.title = 'Right Visual - Right Trials'; ft_singleplotTFR(cfg_plot, rv_rt_TFR);
sgtitle(sprintf('Visual Area TFRs - %dmm', volRes), 'FontSize', 16);

saveas(fig_vis, fullfile(figures_base_dir, sprintf('tfr_visual_%dmm.png', volRes)));
saveas(fig_vis, fullfile(figures_base_dir, sprintf('tfr_visual_%dmm.fig', volRes)));
saveas(fig_vis, fullfile(figures_base_dir, sprintf('tfr_visual_%dmm.svg', volRes)));

%% FRONTAL FIGURE
fig_front = figure('Position', [200, 200, 1400, 1000]);
subplot(2, 2, 1); cfg_plot.title = 'Left Frontal - Left Trials'; ft_singleplotTFR(cfg_plot, lf_lt_TFR);
subplot(2, 2, 2); cfg_plot.title = 'Left Frontal - Right Trials'; ft_singleplotTFR(cfg_plot, lf_rt_TFR);
subplot(2, 2, 3); cfg_plot.title = 'Right Frontal - Left Trials'; ft_singleplotTFR(cfg_plot, rf_lt_TFR);
subplot(2, 2, 4); cfg_plot.title = 'Right Frontal - Right Trials'; ft_singleplotTFR(cfg_plot, rf_rt_TFR);
sgtitle(sprintf('Frontal Area TFRs - %dmm', volRes), 'FontSize', 16);

saveas(fig_front, fullfile(figures_base_dir, sprintf('tfr_frontal_%dmm.png', volRes)));
saveas(fig_front, fullfile(figures_base_dir, sprintf('tfr_frontal_%dmm.fig', volRes)));
saveas(fig_front, fullfile(figures_base_dir, sprintf('tfr_frontal_%dmm.svg', volRes)));

%% AVERAGE FIGURE
vis_pow_avg = (mean(lv_lt_TFR.powspctrm, 1, 'omitnan') + ...
               mean(lv_rt_TFR.powspctrm, 1, 'omitnan') + ...
               mean(rv_lt_TFR.powspctrm, 1, 'omitnan') + ...
               mean(rv_rt_TFR.powspctrm, 1, 'omitnan')) / 4;
vis_avg_TFR = lv_lt_TFR;
vis_avg_TFR.powspctrm = vis_pow_avg;
vis_avg_TFR.label = vis_avg_TFR.label(1);

front_pow_avg = (mean(lf_lt_TFR.powspctrm, 1, 'omitnan') + ...
                 mean(lf_rt_TFR.powspctrm, 1, 'omitnan') + ...
                 mean(rf_lt_TFR.powspctrm, 1, 'omitnan') + ...
                 mean(rf_rt_TFR.powspctrm, 1, 'omitnan')) / 4;
front_avg_TFR = lf_lt_TFR;
front_avg_TFR.powspctrm = front_pow_avg;
front_avg_TFR.label = front_avg_TFR.label(1);

fig_avg = figure('Position', [300, 300, 1400, 500]);
subplot(1, 2, 1); cfg_plot.title = 'Visual Area - Average All Conditions'; ft_singleplotTFR(cfg_plot, vis_avg_TFR);
subplot(1, 2, 2); cfg_plot.title = 'Frontal Area - Average All Conditions'; ft_singleplotTFR(cfg_plot, front_avg_TFR);
sgtitle(sprintf('Average TFRs across all conditions - %dmm', volRes), 'FontSize', 16);

saveas(fig_avg, fullfile(figures_base_dir, sprintf('tfr_average_%dmm.png', volRes)));
saveas(fig_avg, fullfile(figures_base_dir, sprintf('tfr_average_%dmm.fig', volRes)));
saveas(fig_avg, fullfile(figures_base_dir, sprintf('tfr_average_%dmm.svg', volRes)));

end
