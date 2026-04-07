function Fs01C_exploreTFplots_IpsiContra(volRes)
% Fs01C_exploreTFplots_IpsiContra - Explore Ipsi vs Contra Time-Frequency plots
%
% This script remaps physical ROIs (Left/Right) to functional streams (Ipsi/Contra)
% relative to the target hemifield for Visual and Frontal regions.
%
% Inputs:
%   volRes - Volumetric resolution (5, 8, 10 mm)
%
% Outputs:
%   - 2x2 Grand Average Master Figure: [Ipsi-Vis, Contra-Vis; Ipsi-Fro, Contra-Fro]
%   - Saves figures in derivatives/figures/Fs01C/
%
% Author: Antigravity

clearvars -except volRes;
close all; clc;

%% Path Setup
fprintf('=== MEG TF Exploration: Functional Ipsi/Contra Streams ===\n');

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

figures_base_dir = fullfile(data_base_path, 'figures', 'Fs01C');
if ~exist(figures_base_dir, 'dir'); mkdir(figures_base_dir); end

subjects = [1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32];

%% Load Atlas Indices
atlas_path = fullfile(data_base_path, 'atlas', sprintf('rois_%dmm.mat', volRes));
load(atlas_path, 'left_visual_points', 'right_visual_points', 'left_frontal_points', 'right_frontal_points');

lvis_idx = find(left_visual_points == 1);
rvis_idx = find(right_visual_points == 1);
lfro_idx = find(left_frontal_points == 1);
rfro_idx = find(right_frontal_points == 1);

fprintf('Aggregating %d subjects...\n', length(subjects));
first_subj = 1;

for subjID = subjects
    fprintf('  Loading Sub-%02d...\n', subjID);
    subj_mat = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
        sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, volRes));
    
    if ~exist(subj_mat, 'file'); continue; end
    load(subj_mat, 'sourcedataCombined');
    
    tgt = sourcedataCombined.trialinfo(:, 2);
    L_trials = (tgt >= 4 & tgt <= 8);
    R_trials = (tgt >= 1 & tgt <= 3) | (tgt >= 9 & tgt <= 10);
    
    % Extraction helper
    get_tf = @(ch, tr) compute_TFRs(ft_selectdata(struct('channel', ch, 'trials', tr), sourcedataCombined), 0);
    
    % VISUAL: Ipsi = (L-ROI @ L-Trial) + (R-ROI @ R-Trial), Contra = (R-ROI @ L-Trial) + (L-ROI @ R-Trial)
    v_ll = get_tf(lvis_idx, L_trials); v_rr = get_tf(rvis_idx, R_trials);
    v_rl = get_tf(rvis_idx, L_trials); v_lr = get_tf(lvis_idx, R_trials);
    
    % FRONTAL
    f_ll = get_tf(lfro_idx, L_trials); f_rr = get_tf(rfro_idx, R_trials);
    f_rl = get_tf(rfro_idx, L_trials); f_lr = get_tf(lfro_idx, R_trials);
    
    % Aggregate Power (Mean across trials/channels for each subject condition)
    ipsi_vis_pow = (mean(v_ll.powspctrm, 1, 'omitnan') + mean(v_rr.powspctrm, 1, 'omitnan')) / 2;
    cont_vis_pow = (mean(v_rl.powspctrm, 1, 'omitnan') + mean(v_lr.powspctrm, 1, 'omitnan')) / 2;
    ipsi_fro_pow = (mean(f_ll.powspctrm, 1, 'omitnan') + mean(f_rr.powspctrm, 1, 'omitnan')) / 2;
    cont_fro_pow = (mean(f_rl.powspctrm, 1, 'omitnan') + mean(f_lr.powspctrm, 1, 'omitnan')) / 2;

    if first_subj
        iv_all = ipsi_vis_pow; cv_all = cont_vis_pow;
        if_all = ipsi_fro_pow; cf_all = cont_fro_pow;
        ref_time = v_ll.time; ref_freq = v_ll.freq; ref_TFR = v_ll;
        first_subj = 0;
    else
        iv_all = cat(4, iv_all, ipsi_vis_pow); cv_all = cat(4, cv_all, cont_vis_pow);
        if_all = cat(4, if_all, ipsi_fro_pow); cf_all = cat(4, cf_all, cont_fro_pow);
    end
end

%% Grand Average & Baseline Correction (dB)
fprintf('Computing Grand Averages and Baseline Correction...\n');
pow_to_dB = @(x) 10*log10(x);
baseline_tidx = find(ref_time >= -1 & ref_time < 0);
baseline_corr = @(data_dB) data_dB - repmat(mean(data_dB(:, :, baseline_tidx, :), 3, 'omitnan'), [1 1 length(ref_time) 1]);

% 1. Convert to dB unit, then Baseline mean per subject, then Mean over subjects
iv_db = baseline_corr(pow_to_dB(iv_all));
cv_db = baseline_corr(pow_to_dB(cv_all));
if_db = baseline_corr(pow_to_dB(if_all));
cf_db = baseline_corr(pow_to_dB(cf_all));

% 2. Average across 4th dimension (Subjects)
avg_tfr = @(d) struct('powspctrm', mean(d, 4, 'omitnan'), 'label', {{'ROI_avg'}}, ...
                    'time', ref_time, 'freq', ref_freq, 'dimord', 'chan_freq_time');
IV_TFR = avg_tfr(iv_db); CV_TFR = avg_tfr(cv_db);
IF_TFR = avg_tfr(if_db); CF_TFR = avg_tfr(cf_db);

%% Visualization
fprintf('Generating Master Figure...\n');
fig = figure('Position', [100, 100, 1600, 1000], 'Visible', 'on', 'Renderer', 'painters');
cfg_plot = []; cfg_plot.figure = 'gcf'; cfg_plot.baseline = 'no'; cfg_plot.colorbar = 'yes';
cfg_plot.colormap = '*RdBu'; cfg_plot.xlim = [-0.5 1.7]; cfg_plot.ylim = [5 40];
cfg_plot.zlim = [-0.4 0.4]; cfg_plot.interactive = 'no';

subplot(2, 2, 1); cfg_plot.title = 'Visual Area - IPSILATERAL'; ft_singleplotTFR(cfg_plot, IV_TFR);
subplot(2, 2, 2); cfg_plot.title = 'Visual Area - CONTRALATERAL'; ft_singleplotTFR(cfg_plot, CV_TFR);
subplot(2, 2, 3); cfg_plot.title = 'Frontal Area - IPSILATERAL'; ft_singleplotTFR(cfg_plot, IF_TFR);
subplot(2, 2, 4); cfg_plot.title = 'Frontal Area - CONTRALATERAL'; ft_singleplotTFR(cfg_plot, CF_TFR);

sgtitle(sprintf('Functional TFR Split (Ipsi/Contra) - Grand Average (N=%d, %dmm)', length(subjects), volRes), 'FontSize', 18, 'FontWeight', 'bold');

saveas(fig, fullfile(figures_base_dir, sprintf('tfr_IpsiContra_Split_%dmm.png', volRes)));
saveas(fig, fullfile(figures_base_dir, sprintf('tfr_IpsiContra_Split_%dmm.fig', volRes)));
saveas(fig, fullfile(figures_base_dir, sprintf('tfr_IpsiContra_Split_%dmm.svg', volRes)));

fprintf('=== Processing Complete. Figures saved in %s ===\n', figures_base_dir);

end
