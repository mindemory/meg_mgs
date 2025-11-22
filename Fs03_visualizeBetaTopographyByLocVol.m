function Fs03_visualizeBetaTopographyByLocVol(subjID, vol_resolution)
% Fs03_visualizeBetaTopographyByLocVol - Visualize beta power topography by target location (Volumetric)
%
% This script loads source space data and computes relative power by:
% 1. Computing average over all trials first (baseline)
% 2. Dividing each trial by this baseline
% 3. Averaging for each location
%
% Inputs:
%   subjID - Subject ID (1-21) or 'all' for group average
%   vol_resolution - Volumetric resolution (default: 10mm)
%
% Outputs:
%   - Saves figures as PNG and .fig files
%   - Individual subject figures: MEG_HPC/derivatives/figures/S03/sub-XX/
%   - Group average figures: MEG_HPC/derivatives/figures/S03/group/
%
% Dependencies:
%   - Source space data files with sourcedataCombined structure
%
% Example:
%   Fs03_visualizeBetaTopographyByLocVol(1, 10)  % Individual subject
%   Fs03_visualizeBetaTopographyByLocVol('all', 10)  % Group average
%
% Author: Mrugank Dake
% Date: 2025-01-20

% Set default volumetric resolution if not provided
if nargin < 2
    vol_resolution = 10; % Default to 10mm
end

% restoredefaultpath;
clearvars -except subjID vol_resolution; % Keep inputs
close all; clc;

%% Path Setup - Auto-detect HPC vs Local
fprintf('=== MEG Beta Power Topography Visualization (Volumetric) ===\n');

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
else
    % Fallback - try to detect from current working directory
    current_dir = pwd;
    if contains(current_dir, 'scratch')
        fprintf('Detected HPC environment (from working directory)\n');
        fieldtrip_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/';
        project_path = '/scratch/mdd9787/meg_prf_greene/megScripts';
        data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
    else
        fprintf('Detected local environment (fallback)\n');
        fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
        project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
        data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    end
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
addpath(project_path);

fprintf('Subject: %s\n', string(subjID));
fprintf('Volumetric resolution: %dmm\n', vol_resolution);

%% Create Figures Directory Structure
figures_base_dir = fullfile(data_base_path, 'figures', 'Fs03');
if strcmp(subjID, 'all')
    figures_dir = fullfile(figures_base_dir, 'group');
else
    figures_dir = fullfile(figures_base_dir, sprintf('sub-%02d', subjID));
end

% Create directories if they don't exist
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

%% Load Data
if strcmp(subjID, 'all')
    % Load data from all subjects for group average
    fprintf('Loading data from all subjects for group average...\n');
    
    % Define subjects array
    subjects = [1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32];
    % subjects = [1 2];
    
    % Define 6 locations by grouping targets
    location_groups = {1, [2,3], [4,5], 6, [7,8], [9,10]};
    location_angles = [0, 37.5, 142.5, 180, 217.5, 322.5]; % Average angles for each location
    n_locations = length(location_groups);
    
    all_power_data = [];
    
    for s = subjects
        % Load complex beta data
        subj_data_path = fullfile(data_base_path, sprintf('sub-%02d', s), 'sourceRecon', 'freqSpace', ...
            sprintf('sub-%02d_task-mgs_complexbeta_allTargets_%d.mat', s, vol_resolution));
        
        if exist(subj_data_path, 'file')
            fprintf('  Loading subject %02d complex beta data...\n', s);
            loaded_data = load(subj_data_path, 'sourceDataByTarget');
            sourceDataByTarget = loaded_data.sourceDataByTarget;
            
            % Extract trial information
            n_sources = length(sourceDataByTarget{1}.label);
            n_targets = length(sourceDataByTarget);
            n_trialsAll = sum(arrayfun(@(x) length(sourceDataByTarget{x}.trial), 1:10));
            
            % Compute power for each location and trial
            fprintf('    Computing power for subject %02d...\n', s);
            
            % Initialize power data matrix (sources x locations)
            subject_power_data = zeros(n_sources, n_locations);
            grand_avg_power = zeros(n_sources, n_trialsAll); % Pre-allocate large enough
            trlCounter = 0;
            TOI = [0.5, 1.0];
            TOI_idx = sourceDataByTarget{1}.time{1} >= TOI(1) & sourceDataByTarget{1}.time{1} <= TOI(2);
            
            % Step 1: Compute power for each location (grouped targets)
            for location = 1:n_locations
                target_group = location_groups{location};
                fprintf('      Processing location %d (targets %s)...\n', location, mat2str(target_group));
                
                % Collect power data for all targets in this location
                location_power_data = [];
                
                for target_idx = 1:length(target_group)
                    target = target_group(target_idx);
                    target_data = sourceDataByTarget{target};
                    n_trials = length(target_data.trial);
                    
                    fprintf('        Processing target %d (%d trials)...\n', target, n_trials);
                    
                    % Compute power for each trial (magnitude squared of complex data)
                    trial_powers = zeros(n_sources, n_trials);
                    for trial = 1:n_trials
                        % Get complex data for this trial
                        complex_data = target_data.trial{trial}(:, TOI_idx);
                        trlPower = mean(abs(complex_data).^2, 2, 'omitnan');
                        % Compute power (magnitude squared)
                        trial_powers(:, trial) = trlPower;
                        trlCounter = trlCounter + 1;
                        grand_avg_power(:, trlCounter) = trlPower;
                    end
                    
                    % Average power across trials for this target
                    target_power = mean(trial_powers, 2, 'omitnan');
                    location_power_data = [location_power_data, target_power];
                end
                
                % Average power across all targets in this location
                subject_power_data(:, location) = mean(location_power_data, 2, 'omitnan');
            end
            
            % Step 2: Compute grand average power across all targets and trials
            grand_average_power = mean(grand_avg_power, 2, 'omitnan');
            
            % Step 3: Compute relative power (divide each target's average by grand average)
            relative_power_data = (subject_power_data ./ grand_average_power) - 1;
            
            % Store data for group average
            if isempty(all_power_data)
                all_power_data = zeros(n_sources, n_locations, length(subjects));
            end
            all_power_data(:, :, s == subjects) = relative_power_data;
        else
            fprintf('  Subject %02d data not found, skipping...\n', s);
        end
    end
    
    % Compute group average
    group_power_data = mean(all_power_data, 3, 'omitnan');
    
    % Create group visualization
    fprintf('Creating group visualization...\n');
    
    % Calculate symmetric color limits around 1.0 (baseline)
    all_power_values = group_power_data(:);
    all_power_values = all_power_values(~isnan(all_power_values)); % Remove NaN values
    % data_mean = mean(all_power_values);
    phigh = quantile(all_power_values, 0.99);
    plow = quantile(all_power_values, 0.01);
    max_deviation = max(abs(phigh - 0), abs(plow - 0));
    color_min = 0.0 - max_deviation;
    color_max = 0.0 + max_deviation;
    fprintf('  Color range for group (symmetric around 0.0): %.3f to %.3f (deviation: %.3f)\n', color_min, color_max, max_deviation);
    
    % Load volumetric sourcemodel
    sourcemodel_path = sprintf('/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/standard_sourcemodel3d%dmm.mat', vol_resolution);
    
    fprintf('Loading volumetric sourcemodel from: %s\n', sourcemodel_path);
    load(sourcemodel_path);
    template_sourcemodel = sourcemodel;
    sourcemodel_pos = template_sourcemodel.pos(template_sourcemodel.inside, :);
    clear sourcemodel; % Clean up
    
    %% Figure 1: MNI Template Volumetric Visualization
    fprintf('Creating MNI template volumetric visualization...\n');
    
    figure('Position', [100, 400, 1500, 1000], 'Renderer', 'painters');
    
    % Create circular subplot layout
    for i = 1:n_locations
        location = i;
        
        % Calculate position for circular layout
        angle_rad = deg2rad(location_angles(i));
        radius = 0.35;
        x_pos = 0.5 + radius * cos(angle_rad);
        y_pos = 0.5 + radius * sin(angle_rad);
        subplot_size = 0.12;
        
        % Create subplot at calculated position
        subplot('Position', [x_pos - subplot_size/2, y_pos - subplot_size/2, subplot_size, subplot_size]);
        
        % Get relative power data for this location
        target_power_data_group = group_power_data(:, location);
        
        % Create volumetric scatter plot
        scatter3(sourcemodel_pos(:,1), sourcemodel_pos(:,2), sourcemodel_pos(:,3), ...
            20, target_power_data_group, 'filled');
        
        % Set color limits and colormap
        % caxis([color_min, color_max]);
        caxis([-0.03 0.03]);
        colormap(flipud(brewermap([], 'RdBu')))

        colorbar;
        view(0, 40)
        
        % Labels and title
        title(sprintf('Location %d\n%d°', location, location_angles(i)));
        
        % Set axis properties
        axis equal;
        axis off;
        grid off;
    end
    
    % Overall title
    sgtitle(sprintf('Group Average: Beta Relative Power in Volumetric Space (6 Locations, %dmm)', vol_resolution));
    
    % Save figure
    fig_name = sprintf('group_betaPower_vol_%dmm', vol_resolution);
    
    saveas(gcf, fullfile(figures_dir, [fig_name, '.fig']));
    saveas(gcf, fullfile(figures_dir, [fig_name, '.png']));
    fprintf('Saved: %s\n', fig_name);
    
    % close all;
    
else
    % Load data for individual subject
    fprintf('Loading data for subject %02d...\n', subjID);
    
    % Load complex beta data
    subj_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'freqSpace', ...
        sprintf('sub-%02d_task-mgs_complexbeta_allTargets_%d.mat', subjID, vol_resolution));
    
    if ~exist(subj_data_path, 'file')
        error('Complex beta data not found at: %s', subj_data_path);
    end
    
    loaded_data = load(subj_data_path, 'sourceDataByTarget');
    sourceDataByTarget = loaded_data.sourceDataByTarget;
    
    % Extract trial information
    n_sources = length(sourceDataByTarget{1}.label);
    n_targets = length(sourceDataByTarget);
    n_trialsAll = sum(arrayfun(@(x) length(sourceDataByTarget{x}.trial), 1:10));
    
    fprintf('Loaded complex beta data with %d sources and %d targets\n', n_sources, n_targets);
    
    % Define 6 locations by grouping targets
    location_groups = {1, [2,3], [4,5], 6, [7,8], [9,10]};
    location_angles = [0, 37.5, 142.5, 180, 217.5, 322.5]; % Average angles for each location
    n_locations = length(location_groups);
    
    % Compute power for each location and trial
    fprintf('Computing power for each target location...\n');
    
    % Initialize power data matrix (sources x locations)
    subject_power_data = NaN(n_sources, n_locations);
    grand_avg_power = NaN(n_sources, n_trialsAll);
    trlCounter = 0;
    TOI = [0.5, 1.0];
    TOI_idx = sourceDataByTarget{1}.time{1} >= TOI(1) & sourceDataByTarget{1}.time{1} <= TOI(2);
    
    % Step 1: Compute power for each location (grouped targets)
    for location = 1:n_locations
        target_group = location_groups{location};
        fprintf('  Processing location %d (targets %s)...\n', location, mat2str(target_group));
        
        % Collect power data for all targets in this location
        location_power_data = [];
        
        for target_idx = 1:length(target_group)
            target = target_group(target_idx);
            target_data = sourceDataByTarget{target};
            n_trials = length(target_data.trial);
            
            fprintf('    Processing target %d (%d trials)...\n', target, n_trials);
            
            % Compute power for each trial (magnitude squared of complex data)
            trial_powers = NaN(n_sources, n_trials);
            for trial = 1:n_trials
                % Get complex data for this trial
                complex_data = target_data.trial{trial}(:, TOI_idx);
                trlPower = mean(abs(complex_data).^2, 2, 'omitnan');
                % Compute power (magnitude squared)
                trial_powers(:, trial) = trlPower;
                trlCounter = trlCounter + 1;
                grand_avg_power(:, trlCounter) = trlPower;
            end
            
            % Average power across trials for this target
            target_power = mean(trial_powers, 2, 'omitnan');
            location_power_data = [location_power_data, target_power];
        end
        
        % Average power across all targets in this location
        subject_power_data(:, location) = mean(location_power_data, 2, 'omitnan');
    end
    
    % Step 2: Compute grand average power across all targets and trials
    grand_average_power = mean(grand_avg_power, 2, 'omitnan');
    fprintf('Grand average power: %.6f\n', grand_average_power);
    
    % Step 3: Compute relative power (divide each target's average by grand average)
    relative_power_data = (subject_power_data ./ grand_average_power) - 1;
    
    % Create individual subject visualization
    fprintf('Creating individual subject visualization...\n');
    
    % Calculate symmetric color limits around 1.0 (baseline)
    all_power_values = relative_power_data(:);
    all_power_values = all_power_values(~isnan(all_power_values)); % Remove NaN values
    % data_mean = mean(all_power_values);
    phigh = quantile(all_power_values, 0.99);
    plow = quantile(all_power_values, 0.01);
    max_deviation = max(abs(phigh - 0), abs(plow - 0));
    color_min = 0.0 - max_deviation;
    color_max = 0.0 + max_deviation;
    fprintf('  Color range for subject (symmetric around 0.0): %.3f to %.3f (deviation: %.3f)\n', color_min, color_max, max_deviation);
    
    % Load volumetric sourcemodel
    sourcemodel_path = sprintf('/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/standard_sourcemodel3d%dmm.mat', vol_resolution);
    
    fprintf('Loading volumetric sourcemodel from: %s\n', sourcemodel_path);
    load(sourcemodel_path);
    template_sourcemodel = sourcemodel;
    sourcemodel_pos = template_sourcemodel.pos(template_sourcemodel.inside, :);
    clear sourcemodel; % Clean up
    
    % Define 6 locations by grouping targets
    location_groups = {1, [2,3], [4,5], 6, [7,8], [9,10]};
    location_angles = [0, 37.5, 142.5, 180, 217.5, 322.5]; % Average angles for each location
    n_locations = length(location_groups);
    
    %% Figure 1: MNI Template Volumetric Visualization
    fprintf('Creating MNI template volumetric visualization...\n');
    
    figure('Position', [100, 400, 1500, 1000], 'Renderer', 'painters');
    
    % Create circular subplot layout
    for i = 1:n_locations
        location = i;
        
        % Calculate position for circular layout
        angle_rad = deg2rad(location_angles(i));
        radius = 0.35;
        x_pos = 0.5 + radius * cos(angle_rad);
        y_pos = 0.5 + radius * sin(angle_rad);
        subplot_size = 0.12;
        
        % Create subplot at calculated position
        subplot('Position', [x_pos - subplot_size/2, y_pos - subplot_size/2, subplot_size, subplot_size]);
        
        % Get relative power data for this location
        target_power_data = relative_power_data(:, location);
        
        % Create volumetric scatter plot
        scatter3(sourcemodel_pos(:,1), sourcemodel_pos(:,2), sourcemodel_pos(:,3), ...
            50, target_power_data, 'filled');
        
        % Set color limits and colormap
        caxis([color_min, color_max]);
        colormap(flipud(brewermap([], 'RdBu')))

        colorbar;
        view(0, 40)
        
        % Labels and title
        title(sprintf('Location %d\n%d°', location, location_angles(i)));
        
        % Set axis properties
        axis equal;
        axis off;
        grid off;
    end
    
    % Overall title
    sgtitle(sprintf('Subject %02d: Beta Relative Power in Volumetric Space (6 Locations, %dmm)', subjID, vol_resolution));
    
    % Save figure
    fig_name = sprintf('sub-%02d_betaPower_vol_%dmm', subjID, vol_resolution);
    
    saveas(gcf, fullfile(figures_dir, [fig_name, '.fig']));
    saveas(gcf, fullfile(figures_dir, [fig_name, '.png']));
    fprintf('Saved: %s\n', fig_name);
    
    
    close all;
end
