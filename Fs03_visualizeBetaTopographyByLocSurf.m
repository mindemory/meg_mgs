function Fs03_visualizeBetaTopographyByLocSurf(subjID, surface_resolution)
% Fs03_visualizeBetaTopographyByLocSurf - Visualize beta power topography by target location (Surface)
%
% This script loads source space data and computes relative power by:
% 1. Computing average over all trials first (baseline)
% 2. Dividing each trial by this baseline
% 3. Averaging for each location
%
% Inputs:
%   subjID - Subject ID (1-21) or 'all' for group average
%   surface_resolution - Surface resolution (default: 5124)
%
% Outputs:
%   - Saves figures as PNG and .fig files
%   - Individual subject figures: MEG_HPC/derivatives/figures/Fs03/sub-XX/
%   - Group average figures: MEG_HPC/derivatives/figures/Fs03/group/
%
% Dependencies:
%   - Source space data files with sourcedataCombined structure
%
% Example:
%   Fs03_visualizeBetaTopographyByLocSurf(1, 5124)  % Individual subject
%   Fs03_visualizeBetaTopographyByLocSurf('all', 5124)  % Group average
%
% Author: Mrugank Dake
% Date: 2025-01-20

% Set default surface resolution if not provided
if nargin < 2
    surface_resolution = 5124; % Default to 5124 vertices
end

% restoredefaultpath;
clearvars -except subjID surface_resolution; % Keep inputs
close all; clc;

%% Path Setup - Auto-detect HPC vs Local
fprintf('=== MEG Beta Power Topography Visualization (Surface) ===\n');

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
fprintf('Surface resolution: %d vertices\n', surface_resolution);

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
        subj_data_path = fullfile(data_base_path, sprintf('sub-%02d', s), 'sourceRecon', ...
            sprintf('sub-%02d_task-mgs_complexBeta_allTargets_%d.mat', s, surface_resolution));
        
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
                all_power_data = NaN(n_sources, n_locations, length(subjects));
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
    
    % Calculate symmetric color limits around 0.0 (baseline)
    all_power_values = group_power_data(:);
    all_power_values = all_power_values(~isnan(all_power_values)); % Remove NaN values
    % data_mean = mean(all_power_values);
    phigh = quantile(all_power_values, 0.99);
    plow = quantile(all_power_values, 0.01);
    max_deviation = max(abs(phigh - 0), abs(plow - 0));
    color_min = 0.0 - max_deviation;
    color_max = 0.0 + max_deviation;
    fprintf('  Color range for group (symmetric around 0.0): %.3f to %.3f (deviation: %.3f)\n', color_min, color_max, max_deviation);
    
    % Load template surface - check multiple locations
    surface_file = sprintf('cortex_%d.surf.gii', surface_resolution);
    
    % Try current directory first, then common locations
    if exist(surface_file, 'file')
        template_mesh = ft_read_headshape(surface_file);
    elseif exist(fullfile('/d/DATD/hyper/software/fieldtrip-20250318/template/anatomy', surface_file), 'file')
        template_mesh = ft_read_headshape(fullfile('/d/DATD/hyper/software/fieldtrip-20250318/template/anatomy', surface_file));
    elseif exist(fullfile('/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/template/anatomy', surface_file), 'file')
        template_mesh = ft_read_headshape(fullfile('/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/template/anatomy', surface_file));
    else
        error('Surface file not found: %s. Please ensure cortex_*.surf.gii files are available.', surface_file);
    end
    
    fprintf('Using surface file: %s\n', surface_file);
    
    % Separate hemispheres using template mesh coordinates
    left_hemisphere_idx = template_mesh.pos(:,1) < 0;
    right_hemisphere_idx = template_mesh.pos(:,1) > 0;
    
    % Create vertex maps for reindexing
    left_vertex_map = find(left_hemisphere_idx);
    right_vertex_map = find(right_hemisphere_idx);
    
    %% Figure 1: Combined Left and Right Hemisphere Surface Visualization
    fprintf('Creating combined left and right hemisphere surface visualization...\n');
    
    figure('Position', [100, 400, 1500, 1000], 'Renderer', 'painters');
    
    % Create 2x6 subplot layout (2 hemispheres × 6 locations)
    for i = 1:n_locations
        location = i;
        
        % Left hemisphere subplot (row 1)
        subplot(2, 6, i);
        
        % Get relative power data for this location
        target_power_data_group = group_power_data(:, location);
        
        % Create source structure for visualization
        sourceVisualize = struct();
        sourceVisualize.pos = template_mesh.pos;
        sourceVisualize.tri = template_mesh.tri;
        sourceVisualize.unit = 'mm';
        sourceVisualize.coordsys = 'mni';
        sourceVisualize.pow = target_power_data_group;
        
        % Create separate structure for left hemisphere
        sourceVisualize_left = sourceVisualize;
        sourceVisualize_left.pos = template_mesh.pos(left_hemisphere_idx, :);
        sourceVisualize_left.pow = target_power_data_group(left_hemisphere_idx); % Filter power data to match hemisphere
        
        % Reindex triangulation for left hemisphere
        left_tri = template_mesh.tri;
        left_tri_valid = all(left_hemisphere_idx(left_tri), 2);
        left_tri = left_tri(left_tri_valid, :);
        % Create new vertex indices
        [~, left_new_indices] = ismember(left_tri, left_vertex_map);
        sourceVisualize_left.tri = left_new_indices;
        
        % Use ft_sourceplot for surface visualization
        cfg = [];
        cfg.method = 'surface';
        cfg.figure = 'gcf';
        cfg.funparameter = 'pow';
        cfg.maskparameter = "";
        cfg.surffile = surface_file;
        cfg.colorbar = 'no'; % Remove individual colorbars
        cfg.funcolormap = colormap(flipud(brewermap([], 'RdBu')));
        cfg.funcolorlim = [color_min, color_max];
        ft_sourceplot(cfg, sourceVisualize_left);
        
        % Set view angle and lighting
        view(-45, 30); % Left hemisphere view
        lighting gouraud;
        material dull;
        
        % Labels and title
        title(sprintf('Location %d\n%d°', location, location_angles(i)));
        
        % Right hemisphere subplot (row 2)
        subplot(2, 6, i + 6);
        
        % Create separate structure for right hemisphere
        sourceVisualize_right = sourceVisualize;
        sourceVisualize_right.pos = template_mesh.pos(right_hemisphere_idx, :);
        sourceVisualize_right.pow = target_power_data_group(right_hemisphere_idx); % Filter power data to match hemisphere
        
        % Reindex triangulation for right hemisphere
        right_tri = template_mesh.tri;
        right_tri_valid = all(right_hemisphere_idx(right_tri), 2);
        right_tri = right_tri(right_tri_valid, :);
        % Create new vertex indices
        [~, right_new_indices] = ismember(right_tri, right_vertex_map);
        sourceVisualize_right.tri = right_new_indices;
        
        % Use ft_sourceplot for surface visualization
        cfg = [];
        cfg.method = 'surface';
        cfg.figure = 'gcf';
        cfg.funparameter = 'pow';
        cfg.maskparameter = "";
        cfg.surffile = surface_file;
        cfg.colorbar = 'no'; % Remove individual colorbars
        cfg.funcolormap = colormap(flipud(brewermap([], 'RdBu')));
        cfg.funcolorlim = [color_min, color_max];
        ft_sourceplot(cfg, sourceVisualize_right);
        
        % Set view angle and lighting
        view(45, 30); % Right hemisphere view
        lighting gouraud;
        material dull;
        
        % Labels and title
        title(sprintf('Location %d\n%d°', location, location_angles(i)));
    end
    
    % Overall title
    sgtitle(sprintf('Group Average: Beta Relative Power - Left vs Right Hemisphere (6 Locations, %d vertices)', surface_resolution));
    
    % Add overall colorbar
    c = colorbar('Position', [0.92, 0.15, 0.02, 0.7]);
    c.Label.String = 'Relative Power';
    caxis([color_min, color_max]);
    
    % Save combined figure
    fig_name = sprintf('group_betaPower_surf_combined_%d', surface_resolution);
    saveas(gcf, fullfile(figures_dir, [fig_name, '.fig']));
    saveas(gcf, fullfile(figures_dir, [fig_name, '.png']));
    fprintf('Saved: %s\n', fig_name);
    
    % close all;
    
else
    % Load data for individual subject
    fprintf('Loading data for subject %02d...\n', subjID);
    
    % Load complex beta data
    subj_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
        sprintf('sub-%02d_task-mgs_complexBeta_allTargets_%d.mat', subjID, surface_resolution));
    
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
    
    % Calculate symmetric color limits around 0.0 (baseline)
    all_power_values = relative_power_data(:);
    all_power_values = all_power_values(~isnan(all_power_values)); % Remove NaN values
    phigh = quantile(all_power_values, 0.99);
    plow = quantile(all_power_values, 0.01);
    max_deviation = max(abs(phigh - 0), abs(plow - 0));
    color_min = 0.0 - max_deviation;
    color_max = 0.0 + max_deviation;
    fprintf('  Color range for subject (symmetric around 0.0): %.3f to %.3f (deviation: %.3f)\n', color_min, color_max, max_deviation);
    
    % Load template surface - check multiple locations
    surface_file = sprintf('cortex_%d.surf.gii', surface_resolution);
    
    % Try current directory first, then common locations
    if exist(surface_file, 'file')
        template_mesh = ft_read_headshape(surface_file);
    elseif exist(fullfile('/d/DATD/hyper/software/fieldtrip-20250318/template/anatomy', surface_file), 'file')
        template_mesh = ft_read_headshape(fullfile('/d/DATD/hyper/software/fieldtrip-20250318/template/anatomy', surface_file));
    elseif exist(fullfile('/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/template/anatomy', surface_file), 'file')
        template_mesh = ft_read_headshape(fullfile('/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/template/anatomy', surface_file));
    else
        error('Surface file not found: %s. Please ensure cortex_*.surf.gii files are available.', surface_file);
    end
    
    fprintf('Using surface file: %s\n', surface_file);
    
    % Separate hemispheres using template mesh coordinates
    left_hemisphere_idx = template_mesh.pos(:,1) < 0;
    right_hemisphere_idx = template_mesh.pos(:,1) > 0;
    
    % Create vertex maps for reindexing
    left_vertex_map = find(left_hemisphere_idx);
    right_vertex_map = find(right_hemisphere_idx);
    
    %% Figure 1: Combined Left and Right Hemisphere Surface Visualization
    fprintf('Creating combined left and right hemisphere surface visualization...\n');
    
    figure('Position', [100, 400, 1500, 1000], 'Renderer', 'painters');
    
    % Create 2x6 subplot layout (2 hemispheres × 6 locations)
    for i = 1:n_locations
        location = i;
        
        % Left hemisphere subplot (row 1)
        subplot(2, 6, i);
        
        % Get relative power data for this location
        target_power_data = relative_power_data(:, location);
        
        % Create source structure for visualization
        sourceVisualize = struct();
        sourceVisualize.pos = template_mesh.pos;
        sourceVisualize.tri = template_mesh.tri;
        sourceVisualize.unit = 'mm';
        sourceVisualize.coordsys = 'mni';
        sourceVisualize.pow = target_power_data;
        
        % Create separate structure for left hemisphere
        sourceVisualize_left = sourceVisualize;
        sourceVisualize_left.pos = template_mesh.pos(left_hemisphere_idx, :);
        sourceVisualize_left.pow = target_power_data(left_hemisphere_idx); % Filter power data to match hemisphere
        
        % Reindex triangulation for left hemisphere
        left_tri = template_mesh.tri;
        left_tri_valid = all(left_hemisphere_idx(left_tri), 2);
        left_tri = left_tri(left_tri_valid, :);
        % Create new vertex indices
        [~, left_new_indices] = ismember(left_tri, left_vertex_map);
        sourceVisualize_left.tri = left_new_indices;
        
        % Use ft_sourceplot for surface visualization
        cfg = [];
        cfg.method = 'surface';
        cfg.figure = 'gcf';
        cfg.funparameter = 'pow';
        cfg.maskparameter = "";
        cfg.surffile = surface_file;
        cfg.colorbar = 'no'; % Remove individual colorbars
        cfg.funcolormap = colormap(flipud(brewermap([], 'RdBu')));
        cfg.funcolorlim = [color_min, color_max];
        ft_sourceplot(cfg, sourceVisualize_left);
        
        % Set view angle and lighting
        view(-45, 30); % Left hemisphere view
        lighting gouraud;
        material dull;
        
        % Labels and title
        title(sprintf('Location %d\n%d°', location, location_angles(i)));
        
        % Right hemisphere subplot (row 2)
        subplot(2, 6, i + 6);
        
        % Create separate structure for right hemisphere
        sourceVisualize_right = sourceVisualize;
        sourceVisualize_right.pos = template_mesh.pos(right_hemisphere_idx, :);
        sourceVisualize_right.pow = target_power_data(right_hemisphere_idx); % Filter power data to match hemisphere
        
        % Reindex triangulation for right hemisphere
        right_tri = template_mesh.tri;
        right_tri_valid = all(right_hemisphere_idx(right_tri), 2);
        right_tri = right_tri(right_tri_valid, :);
        % Create new vertex indices
        [~, right_new_indices] = ismember(right_tri, right_vertex_map);
        sourceVisualize_right.tri = right_new_indices;
        
        % Use ft_sourceplot for surface visualization
        cfg = [];
        cfg.method = 'surface';
        cfg.figure = 'gcf';
        cfg.funparameter = 'pow';
        cfg.maskparameter = "";
        cfg.surffile = surface_file;
        cfg.colorbar = 'no'; % Remove individual colorbars
        cfg.funcolormap = colormap(flipud(brewermap([], 'RdBu')));
        cfg.funcolorlim = [color_min, color_max];
        ft_sourceplot(cfg, sourceVisualize_right);
        
        % Set view angle and lighting
        view(45, 30); % Right hemisphere view
        lighting gouraud;
        material dull;
        
        % Labels and title
        title(sprintf('Location %d\n%d°', location, location_angles(i)));
    end
    
    % Overall title
    sgtitle(sprintf('Subject %02d: Beta Relative Power - Left vs Right Hemisphere (6 Locations, %d vertices)', subjID, surface_resolution));
    
    % Add overall colorbar
    c = colorbar('Position', [0.92, 0.15, 0.02, 0.7]);
    c.Label.String = 'Relative Power';
    caxis([color_min, color_max]);
    
    % Save combined figure
    fig_name = sprintf('sub-%02d_betaPower_surf_combined_%d', subjID, surface_resolution);
    saveas(gcf, fullfile(figures_dir, [fig_name, '.fig']));
    saveas(gcf, fullfile(figures_dir, [fig_name, '.png']));
    fprintf('Saved: %s\n', fig_name);
    
    
    close all;
end
