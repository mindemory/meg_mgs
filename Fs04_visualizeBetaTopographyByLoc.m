function Fs04_visualizeBetaTopographyByLoc(subjID, surface_resolution)
% Fs04_visualizeBetaTopographyByLoc - Visualize beta power topography by target location
%
% This script loads source space data and computes relative power by:
% 1. Computing average over all trials first (baseline)
% 2. Dividing each trial by this baseline
% 3. Averaging for each location
%
% Inputs:
%   subjID - Subject ID (1-21) or 'all' for group average
%   surface_resolution - Surface resolution (5124 or 8196)
%
% Outputs:
%   - Saves figures as PNG and .fig files
%   - Individual subject figures: MEG_HPC/derivatives/figures/S04/sub-XX/
%   - Group average figures: MEG_HPC/derivatives/figures/S04/group/
%
% Dependencies:
%   - Source space data files with sourcedataCombined structure
%
% Example:
%   Fs04_visualizeBetaTopographyByLoc(1, 5124)  % Individual subject
%   Fs04_visualizeBetaTopographyByLoc('all', 5124)  % Group average
%
% Author: Mrugank Dake
% Date: 2025-01-20

% restoredefaultpath;
clearvars -except subjID surface_resolution; % Keep inputs
close all; clc;

%% Path Setup - Auto-detect HPC vs Local
fprintf('=== MEG Beta Power Topography Visualization ===\n');

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
figures_base_dir = fullfile(data_base_path, 'figures', 'S04');
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
    % subjects = [1 2 3 4 5 6 7 9 10 12 13 15 17 18 19 23 24 25 29 31 32];
    subjects = [1 2];
    
    all_power_data = [];
    valid_subjects = [];
    
    for s = subjects
        subj_data_path = fullfile(data_base_path, sprintf('sub-%02d', s), 'sourceRecon', ...
            sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', s, surface_resolution));
        
        if exist(subj_data_path, 'file')
            fprintf('  Loading subject %02d...\n', s);
            loaded_data = load(subj_data_path, 'sourcedataCombined');
            sourcedata = loaded_data.sourcedataCombined;
            
            % Extract trial information
            n_sources = length(sourcedata.label);
            target_info = sourcedata.trialinfo(:, 2); % 2nd column has target info
            
            % Compute relative power using new method for this subject
            fprintf('    Computing relative power for subject %02d...\n', s);
            
            % Define time window for analysis (0.5-1.0s)
            time_start = 0.5;
            time_end = 1.0;
            
            % Get time vector from first trial to determine time indices
            time_vector = sourcedata.time{1};
            time_idx = time_vector >= time_start & time_vector <= time_end;
            
            % Step 1: Compute average over all trials first (baseline) - only for time window
            % Use cellfun to extract and average time window for all trials
            trial_data_windows = cellfun(@(trial_data) mean(trial_data(:, time_idx), 2, 'omitnan'), ...
                                        sourcedata.trial, 'UniformOutput', false);
            all_trial_data = [trial_data_windows{:}]; % Concatenate all trials
            baseline = mean(all_trial_data, 2, 'omitnan'); % Average across all trials
            
            % Step 2: Compute relative power for each target
            subject_power_data = zeros(n_sources, 10);
            
            for target = 1:10
                % Find trials for this target
                target_trials = find(target_info == target);
                
                if ~isempty(target_trials)
                    % Compute relative power for each trial of this target using cellfun
                    target_trial_data = sourcedata.trial(target_trials);
                    relative_power_trials = cellfun(@(trial_data) mean(trial_data(:, time_idx), 2, 'omitnan') - baseline , ...
                                                   target_trial_data, 'UniformOutput', false);
                    relative_power_matrix = [relative_power_trials{:}]; % Concatenate all trials
                    
                    % Step 3: Average for this location (across trials)
                    subject_power_data(:, target) = mean(relative_power_matrix, 2);
                else
                    subject_power_data(:, target) = NaN;
                end
            end
            
            % Store data for group average
            if isempty(all_power_data)
                all_power_data = zeros(n_sources, 10, length(subjects));
            end
            all_power_data(:, :, s == subjects) = subject_power_data;
            valid_subjects = [valid_subjects, s];
        else
            fprintf('  Subject %02d data not found, skipping...\n', s);
        end
    end
    
    % Compute group average
    group_power_data = mean(all_power_data, 3, 'omitnan');
    fprintf('Group average computed from %d subjects\n', length(valid_subjects));
    
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
    
    % Define target locations and their angles
    target_locations = 1:10;
    target_angles = [0, 25, 50, 130, 155, 180, 205, 230, 310, 335]; % degrees
    n_targets = length(target_locations);
    
    %% Figure 1: MNI Template Surface Visualization
    fprintf('Creating MNI template surface visualization...\n');
    
    figure('Position', [100, 400, 1500, 1000], 'Renderer', 'painters');
    
    % Create circular subplot layout
    for i = 1:n_targets
        target = target_locations(i);
        
        % Calculate position for circular layout
        angle_rad = deg2rad(target_angles(i));
        radius = 0.35;
        x_pos = 0.5 + radius * cos(angle_rad);
        y_pos = 0.5 + radius * sin(angle_rad);
        subplot_size = 0.12;
        
        % Create subplot at calculated position
        subplot('Position', [x_pos - subplot_size/2, y_pos - subplot_size/2, subplot_size, subplot_size]);
        
        % Get relative power data for this target
        relative_power_data = group_power_data(:, target);
        
        % Create source structure for visualization
        sourceVisualize = struct();
        sourceVisualize.pos = template_mesh.pos;
        sourceVisualize.tri = template_mesh.tri;
        sourceVisualize.unit = 'mm';
        sourceVisualize.coordsys = 'mni';
        sourceVisualize.pow = relative_power_data;
        
        % Use ft_sourceplot for surface visualization
        cfg = [];
        cfg.method = 'surface';
        cfg.figure = 'gcf';
        cfg.funparameter = 'pow';
        cfg.maskparameter = "";
        cfg.surffile = sprintf('cortex_%d.surf.gii', surface_resolution);
        cfg.colorbar = 'yes';
        cfg.funcolormap = 'jet';
        cfg.funcolorlim = [color_min, color_max];
        ft_sourceplot(cfg, sourceVisualize);
        
        % Set view angle and lighting
        view(0, 40);
        lighting gouraud;
        material dull;
        light('Position', [-1, -1, 1], 'Style', 'infinite', 'Color', [0.4, 0.4, 0.4]);
        
        % Labels and title
        title(sprintf('Target %d\n%d°', target, target_angles(i)));
        xlabel('X (mm)');
        ylabel('Y (mm)');
        zlabel('Z (mm)');
        
        % Set axis properties
        axis equal;
        axis tight;
        grid on;
    end
    
    % Overall title
    sgtitle(sprintf('Group Average: Beta Relative Power on Template Midthickness Surface (Targets 1-10, %d vertices)', surface_resolution));
    
    % % Save figure
    % fig_name = sprintf('group_betaPower_cortex_%d', surface_resolution);
    
    % saveas(gcf, fullfile(figures_dir, [fig_name, '.fig']));
    % saveas(gcf, fullfile(figures_dir, [fig_name, '.png']));
    % fprintf('Saved: %s\n', fig_name);
    
    % close all;
    
else
    % Load data for individual subject
    fprintf('Loading data for subject %02d...\n', subjID);
    
    subj_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
        sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, surface_resolution));
    
    if ~exist(subj_data_path, 'file')
        error('Source space data not found at: %s', subj_data_path);
    end
    
    loaded_data = load(subj_data_path, 'sourcedataCombined');
    sourcedata = loaded_data.sourcedataCombined;
    
    % Extract trial information
    n_trials = length(sourcedata.trial);
    n_sources = length(sourcedata.label);
    target_info = sourcedata.trialinfo(:, 2); % 2nd column has target info
    
    fprintf('Loaded %d trials with %d sources\n', n_trials, n_sources);
    
    % Compute relative power using new method
    fprintf('Computing relative power using baseline method...\n');
    
    % Define time window for analysis (0.5-1.0s)
    time_start = 0.5;
    time_end = 1.0;
    fprintf('  Using time window: %.1f-%.1fs\n', time_start, time_end);
    
    % Get time vector from first trial to determine time indices
    time_vector = sourcedata.time{1};
    time_idx = time_vector >= time_start & time_vector <= time_end;
    n_timepoints_window = sum(time_idx);
    fprintf('  Found %d time points in window\n', n_timepoints_window);
    
    % Step 1: Compute average over all trials first (baseline) - only for time window
    fprintf('  Computing baseline (average over all trials in time window)...\n');
    % Use cellfun to extract and average time window for all trials
    trial_data_windows = cellfun(@(trial_data) mean(trial_data(:, time_idx), 2, 'omitnan'), ...
                                sourcedata.trial, 'UniformOutput', false);
    all_trial_data = [trial_data_windows{:}]; % Concatenate all trials
    baseline = mean(all_trial_data, 2); % Average across all trials
    
    % Step 2: Divide each trial by baseline and compute relative power
    fprintf('  Computing relative power for each target location...\n');
    
    % Initialize power data matrix
    subject_power_data = zeros(n_sources, 10);
    
    for target = 1:10
        % Find trials for this target
        target_trials = find(target_info == target);
        
        if ~isempty(target_trials)
            fprintf('    Processing target %d (%d trials)...\n', target, length(target_trials));
            
            % Compute relative power for each trial of this target using cellfun
            target_trial_data = sourcedata.trial(target_trials);
            relative_power_trials = cellfun(@(trial_data) mean(trial_data(:, time_idx), 2, 'omitnan') ./ baseline, ...
                                           target_trial_data, 'UniformOutput', false);
            relative_power_matrix = [relative_power_trials{:}]; % Concatenate all trials
            
            % Step 3: Average for this location (across trials)
            subject_power_data(:, target) = mean(relative_power_matrix, 2);
        else
            fprintf('    No trials found for target %d\n', target);
            subject_power_data(:, target) = NaN;
        end
    end
    
    % Create individual subject visualization
    fprintf('Creating individual subject visualization...\n');
    
    % Calculate symmetric color limits around 1.0 (baseline)
    all_power_values = subject_power_data(:);
    all_power_values = all_power_values(~isnan(all_power_values)); % Remove NaN values
     % data_mean = mean(all_power_values);
     phigh = quantile(all_power_values, 0.99);
     plow = quantile(all_power_values, 0.01);
     max_deviation = max(abs(phigh - 1), abs(plow - 1));
    color_min = 1.0 - max_deviation;
    color_max = 1.0 + max_deviation;
    fprintf('  Color range for subject (symmetric around 1.0): %.3f to %.3f (deviation: %.3f)\n', color_min, color_max, max_deviation);
    
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
    
    % Define target locations and their angles
    target_locations = 1:10;
    target_angles = [0, 25, 50, 130, 155, 180, 205, 230, 310, 335]; % degrees
    n_targets = length(target_locations);
    
    %% Figure 1: MNI Template Surface Visualization
    fprintf('Creating MNI template surface visualization...\n');
    
    figure('Position', [100, 400, 1500, 1000], 'Renderer', 'painters');
    
    % Create circular subplot layout
    for i = 1:n_targets
        target = target_locations(i);
        
        % Calculate position for circular layout
        angle_rad = deg2rad(target_angles(i));
        radius = 0.35;
        x_pos = 0.5 + radius * cos(angle_rad);
        y_pos = 0.5 + radius * sin(angle_rad);
        subplot_size = 0.12;
        
        % Create subplot at calculated position
        subplot('Position', [x_pos - subplot_size/2, y_pos - subplot_size/2, subplot_size, subplot_size]);
        
        % Get relative power data for this target
        relative_power_data = subject_power_data(:, target);
        
        % Create source structure for visualization
        sourceVisualize = struct();
        sourceVisualize.pos = template_mesh.pos;
        sourceVisualize.tri = template_mesh.tri;
        sourceVisualize.unit = 'mm';
        sourceVisualize.coordsys = 'mni';
        sourceVisualize.pow = relative_power_data;
        
        % Use ft_sourceplot for surface visualization
        cfg = [];
        cfg.method = 'surface';
        cfg.figure = 'gcf';
        cfg.funparameter = 'pow';
        cfg.maskparameter = "";
        cfg.surffile = sprintf('cortex_%d.surf.gii', surface_resolution);
        cfg.colorbar = 'yes';
        cfg.funcolormap = 'jet';
        cfg.funcolorlim = [color_min, color_max];
        ft_sourceplot(cfg, sourceVisualize);
        
        % Set view angle and lighting
        view(0, 40);
        lighting gouraud;
        material dull;
        light('Position', [-1, -1, 1], 'Style', 'infinite', 'Color', [0.4, 0.4, 0.4]);
        
        % Labels and title
        title(sprintf('Target %d\n%d°', target, target_angles(i)));
        xlabel('X (mm)');
        ylabel('Y (mm)');
        zlabel('Z (mm)');
        
        % Set axis properties
        axis equal;
        axis tight;
        grid on;
    end
    
    % Overall title
    sgtitle(sprintf('Subject %02d: Beta Relative Power on Template Midthickness Surface (Targets 1-10, %d vertices)', subjID, surface_resolution));
    
    % % Save figure
    % fig_name = sprintf('sub-%02d_betaPower_cortex_%d', subjID, surface_resolution);
    
    % saveas(gcf, fullfile(figures_dir, [fig_name, '.fig']));
    % saveas(gcf, fullfile(figures_dir, [fig_name, '.png']));
    % fprintf('Saved: %s\n', fig_name);
    
    %% Figure 2: Native Subject Space Scatter Plot (only for individual subjects)
    % Check for the correct source model field based on surface resolution
    sourcemodel_field = sprintf('sourcemodel_aligned_%d', surface_resolution);
    if exist('loaded_forward', 'var') && isfield(loaded_forward, sourcemodel_field)
        fprintf('Creating native subject space scatter plot...\n');
        sourcemodel_subject = loaded_forward.(sourcemodel_field);
        fprintf('  Source model points: %d\n', size(sourcemodel_subject.pos, 1));
        fprintf('  Power data dimensions: %s\n', mat2str(size(subject_power_data)));
        fprintf('  Using symmetric color range around 1.0: %.3f to %.3f\n', color_min, color_max);
        
        figure('Position', [200, 500, 1500, 1000], 'Renderer', 'painters');
        
        % Create circular subplot layout for native space
        for i = 1:n_targets
            target = target_locations(i);
            
            % Calculate position for circular layout
            angle_rad = deg2rad(target_angles(i));
            radius = 0.35;
            x_pos = 0.5 + radius * cos(angle_rad);
            y_pos = 0.5 + radius * sin(angle_rad);
            subplot_size = 0.12;
            
            % Create subplot at calculated position
            subplot('Position', [x_pos - subplot_size/2, y_pos - subplot_size/2, subplot_size, subplot_size]);
            
            % Get relative power data for this target
            relative_power_data = subject_power_data(:, target);
            
            % Check data dimensions and ensure they match
            n_source_points = size(sourcemodel_subject.pos, 1);
            n_power_points = length(relative_power_data);
            
            fprintf('Target %d: sourcemodel.pos size: %s, relative_power_data size: %s\n', ...
                target, mat2str(size(sourcemodel_subject.pos)), mat2str(size(relative_power_data)));
            
            if n_source_points ~= n_power_points
                fprintf('  Warning: Dimension mismatch - source: %d, power: %d\n', n_source_points, n_power_points);
                
                % Interpolate power data to match source model size
                if n_power_points < n_source_points
                    % Pad with NaN if power data is smaller
                    relative_power_data = [relative_power_data; NaN(n_source_points - n_power_points, 1)];
                else
                    % Truncate if power data is larger
                    relative_power_data = relative_power_data(1:n_source_points);
                end
            end

            % Create scatter plot in native subject space
            scatter3(sourcemodel_subject.pos(:,1), sourcemodel_subject.pos(:,2), sourcemodel_subject.pos(:,3), ...
                20, relative_power_data, 'filled');

            % Set color limits for consistent scaling across all subplots
            caxis([color_min, color_max]);
            colormap('jet');
            colorbar;
            
            % Set view angle
            view(0, 40);
            
            % Labels and title
            title(sprintf('Target %d\n%d°\n(Native)', target, target_angles(i)));
            xlabel('X (mm)');
            ylabel('Y (mm)');
            zlabel('Z (mm)');
            
            % Set axis properties
            axis equal;
            axis tight;
            grid on;
        end
        
        % Overall title for native space
        sgtitle(sprintf('Subject %02d: Beta Relative Power in Native Subject Space (Targets 1-10, %d vertices)', subjID, surface_resolution));
        
        % Save figure
        % fig_name = sprintf('sub-%02d_betaPower_native_%d', subjID, surface_resolution);
        % saveas(gcf, fullfile(figures_dir, [fig_name, '.fig']));
        % saveas(gcf, fullfile(figures_dir, [fig_name, '.png']));
        % fprintf('Saved: %s\n', fig_name);
    else
        fprintf('Source model field %s not found in loaded_forward. Skipping scatter plot.\n', sourcemodel_field);
    end
    
    close all;
end  