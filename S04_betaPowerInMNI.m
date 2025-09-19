function S04_betaPowerInMNI(subjID, surface_resolution, displayFigure)
%% MEG Beta Power Analysis in MNI Space
% Load complex beta data, compute power, lateralization, and visualize in MNI space
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%   surface_resolution - Surface resolution (default: 5124)
%   displayFigure - Display figures (true/false, default: false)
%
% Example:
%   S04_betaPowerInMNI(1, 5124, false)

if nargin < 1
    error('Subject ID is required');
end
if nargin < 2
    surface_resolution = 5124; % Default resolution
end
if nargin < 3
    displayFigure = true; % Default: don't display figures
end

restoredefaultpath;
clearvars -except subjID surface_resolution displayFigure; % Keep inputs
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

fprintf('=== MEG Beta Power Analysis in MNI Space ===\n');
fprintf('Environment: %s\n', hostname);
fprintf('Detected HPC: %s\n', string(is_hpc));
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Setup paths based on environment
if is_hpc
    % HPC paths
    fieldtrip_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/';
    project_path = '/scratch/mdd9787/meg_prf_greene/megScripts';
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
    ft_gifti_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/external/gifti';
else
    % Local machine paths
    fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    ft_gifti_path = '/d/DATD/hyper/software/fieldtrip-20250318/external/gifti'; % Add Gifti toolbox for .surf.gii files

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
addpath(ft_gifti_path);
addpath(genpath(project_path));
ft_defaults;
ft_hastoolbox('spm12', 1);

%% Initialize File Paths
% Complex beta data path (input)
beta_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_complexBeta_allTargets_%d.mat', subjID, surface_resolution));

% Forward model path (input)
forward_model_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_forwardModel.mat', subjID));

% Output directory and file paths
output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon');
betaPow_data_path = fullfile(output_dir, sprintf('sub-%02d_task-mgs_relativePowerBetaBand_%d.mat', subjID, surface_resolution));

% Check if output file already exists
if exist(betaPow_data_path, 'file')
    fprintf('MNI space analysis data already exists at: %s\n', betaPow_data_path);
    fprintf('Skipping processing to avoid overwriting existing data.\n');
    fprintf('To reprocess, delete the existing file first.\n');
    return;
else


    %% Load Complex Beta Data
    % Load the complex beta data created by S03_betaPowerInSource.m
    if ~exist(beta_data_path, 'file')
        error('Complex beta data not found at: %s\nPlease run S03_betaPowerInSource.m first!', beta_data_path);
    end

    fprintf('Loading complex beta data from: %s\n', beta_data_path);
    load(beta_data_path);

    fprintf('Loaded complex beta data for %d targets\n', length(target_locations));

    %% Load Forward Model for MNI Space Information
    % Load the forward model to get MNI space coordinates and source model
    if ~exist(forward_model_path, 'file')
        error('Forward model not found at: %s\nPlease run S01_ForwardModelMNI.m first!', forward_model_path);
    end

    fprintf('Loading forward model from: %s\n', forward_model_path);
    load(forward_model_path);

    %% Compute Global Average Power Across All Targets
    fprintf('Computing global average power across all targets...\n');

    % Define target locations (1-10)
    target_locations = 1:10;
    all_power_data = [];

    % Collect power data from all targets
    for target = target_locations
        if isempty(sourceDataByTarget{target})
            fprintf('  No data for target %d\n', target);
            continue;
        end

        fprintf('  Collecting power data from target %d...\n', target);

        % Get the complex beta data for this target
        sourcedataTarget = sourceDataByTarget{target};

        % Compute power (magnitude squared of complex signal) for all trials
        powerAll = sourcedataTarget;
        powerAll.trial = cellfun(@(x) abs(x).^2, sourcedataTarget.trial, 'UniformOutput', false);

        % Average across trials
        cfg = [];
        cfg.keeptrials = 'no';
        powerAll_avg = ft_timelockanalysis(cfg, powerAll);

        % Store power data for this target
        all_power_data = cat(3, all_power_data, powerAll_avg.avg);
    end

    % Compute global average power across all targets and all locations
    fprintf('Computing global average power...\n');
    global_avg_power_time = mean(all_power_data, 2, 'omitnan'); % Average across targets for each time point

    fprintf('Global average power computed across %d targets\n', size(all_power_data, 3));

    %% Process Each Target Location for Relative Power
    fprintf('Processing each target location for relative power...\n');

    powerDataByTarget = cell(10, 1);

    for target = target_locations
        fprintf('Processing target location %d...\n', target);

        if isempty(sourceDataByTarget{target})
            fprintf('  No data for target %d\n', target);
            continue;
        end

        % Get the complex beta data for this target
        sourcedataTarget = sourceDataByTarget{target};

        % Compute power (magnitude squared of complex signal) for all trials
        fprintf('  Computing power for all trials...\n');

        % Compute power for all trials
        powerAll = sourcedataTarget;
        powerAll.trial = cellfun(@(x) abs(x).^2, sourcedataTarget.trial, 'UniformOutput', false);

        % Average across trials
        cfg = [];
        cfg.keeptrials = 'no';
        powerAll_avg = ft_timelockanalysis(cfg, powerAll);

        % Compute relative power: power at each location / global average power
        fprintf('  Computing relative power...\n');

        % Compute relative power for each location and time point
        relative_power_all = powerAll_avg.avg ./ global_avg_power_time;

        % Define time windows for analysis (-1.5 to 1.7s with 200ms steps)
        time_start = -1.5;
        time_end = 1.7;
        time_step = 0.2; % 200ms
        time_windows = time_start:time_step:(time_end - time_step);

        % Compute relative power for each time window
        relative_power_windows = zeros(size(relative_power_all, 1), length(time_windows), 'single');

        for tw = 1:length(time_windows)
            window_start = time_windows(tw);
            window_end = window_start + time_step;

            % Find time indices for this window
            time_idx = powerAll_avg.time >= window_start & powerAll_avg.time < window_end;

            if sum(time_idx) > 0
                % Average relative power across this time window
                relative_power_windows(:, tw) = single(mean(relative_power_all(:, time_idx), 2, 'omitnan'));
            else
                % No data in this window, set to NaN
                relative_power_windows(:, tw) = single(NaN);
            end
        end

        % Store relative power data in single precision for efficiency
        powerDataByTarget{target} = struct();
        powerDataByTarget{target}.relative_power_windows = relative_power_windows;
        powerDataByTarget{target}.time_windows = single(time_windows);
        powerDataByTarget{target}.time_step = single(time_step);

        fprintf('  Target %d processing complete\n', target);
    end

    fprintf('Relative power analysis complete for all targets\n');

     %% Create MNI Space Visualization (Local Only)
     if ~is_hpc && displayFigure
        fprintf('Creating MNI space visualization on template midthickness surface...\n');

        % Load FieldTrip's standard MNI template surface for visualization
        template_mesh = ft_read_headshape(sprintf('cortex_%d.surf.gii', surface_resolution));

        % Use the subject-specific source model for data interpolation
        sourcemodel_subject = sourcemodel_aligned_5124; % Subject-specific source model

        % Create figure for all targets
        figure('Position', [100, 400, 1500, 1000]);

        % Define target locations and their angles
        target_angles = [0, 25, 50, 130, 155, 180, 205, 230, 310, 335]; % degrees
        n_targets = length(target_locations);

        % Create circular subplot layout
        for i = 1:n_targets
            target = target_locations(i);

            if isempty(powerDataByTarget{target})
                continue;
            end

            % Calculate position for circular layout
            angle_rad = deg2rad(target_angles(i));
            radius = 0.35; % Distance from center (reduced to keep all subplots visible)
            x_pos = 0.5 + radius * cos(angle_rad); % Center around 0.5
            y_pos = 0.5 + radius * sin(angle_rad); % Center around 0.5
            subplot_size = 0.12; % Size of each subplot (reduced)

            % Debug: print positions for targets 2 and 4
            if target == 2 || target == 4
                fprintf('Target %d: angle=%.1f°, x=%.3f, y=%.3f\n', target, target_angles(i), x_pos, y_pos);
            end

            % Create subplot at calculated position
            subplot('Position', [x_pos - subplot_size/2, y_pos - subplot_size/2, subplot_size, subplot_size]);

            % Get relative power data for time window 0.8-1.5s
            relative_power_windows = powerDataByTarget{target}.relative_power_windows;
            time_windows = powerDataByTarget{target}.time_windows;

            % Find time windows between 0.8 and 1.5 seconds
            target_window_start = 0.8;
            target_window_end = 1.5;
            window_indices = find(time_windows >= target_window_start & time_windows <= target_window_end);

            if isempty(window_indices)
                fprintf('  Warning: No time windows found between %.1f and %.1f s for target %d\n', ...
                    target_window_start, target_window_end, target);
                continue;
            end

            % Average relative power across the time window
            relative_power_data = mean(relative_power_windows(:, window_indices), 2, 'omitnan');

            % % Interpolate data from subject source model to standard MNI template
            % % Find nearest neighbors between subject and template surfaces
            % [~, nearest_indices] = pdist2(template_mesh.pos, sourcemodel_subject.pos, 'euclidean', 'Smallest', 1);
            %
            % % Interpolate relative power data to template surface
            % relative_power_interpolated = relative_power_data(nearest_indices);

            % Create source visualization structure for this target (standard MNI template)
            sourceVisualize = struct();
            sourceVisualize.pos = template_mesh.pos;
            sourceVisualize.tri = template_mesh.tri;
            sourceVisualize.unit = 'mm';
            sourceVisualize.coordsys = 'mni';
            sourceVisualize.pow = relative_power_data; % Use original data (will need interpolation)

            % Use ft_sourceplot for proper surface visualization
            cfg = [];
            cfg.method = 'surface';
            cfg.figure = 'gcf';
            cfg.funparameter = 'pow';
            cfg.maskparameter = ""; %cfg.funparameter;
            cfg.surffile = sprintf('cortex_%d.surf.gii', surface_resolution);
            % cfg.surfdownsample = 10;
            cfg.colorbar = 'yes';
            cfg.funcolormap = '*RdBu';
            cfg.funcolorlim = [0.5, 1.5];
            ft_sourceplot(cfg, sourceVisualize);

            % Set view angle (posterior view)
            view(0, 40);

            % Adjust lighting for better visualization
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
        sgtitle(sprintf('Subject %02d: Beta Relative Power on Template Midthickness Surface (Targets 1-10)', subjID));

        %% Create Second Visualization: Native Subject Space Surface Dots
        fprintf('Creating native subject space surface dot visualization...\n');

        % Create second figure for native space visualization
        figure('Position', [200, 500, 1500, 1000]);

        % Create circular subplot layout for native space
        for i = 1:n_targets
            target = target_locations(i);

            if isempty(powerDataByTarget{target})
                continue;
            end

            % Calculate position for circular layout
            angle_rad = deg2rad(target_angles(i));
            radius = 0.35; % Distance from center
            x_pos = 0.5 + radius * cos(angle_rad); % Center around 0.5
            y_pos = 0.5 + radius * sin(angle_rad); % Center around 0.5
            subplot_size = 0.12; % Size of each subplot

            % Create subplot at calculated position
            subplot('Position', [x_pos - subplot_size/2, y_pos - subplot_size/2, subplot_size, subplot_size]);

            % Get relative power data for time window 0.8-1.5s
            relative_power_windows = powerDataByTarget{target}.relative_power_windows;
            time_windows = powerDataByTarget{target}.time_windows;

            % Find time windows between 0.8 and 1.5 seconds
            target_window_start = 0.8;
            target_window_end = 1.5;
            window_indices = find(time_windows >= target_window_start & time_windows <= target_window_end);

            if isempty(window_indices)
                continue;
            end

            % Average relative power across the time window
            relative_power_data = mean(relative_power_windows(:, window_indices), 2, 'omitnan');

            % Check data dimensions
            fprintf('Target %d: sourcemodel_subject.pos size: %s, relative_power_data size: %s\n', ...
                target, mat2str(size(sourcemodel_subject.pos)), mat2str(size(relative_power_data)));

            % Create scatter plot in native subject space
            scatter3(sourcemodel_subject.pos(:,1), sourcemodel_subject.pos(:,2), sourcemodel_subject.pos(:,3), ...
                20, relative_power_data, 'filled');

            % Set color limits for consistent scaling
            caxis([0.5, 1.5]);
            colormap('jet');
            colorbar;

            % Set view angle (posterior view)
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
        sgtitle(sprintf('Subject %02d: Beta Relative Power in Native Subject Space (Targets 1-10)', subjID));

     else
         if is_hpc
             fprintf('Skipping MNI visualization on HPC - only computing and saving data\n');
         else
             fprintf('Skipping MNI visualization - displayFigure is false\n');
         end
     end

    %% Save powerDataByTarget
    fprintf('Saving powerDataByTarget...\n');
    save(betaPow_data_path, 'powerDataByTarget', '-v7.3');
    fprintf('Saved powerDataByTarget to: %s\n', betaPow_data_path);

    % %% Save Results
    % fprintf('Saving MNI space analysis results...\n');

    % % Save power and relative power data in single precision
    % save(mni_data_path, 'powerDataByTarget', 'target_locations', '-v7.3');
    % fprintf('MNI space analysis saved to: %s\n', mni_data_path);

    % % Print summary
    % fprintf('Summary of MNI space analysis:\n');
    % fprintf('  Time range: %.1f to %.1f s\n', time_start, time_end);
    % fprintf('  Time step: %.1f s (%.0f ms)\n', time_step, time_step*1000);
    % fprintf('  Number of time windows: %d\n', length(time_windows));
    % for target = target_locations
    %     if ~isempty(powerDataByTarget{target}) && isfield(powerDataByTarget{target}, 'sourceVisualize')
    %         fprintf('  Target %d: Relative power computed for %d time windows\n', target, size(powerDataByTarget{target}.relative_power_windows, 2));
    %     else
    %         fprintf('  Target %d: No data\n', target);
    %     end
    % end

    % fprintf('\nMNI space relative power analysis complete!\n');
end