function S01A_VolSources2SubSpace(subjID)
% S01A_VolSources2SubSpace - Transform volumetric sources to subject space
%
% This script loads the forward model data from S01_ForwardModelMNI and
% transforms volumetric sources (3d8mm and 3d10mm) to subject space
% for visualization.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, etc.)
%
% Outputs:
%   Saves transformed source data and creates visualization plots
%
% Example:
%   S01A_VolSources2SubSpace(1);
%   S01A_VolSources2SubSpace(2);

%% Environment Detection and Path Setup
% Detect if running on HPC or local machine
[~, hostname] = system('hostname');
hostname = strtrim(hostname);

% Check for common HPC indicators
is_hpc = contains(hostname, {'login', 'compute', 'node', 'hpc'}) || ...
    exist('/etc/slurm', 'dir') || ...
    ~isempty(getenv('SLURM_JOB_ID')) || ...
    ~isempty(getenv('PBS_JOBID'));

fprintf('=== MEG Volumetric Sources to Subject Space Transformation ===\n');
fprintf('Environment: %s\n', hostname);
fprintf('Detected HPC: %s\n', string(is_hpc));
fprintf('Subject: %d\n', subjID);

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
    ft_gifti_path = '/d/DATD/hyper/software/fieldtrip-20250318/external/gifti';
end

% Add FieldTrip to path
addpath(fieldtrip_path);
ft_defaults;

% Add Gifti toolbox for .surf.gii files
addpath(ft_gifti_path);

addpath(genpath(project_path));

%% Verify paths exist
if ~exist(fieldtrip_path, 'dir')
    error('FieldTrip path not found: %s', fieldtrip_path);
end

if ~exist(data_base_path, 'dir')
    error('Data base path not found: %s', data_base_path);
end

%% Define file paths
subj_str = sprintf('sub-%02d', subjID);
subj_dir = fullfile(data_base_path, subj_str, 'sourceRecon');

% Forward model file
forward_model_path = fullfile(subj_dir, sprintf('%s_task-mgs_forwardModel.mat', subj_str));

% Check if forward model exists
if ~exist(forward_model_path, 'file')
    error('Forward model not found at: %s', forward_model_path);
end

%% Load forward model data
fprintf('Loading forward model data...\n');
load(forward_model_path, 'transformApplied', 'mri_aligned', 'singleShellHeadModel', ...
    'gradData', 'subject_fid', 'hspCorrected_ras');

fprintf('Forward model loaded successfully!\n');
fprintf('Transform matrix loaded: %s\n', mat2str(size(transformApplied)));

%% Define volumetric resolutions to process
volumetric_resolutions = [5, 8, 10]; % mm

%% Process each volumetric resolution
for vol_res = 1:length(volumetric_resolutions)
    current_vol_res = volumetric_resolutions(vol_res);
    fprintf('\n=== Processing %dmm resolution ===\n', current_vol_res);

    % Create output file path
    output_file = fullfile(subj_dir, sprintf('%s_task-mgs_volumetricSources_%dmm.mat', ...
        subj_str, current_vol_res));

    % Check if already processed
    if exist(output_file, 'file')
        fprintf('Volumetric sources already processed for %dmm resolution\n', current_vol_res);
        continue;
    end

    %% Load FieldTrip volumetric source model from template
    fprintf('Loading FieldTrip volumetric source model for %dmm...\n', current_vol_res);

    % Load FieldTrip volumetric source model from template directory
    if is_hpc
        sourcemodel_path = sprintf('/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/template/sourcemodel/standard_sourcemodel3d%dmm.mat', current_vol_res);
    else
        sourcemodel_path = sprintf('/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/standard_sourcemodel3d%dmm.mat', current_vol_res);
    end
    load(sourcemodel_path);
    sourcemodel = ft_convert_units(sourcemodel, 'mm');
    % sourcemodel.unit = 'mm';

    fprintf('Loaded %d grid points for %dmm resolution\n', size(sourcemodel.pos, 1), current_vol_res);

    %% Transform to subject space
    fprintf('Transforming volumetric sources to subject space...\n');

    % Use FieldTrip's transformation function (same as S01)
    sourcemodel = ft_transform_geometry(transformApplied, sourcemodel);

    fprintf('Transformed %d points to subject space\n', size(sourcemodel.pos, 1));


    %% Save volumetric source data
    fprintf('Saving volumetric source data...\n');
    save(output_file, 'sourcemodel', 'current_vol_res', ...
        'sourcemodel', '-v7.3');

    fprintf('Saved volumetric sources to: %s\n', output_file);

end

%% Create summary visualization
fprintf('\nCreating summary visualization...\n');

% Load all processed volumetric sources
all_sources = struct();
for vol_res = 1:length(volumetric_resolutions)
    current_vol_res = volumetric_resolutions(vol_res);
    output_file = fullfile(subj_dir, sprintf('%s_task-mgs_volumetricSources_%dmm.mat', ...
        subj_str, current_vol_res));

    if exist(output_file, 'file')
        load(output_file);
        field_name = sprintf('res_%dmm', current_vol_res);
        all_sources.(field_name) = sourcemodel;
    end
end

% Create comparison figure (same style as S01)
figure('Name', 'MEG Volumetric Sources Setup', 'Position', [100, 100, 1600, 800]);

for vol_res = 1:length(volumetric_resolutions)
    current_vol_res = volumetric_resolutions(vol_res);
    field_name = sprintf('res_%dmm', current_vol_res);

    if isfield(all_sources, field_name)
        subplot(1, length(volumetric_resolutions), vol_res);

        % Plot headshape (if available)
        if exist('hspCorrected_ras', 'var')
            ft_plot_headshape(hspCorrected_ras, 'vertexcolor', 'k');
            hold on;
        end

        % Plot head model
        ft_plot_mesh(singleShellHeadModel.bnd, 'facecolor', 'brain', 'facealpha', 0.3);
        hold on;

        % Plot volumetric sources as points
        plot3(all_sources.(field_name).pos(:, 1), ...
            all_sources.(field_name).pos(:, 2), ...
            all_sources.(field_name).pos(:, 3), ...
            'r.', 'MarkerSize', 5);

        % Plot gradiometers (if available)
        if exist('gradData', 'var')
            ft_plot_sens(gradData, 'facecolor', 'green', 'facealpha', 1);
        end

        % Plot fiducials (if available)
        if exist('subject_fid', 'var')
            plot3(subject_fid.pos(:,1), subject_fid.pos(:,2), subject_fid.pos(:,3), ...
                'bs', 'MarkerSize', 10, 'LineWidth', 2);
        end

        title(sprintf('Volumetric Sources - %dmm (%d points)', current_vol_res, size(all_sources.(field_name).pos, 1)));
        xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
        axis equal; grid on;
    end
end

sgtitle(sprintf('MEG Volumetric Sources Setup (Subject %d)', subjID), ...
    'FontSize', 16, 'FontWeight', 'bold');
end
