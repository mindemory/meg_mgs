function Fs02_visualizeWangAtlas()
% Fs02_visualizeWangAtlas - Visualize Wang Atlas ROIs in 8mm space
%
% This script loads the saved visual, parietal, and frontal points from the
% 8mm volumetric space and creates comprehensive visualizations of the
% Wang atlas regions.
%
% Outputs:
%   - 3D scatter plots showing ROI distributions
%   - Separate plots for left/right hemispheres
%   - Group-level and individual ROI visualizations
%   - Saves figures as SVG format
%
% Dependencies:
%   - Saved ROI files from exploreWangAtlasVolumetric.m (8mm space)
%
% Author: Mrugank Dake
% Date: 2025-01-20

clearvars;
close all; clc;

%% Path Setup - Auto-detect HPC vs Local
fprintf('=== Wang Atlas Visualization (8mm Space) ===\n');

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

%% Load Source Model and ROI Data
fprintf('Loading source model and ROI data...\n');

% Load 8mm source model
sourcemodel_path = sprintf('%s/sub-01/sourceRecon/sub-01_task-mgs_volumetricSources_8mm.mat', data_base_path);
if ~exist(sourcemodel_path, 'file')
    error('Source model file not found: %s', sourcemodel_path);
end
load(sourcemodel_path, 'sourcemodel');

% Load ROI points from 8mm space
roi_path = sprintf('%s/atlas/rois_8mm.mat', data_base_path);
if ~exist(roi_path, 'file')
    error('ROI file not found: %s', roi_path);
end
load(roi_path, 'visual_points', 'parietal_points', 'frontal_points', ...
     'left_visual_points', 'right_visual_points', ...
     'left_parietal_points', 'right_parietal_points', ...
     'left_frontal_points', 'right_frontal_points');

fprintf('Loaded source model with %d inside points\n', sum(sourcemodel.inside));
fprintf('Visual points: %d\n', sum(visual_points));
fprintf('Parietal points: %d\n', sum(parietal_points));
fprintf('Frontal points: %d\n', sum(frontal_points));

%% Get Source Model Positions
sm_pos = sourcemodel.pos(sourcemodel.inside, :);

%% Define Colors
group_color = [0 0.447 0.741;  % blue  - visual
               0.85 0.33 0.1;  % red   - parietal
               0.47 0.67 0.19];% green - frontal

hemisphere_color = [0.8 0.2 0.2;  % red   - left
                    0.2 0.2 0.8]; % blue  - right

%% Create Figure Directory
figures_dir = fullfile(data_base_path, 'figures', 'Fs02');
if ~exist(figures_dir, 'dir')
    mkdir(figures_dir);
end

%% Figure 1: Wang Atlas Visualization
fprintf('Creating Wang atlas visualization...\n');
figure('Position', [100, 100, 600, 600], 'Renderer', 'painters');

hold on;
% Background points
scatter3(sm_pos(:,1), sm_pos(:,2), sm_pos(:,3), 5, [0.7 0.7 0.7], 'filled');
% Visual group
scatter3(sm_pos(visual_points,1), sm_pos(visual_points,2), sm_pos(visual_points,3), 20, group_color(1,:), 'filled');
% Parietal group
scatter3(sm_pos(parietal_points,1), sm_pos(parietal_points,2), sm_pos(parietal_points,3), 20, group_color(2,:), 'filled');
% Frontal group
scatter3(sm_pos(frontal_points,1), sm_pos(frontal_points,2), sm_pos(frontal_points,3), 20, group_color(3,:), 'filled');
axis equal; view(3);
axis off;
title('Wang Atlas ROIs - 8mm Space', 'FontSize', 16);
legend({'Other', 'Visual', 'Parietal', 'Frontal'}, 'Location', 'best');

% Save as SVG
saveas(gcf, fullfile(figures_dir, 'wang_atlas_groups_8mm.svg'));

%% Print Summary
fprintf('\n=== Wang Atlas Visualization Complete ===\n');
fprintf('Figure saved to: %s\n', figures_dir);
fprintf('File created: wang_atlas_groups_8mm.svg\n');
fprintf('\nROI Summary:\n');
fprintf('  Visual: %d points (%.1f%%)\n', sum(visual_points), 100*sum(visual_points)/sum(sourcemodel.inside));
fprintf('  Parietal: %d points (%.1f%%)\n', sum(parietal_points), 100*sum(parietal_points)/sum(sourcemodel.inside));
fprintf('  Frontal: %d points (%.1f%%)\n', sum(frontal_points), 100*sum(frontal_points)/sum(sourcemodel.inside));

end
