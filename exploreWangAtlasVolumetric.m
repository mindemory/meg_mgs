% exploreWangAtlasVolumetric - Explore and visualize Wang atlas using volumetric source model
%
% This script loads the Wang atlas and the 8mm volumetric source model to visualize
% atlas regions in volumetric space using 3D scatter plots.
%
% Author: Mrugank Dake
% Date: 2025-01-24

restoredefaultpath;
clear; close all; clc;

%% Path Setup
fprintf('=== Wang Atlas Explorer - Volumetric ===\n');

% Local paths
fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';

% Add FieldTrip to path
addpath(fieldtrip_path);
addpath(genpath(project_path));
ft_defaults;

%% Load Volumetric Source Model
fprintf('Loading 5mm volumetric source model...\n');

% Load the standard 5mm volumetric source model from FieldTrip
sourcemodel_path = '/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/standard_sourcemodel3d5mm.mat';
if ~exist(sourcemodel_path, 'file')
    error('Volumetric source model not found at: %s', sourcemodel_path);
end

load(sourcemodel_path);

fprintf('Loaded 5mm volumetric source model with %d sources\n', size(sourcemodel.pos, 1));

%% Load Wang Atlas
fprintf('Loading Wang atlas...\n');

% Load the Wang atlas (VTPM - Visual Topographic Parcellation Model)
atlas_path = '/d/DATD/hyper/software/fieldtrip-20250318/template/atlas/vtpm/vtpm.mat';
% atlas_path = '/d/DATD/hyper/software/fieldtrip-20250318/template/atlas/aal/ROI_MNI_V4.nii';
if ~exist(atlas_path, 'file')
    error('Wang atlas not found at: %s', atlas_path);
end

wangatlas = ft_read_atlas(atlas_path);
% load(atlas_path);
fprintf('Loaded Wang atlas with %d regions\n', length(wangatlas.tissuelabel));

%% Define ROI Categories
fprintf('Defining ROI categories...\n');

% Visual ROIs
visual_rois = {'left_V1d', 'left_V1v', 'left_V2d', 'left_V2v', 'left_V3d', 'left_V3v', ...
               'left_V3a', 'left_V3b', 'left_hV4', 'right_V1d', 'right_V1v', 'right_V2d', 'right_V2v', ...
               'right_V3d', 'right_V3v', 'right_V3a', 'right_V3b', 'right_hV4'};

% Parietal ROIs  
parietal_rois = {'left_IPS0', 'left_IPS1', 'left_IPS2', 'left_IPS3', 'left_IPS4', 'left_IPS5', ...
                 'right_IPS0', 'right_IPS1', 'right_IPS2', 'right_IPS3', 'right_IPS4', 'right_IPS5'};

% Frontal ROIs
frontal_rois = {'left_FEF', 'right_FEF'};

fprintf('Visual ROIs: %d\n', length(visual_rois));
fprintf('Parietal ROIs: %d\n', length(parietal_rois));
fprintf('Frontal ROIs: %d\n', length(frontal_rois));

%% Find Region IDs for Each Category
fprintf('Finding region IDs...\n');

visual_region_ids = [];
parietal_region_ids = [];
frontal_region_ids = [];

% Find region IDs for visual ROIs
for i = 1:length(visual_rois)
    roi_name = visual_rois{i};
    for region_id = 1:length(vtpm.tissuelabel)
        if strcmp(vtpm.tissuelabel{region_id}, roi_name)
            visual_region_ids = [visual_region_ids, region_id];
            break;
        end
    end
end

% Find region IDs for parietal ROIs
for i = 1:length(parietal_rois)
    roi_name = parietal_rois{i};
    for region_id = 1:length(vtpm.tissuelabel)
        if strcmp(vtpm.tissuelabel{region_id}, roi_name)
            parietal_region_ids = [parietal_region_ids, region_id];
            break;
        end
    end
end

% Find region IDs for frontal ROIs
for i = 1:length(frontal_rois)
    roi_name = frontal_rois{i};
    for region_id = 1:length(vtpm.tissuelabel)
        if strcmp(vtpm.tissuelabel{region_id}, roi_name)
            frontal_region_ids = [frontal_region_ids, region_id];
            break;
        end
    end
end

fprintf('Found %d visual regions, %d parietal regions, %d frontal regions\n', ...
        length(visual_region_ids), length(parietal_region_ids), length(frontal_region_ids));

% Debug: Print the region IDs found
fprintf('Visual region IDs: %s\n', mat2str(visual_region_ids));
fprintf('Parietal region IDs: %s\n', mat2str(parietal_region_ids));
fprintf('Frontal region IDs: %s\n', mat2str(frontal_region_ids));

%% Create Atlas Visualization on Volumetric Grid
fprintf('Creating volumetric atlas visualization...\n');

% Check coordinate systems
fprintf('Atlas coordinate system: %s\n', vtpm.coordsys);
fprintf('Atlas unit: %s\n', vtpm.unit);
if isfield(sourcemodel, 'coordsys')
    fprintf('Source model coordinate system: %s\n', sourcemodel.coordsys);
else
    fprintf('Source model coordinate system: not specified\n');
end
if isfield(sourcemodel, 'unit')
    fprintf('Source model unit: %s\n', sourcemodel.unit);
else
    fprintf('Source model unit: not specified\n');
end

% Convert source model to mm if it's in cm
if strcmp(sourcemodel.unit, 'cm')
    fprintf('Converting source model from cm to mm...\n');
    sourcemodel = ft_convert_units(sourcemodel, 'mm');
end

% Create a source structure for visualization
sourceVisualize = struct();
sourceVisualize.pos = sourcemodel.pos;
sourceVisualize.inside = sourcemodel.inside;
sourceVisualize.unit = 'mm';
sourceVisualize.coordsys = 'mni';


sourcemodel_mm = ft_convert_units(sourcemodel, 'mm');
% Interpolate atlas to volumetric source model
cfg = [];
cfg.interpmethod = 'nearest';
cfg.parameter = 'tissue';
% sourceVisualize_interpolated = ft_sourceinterpolate(cfg, vtpm, sourceVisualize);
sourceVisualize_interpolated = ft_sourceinterpolate(cfg, wangatlas, sourcemodel_mm);

% Check if interpolation worked
fprintf('Interpolated tissue values: %d non-zero values out of %d total\n', ...
        sum(sourceVisualize_interpolated.tissue > 0), length(sourceVisualize_interpolated.tissue));

% Debug: Check the atlas dimensions
fprintf('Atlas dimensions: %s\n', mat2str(size(vtpm.tissue)));
fprintf('Atlas transform matrix:\n');
disp(vtpm.transform);

fprintf('Source model coordinate range: X=[%.1f, %.1f], Y=[%.1f, %.1f], Z=[%.1f, %.1f]\n', ...
        min(sourcemodel.pos(:,1)), max(sourcemodel.pos(:,1)), ...
        min(sourcemodel.pos(:,2)), max(sourcemodel.pos(:,2)), ...
        min(sourcemodel.pos(:,3)), max(sourcemodel.pos(:,3)));

% Debug: Check unique tissue values
unique_tissues = unique(sourceVisualize_interpolated.tissue);
fprintf('Unique tissue values found: %s\n', mat2str(unique_tissues));
fprintf('Number of unique tissue values: %d\n', length(unique_tissues));

% Create masks for each category
visual_mask = ismember(sourceVisualize_interpolated.tissue, visual_region_ids);
parietal_mask = ismember(sourceVisualize_interpolated.tissue, parietal_region_ids);
frontal_mask = ismember(sourceVisualize_interpolated.tissue, frontal_region_ids);

% Create ROI data vector (0 = background, 1 = visual, 2 = parietal, 3 = frontal)
roi_data = zeros(size(sourceVisualize_interpolated.tissue));
roi_data(visual_mask) = 1;    % Visual
roi_data(parietal_mask) = 2;  % Parietal
roi_data(frontal_mask) = 3;   % Frontal

% Add to source structure
sourceVisualize_interpolated.roi_data = roi_data;

%% Create 3D Visualization
fprintf('Creating 3D visualization...\n');

% Create figure
figure('Position', [100, 100, 1200, 800], 'Name', 'Wang Atlas - Volumetric Visualization');

% Define colors for each category
colors = [0.0 0.0 0.0;    % Background (black)
          1.0 0.0 0.0;    % Visual (red)
          0.0 0.0 1.0;    % Parietal (blue)
          0.0 0.8 0.0];   % Frontal (green)

% Get positions for each category
visual_pos = sourceVisualize_interpolated.pos(visual_mask, :);
parietal_pos = sourceVisualize_interpolated.pos(parietal_mask, :);
frontal_pos = sourceVisualize_interpolated.pos(frontal_mask, :);

% Get non-ROI positions (background) - only consider inside sources
non_roi_mask = ~(visual_mask | parietal_mask | frontal_mask) & sourcemodel.inside;
non_roi_pos = sourceVisualize_interpolated.pos(non_roi_mask, :);

% Plot non-ROI sources in black
if ~isempty(non_roi_pos)
    scatter3(non_roi_pos(:, 1), non_roi_pos(:, 2), non_roi_pos(:, 3), 10, colors(1, :), 'filled', 'DisplayName', 'Non-ROI');
    hold on;
end

% Plot Visual ROIs in red
if ~isempty(visual_pos)
    scatter3(visual_pos(:, 1), visual_pos(:, 2), visual_pos(:, 3), 20, colors(2, :), 'filled', 'DisplayName', 'Visual');
    hold on;
end

% Plot Parietal ROIs in blue
if ~isempty(parietal_pos)
    scatter3(parietal_pos(:, 1), parietal_pos(:, 2), parietal_pos(:, 3), 20, colors(3, :), 'filled', 'DisplayName', 'Parietal');
    hold on;
end

% Plot Frontal ROIs in green
if ~isempty(frontal_pos)
    scatter3(frontal_pos(:, 1), frontal_pos(:, 2), frontal_pos(:, 3), 20, colors(4, :), 'filled', 'DisplayName', 'Frontal');
    hold on;
end

title('Wang Atlas ROIs on 5mm Volumetric Grid', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
axis equal; grid on; view(3);
legend('Location', 'best');

%% Print Summary Statistics
fprintf('\n=== Summary Statistics ===\n');
fprintf('Total volumetric sources: %d\n', size(sourcemodel.pos, 1));
fprintf('Visual ROI voxels: %d\n', sum(visual_mask));
fprintf('Parietal ROI voxels: %d\n', sum(parietal_mask));
fprintf('Frontal ROI voxels: %d\n', sum(frontal_mask));
fprintf('Total atlas voxels: %d\n', sum(visual_mask | parietal_mask | frontal_mask));

fprintf('\nVisualization complete!\n');
