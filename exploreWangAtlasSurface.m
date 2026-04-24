

restoredefaultpath;
clear; close all; clc;
fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
ft_gifti_path = '/d/DATD/hyper/software/fieldtrip-20250318/external/gifti';


% Add FieldTrip to path
addpath(fieldtrip_path);
ft_defaults;

% Add Gifti toolbox for .surf.gii files
addpath(ft_gifti_path);

addpath(genpath(project_path));

% Load inflated surface (for visualization) and pial surface (for coordinate matching)
inflated_file = '/d/DATD/hyper/software/fieldtrip-20250318/template/anatomy/surface_inflated_both.mat';
pial_file = '/d/DATD/hyper/software/fieldtrip-20250318/template/anatomy/surface_pial_both.mat';

if ~exist(inflated_file, 'file') || ~exist(pial_file, 'file')
    error('Inflated or Pial surface files not found.');
end

mesh_inflated = load(inflated_file); mesh_inflated = mesh_inflated.mesh;
mesh_pial = load(pial_file); mesh_pial = mesh_pial.mesh;

% Use inflated mesh as our template for visualization
template_mesh = mesh_inflated;

fprintf('Loaded inflated surface with %d vertices.\n', size(template_mesh.pos, 1));

%%
atlas_path = '/d/DATD/hyper/software/fieldtrip-20250318/template/atlas/vtpm/vtpm.mat';
wangatlas = ft_read_atlas(atlas_path);
%%
% Positions of surface vertices (Nx3) - WE USE PIAL FOR KNN SEARCH!
% The inflated mesh distances are distorted, pial mesh matches MNI volume coordinates.
sm_pos = mesh_pial.pos;

% Atlas labeled voxels & their labels
[ind_x, ind_y, ind_z] = ind2sub(size(wangatlas.tissue), find(wangatlas.tissue > 0));
V = [ind_x, ind_y, ind_z, ones(numel(ind_x),1)];
mm_coords = (wangatlas.transform * V')';  % Mx4
mm_coords = mm_coords(:,1:3);
label_atlas = double(wangatlas.tissue(wangatlas.tissue > 0)); % region index at each voxel

% Nearest neighbor mapping from each vertex to nearest labeled atlas voxel
[idx_nearest, dist] = knnsearch(mm_coords, sm_pos);

% Each vertex's assigned Wang atlas index (in tissuelabel/cell format)
sourcemodel_atlas_label = label_atlas(idx_nearest);

% Filter points too far from any labeled ROI
good = (dist < 10); % Relaxed to 15mm to properly cover deep sulci on the high-res pial surface
sourcemodel_atlas_label(~good) = 0; % 0 = unlabeled/outside cortex
fprintf('Filtered out %d vertices (distance > 15mm from any ROI)\n', sum(~good));


%%
visualROIs = {'left_V1v', 'left_V1d', 'left_V2v', 'left_V2d', 'left_V3v', 'left_V3d', ...
              'left_hV4', 'left_VO1', 'left_VO2', 'left_V3b', 'left_V3a', ...
              'right_V1v', 'right_V1d', 'right_V2v', 'right_V2d', 'right_V3v', 'right_V3d', ...
              'right_hV4', 'right_VO1', 'right_VO2', 'right_V3b', 'right_V3a'};
leftVisualROIs = {'left_V1v', 'left_V1d', 'left_V2v', 'left_V2d', 'left_V3v', 'left_V3d', ...
              'left_hV4', 'left_VO1', 'left_VO2', 'left_V3b', 'left_V3a'};
rightVisualROIs = {'right_V1v', 'right_V1d', 'right_V2v', 'right_V2d', 'right_V3v', 'right_V3d', ...
              'right_hV4', 'right_VO1', 'right_VO2', 'right_V3b', 'right_V3a'};
parietalROIs = {'left_IPS0', 'left_IPS1', 'left_IPS2', 'left_IPS3', 'left_IPS4', 'left_IPS5', 'left_SPL1', ...
                'right_IPS0', 'right_IPS1', 'right_IPS2', 'right_IPS3', 'right_IPS4', 'right_IPS5', 'right_SPL1'};
leftParietalROIs = {'left_IPS0', 'left_IPS1', 'left_IPS2', 'left_IPS3', 'left_IPS4', 'left_IPS5', 'left_SPL1'};
rightParietalROIs = {'right_IPS0', 'right_IPS1', 'right_IPS2', 'right_IPS3', 'right_IPS4', 'right_IPS5', 'right_SPL1'};
frontalROIs = {'left_FEF', 'right_FEF'};
leftFrontalROIs = {'left_FEF'};
rightFrontalROIs = {'right_FEF'};

% Convert label sets to index sets
visual_idx = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), visualROIs, 'UniformOutput', true);
parietal_idx = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), parietalROIs, 'UniformOutput', true);
frontal_idx = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), frontalROIs, 'UniformOutput', true);
left_visual_idx = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), leftVisualROIs, 'UniformOutput', true);
right_visual_idx = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), rightVisualROIs, 'UniformOutput', true);
left_parietal_idx = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), leftParietalROIs, 'UniformOutput', true);
right_parietal_idx = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), rightParietalROIs, 'UniformOutput', true);
left_frontal_idx = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), leftFrontalROIs, 'UniformOutput', true);
right_frontal_idx = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), rightFrontalROIs, 'UniformOutput', true);


% Get corresponding points
visual_points = ismember(sourcemodel_atlas_label, visual_idx);
parietal_points = ismember(sourcemodel_atlas_label, parietal_idx);
frontal_points = ismember(sourcemodel_atlas_label, frontal_idx);
left_visual_points = ismember(sourcemodel_atlas_label, left_visual_idx);
right_visual_points = ismember(sourcemodel_atlas_label, right_visual_idx);
left_parietal_points = ismember(sourcemodel_atlas_label, left_parietal_idx);
right_parietal_points = ismember(sourcemodel_atlas_label, right_parietal_idx);
left_frontal_points = ismember(sourcemodel_atlas_label, left_frontal_idx);
right_frontal_points = ismember(sourcemodel_atlas_label, right_frontal_idx);


% Create a color map for visualization on surface
% Assign colors with larger gaps to prevent interpolation blending
% Use: 0 = gray (other), 10 = orange (visual), 20 = purple (frontal)
% Create a color map for visualization on surface
roi_color_map = zeros(size(sourcemodel_atlas_label)); % Initialize with zeros (Gyri/light gray)

% Use curvature to define Sulci (dark gray) which will be value 1
if isfield(mesh_inflated, 'curv')
    is_sulcus = mesh_inflated.curv > 0;
    roi_color_map(is_sulcus) = 1; % Sulci = 1
end

roi_color_map(visual_points & ~frontal_points) = 10; % Visual regions only (not frontal)
roi_color_map(frontal_points) = 20; % Frontal regions (including any that overlap with visual)
% roi_color_map(parietal_points) = 30; % Parietal regions = 30 (if needed)

% Debug: Check for any overlap and verify assignments
fprintf('\nColor assignment summary:\n');
fprintf('  Visual only: %d vertices\n', sum(visual_points & ~frontal_points));
fprintf('  Frontal (may include visual): %d vertices\n', sum(frontal_points));
fprintf('  Visual AND Frontal overlap: %d vertices\n', sum(visual_points & frontal_points));
fprintf('  Other: %d vertices\n', sum(roi_color_map == 0));
fprintf('  ROI color map values: min=%.1f, max=%.1f\n', min(roi_color_map), max(roi_color_map));
fprintf('  Vertices with value 10 (visual): %d\n', sum(roi_color_map == 10));
fprintf('  Vertices with value 20 (frontal): %d\n', sum(roi_color_map == 20));

% Verify that frontal vertices are correctly assigned
frontal_vertices_check = roi_color_map(frontal_points);
fprintf('  Frontal vertices color values: %d have value 10 (wrong!), %d have value 20 (correct)\n', ...
    sum(frontal_vertices_check == 10), sum(frontal_vertices_check == 20));
if sum(frontal_vertices_check == 10) > 0
    warning('Some frontal vertices are still assigned to visual! This should not happen.');
end

% Separate hemispheres using template mesh coordinates
left_hemisphere_idx = template_mesh.pos(:,1) < 0;
right_hemisphere_idx = template_mesh.pos(:,1) > 0;

% Create vertex maps for reindexing
left_vertex_map = find(left_hemisphere_idx);
right_vertex_map = find(right_hemisphere_idx);

% Define colors for each ROI group
group_color = [241 90 41;  % orange  - visual
               150 90 164];% purple - frontal
group_color = group_color ./ 255;

% Create custom colormap with 21 explicit steps (mapping 0 to 20 exactly)
% Value 0: Gyri (Light Gray)
% Value 1: Sulci (Dark Gray)
% Value 10: Visual (Orange)
% Value 20: Frontal (Purple)
custom_colormap = repmat([0.7 0.7 0.7], 21, 1); % Default all to light gray
custom_colormap(2, :) = [0.4 0.4 0.4];          % Index 2 corresponds to Value 1 (Sulci)
custom_colormap(11, :) = group_color(1,:);      % Index 11 corresponds to Value 10 (Visual)
custom_colormap(21, :) = group_color(2,:);      % Index 21 corresponds to Value 20 (Frontal)

% Visualize on surface using ft_sourceplot
figure('Position', [100, 100, 1200, 600], 'Renderer','painters');

% Left hemisphere view
subplot(1, 2, 1);

% Create source structure for visualization
sourceVisualize = struct();
sourceVisualize.pos = template_mesh.pos;
sourceVisualize.tri = template_mesh.tri;
sourceVisualize.unit = 'mm';
sourceVisualize.coordsys = 'mni';
sourceVisualize.roi = roi_color_map; % ROI assignment for each vertex

% Create separate structure for left hemisphere
sourceVisualize_left = sourceVisualize;
sourceVisualize_left.pos = template_mesh.pos(left_hemisphere_idx, :);
sourceVisualize_left.roi = roi_color_map(left_hemisphere_idx); % Filter ROI data to match hemisphere

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
cfg.funparameter = 'roi';
cfg.funcolormap = custom_colormap;
cfg.colorbar = 'yes';
cfg.funcolorlim = [0 20]; % Limits for categorical mapping (0=gray, 10=orange, 20=purple)
% We do NOT provide cfg.surffile, so fieldtrip plots the mesh directly from sourceVisualize_left
ft_sourceplot(cfg, sourceVisualize_left);

% Set view angle and lighting
view([-1, -0.5, 1]); % Left hemisphere view
lighting gouraud;
material dull;
cl = camlight('headlight'); % Attaches light perfectly to the camera 
cl.Color = [0.4, 0.4, 0.4];
title('Wang Atlas ROIs on Inflated Surface - Left View');

% Right hemisphere view
subplot(1, 2, 2);

% Create separate structure for right hemisphere
sourceVisualize_right = sourceVisualize;
sourceVisualize_right.pos = template_mesh.pos(right_hemisphere_idx, :);
sourceVisualize_right.roi = roi_color_map(right_hemisphere_idx); % Filter ROI data to match hemisphere

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
cfg.funparameter = 'roi';
cfg.funcolormap = custom_colormap;
cfg.colorbar = 'yes';
cfg.funcolorlim = [0 20]; % Limits for categorical mapping (0=gray, 10=orange, 20=purple)
% We do NOT provide cfg.surffile, so fieldtrip plots the mesh directly from sourceVisualize_right
ft_sourceplot(cfg, sourceVisualize_right);

% Set view angle and lighting
view([1, -0.5, 1]); % Right hemisphere view
lighting gouraud;
material dull;
cl = camlight('headlight'); % Attaches light perfectly to the camera 
cl.Color = [0.4, 0.4, 0.4];

title('Wang Atlas ROIs on Inflated Surface - Right View');
