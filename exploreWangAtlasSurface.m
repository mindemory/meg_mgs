

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


% ── Vertex -> ROI colour code ─────────────────────────────────────────────
% Categorical codes, spaced out so ft_sourceplot's colour interpolation
% cannot blend two ROIs into a third ROI's colour:
%   0  = gyrus  (light gray)      10 = visual
%   1  = sulcus (dark gray)       20 = parietal
%                                 30 = frontal
% Colours below match the ROI palette used throughout glue_decoding's
% figures (plot_decoding_ts.py, plot_timeseries.py, intrinsic_dim_epochs.py,
% aggregate_glue_capacity.py, ...), so this anatomy panel reads as the legend
% for every results figure in the same set. See ROI_COLOUR_HEX below.
roi_color_map = zeros(size(sourcemodel_atlas_label)); % Initialize with zeros (Gyri/light gray)

% Use curvature to define Sulci (dark gray) which will be value 1
if isfield(mesh_inflated, 'curv')
    is_sulcus = mesh_inflated.curv > 0;
    roi_color_map(is_sulcus) = 1; % Sulci = 1
end

% Precedence: visual < parietal < frontal, i.e. a vertex claimed by more than
% one group (possible because the vertex->voxel mapping is nearest-neighbour,
% not an exact parcellation) is drawn as the LAST group listed here. Overlap
% counts are printed below so a large number is visible rather than silently
% painted over.
roi_color_map(visual_points)   = 10;
roi_color_map(parietal_points) = 20;
roi_color_map(frontal_points)  = 30;

fprintf('\nColor assignment summary (precedence visual < parietal < frontal):\n');
fprintf('  Visual   : %6d vertices labelled, %6d drawn\n', sum(visual_points),   sum(roi_color_map == 10));
fprintf('  Parietal : %6d vertices labelled, %6d drawn\n', sum(parietal_points), sum(roi_color_map == 20));
fprintf('  Frontal  : %6d vertices labelled, %6d drawn\n', sum(frontal_points),  sum(roi_color_map == 30));
fprintf('  Overlaps : visual&parietal %d, visual&frontal %d, parietal&frontal %d\n', ...
    sum(visual_points & parietal_points), sum(visual_points & frontal_points), ...
    sum(parietal_points & frontal_points));
fprintf('  Unlabelled cortex: %d vertices\n', sum(roi_color_map == 0 | roi_color_map == 1));

% Separate hemispheres using template mesh coordinates
left_hemisphere_idx = template_mesh.pos(:,1) < 0;
right_hemisphere_idx = template_mesh.pos(:,1) > 0;

% Create vertex maps for reindexing
left_vertex_map = find(left_hemisphere_idx);
right_vertex_map = find(right_hemisphere_idx);

% ROI palette -- these hex values are the single source of truth shared with
% glue_decoding's Python figures (ROI_COLOURS in plot_decoding_ts.py etc.):
%   visual   #FFC629  mango / amber
%   parietal #A78BFA  soft violet
%   frontal  #34D399  emerald mint
% Keep the two in sync: changing a colour here without changing it there
% silently breaks the cross-figure correspondence this panel exists to provide.
ROI_NAMES      = {'Visual', 'Parietal', 'Frontal'};
ROI_COLOUR_HEX = {'#FFC629', '#A78BFA', '#34D399'};
% Hex -> Nx3 [0,1] RGB, so the hex strings above stay the only place a colour
% is written down (no hand-converted RGB triplet to drift out of sync).
group_color = cell2mat(cellfun(@(h) sscanf(h(2:end), '%2x%2x%2x')' / 255, ...
                                ROI_COLOUR_HEX', 'UniformOutput', false));

% Custom colormap with 31 explicit steps (mapping values 0..30 exactly).
% MATLAB indices are value+1, hence the 11/21/31 below.
custom_colormap = repmat([0.7 0.7 0.7], 31, 1); % Default all to light gray (gyri)
custom_colormap(2,  :) = [0.4 0.4 0.4];         % Value 1  = sulci (dark gray)
custom_colormap(11, :) = group_color(1,:);      % Value 10 = visual
custom_colormap(21, :) = group_color(2,:);      % Value 20 = parietal
custom_colormap(31, :) = group_color(3,:);      % Value 30 = frontal

% ── Camera angles ─────────────────────────────────────────────────────────
% [azimuth elevation] for MATLAB's view(az, el). az is measured from the -y
% axis, i.e. az=0 is a straight-on POSTERIOR view of the brain (looking at
% the occipital pole) and az=+/-90 is a lateral view; el is elevation above
% the axial plane. The previous hard-coded direction vectors [-1 -0.5 1] /
% [1 -0.5 1] work out to az=+/-63, el=42 -- a near-lateral view from well
% above, which showed FEF nicely but rotated the occipital pole out of sight
% and foreshortened the IPS band. The pair below tilts further back and drops
% the elevation a little: posterior-dorsal oblique, which is the angle that
% shows the visual ROIs (V1/V2/V3/V3a/V3b/hV4/VO, occipital pole + lateral
% occipital) and the parietal band (IPS0-5, SPL1) at the same time. FEF sits
% further forward and is correspondingly more foreshortened -- that is the
% trade-off being made deliberately.
% Re-frame both panels by editing these two numbers:
%   az -> 0    more posterior (more occipital pole, less lateral surface)
%   az -> +/-90 more lateral  (more FEF, occipital pole rotates away)
%   el up      more dorsal    (more IPS/SPL1, less ventral hV4/VO)
%   el down    more lateral/ventral
VIEW_AZEL_LEFT  = [-42 28];
VIEW_AZEL_RIGHT = [ 42 28];

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
cfg.colorbar = 'no';   % categorical codes -- a continuous colorbar is meaningless here;
                       % roi_legend() below draws a labelled patch legend instead
cfg.funcolorlim = [0 30]; % Categorical mapping: 0/1=gray, 10=visual, 20=parietal, 30=frontal
% We do NOT provide cfg.surffile, so fieldtrip plots the mesh directly from sourceVisualize_left
ft_sourceplot(cfg, sourceVisualize_left);

% Set view angle and lighting
view(VIEW_AZEL_LEFT(1), VIEW_AZEL_LEFT(2));   % see VIEW_AZEL_* above
lighting gouraud;
material dull;
cl = camlight('headlight'); % Attaches light perfectly to the camera 
cl.Color = [0.4, 0.4, 0.4];
title('Left hemisphere', 'FontSize', 13);

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
cfg.colorbar = 'no';   % categorical codes -- a continuous colorbar is meaningless here;
                       % roi_legend() below draws a labelled patch legend instead
cfg.funcolorlim = [0 30]; % Categorical mapping: 0/1=gray, 10=visual, 20=parietal, 30=frontal
% We do NOT provide cfg.surffile, so fieldtrip plots the mesh directly from sourceVisualize_right
ft_sourceplot(cfg, sourceVisualize_right);

% Set view angle and lighting
view(VIEW_AZEL_RIGHT(1), VIEW_AZEL_RIGHT(2)); % see VIEW_AZEL_* above
lighting gouraud;
material dull;
cl = camlight('headlight'); % Attaches light perfectly to the camera 
cl.Color = [0.4, 0.4, 0.4];

title('Right hemisphere', 'FontSize', 13);

% ── Shared ROI legend ─────────────────────────────────────────────────────
% ft_sourceplot's colorbar is switched off above (the plotted values are
% categorical codes, not a continuous scale), so the ROI colours are labelled
% here instead -- one legend for both hemisphere panels, drawn on an
% invisible full-figure axes so it belongs to neither subplot and cannot be
% clipped by either. The patches carry no data (NaN vertices); they exist
% only as legend proxies.
sgtitle('Wang atlas ROIs on inflated surface', 'FontSize', 14, 'FontWeight', 'bold');

lgd_ax = axes('Position', [0 0 1 1], 'Visible', 'off');
hold(lgd_ax, 'on');
lgd_h = gobjects(1, numel(ROI_NAMES));
for k = 1:numel(ROI_NAMES)
    lgd_h(k) = patch(lgd_ax, NaN, NaN, group_color(k,:), 'EdgeColor', 'none');
end
lgd = legend(lgd_ax, lgd_h, ROI_NAMES, 'Orientation', 'horizontal', 'FontSize', 12);
lgd.Box = 'off';
lgd.Position = [0.5 - lgd.Position(3)/2, 0.02, lgd.Position(3), lgd.Position(4)];
