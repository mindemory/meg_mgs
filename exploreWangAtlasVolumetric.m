

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

sourcemodel_path = sprintf('/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/standard_sourcemodel3d5mm.mat');
load(sourcemodel_path);
sourcemodel_orig = ft_convert_units(sourcemodel, 'mm');
subsourcemodel_path = sprintf('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-01/sourceRecon/sub-01_task-mgs_volumetricSources_5mm.mat');
load(subsourcemodel_path);


figure;
subplot(1, 2, 1)
scatter3(sourcemodel_orig.pos(sourcemodel_orig.inside, 1), sourcemodel_orig.pos(sourcemodel_orig.inside, 2), ...
         sourcemodel_orig.pos(sourcemodel_orig.inside, 3))
subplot(1, 2, 2)
scatter3(sourcemodel.pos(sourcemodel.inside, 1), sourcemodel.pos(sourcemodel.inside, 2), ...
         sourcemodel.pos(sourcemodel.inside, 3))

%%
atlas_path = '/d/DATD/hyper/software/fieldtrip-20250318/template/atlas/vtpm/vtpm.mat';
wangatlas = ft_read_atlas(atlas_path);

% % Get all labeled voxels
% [ind_x, ind_y, ind_z] = ind2sub(size(wangatlas.tissue), find(wangatlas.tissue > 0));
% labels = wangatlas.tissue(wangatlas.tissue > 0); % label index at each voxel
% 
% % Atlas transform: convert voxel indices to real-world (mm) coordinates
% V = [ind_x, ind_y, ind_z, ones(numel(ind_x),1)]; % Homogeneous coordinates
% mm_coords = (wangatlas.transform * V')';  % Nx4, columns are [x y z 1]
% mm_coords = mm_coords(:,1:3);
% 
% % Pick a colormap with enough distinct colors for regions
% cmap = lines(length(wangatlas.tissuelabel)+1);
% colors = cmap(double(labels)+1, :);
% 
% % Plot as 3D scatter (this can get heavy for all voxels—subsample if needed)
% figure;
% scatter3(sourcemodel.pos(sourcemodel.inside, 1), sourcemodel.pos(sourcemodel.inside, 2), ...
%          sourcemodel.pos(sourcemodel.inside, 3))
% hold on;
% scatter3(mm_coords(:,1), mm_coords(:,2), mm_coords(:,3), 3, colors, 'filled');
% xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
% axis equal;
% title('Wang atlas: 3D scatter plot by region label');

%%
% Positions of inside sourcemodel grid points (Nx3)
sm_orig_pos = sourcemodel_orig.pos(sourcemodel_orig.inside, :);
sm_pos = sourcemodel.pos(sourcemodel.inside, :);

% Atlas labeled voxels & their labels
[ind_x, ind_y, ind_z] = ind2sub(size(wangatlas.tissue), find(wangatlas.tissue > 0));
V = [ind_x, ind_y, ind_z, ones(numel(ind_x),1)];
mm_coords = (wangatlas.transform * V')';  % Mx4
mm_coords = mm_coords(:,1:3);
label_atlas = double(wangatlas.tissue(wangatlas.tissue > 0)); % region index at each voxel

% Nearest neighbor mapping from each grid point to nearest labeled atlas voxel
[idx_nearest, dist] = knnsearch(mm_coords, sm_orig_pos);

% Each gridpoint's assigned Wang atlas index (in tissuelabel/cell format)
sourcemodel_atlas_label = label_atlas(idx_nearest);

% OPTIONAL: filter points too far from any labeled ROI (e.g., >2mm)
good = (dist < 20); sourcemodel_atlas_label(~good) = 0; % 0 = unlabeled/outside cortex


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


group_color = [241 90 41;  % orange  - visual
               % 0.85 0.33 0.1;  % red   - parietal
               150 90 164];% purple - frontal
group_color = group_color ./ 255;

% figure('Renderer', 'painters'); 
% % subplot(1, 2, 1)
% hold on
% scatter3(sm_orig_pos(:,1), sm_orig_pos(:,2), sm_orig_pos(:,3), 5, [0.7 0.7 0.7], 'filled')
% % Visual group
% scatter3(sm_orig_pos(visual_points,1), sm_orig_pos(visual_points,2), sm_orig_pos(visual_points,3), 20, group_color(1,:), 'filled');
% % Parietal group
% % scatter3(sm_orig_pos(parietal_points,1), sm_orig_pos(parietal_points,2), sm_orig_pos(parietal_points,3), 20, group_color(2,:), 'filled');
% % Frontal group
% scatter3(sm_orig_pos(frontal_points,1), sm_orig_pos(frontal_points,2), sm_orig_pos(frontal_points,3), 20, group_color(2,:), 'filled');
% xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
% axis equal; view(3);
% title('Sourcemodel grid points: Visual (blue), Parietal (red), Frontal (green)');
% % legend({'Other','Visual','Parietal','Frontal'});
% legend({'Other', 'Visual', 'Frontal'});

% subplot(1, 2, 2)
% hold on
% scatter3(sm_pos(:,1), sm_pos(:,2), sm_pos(:,3), 5, [0.7 0.7 0.7], 'filled')
% % Visual group
% scatter3(sm_pos(visual_points,1), sm_pos(visual_points,2), sm_pos(visual_points,3), 20, group_color(1,:), 'filled');
% % Parietal group
% scatter3(sm_pos(parietal_points,1), sm_pos(parietal_points,2), sm_pos(parietal_points,3), 20, group_color(2,:), 'filled');
% % Frontal group
% scatter3(sm_pos(frontal_points,1), sm_pos(frontal_points,2), sm_pos(frontal_points,3), 20, group_color(3,:), 'filled');
% xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
% axis equal; view(3);
% title('Sourcemodel grid points: Visual (blue), Parietal (red), Frontal (green)');
% legend({'Other','Visual','Parietal','Frontal'});

% save('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/atlas/rois_5mm.mat', "visual_points", "parietal_points", "frontal_points", ...
%     "left_visual_points", "right_visual_points", "left_parietal_points", "right_parietal_points", "left_frontal_points", "right_frontal_points");