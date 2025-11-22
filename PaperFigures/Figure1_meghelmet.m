restoredefaultpath;
clear; close all; clc;

addpath('/d/DATD/hyper/software/fieldtrip-20250318/'); 
ft_defaults;
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'));

% Load data
load('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-01/meg/sub-01_task-mgs_run-01_raw.mat')

% Load T1 anatomical image
mri = ft_read_mri('/System/Volumes/Data/d/DATD/datd/fs_subjects/MDanat/SUMA/T1.nii');
mri = ft_convert_units(mri, 'cm');
mri.coordsys = 'ras';
mri = ft_convert_coordsys(mri, 'als');
% Segment MRI to identify scalp/head tissue
cfg = [];
cfg.output = {'scalp', 'skull', 'brain'};
segmentedmri = ft_volumesegment(cfg, mri);
segmentedmri = ft_convert_units(segmentedmri, 'cm');

% Create mesh from scalp segmentation
cfg = [];
cfg.tissue = 'scalp';
cfg.method = 'isosurface';
cfg.numvertices = 10000;
scalp_mesh = ft_prepare_mesh(cfg, segmentedmri);

% Align scalp mesh center with grad center
% Ensure grad is in same units as scalp mesh
% if isfield(data.grad, 'unit')
%     data.grad = ft_convert_units(data.grad, 'cm');
% end
% grad_center = mean(data.grad.chanpos, 1);
% scalp_center = mean(scalp_mesh.pos, 1);
% translation = grad_center - scalp_center;
% scalp_mesh.pos = scalp_mesh.pos + translation;

%%
scalp_mesh_c = scalp_mesh;
scalp_mesh_c.pos = scalp_mesh_c.pos + [-1 2 0];
% Visualize MEG helmet in 3D with scalp/face surface
figure('Renderer', 'painters');
hold on;

% Plot scalp with yellowish color
ft_plot_mesh(scalp_mesh_c, 'facecolor', [0.9 0.85 0.7], 'facealpha', 1.0, 'edgecolor', 'none');

% Plot gradiometers as 3D square discs
sensor_size = 2.0; % Size of square sensor in cm
sensor_thickness = 0.3; % Thickness of sensor disc in cm
for i = 1:size(data.grad.chanpos, 1)
    pos = data.grad.chanpos(i, :);
    ori = data.grad.chanori(i, :);
    
    % Create square base in plane perpendicular to sensor orientation
    % Find two perpendicular vectors to the sensor orientation
    if abs(ori(3)) < 0.9
        v1 = cross(ori, [0 0 1]);
    else
        v1 = cross(ori, [1 0 0]);
    end
    v1 = v1 / norm(v1);
    v2 = cross(ori, v1);
    v2 = v2 / norm(v2);
    
    % Create square corners
    half_size = sensor_size / 2;
    corners = [
        pos + half_size * v1 + half_size * v2;
        pos - half_size * v1 + half_size * v2;
        pos - half_size * v1 - half_size * v2;
        pos + half_size * v1 - half_size * v2;
    ];
    
    % Create top and bottom faces of the square disc
    top_corners = corners + (sensor_thickness/2) * ori;
    bottom_corners = corners - (sensor_thickness/2) * ori;
    
    % Plot top face
    patch('Faces', [1 2 3 4], 'Vertices', top_corners, 'FaceColor', 'b', 'EdgeColor', 'b', 'FaceLighting', 'gouraud');
    % Plot bottom face
    patch('Faces', [1 2 3 4], 'Vertices', bottom_corners, 'FaceColor', 'b', 'EdgeColor', 'b', 'FaceLighting', 'gouraud');
    % Plot side faces
    for j = 1:4
        next_j = mod(j, 4) + 1;
        side_verts = [top_corners(j, :); top_corners(next_j, :); bottom_corners(next_j, :); bottom_corners(j, :)];
        patch('Faces', [1 2 3 4], 'Vertices', side_verts, 'FaceColor', 'b', 'EdgeColor', 'b', 'FaceLighting', 'gouraud');
    end
end

view([160 10]); % Right side view - adjust azimuth if needed (try 0, 90, 180, 270 or -90)
title('MEG Helmet - 3D Visualization');

% Add lighting
lighting gouraud;
material([0.3 0.8 0.2 10 1]); % [ka kd ks n sc] - ambient, diffuse, specular, shininess, specular color
light('Position', [100 100 100], 'Style', 'infinite');
% light('Position', [-100 -100 100], 'Style', 'infinite');
% light('Position', [0 0 200], 'Style', 'infinite');

