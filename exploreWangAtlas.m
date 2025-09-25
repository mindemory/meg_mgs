% function exploreWangAtlas()
% exploreWangAtlas - Explore and visualize Wang atlas (volumetric MNI space)
%
% This script loads the Wang atlas (volumetric MNI space) and displays various atlases available
% in FieldTrip, allowing you to see different brain parcellations and regions.
%
% Author: Mrugank Dake
% Date: 2025-09-24

restoredefaultpath;
close all; clc;

%% Path Setup
fprintf('=== Wang Atlas Explorer ===\n');

% Local paths
fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
ft_gifti_path = '/d/DATD/hyper/software/fieldtrip-20250318/external/gifti'; % Add Gifti toolbox for .surf.gii files

% Add FieldTrip to path
addpath(fieldtrip_path);
addpath(ft_gifti_path);
ft_defaults;

% Add project path
addpath(genpath(project_path));

%% Load Template Surface
fprintf('Loading template surface...\n');

% Try different surface resolutions
surface_resolutions = [20484]; %, 8196, 20484];
surface_loaded = false;

for res = surface_resolutions
    try
        surface_file = sprintf('cortex_%d.surf.gii', res);
        if exist(surface_file, 'file')
            template_mesh = ft_read_headshape(surface_file);
            surface_resolution = res;
            surface_loaded = true;
            fprintf('Loaded surface: %s (%d vertices)\n', surface_file, size(template_mesh.pos, 1));
            break;
        end
    catch
        continue;
    end
end

if ~surface_loaded
    error('Could not load any template surface. Please check if cortex_*.surf.gii files exist.');
end

%% Load VTPM Atlas
fprintf('\nLoading VTPM atlas...\n');

% VTPM atlas path
vtpm_path = '/d/DATD/hyper/software/fieldtrip-20250318/template/atlas/vtpm/';

% Find VTPM atlas file (try both .mat and .nii)
vtpm_files = dir(fullfile(vtpm_path, '*.mat'));

vtpm_file = fullfile(vtpm_path, vtpm_files(1).name);
fprintf('Using VTPM file: %s\n', vtpm_files(1).name);

% Load the VTPM atlas
vtpm_atlas = ft_read_atlas(vtpm_file);

% Display atlas properties including voxel size
fprintf('VTPM Atlas Properties:\n');
fprintf('  Total regions: %d\n', length(unique(vtpm_atlas.tissue)));
fprintf('  Tissue data size: %s\n', mat2str(size(vtpm_atlas.tissue)));

if isfield(vtpm_atlas, 'unit')
    fprintf('  Unit: %s\n', vtpm_atlas.unit);
end

if isfield(vtpm_atlas, 'coordsys')
    fprintf('  Coordinate system: %s\n', vtpm_atlas.coordsys);
end

% Check for voxel size information
if isfield(vtpm_atlas, 'transform')
    fprintf('  Transform matrix available: %s\n', mat2str(size(vtpm_atlas.transform)));
    % Extract voxel size from transform matrix (diagonal elements)
    voxel_size = abs(diag(vtpm_atlas.transform(1:3, 1:3)));
    fprintf('  Voxel size: [%.2f, %.2f, %.2f] %s\n', voxel_size(1), voxel_size(2), voxel_size(3), vtpm_atlas.unit);
elseif isfield(vtpm_atlas, 'dim')
    fprintf('  Dimensions: %s\n', mat2str(vtpm_atlas.dim));
    if isfield(vtpm_atlas, 'transform')
        voxel_size = abs(diag(vtpm_atlas.transform(1:3, 1:3)));
        fprintf('  Voxel size: [%.2f, %.2f, %.2f] %s\n', voxel_size(1), voxel_size(2), voxel_size(3), vtpm_atlas.unit);
    end
else
    fprintf('  No explicit voxel size information found\n');
end

% Check if it's a volume or surface atlas
if isfield(vtpm_atlas, 'pos')
    fprintf('  Surface atlas detected (pos field present)\n');
elseif isfield(vtpm_atlas, 'dim')
    fprintf('  Volume atlas detected (dim field present)\n');
else
    fprintf('  Atlas type: Unknown\n');
end

%% Create VTPM Information Display
fprintf('\nCreating VTPM atlas information display...\n');

% Create figure for VTPM atlas
figure('Position', [100, 100, 1200, 800], 'Name', 'VTPM Atlas Information');

% Region size distribution
if isfield(vtpm_atlas, 'tissue') && ~isempty(vtpm_atlas.tissue)
    unique_regions = unique(vtpm_atlas.tissue);
    unique_regions = unique_regions(2:end); % Remove background (0)
    region_counts = histcounts(vtpm_atlas.tissue, [unique_regions; max(unique_regions)+1]);
    
    % Create bar plot
    bar(1:length(unique_regions), region_counts, 'FaceColor', [0.3, 0.6, 0.9], 'EdgeColor', 'none');
    title('VTPM Region Size Distribution', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Brain Regions', 'FontSize', 12);
    ylabel('Number of Voxels', 'FontSize', 12);
    grid on;
    
    % Set x-axis labels to tissue names if available
    if isfield(vtpm_atlas, 'tissuelabel') && length(vtpm_atlas.tissuelabel) >= length(unique_regions)
        % Get tissue labels for the unique regions (skip background)
        tissue_labels = vtpm_atlas.tissuelabel(2:end); % Skip background label
        if length(tissue_labels) >= length(unique_regions)
            % Set x-axis labels
            set(gca, 'XTick', 1:length(unique_regions));
            set(gca, 'XTickLabel', tissue_labels(1:length(unique_regions)));
            % Rotate labels for better readability
            xtickangle(45);
        end
    end
    
    % Add statistics
    text(0.02, 0.98, sprintf('Total: %d voxels\nRegions: %d\nMean: %.1f voxels/region', ...
        sum(region_counts), length(unique_regions), mean(region_counts)), ...
        'Units', 'normalized', 'FontSize', 10, 'VerticalAlignment', 'top', ...
        'BackgroundColor', 'white', 'EdgeColor', 'black');
end


%%
figure;
% 3D scatter plot with ROI coloring and legend
if isfield(vtpm_atlas, 'tissue') && isfield(vtpm_atlas, 'dim')
    % Sample voxels for each region (to avoid too many points)
    all_coords = [];
    all_colors = [];
    all_labels = {};
    
    % Background (region 0) - gray
    % bg_coords = find(vtpm_atlas.tissue == 0);
    % if ~isempty(bg_coords)
    %     [x, y, z] = ind2sub(vtpm_atlas.dim, bg_coords(1:length(bg_coords))); % Sample max 1000 points
    %     all_coords = [all_coords; x, y, z];
    %     all_colors = [all_colors; repmat([0.5, 0.5, 0.5], length(x), 1)];
    %     all_labels{end+1} = 'Background';
    % end
    
    % Create separate scatter plots for each ROI (one legend entry per ROI)
    hold on;
    
    % Limit to first 15 ROIs to avoid too many legend entries
    max_rois = length(unique_regions);
    
    for i = 1:max_rois
        region_id = unique_regions(i);
        region_coords = find(vtpm_atlas.tissue == region_id);
        
        if ~isempty(region_coords)
            [x, y, z] = ind2sub(vtpm_atlas.dim, region_coords);
            
            % Get very distinct color for this ROI
            % Create 25 highly distinct colors (left/right can share colors)
            distinct_colors = [
                1.0, 0.0, 0.0;  % Red
                0.0, 1.0, 0.0;  % Green
                0.0, 0.0, 1.0;  % Blue
                1.0, 1.0, 0.0;  % Yellow
                1.0, 0.0, 1.0;  % Magenta
                0.0, 1.0, 1.0;  % Cyan
                1.0, 0.5, 0.0;  % Orange
                0.5, 0.0, 1.0;  % Purple
                0.0, 0.5, 0.0;  % Dark Green
                0.5, 0.5, 0.0;  % Olive
                0.5, 0.0, 0.5;  % Maroon
                0.0, 0.5, 0.5;  % Teal
                1.0, 0.3, 0.3;  % Light Red
                0.3, 1.0, 0.3;  % Light Green
                0.3, 0.3, 1.0;  % Light Blue
                1.0, 0.8, 0.0;  % Gold
                0.8, 0.0, 1.0;  % Violet
                0.0, 0.8, 1.0;  % Sky Blue
                1.0, 0.4, 0.4;  % Pink
                0.4, 1.0, 0.4;  % Lime
                0.4, 0.4, 1.0;  % Light Blue
                1.0, 0.6, 0.0;  % Dark Orange
                0.6, 0.0, 1.0;  % Indigo
                0.0, 0.6, 0.6;  % Dark Cyan
                0.8, 0.8, 0.0;  % Dark Yellow
            ];
            
            % Use modulo to cycle through the 25 distinct colors
            color_idx = mod(i-1, size(distinct_colors, 1)) + 1;
            roi_color = distinct_colors(color_idx, :);
            
            % Get ROI label safely
            if isfield(vtpm_atlas, 'tissuelabel') && region_id <= length(vtpm_atlas.tissuelabel)
                roi_label = vtpm_atlas.tissuelabel{region_id};
            else
                roi_label = sprintf('ROI %d', region_id);
            end
            
            % Create scatter plot for this ROI
            scatter3(x, y, z, 20, roi_color, 'filled', 'DisplayName', roi_label);
        end
    end
    
    % Set plot properties
    title('3D ROI Visualization', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('X (voxels)', 'FontSize', 12);
    ylabel('Y (voxels)', 'FontSize', 12);
    zlabel('Z (voxels)', 'FontSize', 12);
    grid on;
    
    % Add legend (one entry per ROI)
    legend('Location', 'eastoutside', 'FontSize', 8);
    view(45, 30); % Set nice viewing angle
    hold off;
end

%% Visualize Specific ROIs
fprintf('\nCreating specific ROI visualization...\n');

% Define ROIs (using full names with hemisphere prefixes)
visual_rois = {'left_V1d', 'left_V1v', 'left_V2d', 'left_V2v', 'left_V3d', 'left_V3v', 'left_V3a', 'left_V3b', ...
               'right_V1d', 'right_V1v', 'right_V2d', 'right_V2v', 'right_V3d', 'right_V3v', 'right_V3a', 'right_V3b'};
parietal_rois = {'left_IPS0', 'left_IPS1', 'left_IPS2', 'left_IPS3', 'left_IPS4', 'left_IPS5', ...
                'right_IPS0', 'right_IPS1', 'right_IPS2', 'right_IPS3', 'right_IPS4', 'right_IPS5'};
frontal_rois = {'left_FEF', 'right_FEF'};

% Create single figure
figure('Position', [100, 100, 1200, 800], 'Name', 'Wang Atlas - Specific ROIs');

hold on;

% Visual ROIs - Red color
for i = 1:length(visual_rois)
    roi_name = visual_rois{i};
    for region_id = 1:length(vtpm_atlas.tissuelabel)
        if strcmp(vtpm_atlas.tissuelabel{region_id}, roi_name)
            [x, y, z] = ind2sub(vtpm_atlas.dim, find(vtpm_atlas.tissue == region_id));
            scatter3(x, y, z, 30, [1, 0, 0], 'filled', 'DisplayName', roi_name);
            break;
        end
    end
end

% Parietal ROIs - Blue color
for i = 1:length(parietal_rois)
    roi_name = parietal_rois{i};
    for region_id = 1:length(vtpm_atlas.tissuelabel)
        if strcmp(vtpm_atlas.tissuelabel{region_id}, roi_name)
            [x, y, z] = ind2sub(vtpm_atlas.dim, find(vtpm_atlas.tissue == region_id));
            scatter3(x, y, z, 30, [0, 0, 1], 'filled', 'DisplayName', roi_name);
            break;
        end
    end
end

% Frontal ROIs - Green color
for i = 1:length(frontal_rois)
    roi_name = frontal_rois{i};
    for region_id = 1:length(vtpm_atlas.tissuelabel)
        if strcmp(vtpm_atlas.tissuelabel{region_id}, roi_name)
            [x, y, z] = ind2sub(vtpm_atlas.dim, find(vtpm_atlas.tissue == region_id));
            scatter3(x, y, z, 30, [0, 1, 0], 'filled', 'DisplayName', roi_name);
            break;
        end
    end
end

title('Wang Atlas - Specific ROIs by Category', 'FontSize', 16, 'FontWeight', 'bold');
xlabel('X (voxels)', 'FontSize', 12);
ylabel('Y (voxels)', 'FontSize', 12);
zlabel('Z (voxels)', 'FontSize', 12);
grid on;
legend('Location', 'eastoutside', 'FontSize', 10);
view(45, 30);
hold off;

%% Visualize ROIs on Cortex Surface
fprintf('\nCreating cortex surface visualization...\n');

% Create surface visualization figure
figure('Position', [200, 200, 1600, 1000], 'Name', 'Wang Atlas - ROIs on Cortex Surface');

% Create source structure for surface visualization
sourceVisualize = struct();
sourceVisualize.pos = template_mesh.pos;
sourceVisualize.tri = template_mesh.tri;
sourceVisualize.unit = 'mm';
sourceVisualize.coordsys = 'mni';

% Initialize ROI data on surface
n_vertices = size(template_mesh.pos, 1);
roi_data = zeros(n_vertices, 1); % 0 = background, 1 = visual, 2 = parietal, 3 = frontal

% Transform atlas coordinates to MNI space using transform matrix
transform_matrix = vtpm_atlas.transform;

% Find vertices that belong to each ROI category
for i = 1:length(visual_rois)
    roi_name = visual_rois{i};
    for region_id = 1:length(vtpm_atlas.tissuelabel)
        if strcmp(vtpm_atlas.tissuelabel{region_id}, roi_name)
            % Get ROI voxels
            [x, y, z] = ind2sub(vtpm_atlas.dim, find(vtpm_atlas.tissue == region_id));
            roi_coords_voxel = [x, y, z, ones(size(x))];
            
            % Transform to MNI space (mm)
            roi_coords_mni = (transform_matrix * roi_coords_voxel')';
            roi_coords_mni = roi_coords_mni(:, 1:3); % Remove homogeneous coordinate
            
            % Find closest surface vertices to ROI coordinates
            for j = 1:size(roi_coords_mni, 1)
                distances = sqrt(sum((template_mesh.pos - roi_coords_mni(j, :)).^2, 2));
                [~, closest_vertex] = min(distances);
                roi_data(closest_vertex) = 1; % Visual
            end
            break;
        end
    end
end

for i = 1:length(parietal_rois)
    roi_name = parietal_rois{i};
    for region_id = 1:length(vtpm_atlas.tissuelabel)
        if strcmp(vtpm_atlas.tissuelabel{region_id}, roi_name)
            % Get ROI voxels
            [x, y, z] = ind2sub(vtpm_atlas.dim, find(vtpm_atlas.tissue == region_id));
            roi_coords_voxel = [x, y, z, ones(size(x))];
            
            % Transform to MNI space (mm)
            roi_coords_mni = (transform_matrix * roi_coords_voxel')';
            roi_coords_mni = roi_coords_mni(:, 1:3); % Remove homogeneous coordinate
            
            % Find closest surface vertices to ROI coordinates
            for j = 1:size(roi_coords_mni, 1)
                distances = sqrt(sum((template_mesh.pos - roi_coords_mni(j, :)).^2, 2));
                [~, closest_vertex] = min(distances);
                roi_data(closest_vertex) = 2; % Parietal
            end
            break;
        end
    end
end

for i = 1:length(frontal_rois)
    roi_name = frontal_rois{i};
    for region_id = 1:length(vtpm_atlas.tissuelabel)
        if strcmp(vtpm_atlas.tissuelabel{region_id}, roi_name)
            % Get ROI voxels
            [x, y, z] = ind2sub(vtpm_atlas.dim, find(vtpm_atlas.tissue == region_id));
            roi_coords_voxel = [x, y, z, ones(size(x))];
            
            % Transform to MNI space (mm)
            roi_coords_mni = (transform_matrix * roi_coords_voxel')';
            roi_coords_mni = roi_coords_mni(:, 1:3); % Remove homogeneous coordinate
            
            % Find closest surface vertices to ROI coordinates
            for j = 1:size(roi_coords_mni, 1)
                distances = sqrt(sum((template_mesh.pos - roi_coords_mni(j, :)).^2, 2));
                [~, closest_vertex] = min(distances);
                roi_data(closest_vertex) = 3; % Frontal
            end
            break;
        end
    end
end

% Add ROI data to source structure
sourceVisualize.roi_data = roi_data;

% Create subplots for different views
subplot(2, 2, 1);
% Left hemisphere view
cfg = [];
cfg.method = 'surface';
cfg.figure = 'gcf';
cfg.funparameter = 'roi_data';
cfg.surffile = sprintf('cortex_%d.surf.gii', surface_resolution);
cfg.colorbar = 'yes';
cfg.funcolormap = [0.8, 0.8, 0.8; 1, 0, 0; 0, 0, 1; 0, 1, 0]; % Gray, Red, Blue, Green
cfg.funcolorlim = [0, 3.5];
ft_sourceplot(cfg, sourceVisualize);
view(-90, 0); % Left hemisphere
title('Left Hemisphere', 'FontSize', 14, 'FontWeight', 'bold');
lighting gouraud;
material dull;

subplot(2, 2, 2);
% Right hemisphere view
ft_sourceplot(cfg, sourceVisualize);
view(90, 0); % Right hemisphere
title('Right Hemisphere', 'FontSize', 14, 'FontWeight', 'bold');
lighting gouraud;
material dull;

subplot(2, 2, 3);
% Top view
ft_sourceplot(cfg, sourceVisualize);
view(0, 90); % Top view
title('Top View', 'FontSize', 14, 'FontWeight', 'bold');
lighting gouraud;
material dull;

subplot(2, 2, 4);
% Back view
ft_sourceplot(cfg, sourceVisualize);
view(180, 0); % Back view
title('Back View', 'FontSize', 14, 'FontWeight', 'bold');
lighting gouraud;
material dull;

sgtitle('Wang Atlas ROIs on Cortex Surface', 'FontSize', 16, 'FontWeight', 'bold');

fprintf('Cortex surface visualization complete!\n');




%%
vroi = sourceVisualize.roi_data == 1;
proi = sourceVisualize.roi_data == 2;
froi = sourceVisualize.roi_data == 3;
emptyVertices = sourceVisualize.roi_data == 0;

figure;
scatter3(sourceVisualize.pos(vroi, 1), sourceVisualize.pos(vroi, 2), sourceVisualize.pos(vroi, 3), 'w', 'filled')
% hold on;
% scatter3(sourceVisualize.pos(proi, 1), sourceVisualize.pos(proi, 2), sourceVisualize.pos(proi, 3), 'w', 'filled')
% 
% scatter3(sourceVisualize.pos(froi, 1), sourceVisualize.pos(froi, 2), sourceVisualize.pos(froi, 3), 'w', 'filled')

scatter3(sourceVisualize.pos(emptyVertices, 1), sourceVisualize.pos(emptyVertices, 2), sourceVisualize.pos(emptyVertices, 3), 'k')

%% Recommended Solution: Using ft_sourceinterpolate
fprintf('\n=== Using ft_sourceinterpolate for proper atlas-to-surface mapping ===\n');

% Create source structure from template mesh
atlas_source = struct();
atlas_source.pos = template_mesh.pos;
atlas_source.tri = template_mesh.tri;
atlas_source.unit = 'mm';
atlas_source.coordsys = 'mni';
atlas_source.inside = sourcemodel_aligned_20484.inside;

% Interpolate atlas to surface using FieldTrip's built-in function
cfg = [];
cfg.interpmethod = 'nearest';
cfg.parameter = 'tissue';
% sourceVisualize_interpolated = ft_sourceinterpolate(cfg, vtpm_atlas, atlas_source);
sourceVisualize_interpolated = ft_sourceinterpolate(cfg, vtpm_atlas, atlas_source);


% Get region IDs for each category
visual_region_ids = [];
parietal_region_ids = [];
frontal_region_ids = [];

% Find region IDs for visual ROIs
for i = 1:length(visual_rois)
    roi_name = visual_rois{i};
    for region_id = 1:length(vtpm_atlas.tissuelabel)
        if strcmp(vtpm_atlas.tissuelabel{region_id}, roi_name)
            visual_region_ids = [visual_region_ids, region_id];
            break;
        end
    end
end

% Find region IDs for parietal ROIs
for i = 1:length(parietal_rois)
    roi_name = parietal_rois{i};
    for region_id = 1:length(vtpm_atlas.tissuelabel)
        if strcmp(vtpm_atlas.tissuelabel{region_id}, roi_name)
            parietal_region_ids = [parietal_region_ids, region_id];
            break;
        end
    end
end

% Find region IDs for frontal ROIs
for i = 1:length(frontal_rois)
    roi_name = frontal_rois{i};
    for region_id = 1:length(vtpm_atlas.tissuelabel)
        if strcmp(vtpm_atlas.tissuelabel{region_id}, roi_name)
            frontal_region_ids = [frontal_region_ids, region_id];
            break;
        end
    end
end

fprintf('Visual region IDs: %s\n', mat2str(visual_region_ids));
fprintf('Parietal region IDs: %s\n', mat2str(parietal_region_ids));
fprintf('Frontal region IDs: %s\n', mat2str(frontal_region_ids));

% Create masks for each category
visual_mask = ismember(sourceVisualize_interpolated.tissue, visual_region_ids);
parietal_mask = ismember(sourceVisualize_interpolated.tissue, parietal_region_ids);
frontal_mask = ismember(sourceVisualize_interpolated.tissue, frontal_region_ids);

% Create ROI data vector
roi_data_interpolated = zeros(size(sourceVisualize_interpolated.tissue));
roi_data_interpolated(visual_mask) = 1;    % Visual
roi_data_interpolated(parietal_mask) = 2; % Parietal
roi_data_interpolated(frontal_mask) = 3;  % Frontal

% Add to source structure
sourceVisualize_interpolated.roi_data = roi_data_interpolated;

% Create visualization using interpolated data
figure('Position', [300, 300, 1600, 1000], 'Name', 'Wang Atlas - ROIs on Cortex Surface (ft_sourceinterpolate)');

% Create subplots for different views
subplot(2, 2, 1);
% Left hemisphere view
cfg = [];
cfg.method = 'surface';
cfg.figure = 'gcf';
cfg.funparameter = 'roi_data';
cfg.surffile = sprintf('cortex_%d.surf.gii', surface_resolution);
cfg.colorbar = 'yes';
cfg.funcolormap = [0.8, 0.8, 0.8; 1, 0, 0; 0, 0, 1; 0, 1, 0]; % Gray, Red, Blue, Green
cfg.funcolorlim = [0, 3.5];
ft_sourceplot(cfg, sourceVisualize_interpolated);
view(-90, 0); % Left hemisphere
title('Left Hemisphere (ft_sourceinterpolate)', 'FontSize', 14, 'FontWeight', 'bold');
lighting gouraud;
material dull;

subplot(2, 2, 2);
% Right hemisphere view
ft_sourceplot(cfg, sourceVisualize_interpolated);
view(90, 0); % Right hemisphere
title('Right Hemisphere (ft_sourceinterpolate)', 'FontSize', 14, 'FontWeight', 'bold');
lighting gouraud;
material dull;

subplot(2, 2, 3);
% Top view
ft_sourceplot(cfg, sourceVisualize_interpolated);
view(0, 90); % Top view
title('Top View (ft_sourceinterpolate)', 'FontSize', 14, 'FontWeight', 'bold');
lighting gouraud;
material dull;

subplot(2, 2, 4);
% Back view
ft_sourceplot(cfg, sourceVisualize_interpolated);
view(180, 0); % Back view
title('Back View (ft_sourceinterpolate)', 'FontSize', 14, 'FontWeight', 'bold');
lighting gouraud;
material dull;

sgtitle('Wang Atlas ROIs on Cortex Surface (ft_sourceinterpolate)', 'FontSize', 16, 'FontWeight', 'bold');

% Compare the two methods
fprintf('\n=== Comparison of Methods ===\n');
fprintf('Manual mapping - Visual vertices: %d, Parietal: %d, Frontal: %d\n', ...
        sum(roi_data == 1), sum(roi_data == 2), sum(roi_data == 3));
fprintf('ft_sourceinterpolate - Visual vertices: %d, Parietal: %d, Frontal: %d\n', ...
        sum(roi_data_interpolated == 1), sum(roi_data_interpolated == 2), sum(roi_data_interpolated == 3));

fprintf('ft_sourceinterpolate method complete!\n');



