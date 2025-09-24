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
surface_resolutions = [5124]; %, 8196, 20484];
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






