% function explore_fieldtrip_atlases()
% explore_fieldtrip_atlases - Explore and visualize different atlases in FieldTrip
%
% This script loads the cortex surface and displays various atlases available
% in FieldTrip, allowing you to see different brain parcellations and regions.
%
% Outputs:
%   - Interactive figures showing different atlas overlays
%   - Information about available atlases and their regions
%
% Author: MEG Analysis Pipeline
% Date: 2024

restoredefaultpath;
close all; clc;

%% Path Setup
fprintf('=== FieldTrip Atlas Explorer ===\n');

% Local paths
fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';

% Add FieldTrip to path
addpath(fieldtrip_path);
ft_defaults;

% Add project path
addpath(genpath(project_path));

%% Load Template Surface
fprintf('Loading template surface...\n');

% Try different surface resolutions
surface_resolutions = [5124, 8196, 20484];
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
if isempty(vtpm_files)
    vtpm_files = dir(fullfile(vtpm_path, '*.nii'));
end

if isempty(vtpm_files)
    error('No VTPM atlas files found in: %s', vtpm_path);
end

vtpm_file = fullfile(vtpm_path, vtpm_files(1).name);
fprintf('Using VTPM file: %s\n', vtpm_files(1).name);

if exist(vtpm_file, 'file')
    try
        % Load the VTPM atlas
        vtpm_atlas = ft_read_atlas(vtpm_file);
        fprintf('  ✓ VTPM atlas loaded: %d regions\n', length(unique(vtpm_atlas.tissue)));
    catch ME
        error('Error loading VTPM atlas: %s', ME.message);
    end
else
    error('VTPM atlas file not found at: %s', vtpm_file);
end

%% Resample Atlas to Surface and Visualize
fprintf('\nResampling VTPM atlas to surface vertices...\n');

% Create source structure for visualization
sourceVisualize = struct();
sourceVisualize.pos = template_mesh.pos;
sourceVisualize.tri = template_mesh.tri;
sourceVisualize.unit = 'mm';
sourceVisualize.coordsys = 'mni';

% Resample atlas to surface vertices
try
    % Use ft_sourceinterpolate to resample volume atlas to surface
    cfg = [];
    cfg.method = 'nearest';
    cfg.interpmethod = 'nearest';
    sourceVisualize = ft_sourceinterpolate(cfg, vtpm_atlas, sourceVisualize);
    
    fprintf('  ✓ Atlas resampled to surface successfully\n');
    
    % Create figure
    figure('Position', [100, 100, 1200, 800], 'Name', 'VTPM Atlas on Surface');
    
    % Visualize on surface
    cfg = [];
    cfg.method = 'surface';
    cfg.funparameter = 'tissue';
    cfg.surffile = sprintf('cortex_%d.surf.gii', surface_resolution);
    cfg.colorbar = 'yes';
    cfg.funcolormap = 'jet';
    ft_sourceplot(cfg, sourceVisualize);
    
    % Set view and lighting
    view(0, 40);
    lighting gouraud;
    material dull;
    light('Position', [-1, -1, 1], 'Style', 'infinite', 'Color', [0.4, 0.4, 0.4]);
    
    % Title and labels
    title(sprintf('VTPM Atlas on Surface (%d regions)', length(unique(vtpm_atlas.tissue))));
    xlabel('X (mm)');
    ylabel('Y (mm)');
    zlabel('Z (mm)');
    
    % Set axis properties
    axis equal;
    axis tight;
    grid on;
    
    fprintf('  ✓ VTPM atlas visualization complete\n');
    
catch ME
    fprintf('  Error during resampling/visualization: %s\n', ME.message);
    fprintf('  Falling back to atlas information display...\n');
    
    % Fallback: just display atlas information
    fprintf('\nVTPM Atlas Information:\n');
    fprintf('Atlas dimensions: %s\n', mat2str(size(vtpm_atlas.tissue)));
    fprintf('Number of unique regions: %d\n', length(unique(vtpm_atlas.tissue)));
    
    if isfield(vtpm_atlas, 'tissuelabel')
        fprintf('\nSample region names:\n');
        for i = 1:min(10, length(vtpm_atlas.tissuelabel))
            fprintf('  %d. %s\n', i, vtpm_atlas.tissuelabel{i});
        end
        if length(vtpm_atlas.tissuelabel) > 10
            fprintf('  ... and %d more regions\n', length(vtpm_atlas.tissuelabel) - 10);
        end
    end
end

fprintf('\n=== VTPM Atlas Complete ===\n');
fprintf('VTPM atlas loaded from: %s\n', vtpm_file);

% end
