%% Simple dimensionality plot for single subject
% Loads S05_DimensionalityAnalysis output for subject 1, 5124 vertices
% and plots the effective dimensionality over time

clear; close all; clc;

fprintf('=== Simple Dimensionality Plot ===\n');

%% Set up paths
[~, hostname] = system('hostname');
hostname = strtrim(hostname);

% Check for common HPC indicators
is_hpc = contains(hostname, {'login', 'compute', 'node', 'hpc'}) || ...
         exist('/etc/slurm', 'dir') || ...
         ~isempty(getenv('SLURM_JOB_ID')) || ...
         ~isempty(getenv('PBS_JOBID'));

if is_hpc
    % HPC paths
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
    project_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC';
    fieldtrip_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/';
    ft_gifti_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/external/gifti';

else
    % Local paths
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
    ft_gifti_path = '/d/DATD/hyper/software/fieldtrip-20250318/external/gifti'; % Add Gifti toolbox for .surf.gii files

end

%% Setup and Initialization
addpath(fieldtrip_path);
addpath(ft_gifti_path);
addpath(genpath(project_path));
ft_defaults;
ft_hastoolbox('spm12', 1);

% Add project path
addpath(genpath(project_path));

%% Load data for subject 1, 8196 vertices
subjID = 2;
surface_resolution = 8196;

% Define output directory for results
output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'dimensionality_analysis');

% Load template mesh for surface visualization
fprintf('Loading template mesh from cortex surface file...\n');
template_mesh = ft_read_headshape(sprintf('cortex_%d.surf.gii', surface_resolution));

% Create source visualization structure
sourceVisualize = struct();
sourceVisualize.pos = template_mesh.pos;
sourceVisualize.tri = template_mesh.tri;
sourceVisualize.unit = 'mm';
sourceVisualize.coordsys = 'mni';

% Split data by hemisphere (assuming X < 0 is left, X > 0 is right)
fprintf('Splitting hemispheres...\n');
left_hemisphere_idx = template_mesh.pos(:, 1) < 0;
right_hemisphere_idx = template_mesh.pos(:, 1) > 0;

% Create mapping for reindexing vertices
left_vertex_map = find(left_hemisphere_idx);
right_vertex_map = find(right_hemisphere_idx);

% Create separate structures for each hemisphere
sourceVisualize_left = sourceVisualize;
sourceVisualize_left.pos = template_mesh.pos(left_hemisphere_idx, :);

% Reindex triangulation for left hemisphere
left_tri = template_mesh.tri;
left_tri_valid = all(left_hemisphere_idx(left_tri), 2);
left_tri = left_tri(left_tri_valid, :);
% Create new vertex indices
[~, left_new_indices] = ismember(left_tri, left_vertex_map);
sourceVisualize_left.tri = left_new_indices;

sourceVisualize_right = sourceVisualize;
sourceVisualize_right.pos = template_mesh.pos(right_hemisphere_idx, :);

% Reindex triangulation for right hemisphere
right_tri = template_mesh.tri;
right_tri_valid = all(right_hemisphere_idx(right_tri), 2);
right_tri = right_tri(right_tri_valid, :);
% Create new vertex indices
[~, right_new_indices] = ismember(right_tri, right_vertex_map);
sourceVisualize_right.tri = right_new_indices;

fprintf('Left hemisphere: %d vertices, %d triangles\n', sum(left_hemisphere_idx), size(sourceVisualize_left.tri, 1));
fprintf('Right hemisphere: %d vertices, %d triangles\n', sum(right_hemisphere_idx), size(sourceVisualize_right.tri, 1));

fprintf('Loading data for Subject %d, Resolution %d...\n', subjID, surface_resolution);

% Construct file path
result_file = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    'dimensionality_analysis', sprintf('S05_dimensionality_sub%02d_%d.mat', subjID, surface_resolution));

if ~exist(result_file, 'file')
    error('Dimensionality analysis file not found: %s\nPlease run S05_DimensionalityAnalysis first!', result_file);
end

fprintf('Loading from: %s\n', result_file);
load(result_file, 'dimensionality_results');

% Extract data
time_windows = dimensionality_results.time_windows;
effective_dimensionality = dimensionality_results.effective_dimensionality;

fprintf('Loaded %d time points\n', length(time_windows));
fprintf('Time range: %.2f to %.2f seconds\n', min(time_windows), max(time_windows));
fprintf('Dimensionality range: %.2f to %.2f\n', min(effective_dimensionality), max(effective_dimensionality));

%% Create simple plot
fprintf('Creating plot...\n');

figure('Position', [100, 100, 1000, 600]);

% Plot the dimensionality over time
plot(time_windows, effective_dimensionality, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);

% Add stimulus onset marker
xline(0, 'k--', 'LineWidth', 2, 'Alpha', 0.8, 'Label', 'Stimulus Onset');

% Formatting
xlabel('Time (s)', 'FontSize', 14);
ylabel('Effective Dimensionality (Participation Ratio)', 'FontSize', 14);
title(sprintf('Subject %d: Effective Dimensionality Over Time (Surface Resolution %d)', subjID, surface_resolution), ...
    'FontSize', 16, 'FontWeight', 'bold');
grid on;
xlim([min(time_windows), max(time_windows)]);

% Add some statistics as text
stats_text = sprintf('Mean: %.2f\nStd: %.2f\nRange: %.2f - %.2f', ...
    mean(effective_dimensionality), ...
    std(effective_dimensionality), ...
    min(effective_dimensionality), ...
    max(effective_dimensionality));

text(0.05, 0.95, stats_text, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'FontSize', 12, ...
    'BackgroundColor', 'white', 'EdgeColor', 'black');

set(gca, 'FontSize', 12);

%% Create movie of PC loadings over time
fprintf('Creating movie of PC loadings over time...\n');

% Check if we want to create a movie or just single timepoint
create_movie = true;  % Set to false for single timepoint visualization

if create_movie
    % Create movie showing PC loadings over time
    fprintf('Creating movie with %d time points...\n', length(time_windows));
    
    % Create figure for movie
    fig_movie = figure('Position', [200, 200, 1600, 800]);
    
    % Set up video writer - save in the same directory as the data
    movie_filename = fullfile(output_dir, sprintf('PC_loadings_movie_sub%02d_%d.mp4', subjID, surface_resolution));
    v = VideoWriter(movie_filename, 'MPEG-4');
    v.FrameRate = 2;   % 2 frames per second (very slow movie)
    v.Quality = 90;    % High quality
    open(v);
    
    % Set common colormap and color limits
    common_colormap = '*RdBu';
    common_colorlim = [-2, 2];
    
    % Process each time point
    for t = 1:length(time_windows)
        fprintf('Processing time point %d/%d (%.2fs)...\n', t, length(time_windows), time_windows(t));
        
        % Get eigenvalues and eigenvectors for this time point
        eigenvalues = dimensionality_results.all_eigenvalues{t};
        eigenvectors = dimensionality_results.all_eigenvectors{t};
        
        if isempty(eigenvalues) || isempty(eigenvectors)
            fprintf('No data for time point %.2fs, skipping...\n', time_windows(t));
            continue;
        end
        
        % Extract eigenvalues and sort
        eigenvals = diag(eigenvalues);
        [sorted_eigenvals, sort_idx] = sort(eigenvals, 'descend');
        sorted_eigenvectors = eigenvectors(:, sort_idx);
        
        % Keep only positive eigenvalues
        positive_idx = sorted_eigenvals > 0;
        positive_eigenvals = sorted_eigenvals(positive_idx);
        positive_eigenvectors = sorted_eigenvectors(:, positive_idx);
        
        % Use top 50 components (or all positive if fewer than 50)
        n_components = min(100, length(positive_eigenvals));
        top_eigenvals = positive_eigenvals(1:n_components);
        top_eigenvectors = positive_eigenvectors(:, 1:n_components);
        
        % Calculate PC loadings (sum of squared loadings weighted by eigenvalues)
        pc_loadings = sum((top_eigenvectors.^2) .* top_eigenvals', 2);
        
        % Z-score the PC loadings
        pc_loadings_zscore = (pc_loadings - mean(pc_loadings)) / std(pc_loadings);
        
        % Add z-scored PC loadings to hemisphere structures
        sourceVisualize_left.pow = pc_loadings_zscore(left_hemisphere_idx);
        sourceVisualize_right.pow = pc_loadings_zscore(right_hemisphere_idx);
        
        % Clear previous plots
        clf;
        
        % Left hemisphere
        subplot(1, 2, 1);
        cfg = [];
        cfg.method = 'surface';
        cfg.figure = 'gcf';
        cfg.funparameter = 'pow';
        cfg.maskparameter = "";
        cfg.surffile = sprintf('cortex_%d.surf.gii', surface_resolution);
        cfg.colorbar = 'no';  % No colorbar for left hemisphere
        cfg.funcolormap = common_colormap;
        cfg.funcolorlim = common_colorlim;
        
        try
            ft_sourceplot(cfg, sourceVisualize_left);
        catch
            fprintf('Left hemisphere surface plotting failed, using scatter plot fallback...\n');
            scatter3(sourceVisualize_left.pos(:, 1), sourceVisualize_left.pos(:, 2), sourceVisualize_left.pos(:, 3), 50, sourceVisualize_left.pow, 'filled');
            colormap(common_colormap);
            caxis(common_colorlim);
        end
        
        view(-45, 30); % Left hemisphere view
        lighting gouraud;
        material dull;
        light('Position', [-1, -1, 1], 'Style', 'infinite', 'Color', [0.4, 0.4, 0.4]);
        title(sprintf('Left Hemisphere (t=%.2fs)', time_windows(t)), 'FontSize', 14, 'FontWeight', 'bold');
        axis equal;
        axis tight;
        grid on;
        set(gca, 'FontSize', 12);
        
        % Right hemisphere
        subplot(1, 2, 2);
        cfg.colorbar = 'yes';  % Show colorbar only on right hemisphere
        cfg.funcolormap = common_colormap;
        cfg.funcolorlim = common_colorlim;
        
        try
            ft_sourceplot(cfg, sourceVisualize_right);
        catch
            fprintf('Right hemisphere surface plotting failed, using scatter plot fallback...\n');
            scatter3(sourceVisualize_right.pos(:, 1), sourceVisualize_right.pos(:, 2), sourceVisualize_right.pos(:, 3), 50, sourceVisualize_right.pow, 'filled');
            colorbar;
            colormap(common_colormap);
            caxis(common_colorlim);
        end
        
        view(45, 30); % Right hemisphere view
        lighting gouraud;
        material dull;
        light('Position', [1, -1, 1], 'Style', 'infinite', 'Color', [0.4, 0.4, 0.4]);
        title(sprintf('Right Hemisphere (t=%.2fs)', time_windows(t)), 'FontSize', 14, 'FontWeight', 'bold');
        axis equal;
        axis tight;
        grid on;
        set(gca, 'FontSize', 12);
        
        % Overall title
        sgtitle(sprintf('Subject %d: PC Loadings Over Time (Top %d Components)', subjID, n_components), ...
            'FontSize', 16, 'FontWeight', 'bold');
        
        % Add time indicator
        text(0.5, 0.95, sprintf('Time: %.2fs', time_windows(t)), 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontSize', 14, 'FontWeight', 'bold', ...
            'BackgroundColor', 'white', 'EdgeColor', 'black');
        
        % Capture frame
        frame = getframe(fig_movie);
        writeVideo(v, frame);
    end
    
    % Close video writer
    close(v);
    fprintf('Movie saved as: %s\n', movie_filename);
    
else
    %% Create 3D scatter plot for time point 0 (stimulus onset)
    fprintf('Creating 3D scatter plot for time point 0...\n');

    % Find time point closest to 0
    [~, time_zero_idx] = min(abs(time_windows));
    fprintf('Using time point %.2fs (index %d)\n', time_windows(time_zero_idx), time_zero_idx);

% Get eigenvalues and eigenvectors for this time point
eigenvalues = dimensionality_results.all_eigenvalues{time_zero_idx};
eigenvectors = dimensionality_results.all_eigenvectors{time_zero_idx};

if isempty(eigenvalues) || isempty(eigenvectors)
    fprintf('No eigenvalue/eigenvector data available for time point %.2fs\n', time_windows(time_zero_idx));
else
    % Extract eigenvalues and sort
    eigenvals = diag(eigenvalues);
    [sorted_eigenvals, sort_idx] = sort(eigenvals, 'descend');
    sorted_eigenvectors = eigenvectors(:, sort_idx);
    
    % Keep only positive eigenvalues
    positive_idx = sorted_eigenvals > 0;
    positive_eigenvals = sorted_eigenvals(positive_idx);
    positive_eigenvectors = sorted_eigenvectors(:, positive_idx);
    
    % Use top 50 components (or all positive if fewer than 50)
    n_components = min(50, length(positive_eigenvals));
    top_eigenvals = positive_eigenvals(1:n_components);
    top_eigenvectors = positive_eigenvectors(:, 1:n_components);
    
    % Calculate PC loadings (sum of squared loadings weighted by eigenvalues)
    pc_loadings = sum((top_eigenvectors.^2) .* top_eigenvals', 2);

    % Create surface visualization using FieldTrip
    fprintf('Creating surface visualization for left and right hemispheres...\n');
    
    % Z-score the PC loadings
    pc_loadings_zscore = (pc_loadings - mean(pc_loadings)) / std(pc_loadings);
    
    % Add z-scored PC loadings to hemisphere structures
    sourceVisualize_left.pow = pc_loadings_zscore(left_hemisphere_idx);
    sourceVisualize_right.pow = pc_loadings_zscore(right_hemisphere_idx);
    
    % Create figure with subplots for both hemispheres
    fig2 = figure('Position', [200, 200, 1600, 800]);
    
    % Set common colormap and color limits
    common_colormap = '*RdBu';
    common_colorlim = [-2, 2];
    
    % Left hemisphere
    subplot(1, 2, 1);
    cfg = [];
    cfg.method = 'surface';
    cfg.figure = 'gcf';
    cfg.funparameter = 'pow';
    cfg.maskparameter = "";
    cfg.surffile = sprintf('cortex_%d.surf.gii', surface_resolution);
    cfg.colorbar = 'no';  % No colorbar for left hemisphere
    cfg.funcolormap = common_colormap;
    cfg.funcolorlim = common_colorlim;
    
    try
        ft_sourceplot(cfg, sourceVisualize_left);
    catch
        fprintf('Left hemisphere surface plotting failed, using scatter plot fallback...\n');
        scatter3(sourceVisualize_left.pos(:, 1), sourceVisualize_left.pos(:, 2), sourceVisualize_left.pos(:, 3), 50, sourceVisualize_left.pow, 'filled');
        colorbar;
        colormap('hot');
    end
    
    view(-45, 30); % Left hemisphere view
    lighting gouraud;
    material dull;
    light('Position', [-1, -1, 1], 'Style', 'infinite', 'Color', [0.4, 0.4, 0.4]);
    title('Left Hemisphere', 'FontSize', 14, 'FontWeight', 'bold');
    axis equal;
    axis tight;
    grid on;
    set(gca, 'FontSize', 12);
    
    % Right hemisphere
    subplot(1, 2, 2);
    cfg.colorbar = 'yes';  % Show colorbar only on right hemisphere
    cfg.funcolormap = common_colormap;
    cfg.funcolorlim = common_colorlim;
    
    try
        ft_sourceplot(cfg, sourceVisualize_right);
    catch
        fprintf('Right hemisphere surface plotting failed, using scatter plot fallback...\n');
        scatter3(sourceVisualize_right.pos(:, 1), sourceVisualize_right.pos(:, 2), sourceVisualize_right.pos(:, 3), 50, sourceVisualize_right.pow, 'filled');
        colorbar;
        colormap(common_colormap);
    end
    
    view(45, 30); % Right hemisphere view
    lighting gouraud;
    material dull;
    light('Position', [1, -1, 1], 'Style', 'infinite', 'Color', [0.4, 0.4, 0.4]);
    title('Right Hemisphere', 'FontSize', 14, 'FontWeight', 'bold');
    axis equal;
    axis tight;
    grid on;
    set(gca, 'FontSize', 12);
    
    % Overall title
    sgtitle(sprintf('Subject %d: PC Loadings at %.2fs (Top %d Components)', subjID, time_windows(time_zero_idx), n_components), ...
        'FontSize', 16, 'FontWeight', 'bold');

    end
end

