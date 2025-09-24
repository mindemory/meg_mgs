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

%% Load data for multiple subjects, 5124 vertices
subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32];
surface_resolution = 5124;

% Initialize aggregation variables
all_effective_dimensionality = [];
all_pc_loadings = {};  % Store PC loadings instead of raw eigenvalues/eigenvectors
loaded_subjects = [];

fprintf('Loading data for %d subjects...\n', length(subjects));

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

% Loop over subjects to load and aggregate data
for s = 1:length(subjects)
    subjID = subjects(s);
    fprintf('Loading data for Subject %d, Resolution %d...\n', subjID, surface_resolution);
        
        % Construct file path
        result_file = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
            'dimensionality_analysis', sprintf('S05_dimensionality_sub%02d_%d.mat', subjID, surface_resolution));
        
    if ~exist(result_file, 'file')
        fprintf('Warning: Dimensionality analysis file not found for Subject %d: %s\n', subjID, result_file);
        continue;
    end
        
    fprintf('Loading from: %s\n', result_file);
                load(result_file, 'dimensionality_results');
                
    % Extract data for this subject
    if s == 1
        % First subject - initialize time_windows
        time_windows = dimensionality_results.time_windows;
        fprintf('Loaded %d time points\n', length(time_windows));
        fprintf('Time range: %.2f to %.2f seconds\n', min(time_windows), max(time_windows));
    end
    
    % Store effective dimensionality
    all_effective_dimensionality = [all_effective_dimensionality; dimensionality_results.effective_dimensionality'];
    
    % Project to PC space and store only PC loadings (memory optimization)
    fprintf('  Projecting to PC space for subject %d...\n', subjID);
    subject_pc_loadings = cell(length(dimensionality_results.all_eigenvalues), 1);
    
    for t = 1:length(dimensionality_results.all_eigenvalues)
        eigenvalues = dimensionality_results.all_eigenvalues{t};
        eigenvectors = dimensionality_results.all_eigenvectors{t};
        
        if isempty(eigenvalues) || isempty(eigenvectors)
            subject_pc_loadings{t} = [];
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
        
        % Use top 100 components (or all positive if fewer than 100)
        n_components = min(50, length(positive_eigenvals));
        top_eigenvals = positive_eigenvals(1:n_components);
        top_eigenvectors = positive_eigenvectors(:, 1:n_components);
        
        % Calculate PC loadings for this time point (sum of squared loadings weighted by eigenvalues)
        pc_loadings = sum((top_eigenvectors.^2) .* top_eigenvals', 2);
        subject_pc_loadings{t} = pc_loadings;
    end
    
    % Store PC loadings for this subject
    all_pc_loadings{end+1} = subject_pc_loadings;
    
    % Clear large matrices to save memory
    clear eigenvalues eigenvectors eigenvals sorted_eigenvals sorted_eigenvectors;
    clear positive_eigenvals positive_eigenvectors top_eigenvals top_eigenvectors;
    
    % Track loaded subjects
    loaded_subjects = [loaded_subjects, subjID];
                
    fprintf('Subject %d: Dimensionality range %.2f to %.2f\n', subjID, ...
        min(dimensionality_results.effective_dimensionality), max(dimensionality_results.effective_dimensionality));
end

% Compute group statistics
n_subjects_loaded = length(loaded_subjects);
fprintf('\nSuccessfully loaded %d subjects: %s\n', n_subjects_loaded, mat2str(loaded_subjects));

% Report memory optimization
fprintf('\nMemory optimization applied:\n');
fprintf('  - Stored only PC loadings instead of raw eigenvalues/eigenvectors\n');
fprintf('  - Cleared large matrices after projection\n');
fprintf('  - Reduced memory footprint significantly\n');

if n_subjects_loaded == 0
    error('No subjects loaded! Please check that S05_DimensionalityAnalysis has been run.');
end

% Compute mean and SEM for effective dimensionality
mean_effective_dimensionality = mean(all_effective_dimensionality, 1);
sem_effective_dimensionality = std(all_effective_dimensionality, 0, 1) / sqrt(n_subjects_loaded);

fprintf('Group statistics:\n');
fprintf('Mean dimensionality range: %.2f to %.2f\n', min(mean_effective_dimensionality), max(mean_effective_dimensionality));
fprintf('SEM range: %.2f to %.2f\n', min(sem_effective_dimensionality), max(sem_effective_dimensionality));

%% Create simple plot
fprintf('Creating plot...\n');

figure('Position', [100, 100, 1000, 600]);

% Plot the mean dimensionality over time with SEM shading
plot(time_windows, mean_effective_dimensionality, 'b-o', 'LineWidth', 2, 'MarkerSize', 6);

hold on;
% Add SEM shading
fill([time_windows, fliplr(time_windows)], ...
     [mean_effective_dimensionality + sem_effective_dimensionality, ...
      fliplr(mean_effective_dimensionality - sem_effective_dimensionality)], ...
     'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none');

% Add stimulus onset marker
xline(0, 'k--', 'LineWidth', 2, 'Alpha', 0.8, 'Label', 'Stimulus Onset');

% Formatting
xlabel('Time (s)', 'FontSize', 14);
ylabel('Effective Dimensionality (Participation Ratio)', 'FontSize', 14);
title(sprintf('Group Average: Effective Dimensionality Over Time (n=%d, Surface Resolution %d)', n_subjects_loaded, surface_resolution), ...
    'FontSize', 16, 'FontWeight', 'bold');
grid on;
xlim([min(time_windows), max(time_windows)]);

set(gca, 'FontSize', 12);

%% Create movie of PC loadings over time
fprintf('Creating movie of PC loadings over time...\n');

output_dir = pwd;
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
        
        % Get pre-computed PC loadings for each subject at this time point
        time_pc_loadings = [];
        
        for s = 1:length(loaded_subjects)
            % Get pre-computed PC loadings for this subject and time point
            pc_loadings = all_pc_loadings{s}{t};
            
            if isempty(pc_loadings)
                fprintf('No PC data for subject %d, time point %.2fs, skipping...\n', loaded_subjects(s), time_windows(t));
                continue;
            end
            
            % Store PC loadings for this subject
            time_pc_loadings = [time_pc_loadings, pc_loadings];
        end
        
        % Average PC loadings across subjects
        if ~isempty(time_pc_loadings)
            mean_pc_loadings = mean(time_pc_loadings, 2);
        else
            fprintf('No valid data for time point %.2fs, skipping...\n', time_windows(t));
            continue;
        end
        
        % Z-score the averaged PC loadings
        mean_pc_loadings_zscore = (mean_pc_loadings - mean(mean_pc_loadings)) / std(mean_pc_loadings);
        
        % Add z-scored PC loadings to hemisphere structures
        sourceVisualize_left.pow = mean_pc_loadings_zscore(left_hemisphere_idx);
        sourceVisualize_right.pow = mean_pc_loadings_zscore(right_hemisphere_idx);
        
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
        sgtitle(sprintf('Group Average: PC Loadings Over Time (n=%d, Top 100 Components)', n_subjects_loaded), ...
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

