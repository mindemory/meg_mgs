function S05_DimensionalityAnalysis(subjID, surface_resolution)
% S05_DimensionalityAnalysis - Analyze dimensionality of beta power data
%
% This script performs eigen decomposition on beta power data at each time point
% to estimate the intrinsic dimensionality of the neural data.
%
% Inputs:
%   subjID - Subject ID (e.g., 1, 2, 3, ...)
%   surface_resolution - Surface resolution (5124 or 8196)
%
% Outputs:
%   - Saves results to MAT file
%   - Creates visualization plots
%   - Saves figures as FIG and PNG files
%
% Example:
%   S05_DimensionalityAnalysis(1, 5124);

if nargin < 2
    surface_resolution = 5124;
end

[~, hostname] = system('hostname');
hostname = strtrim(hostname);

% Check for common HPC indicators
is_hpc = contains(hostname, {'login', 'compute', 'node', 'hpc'}) || ...
    exist('/etc/slurm', 'dir') || ...
    ~isempty(getenv('SLURM_JOB_ID')) || ...
    ~isempty(getenv('PBS_JOBID'));

%% Initialize
fprintf('\n=== MEG Beta Power Dimensionality Analysis ===\n');
fprintf('Environment: %s\n', getenv('HOSTNAME'));

% Check available memory
if ispc
    [~, meminfo] = memory;
    fprintf('Available memory: %.1f GB\n', meminfo.PhysicalMemory.Available / 1e9);
else
    fprintf('Memory check not available on this system\n');
end
fprintf('Detected HPC: %s\n', string(~isempty(getenv('SLURM_JOB_ID'))));
fprintf('Subject: %d\n', subjID);
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Define time windows for analysis (-1.0 to 2.0s with 100ms steps w/ moving window of 200ms)
time_start = -1.0;
time_end = 2;
time_step = 0.1; % 100ms
time_windows = time_start:time_step:time_end;

% Define target locations
target_locations = 1:10;

%% Set up paths
if is_hpc
    % HPC paths
    data_base_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC/derivatives';
    project_path = '/scratch/mdd9787/meg_prf_greene/MEG_HPC';
    fieldtrip_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318';
    ft_gifti_path = '/scratch/mdd9787/meg_prf_greene/fieldtrip-20250318/external/gifti';
else
    % Local paths
    data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318';
    project_path = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
    ft_gifti_path = '/d/DATD/hyper/software/fieldtrip-20250318/external/gifti';
end

% Add paths
addpath(fieldtrip_path);
addpath(ft_gifti_path);
addpath(genpath(project_path));
ft_defaults;
ft_hastoolbox('spm12', 1);

%% Load Data
fprintf('Loading data for subject %d, surface resolution %d...\n', subjID, surface_resolution);

% Load complex beta data
beta_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_complexBeta_allTargets_%d.mat', subjID, surface_resolution));

if ~exist(beta_data_path, 'file')
    error('Complex beta data not found at: %s\nPlease run S03_betaPowerInSource.m first!', beta_data_path);
end

fprintf('Loading complex beta data from: %s\n', beta_data_path);
load(beta_data_path);

% Load forward model for source positions
forward_model_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_forwardModel.mat', subjID));

if ~exist(forward_model_path, 'file')
    error('Forward model not found at: %s\nPlease run S01_ForwardModelMNI.m first!', forward_model_path);
end

fprintf('Loading forward model from: %s\n', forward_model_path);
load(forward_model_path);

% Create output directory
output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'dimensionality_analysis');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

dataFile = fullfile(output_dir, sprintf('S05_dimensionality_sub%02d_%d.mat', subjID, surface_resolution));

if ~exist(dataFile, 'file')
    fprintf('Output file does not exist, proceeding with analysis...\n');


    %% Collect All Trial Data Efficiently
    fprintf('Collecting power data for all trials and time points...\n');

    % First pass: count total trials to preallocate
    total_trials = 0;
    for target = target_locations
        if ~isempty(sourceDataByTarget{target})
            total_trials = total_trials + length(sourceDataByTarget{target}.trial);
        end
    end

    % Preallocate 3D arrays: [surface_resolution, n_trials, n_time_windows]
    all_trial_data = zeros(surface_resolution, total_trials, length(time_windows), 'single');
    all_target_labels = zeros(total_trials, 1);
    trial_count = 0;

    % Process each target location
    for target = target_locations
        fprintf('Processing target %d...\n', target);

        if isempty(sourceDataByTarget{target})
            fprintf('  No data for target %d\n', target);
            continue;
        end

        % Get the complex beta data for this target
        sourcedataTarget = sourceDataByTarget{target};
        n_trials = length(sourcedataTarget.trial);

        % Process each trial
        for trial = 1:n_trials
            trial_count = trial_count + 1;

            % Compute power (magnitude squared)
            power_data = abs(sourcedataTarget.trial{trial}).^2;

            % Process each time window
            for tw = 1:length(time_windows)
                window_start = time_windows(tw) - time_step;
                window_end = time_windows(tw) + time_step;

                % Find time indices for this window
                time_vec = sourcedataTarget.time{1};
                time_idx = time_vec >= window_start & time_vec < window_end;

                if sum(time_idx) > 0
                    % Average power across this time window
                    window_power = mean(power_data(:, time_idx), 2);

                    % Store data in 3D array
                    all_trial_data(:, trial_count, tw) = single(window_power);
                end
            end

            % Store target label
            all_target_labels(trial_count) = target;
        end

        fprintf('  Target %d: %d trials processed\n', target, n_trials);
    end

    %% Compute Relative Power Efficiently
    fprintf('Computing relative power...\n');

    global_avg_power_time = mean(all_trial_data, 2);
    % Compute relative power for all trials at once
    relative_power_data = all_trial_data ./ global_avg_power_time;

    fprintf('Relative power computation complete\n');

    %% Compute Effective Dimensionality as a Function of Time
    fprintf('Computing effective dimensionality as a function of time...\n');

    % Initialize arrays to store dimensionality results
    n_time_windows = length(time_windows);
    effective_dimensionality = zeros(n_time_windows, 1);

    % Initialize cell arrays to store eigenvalues and eigenvectors for each time window
    all_eigenvalues = cell(n_time_windows, 1);
    all_eigenvectors = cell(n_time_windows, 1);
    all_normalized_eigenvals = cell(n_time_windows, 1);

    % Process each time window
    fprintf('Processing %d time windows...\n', n_time_windows);

    for tw = 1:n_time_windows
        fprintf('Processing time window %d/%d (%.2fs)...\n', tw, n_time_windows, time_windows(tw));
        % Get data for this time window: [n_sources × n_trials]
        time_data = relative_power_data(:, :, tw);

        % Remove any NaN values
        valid_trials = ~any(isnan(time_data), 1);
        if sum(valid_trials) < 2
            effective_dimensionality(tw) = NaN;
            all_eigenvalues{tw} = [];
            all_eigenvectors{tw} = [];
            all_normalized_eigenvals{tw} = [];
            continue;
        end

        time_data_clean = time_data(:, valid_trials);

        % Compute covariance matrix
        cov_matrix = cov(time_data_clean');

        % Perform eigendecomposition
        [eigenvectors, eigenvalues] = eig(cov_matrix);

        % Store original eigenvalues and eigenvectors
        all_eigenvalues{tw} = eigenvalues;
        all_eigenvectors{tw} = eigenvectors;

        eigenvalues = diag(eigenvalues);

        % Sort eigenvalues in descending order
        [eigenvalues, sort_idx] = sort(eigenvalues, 'descend');
        eigenvectors = eigenvectors(:, sort_idx);

        % Remove negative eigenvalues (numerical artifacts)
        positive_eigenvals = eigenvalues(eigenvalues > 0);

        if isempty(positive_eigenvals)
            effective_dimensionality(tw) = NaN;
            all_normalized_eigenvals{tw} = [];
            continue;
        end

        % Normalize eigenvalues
        normalized_eigenvals = positive_eigenvals / sum(positive_eigenvals);
        all_normalized_eigenvals{tw} = normalized_eigenvals;

        % Compute effective dimensionality (participation ratio)
        effective_dimensionality(tw) = 1 / sum(normalized_eigenvals.^2);

        fprintf('  Time %.2fs: Effective dimensionality = %.2f\n', time_windows(tw), effective_dimensionality(tw));
    end

    %% Create dimensionality visualization
    fprintf('Creating dimensionality visualization...\n');

    figure('Position', [100, 100, 800, 600]);
    plot(time_windows, effective_dimensionality, 'b-', 'LineWidth', 2);
    xlabel('Time (s)');
    ylabel('Effective Dimensionality');
    title(sprintf('Subject %02d: Effective Dimensionality Over Time (Surface Resolution %d)', subjID, surface_resolution));
    grid on;
    xline(0, 'k--', 'Stimulus Onset', 'LineWidth', 1);

    %% Create 3D scatter plot of PC loadings for time period 0.5-1.5s
    % fprintf('Creating 3D scatter plot of PC loadings...\n');
    %
    % % Find time windows in the 0.5-1.5s range
    % target_time_start = 0.1;
    % target_time_end = 0.3;
    % time_indices = find(time_windows >= target_time_start & time_windows <= target_time_end);
    %
    % if ~isempty(time_indices)
    %     % Average PC loadings across the time period
    %     fprintf('Averaging PC loadings from %.1fs to %.1fs...\n', target_time_start, target_time_end);
    %
    %     % Initialize arrays to store averaged loadings
    %     n_sources = surface_resolution;
    %     n_components = 50; % Use first 50 components
    %     avg_pc_loadings = zeros(n_sources, 1);
    %
    %     % Average PC loadings across time windows with eigenvalue weighting
    %     for tw_idx = 1:length(time_indices)
    %         tw = time_indices(tw_idx);
    %
    %         if ~isempty(all_eigenvectors{tw}) && ~isempty(all_eigenvalues{tw})
    %             % Get eigenvectors and eigenvalues
    %             eigenvectors = all_eigenvectors{tw};
    %             eigenvalues = diag(all_eigenvalues{tw});
    %
    %             % Sort eigenvalues in descending order and get sort indices
    %             [sorted_eigenvals, sort_idx] = sort(eigenvalues, 'descend');
    %
    %             % Sort eigenvectors according to eigenvalue order
    %             sorted_eigenvectors = eigenvectors(:, sort_idx);
    %
    %             n_comp_available = min(n_components, size(sorted_eigenvectors, 2));
    %
    %             % Get eigenvalues for the first n_comp_available components
    %             comp_eigenvalues = sorted_eigenvals(1:n_comp_available);
    %
    %             % Compute weighted PC loadings (eigenvalue-weighted sum of squared loadings)
    %             weighted_pc_loadings = abs(sum((sorted_eigenvectors(:, 1:n_comp_available).^2) .* comp_eigenvalues', 2));
    %             avg_pc_loadings = avg_pc_loadings + weighted_pc_loadings;
    %         end
    %     end
    %
    %     % Average across time windows
    %     avg_pc_loadings = avg_pc_loadings / length(time_indices);
    %
    %     % Create 3D scatter plot
    %     figure('Position', [200, 200, 1000, 800]);
    %
    %     % Load source positions for visualization
    %     source_file = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    %         sprintf('sub-%02d_task-mgs_sourceSpaceData_%d.mat', subjID, surface_resolution));
    %
    %     if exist(source_file, 'file')
    %         load(source_file, "sourcemodel_aligned_5124");
    %         pos = sourcemodel_aligned_5124.pos;
    %
    %
    %         % Create 3D scatter plot with PC loadings as color
    %         scatter3(pos(:, 1), pos(:, 2), pos(:, 3), 50, avg_pc_loadings, 'filled');
    %         xlabel('X (mm)');
    %         ylabel('Y (mm)');
    %         zlabel('Z (mm)');
    %         title(sprintf('Subject %02d: PC Loadings (0.5-1.5s, Surface Resolution %d)', subjID, surface_resolution));
    %         caxis([ quantile(avg_pc_loadings, 0.05)  quantile(avg_pc_loadings, 0.95)]);
    %         colorbar;
    %         colormap('jet');
    %         axis equal;
    %         view(45, 30);
    %
    %         % Add some statistics
    %         fprintf('PC loading statistics:\n');
    %         fprintf('  Mean: %.4f\n', mean(avg_pc_loadings));
    %         fprintf('  Std: %.4f\n', std(avg_pc_loadings));
    %         fprintf('  Min: %.4f\n', min(avg_pc_loadings));
    %         fprintf('  Max: %.4f\n', max(avg_pc_loadings));
    %
    %     else
    %         fprintf('Warning: Source positions not found, skipping 3D visualization\n');
    %     end
    % else
    %     fprintf('Warning: No time windows found in the 0.5-1.5s range\n');
    % end

    %% Save results
    fprintf('Saving dimensionality results...\n');
    dimensionality_results = struct();
    dimensionality_results.time_windows = time_windows;
    dimensionality_results.effective_dimensionality = effective_dimensionality;
    dimensionality_results.subject_id = subjID;
    dimensionality_results.surface_resolution = surface_resolution;

    % Save eigenvalues and eigenvectors for PCA analysis
    dimensionality_results.all_eigenvalues = all_eigenvalues;
    dimensionality_results.all_eigenvectors = all_eigenvectors;
    dimensionality_results.all_normalized_eigenvals = all_normalized_eigenvals;

    % Additional metadata for PCA analysis
    dimensionality_results.n_sources = surface_resolution;
    dimensionality_results.n_time_windows = n_time_windows;
    dimensionality_results.total_trials = total_trials;

    % Save PC loadings if computed
    if exist('avg_pc_loadings', 'var')
        dimensionality_results.avg_pc_loadings = avg_pc_loadings;
        dimensionality_results.pc_loading_time_range = [target_time_start, target_time_end];
    end
    % Save results
    save(dataFile, 'dimensionality_results', '-v7.3');

    % Save figures
    savefig(fullfile(output_dir, sprintf('S05_dimensionality_sub%02d_%d.fig', subjID, surface_resolution)));
    print(fullfile(output_dir, sprintf('S05_dimensionality_sub%02d_%d.png', subjID, surface_resolution)), '-dpng', '-r300');

    % Save PC loadings figure if created
    if exist('avg_pc_loadings', 'var')
        savefig(fullfile(output_dir, sprintf('S05_pc_loadings_sub%02d_%d.fig', subjID, surface_resolution)));
        print(fullfile(output_dir, sprintf('S05_pc_loadings_sub%02d_%d.png', subjID, surface_resolution)), '-dpng', '-r300');
    end

    fprintf('Dimensionality analysis complete!\n');

end