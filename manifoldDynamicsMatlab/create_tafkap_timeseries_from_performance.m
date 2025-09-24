function create_tafkap_timeseries_from_performance()
% CREATE_TAFKAP_TIMESERIES_FROM_PERFORMANCE
% Creates time series summary files from individual TAFKAP PCA performance files
%
% This script aggregates individual TAFKAP PCA performance files into time series
% summary files that can be used by Fs07_visualizeTAFKAPPCATimeSeries.m

fprintf('Creating TAFKAP PCA time series from individual performance files...\n');

% Define subjects and surface resolutions
subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32];
surface_resolutions = [5124, 8196];

% Define time points (from -0.5 to 1.7s with 0.1s steps)
time_points = -0.5:0.1:1.7;
n_timepoints = length(time_points);

% Set up paths
if ispc
    % HPC paths
    base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    tafkap_path = '/d/DATD/hyper/toolboxes/TAFKAP';
else
    % Local paths
    base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
    tafkap_path = '/d/DATD/hyper/toolboxes/TAFKAP';
end

% Add TAFKAP to path
addpath(tafkap_path);

% Process each subject and surface resolution
for subj_idx = 1:length(subjects)
    subjID = subjects(subj_idx);
    
    for surf_idx = 1:length(surface_resolutions)
        surface_resolution = surface_resolutions(surf_idx);
        
        fprintf('Processing Subject %d, surface resolution %d...\n', subjID, surface_resolution);
        
        % Initialize arrays for this subject/resolution
        mean_errors = NaN(1, n_timepoints);
        std_errors = NaN(1, n_timepoints);
        success_flags = false(1, n_timepoints);
        
        % Process each time point
        for t_idx = 1:n_timepoints
            time_point = time_points(t_idx);
            
            % Construct filename
            filename = sprintf('sub-%02d_tafkap_pca_performance_%d_t%.1f.mat', ...
                subjID, surface_resolution, time_point);
            filepath = fullfile(base_path, sprintf('sub-%02d', subjID), ...
                'sourceRecon', 'tafkap_decoding', filename);
            
            % Check if file exists
            if exist(filepath, 'file')
                try
                    % Load performance data
                    load(filepath, 'performance');
                    
                    % Extract mean and std error
                    mean_errors(t_idx) = mean(performance.mean_error);
                    std_errors(t_idx) = mean(performance.std_error);
                    success_flags(t_idx) = true;
                    
                    fprintf('  Time %.1fs: Mean error = %.2f° ± %.2f°\n', ...
                        time_point, mean_errors(t_idx), std_errors(t_idx));
                    
                catch ME
                    fprintf('  Time %.1fs: Error loading file - %s\n', time_point, ME.message);
                    success_flags(t_idx) = false;
                end
            else
                fprintf('  Time %.1fs: File not found\n', time_point);
                success_flags(t_idx) = false;
            end
        end
        
        % Create summary results structure
        results = struct();
        results.subject = subjID;
        results.surface_resolution = surface_resolution;
        results.time_points = time_points;
        results.mean_error = mean_errors;
        results.std_error = std_errors;
        results.success_flags = success_flags;
        results.n_timepoints = n_timepoints;
        results.n_successful = sum(success_flags);
        results.success_rate = sum(success_flags) / n_timepoints;
        
        % Calculate summary statistics
        valid_errors = mean_errors(success_flags);
        if ~isempty(valid_errors)
            results.overall_mean_error = mean(valid_errors);
            results.overall_std_error = std(valid_errors);
            [~, best_idx] = min(abs(valid_errors));
            results.best_timepoint = time_points(success_flags);
            results.best_timepoint = results.best_timepoint(best_idx);
            results.best_error = valid_errors(best_idx);
        else
            results.overall_mean_error = NaN;
            results.overall_std_error = NaN;
            results.best_timepoint = NaN;
            results.best_error = NaN;
        end
        
        % Create output directory
        output_dir = fullfile(base_path, sprintf('sub-%02d', subjID), ...
            'sourceRecon', 'tafkap_pca_decoding');
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end
        
        % Save summary results
        summary_filename = sprintf('sub-%02d_tafkap_pca_timeseries_summary_%d.mat', ...
            subjID, surface_resolution);
        summary_filepath = fullfile(output_dir, summary_filename);
        
        save(summary_filepath, 'results');
        fprintf('  Summary saved to: %s\n', summary_filepath);
        
        % Print summary
        fprintf('  === TAFKAP PCA Time Series Summary ===\n');
        fprintf('  Subject: %d\n', subjID);
        fprintf('  Surface resolution: %d\n', surface_resolution);
        fprintf('  Time points analyzed: %d\n', n_timepoints);
        fprintf('  Successful analyses: %d/%d (%.1f%%)\n', ...
            results.n_successful, n_timepoints, results.success_rate * 100);
        if ~isnan(results.overall_mean_error)
            fprintf('  Best performance: %.2f° at %.1fs\n', ...
                results.best_error, results.best_timepoint);
            fprintf('  Overall mean error: %.2f° ± %.2f°\n', ...
                results.overall_mean_error, results.overall_std_error);
        else
            fprintf('  No successful analyses found!\n');
        end
        fprintf('\n');
    end
end

fprintf('TAFKAP PCA time series creation completed!\n');
end
