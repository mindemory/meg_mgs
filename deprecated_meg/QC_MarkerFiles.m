%% Quality Control: HPI Marker Files Analysis
% Automatically detects and visualizes HPI marker files for multiple subjects
% Helps determine optimal nMrkFiles parameter for each subject

clear; close all; clc;

%% Setup
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;

%% Configuration
% Process subjects in batches of 6
subject_batches = {
    [1, 2, 3, 4, 5, 6],      % Batch 1
    [7, 9, 10, 12, 13, 15],     % Batch 2  
    [17, 18, 19, 23, 24, 25], % Batch 3
    [26, 27, 29, 31, 32]  % Batch 4 (add more as needed)
};

% Select which batch to process
BATCH_TO_PROCESS = 4; % Change this to process different batches
subjects = subject_batches{BATCH_TO_PROCESS};

fprintf('=== HPI MARKER FILES QUALITY CONTROL ===\n');
fprintf('Processing batch %d: subjects %s\n', BATCH_TO_PROCESS, mat2str(subjects));

%% Process Each Subject
results = struct();

for subj_idx = 1:length(subjects)
    subjID = subjects(subj_idx);
    
    fprintf('\n--- Subject %02d ---\n', subjID);
    
    % Define paths
    subRoot = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-%02d/meg/sub-%02d_task-mgs_', subjID, subjID);
    
    % Find all marker files
    marker_pattern = sprintf('%smarker-*.sqd', subRoot);
    marker_files = dir(marker_pattern);
    
    if isempty(marker_files)
        fprintf('  ❌ No marker files found for subject %02d\n', subjID);
        results(subj_idx).subjID = subjID;
        results(subj_idx).n_markers = 0;
        results(subj_idx).hpi_positions = [];
        results(subj_idx).marker_quality = 'No files';
        continue;
    end
    
    n_markers = length(marker_files);
    fprintf('  📁 Found %d marker files\n', n_markers);
    
    % Load HPI positions from each marker file
    hpi_data = NaN(n_markers, 5, 3); % [n_files, 5_coils, xyz]
    
    % Also load headshape and fiducials
    try
        % Load headshape
        hspPath = sprintf('%sheadshape.hsp', subRoot);
        if exist(hspPath, 'file')
            hspData = ft_read_headshape(hspPath, 'unit', 'm');
            hspData.fid.label = {'Nasion', 'LPA', 'RPA'};
            
            % Special handling for subject 29 - truncate BEFORE unit conversion
            if subjID == 29 && size(hspData.pos, 1) > 149
                hspData.pos = hspData.pos(1:149, :) .* 1000;
                fprintf('  ✓ Headshape truncated to %d points for sub-29\n', size(hspData.pos, 1));
            end
            
            % Convert units to mm (applies to all subjects including truncated sub-29)
            hspData = ft_convert_units(hspData, 'mm');
            fprintf('  ✓ Headshape loaded: %d points\n', size(hspData.pos, 1));
        else
            hspData = [];
            fprintf('  ⚠️ Headshape file not found\n');
        end
        
        % Load fiducials from ELP file
        elpPath = sprintf('%selectrodes.elp', subRoot);
        if exist(elpPath, 'file')
            [fidData, ~] = readelpFile(elpPath);
            fidData = fidData * 1000; % Convert to mm
            fprintf('  ✓ Fiducials loaded from ELP\n');
        else
            fidData = [];
            fprintf('  ⚠️ ELP file not found\n');
        end
        
        % Load HPI positions from marker files
        for mrk_idx = 1:n_markers
            marker_file = fullfile(marker_files(mrk_idx).folder, marker_files(mrk_idx).name);
            hdr_mrk = ft_read_header(marker_file);
            
            if isfield(hdr_mrk.orig, 'coregist') && isfield(hdr_mrk.orig.coregist, 'hpi')
                hpi_pos = cat(1, hdr_mrk.orig.coregist.hpi.meg_pos);
                if size(hpi_pos, 1) == 5 && size(hpi_pos, 2) == 3
                    hpi_data(mrk_idx, :, :) = hpi_pos * 1000; % Convert to mm
                    fprintf('  ✓ Marker %d: HPI positions loaded\n', mrk_idx);
                else
                    fprintf('  ⚠️ Marker %d: Unexpected HPI data size\n', mrk_idx);
                end
            else
                fprintf('  ❌ Marker %d: No HPI data found\n', mrk_idx);
            end
        end
        
        % Calculate quality metrics
        hpi_mean = squeeze(nanmean(hpi_data, 1));
        hpi_std = squeeze(nanstd(hpi_data, 0, 1));
        max_movement = max(hpi_std(:));
        mean_movement = mean(hpi_std(:));
        
        % Quality assessment
        if max_movement < 2 % mm
            quality = 'Excellent';
        elseif max_movement < 5
            quality = 'Good';
        elseif max_movement < 10
            quality = 'Fair';
        else
            quality = 'Poor';
        end
        
        fprintf('  📊 Movement: Max=%.2f mm, Mean=%.2f mm (%s)\n', ...
            max_movement, mean_movement, quality);
        
        % Store results
        results(subj_idx).subjID = subjID;
        results(subj_idx).n_markers = n_markers;
        results(subj_idx).hpi_positions = hpi_data;
        results(subj_idx).hpi_mean = hpi_mean;
        results(subj_idx).hpi_std = hpi_std;
        results(subj_idx).max_movement = max_movement;
        results(subj_idx).mean_movement = mean_movement;
        results(subj_idx).marker_quality = quality;
        results(subj_idx).headshape = hspData;
        results(subj_idx).fiducials = fidData;
        
    catch ME
        fprintf('  ❌ Error processing subject %02d: %s\n', subjID, ME.message);
        results(subj_idx).subjID = subjID;
        results(subj_idx).n_markers = n_markers;
        results(subj_idx).hpi_positions = [];
        results(subj_idx).marker_quality = 'Error';
    end
end

%% 3D Visualization
figure('Name', sprintf('HPI Marker Quality 3D - Batch %d', BATCH_TO_PROCESS), 'Position', [100, 100, 1600, 1000]);

n_valid_subjects = sum([results.n_markers] > 0);
if n_valid_subjects == 0
    fprintf('\n❌ No valid subjects found in this batch.\n');
    return;
end

% Calculate subplot layout
n_cols = min(3, n_valid_subjects);
n_rows = ceil(n_valid_subjects / n_cols);

plot_idx = 1;
for subj_idx = 1:length(subjects)
    if results(subj_idx).n_markers == 0
        continue;
    end
    
    subjID = results(subj_idx).subjID;
    hpi_data = results(subj_idx).hpi_positions;
    headshape = results(subj_idx).headshape;
    fiducials = results(subj_idx).fiducials;
    
    if isempty(hpi_data)
        continue;
    end
    
    % 3D Plot
    subplot(n_rows, n_cols, plot_idx);
    hold on;
    
    % Plot headshape if available
    if ~isempty(headshape)
        plot3(headshape.pos(:,1), headshape.pos(:,2), headshape.pos(:,3), ...
              '.', 'Color', [0.7 0.7 0.7], 'MarkerSize', 1, 'DisplayName', 'Headshape');
        
        % Plot headshape fiducials if available
        if isfield(headshape, 'fid') && ~isempty(headshape.fid.pos)
            plot3(headshape.fid.pos(:,1), headshape.fid.pos(:,2), headshape.fid.pos(:,3), ...
                  'ks', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'HSP Fiducials');
        end
    end
    
    % Plot ELP fiducials if available
    if ~isempty(fiducials)
        plot3(fiducials(:,1), fiducials(:,2), fiducials(:,3), ...
              'bs', 'MarkerSize', 12, 'LineWidth', 2, 'DisplayName', 'ELP Fiducials');
    end
    
    % Plot HPI positions with different colors for each marker file
    colors = hsv(size(hpi_data, 1)); % Different colors for each marker file
    coil_names = {'LPA', 'RPA', 'NAS', 'HPI4', 'HPI5'};
    coil_shapes = {'o', 's', '^', 'd', 'v'};
    
    % Plot all HPI positions with labels
    for mrk_idx = 1:size(hpi_data, 1)
        for coil = 1:5
            x_pos = hpi_data(mrk_idx, coil, 1);
            y_pos = hpi_data(mrk_idx, coil, 2);
            z_pos = hpi_data(mrk_idx, coil, 3);
            
            % Skip if NaN
            if isnan(x_pos) || isnan(y_pos) || isnan(z_pos)
                continue;
            end
            
            % Plot HPI coil position (smaller markers)
            plot3(x_pos, y_pos, z_pos, coil_shapes{coil}, 'Color', colors(mrk_idx, :), ...
                  'MarkerSize', 6, 'LineWidth', 1.5, 'MarkerFaceColor', colors(mrk_idx, :), ...
                  'HandleVisibility', 'off');
            
            % Add text label showing marker index and coil name
            text(x_pos, y_pos, z_pos, sprintf('M%d-%s', mrk_idx, coil_names{coil}), ...
                 'FontSize', 7, 'Color', colors(mrk_idx, :), 'FontWeight', 'bold', ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    end
    
    % Add lines connecting the same coil across marker files to show movement
    for coil = 1:5
        coil_positions = squeeze(hpi_data(:, coil, :));
        valid_idx = ~any(isnan(coil_positions), 2);
        
        if sum(valid_idx) > 1
            plot3(coil_positions(valid_idx, 1), coil_positions(valid_idx, 2), coil_positions(valid_idx, 3), ...
                  '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 1, 'HandleVisibility', 'off');
        end
    end
    
    % Formatting
    xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
    title(sprintf('Sub-%02d: %s (Max: %.1fmm)', subjID, results(subj_idx).marker_quality, ...
                  results(subj_idx).max_movement));
    grid on; axis equal;
    view(3); % 3D view
    
    % Color-code title based on quality
    ax = gca;
    switch results(subj_idx).marker_quality
        case 'Excellent'
            ax.Title.Color = [0 0.7 0]; % Green
        case 'Good' 
            ax.Title.Color = [0 0.5 1]; % Blue
        case 'Fair'
            ax.Title.Color = [1 0.7 0]; % Orange
        case 'Poor'
            ax.Title.Color = [1 0 0]; % Red
    end
    
    % Add legend for first subplot only to avoid clutter
    if plot_idx == 1
        legend('Location', 'best', 'FontSize', 8);
        
        % Add custom legend for marker colors
        legend_text = {};
        for m = 1:size(hpi_data, 1)
            legend_text{end+1} = sprintf('Marker %d', m);
        end
        
        % Create dummy plots for marker color legend
        for m = 1:size(hpi_data, 1)
            h_dummy = plot3(NaN, NaN, NaN, 'o', 'Color', colors(m, :), ...
                           'MarkerFaceColor', colors(m, :), 'MarkerSize', 6, ...
                           'DisplayName', sprintf('Marker %d', m));
        end
    end
    
    plot_idx = plot_idx + 1;
end

%% Summary Table
fprintf('\n=== BATCH %d SUMMARY ===\n', BATCH_TO_PROCESS);
fprintf('Subject | Files | Quality   | Max Move | Recommendation\n');
fprintf('--------|-------|-----------|----------|---------------\n');

for subj_idx = 1:length(subjects)
    subjID = results(subj_idx).subjID;
    n_files = results(subj_idx).n_markers;
    quality = results(subj_idx).marker_quality;
    
    if n_files == 0
        fprintf(' %02d     |   0   | No files  |   N/A    | ❌ Check data\n', subjID);
        continue;
    end
    
    if isempty(results(subj_idx).hpi_positions)
        fprintf(' %02d     |  %2d   | Error     |   N/A    | ❌ Check files\n', subjID, n_files);
        continue;
    end
    
    max_move = results(subj_idx).max_movement;
    
    % Recommendation based on quality and number of files
    if strcmp(quality, 'Excellent') || strcmp(quality, 'Good')
        if n_files >= 3
            rec = sprintf('✅ Use all %d', n_files);
        else
            rec = sprintf('✅ Use %d', n_files);
        end
    elseif strcmp(quality, 'Fair')
        if n_files >= 3
            rec = '⚠️ Use best 2-3';
        else
            rec = sprintf('⚠️ Use %d', n_files);
        end
    else % Poor
        rec = '❌ Manual check';
    end
    
    fprintf(' %02d     |  %2d   | %-9s | %7.2f  | %s\n', subjID, n_files, quality, max_move, rec);
end

fprintf('\n🎯 Next steps:\n');
fprintf('1. Review the 3D plots and summary table\n');
fprintf('2. Update nMrkFiles in S01_ForwardModelMNI.m based on recommendations\n');
fprintf('3. Process next batch by changing BATCH_TO_PROCESS\n');
