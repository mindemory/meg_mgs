%% Analyze Head Model to Head Shape Distance
% Computes mean perpendicular distance between head model and head shape for each subject
% and plots results with error bars

restoredefaultpath;
clear; close all; clc;

%% Setup and Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;

fprintf('=== Head Model to Head Shape Distance Analysis ===\n');

%% Find all subjects with forward models
derivatives_dir = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
forward_model_files = dir(fullfile(derivatives_dir, 'sub-*/sourceRecon/*forwardModel.mat'));

% Extract subject IDs from file paths
subject_ids = [];
for i = 1:length(forward_model_files)
    file_path = forward_model_files(i).folder;
    subj_match = regexp(file_path, 'sub-(\d+)', 'tokens');
    if ~isempty(subj_match)
        subject_ids(end+1) = str2double(subj_match{1}{1});
    end
end
subject_ids = sort(subject_ids);

fprintf('Found %d subjects with forward models: %s\n', length(subject_ids), mat2str(subject_ids));

%% Initialize storage arrays
mean_distances = zeros(length(subject_ids), 1);
std_distances = zeros(length(subject_ids), 1);
valid_subjects = [];

%% Process each subject
for i = 1:length(subject_ids)
    subjID = subject_ids(i);
    
    fprintf('\nProcessing Subject %02d...\n', subjID);
    
    % Load forward model
    forward_model_file = fullfile(derivatives_dir, sprintf('sub-%02d/sourceRecon/sub-%02d_task-mgs_forwardModel.mat', subjID, subjID));
    
    if ~exist(forward_model_file, 'file')
        fprintf('  Warning: Forward model file not found for subject %02d\n', subjID);
        continue;
    end
    
    try
        load(forward_model_file);
        
        % Check if required data exists
        if ~exist('singleShellHeadModel', 'var') || ~exist('hspCorrected_ras', 'var')
            fprintf('  Warning: Missing head model or head shape data for subject %02d\n', subjID);
            continue;
        end
        
        % Extract head model surface points
        head_model_points = singleShellHeadModel.bnd.pos;
        
        % Extract head shape points
        if isstruct(hspCorrected_ras)
            head_shape_points = hspCorrected_ras.pos;
        else
            head_shape_points = hspCorrected_ras;
        end
        
        fprintf('  Head model points: %d\n', size(head_model_points, 1));
        fprintf('  Head shape points: %d\n', size(head_shape_points, 1));
        
        % Compute distances from each head shape point to the nearest head model point
        % This is a simplified approach - for true perpendicular distance, we'd need
        % to compute distance to the surface, but this gives a good approximation
        distances = zeros(size(head_shape_points, 1), 1);
        
        for j = 1:size(head_shape_points, 1)
            % Find distance to nearest head model point
            point = head_shape_points(j, :);
            dists_to_model = sqrt(sum((head_model_points - point).^2, 2));
            distances(j) = min(dists_to_model);
        end
        
        % Compute mean and standard deviation
        mean_dist = mean(distances);
        std_dist = std(distances);
        
        mean_distances(i) = mean_dist;
        std_distances(i) = std_dist;
        valid_subjects(end+1) = subjID;
        
        fprintf('  Mean distance: %.2f mm (SD: %.2f mm)\n', mean_dist, std_dist);
        
    catch ME
        fprintf('  Error processing subject %02d: %s\n', subjID, ME.message);
        continue;
    end
end

%% Remove invalid entries
valid_idx = mean_distances > 0;
mean_distances = mean_distances(valid_idx);
std_distances = std_distances(valid_idx);

fprintf('\n=== Results Summary ===\n');
fprintf('Successfully processed %d subjects\n', length(valid_subjects));

if length(valid_subjects) == 0
    fprintf('No valid subjects found. Exiting.\n');
    return;
end

fprintf('Overall mean distance: %.2f mm (SD: %.2f mm)\n', mean(mean_distances), std(mean_distances));

%% Create visualization
figure('Position', [100, 100, 1200, 600]);

% Main plot
errorbar(1:length(valid_subjects), mean_distances, std_distances, 'o', ...
    'MarkerSize', 8, 'MarkerFaceColor', 'blue', 'MarkerEdgeColor', 'blue', ...
    'LineWidth', 2, 'CapSize', 10);
xlabel('Subject Index');
ylabel('Mean Distance (mm)');
title('Head Model to Head Shape Distance');
grid on;
set(gca, 'FontSize', 12);

% Add subject IDs as x-axis labels
set(gca, 'XTick', 1:length(valid_subjects));
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('S%02d', x), valid_subjects, 'UniformOutput', false));
xtickangle(45);

%% Display detailed results table
fprintf('\n=== Detailed Results ===\n');
fprintf('Subject\tMean Dist (mm)\tStd Dist (mm)\n');
fprintf('-------\t--------------\t-------------\n');
for i = 1:length(valid_subjects)
    fprintf('S%02d\t%.2f\t\t%.2f\n', valid_subjects(i), mean_distances(i), std_distances(i));
end

fprintf('\nAnalysis complete!\n');
