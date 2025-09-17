%% Helper Script to Run S03 Beta Power Analysis for Multiple Subjects
% Runs S03_betaPowerInSource.m for subjects 1, 2, and 3

restoredefaultpath;
clear; close all; clc;

%% Setup and Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);

%% Configuration
subjects = [1];
surface_resolution = 5124; % Use 5124 vertices for testing

fprintf('=== Running S03 Beta Power Analysis ===\n');
fprintf('Subjects: %s\n', mat2str(subjects));
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Run S03 for each subject
for i = 1:length(subjects)
    subjID = subjects(i);
    
    fprintf('\n=== Processing Subject %02d ===\n', subjID);
    
    try
        % Run S03_betaPowerInSource.m
        S03_betaPowerInSource(subjID, surface_resolution);
        
        fprintf('Subject %02d completed successfully!\n', subjID);
        
    catch ME
        fprintf('Error processing subject %02d: %s\n', subjID, ME.message);
        fprintf('Stack trace:\n');
        for j = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(j).name, ME.stack(j).line);
        end
        continue;
    end
end

fprintf('\n=== S03 Analysis Complete ===\n');
fprintf('Processed %d subjects\n', length(subjects));
