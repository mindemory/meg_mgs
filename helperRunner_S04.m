restoredefaultpath;
clear; close all; clc;

%% Setup and Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);

fprintf('=== Running S04 Beta Power Analysis in MNI Space ===\n');

%% Configuration
subjects = [1]; % Subjects to process
surface_resolution = 5124; % Use 5124 vertices

fprintf('Subjects: %s\n', mat2str(subjects));
fprintf('Surface resolution: %d vertices\n', surface_resolution);

%% Loop through subjects and run S04_betaPowerInMNI
for subjID = subjects
    fprintf('\n=== Processing Subject %02d ===\n', subjID);
    try
        S04_betaPowerInMNI(subjID, surface_resolution);
        fprintf('Subject %02d completed successfully!\n', subjID);
    catch ME
        fprintf('Error processing subject %02d: %s\n', subjID, ME.message);
        for i=1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
    end
end

fprintf('\n=== S04 MNI Analysis Complete ===\n');
fprintf('Processed %d subjects\n', length(subjects));
