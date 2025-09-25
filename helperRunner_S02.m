%% Run S02 for Subjects 1-5
% Simple script to run S02_ReverseModelMNI for subjects 1, 2, 3, 4, 5

% Clear workspace
clear; close all; clc;

% Define subjects and parameters
subjects = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, 18, 19, 23, 24, 25, 29, 31, 32];
surface_resolutions = [8, 10];

% Run the master script
for subjID = subjects
    for surface_resolution = surface_resolutions
        S02A_ReverseModelMNIVolumetric(subjID, surface_resolution);
    end
end
