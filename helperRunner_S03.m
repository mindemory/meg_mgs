%% Run S02 for Subjects 1-5
% Simple script to run S02_ReverseModelMNI for subjects 1, 2, 3, 4, 5

% Clear workspace
clear; close all; clc;

% Define subjects and parameters
subjects = [1,2, 3,4,5];
surface_resolutions = [8196];

% Run the master script
for subjID = subjects
    for surface_resolution = surface_resolutions
        S03_betaPowerInSource(subjID, surface_resolution);
    end
end
