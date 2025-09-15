%% Run S02 for Subjects 1-5
% Simple script to run S02_ReverseModelMNI for subjects 1, 2, 3, 4, 5

% Clear workspace
clear; close all; clc;

% Define subjects and parameters
subjects = [2, 3, 4, 5,7, 9, 12, 13, 15, 17, 18, 19, 23, 25];
surface_resolution = 5124;

% Run the master script
for subjID = subjects
    S02_ReverseModelMNI(subjID, surface_resolution);
end
