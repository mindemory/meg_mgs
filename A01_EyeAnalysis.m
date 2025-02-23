function A01_EyeAnalysis(subjID)
% Created by Mrugank (01/24/2025) 
% clear; close all; clc;
clearvars -except subjID;
close all; clc;
warning('off', 'all');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% p.subjID = [num2str(subjID,'%02d') 'MGS'];
% [p] = initialization(p, 'eye');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/iEye'));
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'));
 
% subjID = 5 ;

% Initializing variables
bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
derivativesRoot = [bidsRoot filesep 'derivatives/sub-' num2str(subjID, '%02d') '/eyetracking'];
if ~exist("derivativesRoot", "dir")
    mkdir(derivativesRoot)
end

taskName = 'mgs';
subName = ['sub-' num2str(subjID, '%02d')];
eyeRoot = [bidsRoot filesep subName filesep 'eyetracking'];
stimRoot = [bidsRoot filesep subName filesep 'stimfiles'];
fNameRoot = [subName '_task-' taskName];

% List edf files
edfFiles = dir([eyeRoot '/*.edf']);


nruns = length(edfFiles);
% Copy edf and stimfiles to datc
for run = 1:nruns
    runPath = [derivativesRoot filesep '/run-' num2str(run, '%02d')];
    if ~exist(runPath, 'dir')
        mkdir(runPath)
    end
    copyfile([eyeRoot filesep edfFiles(run).name], [runPath filesep edfFiles(run).name]);
end

%% Load ii_sess files
iisessRoot = [derivativesRoot filesep fNameRoot '-iisess'];
iisessfName = [iisessRoot '.mat'];
if exist(iisessfName, 'file') == 2
    disp('Loading existing ii_sess file.')
    load(iisessfName);
else
    disp('ii_sess file does not exist. running ieye')        
    ii_sess = RuniEye(subjID,    derivativesRoot, stimRoot, nruns);
    save(iisessRoot,'ii_sess')
end

%% QC plots
% Run QC
% which_excl = [20 22];
% disp('Running QC')
% RunQC_EyeData(ii_sess, p, which_excl, {'all_trials'});
%RunQC_EyeData(ii_sess, p, which_excl);
end