function A01_EyeAnalysis(subjID)
% Created by Mrugank (04/24/2023) 
clearvars -except subjID;
close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p.subjID = subjID;

% Initialize all the relevant paths
[p] = initialization(p, 'eye');

% List edf files
p.edf_dir = [p.subjDIR '/eyedata'];
dir_list = dir(p.edf_dir);
fNames_all = {dir_list.name};
fNames_edf = fNames_all(endsWith(fNames_all, 'edf'));
num_blocks = length(fNames_edf);

% List stim files
p.stim_dir = [p.subjDIR '/stimfiles'];
tar_dir = dir(p.stim_dir);
tar_list = {tar_dir.name};

if strcmp(p.subjID, 'NY098')
    tar_list = tar_list(endsWith(tar_list, 'mat'));
elseif strcmp(p.subjID, 'NY272')
    tar_list               = tar_list(startsWith(tar_list, '10071'));
elseif strcmp(p.subjID, 'NY190')
    tar_list               = tar_list(startsWith(tar_list, '101017'));
elseif strcmp(p.subjID, 'NY276')
    tar_list               = tar_list(startsWith(tar_list, '101024'));
    tar_list               = tar_list(~contains(tar_list, 'stimorig'));
    fNames_edf             = fNames_edf(startsWith(fNames_edf, '04'));
    num_blocks = length(fNames_edf);
end

% Copy edf and stimfiles to datc
for ii = 1:num_blocks
    block_path = [p.save_eyedata filesep '/block' num2str(ii, '%02d')];
    if ~exist(block_path, 'dir')
        mkdir(block_path)
    end
    copyfile([p.edf_dir filesep fNames_edf{ii}], [block_path filesep fNames_edf{ii}]);
    copyfile([p.stim_dir filesep tar_list{ii}], [block_path filesep tar_list{ii}(1:end-4) '_stiminfo.mat']);
end

%% Load ii_sess files
ii_sess_saveName = [p.save_eyedata '/ii_sess_sub_' p.subjID];
ii_sess_saveName_mat = [ii_sess_saveName '.mat'];
if exist(ii_sess_saveName_mat, 'file') == 2
    disp('Loading existing ii_sess file.')
    load(ii_sess_saveName_mat);
else
    disp('ii_sess file does not exist. running ieye')        
    ii_sess = RuniEye(p);
    save(ii_sess_saveName,'ii_sess')
end
end