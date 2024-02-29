function [p]             = initialization(p, analysis_type)

% Adding all the folders and the helper functions
tmp                      = pwd; 
tmp2                     = strfind(tmp,filesep);
addpath(genpath(tmp));
p.master                 = tmp(1:(tmp2(end-1)-1));

% Check the system running on: currently accepted: syndrome and my macbook
[ret, hostname]          = system('hostname');
if ret                   ~= 0
    hostname             = getenv('HOSTNAME');
end
p.hostname               = strtrim(hostname);

% Find which git branch I am working on
[~, gitbranch]           = system('git rev-parse --abbrev-ref HEAD');
p.gitbranch              = strtrim(gitbranch);
disp(['You are currently working on ' p.gitbranch ' branch.']);

% Initialize all the paths
if strcmp(p.hostname, 'syndrome') || strcmp(p.hostname, 'vader') || strcmp(p.hostname, 'zod') ...
        || strcmp(p.hostname, 'zod.psych.nyu.edu') || strcmp(p.hostname, 'loki.psych.nyu.edu') % If running on Syndrome or Vader
    p.datb               = '/d/DATD/datd/MGS_iEEG';
    p.datc               = '/d/DATD/datd/MGS_ECoG';
    p.datd               = '/d/DATD/datd/MGS_iEEG';
    p.fieldtrip          = '/d/DATA/hyper/software/fieldtrip-20220104/';
    p.nspike             = '/d/DATA/hyper/toolboxes/sangi-matlab/NSpike_Code';
elseif strcmp(p.hostname, 'zod.psych.nyu.edu') || strcmp(p.hostname, 'zod') % If running on Zod
    p.datb               = '/datd/MGS_iEEG';
    p.datc               = '/datd/MGS_ECoG';
    p.datd               = '/datd/MGS_iEEG';
    p.fieldtrip          = '/clayspace/hyper/software/fieldtrip-20220104/';
    p.nspike             = '/clayspace/hyper/toolboxes/sangi-matlab/NSpike_Code';
else 
    disp('Device not identified!')
end

% subject directory to extract raw data
p.subjDIR                = [p.datd filesep p.subjID];

% Folder to save analysis data
p.save                   = [p.datc filesep p.gitbranch filesep p.subjID];
if ~exist(p.save, 'dir')
    mkdir(p.save);
end

% Add toolbox to path (either iEye or fieldtrip)
if strcmp(analysis_type, 'eye')
    p.iEye               = [p.master '/Mrugank/TMS/mgs_stimul/iEye'];
    p.save_eyedata       = [p.save '/EyeData'];
    if ~exist(p.save_eyedata, 'dir')
        mkdir(p.save_eyedata);
    end
    addpath(genpath(p.iEye));
elseif strcmp(analysis_type, 'ecog')
    p.saveECoG           = [p.save '/ECoG'];
    if ~exist(p.saveECoG, 'dir')
        mkdir(p.saveECoG);
    end
    addpath(p.fieldtrip);
    ft_defaults;
    addpath(genpath(p.nspike));
end
end