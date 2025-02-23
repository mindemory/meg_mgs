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

% Initialize all the paths
if strcmp(p.hostname, 'syndrome') || strcmp(p.hostname, 'vader') || strcmp(p.hostname, 'zod') ...
        || strcmp(p.hostname, 'zod.psych.nyu.edu') || strcmp(p.hostname, 'loki.psych.nyu.edu') % If running on Syndrome or Vader
    p.raw_dir            = '/d/DATD/datd/MEG_MGS';
    p.save_dir           = '/d/DATD/datd/MEG_MGS/mrugankAnalysis';
    p.fieldtrip          = '/d/DATD/hyper/software/fieldtrip-20220104/';
else 
    disp('Device not identified!')
end

% subject directory to extract raw data
p.subjDIR                = [p.raw_dir filesep p.subjID];

% Folder to save analysis data
p.save                   = [p.save_dir filesep p.subjID];
if ~exist(p.save, 'dir')
    mkdir(p.save);
end

% Add toolbox to path (either iEye or fieldtrip)
if strcmp(analysis_type, 'eye')
    p.iEye               = [p.master '/Mrugank/iEye'];
    p.save_eyedata       = [p.save '/EyeData'];
    if ~exist(p.save_eyedata, 'dir')
        mkdir(p.save_eyedata);
    end
    addpath(genpath(p.iEye));
elseif strcmp(analysis_type, 'meg')
    p.saveMEG           = [p.save '/MEG'];
    if ~exist(p.saveMEG, 'dir')
        mkdir(p.saveMEG);
    end
    addpath(p.fieldtrip);
    ft_defaults;
    addpath(genpath(p.nspike));
end
end