close all; clc;
%% Initialization
% addpath('/d/DATD/hyper/software/fieldtrip-20220104/');
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');

addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;

%%
% Load anatomicals
mri_path             = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/SUMA/T1.nii';
pial_l_path          = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/surf/lh.pial.asc';
pial_r_path          = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/surf/rh.pial.asc';

surfL                = readPial(pial_l_path);
surfR                = readPial(pial_r_path);

figure;
hold on;
trisurf(surfL.faces, surfL.vertices(:,1), surfL.vertices(:,2), surfL.vertices(:,3), ...
        'FaceColor', '#808080', 'EdgeColor', 'none');
trisurf(surfR.faces, surfR.vertices(:,1), surfR.vertices(:,2), surfR.vertices(:,3), ...
        'FaceColor', '#808080', 'EdgeColor', 'none');
axis equal;
camlight; 
lighting gouraud; 

%%
anatMRI              = ft_read_mri(mri_path);
% anatMRI.tra          = eye(4);
coordsys             = ft_determine_coordsys(anatMRI);
anatMRI.coordsys     = coordsys.coordsys;

%% Read headshape file
hspPath              = '/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-12/meg/sub-12_task-mgs_headshape.hsp';
hspData              = ft_read_headshape(hspPath);
hspData.unit         = 'm';
hspData.fid.label    = {'Nasion', 'RPA', 'LPA'}';

gradData             = data.grad;
gradData             = ft_convert_units(gradData, 'm');

% Load layout
load('NYUKIT_helmet.mat')

figure;
ft_plot_mesh(hspData, 'vertexcolor', 'k', 'facealpha', 0.5);
hold on;
if isfield(hspData, 'fid') && ~isempty(hspData.fid)
    plot3(hspData.fid.pos(:,1), hspData.fid.pos(:,2), hspData.fid.pos(:,3), ...
        'go', 'MarkerSize', 5, 'LineWidth', 2);
    
    text(hspData.fid.pos(:,1), hspData.fid.pos(:,2), hspData.fid.pos(:,3), ...
        hspData.fid.label, 'FontSize', 12, 'Color', 'r');
end
% ft_plot_sens(gradData, 'box', 'k*');
ft_plot_sens(gradData, 'box', 1)

%% Align the headshape and fiducials to MEG
% elec_coil.elecpos    = hspData.fid.pos;
% elec_coil.label      = hspData.fid.label;
% elec_coil.unit       = hspData.unit;
% 
% elec_dummy.pos       = hspData.pos;
% elec_dummy.unit      = hspData.unit;
% elec_dummy.label     = arrayfun(@(x) ['Head' num2str(x)], 1:653, 'UniformOutput', false)';
% elec_dummy.elecpos   = hspData.pos;
% elec_dummy.chanpos   = elec_dummy.elecpos;
% 
% 
% % Align fiducials with MEG sensors
% cfg                  = [];
% cfg.method           = 'fiducial';
% cfg.template         = elec_coil;
% cfg.elec             = elec_dummy;
% cfg.feedback         = 'yes';
% cfg.fiducial         = {'Nasion', 'LPA', 'RPA'};
% elecAligned          = ft_electroderealign(cfg);


%% Align anatomical with hsp
cfg                  = [];
cfg.method           = 'headshape';
cfg.headshape.headshape ...
                     = hspData;
cfg.headshape.icp    = 'yes';
cfg.headshape.interactive ...
                     = 'yes';
mri_aligned          = ft_volumerealign(cfg, anatMRI);

% For sub-12:
% rotate (0, 30, 270)
% scale (1, 1, 1)
% translate (5, 0, 40)

%% Slice, segment and create headmodel
% Slice the data
mri_reslice          = ft_volumereslice([], mri_aligned);
% Segment the mri
cfg                  = [];
cfg.output           = {'brain'};
segmentedmri         = ft_volumesegment(cfg, mri_reslice);
% 
% figure;
% ft_plot_ortho(segmentedmri.brain, 'style', 'intersect','transform',segmentedmri.transform);
% % %% first manually rotate and translate to help the icp algorithm. After that ICP is always run
% 
% ft_plot_sens(gradData, 'style', '*b', ...
%     'facecolor' , 'y', 'facealpha' , 0.5);

cfg                  = [];
cfg.method           = 'singleshell';
singleShellHeadModel = ft_prepare_headmodel(cfg, segmentedmri);

% Volume conduction model
cfg                  = [];
cfg.method           = 'dipoli';
dipoliHeadModel      = ft_prepare_headmodel(cfg, singleShellHeadModel);

% Visualize the headmodel
figure(); ft_plot_mesh(singleShellHeadModel.bnd);


% Create a grid for this subject
cfg                  = [];
cfg.grid.resolution  = 10; % in mm
cfg.grid.unit        = 'mm';
cfg.tight            = 'yes';
cfg.inwardshift      = 2;
cfg.headmodel        = dipoliHeadModel;
grid                 = ft_prepare_sourcemodel(cfg);

% Create leadfield
cfg                  = [];
cfg.grad             = gradData;
cfg.channel          = data.label;
cfg.grid             = grid;
cfg.headmodel        = dipoliHeadModel;
leadfield            = ft_prepare_leadfield(cfg);

%% plot individual grid
figure();
ft_plot_mesh(dipoliHeadModel.bnd, 'facecolor', 'k',...
    'facealpha', 0.2); % scalp
hold on;
ft_plot_mesh(grid.pos(grid.inside,:), ...
    'vertexcolor', 'r');


sourcemodelfPath     = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/anatomy/sourcemodel.mat';
save(sourcemodelfPath, 'mri_aligned', 'mri_reslice', 'segmentedmri', ...
    'singleShellHeadModel', 'dipoliHeadModel', 'grid', '-v7.3')

load(sourcemodelfPath);

%% lcmv beamformer
cfg                 = [];
cfg.frequency       = [5 8];
cfg.latency         = [-0.1 1.7];
TFR_selected        = ft_selectdata(cfg, TFR_fourier_left);


% cfg                 = [];
% cfg.method          = 'lcmv';
% cfg.sourcemodel     = grid;
% cfg.headmodel       = singleShellHeadModel;
% cfg.grad            = gradData;
% cfg.keepleadfield   = 'yes';
% cfg.lcmv.keepfilter = 'yes';
% cfg.lcmv.fixedori   = 'yes';
% cfg.lcmv.lambda     = '1%';  % must be '1%' not 1
% cfg.normalize       = 'yes'; % normalize LF to attenuate depth bias
% cfg.normalizeparam  = 0.5; % default = 0.5
% source              = ft_sourceanalysis(cfg, TFR_selected);
% inside_pos          = find(source.inside);

cfg                 = [];
cfg.method          = 'dics';
cfg.fre

%% Extract all filters for inside voxels
W_meg = cell2mat(source.avg.filter(source.inside));