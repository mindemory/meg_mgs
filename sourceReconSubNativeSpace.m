clear; close all; clc;
%% Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/'); % 2022 doesn't work well for sourerecon
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);

%%
subjID                 = 12;
% Initialize rootpaths
subRoot                = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-' num2str(subjID, '%02d') ...
                          '/meg/sub-' num2str(subjID, '%02d') '_task-mgs_'];
subDerivativesRoot     = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-' num2str(subjID, '%02d') ...
                          '/meg/sub-' num2str(subjID, '%02d') '_task-mgs_'];
anatRoot               = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-' num2str(subjID, '%02d') ...
                          '/anatomy/sub-' num2str(subjID, '%02d') '_task-mgs_'];
% Load anatomicals
% mri_path             = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/SUMA/T1.nii';
if subjID              == 12
    nMrkFiles            =  3;
    mri_path               = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/mri/orig.mgz';
    roiRoot                = '/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-12/anat/rois/';
    roiPaths = {
        [roiRoot 'lh.V1.nii.gz']
        [roiRoot 'lh.V2d.nii.gz']
        [roiRoot 'lh.V3d.nii.gz']
        [roiRoot 'rh.V1.nii.gz']
        [roiRoot 'rh.V2d.nii.gz']
        [roiRoot 'rh.V3d.nii.gz']
        [roiRoot 'lh.sPCS.nii.gz']
        [roiRoot 'rh.sPCS.nii.gz']
        };
    roiNames = {'lh_V1', 'lh_V2d', 'lh_V3d', 'rh_V1', 'rh_V2d', 'rh_V3d', 'lh_sPCS', 'rh_sPCS'}; % Manually defined ROI names
    
    % pial_l_path          = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/surf/lh.pial.asc';
    % pial_r_path          = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/surf/rh.pial.asc';
    
    % mri_path             = '/d/DATD/hyper/software/fieldtrip-20250318/template/headmodel/standard_mri.mat';
    % load(mri_path, 'mri');
    anatMRI              = ft_read_mri(mri_path);
    anatMRI.coordsys = 'ras';
    roiData = struct();
    for i = 1:numel(roiPaths)
        roiData.(roiNames{i}) = ft_read_mri(roiPaths{i});
        % roiData.(roiNames{i}).coordsys = 'ras';
        % roiData.(roiNames{i}) = ft_convert_coordsys(roiData.(roiNames{i}), 'als');
    end
    lhVisual               = roiData.lh_V1;
    lhVisual.anatomy       = roiData.lh_V1.anatomy | roiData.lh_V2d.anatomy | roiData.lh_V3d.anatomy;
    lhVisual.coordsys      = 'ras';
    lhVisual               = ft_convert_coordsys(lhVisual, 'als');
    rhVisual               = roiData.rh_V1;
    rhVisual.anatomy       = roiData.rh_V1.anatomy | roiData.rh_V2d.anatomy | roiData.rh_V3d.anatomy;
    % cfg.parameter          = 'anatomy'; % Assuming 'anatomy' field holds your ROI mask
    % cfg.resampleto         = anatMRI;  % Target MRI
    % cfg.interpmethod       = 'nearest'; % Important! For ROIs (discrete labels), use nearest neighbor
    % roi_resliced           = ft_volumereslice(cfg, roiData.lh_V1);
    % d.anatomy | roiData.rh_V3d.anatomy;
    rhVisual.coordsys      = 'ras';
    rhVisual               = ft_convert_coordsys(rhVisual, 'als');
    lhFrontal              = roiData.lh_sPCS;
    lhFrontal.coordsys      = 'ras';
    lhFrontal               = ft_convert_coordsys(lhFrontal, 'als');
    rhFrontal              = roiData.rh_sPCS;
    rhFrontal.coordsys      = 'ras';
    rhFrontal               = ft_convert_coordsys(rhFrontal, 'als');
elseif subjID              == 4
    nMrkFiles            =  1;
    mri_path               = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/sub-04/mri/orig.mgz';
    anatMRI              = ft_read_mri(mri_path);
    anatMRI.coordsys = 'ras';
elseif subjID              == 1
    nMrkFiles            =  2;
    mri_path               = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/sub-01/mri/orig.mgz';
    anatMRI              = ft_read_mri(mri_path);
    anatMRI.coordsys = 'ras';
end

% anatMRI              = mri;

% surfL                = readPial(pial_l_path);
% surfR                = readPial(pial_r_path);
% 
% figure;
% hold on;
% trisurf(surfL.faces, surfL.vertices(:,1), surfL.vertices(:,2), surfL.vertices(:,3), ...
%         'FaceColor', '#808080', 'EdgeColor', 'none');
% trisurf(surfR.faces, surfR.vertices(:,1), surfR.vertices(:,2), surfR.vertices(:,3), ...
%         'FaceColor', '#808080', 'EdgeColor', 'none');
% axis equal;
% camlight; 
% lighting gouraud; 


% coordsys             = ft_determine_coordsys(anatMRI);
% % Set the coordinate system to 'r a s'
% anatMRI.coordsys     = coordsys.coordsys;
%%
% cfg = [];
% cfg.roi = roiData.lh_V1.anatomy; % Use the anatomy field of the ROI structure
% cfg.roicolor = [1 0 0]; % Red color for V1
% cfg.atlas = anatMRI;
% ft_sourceplot(cfg, roiData.lh_V1, anatMRI);
% title('Left V1 ROI');

% ft_plot_ortho(anatMRI.anatomy, 'style', 'intersect', 'transform', anatMRI.transform)
% hold on;
% ft_plot_ortho(roiData.lh_V1.anatomy, 'style', 'intersect', 'transform', roiData.lh_V1.transform)


%% Read headshape file
% COORDSYS NOTE: 
%   hspData: ALS
%   gradData: ALS
%   mri: RAS

hspPath              = [subRoot 'headshape.hsp'];
% COORDSYS NOTE: 
%   hspData: ALS
%   gradData: ALS
%
hspData              = ft_read_headshape(hspPath, 'unit', 'm');
hspData.fid.label    = {'Nasion', 'LPA', 'RPA'};

% Read elpData
elpPath              = [subRoot 'electrodes.elp'];
[fidData, hpiData]   = readelpFile(elpPath);

% Read raw data for gradiometers
load([subDerivativesRoot 'run-01_raw.mat'])
gradData             = data.grad;
gradData             = ft_convert_units(gradData, 'm');
clearvars data;

% Read hpilocations in gradiometer space

hpiMrkData           = NaN(nMrkFiles, size(hpiData, 1), size(hpiData, 2));
% 
% raw_file1 = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-12/meg/sub-12_task-mgs_run-01_meg.sqd';
for mrkIdx           = 1:nMrkFiles
    raw_file_mrk     = [subRoot 'marker_' num2str(mrkIdx, '%02d') '.sqd'];
    hdr_Mrk          = ft_read_header(raw_file_mrk);
    hpiMrkData(mrkIdx, :, :) ...
                     = cat(1, hdr_Mrk.orig.coregist.hpi.meg_pos);
end
hpiMrkData           = squeeze(mean(hpiMrkData, 1));

%%
headshape.pos        = [hpiData; fidData; hspData.pos];
headshape.unit       = 'm'; 
num_remaining_points = size(hspData.pos, 1); 
remaining_labels     = cell(num_remaining_points, 1);
for i                = 1:num_remaining_points
    remaining_labels{i} ...
                     = sprintf('head_%d', i);
end
headshape.label      = [{'LPA', 'RPA', 'NAS', 'HPI4', 'HPI5', 'nreal', ...
                         'lreal', 'rreal'}, remaining_labels']';

elec_dummy           = headshape;
elec_dummy.elecpos   = headshape.pos;
elec_dummy.chanpos   = elec_dummy.elecpos;

hpi_coil.elecpos     = hpiMrkData;%(1:3, :);
hpi_coil.label       = {'LPA', 'RPA', 'NAS', 'HPI4', 'HPI5'}';
hpi_coil.unit        = 'm';

cfg                  = [];
cfg.method           = 'fiducial';
cfg.template         = hpi_coil;
cfg.elec             = elec_dummy;
cfg.feedback         = 'yes';
cfg.fiducial         = {'NAS', 'LPA', 'RPA'}';
elec_aligned         = ft_electroderealign(cfg);
%%
hspCorrected         = hspData;
hspCorrected.pos     = elec_aligned.chanpos(9:end, :);
hspCorrected.fid.pos = elec_aligned.chanpos(6:8, :);
hspCorrected.fid.label ...
                     = {'NAS', 'LPA', 'RPA'};
% Convert units back to mm
hspCorrected         = ft_convert_units(hspCorrected, 'mm');
gradData             = ft_convert_units(gradData, 'mm');
hpiMrkData           = hpiMrkData .* 1000;

figure;
ft_plot_mesh(hspCorrected, 'vertexcolor', 'k', 'facealpha', 0.5);

hold on;
if isfield(hspCorrected, 'fid') && ~isempty(hspCorrected.fid)
    plot3(hspCorrected.fid.pos(:,1), hspCorrected.fid.pos(:,2), hspCorrected.fid.pos(:,3), ...
          'go', 'MarkerSize', 5, 'LineWidth', 2);
    text(hspCorrected.fid.pos(:,1), hspCorrected.fid.pos(:,2), hspCorrected.fid.pos(:,3), ...
         hspCorrected.fid.label, 'FontSize', 12, 'Color', 'r');
end
ft_plot_sens(gradData, 'box', 1)
ft_plot_mesh(hpiMrkData, 'vertexcolor', 'b', 'vertexsize', 20)

%% Align anatomical with hsp
restoredefaultpath;
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;

anatMRItransformed = ft_convert_coordsys(anatMRI, 'als');

cfg                  = [];
cfg.method           = 'headshape';
cfg.spmversion       = 'spm12';
cfg.headshape.headshape ...
                     = hspCorrected;
% cfg.headshape.coordsys = 'als';
cfg.headshape.icp    = 'yes';
cfg.headshape.interactive = 'no';  
mri_aligned          = ft_volumerealign(cfg, anatMRItransformed);

cfg.headshape.interactive = 'yes';  
mri_aligned          = ft_volumerealign(cfg, mri_aligned);
%%
figure;
ft_plot_headshape(hspCorrected);
hold on;
ft_plot_ortho(mri_aligned.anatomy,'transform',mri_aligned.transform,'style','intersect');
%%
figure;
ft_plot_headshape(hspCorrected);
hold on;
ft_plot_ortho(rhFrontal.anatomy,'transform',rhFrontal.transform,'style','intersect');
%% Transform pial accordingly
% pial                 = [];
% pial.pos             = [surfL.vertices; surfR.vertices];
% pial.tri             = [surfL.faces; surfR.faces + size(surfL.vertices, 1)];
% 
% % Calculate the combined transformation
% T_orig               = mri_aligned.transformorig;
% T_new                = mri_aligned.transform;
% T_combined           = T_new / T_orig;  % This is equivalent to T_new * inv(T_orig)
% 
% % Apply the transformation to the pial surface vertices
% homogeneous_coords   = [pial.pos, ones(size(pial.pos, 1), 1)]';
% transformed_coords   = T_combined * homogeneous_coords;
% pial.pos             = transformed_coords(1:3, :)';
% pial                 = ft_convert_units(pial, 'mm');

%% Slice, segment and create headmodel
% % Slice the data
mri_reslice          = ft_volumereslice([], mri_aligned);
% % Segment the mri
cfg                  = [];
cfg.output           = {'brain'};
segmentedmri         = ft_volumesegment(cfg, mri_reslice);

%%
cfg                  = [];
cfg.method           = 'singleshell';
singleShellHeadModel = ft_prepare_headmodel(cfg, segmentedmri);

%%
figure;
plot3(singleShellHeadModel.bnd.pos(:, 1), singleShellHeadModel.bnd.pos(:, 2), singleShellHeadModel.bnd.pos(:, 3), 'ro')
hold on;
ft_plot_sens(gradData, 'facecolor', 'green', 'facealpha', 1)
ft_plot_ortho(mri_reslice.anatomy, 'style', 'intersect', ...
    'transform', mri_reslice.transform)
axis equal;
%%
% Create a grid for this subject
cfg                  = [];
cfg.sourcemodel.resolution  = 10; % in mm
cfg.sourcemodel.unit        = 'mm';
cfg.tight            = 'yes';
cfg.inwardshift      = -2;
cfg.headmodel        = singleShellHeadModel;
grid                 = ft_prepare_sourcemodel(cfg);


figure;
hold on;
ft_plot_headmodel(singleShellHeadModel, 'edgecolor', 'g', 'facealpha', 0.4);
ft_plot_mesh(grid.pos(grid.inside, :))
%% plot the shared individual grid
figure;
ft_plot_mesh(singleShellHeadModel.bnd, 'facecolor', 'g',...
    'facealpha', 0.2); % brain
hold on;
ft_plot_mesh(grid.pos(grid.inside,:), ...
    'vertexcolor', 'r');
ft_plot_ortho(mri_reslice.anatomy, 'style', 'intersect', ...
    'transform', mri_reslice.transform)
ft_plot_sens(gradData, 'facecolor', 'g','edgecolor', 'g')
%%
idxChosen            = grid.inside;
figure;
% plot3(singleShellHeadModel.bnd.pos(:, 1), ...
%       singleShellHeadModel.bnd.pos(:, 2), ...
%       singleShellHeadModel.bnd.pos(:, 3), 'ro')
hold on;
plot3(gradData.chanpos(:, 1), gradData.chanpos(:, 2), gradData.chanpos(:, 3), 'bs')
plot3(grid.pos(idxChosen, 1), grid.pos(idxChosen, 2), grid.pos(idxChosen, 3), 'ko')
axis equal;
%%

%% plot individual grid
forwardmodelfPath     = [anatRoot 'forwardmodel-nativespace-10mm.mat'];
save(forwardmodelfPath, 'gradData', 'hpiMrkData', 'hspCorrected', 'mri_reslice', ...
    'mri_aligned', 'segmentedmri', 'singleShellHeadModel', 'grid', '-v7.3');
load(forwardmodelfPath);

%% lcmv beamformer
% Load stimlocked data
load([subDerivativesRoot 'stimlocked_lineremoved.mat'])
% Load the data
epocThis                          = epocStimLocked;

% Logical mask to find epochs matching your criteria
trial_criteria_left               = (epocThis.trialinfo(:,2) == 4) | ...
                                    (epocThis.trialinfo(:,2) == 5) | ...
                                    (epocThis.trialinfo(:,2) == 6) | ...
                                    (epocThis.trialinfo(:,2) == 7) | ...
                                    (epocThis.trialinfo(:,2) == 8);

trial_criteria_right              = (epocThis.trialinfo(:,2) == 1) | ...
                                    (epocThis.trialinfo(:,2) == 2) | ...
                                    (epocThis.trialinfo(:,2) == 3) | ...
                                    (epocThis.trialinfo(:,2) == 9) | ...
                                    (epocThis.trialinfo(:,2) == 10);

% Logical mask to find epochs without NaNs
has_no_nans                       = cellfun(@(x) ~any(isnan(x(:))), epocThis.trial)';

% Combine both criteria
valid_trialsLeft                  = find(trial_criteria_left & has_no_nans);
valid_trialsRight                 = find(trial_criteria_right & has_no_nans);

% Select Left trials
cfg                               = [];
cfg.trials                        = valid_trialsLeft;
epocLeft                          = ft_selectdata(cfg, epocThis);

% Select Right trials
cfg                               = [];
cfg.trials                        = valid_trialsRight;
epocRight                         = ft_selectdata(cfg, epocThis);
epocLeft_filtered                 = epocLeft;
epocRight_filtered                = epocRight;
% cfg                               = [];
% cfg.lpfreq                        = 100;
% cfg.lpfilter                      = 'yes';
% epocLeft_filtered                 = ft_preprocessing(cfg, epocLeft);
% epocRight_filtered                = ft_preprocessing(cfg, epocRight);
% cfg                               = [];
% cfg.resamplefs                    = 100;
% cfg.resamplemethod                = 'resample';
% epocLeft_filtered                 = ft_resampledata(cfg, epocLeft_filtered);
% epocRight_filtered                = ft_resampledata(cfg, epocRight_filtered);
% 
% %%%%% Average reference >> absolutely not
% cfg                               = [];
% cfg.reref                         = 'yes';
% cfg.refchannel                    = 'all';
% cfg.refmethod                     = 'avg';
% epocLeft                          = ft_preprocessing(cfg, epocLeft);
% epocRight                         = ft_preprocessing(cfg, epocRight);
% epocCombined                      = ft_appenddata([], epocLeft, epocRight);
epocCombined_filtered             = ft_appenddata([], epocLeft_filtered, epocRight_filtered);


%%
% Compute timelocked data with covariance
cfg                               = [];
cfg.covariance                    = 'yes';
cfg.covariancewindow              = 'all';
cfg.keeptrials                    = 'no';
% timelockedLeft                    = ft_timelockanalysis(cfg, epocLeft);
% timelockedRight                   = ft_timelockanalysis(cfg, epocRight);
% timelockedCombined                = ft_timelockanalysis(cfg, epocCombined);
timelockedCombined_filtered       = ft_timelockanalysis(cfg, epocCombined_filtered);
%%        
cfg                               = [];
cfg.method                        = 'lcmv';
cfg.sourcemodel                   = grid;
cfg.headmodel                     = singleShellHeadModel;
cfg.grad                          = gradData;
cfg.keepleadfield                 = 'yes';
cfg.lcmv.keepfilter               = 'yes';
cfg.lcmv.fixedori                 = 'yes';
cfg.lcmv.lambda                   = '5%';  % must be '1%' not 1
% cfg.lcmv.weightnorm               = 'unitnoisegain';
% cfg.normalize                     = 'yes'; % normalize LF to attenuate depth bias
% cfg.normalizeparam                = 0.5; % default = 0.5
% source                            = ft_sourceanalysis(cfg, timelockedCombined);
source                            = ft_sourceanalysis(cfg, timelockedCombined_filtered);

inside_pos                        = find(source.inside);


%% Extract all filters for inside voxels
W_meg               = cell2mat(cellfun(@(x) x, source.avg.filter(inside_pos), 'UniformOutput', false));

sourcedata          = [];
sourcedata.label    = cell(numel(inside_pos), 1);
for i               = 1:numel(inside_pos)
    sourcedata.label{i} ...
                    = sprintf('S_%d', inside_pos(i));
end
sourcedataLeft      = sourcedata;
sourcedataRight     = sourcedata;
sourcedataCombined  = sourcedata;

%% Create sourcedata
% sourcedata          = [];
% sourcedata.label    = cell(numel(inside_pos), 1);
% for i               = 1:numel(inside_pos)
%     sourcedata.label{i} ...
%                     = sprintf('S_%d', inside_pos(i));
% end
% sourcedataPost      = sourcedata;
% sourcedataPre     = sourcedata;
% 
% epocLeft_filt       = epocLeft;
% epocRight_filt      = epocRight;
% %%
% for iTrial          = 1:size(epoc_pre.trialinfo, 1)
%     sourcedataPre.trial{iTrial} ...
%                     = W_meg * epoc_pre.trial{iTrial};
% end
% for iTrial          = 1:size(epoc_post.trialinfo, 1)
%     sourcedataPost.trial{iTrial} ...
%                     = W_meg * epoc_post.trial{iTrial};
% end
% 
% vars_pre = zeros(size(var(sourcedataPre.trial{tt},0, 2)));
% vars_post  = zeros(size(var(sourcedataPre.trial{tt},0, 2)));
% 
% for tt = 1: numel(sourcedataPre.trial)
%     vars_pre = vars_pre + var(sourcedataPre.trial{tt},0, 2);
% end
% vars_pre = vars_pre./numel(sourcedataPre.trial)
% for tt = 1: numel(sourcedataPost.trial)
%     vars_post = vars_post + var(sourcedataPost.trial{tt},0, 2);
% end
% vars_post = vars_post./numel(sourcedataPost.trial);
% 
% source_val = (vars_post - vars_pre)./(vars_post + vars_pre);

%%
epocLeft_filt              = epocLeft_filtered;
epocRight_filt             = epocRight_filtered;
epocCombined_filt          = epocCombined_filtered;
% Project data to source space
% sourceSpacePower     = [anatRoot '_sourceSpaceBetaPower-highres.mat'];
% if ~exist("sourceSpacePower", 'file')
sourcedataLeft.trial       = cellfun(@(x) W_meg * x, epocLeft_filt.trial, 'UniformOutput', false);
sourcedataRight.trial      = cellfun(@(x) W_meg * x, epocRight_filt.trial, 'UniformOutput', false);
sourcedataCombined.trial   = cellfun(@(x) W_meg * x, epocCombined_filt.trial, 'UniformOutput', false);
sourcedataLeft.time        = epocLeft_filt.time;
sourcedataRight.time       = epocRight_filt.time;
sourcedataCombined.time    = epocCombined_filt.time;

% Low-pass filter
cfg                        = [];
cfg.lpfilter               = 'yes';
cfg.lpfreq                 = 70;  % Or 70, safely below 75 Hz (Nyquist of 150 Hz)
sourcedataLeft             = ft_preprocessing(cfg, sourcedataLeft);
sourcedataRight            = ft_preprocessing(cfg, sourcedataRight);
sourcedataCombined         = ft_preprocessing(cfg, sourcedataCombined);

% Step 2: Downsample
cfg                        = [];
cfg.resamplefs             = 150;
cfg.detrend                = 'no';
sourcedataLeft             = ft_resampledata(cfg, sourcedataLeft);
sourcedataRight            = ft_resampledata(cfg, sourcedataRight);
sourcedataCombined         = ft_resampledata(cfg, sourcedataCombined);
cfg                        = [];
cfg.bpfilter               = 'yes';
cfg.bpfreq                 = [18 27];
sourcedataLeft             = ft_preprocessing(cfg, sourcedataLeft);
sourcedataRight            = ft_preprocessing(cfg, sourcedataRight);
sourcedataCombined         = ft_preprocessing(cfg, sourcedataCombined);
hilbert_compute            = @(x) hilbert(x')'; 
sourcedataLeft.trial       = cellfun(hilbert_compute, sourcedataLeft.trial, ...
                                     'UniformOutput', false);
sourcedataRight.trial      = cellfun(hilbert_compute, sourcedataRight.trial, ...
                                     'UniformOutput', false);
sourcedataCombined.trial   = cellfun(hilbert_compute, sourcedataCombined.trial, ...
                                     'UniformOutput', false);
%     save(sourceSpacePower, 'sourcedataLeft','sourcedataRight', 'sourcedataCombined', '-v7.3');
% else
%     load(sourceSpacePower);
% end

%% Compute Power
sourceLeftPow             = sourcedataLeft;
sourceLeftPow.trial       = cellfun(@(x) abs(x), sourceLeftPow.trial, 'UniformOutput', false);
sourceRightPow            = sourcedataRight;
sourceRightPow.trial      = cellfun(@(x) abs(x), sourceRightPow.trial, 'UniformOutput', false);

% 4. Contrast Calculation - Proper Normalization
cfg                       = [];
sourceDataLeft_avg        = ft_timelockanalysis(cfg, sourceLeftPow);
sourceDataRight_avg       = ft_timelockanalysis(cfg, sourceRightPow);

cfg                       = [];
cfg.parameter             = 'avg';  
cfg.operation             = '(x1-x2)/(x1+x2)';
sourceDiff                = ft_math(cfg, sourceDataLeft_avg, sourceDataRight_avg);


%% Visualize source
% TOI                 = find(sourceDiff.time > 1.0 & sourceDiff.time < 1.5);
TOI                       = find(sourceDiff.time > 0.8 & sourceDiff.time < 1.5);

sourceVisualize           = source;
sourceVisualize.lateralizedPow ...
                          = NaN(size(source.inside));
sourceVisualize.lateralizedPow(source.inside) ...
                          = squeeze(mean(sourceDiff.avg(:, TOI), 2, 'omitnan'));

%%
cfg                       = [];
cfg.parameter             = {'lateralizedPow'};
[interp]                  = ft_sourceinterpolate(cfg, sourceVisualize, mri_reslice);

cfg                       = [];
cfg.method                = 'ortho';
cfg.crosshair             = 'yes';
cfg.funparameter          = 'lateralizedPow';
cfg.funcolormap           = '*RdBu';
cfg.funcolorlim           = [-0.1 0.1];
ft_sourceplot(cfg, interp);
% hold on;
% ft_plot_sens(gradData, 'unit', 'mm', 'style', 'r*');
% hold off;
%% COHERENCE
% right_sensors               = {'AG001', 'AG002', 'AG007', 'AG008', 'AG020', 'AG022', 'AG023', ...
%                                'AG024', 'AG034', 'AG036', 'AG050', 'AG055', 'AG065', 'AG066', ...
%                                'AG098', 'AG103'};
% left_sensors                = {'AG013', 'AG014', 'AG015', 'AG016', 'AG023', 'AG025', 'AG026', ...
%                                'AG027', 'AG028', 'AG041', 'AG042', 'AG043', 'AG059', 'AG060', ...
%                                'AG066', 'AG092'};
% 
% % Find idx of left sensors
% left_idx                    = find(ismember(left_sensors, gradData.label));
% right_idx                   = find(ismember(right_sensors, gradData.label));

% contrib                     = mean(W_meg(:, left_idx), 2);
% contribution = vecnorm(contrib, 2, 2);
% sorted_contrib              = sort(contrib, 'descend');
% thresholdCenterOfHead       = sorted_contrib(ceil(0.04 * numel(sorted_contrib)));
% thresholdEffect             = sorted_contrib(ceil(0.1 * numel(sorted_contrib)));
% top_source_idx              = find((contrib >= thresholdEffect) & ...
%                                    (contrib <= thresholdCenterOfHead));
% top_source_idx              = find(contrib >= thresholdCenterOfHead);
% top_source_idx = find(sourceVisualize.lateralizedPow < -0.02);

% is_top_source               = zeros(size(source.inside));
% cntr                        = 1;
% for iii                     = 1:size(source.inside)
%     if source.inside(iii)
%         if ismember(cntr, top_source_idx)
%             is_top_source(iii) = 1;
%         end
%         cntr                = cntr + 1;
%     end
% 
% end
% is_top_source(source.inside(source.inside(top_source_idx))) ...
%                             = 1;

sourceVisualizeWithMask     = sourceVisualize;
sourceVisualizeWithMask.mask ...
                            = zeros(size(source.inside));
% sourceVisualizeWithMask.mask(find((sourceVisualize.lateralizedPow > 0.045) & (sourceVisualize.pos(1, :, :) >= 0))) ...
%                             = 1;
positive_x_indices = find(sourceVisualize.pos(:, 2) > 0);
mask_indices = find(sourceVisualize.lateralizedPow > 0.045);

% Find the intersection of these indices
% combined_indices = intersect(mask_indices, positive_x_indices);
% combined_indices = positive_x_indices;
posterior_indices = find(sourceVisualize.pos(:, 1) < -55);
superior_indices = find(sourceVisualize.pos(:, 3) > 0);
% pos_pow = find(sourceVisualize.lateralizedPow > 0.03);
% neg_pow = find(sourceVisualize.lateralizedPow < -0.03);

combined_indices = intersect(posterior_indices, superior_indices);
% combined_indices = find( ...
%     sourceVisualize.pos(:, 1) < -55 & ...
%     sourceVisualize.pos(:, 3) > 0 & ...
%     (sourceVisualize.lateralizedPow > 0.03 | sourceVisualize.lateralizedPow < -0.03));

% Now use these indices to set the mask
sourceVisualizeWithMask.mask(combined_indices) = 1;
% sourceVisualizeWithMask.mask()
%%
close all;
cfg                 = [];
cfg.parameter       = {'lateralizedPow', 'mask'};
[interp]            = ft_sourceinterpolate(cfg, sourceVisualizeWithMask, mri_reslice);

cfg = [];
cfg.method        = 'ortho';
cfg.crosshair = 'yes';
cfg.funparameter  = 'lateralizedPow';
cfg.funcolormap = '*RdBu';
cfg.maskparameter = 'mask';
cfg.funcolorlim = [-0.05 0.05];
% sourceVisualize.mask = is_top_source;
% cfg.funcolorlim   = [0 0.06];
% cfg.location = [2 38 48];
ft_sourceplot(cfg, interp);
%% COHERENCE in source space
TOI                             = find(sourcedataLeft.time{1} > 0.8 & sourcedataLeft.time{1} < 1.5);
% seedSources                       = find(sourceVisualize.lateralizedPow(source.inside) > 0.045);
% positive_x_indices = find(sourceVisualize.pos(source.inside, 2) > 0);
% mask_indices = find(sourceVisualize.lateralizedPow(source.inside) > 0.045);
% 
% % Find the intersection of these indices
% seedSources = intersect(mask_indices, positive_x_indices);
posterior_indices               = find(source.pos(inside_pos, 1) < -55);
superior_indices                = find(source.pos(inside_pos, 3) > 0);
seedSources                     = intersect(posterior_indices, superior_indices);
% seedSources = find( ...
%     source.pos(inside_pos, 1) < -55 & ...
%     source.pos(inside_pos, 3) > 0 & ...
%     (sourceVisualize.lateralizedPow(inside_pos) > 0.03 | sourceVisualize.lateralizedPow(inside_pos) < -0.03));
% seedSources = find( ...
%     source.pos(inside_pos, 1) < -55 & ...
%     source.pos(inside_pos, 3) > 0 & ...
%     sourceVisualize.lateralizedPow(inside_pos) < -0.03);

% [~,seedSources] = min(sourceVisualize.lateralizedPow(source.inside));
[cohLeftMat, icohLeftMat, pcohLeftMat, ipcohLeftMat, plvLeftMat, pliLeftMat] ...
                                = computeConnectivitySourceSpace(sourcedataLeft, seedSources, TOI);
[cohRightMat, icohRightMat, pcohRightMat, ipcohRightMat, plvRightMat, pliRightMat] ...
                                = computeConnectivitySourceSpace(sourcedataRight, seedSources, TOI);
[cohMat, icohMat, pcohMat, ipcohMat, plvMat, pliMat] ...
                                = computeConnectivitySourceSpace(sourcedataCombined, seedSources, TOI);
%% 
% clearvars epocLeft epochRight epocLeft_filtered epocRight_filtered epocStimLocked epocThis
sourceVisualizeConnectivity     = source;
sourceVisualizeConnectivity.cohLeft ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.icohLeft ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.pcohLeft ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.ipcohLeft ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.plvLeft ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.pliLeft ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.cohRight ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.icohRight ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.pcohRight ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.ipcohRight ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.plvRight ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.pliRight ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.coh ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.icoh ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.pcoh ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.ipcoh ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.plv ...
                                = NaN(size(source.inside));
sourceVisualizeConnectivity.pli ...
                                = NaN(size(source.inside));

meancohLeftMat                  = squeeze(mean(cohLeftMat, [1, 3]));
meanicohLeftMat                 = squeeze(mean(abs(icohLeftMat), [1, 3]));
meanpcohLeftMat                 = squeeze(mean(pcohLeftMat, [1, 3]));
meanipcohLeftMat                = squeeze(mean(abs(ipcohLeftMat), [1, 3]));
meanplvLeftMat                  = squeeze(mean(abs(plvLeftMat), [1, 3]));
meanpliLeftMat                  = squeeze(mean(abs(pliLeftMat), [1, 3]));

meancohRightMat                 = squeeze(mean(cohRightMat, [1, 3]));
meanicohRightMat                = squeeze(mean(abs(icohRightMat), [1, 3]));
meanpcohRightMat                = squeeze(mean(pcohRightMat, [1, 3]));
meanipcohRightMat               = squeeze(mean(abs(ipcohRightMat), [1, 3]));
meanplvRightMat                 = squeeze(mean(abs(plvRightMat), [1, 3]));
meanpliRightMat                 = squeeze(mean(abs(pliRightMat), [1, 3]));

meancohMat                      = squeeze(mean(cohMat, [1, 3]));
meanicohMat                     = squeeze(mean(abs(icohMat), [1, 3]));
meanpcohMat                     = squeeze(mean(pcohMat, [1, 3]));
meanipcohMat                    = squeeze(mean(abs(ipcohMat), [1, 3]));
meanplvMat                      = squeeze(mean(abs(plvMat), [1, 3]));
meanpliMat                      = squeeze(mean(abs(pliMat), [1, 3]));

% meancohMat                      = squeeze(mean((cohLeftMat - cohRightMat) ./ (cohLeftMat + cohRightMat), [1, 3]));
% meanicohMat                     = squeeze(mean((abs(icohLeftMat) - abs(icohRightMat)) ./ (abs(icohLeftMat) - abs(icohRightMat)), [1, 3]));
% meanpcohMat                     = squeeze(mean(pcohRightMat, [1, 3]));
% meanipcohMat                    = squeeze(mean(abs(ipcohRightMat), [1, 3]));
% meanplvMat                      = squeeze(mean(abs(plvRightMat), [1, 3]));
% meanpliMat                      = squeeze(mean(abs(pliRightMat), [1, 3]));

sourceVisualizeConnectivity.cohLeft(inside_pos) ...
                                = meancohLeftMat;% - squeeze(mean(cohRightMat, [1, 3]));
sourceVisualizeConnectivity.icohLeft(inside_pos) ...
                                = meanicohLeftMat;% - squeeze(mean(icohRightMat, [1, 3]));
sourceVisualizeConnectivity.pcohLeft(inside_pos) ...
                                = meanpcohLeftMat;% - squeeze(mean(pcohRightMat, [1, 3]));
sourceVisualizeConnectivity.ipcohLeft(inside_pos) ...
                                = meanipcohLeftMat;% - squeeze(mean(ipcohRightMat, [1, 3]));
sourceVisualizeConnectivity.plvLeft(inside_pos) ...
                                = meanplvLeftMat;
sourceVisualizeConnectivity.pliLeft(inside_pos) ...
                                = meanpliLeftMat;

sourceVisualizeConnectivity.cohRight(inside_pos) ...
                                = meancohRightMat;
sourceVisualizeConnectivity.icohRight(inside_pos) ...
                                = meanicohRightMat;
sourceVisualizeConnectivity.pcohRight(inside_pos) ...
                                = meanpcohRightMat;
sourceVisualizeConnectivity.ipcohRight(inside_pos) ...
                                = meanipcohRightMat;
sourceVisualizeConnectivity.plvRight(inside_pos) ...
                                = meanplvRightMat;
sourceVisualizeConnectivity.pliRight(inside_pos) ...
                                = meanpliRightMat;

sourceVisualizeConnectivity.coh(inside_pos) ...
                                = meancohMat; %(meancohLeftMat - meancohRightMat) ./ (meancohLeftMat + meancohRightMat);
sourceVisualizeConnectivity.icoh(inside_pos) ... %= (meancohLeftMat + meancohRightMat) ./ 2;
                                = meanicohMat; %(meanicohLeftMat - meanicohRightMat) ./ (meanicohLeftMat + meanicohRightMat);
sourceVisualizeConnectivity.pcoh(inside_pos) ...
                                = meanpcohMat; %(meanpcohLeftMat - meanpcohRightMat) ./ (meanpcohLeftMat + meanpcohRightMat);
sourceVisualizeConnectivity.ipcoh(inside_pos) ...
                                = meanipcohRightMat; %(meanipcohLeftMat - meanipcohRightMat) ./ (meanipcohLeftMat + meanipcohRightMat);
sourceVisualizeConnectivity.plv(inside_pos) ...
                                = meanplvMat; %(meanpcohLeftMat - meanpcohRightMat) ./ (meanpcohLeftMat + meanpcohRightMat);
sourceVisualizeConnectivity.pli(inside_pos) ...
                                = meanpliMat; %(meanipcohLeftMat - meanipcohRightMat) ./ (meanipcohLeftMat + meanipcohRightMat);

%%
cfg                            = [];
cfg.parameter                  = {'cohLeft', 'icohLeft', 'pcohLeft', 'ipcohLeft', 'plvLeft', 'pliLeft', ...
                                  'cohRight', 'icohRight', 'pcohRight', 'ipcohRight', 'plvRight', 'pliRight', ...
                                  'coh', 'icoh', 'pcoh', 'ipcoh', 'plv', 'pli'};
[interpConnec]                 = ft_sourceinterpolate(cfg, sourceVisualizeConnectivity, mri_reslice);
%%
lThresh                        = 0.05;
hThresh                        = 0.9;
ArrayForThresh.cohLeft         = [quantile(meancohLeftMat, lThresh), ...
                                  quantile(meancohLeftMat, 1-lThresh)];
ArrayForThresh.icohLeft        = [quantile(meanicohLeftMat, lThresh), ...
                                  quantile(meanicohLeftMat, 1-lThresh)];
ArrayForThresh.pcohLeft        = [quantile(meanpcohLeftMat, lThresh), ...
                                  quantile(meanpcohLeftMat, 1-lThresh)];
ArrayForThresh.ipcohLeft       = [quantile(meanipcohLeftMat, lThresh), ...
                                  quantile(meanipcohLeftMat, 1-lThresh)];
ArrayForThresh.plvLeft         = [quantile(meanplvLeftMat, lThresh), ...
                                  quantile(meanplvLeftMat, 1-lThresh)];
ArrayForThresh.pliLeft         = [quantile(meanpliLeftMat, lThresh), ...
                                  quantile(meanpliLeftMat, 1-lThresh)];

ArrayForThresh.cohRight        = [quantile(meancohRightMat, lThresh), ...
                                  quantile(meancohRightMat, 1 - lThresh)];
ArrayForThresh.icohRight       = [quantile(meanicohRightMat, lThresh), ...
                                  quantile(meanicohRightMat, 1 - lThresh)];
ArrayForThresh.pcohRight       = [quantile(meanpcohRightMat, lThresh), ...
                                  quantile(meanpcohRightMat, 1 - lThresh)];
ArrayForThresh.ipcohRight      = [quantile(meanipcohRightMat, lThresh), ...
                                  quantile(meanipcohRightMat, 1 - lThresh)];
ArrayForThresh.plvRight        = [quantile(meanplvRightMat, lThresh), ...
                                  quantile(meanplvRightMat, 1 - lThresh)];
ArrayForThresh.pliRight        = [quantile(meanpliRightMat, lThresh), ...
                                  quantile(meanpliRightMat, 1 - lThresh)];

ArrayForThresh.coh             = [quantile(meancohMat, lThresh), ...
                                  quantile(meancohMat, 1 - lThresh)];
ArrayForThresh.icoh            = [quantile(meanicohMat, lThresh), ...
                                  quantile(meanicohMat, 1 - lThresh)];
ArrayForThresh.pcoh            = [quantile(meanpcohMat, lThresh), ...
                                  quantile(meanpcohMat, 1 - lThresh)];
ArrayForThresh.ipcoh           = [quantile(meanipcohMat, lThresh), ...
                                  quantile(meanipcohMat, 1 - lThresh)];
ArrayForThresh.plv             = [quantile(meanplvMat, lThresh), ...
                                  quantile(meanplvMat, 1 - lThresh)];
ArrayForThresh.pli             = [quantile(meanpliMat, lThresh), ...
                                  quantile(meanpliMat, 1 - lThresh)];

%%
% close all;
cfg                            = [];
cfg.method                     = 'ortho';
cfg.crosshair                  = 'yes';
cfg.funparameter               = 'pliLeft';
% cfg.funcolormap                = '*RdBu';
cfg.funcolormap                = 'YlOrRd';
cfg.funcolorlim                = ArrayForThresh.(cfg.funparameter);
cfg.location                   = [-50 25 61];
ft_sourceplot(cfg, interpConnec);

%%
% close all;
cfg = [];
cfg.method        = 'ortho';
cfg.crosshair = 'yes';
cfg.funparameter  = 'pliLeft';
% cfg.funcolormap = '*RdBu';
cfg.funcolormap = 'YlOrRd';
% cfg.maskparameter = 'mask';
% cfg.funcolorlim = [-0.015 0.015];
% cfg.funcolorlim = [6 10] .* 0.001;
cfg.funcolorlim = [0.31 0.36];

% sourceVisualize.mask = is_top_source;
% cfg.funcolorlim   = [0 0.06];
% cfg.location = [2 38 48];
ft_sourceplot(cfg, interpConnec);