clear; close all; clc;
%% Initialization
% addpath('/d/DATD/hyper/software/fieldtrip-20220104/');
addpath('/d/DATD/hyper/software/fieldtrip-20250318/'); % 2022 doesn't work well for sourerecon

addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;

%%
% Load anatomicals
mri_path             = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/SUMA/T1.nii';
pial_l_path          = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/surf/lh.pial.asc';
pial_r_path          = '/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/surf/rh.pial.asc';

anatMRI              = ft_read_mri(mri_path);
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


coordsys             = ft_determine_coordsys(anatMRI);
% Set the coordinate system to 'r a s'
anatMRI.coordsys     = coordsys.coordsys;

%% Read headshape file
hspPath              = '/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-12/meg/sub-12_task-mgs_headshape.hsp';
hspData              = ft_read_headshape(hspPath);
hspData.unit         = 'm';
hspData.fid.label    = {'Nasion', 'LPA', 'RPA'}';

% Read raw data for gradiometers
load('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/meg/sub-12_task-mgs_run-01_raw.mat')
gradData             = data.grad;
gradData             = ft_convert_units(gradData, 'm');
clearvars data;

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
ft_plot_sens(gradData, 'box', 1)


%% Align anatomical with hsp
cfg                  = [];
cfg.method           = 'headshape';
cfg.headshape.headshape ...
                     = hspData;
cfg.headshape.icp    = 'yes';
cfg.headshape.interactive ...66
                     = 'yes';
mri_aligned          = ft_volumerealign(cfg, anatMRI);


% For sub-12:
% rotate (0, 30, -92)
% scale (0.96, 0.96, 0.96)
% translate (0, 0, 42)
% ft_volumerealign(cfg, mri_aligned)

%% Transform pial accordingly
pial                 = [];
pial.pos             = [surfL.vertices; surfR.vertices];
pial.tri             = [surfL.faces; surfR.faces + size(surfL.vertices, 1)];

% Calculate the combined transformation
T_orig               = mri_aligned.transformorig;
T_new                = mri_aligned.transform;
T_combined           = T_new / T_orig;  % This is equivalent to T_new * inv(T_orig)

% Apply the transformation to the pial surface vertices
homogeneous_coords   = [pial.pos, ones(size(pial.pos, 1), 1)]';
transformed_coords   = T_combined * homogeneous_coords;
pial.pos             = transformed_coords(1:3, :)';
pial                 = ft_convert_units(pial, 'm');

%% Slice, segment and create headmodel
% Slice the data
mri_reslice          = ft_volumereslice([], mri_aligned);
mri_reslice          = ft_convert_units(mri_reslice, 'm');
% Segment the mri
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
plot3(gradData.chanpos(:, 1), gradData.chanpos(:, 2), gradData.chanpos(:, 3), 'bs')
axis equal;
%%
% Volume conduction model
% cfg                  = [];
% cfg.method           = 'dipoli';
% dipoliHeadModel      = ft_prepare_headmodel(cfg, singleShellHeadModel);
% % 
% % % Visualize the headmodel
% figure;
% plot3(dipoliHeadModel.bnd.pos(:, 1), dipoliHeadModel.bnd.pos(:, 2), dipoliHeadModel.bnd.pos(:, 3), 'ro')
% hold on;
% plot3(gradData.chanpos(:, 1), gradData.chanpos(:, 2), gradData.chanpos(:, 3), 'bs')
% axis equal;


% Create a grid for this subject
% cfg                  = [];
% cfg.sourcemodel.resolution  = 0.01; % in mm
% cfg.sourcemodel.unit        = 'm';
% cfg.tight            = 'yes';
% cfg.inwardshift      = -2;
% cfg.headmodel        = singleShellHeadModel;
% grid                 = ft_prepare_sourcemodel(cfg);


cfg                  = [];
cfg.method           = 'basedonmri';
cfg.mri              = segmentedmri;
cfg.threshold        = 0.1;
cfg.smooth           = 5;
cfg.resolution       = 0.004; % in m
cfg.sourcemodel.unit = 'm';
cfg.tight            = 'yes';
% cfg.inwardshift      = -0.002;
cfg.headmodel        = singleShellHeadModel;
grid                 = ft_prepare_sourcemodel(cfg);
%%
idxChosen = grid.inside;
figure;
plot3(singleShellHeadModel.bnd.pos(:, 1), singleShellHeadModel.bnd.pos(:, 2), singleShellHeadModel.bnd.pos(:, 3), 'ro')
hold on;
plot3(gradData.chanpos(:, 1), gradData.chanpos(:, 2), gradData.chanpos(:, 3), 'bs')
plot3(grid.pos(idxChosen, 1), grid.pos(idxChosen, 2), grid.pos(idxChosen, 3), 'ko')
axis equal;
%%
% Create leadfield
cfg                  = [];
cfg.grad             = gradData;
cfg.channel          = gradData.label;
cfg.grid             = grid;
cfg.headmodel        = singleShellHeadModel;
leadfield            = ft_prepare_leadfield(cfg);

%%
figure;
plot3(singleShellHeadModel.bnd.pos(:, 1), singleShellHeadModel.bnd.pos(:, 2), singleShellHeadModel.bnd.pos(:, 3), 'ro')
hold on;
plot3(gradData.chanpos(:, 1), gradData.chanpos(:, 2), gradData.chanpos(:, 3), 'bs')
plot3(leadfield.pos(:, 1), leadfield.pos(:, 2), leadfield.pos(:, 3), 'yo')
axis equal;

%% plot individual grid
figure();
ft_plot_mesh(dipoliHeadModel.bnd, 'facecolor', 'k',...
    'facealpha', 0.2); % scalp
hold on;
ft_plot_mesh(grid.pos(grid.inside,:), ...
    'vertexcolor', 'r');


sourcemodelfPath     = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/anatomy/sourcemodel.mat';
save(sourcemodelfPath, 'pial', 'mri_aligned', 'mri_reslice', 'segmentedmri', ...
    'singleShellHeadModel', 'grid', '-v7.3')

load(sourcemodelfPath);

%% lcmv beamformer
% cfg                 = [];
% cfg.frequency       = [5 8];
% cfg.latency         = [-0.1 1.7];
% TFR_selected        = ft_selectdata(cfg, TFR_fourier_left);

% Load stimlocked data
load('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/meg/sub-12_task-mgs_stimlocked.mat')
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

%%%%% Average reference
% cfg                               = [];
% cfg.reref                         = 'yes';
% cfg.refchannel                    = 'all';
% cfg.refmethod                     = 'avg';
% epocLeft                          = ft_preprocessing(cfg, epocLeft);
% epocRight                         = ft_preprocessing(cfg, epocRight);


% cfg                               = [];
% cfg.covariance                    = 'yes';
% cfg.covariancewindow              = 'all';
% timelockedLeft                    = ft_timelockanalysis(cfg, epocLeft);
% timelockedRight                   = ft_timelockanalysis(cfg, epocRight);

%%        
cfg                               = [];
cfg.method                        = 'lcmv';
cfg.sourcemodel                   = grid;
cfg.headmodel                     = singleShellHeadModel;
cfg.grad                          = gradData;
cfg.keepleadfield                 = 'yes';
cfg.lcmv.keepfilter               = 'yes';
cfg.lcmv.fixedori                 = 'yes';
cfg.lcmv.lambda                   = '1%';  % must be '1%' not 1
cfg.normalize                     = 'yes'; % normalize LF to attenuate depth bias
cfg.normalizeparam                = 0.5; % default = 0.5
source                            = ft_sourceanalysis(cfg, timelockedLeft);
inside_pos                        = find(source.inside);

% figure();
% ft_plot_mesh(source); , 'vertexcolor', 100)
% 
% cfg = [];
% cfg.funparameter = 'pow';
% ft_sourcemovie(cfg, source);

%% Extract all filters for inside voxels
W_meg               = cell2mat(source.avg.filter(source.inside));

%% Create sourcedata
sourcedata          = [];
sourcedata.label    = cell(numel(inside_pos), 1);
for i               = 1:numel(inside_pos)
    sourcedata.label{i} ...
                    = sprintf('S_%d', inside_pos(i));
end
sourcedataLeft      = sourcedata;
sourcedataRight     = sourcedata;

epocLeft_filt       = epocLeft;
epocRight_filt      = epocRight;
cfg                 = [];
cfg.lpfreq          = 50;
cfg.lpfilter        = 'yes';
epocLeft_filt       = ft_preprocessing(cfg, epocLeft);
epocRight_filt      = ft_preprocessing(cfg, epocRight);

cfg                 = [];
cfg.resamplefs      = 50;
cfg.demean          = 'no';
cfg.detrend         = 'no';
epocLeft_filt       = ft_resampledata(cfg, epocLeft_filt);
epocRight_filt      = ft_resampledata(cfg, epocRight_filt);

% Bandpass filter data in frequency of interest
cfg                 = [];
cfg.bpfilter        = 'yes';
cfg.bpfreq          = [8 12];
sourcedataLeft.trial ...
                    = cell(1, size(epocLeft_filt.trialinfo, 1));
sourcedataLeft.time = epocLeft_filt.time; % thEpocData.time{1};
sourcedataRight.trial ...
                    = cell(1, size(epocRight_filt.trialinfo, 1));
sourcedataRight.time ...
                    = epocRight_filt.time;
for iTrial          = 1:size(epocLeft_filt.trialinfo, 1)
    sourcedataLeft.trial{iTrial} ...
                    = W_meg * epocLeft_filt.trial{iTrial};
end
for iTrial          = 1:size(epocRight_filt.trialinfo, 1)
    sourcedataRight.trial{iTrial} ...
                    = W_meg * epocRight_filt.trial{iTrial};
end

sourcedataLeft      = ft_preprocessing(cfg, sourcedataLeft);
sourcedataRight     = ft_preprocessing(cfg, sourcedataRight);

% Estimate power using hilbert transform
for iTrial          = 1:size(epocLeft_filt.trialinfo, 1)
    for iSource     = 1:numel(inside_pos)               
        sourcedataLeft.trial{iTrial}(iSource, :) ...
                    = abs(hilbert(sourcedataLeft.trial{iTrial}(iSource, :)).^2);
    end
end

for iTrial          = 1:size(epocRight_filt.trialinfo, 1)
    for iSource     = 1:numel(inside_pos)               
        sourcedataRight.trial{iTrial}(iSource, :) ...
                    = abs(hilbert(sourcedataRight.trial{iTrial}(iSource, :)).^2);
    end
end


%%
cfg                 = [];
sourceDataLeft_avg  = ft_timelockanalysis(cfg, sourcedataLeft);
sourceDataRight_avg = ft_timelockanalysis(cfg, sourcedataRight);

cfg                 = [];
cfg.parameter       = 'avg';  % Specify the parameter to operate on
% cfg.operation       = '(10^(x1/10) - 10^(x2/10)) / (10^(x1/10) + 10^(x2/10))';
cfg.operation       = '(x1-x2)/(x1+x2)';
sourceDiff          = ft_math(cfg, sourceDataLeft_avg, sourceDataRight_avg);

%% Visualize source
TOI                 = find(sourceDiff.time > 0.5 & sourceDiff.time < 1.0);
sourceVisualize     = source;
sourceVisualize.lateralizedPow ...
                    = NaN(size(source.inside));
sourceVisualize.lateralizedPow(source.inside) ...
                    = squeeze(mean(sourceDiff.avg(:, TOI), 2, 'omitnan'));

%%
cfg                 = [];
cfg.parameter       = {'lateralizedPow'};
[interp]            = ft_sourceinterpolate(cfg, sourceVisualize, mri_reslice);

cfg = [];
cfg.method        = 'ortho';
cfg.crosshair = 'yes';
cfg.funparameter  = 'lateralizedPow';
cfg.funcolormap = 'jet';
% cfg.funcolorlim   = [0 0.06];
cfg.location = [2 38 48];
ft_sourceplot(cfg, interp);

%% try on pial surface
pial = [];
pial.pos = [surfL.vertices; surfR.vertices];
pial.tri = [surfL.faces; surfR.faces + size(surfL.vertices, 1)];
% pial.unit = 'm';
% pial.pos(:, [2, 3, 1]) = pial.pos(:, [3, 1, 2]);
% pial.coordsys = 'ras';

T_orig = mri_aligned.transformorig;
T_new = mri_aligned.transform;

% Calculate the combined transformation
T_combined = T_new / T_orig;  % This is equivalent to T_new * inv(T_orig)

% T_combined = [[ 0      0     -1 5]; ...
%               [-0.866 -0.5    0 0]; ...
%               [-0.5   -0.866  0 4]; ...
%               [ 0      0      0 1]];
% Apply the transformation to the pial surface vertices
homogeneous_coords = [pial.pos, ones(size(pial.pos, 1), 1)]';
transformed_coords = T_combined * homogeneous_coords;
pial.pos = transformed_coords(1:3, :)';
pial = ft_convert_units(pial, 'm');

% theta = -pi/2; % 90 degrees in radians
% rotMat = [
%     cos(theta), -sin(theta), 0;
%     sin(theta),  cos(theta), 0;
%     0,           0,          1
% ];
% rotated_pial_pos = (rotMat * pial.pos')'; % Transpose for matrix multiplication
% pial.pos = rotated_pial_pos;
% transform = mri_aligned.transform;
% % 
% % % Apply the transformation to the vertices
% homogeneous_coords = [pial.pos, ones(size(pial.pos, 1), 1)];
% transformed_coords = homogeneous_coords  * transform;
% pial.pos = transformed_coords(:, 1:3);

% coordsys = ft_determine_coordsys(pial);
% pial.coordsys = coordsys;

%%
figure;
plot3(singleShellHeadModel.bnd.pos(:, 1), singleShellHeadModel.bnd.pos(:, 2), singleShellHeadModel.bnd.pos(:, 3), 'ro')
hold on;
plot3(gradData.chanpos(:, 1), gradData.chanpos(:, 2), gradData.chanpos(:, 3), 'bs')
% plot3(leadfield.pos(:, 1), leadfield.pos(:, 2), leadfield.pos(:, 3), 'yo')
plot3(pial.pos(1:100:400000, 1), pial.pos(1:100:400000, 2), pial.pos(1:100:400000, 3), 'yo')
axis equal;

%%
cfg = [];
cfg.parameter = 'lateralizedPow';
cfg.interpmethod = 'nearest';
cfg.sphereradius = 20; % Adjust based on your needs
sourceInterp = ft_sourceinterpolate(cfg, sourceVisualize, pial);

cfg = [];
cfg.method = 'surface';
cfg.funparameter = 'lateralizedPow';
cfg.funcolormap = '*RdBu';
% cfg.opacitymap = 'rampup'; 
% cfg.opacitylim = [0 0.1]; % Adjust for desired transparency
% cfg.funcolorlim = [-max(abs(sourceInterp.lateralizedPow)) max(abs(sourceInterp.lateralizedPow))]; % Symmetric color scaling
% cfg.camlight = 'yes';
ft_sourceplot(cfg, sourceInterp);









%%
cfg                        = [];
cfg.frequency              = [8 12];
cfg.avgoverfreq            = 'yes';
cfg.keepfreqdim            = 'yes';
TFR_fourier_left_alpha     = ft_selectdata(cfg, TFR_fourier_left);
TFR_fourier_right_alpha    = ft_selectdata(cfg, TFR_fourier_right);


cfg                        = [];
cfg.parameter              = 'fourierspctrm';
TFR_fourier_alpha          = ft_appendfreq(cfg, TFR_fourier_left_alpha, TFR_fourier_right_alpha);

cfg                  = [];
cfg.grad             = gradData;
cfg.channel          = gradData.label;
cfg.grid             = grid;
cfg.headmodel        = singleShellHeadModel;
leadfield            = ft_prepare_leadfield(cfg, epocThis);


cfg                        = [];
cfg.frequency              = freqThis.freq;
cfg.method                 = 'pcc';
cfg.grid                   = leadfield;
cfg.headmodel              = singleShellHeadModel;
cfg.keeptrials             = 'yes';
cfg.pcc.lambda             = '10%';
cfg.pcc.projectnoise       = 'yes';
cfg.pcc.fixedori           = 'yes';
sourceFreq                 = ft_sourceanalysis(cfg, freqThis);