clear; close all; clc;
%% Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/'); % 2022 doesn't work well for sourerecon
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);

%%
subjID                             = 2;
if ismember(subjID, [2, 5])
    nMrkFiles                      = 2;
end
subRoot                            = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-' num2str(subjID, '%02d') ...
                                      '/meg/sub-' num2str(subjID, '%02d') '_task-mgs_'];
subDerivativesRoot                 = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-' num2str(subjID, '%02d') ...
                                      '/meg/sub-' num2str(subjID, '%02d') '_task-mgs_'];
% Load template anatomical
mri_path                           = '/d/DATD/hyper/software/fieldtrip-20250318/template/anatomy/single_subj_T1.nii';
anatMRI                            = ft_read_mri(mri_path);
anatMRI.coordsys                   = 'ras';

%% Read headshape file for the subject
hspPath                            = [subRoot 'headshape.hsp'];
hspData                            = ft_read_headshape(hspPath, 'unit', 'm');
if subjID                          == 2
    rows_to_remove                 = hspData.pos(:,2) > 0.1;
    hspData.pos(rows_to_remove,:)  = [];
end
hspData.fid.label                  = {'Nasion', 'LPA', 'RPA'};

% Read elpData
elpPath                            = [subRoot 'electrodes.elp'];
[fidData, hpiData]                 = readelpFile(elpPath);

% Read raw data for gradiometers
load([subDerivativesRoot 'run-02_raw.mat'])
gradData                           = data.grad;
gradData                           = ft_convert_units(gradData, 'm');
clearvars data;

% Read hpilocations in gradiometer space
hpiMrkData                         = NaN(nMrkFiles, size(hpiData, 1), size(hpiData, 2));
for mrkIdx                         = 1:nMrkFiles
    raw_file_mrk                   = [subRoot 'marker_' num2str(mrkIdx, '%02d') '.sqd'];
    hdr_Mrk                        = ft_read_header(raw_file_mrk);
    hpiMrkData(mrkIdx, :, :)       = cat(1, hdr_Mrk.orig.coregist.hpi.meg_pos);
end
hpiMrkData                         = squeeze(mean(hpiMrkData, 1));

%hspData is the headshape of the subject, fid in it has the fiducials
%fidData is fiducials in the scanner space
%hpiData is marker locations in scanner space

%%
% Here we realign the headshape to hpi by aligning the hpicoil locations,
% treating them as fiducials
headshape.pos                     = [hpiData; fidData; hspData.pos];
headshape.unit                    = 'm'; 
num_remaining_points              = size(hspData.pos, 1); 
remaining_labels                  = cell(num_remaining_points, 1);
for i                             = 1:num_remaining_points
    remaining_labels{i}           = sprintf('head_%d', i);
end
headshape.label                   = [{'LPA', 'RPA', 'NAS', 'HPI4', 'HPI5', 'nreal', ...
                                      'lreal', 'rreal'}, remaining_labels']';

headshape_dummy                   = headshape;
headshape_dummy.elecpos           = headshape.pos;
headshape_dummy.chanpos           = headshape_dummy.elecpos;

hpi_coil.elecpos                  = hpiMrkData;%(1:3, :);
hpi_coil.label                    = {'LPA', 'RPA', 'NAS', 'HPI4', 'HPI5'}';
hpi_coil.unit                     = 'm';

cfg                               = [];
cfg.method                        = 'fiducial';
cfg.template                      = hpi_coil;
cfg.elec                          = headshape_dummy;
cfg.feedback                      = 'yes';
cfg.fiducial                      = {'NAS', 'LPA', 'RPA'}';
headshape_aligned                 = ft_electroderealign(cfg);
%%
% hspCorrected is now the headshape points and fiducials aligned to the
% hpi locations in scanner space
hspCorrected                      = hspData;
hspCorrected.pos                  = headshape_aligned.chanpos(9:end, :);
hspCorrected.fid.pos              = headshape_aligned.chanpos(6:8, :); % real fiducials
hspCorrected.fid.label            = {'NAS', 'LPA', 'RPA'}; 
% Convert units back to mm
hspCorrected                      = ft_convert_units(hspCorrected, 'mm');
gradData                          = ft_convert_units(gradData, 'mm');
hpiMrkData                        = hpiMrkData .* 1000;

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
ft_plot_mesh(hpiData .* 1000, 'vertexcolor', 'b', 'vertexsize', 20)
ft_plot_mesh(hpiMrkData, 'vertexcolor', 'm', 'vertexsize', 20)

%% Align anatomical with hsp
restoredefaultpath;
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;

anatMRItransformed                = ft_convert_coordsys(anatMRI, 'als');

cfg                               = [];
cfg.method                        = 'interactive';
cfg.spmversion                    = 'spm12';
% cfg.coordsys                      = 'als';
mri_realigned                     = ft_volumerealign(cfg, anatMRItransformed);

cfg                               = [];
cfg.spmversion                    = 'spm12';
cfg.method                        = 'headshape';
cfg.headshape.headshape           = hspCorrected;
cfg.headshape.coordsys            = 'als';
cfg.headshape.icp                 = 'yes';
cfg.headshape.interactive         = 'yes';
mri_aligned                    = ft_volumerealign(cfg, mri_realigned);

cfg = [];
cfg.method = 'ortho'; % Orthographic view
cfg.interactive = 'yes';
ft_sourceplot(cfg, anatMRI);
% 
% % To check fiducial alignment
% hold on;
% ft_plot_sens(mri_aligned.cfg.fiducial, 'label', 'yes', 'orientation', true);
% 
% cfg = [];
% cfg.interactive = 'yes';
% ft_volumerealign(cfg, mri_aligned);
% gradData.fid.pnt     = hspCorrected.fid.pos;
% gradData.fid.label   = hspCorrected.fid.label;

% For sub-12:
% rotate (0, 15, 270)
% scale (1, 1, 1)
% translate (0, 2, 13)

% cfg = [];
% cfg.method = 'ortho';
% cfg.interactive = 'yes';
% ft_sourceplot(cfg, anatMRI)
figure;
ft_plot_headshape(hspCorrected);
hold on;
ft_plot_ortho(mri_aligned.anatomy,'transform',mri_aligned.transform,'style','intersect');

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

% % cfg                  = [];
% % cfg.nonlinear        = 'no';
% % cfg.templatecoordys  = 'ras';
% % mri_reslice          = ft_volumenormalise(cfg, mri_reslice);
% % Segment the mri
cfg                  = [];
cfg.output           = {'brain'};
segmentedmri         = ft_volumesegment(cfg, mri_reslice);

% ft_volumegui([], segmentedmri);  % Check segmentation

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
restoredefaultpath;
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
cfg                  = [];
cfg.nonlinear        = 'yes';
mri_reslice_mni      = ft_volumenormalise(cfg, mri_reslice);

transform_vox2ctf    = mri_reslice.transform;
transform_vox2acpc   = inv(mri_reslice_mni.transform);
transform_acpc2ctf   = transform_vox2ctf / transform_vox2acpc;

cfg                  = [];
cfg.output           = {'brain'};
segmentedmri_mni         = ft_volumesegment(cfg, mri_reslice_mni);

cfg                  = [];
cfg.method           = 'singleshell';
singleShellHeadModel_mni = ft_prepare_headmodel(cfg, segmentedmri_mni);
%%
singleShellHeadModel_mni_transformed = ft_transform_geometry(transform_vox2acpc, singleShellHeadModel_mni);
figure;
ft_plot_mesh(singleShellHeadModel.bnd, 'facecolor', 'g',...
    'facealpha', 0.2); % brain
hold on;
ft_plot_mesh(singleShellHeadModel_mni_transformed.bnd, 'facecolor', 'g',...
    'facealpha', 0.2);
%%
figure;
plot3(templateheadmodel.bnd.pos(:, 1), templateheadmodel.bnd.pos(:, 2), templateheadmodel.bnd.pos(:, 3), 'ro')
hold on;
ft_plot_sens(gradData, 'facecolor', 'green', 'facealpha', 1)
ft_plot_ortho(mri_reslice_mni.anatomy, 'style', 'intersect', ...
    'transform', mri_reslice_mni.transform)
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


% T_orig               = mri_aligned.transformorig;
% T_new                = mri_aligned.transform;
% T_gradspace           = T_new * T_orig; 
% load('/d/DATD/hyper/software/fieldtrip-20250318/template/headmodel/standard_seg.mat');
% load('/d/DATD/hyper/software/fieldtrip-20250318/template/headmodel/standard_singleshell.mat');
load('/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/standard_sourcemodel3d8mm.mat');
% mri_reslice          = mri;
% mri_reslice.coorsys  = 'ras';
% 
templateheadmodel    = ft_convert_units(vol, 'mm');
templateheadmodel    = ft_transform_geometry(transform_acpc2ctf, templateheadmodel);
% cfg                  = [];
% cfg.grid.resolution  = 8;
% cfg.grid.unit        = 'mm';
% cfg.tight            = 'yes';
% % cfg.inwardshift      = 5;
% cfg.headmodel        = templateheadmodel;
% templateGrid         = ft_prepare_sourcemodel(cfg);
% templateGrid         = sourcemodel;
templateGrid         = ft_convert_units(sourcemodel, 'mm');
% templateGrid         = ft_transform_geometry(transform_acpc2vox, templateGrid);

% cfg                  = [];
% cfg.warpmni          = 'yes';
% cfg.template         = templateGrid;
% cfg.nonlinear        = 'yes';
% cfg.mri              = mri_reslice;
% cfg.sourcemodel.unit = 'mm';
% grid                 = ft_prepare_sourcemodel(cfg);

cfg                    = [];
cfg.method           = 'basedonmni';
cfg.template         = templateGrid;
cfg.nonlinear        = 'yes';
cfg.mri              = mri_aligned;
cfg.unit             = 'mm';
% % cfg.grid.resolution  = 8;
% % cfg.inwardshift      = 'yes';
% % cfg.tight            = 'yes';
% % cfg.sourcemodel.unit    = 'mm';
% % 
grid                 = ft_prepare_sourcemodel(cfg);

% cfg                  = [];
% cfg.method           = 'basedonmri';
% cfg.mri              = segmentedmri;
% cfg.threshold        = 0.1;
% cfg.smooth           = 5;
% cfg.resolution       = 0.01; % in m
% cfg.sourcemodel.unit = 'm';
% cfg.tight            = 'yes';
% cfg.inwardshift      = 0.005; % Around 5mm inward shift
% cfg.headmodel        = singleShellHeadModel;
% grid                 = ft_prepare_sourcemodel(cfg);
%%
% cfg                  = [];
% cfg.nonlinear        = 'no';
% mri_reslice_mni      = ft_volumenormalise(cfg, mri_reslice);
figure;
hold on;
ft_plot_headmodel(singleShellHeadModel, 'edgecolor', 'g', 'facealpha', 0.4);
ft_plot_mesh(grid.pos(grid.inside, :))
%% plot the shared individual grid
figure;
ft_plot_mesh(singleShellHeadModel.bnd, 'facecolor', 'g',...
    'facealpha', 0.2); % brain
hold on;
ft_plot_mesh(templateheadmodel.bnd, 'facecolor', 'g',...
    'facealpha', 0.2);
% ft_plot_mesh(grid.pos(grid.inside,:), ...
%     'vertexcolor', 'r');
% %
% 
% ft_plot_ortho(mri_reslice.anatomy, 'style', 'intersect', ...
%     'transform', mri_reslice.transform)
% hold on;
% % ft_plot_mesh(headmodel.bnd, 'facecolor', 'k', 'facealpha', 0.2); % brain
% % view([0 -1 0]); % from the right side
% % 
% % 
% % ft_plot_mesh(grid.pos(grid.inside,:), ...
% %     'vertexcolor', [1 0 0]);
% ft_plot_sens(gradData, 'facecolor', 'g','edgecolor', 'g')
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
% % Create leadfield
% cfg                  = [];
% cfg.grad             = gradData;
% cfg.channel          = gradData.label;
% cfg.grid             = grid;
% cfg.headmodel        = singleShellHeadModel;
% leadfield            = ft_prepare_leadfield(cfg);
% 
% %%
% figure;
% plot3(singleShellHeadModel.bnd.pos(:, 1), singleShellHeadModel.bnd.pos(:, 2), singleShellHeadModel.bnd.pos(:, 3), 'ro')
% hold on;
% plot3(gradData.chanpos(:, 1), gradData.chanpos(:, 2), gradData.chanpos(:, 3), 'bs')
% plot3(leadfield.pos(:, 1), leadfield.pos(:, 2), leadfield.pos(:, 3), 'yo')
% axis equal;

%% plot individual grid
sourcemodelfPath     = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/anatomy/sourcemodel.mat';
save(sourcemodelfPath, 'gradData', 'mrkData', 'hspCorrected', 'mri_reslice', ...
    'mri_aligned', 'pial', 'segmentedmri', 'singleShellHeadModel', 'grid', '-v7.3');
load(sourcemodelfPath);

%% lcmv beamformer
% Load stimlocked data
load('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/meg/sub-12_task-mgs_stimlocked_lineremoved.mat')
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
% 
% %%%%% Average reference >> absolutely not
% cfg                               = [];
% cfg.reref                         = 'yes';
% cfg.refchannel                    = 'all';
% cfg.refmethod                     = 'avg';
% epocLeft                          = ft_preprocessing(cfg, epocLeft);
% epocRight                         = ft_preprocessing(cfg, epocRight);
epocCombined                      = ft_appenddata([], epocLeft, epocRight);

%%
cfg = [];
cfg.bpfilter = 'yes';
cfg.bpfreq = [20 25];
epoc_filter = ft_preprocessing(cfg, epocCombined);
cfg = [];
cfg.latency = [-0.5 0];
epoc_pre = ft_selectdata(cfg, epoc_filter);
cfg.latency = [1 1.5];
epoc_post = ft_selectdata(cfg, epoc_filter);
epocCombined                      = ft_appenddata([], epoc_pre, epoc_post);

%%
% Compute timelocked data with covariance
cfg                               = [];
cfg.covariance                    = 'yes';
cfg.covariancewindow              = 'all';
cfg.keeptrials                    = 'no';
% timelockedLeft                    = ft_timelockanalysis(cfg, epocLeft);
% timelockedRight                   = ft_timelockanalysis(cfg, epocRight);
timelockedCombined                = ft_timelockanalysis(cfg, epocCombined);
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
% cfg.normalize                     = 'yes'; % normalize LF to attenuate depth bias
% cfg.normalizeparam                = 0.5; % default = 0.5
source                            = ft_sourceanalysis(cfg, timelockedCombined);
inside_pos                        = find(source.inside);

%%
% figure();
% ft_plot_mesh(source); , 'vertexcolor', 100)
% 
% cfg = [];
% cfg.funparameter = 'pow';
% ft_sourcemovie(cfg, source);

% % Reconstruct sensor data
% source_signal = cat(1, source.avg.mom{inside_pos});
% % reconstructed_data = source.leadfield{inside_pos(1)} * source_signal(1:3,:); % For first source
% reconstructed_data = W_meg' * source_signal;
% 
% % Compare to original ERP at peak latency
% [~, peak_idx] = max(abs(squeeze(mean(timelockedCombined.avg, 1, 'omitnan'))));
% figure; plot(timelockedCombined.time, timelockedCombined.avg(1, :), 'k'); hold on;
% plot(timelockedCombined.time, reconstructed_data(1, :), 'r--');
% axvline(timelockedCombined.time(peak_idx))
% legend('Original', 'Reconstructed');


%% Extract all filters for inside voxels
W_meg               = cell2mat(source.avg.filter(source.inside));

%% Create sourcedata
sourcedata          = [];
sourcedata.label    = cell(numel(inside_pos), 1);
for i               = 1:numel(inside_pos)
    sourcedata.label{i} ...
                    = sprintf('S_%d', inside_pos(i));
end
sourcedataPost      = sourcedata;
sourcedataPre     = sourcedata;

epocLeft_filt       = epocLeft;
epocRight_filt      = epocRight;
%%
for iTrial          = 1:size(epoc_pre.trialinfo, 1)
    sourcedataPre.trial{iTrial} ...
                    = W_meg * epoc_pre.trial{iTrial};
end
for iTrial          = 1:size(epoc_post.trialinfo, 1)
    sourcedataPost.trial{iTrial} ...
                    = W_meg * epoc_post.trial{iTrial};
end

vars_pre = zeros(size(var(sourcedataPre.trial{tt},0, 2)));
vars_post  = zeros(size(var(sourcedataPre.trial{tt},0, 2)));

for tt = 1: numel(sourcedataPre.trial)
    vars_pre = vars_pre + var(sourcedataPre.trial{tt},0, 2);
end
vars_pre = vars_pre./numel(sourcedataPre.trial)
for tt = 1: numel(sourcedataPost.trial)
    vars_post = vars_post + var(sourcedataPost.trial{tt},0, 2);
end
vars_post = vars_post./numel(sourcedataPost.trial);

source_val = (vars_post - vars_pre)./(vars_post + vars_pre);

%%
% Bandpass filter data in frequency of interest
cfg                 = [];
cfg.bpfilter        = 'yes';
cfg.bpfreq          = [20 25];
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
cfg.parameter       = 'avg';  
cfg.operation       = '(x1-x2)/(x1+x2)';
sourceDiff          = ft_math(cfg, sourceDataLeft_avg, sourceDataRight_avg);

% clearvars sourcedataLeft sourcedataRight;

%% Visualize source
TOI                 = find(sourceDiff.time > 1.0 & sourceDiff.time < 1.5);
sourceVisualize     = source;
sourceVisualize.pow =  nan(size(source.inside));
sourceVisualize.pow(source.inside)= source_val;

alize.lateralizedPow ...
                    = NaN(size(source.inside));
sourceVisualize.lateralizedPow(source.inside) ...
                    = squeeze(mean(sourceDiff.avg(:, TOI), 2, 'omitnan'));

%%
cfg                 = [];
cfg.parameter       = {'pow'};
[interp]            = ft_sourceinterpolate(cfg, sourceVisualize, mri_reslice);

cfg = [];
cfg.method        = 'ortho';
cfg.crosshair = 'yes';
cfg.funparameter  = 'pow';
cfg.funcolormap = 'jet';
% cfg.funcolorlim   = [0 0.06];
cfg.location = [2 38 48];
ft_sourceplot(cfg, interp);


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
cfg.sourcemodel      = grid;
cfg.headmodel        = singleShellHeadModel;
leadfield            = ft_prepare_leadfield(cfg, epocCombined);


% cfg                        = [];
% cfg.frequency              = freqThis.freq;
% cfg.method                 = 'pcc';
% cfg.grid                   = g;
% cfg.headmodel              = singleShellHeadModel;
% cfg.keeptrials             = 'yes';
% cfg.pcc.lambda             = '10%';
% cfg.pcc.projectnoise       = 'yes';
% cfg.pcc.fixedori           = 'yes';
% sourceFreq                 = ft_sourceanalysis(cfg, freqThis);


cfg              = [];
cfg.method       = 'dics';
cfg.frequency    = 6.0;
cfg.sourcemodel  = leadfield;
cfg.grad         = gradData;
cfg.channel      = gradData.label;
cfg.headmodel    = singleShellHeadModel;
cfg.dics.projectnoise = 'yes';
cfg.dics.lambda       = '5%';
cfg.dics.keepfilter   = 'yes';
cfg.dics.realfilter   = 'yes';
sourceCombined = ft_sourceanalysis(cfg, freqCombined);

cfg.sourcemodel.filter = sourceCombined.avg.filter;

sourceLeft  = ft_sourceanalysis(cfg, freqLeft );
sourceRight = ft_sourceanalysis(cfg, freqRight);

sourceDiff = sourceLeft;
sourceDiff.avg.pow = (sourceLeft.avg.pow - sourceRight.avg.pow) ./ (sourceLeft.avg.pow + sourceRight.avg.pow);


cfg            = [];
cfg.downsample = 2;
cfg.parameter  = 'pow';
sourceDiff_interp  = ft_sourceinterpolate(cfg, sourceDiff, mri_reslice);

cfg              = [];
cfg.method       = 'slice';
cfg.funparameter = 'pow';
cfg.funcolorlim   = [-0.02 0.02];
cfg.opacitylim    = [-0.02 0.02];
cfg.opacitymap    = 'rampup';
ft_sourceplot(cfg, sourceDiff_interp);


pial.coordsys = 'ras';


cfg = [];
cfg.parameter = 'pow';
cfg.interpmethod = 'nearest';
cfg.sphereradius = 20; % Adjust based on your needs
sourceDiffPial = ft_sourceinterpolate(cfg, sourceDiff, pial);

cfg = [];
cfg.method = 'surface';
cfg.funparameter = 'pow';
cfg.funcolormap = 'parula';
cfg.funcolorlim   = [-0.02 0.02];
cfg.opacitylim = [0 1];
cfg.opacitymap = 'rampup';  % Add transparency effect

ft_sourceplot(cfg, sourceDiffPial);


cfg = [];
cfg.method         = 'surface';
cfg.funparameter   = 'pow';
cfg.maskparameter  = cfg.funparameter;
% cfg.funcolorlim    = [0.0 maxval];
cfg.funcolormap    = 'parula';
% cfg.opacitylim     = [0.0 maxval];
cfg.opacitymap     = 'rampup';
cfg.projmethod     = 'nearest';
% cfg.surffile       = pial;
cfg.surface        = pial;
cfg.surfdownsample = 10;
ft_sourceplot(cfg, sourceDiff_interp);
