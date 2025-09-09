%% MEG Source Reconstruction in Template Space (Clean Version)
% Aligns template MRI to subject headshape using fiducial-based Procrustes analysis
% Enables MEG source reconstruction in standardized template coordinates

clear; close all; clc;

%% Setup and Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);

% Configuration parameters
subjID = 12;
sourcemodel_resolution = 8; % Source model resolution in mm (4, 5, 6, 8, 10)

subRoot = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-' num2str(subjID, '%02d') ...
           '/meg/sub-' num2str(subjID, '%02d') '_task-mgs_'];
subDerivativesRoot = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-' num2str(subjID, '%02d') ...
                      '/meg/sub-' num2str(subjID, '%02d') '_task-mgs_'];

% Load template MRI and set subject-specific parameters
if subjID == 12
    nMrkFiles = 3;
elseif subjID == 4
    nMrkFiles = 1;
elseif subjID == 1
    nMrkFiles = 2;
end

%% Load Template MRI, segmented MRI and headmodel
% Load FieldTrip template MRI (MNI space)
mri_path = '/d/DATD/hyper/software/fieldtrip-20250318/template/headmodel/standard_mri.mat';
load(mri_path, 'mri');
anatMRI = mri;
clear mri;

% Load pre-built template head model
headmodel_path = '/d/DATD/hyper/software/fieldtrip-20250318/template/headmodel/standard_singleshell.mat';
load(headmodel_path, 'vol');
vol = ft_convert_units(vol, 'mm');
vol.coordsys = 'ras';
singleShellHeadModel_template = vol;
clear vol;

% Load Pre-segmented Template MRI (Optional)
seg_path = '/d/DATD/hyper/software/fieldtrip-20250318/template/headmodel/standard_seg.mat';
load(seg_path, 'mri');
segmentedmri_template = mri;
clear mri;

% Load pre-existing forward model
if sourcemodel_resolution == 7.5
    resolution_str = '7point5mm';
else
    resolution_str = sprintf('%dmm', sourcemodel_resolution);
end
sourcemodel_path = sprintf('/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/standard_sourcemodel3d%s.mat', resolution_str);
load(sourcemodel_path, 'sourcemodel');
sourcemodel_template = sourcemodel;
clear sourcemodel;
sourcemodel_template = ft_convert_units(sourcemodel_template, 'mm');

% Skip pial surface loading for speed

% Load VTPM atlas
vtpm_atlas_path = '/d/DATD/hyper/software/fieldtrip-20250318/template/atlas/vtpm/vtpm.mat';
load(vtpm_atlas_path, 'vtpm');
atlas_template = vtpm;
clear vtpm;

% Template Fiducial Definition (MNI space)
template_fid.pos = [
      1.0,  84.0, -43.0;  % Nasion
      89.0, -17.0, -65.0; % RPA (right ear)
     -89.0, -17.0, -65.0  % LPA (left ear)
];
template_fid.label = {'NAS', 'RPA', 'LPA'};
template_fid.unit = 'mm';

%% Subject Data Loading
% Load headshape, MEG sensors, and HPI coil positions
hspPath = [subRoot 'headshape.hsp'];
hspData = ft_read_headshape(hspPath, 'unit', 'm');
hspData.fid.label = {'Nasion', 'LPA', 'RPA'};

elpPath = [subRoot 'electrodes.elp'];
[fidData, hpiData] = readelpFile(elpPath);

load([subDerivativesRoot 'run-01_raw.mat'])
gradData = data.grad;
gradData.coordsys = 'als';
gradData = ft_convert_coordsys(gradData, 'ras');
gradData = ft_convert_units(gradData, 'mm');
clearvars data;

% Process HPI markers
hpiMrkData = NaN(nMrkFiles, size(hpiData, 1), size(hpiData, 2));
for mrkIdx = 1:nMrkFiles
    raw_file_mrk = [subRoot 'marker_' num2str(mrkIdx, '%02d') '.sqd'];
    hdr_Mrk = ft_read_header(raw_file_mrk);
    hpiMrkData(mrkIdx, :, :) = cat(1, hdr_Mrk.orig.coregist.hpi.meg_pos);
end
hpiMrkData = squeeze(mean(hpiMrkData, 1));

% Headshape-HPI Alignment
% Align headshape with HPI coils
headshape.pos = [hpiData; fidData; hspData.pos];
headshape.unit = 'm';
num_remaining_points = size(hspData.pos, 1);
remaining_labels = cell(num_remaining_points, 1);
for i = 1:num_remaining_points
    remaining_labels{i} = sprintf('head_%d', i);
end
headshape.label = [{'LPA', 'RPA', 'NAS', 'HPI4', 'HPI5', 'nreal', ...
                   'lreal', 'rreal'}, remaining_labels']';

elec_dummy = headshape;
elec_dummy.elecpos = headshape.pos;
elec_dummy.chanpos = elec_dummy.elecpos;

hpi_coil.elecpos = hpiMrkData;
hpi_coil.label = {'LPA', 'RPA', 'NAS', 'HPI4', 'HPI5'}';
hpi_coil.unit = 'm';

cfg = [];
cfg.method = 'fiducial';
cfg.target = hpi_coil;
cfg.elec = elec_dummy;
cfg.feedback = 'no';
cfg.fiducial = {'NAS', 'LPA', 'RPA'}';
elec_aligned = ft_electroderealign(cfg);

hspCorrected = hspData;
hspCorrected.pos = elec_aligned.chanpos(9:end, :);
hspCorrected.fid.pos = elec_aligned.chanpos(6:8, :);
hspCorrected.fid.label = {'NAS', 'LPA', 'RPA'};
hspCorrected = ft_convert_units(hspCorrected, 'mm');

% Convert Headshape ALS -> RAS
% ALS: A=anterior, L=left, S=superior  
% RAS: R=right, A=anterior, S=superior
hspCorrected_ras = hspCorrected;
hspCorrected_ras.pos(:, 1) = -hspCorrected.pos(:, 2);
hspCorrected_ras.pos(:, 2) = hspCorrected.pos(:, 1);
hspCorrected_ras.pos(:, 3) = hspCorrected.pos(:, 3);

hspCorrected_ras.fid.pos(:, 1) = -hspCorrected.fid.pos(:, 2);
hspCorrected_ras.fid.pos(:, 2) = hspCorrected.fid.pos(:, 1);
hspCorrected_ras.fid.pos(:, 3) = hspCorrected.fid.pos(:, 3);

% Prepare Subject Fiducials (RAS)
% Reorder to match template: [NAS, RPA, LPA] = [subj(1), subj(3), subj(2)]
subject_fid.pos = [
    hspCorrected_ras.fid.pos(1, :);  % Nasion
    hspCorrected_ras.fid.pos(3, :);  % RPA
    hspCorrected_ras.fid.pos(2, :)   % LPA
];
subject_fid.label = {'NAS', 'RPA', 'LPA'};
subject_fid.unit = 'mm';

%% Fiducial-Based Alignment (Procrustes)
subj_points = subject_fid.pos;
templ_points = template_fid.pos;

[d, Z, transform] = procrustes(subj_points, templ_points, 'reflection', false);

% Extract transformation components
R = transform.T;          % Rotation matrix with scaling
t = transform.c(1, :)';   % Translation vector
s = transform.b;          % Scaling factor

% Create 4x4 transformation matrix
T = eye(4);
T(1:3, 1:3) = R;
T(1:3, 4) = t;

% Apply transformation to template MRI
mri_aligned = anatMRI;
mri_aligned.transform = s * T * anatMRI.transform;

% Calculate alignment quality
fid_errors = [];
for i = 1:size(subject_fid.pos, 1)
    error_dist = norm(Z(i, :) - subject_fid.pos(i, :));
    fid_errors(end+1) = error_dist;
end
fiducial_rms_error = sqrt(mean(fid_errors.^2));

%% Correct alignment manually
cfg                  = [];
cfg.method           = 'headshape';
cfg.spmversion       = 'spm12';
cfg.headshape.headshape ...
                     = hspCorrected_ras;
% cfg.headshape.coordsys = 'als';
cfg.headshape.icp    = 'yes';
cfg.headshape.interactive = 'yes';  
mri_aligned          = ft_volumerealign(cfg, mri_aligned);

cfg.headshape.interactive = 'yes';  
mri_aligned          = ft_volumerealign(cfg, mri_aligned);

%% Apply Transformation to
transformApplied = mri_aligned.transform * pinv(anatMRI.transform); %s * T + manual alignment

% 1: singleShellHeadModel
% Apply Procrustes transformation to head model vertices
singleShellHeadModel = singleShellHeadModel_template;
for i = 1:size(singleShellHeadModel_template.bnd.pos, 1)
    pos_homog = [singleShellHeadModel_template.bnd.pos(i, :)'; 1];
    pos_transformed = transformApplied * pos_homog;
    singleShellHeadModel.bnd.pos(i, :) = pos_transformed(1:3)';
end
if isfield(singleShellHeadModel.bnd, 'tri') && isfield(singleShellHeadModel.bnd, 'nrm')
    R_only = transformApplied(1:3, 1:3);
    R_normals = inv(R_only)'; 
    for i = 1:size(singleShellHeadModel_template.bnd.pos, 1)
        original_normal = singleShellHeadModel_template.bnd.nrm(i, :)';
        transformed_normal = R_normals * original_normal;
        singleShellHeadModel.bnd.nrm(i, :) = transformed_normal';
    end
end

% 2: Segmented MRI
segmentedmri_aligned = segmentedmri_template;
segmentedmri_aligned.transform = transformApplied * segmentedmri_template.transform;

% 3: Forward Model
sourcemodel_aligned = sourcemodel_template;
for i = 1:size(sourcemodel_template.pos, 1)
    pos_homog = [sourcemodel_template.pos(i, :)'; 1];
    pos_transformed = transformApplied * pos_homog;
    sourcemodel_aligned.pos(i, :) = pos_transformed(1:3)';
end

% %% Helper Function: Select Sources from Brain Regions
% selectSourcesFromRegion = @(sourcemodel, atlas, region_name) ...
%     selectSourcesFromAtlas(sourcemodel, atlas, region_name);

% 5. VTPM atlas - Convert coordinate system first
atlas_aligned = atlas_template;

% Convert atlas to RAS to match other components (headmodel, headshape, gradiometers)
atlas_aligned = ft_convert_coordsys(atlas_aligned, 'ras');

% Now apply the subject-specific transformation
atlas_aligned.transform = transformApplied * atlas_aligned.transform;

% Display available regions in a nice format
regions = atlas_aligned.tissuelabel;
left_regions = regions(contains(regions, 'left_'));
right_regions = regions(contains(regions, 'right_'));

%% Comprehensive 3D Atlas Visualization
figure('Name', 'Complete Source Reconstruction Setup', 'Position', [100, 100, 1600, 800]);

% Plot 1: Complete setup with headshape
subplot(1, 2, 1);
ft_plot_headshape(hspCorrected_ras, 'vertexcolor', 'k');
hold on;
ft_plot_mesh(singleShellHeadModel.bnd, 'facecolor', 'brain', 'facealpha', 0.3);
plot3(sourcemodel_aligned.pos(sourcemodel_aligned.inside, 1), ...
      sourcemodel_aligned.pos(sourcemodel_aligned.inside, 2), ...
      sourcemodel_aligned.pos(sourcemodel_aligned.inside, 3), 'r.', 'MarkerSize', 3);
ft_plot_sens(gradData, 'facecolor', 'green', 'facealpha', 0.7);
plot3(subject_fid.pos(:,1), subject_fid.pos(:,2), subject_fid.pos(:,3), 'bs', 'MarkerSize', 10, 'LineWidth', 2);
title('Complete Setup with Headshape');
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
axis equal; grid on;

% Plot 2: All VTPM Atlas ROIs in 3D with Forward Model
subplot(1, 2, 2);
hold on;

% Get all ROI names (excluding background)
all_rois = atlas_aligned.tissuelabel;
num_rois = length(all_rois);

% Generate distinct colors for all ROIs using hsv colormap
colors = hsv(num_rois);

fprintf('Visualizing %d VTPM ROIs in 3D with forward model...\n', num_rois);

% Create all atlas ROI patches first
roi_patch_count = 0;
for i = 1:num_rois
    roi_name = all_rois{i};
    roi_idx = i; % ROI indices start from 1
    
    % Create mask for this ROI in the aligned atlas
    roi_mask = atlas_aligned.tissue == roi_idx;
    
    % Check if ROI has any voxels
    if sum(roi_mask(:)) > 0
        % Create isosurface for the ROI
        [faces, vertices] = isosurface(roi_mask, 0.5);
        
        % Reorder: [Y,X,Z] -> [X,Y,Z]
        vertices_reordered = vertices(:, [2, 1, 3]);
        
        % Transform vertices from voxel space to world coordinates
        % Add homogeneous coordinate (4th column of ones)
        vertices_hom = [vertices_reordered, ones(size(vertices_reordered, 1), 1)];
        vertices_world = (atlas_aligned.transform * vertices_hom')';
        vertices_transformed = vertices_world(:, 1:3);
        
        p = patch('Faces', faces, 'Vertices', vertices_transformed);
        set(p, 'FaceColor', colors(i,:), 'EdgeColor', 'none', 'FaceAlpha', 0.6);
        roi_patch_count = roi_patch_count + 1;
        if mod(roi_patch_count, 10) == 0
            fprintf('  ✓ Created %d ROI patches...\n', roi_patch_count);
        end
    end
end

% Overlay the forward model components
% 1. Head model (transparent)
ft_plot_mesh(singleShellHeadModel.bnd, 'facecolor', 'brain', 'facealpha', 0.1, 'edgecolor', 'none');

% 2. Source model points (small red dots)
plot3(sourcemodel_aligned.pos(sourcemodel_aligned.inside, 1), ...
      sourcemodel_aligned.pos(sourcemodel_aligned.inside, 2), ...
      sourcemodel_aligned.pos(sourcemodel_aligned.inside, 3), 'r.', 'MarkerSize', 2);

% 3. MEG sensors (green)
ft_plot_sens(gradData, 'facecolor', 'green', 'facealpha', 0.8, 'chantype', 'megmag');

% Set up the plot
axis equal; grid on;
lighting gouraud; camlight;
title(sprintf('All %d VTPM Atlas ROIs + Forward Model', num_rois));
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');

% Add colorbar to show region mapping
colormap(colors);
c = colorbar;
c.Label.String = 'ROI Index';
c.Limits = [1 num_rois];
    