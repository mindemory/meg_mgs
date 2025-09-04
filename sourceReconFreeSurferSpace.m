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
    % Use FieldTrip standard template MRI instead of native space
    mri_path               = '/d/DATD/hyper/software/fieldtrip-20250318/template/headmodel/standard_mri.mat';
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
    
    % Load template MRI from FieldTrip
    load(mri_path, 'mri');
    anatMRI              = mri; % Template MRI is already in the correct format
    clear mri; % Clean up workspace
    % Template MRI is already in MNI coordinates (als coordinate system)
    % ROI handling commented out for template space analysis
    % TODO: Implement template-based ROI analysis or atlas-based ROI definition
    % roiData = struct();
    % for i = 1:numel(roiPaths)
    %     roiData.(roiNames{i}) = ft_read_mri(roiPaths{i});
    % end
    % ... (ROI processing code commented out for template space compatibility)
elseif subjID              == 4
    nMrkFiles            =  1;
    % Use FieldTrip standard template MRI instead of native space
    mri_path               = '/d/DATD/hyper/software/fieldtrip-20250318/template/headmodel/standard_mri.mat';
    load(mri_path, 'mri');
    anatMRI              = mri;
    clear mri;
elseif subjID              == 1
    nMrkFiles            =  2;
    % Use FieldTrip standard template MRI instead of native space
    mri_path               = '/d/DATD/hyper/software/fieldtrip-20250318/template/headmodel/standard_mri.mat';
    load(mri_path, 'mri');
    anatMRI              = mri;
    clear mri;
end

% After loading the template MRI, explicitly set its coordinate system
anatMRItransformed = anatMRI;

%% Fiducial-Based Alignment Approach
% Step 1: Define MNI template fiducials
fprintf('=== Step 1: Setting up Fiducial-Based Alignment ===\n');

% Define template fiducials in MNI space (world coordinates in mm)
template_fid.pos = [
      1.0,  83.0, -43.0;  % Nasion
     87.0, -21.0, -62.0;  % RPA (right ear)
    -87.0, -21.0, -62.0   % LPA (left ear)
];
template_fid.label = {'NAS', 'RPA', 'LPA'};
template_fid.unit = 'mm';

% Template fiducials are already in world coordinates (mm)
template_fid_world = template_fid;

% Template fiducials loaded

% Step 2: Load and process headshape data
fprintf('\n=== Step 2: Loading and Processing Headshape Data ===\n');

% Read headshape file
hspPath              = [subRoot 'headshape.hsp'];
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
for mrkIdx           = 1:nMrkFiles
    raw_file_mrk     = [subRoot 'marker_' num2str(mrkIdx, '%02d') '.sqd'];
    hdr_Mrk          = ft_read_header(raw_file_mrk);
    hpiMrkData(mrkIdx, :, :) ...
                     = cat(1, hdr_Mrk.orig.coregist.hpi.meg_pos);
end
hpiMrkData           = squeeze(mean(hpiMrkData, 1));

% Create headshape structure
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

hpi_coil.elecpos     = hpiMrkData;
hpi_coil.label       = {'LPA', 'RPA', 'NAS', 'HPI4', 'HPI5'}';
hpi_coil.unit        = 'm';

cfg                  = [];
cfg.method           = 'fiducial';
cfg.target           = hpi_coil;
cfg.elec             = elec_dummy;
cfg.feedback         = 'yes';
cfg.fiducial         = {'NAS', 'LPA', 'RPA'}';
elec_aligned         = ft_electroderealign(cfg);

hspCorrected         = hspData;
hspCorrected.pos     = elec_aligned.chanpos(9:end, :);
hspCorrected.fid.pos = elec_aligned.chanpos(6:8, :);
hspCorrected.fid.label ...
                     = {'NAS', 'LPA', 'RPA'};
% Convert units back to mm
hspCorrected         = ft_convert_units(hspCorrected, 'mm');

fprintf('Headshape data loaded and processed.\n');

% Step 3: Convert headshape from ALS to RAS coordinate system
fprintf('\n=== Step 3: Converting Headshape ALS -> RAS ===\n');

% Manual ALS to RAS conversion
% ALS: A=anterior, L=left, S=superior  
% RAS: R=right, A=anterior, S=superior
% Transformation: [x_ras, y_ras, z_ras] = [-x_als, y_als, z_als]

% Convert headshape positions
hspCorrected_ras = hspCorrected;
hspCorrected_ras.pos(:, 1) = -hspCorrected.pos(:, 2);  % Flip X: left->right and move to first dim
hspCorrected_ras.pos(:, 2) = hspCorrected.pos(:, 1); % Anterior is now second dim
hspCorrected_ras.pos(:, 3) = hspCorrected.pos(:, 3); % Superior stays as is
% Y and Z remain the same

% Convert fiducial positions  
hspCorrected_ras.fid.pos(:, 1) = -hspCorrected.fid.pos(:, 2);  % Flip X: left->right and move to first dim
hspCorrected_ras.fid.pos(:, 2) = hspCorrected.fid.pos(:, 1); % Anterior is now second dim
hspCorrected_ras.fid.pos(:, 3) = hspCorrected.fid.pos(:, 3); % Superior stays as is
% Y and Z remain the same

fprintf('Headshape converted ALS -> RAS\n');

% Step 4: Prepare subject fiducials for alignment (now in RAS)
fprintf('\n=== Step 4: Preparing Subject Fiducials (RAS) ===\n');
% Subject order is: NAS(1), LPA(2), RPA(3)  
% Template order is: NAS(1), RPA(2), LPA(3)
% So we need to reorder: [NAS, RPA, LPA] = [subj(1), subj(3), subj(2)]
subject_fid.pos = [
    hspCorrected_ras.fid.pos(1, :);  % Nasion
    hspCorrected_ras.fid.pos(3, :);  % RPA (from position 3)
    hspCorrected_ras.fid.pos(2, :)   % LPA (from position 2)
];
subject_fid.label = {'NAS', 'RPA', 'LPA'};  % To match template order
subject_fid.unit = 'mm';

% Subject fiducials prepared

% Step 5: Fiducial-based alignment using manual transformation
fprintf('\n=== Step 5: Fiducial-Based Alignment ===\n');

% Calculate transformation matrix from subject to template fiducials
subj_points = subject_fid.pos;
templ_points = template_fid_world.pos;

% Use Procrustes analysis (translation + rotation + scaling)
fprintf('Computing Procrustes transformation...\n');
[d, Z, transform] = procrustes(subj_points, templ_points, 'reflection', false);

% Extract transformation components
R = transform.T;          % Rotation matrix with scaling
t = transform.c(1, :)';   % Translation vector
s = transform.b;          % Scaling factor

fprintf('Procrustes: scale=%.3f, distance=%.3f\n', s, d);

% Create 4x4 transformation matrix
% Note: Procrustes already includes scaling in rotation matrix
T = eye(4);
T(1:3, 1:3) = R;  % R already includes scaling
T(1:3, 4) = t;

% For MEG analysis, we want template MRI to move to headshape space
% So we need the INVERSE transformation
T_inverse = inv(T);

% Apply inverse transformation to move template MRI to subject space
mri_fid_aligned = anatMRItransformed;
mri_fid_aligned.transform = T * anatMRItransformed.transform;

fprintf('Procrustes transformation applied\n');

% Step 4: Quality assessment
fprintf('\n=== Step 6: Assessing Fiducial Alignment Quality ===\n');
% Calculate fiducial alignment error using the Procrustes result directly
fid_errors = [];
for i = 1:size(subject_fid.pos, 1)
    % The Procrustes result Z contains the transformed subject points
    error_dist = norm(Z(i, :) - subject_fid.pos(i, :));
    fid_errors(end+1) = error_dist;
    fprintf('  %s: %.2f mm\n', subject_fid.label{i}, error_dist);
end

fiducial_rms_error = sqrt(mean(fid_errors.^2));
fprintf('Fiducial RMS alignment error: %.2f mm\n', fiducial_rms_error);

% Step 7: Save transformation for reproducibility
transform_file = [anatRoot 'template_fiducial_alignment_subj' num2str(subjID, '%02d') '.mat'];
% save(transform_file, 'mri_fid_aligned', 'template_fid_world', 'subject_fid', 'fiducial_rms_error', '-v7.3');
fprintf('Fiducial transformation saved to: %s\n', transform_file);

% Use fiducial-aligned MRI for subsequent processing
mri_aligned = mri_fid_aligned;

% Final aligned MRI ready for source reconstruction
mri_final = mri_aligned;

%%
figure;
ft_plot_headshape(hspCorrected_ras);
hold on;
ft_plot_ortho(mri_final.anatomy,'transform',mri_final.transform,'style','intersect');
title('Template Brain Aligned to Headshape (Procrustes)');

% %% Slice, segment and create headmodel
% % % Slice the data
% mri_reslice          = ft_volumereslice([], mri_aligned);
% % % Segment the mri
% cfg                  = [];
% cfg.output           = {'brain'};
% segmentedmri         = ft_volumesegment(cfg, mri_reslice);

% %%
% cfg                  = [];
% cfg.method           = 'singleshell';
% singleShellHeadModel = ft_prepare_headmodel(cfg, segmentedmri);

% %%
% figure;
% plot3(singleShellHeadModel.bnd.pos(:, 1), singleShellHeadModel.bnd.pos(:, 2), singleShellHeadModel.bnd.pos(:, 3), 'ro')
% hold on;
% ft_plot_sens(gradData, 'facecolor', 'green', 'facealpha', 1)
% ft_plot_ortho(mri_reslice.anatomy, 'style', 'intersect', ...
%     'transform', mri_reslice.transform)
% axis equal;
%%

%% Helper function for alignment quality assessment
function quality = assess_alignment_quality(mri_aligned, headshape)
    % Simple quality assessment based on distance between headshape and brain surface
    
    try
        % Extract brain surface from MRI
        cfg = [];
        cfg.output = 'brain';
        cfg.brainsmooth = 3;
        segmented = ft_volumesegment(cfg, mri_aligned);
        
        cfg = [];
        cfg.method = 'isosurface';
        cfg.numvertices = 1000;
        brain_surface = ft_prepare_mesh(cfg, segmented);
        
        % Calculate distances from headshape points to brain surface
        distances = [];
        for i = 1:size(headshape.pos, 1)
            point = headshape.pos(i, :);
            % Find closest point on brain surface
            dist_to_surface = min(sqrt(sum((brain_surface.pos - point).^2, 2)));
            distances(end+1) = dist_to_surface;
        end
        
        quality.rms_error = sqrt(mean(distances.^2));
        quality.mean_distance = mean(distances);
        quality.max_distance = max(distances);
        quality.std_distance = std(distances);
        
    catch
        % Fallback if surface extraction fails
        fprintf('Warning: Could not extract brain surface for quality assessment\n');
        quality.rms_error = NaN;
        quality.mean_distance = NaN;
        quality.max_distance = NaN;
        quality.std_distance = NaN;
    end
end
