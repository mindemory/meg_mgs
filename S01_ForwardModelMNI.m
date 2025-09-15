%% MEG Source Reconstruction in Template Space (Clean Version)
% Aligns template MRI to subject headshape using fiducial-based Procrustes analysis
% Enables MEG source reconstruction in standardized template coordinates
restoredefaultpath;
clear; close all; clc;

%% Setup and Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
addpath('/d/DATD/hyper/software/fieldtrip-20250318/external/gifti'); % Add Gifti toolbox for .surf.gii files
ft_defaults;
ft_hastoolbox('spm12', 1);

% Configuration parameters
subjID = 25;
surface_resolutions = [5124, 8196, 20484]; % All available cortical surface resolutions

%% Subject-Specific Configuration (Based on Preprocessing Logger)
% Default: 2 marker files, normal headshape (most common case)
nMrkFiles = 2;
headshape_condition = 'normal';

% Override for subjects with special requirements
if subjID == 4 || subjID == 15
    nMrkFiles = 1; % Use only 1st marker file
elseif subjID == 12
    nMrkFiles = 3; % Use 3 marker files
elseif subjID == 32
    nMrkFiles = 4; % Use 4 marker files
elseif subjID == 2
    headshape_condition = 'remove_x_neg150'; % Remove points with X < -150mm
elseif subjID == 29
    headshape_condition = 'points_1_149_factor_1000'; % Special sub-29 processing
end

fprintf('Subject %02d: %d marker files, %s headshape\n', subjID, nMrkFiles, headshape_condition);

% Output directory and file (BIDS structure)
output_dir = sprintf('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-%02d/sourceRecon', subjID);
output_file = fullfile(output_dir, sprintf('sub-%02d_task-mgs_forwardModel.mat', subjID));

% Check if forward model already exists
COMPUTE_FORWARD_MODEL = ~exist(output_file, 'file');

if ~COMPUTE_FORWARD_MODEL
    fprintf('Loading existing forward model from: %s\n', output_file);
    load(output_file);

    % Display loaded components
    fprintf('\n=== LOADED FORWARD MODEL ===\n');
    fprintf('Subject ID: %d\n', subjID);
    fprintf('MRI aligned: %s (coordsys: %s)\n', mat2str(size(mri_aligned.anatomy)), mri_aligned.coordsys);
    fprintf('Head model vertices: %d\n', size(singleShellHeadModel.bnd.pos, 1));
    fprintf('Surface models loaded:\n');
    for i = 1:length(surface_resolutions)
        res = surface_resolutions(i);
        varname = sprintf('sourcemodel_aligned_%d', res);
        if exist(varname, 'var')
            eval(sprintf('sm = %s;', varname));
            fprintf('  - %d vertices: %d sources\n', res, size(sm.pos, 1));
        end
    end

    fprintf('\nSkipping computation, jumping to visualization...\n');

else
    fprintf('Forward model not found. Computing from scratch...\n');
end

%% Computation Section (only runs if COMPUTE_FORWARD_MODEL is true)
if COMPUTE_FORWARD_MODEL

    %% Data Paths and Loading
    subRoot = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-' num2str(subjID, '%02d') ...
        '/meg/sub-' num2str(subjID, '%02d') '_task-mgs_'];
    subDerivativesRoot = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-' num2str(subjID, '%02d') ...
        '/meg/sub-' num2str(subjID, '%02d') '_task-mgs_'];

    % Load template MRI (nMrkFiles now set above based on preprocessing log)

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

    % Load source model (surface or volume)
    % Load all cortical surface models
    sourcemodel_templates = struct();
    for i = 1:length(surface_resolutions)
        res = surface_resolutions(i);
        sourcemodel_path = sprintf('/d/DATD/hyper/software/fieldtrip-20250318/template/sourcemodel/cortex_%d.surf.gii', res);

        % Load and format surface model
        sm = ft_read_headshape(sourcemodel_path);
        sm.pos = sm.pos;
        sm.inside = true(size(sm.pos, 1), 1); % All cortical vertices are inside
        sm.unit = 'mm';
        sm.coordsys = 'mni';

        % Store in structure
        sourcemodel_templates.(sprintf('res_%d', res)) = sm;

        fprintf('Loaded cortical surface model %d: %d vertices\n', res, size(sm.pos, 1));
    end


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
    
    if subjID == 5 || subjID == 23
        load([subDerivativesRoot 'run-02_raw.mat'])
    else
        load([subDerivativesRoot 'run-01_raw.mat'])
    end
    gradData = data.grad;
    gradData.coordsys = 'als';
    gradData = ft_convert_coordsys(gradData, 'ras');
    gradData = ft_convert_units(gradData, 'mm');
    clearvars data;

    % Process HPI markers
    hpiMrkData = NaN(nMrkFiles, size(hpiData, 1), size(hpiData, 2));
    for mrkIdx = 1:nMrkFiles
        raw_file_mrk = [subRoot 'marker-' num2str(mrkIdx, '%02d') '.sqd'];
        hdr_Mrk = ft_read_header(raw_file_mrk);
        hpiMrkData(mrkIdx, :, :) = cat(1, hdr_Mrk.orig.coregist.hpi.meg_pos);
    end
    hpiMrkData = squeeze(mean(hpiMrkData, 1));

    % Headshape-HPI Alignment
    % Apply subject-specific headshape preprocessing
    if strcmp(headshape_condition, 'points_1_149_factor_1000')
        % Subject 29: Use only points 1-149 with factor 1000
        hspData.pos = hspData.pos(1:149, :) * 1000;
        fprintf('Applied sub-29 headshape preprocessing: points 1-149 with factor 1000\n');
     elseif strcmp(headshape_condition, 'remove_x_neg150')
         % Subject 2: Remove points with X < -150mm and Z < -100mm
         valid_idx = hspData.pos(:,1) >= -0.15 & hspData.pos(:,3) >= -0.1;
         removed_count = sum(~valid_idx);
         hspData.pos = hspData.pos(valid_idx, :);
         fprintf('Applied sub-02 headshape preprocessing: removed %d points (X<-150 or Z<-100)\n', removed_count);
    end

    % Standard headshape processing
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
    cfg.feedback = 'yes';
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
    cfg.headshape.icp    = 'yes';
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

    % 3: Transform All Surface Models
    sourcemodel_aligned_5124 = sourcemodel_templates.res_5124;
    sourcemodel_aligned_8196 = sourcemodel_templates.res_8196;
    sourcemodel_aligned_20484 = sourcemodel_templates.res_20484;

    % Transform each surface model
    for res_idx = 1:length(surface_resolutions)
        res = surface_resolutions(res_idx);
        template_sm = sourcemodel_templates.(sprintf('res_%d', res));

        % Transform vertex positions
        aligned_sm = template_sm;
        for i = 1:size(template_sm.pos, 1)
            pos_homog = [template_sm.pos(i, :)'; 1];
            pos_transformed = transformApplied * pos_homog;
            aligned_sm.pos(i, :) = pos_transformed(1:3)';
        end

        % Store transformed model
        eval(sprintf('sourcemodel_aligned_%d = aligned_sm;', res));

        fprintf('Transformed surface model %d: %d vertices\n', res, size(aligned_sm.pos, 1));
    end

    %% Save Forward Model
    % Create output directory if it doesn't exist
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
        fprintf('Created output directory: %s\n', output_dir);
    end

    % Save all forward model components
    fprintf('\nSaving forward model to: %s\n', output_file);
    save(output_file, 'subjID', 'mri_aligned', 'transformApplied', 'segmentedmri_aligned', ...
        'singleShellHeadModel', 'sourcemodel_aligned_5124', 'sourcemodel_aligned_8196', ...
        'sourcemodel_aligned_20484', 'surface_resolutions', 'gradData', 'subject_fid', ...
        'hspCorrected_ras', 'fiducial_rms_error', '-v7.3');

    fprintf('Forward model saved successfully!\n');
    fprintf('Contents: MRI, head model, 3 surface models (%s vertices)\n', mat2str(surface_resolutions));

end % End of COMPUTE_FORWARD_MODEL block

%% Visualization (runs for both loaded and computed models)
figure('Name', 'MEG Forward Model Setup', 'Position', [100, 100, 1600, 800]);

subplot(1, 2, 1);
% Plot 1: Complete setup with headshape
ft_plot_headshape(hspCorrected_ras, 'vertexcolor', 'k');
hold on;
ft_plot_mesh(singleShellHeadModel.bnd, 'facecolor', 'brain', 'facealpha', 0.3);

% Plot source model differently for surface vs volume
% Plot as surface mesh (using 8196 resolution for visualization)
ft_plot_mesh(sourcemodel_aligned_8196, 'facecolor', 'cortex', 'facealpha', 0.8, 'edgecolor', 'none');
title_str = sprintf('Setup with Cortical Surface (8196 vertices)');


ft_plot_sens(gradData, 'facecolor', 'green', 'facealpha', 0.7);
plot3(subject_fid.pos(:,1), subject_fid.pos(:,2), subject_fid.pos(:,3), 'bs', 'MarkerSize', 10, 'LineWidth', 2);
title(title_str);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
axis equal; grid on;

subplot(1, 2, 2);
% Plot 2: Surface model detail (8196 vertices)
plot3(sourcemodel_aligned_8196.pos(:, 1), sourcemodel_aligned_8196.pos(:, 2), sourcemodel_aligned_8196.pos(:, 3), ...
    'r.', 'MarkerSize', 2);
title('Cortical Surface Detail (8196 vertices)');

xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
axis equal; grid on;


