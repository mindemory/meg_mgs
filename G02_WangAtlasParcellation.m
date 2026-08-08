function G02_WangAtlasParcellation(volumetric_resolution)
%G02_WANGATLASPARCELLATION  Full per-voxel Wang atlas parcel labels for a
%   given volumetric grid resolution, extending the boolean-ROI-group-only
%   output of exploreWangAtlasVolumetric.m (which computes this same
%   per-grid-point label internally but discards it after building the
%   visual/parietal/frontal group masks).
%
%   Unlike exploreWangAtlasVolumetric.m, this function:
%     (1) is parametrized on volumetric_resolution instead of hardcoding
%         5mm, and
%     (2) persists a FULL-GRID-LENGTH parcel_id/parcel_name array (0 =
%         unlabeled/beyond the 20mm nearest-neighbor threshold), not just
%         boolean group masks -- so a later analysis can look up any
%         individual Wang ROI (e.g. left_V1v), not only the three curated
%         visual/parietal/frontal groups.
%
%   Labeling is performed once, on the TEMPLATE (unwarped) grid
%   (standard_sourcemodel3d{res}mm.mat), which is index-aligned across all
%   subjects -- so this produces one file per resolution, not per subject.
%
%   *** Indexing caveat (important for all consumers) ***
%   parcel_id is indexed by FULL TEMPLATE GRID INDEX (length =
%   numel(sourcemodel.inside), covering both inside and outside points),
%   NOT by position within any one subject's inside_pos list.
%   S02A_ReverseModelMNIVolumetric.m / G03_SourceLocalizationBroadband.m
%   compute inside_pos = find(source.inside) from each SUBJECT'S warped
%   grid + headmodel, which can differ slightly (in count and identity)
%   from the template's inside mask used here. To align a subject's
%   sourcedataCombined rows with parcel labels, consumers MUST do:
%       parcel_id_this_subject = parcel_id(inside_pos);
%   A subject grid point whose template counterpart was "outside" (and
%   therefore never atlas-labeled) will simply read back as 0/unlabeled --
%   this is an expected, documented limitation near the cortical surface,
%   not a bug.
%
%   Output: extends (does not replace) the existing
%   derivatives/atlas/rois_{res}mm.mat file with:
%     parcel_id     - [nGrid x 1] int array, 0 = unlabeled
%     parcel_name   - [nGrid x 1] cellstr, 'unlabeled' where parcel_id==0
%     grid_pos      - [nGrid x 3] template grid positions (mm), for reference
%     grid_inside   - [nGrid x 1] logical, template's own inside mask
%     tissuelabel   - Wang atlas's full tissuelabel list (parcel_id indexes into this)
%   plus the pre-existing boolean group masks (visual_points, parietal_points,
%   frontal_points, and their left_/right_ variants), now derived FROM
%   parcel_id/parcel_name so both representations stay consistent by
%   construction.

if nargin < 1 || isempty(volumetric_resolution)
    volumetric_resolution = 8;
end

restoredefaultpath;
clearvars -except volumetric_resolution; close all; clc;
fieldtrip_path  = '/d/DATD/hyper/software/fieldtrip-20250318/';
project_path    = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
data_base_path  = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';
ft_gifti_path   = '/d/DATD/hyper/software/fieldtrip-20250318/external/gifti';

addpath(fieldtrip_path);
ft_defaults;
addpath(ft_gifti_path);
addpath(genpath(project_path));

resTag = sprintf('%dmm', volumetric_resolution);
fprintf('Building Wang atlas parcellation for %s template grid\n', resTag);

%% Load template (unwarped) grid at the requested resolution
sourcemodel_path = sprintf('%stemplate/sourcemodel/standard_sourcemodel3d%s.mat', fieldtrip_path, resTag);
tmp = load(sourcemodel_path);
sourcemodel = ft_convert_units(tmp.sourcemodel, 'mm');

nGrid       = size(sourcemodel.pos, 1);
insideIdx   = find(sourcemodel.inside);
sm_pos_in   = sourcemodel.pos(insideIdx, :);

%% Load Wang atlas
atlas_path = [fieldtrip_path 'template/atlas/vtpm/vtpm.mat'];
wangatlas  = ft_read_atlas(atlas_path);

% Atlas-labeled voxels & their labels, converted to mm coordinates
[ind_x, ind_y, ind_z] = ind2sub(size(wangatlas.tissue), find(wangatlas.tissue > 0));
V                     = [ind_x, ind_y, ind_z, ones(numel(ind_x), 1)];
mm_coords             = (wangatlas.transform * V')';
mm_coords             = mm_coords(:, 1:3);
label_atlas           = double(wangatlas.tissue(wangatlas.tissue > 0));

%% Nearest-neighbor assign each inside template grid point to nearest atlas voxel
[idx_nearest, dist] = knnsearch(mm_coords, sm_pos_in);
label_for_inside     = label_atlas(idx_nearest);

% Distance threshold: reject points too far from any labeled ROI.
% Chosen originally against the 5mm grid in exploreWangAtlasVolumetric.m;
% re-validate at coarser resolutions before trusting it blindly (a fixed
% 20mm cutoff could over/under-include at 8mm/10mm spacing).
distThreshold_mm       = 20;
good                   = dist < distThreshold_mm;
label_for_inside(~good) = 0;   % 0 = unlabeled / outside cortex

%% Build full-grid-length parcel_id / parcel_name
parcel_id                = zeros(nGrid, 1);
parcel_id(insideIdx)     = label_for_inside;

parcel_name              = repmat({'unlabeled'}, nGrid, 1);
labeledMask               = parcel_id > 0;
parcel_name(labeledMask) = wangatlas.tissuelabel(parcel_id(labeledMask));

%% ROI group definitions (same curated lists as exploreWangAtlasVolumetric.m)
visualROIs = {'left_V1v', 'left_V1d', 'left_V2v', 'left_V2d', 'left_V3v', 'left_V3d', ...
              'left_hV4', 'left_VO1', 'left_VO2', 'left_V3b', 'left_V3a', ...
              'right_V1v', 'right_V1d', 'right_V2v', 'right_V2d', 'right_V3v', 'right_V3d', ...
              'right_hV4', 'right_VO1', 'right_VO2', 'right_V3b', 'right_V3a'};
leftVisualROIs  = {'left_V1v', 'left_V1d', 'left_V2v', 'left_V2d', 'left_V3v', 'left_V3d', ...
              'left_hV4', 'left_VO1', 'left_VO2', 'left_V3b', 'left_V3a'};
rightVisualROIs = {'right_V1v', 'right_V1d', 'right_V2v', 'right_V2d', 'right_V3v', 'right_V3d', ...
              'right_hV4', 'right_VO1', 'right_VO2', 'right_V3b', 'right_V3a'};
parietalROIs = {'left_IPS0', 'left_IPS1', 'left_IPS2', 'left_IPS3', 'left_IPS4', 'left_IPS5', 'left_SPL1', ...
                'right_IPS0', 'right_IPS1', 'right_IPS2', 'right_IPS3', 'right_IPS4', 'right_IPS5', 'right_SPL1'};
leftParietalROIs  = {'left_IPS0', 'left_IPS1', 'left_IPS2', 'left_IPS3', 'left_IPS4', 'left_IPS5', 'left_SPL1'};
rightParietalROIs = {'right_IPS0', 'right_IPS1', 'right_IPS2', 'right_IPS3', 'right_IPS4', 'right_IPS5', 'right_SPL1'};
frontalROIs      = {'left_FEF', 'right_FEF'};
leftFrontalROIs  = {'left_FEF'};
rightFrontalROIs = {'right_FEF'};

visual_idx        = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), visualROIs, 'UniformOutput', true);
parietal_idx      = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), parietalROIs, 'UniformOutput', true);
frontal_idx       = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), frontalROIs, 'UniformOutput', true);
left_visual_idx   = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), leftVisualROIs, 'UniformOutput', true);
right_visual_idx  = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), rightVisualROIs, 'UniformOutput', true);
left_parietal_idx  = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), leftParietalROIs, 'UniformOutput', true);
right_parietal_idx = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), rightParietalROIs, 'UniformOutput', true);
left_frontal_idx   = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), leftFrontalROIs, 'UniformOutput', true);
right_frontal_idx  = cellfun(@(lab) find(strcmp(wangatlas.tissuelabel, lab)), rightFrontalROIs, 'UniformOutput', true);

% Boolean masks now derived from parcel_id (full grid length), so they
% stay consistent with parcel_id/parcel_name by construction.
visual_points          = ismember(parcel_id, visual_idx);
parietal_points        = ismember(parcel_id, parietal_idx);
frontal_points         = ismember(parcel_id, frontal_idx);
left_visual_points     = ismember(parcel_id, left_visual_idx);
right_visual_points    = ismember(parcel_id, right_visual_idx);
left_parietal_points   = ismember(parcel_id, left_parietal_idx);
right_parietal_points  = ismember(parcel_id, right_parietal_idx);
left_frontal_points    = ismember(parcel_id, left_frontal_idx);
right_frontal_points   = ismember(parcel_id, right_frontal_idx);

%% Save (extend existing rois_{res}mm.mat if present)
atlas_dir  = [data_base_path filesep 'atlas'];
if ~exist(atlas_dir, 'dir')
    mkdir(atlas_dir);
end
out_fpath = [atlas_dir filesep 'rois_' resTag '.mat'];

tissuelabel = wangatlas.tissuelabel; %#ok<NASGU>
grid_pos    = sourcemodel.pos; %#ok<NASGU>
grid_inside = sourcemodel.inside; %#ok<NASGU>

fprintf('Saving %s\n', out_fpath);
save(out_fpath, 'parcel_id', 'parcel_name', 'tissuelabel', 'grid_pos', 'grid_inside', ...
    'visual_points', 'parietal_points', 'frontal_points', ...
    'left_visual_points', 'right_visual_points', ...
    'left_parietal_points', 'right_parietal_points', ...
    'left_frontal_points', 'right_frontal_points', '-v7.3');

end
