clear; close all; clc;
%% Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/'); % 2022 doesn't work well for sourerecon
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);
% Load forward model for Clay
forwardmodelfPath                     = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/anatomy/sub-12_task-mgs_forwardmodel-nativespace-10mm.mat';
load(forwardmodelfPath);
%%
subList                               = [ 1,  2,  3,  4,  5,  6,  7,  9, 10, 12, 13, 15, ...
                                         17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
sourceAvgfPath                        = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/allsubs_clayspace/clayspaceProjections/sub-all_task-mgs_avgdiff-clayspace.mat';
if ~exist(sourceAvgfPath, 'file')
    % subList                               = [ 1 2 3 12];
    for sIdx                              = 1:length(subList)
        subjID                            = subList(sIdx);
        disp(['Running ' num2str(sIdx, '%02d') ' of ' num2str(length(subList))])
        subDerivativesRoot                = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-' num2str(subjID, '%02d') ...
            '/meg/sub-' num2str(subjID, '%02d') '_task-mgs_'];
        sourceSpacefPath                  = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/allsubs_clayspace/sub-' ...
            num2str(subjID, '%02d') '_task-mgs_source-clayspace.mat'];
        if sIdx                   == 1
            load(sourceSpacefPath);
        else
            load(sourceSpacefPath, 'sourcedata');
        end
    
        cfg                       = [];
        cfg.trials                = find(sourcedata.trialinfo(:,6)==1);
        sourcedataLeft            = ft_selectdata(cfg, sourcedata);
        cfg.trials                = find(sourcedata.trialinfo(:,6)==2);
        sourcedataRight           = ft_selectdata(cfg, sourcedata);
    
    
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
    
        if sIdx                   == 1
            sourcediffMeta        = sourceDiff;
        else
            sourcediffMeta.avg    = cat(3, sourcediffMeta.avg, sourceDiff.avg);
        end
        
    end
    save(sourceAvgfPath, "sourcediffMeta", "-v7.3")
else
    sourceSpacefPath                  = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/allsubs_clayspace/sub-01_task-mgs_source-clayspace.mat'];
    load(sourceSpacefPath, 'source');
    load(sourceAvgfPath);
end

%% Visualize source
% TOI                 = find(sourceDiff.time > 1.0 & sourceDiff.time < 1.5);
TOI                       = find(sourcediffMeta.time > 1 & sourcediffMeta.time < 1.5);

sourceVisualize           = source;
sourceVisualize.lateralizedPow ...
    = NaN(size(source.inside));
sourceVisualize.lateralizedPow(source.inside) ...
    = squeeze(mean(sourcediffMeta.avg(:, TOI, :), [2,3], 'omitnan'));

%%
cfg                       = [];
cfg.parameter             = {'lateralizedPow'};
[interp]                  = ft_sourceinterpolate(cfg, sourceVisualize, mri_reslice);

cfg                       = [];
cfg.method                = 'ortho';
cfg.crosshair             = 'yes';
cfg.funparameter          = 'lateralizedPow';
cfg.funcolormap           = '*RdBu';
cfg.funcolorlim           = [-0.01 0.01];
ft_sourceplot(cfg, interp);
%% Visualize on surface
surfToProject             = 'graymid';
surf_lh                   = ft_read_headshape(['/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/surf/lh.' surfToProject]);
surf_lh.coordsys          = 'ras';
surf_lh                   = ft_convert_coordsys(surf_lh, 'als');
surf_rh                   = ft_read_headshape(['/d/DATD/datd/MEG_MGS/MEG_BIDS/freesurferOutput/CCanat/surf/rh.' surfToProject]);
surf_rh.coordsys          = 'ras';
surf_rh                   = ft_convert_coordsys(surf_rh, 'als');

cfg                       = [];
cfg.parameter             = 'lateralizedPow';
cfg.interpmethod          = 'nearest'; % or 'linear'
cfg.downsample            = 1;
sourceOnSurf_lh           = ft_sourceinterpolate(cfg, sourceVisualize, surf_lh);
sourceOnSurf_rh           = ft_sourceinterpolate(cfg, sourceVisualize, surf_rh);

%%
figure;
cfg                   = [];
cfg.figure            = 'gcf';
cfg.method            = 'surface';
cfg.funparameter      = 'lateralizedPow';
cfg.funcolormap       = '*RdBu'; % or '*RdBu'
cfg.funcolorlim       = [-0.02 0.02];
cfg.projmethod        = 'project'; % or 'project'
cfg.surffile          = surf_lh;
cfg.colorbar          = 'no';
cfg.camlight          = 'yes'; % 'no', 'yes', or 'infinite'
subplot(1, 2, 1)
ft_sourceplot(cfg, sourceOnSurf_lh);
subplot(1, 2, 2)
ft_sourceplot(cfg, sourceOnSurf_rh);


%%
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

% combined_indices = intersect(posterior_indices, superior_indices);
combined_indices = find( ...
    sourceVisualize.pos(:, 1) < -55 & ...
    sourceVisualize.pos(:, 3) > 0 & ...
    (sourceVisualize.lateralizedPow > 0.01 | sourceVisualize.lateralizedPow < -0.01));

% Now use these indices to set the mask
sourceVisualizeWithMask.mask(combined_indices) = 1;
% sourceVisualizeWithMask.mask()
%%
close all;
% cfg                 = [];
% cfg.parameter       = {'lateralizedPow', 'mask'};
% [interp]            = ft_sourceinterpolate(cfg, sourceVisualizeWithMask, mri_reslice);
cfg                       = [];
cfg.parameter             = {'lateralizedPow', 'mask'};
cfg.interpmethod          = 'nearest'; % or 'linear'
cfg.downsample            = 1;
sourceWithMaskOnSurf_lh           = ft_sourceinterpolate(cfg, sourceVisualizeWithMask, surf_lh);
sourceWithMaskOnSurf_rh           = ft_sourceinterpolate(cfg, sourceVisualizeWithMask, surf_rh);

figure;
cfg                   = [];
cfg.figure            = 'gcf';
cfg.method            = 'surface';
cfg.funparameter      = 'lateralizedPow';
cfg.maskparameter     = 'mask';
cfg.funcolormap       = '*RdBu'; % or '*RdBu'
cfg.funcolorlim       = [-0.02 0.02];
cfg.projmethod        = 'project'; % or 'project'
cfg.surffile          = surf_lh;
cfg.colorbar          = 'no';
cfg.camlight          = 'yes'; % 'no', 'yes', or 'infinite'
subplot(1, 2, 1)
ft_sourceplot(cfg, sourceWithMaskOnSurf_lh);
subplot(1, 2, 2)
ft_sourceplot(cfg, sourceWithMaskOnSurf_rh);

% cfg = [];
% cfg.method        = 'ortho';
% cfg.crosshair = 'yes';
% cfg.funparameter  = 'lateralizedPow';
% cfg.funcolormap = '*RdBu';
% cfg.maskparameter = 'mask';
% cfg.funcolorlim = [-0.05 0.05];
% % sourceVisualize.mask = is_top_source;
% % cfg.funcolorlim   = [0 0.06];
% % cfg.location = [2 38 48];
% ft_sourceplot(cfg, interp);
%% Extract connectivity measure
subList                               = [ 1,  2,  3,  4,  5,  6,  7,  9, 10, 12, 13, 15, ...
                                         17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
sourceAvgfPath                        = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/allsubs_clayspace/clayspaceProjections/sub-all_task-mgs_avgdiff-clayspace.mat';
connectivityRoot                      = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/allsubs_clayspace/connectivityMeasures';
metricToUse                           = 'pli';
trlTypes                              = 'left';
if ~exist(sourceAvgfPath, 'file')
    % subList                               = [ 1 2 3 12];
    for sIdx                           = 1:length(subList)
        subjID                         = subList(sIdx);
        disp(['Running ' num2str(sIdx, '%02d') ' of ' num2str(length(subList))])
        metricfPath                    = [connectivityRoot filesep metricToUse '/sub-' num2str(subjID, '%02d') '_task-mgs_' metricToUse '.mat'];
        load(metricfPath);
    end
end
        
%% Visualize connectivity

sourceConnectivity           = source;
sourceConnectivity.con       = NaN(size(source.inside));
sourceVisualize.con(source.inside) ...
                          = squeeze(mean(sourceDiff.avg(:, TOI), 2, 'omitnan'));

combined_indices = find( ...
    sourceVisualize.pos(:, 1) < -55 & ...
    sourceVisualize.pos(:, 3) > 0 & ...
    (sourceVisualize.lateralizedPow > 0.01 | sourceVisualize.lateralizedPow < -0.01));