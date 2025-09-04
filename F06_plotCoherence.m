clear; close all; clc;
%%
freqband                    = 'beta';
trlLocs                     = 'left';
seedLocs                    = 'right';
connectivityMetric          = 'coherence'; % Valid: coherence (coherency is computed for free),
                                           %        plv


load('NYUKIT_helmet.mat')
right_sensors               = {'AG001', 'AG007', 'AG008', 'AG020', 'AG022', 'AG024', ...
                               'AG034', 'AG036', 'AG050', 'AG055', 'AG065', 'AG098', ...
                               'AG103'};
left_sensors                = {'AG014', 'AG015', 'AG016', 'AG025', 'AG026', 'AG027', ...
                               'AG028', 'AG041', 'AG042', 'AG043', 'AG059', 'AG060', ...
                               'AG092'};
subList                   = [ 1,  2,  3,  4,  5,  6,  7,  9, 10, 12, 13, 15, ...
                             17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
for sIdx                    = 1:length(subList)

    disp(['Running ' num2str(sIdx) ' of ' num2str(length(subList)) ' subjects.'])
    subjID                  = subList(sIdx);

    % Initalizing Paths
    bidsRoot                = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
    taskName                = 'mgs';
    derivativesRoot         = [bidsRoot filesep 'derivatives/sub-' num2str(subjID, '%02d') '/meg'];
    subName                 = ['sub-' num2str(subjID, '%02d')];
    megRoot                 = [bidsRoot filesep subName filesep 'meg'];
    stimRoot                = [bidsRoot filesep subName filesep 'stimfiles'];
    fNameRoot               = [subName '_task-' taskName];
    stimLocked_fpath        = [derivativesRoot filesep fNameRoot '_stimlocked_lineremoved.mat'];
    TFR_fpath               = [derivativesRoot filesep fNameRoot '_TFRfull_evoked_lineremoved.mat'];
    connectivityRoot        = [derivativesRoot 'connectivity'];
    % if ~exist(connectivityRoot, 'dir')
    %     mkdir(connectivityRoot)
    % end
    connectivity_fPath      = [connectivityRoot filesep fNameRoot '_coh-' freqband '-' seedLocs 'sensors-' trlLocs '-trials.mat'];
    load(connectivity_fPath);

    if sIdx                 == 1
        connectivityMeta1   = connectivity;
    elseif sIdx             == 13
        connectivityMeta2   = connectivity;
    elseif sIdx             < 13
        connectivityMeta1.cohspctrm ...
                            = cat(5, connectivityMeta1.cohspctrm, connectivity.cohspctrm);
    else
        connectivityMeta2.cohspctrm ...
                            = cat(5, connectivityMeta2.cohspctrm, connectivity.cohspctrm);
    end
end
%% Average across subjects
% Coherence
connectivityCoh1            = connectivityMeta1;
connectivityCoh1.cohspctrm  = squeeze(mean(abs(connectivityMeta1.cohspctrm), 5, 'omitnan'));
connectivityCoh2            = connectivityMeta2;
connectivityCoh2.cohspctrm  = squeeze(mean(abs(connectivityMeta2.cohspctrm), 5, 'omitnan'));
connectivityCoh             = connectivityCoh1;
connectivityCoh.cohspctrm   = squeeze(mean(cat(5, connectivityCoh1.cohspctrm, connectivityCoh2.cohspctrm), 5, 'omitnan'));

%% Keeping track of ipsi and contra
connectivityCoh_rightTrls_leftSeeds = connectivityCoh;
connectivityCoh_leftTrls_leftSeeds = connectivityCoh;

connectivityCoh_rightTrls_rightSeeds = connectivityCoh;
connectivityCoh_leftTrls_rightSeeds = connectivityCoh;

save('omgg.mat', 'connectivityCoh_rightTrls_leftSeeds', 'connectivityCoh_leftTrls_leftSeeds', ...
                 'connectivityCoh_rightTrls_rightSeeds', 'connectivityCoh_leftTrls_rightSeeds', '-v7.3');
clearvars connectivityMeta1 connectivityMeta2 connectivityCoh1 connectivityCoh2 connectivityCoh;
%% Perform baseline correction
cfg                                             = [];
cfg.baseline                                    = [-0.5 0];
cfg.parameter                                   = 'cohspctrm';
cfg.baselinetype                                = 'db'; 
connectivityCoh_rightTrls_leftSeeds_corr        = ft_freqbaseline(cfg, connectivityCoh_rightTrls_leftSeeds);
connectivityCoh_leftTrls_leftSeeds_corr         = ft_freqbaseline(cfg, connectivityCoh_leftTrls_leftSeeds);
connectivityCoh_rightTrls_rightSeeds_corr       = ft_freqbaseline(cfg, connectivityCoh_rightTrls_rightSeeds);
connectivityCoh_leftTrls_rightSeeds_corr        = ft_freqbaseline(cfg, connectivityCoh_leftTrls_rightSeeds);
connectivityCoh_rightTrls_leftSeeds_corr        = connectivityCoh_rightTrls_leftSeeds;
connectivityCoh_leftTrls_leftSeeds_corr         = connectivityCoh_leftTrls_leftSeeds;
connectivityCoh_rightTrls_rightSeeds_corr       = connectivityCoh_rightTrls_rightSeeds;
connectivityCoh_leftTrls_rightSeeds_corr        = connectivityCoh_leftTrls_rightSeeds;
%% Combine into contra and ipsi (TO DO: FLip sensors
[left_sensors_extracted, right_sensors_extracted] = find_mirror_sensors(lay);
right_new = find_opposite_sensors(lay, left_sensors_extracted(1:end-1), 'left');
left_new = find_opposite_sensors(lay, right_sensors_extracted(1:end-1), 'right');
left_idx = find(ismember(lay.label, left_sensors_extracted(1:end-1)));
right_idx = find(ismember(lay.label, right_sensors_extracted(1:end-1)));
left_new_idx = zeros(length(left_new), 1);
right_new_idx = zeros(length(right_new), 1);
for ii = 1:length(left_new_idx)
    left_new_idx(ii) = find(ismember(lay.label, left_new{ii}));
end
for ii = 1:length(right_new_idx)
    right_new_idx(ii) = find(ismember(lay.label, right_new{ii}));
end
% left_new_idx = find(ismember(lay.label, left_new));
% right_new_idx = find(ismember(lay.label, right_new));
% cfg = [];
% cfg.layout = lay;
% % cfg.chanidx = left_idx;
% ft_plot_layout(lay);
% connectivityCoh_contra = connectivityCoh_rightTrls_leftSeeds;
% connectivityCoh_contra.cohspctrm = cat(5, connectivityCoh_rightTrls_leftSeeds.cohspctrm, ...
%                                           connectivityCoh_leftTrls_rightSeeds.cohspctrm);
% connectivityCoh_contra.cohspctrm = squeeze(mean(connectivityCoh_contra.cohspctrm, 5, 'omitnan'));
% 
% connectivityCoh_ipsi = connectivityCoh_rightTrls_leftSeeds;
% connectivityCoh_contra.cohspctrm = cat(5, connectivityCoh_rightTrls_leftSeeds.cohspctrm, ...
%                                           connectivityCoh_leftTrls_rightSeeds.cohspctrm);
% connectivityCoh_contra.cohspctrm = squeeze(mean(connectivityCoh_contra.cohspctrm, 5, 'omitnan'));

%% FLIP the right seeded cohspctrm
connectivityCoh_rightTrls_rightSeeds_corr_flipped = connectivityCoh_rightTrls_rightSeeds_corr;
cohTempOrig = connectivityCoh_rightTrls_rightSeeds_corr_flipped.cohspctrm;
cohTempDuplicated = cohTempOrig;
cohTempDuplicated(left_new_idx, left_new_idx, :, :) = cohTempOrig(right_idx, right_idx, :, :);
cohTempDuplicated(right_new_idx, right_new_idx, :, :) = cohTempOrig(left_idx, left_idx, :, :);
cohTempDuplicated(left_new_idx, right_new_idx, :, :) = cohTempOrig(right_idx, left_idx, :, :);
cohTempDuplicated(right_new_idx, left_new_idx, :, :) = cohTempOrig(left_idx, right_idx, :, :);
connectivityCoh_rightTrls_rightSeeds_corr_flipped.cohspctrm = cohTempDuplicated;

connectivityCoh_leftTrls_rightSeeds_corr_flipped = connectivityCoh_leftTrls_rightSeeds_corr;
cohTempOrig = connectivityCoh_leftTrls_rightSeeds_corr_flipped.cohspctrm;
cohTempDuplicated = cohTempOrig;
cohTempDuplicated(left_new_idx, left_new_idx, :, :) = cohTempOrig(right_idx, right_idx, :, :);
cohTempDuplicated(right_new_idx, right_new_idx, :, :) = cohTempOrig(left_idx, left_idx, :, :);
cohTempDuplicated(left_new_idx, right_new_idx, :, :) = cohTempOrig(right_idx, left_idx, :, :);
cohTempDuplicated(right_new_idx, left_new_idx, :, :) = cohTempOrig(left_idx, right_idx, :, :);
connectivityCoh_leftTrls_rightSeeds_corr_flipped.cohspctrm = cohTempDuplicated;


%


% cohTemp                                           = connectivityCoh_rightTrls_rightSeeds_corr_flipped.cohspctrm;
% % Make a copy to modify
% cohTemp_flipped = cohTemp;
% 
% % Swap left_idx with right_new_idx
% for ii = 1:length(left_idx)
%     % Swap rows
%     temp_row = cohTemp_flipped(left_idx(ii), :, :, :);
%     cohTemp_flipped(left_idx(ii), :, :, :) = cohTemp(right_new_idx(ii), :, :, :);
%     cohTemp_flipped(right_new_idx(ii), :, :, :) = temp_row;
% 
%     % Swap columns
%     temp_col = cohTemp_flipped(:, left_idx(ii), :, :);
%     cohTemp_flipped(:, left_idx(ii), :, :) = cohTemp(:, right_new_idx(ii), :, :);
%     cohTemp_flipped(:, right_new_idx(ii), :, :) = temp_col;
% end
% 
% % Now, do the same for right_idx and left_new_idx
% for ii = 1:length(right_idx)
%     % Swap rows
%     temp_row = cohTemp_flipped(right_idx(ii), :, :, :);
%     cohTemp_flipped(right_idx(ii), :, :, :) = cohTemp(left_new_idx(ii), :, :, :);
%     cohTemp_flipped(left_new_idx(ii), :, :, :) = temp_row;
% 
%     % Swap columns
%     temp_col = cohTemp_flipped(:, right_idx(ii), :, :);
%     cohTemp_flipped(:, right_idx(ii), :, :) = cohTemp(:, left_new_idx(ii), :, :);
%     cohTemp_flipped(:, left_new_idx(ii), :, :) = temp_col;
% end
% 
% % Update the original data
% connectivityCoh_rightTrls_rightSeeds_corr_flipped.cohspctrm = cohTemp_flipped;

% cohTemp(left_new_idx)
% connectivityCoh             = connectivityMeta;
% connectivityCoh.cohspctrm   = squeeze(mean(abs(connectivityMeta.cohspctrm), 5, 'omitnan'));

% if strcmp(seedLocs, 'left')
%     seed_sensors            = left_sensors;
% elseif strcmp(seedLocs, 'right')
%     seed_sensors            = right_sensors;
% end
% tOnsets                     = [-0.5 0   0.5 1   ];
% tOffsets                    = [ 0   0.3 1   1.5 ];
%% Plot coherence
ff = figure('Renderer','painters');
% sgtitle(['Coh: ' freqband ' ' trlLocs ' trials'])
for i                       = 1:4
    if i                    == 1
        connectivityCoh_corr = connectivityCoh_rightTrls_leftSeeds_corr;
        seed_sensors        = left_sensors;
    elseif i                == 2
        connectivityCoh_corr = connectivityCoh_leftTrls_leftSeeds_corr;
        seed_sensors        = left_sensors;
    elseif i                == 3
        connectivityCoh_corr = connectivityCoh_rightTrls_rightSeeds_corr_flipped; %connectivityCoh_rightTrls_rightSeeds_corr;
        seed_sensors        = left_sensors;
    elseif i                == 4
        connectivityCoh_corr = connectivityCoh_leftTrls_rightSeeds_corr_flipped; %connectivityCoh_leftTrls_rightSeeds_corr;
        seed_sensors        = left_sensors;
    end
    subplot(2, 2, i)
    cfg                     = [];
    cfg.latency             = [-0.5 0];
    thisConnectivity        = ft_selectdata(cfg, connectivityCoh_corr);
    
    cfg                     = [];
    cfg.figure              = 'gcf';
    cfg.parameter           = 'cohspctrm';
    cfg.refchannel          = seed_sensors;
    cfg.layout              = lay; 
    cfg.interactive         = 'yes';
    cfg.colorbar            = 'yes';
    cfg.colormap            = '*RdBu';
    cfg.comment             = 'no';
    cfg.marker              = 'off';
    cfg.zlim                = [0.1 0.2]; %[-0.25 0.25];
    if i                    == 1
        cfg.title           = 'Right Trials, Left Seeds';
    elseif i                == 2
        cfg.title           = 'Left Trials, Left Seeds';
    elseif i                == 3
        cfg.title           = 'Right Trials, Right Seeds';
    elseif i                == 4
        cfg.title           = 'Left Trials, Right Seeds';
    end

    % cfg.zlim                = [-0.02 0.02];
    ft_topoplotER(cfg, thisConnectivity);
end
% fPatheps = ['/d/DATD/datd/MEG_MGS/IllustratorFigs/Coherence/Coh_' freqband '_' trlLocs '-trials_' seedLocs '-seeds.svg'];
% saveas(ff, fPatheps)
% fPathpng = ['/d/DATD/datd/MEG_MGS/IllustratorFigs/Coherence/Coh_' freqband '_' trlLocs '-trials_' seedLocs '-seeds.png'];
% saveas(ff, fPathpng)
%%
connectivityCoh_leftseeds_contrast = connectivityCoh_rightTrls_leftSeeds_corr;
connectivityCoh_leftseeds_contrast.cohspctrm = (connectivityCoh_rightTrls_leftSeeds_corr.cohspctrm - connectivityCoh_leftTrls_leftSeeds_corr.cohspctrm) ./ ...
                                                (connectivityCoh_rightTrls_leftSeeds_corr.cohspctrm + connectivityCoh_leftTrls_leftSeeds_corr.cohspctrm);
connectivityCoh_rightseeds_contrast = connectivityCoh_rightTrls_rightSeeds_corr_flipped;
connectivityCoh_rightseeds_contrast.cohspctrm = (connectivityCoh_leftTrls_rightSeeds_corr_flipped.cohspctrm - connectivityCoh_rightTrls_rightSeeds_corr_flipped.cohspctrm) ./ ...
                                                (connectivityCoh_leftTrls_rightSeeds_corr_flipped.cohspctrm + connectivityCoh_rightTrls_rightSeeds_corr_flipped.cohspctrm);
% connectivityCoh_rightseeds_contrast.cohspctrm = (connectivityCoh_rightTrls_rightSeeds_corr_flipped.cohspctrm - connectivityCoh_leftTrls_rightSeeds_corr_flipped.cohspctrm) ./ ...
%                                                 (connectivityCoh_rightTrls_rightSeeds_corr_flipped.cohspctrm + connectivityCoh_leftTrls_rightSeeds_corr_flipped.cohspctrm);
conectivityCoh_contrast = connectivityCoh_rightseeds_contrast;
conectivityCoh_contrast.cohspctrm = squeeze(mean(cat(5, connectivityCoh_leftseeds_contrast.cohspctrm, connectivityCoh_rightseeds_contrast.cohspctrm), 5, 'omitnan'));

%%
ff = figure('Renderer','painters');
% sgtitle(['Coh: ' freqband ' ' trlLocs ' trials'])
for i                       = 1:4
    if i                    == 1
        connectivityCoh_corr = connectivityCoh_leftseeds_contrast;
        seed_sensors        = left_sensors;
    elseif i                == 2
        connectivityCoh_corr = connectivityCoh_rightseeds_contrast;
        seed_sensors        = left_sensors;
    elseif i                == 3
        connectivityCoh_corr = conectivityCoh_contrast; %connectivityCoh_rightTrls_rightSeeds_corr;
        seed_sensors        = left_sensors;
    elseif i                == 4
        connectivityCoh_corr = connectivityCoh_leftTrls_rightSeeds_corr_flipped; %connectivityCoh_leftTrls_rightSeeds_corr;
        seed_sensors        = left_sensors;
    end
    subplot(2, 2, i)
    cfg                     = [];
    cfg.latency             = [0.5 1];
    thisConnectivity        = ft_selectdata(cfg, connectivityCoh_corr);
    
    cfg                     = [];
    cfg.figure              = 'gcf';
    cfg.parameter           = 'cohspctrm';
    cfg.refchannel          = seed_sensors;
    cfg.layout              = lay; 
    cfg.interactive         = 'yes';
    cfg.colorbar            = 'yes';
    cfg.colormap            = '*RdBu';
    cfg.comment             = 'no';
    cfg.marker              = 'off';
    % cfg.zlim                = [0.1 0.2]; %[-0.25 0.25];
    cfg.zlim                = [-0.05 0.05];
    if i                    == 1
        cfg.title           = 'Left Seeds Contrast';
    elseif i                == 2
        cfg.title           = 'Right Seeds, Contrast';
    elseif i                == 3
        cfg.title           = 'Combiend Contrast';
    elseif i                == 4
        cfg.title           = 'Ignore Please';
    end

    % cfg.zlim                = [-0.02 0.02];
    ft_topoplotER(cfg, thisConnectivity);
end
