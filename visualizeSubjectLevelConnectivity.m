clear; close all; clc;
%%
freqband                    = 'beta';
trlLocs                     = 'right';
seedLocs                    = 'left';
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

% Imaginary part of Coherence
connectivityImCoh1          = connectivityMeta1;
connectivityImCoh1.cohspctrm= squeeze(mean(imag(connectivityMeta1.cohspctrm), 5, 'omitnan'));
connectivityImCoh2          = connectivityMeta2;
connectivityImCoh2.cohspctrm= squeeze(mean(imag(connectivityMeta2.cohspctrm), 5, 'omitnan'));
connectivityImCoh           = connectivityCoh1;
connectivityImCoh.cohspctrm = squeeze(mean(cat(5, connectivityImCoh1.cohspctrm, connectivityImCoh2.cohspctrm), 5, 'omitnan'));
%%
cfg                         = [];
cfg.baseline                = [-0.5 0];
cfg.parameter               = 'cohspctrm';
cfg.baselinetype            = 'db'; 
connectivityCoh_corr        = ft_freqbaseline(cfg, connectivityCoh);
cfg.baselinetype            = 'absolute'; 
connectivityImCoh_corr      = ft_freqbaseline(cfg, connectivityImCoh);

%%
% connectivityCoh             = connectivityMeta;
% connectivityCoh.cohspctrm   = squeeze(mean(abs(connectivityMeta.cohspctrm), 5, 'omitnan'));

if strcmp(seedLocs, 'left')
    seed_sensors            = left_sensors;
elseif strcmp(seedLocs, 'right')
    seed_sensors            = right_sensors;
end
tOnsets                     = [-0.5 0   0.5 1   ];
tOffsets                    = [ 0   0.3 1   1.5 ];
%% Plot coherence
ff = figure('Renderer','painters');
sgtitle(['Coh: ' freqband ' ' trlLocs ' trials'])
for i                       = 1:length(tOnsets)
    subplot(2, 2, i)
    cfg                     = [];
    cfg.latency             = [tOnsets(i) tOffsets(i)];
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
    cfg.zlim                = [-0.25 0.25];
    if i                    == 1
        cfg.title           = 'Fixation';
    elseif i                == 2
        cfg.title           = 'Stim On';
    elseif i                == 3
        cfg.title           = 'Early Delay';
    elseif i                == 4
        cfg.title           = 'Late Delay';
    end

    % cfg.zlim                = [-0.02 0.02];
    ft_topoplotER(cfg, thisConnectivity);
end
fPatheps = ['/d/DATD/datd/MEG_MGS/IllustratorFigs/Coherence/Coh_' freqband '_' trlLocs '-trials_' seedLocs '-seeds.svg'];
saveas(ff, fPatheps)
fPathpng = ['/d/DATD/datd/MEG_MGS/IllustratorFigs/Coherence/Coh_' freqband '_' trlLocs '-trials_' seedLocs '-seeds.png'];
saveas(ff, fPathpng)

%% Plot imaginary part of coherence
fff = figure('Renderer','painters');
sgtitle(['ImagCoh: ' freqband ' ' trlLocs ' trials'])

for i                       = 1:length(tOnsets)
    subplot(2, 2, i)
    cfg                     = [];
    cfg.latency             = [tOnsets(i) tOffsets(i)];
    thisConnectivityIm      = ft_selectdata(cfg, connectivityImCoh_corr);
    
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
    cfg.zlim                = [-0.01 0.01];
    % cfg.zlim                = [0.12 0.22];
    if i                    == 1
        cfg.title           = 'Fixation';
    elseif i                == 2
        cfg.title           = 'Stim On';
    elseif i                == 3
        cfg.title           = 'Early Delay';
    elseif i                == 4
        cfg.title           = 'Late Delay';
    end

    % cfg.zlim                = [-0.02 0.02];
    ft_topoplotER(cfg, thisConnectivityIm);
end
fPatheps = ['/d/DATD/datd/MEG_MGS/IllustratorFigs/Coherence/ImCoh_' freqband '_' trlLocs '-trials_' seedLocs '-seeds.svg'];
saveas(fff, fPatheps)
fPathpng = ['/d/DATD/datd/MEG_MGS/IllustratorFigs/Coherence/ImCoh_' freqband '_' trlLocs '-trials_' seedLocs '-seeds.png'];
saveas(fff, fPathpng)