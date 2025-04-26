% Initalizing variables
bidsRoot                = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
taskName                = 'mgs';
derivativesRoot         = [bidsRoot filesep 'derivatives/sub-12/meg'];
fNameRoot               = 'sub-12_task-mgs';
load('NYUKIT_helmet.mat');

% Sensors
right_sensors           = {'AG001', 'AG002', 'AG007', 'AG008', 'AG020', 'AG022', 'AG023', ...
                           'AG024', 'AG034', 'AG036', 'AG050', 'AG055', 'AG065', 'AG066', ...
                           'AG098', 'AG103'};
left_sensors            = {'AG013', 'AG014', 'AG015', 'AG016', 'AG023', 'AG025', 'AG026', ...
                           'AG027', 'AG028', 'AG041', 'AG042', 'AG043', 'AG059', 'AG060', ...
                           'AG066', 'AG092'};


fieldNames              = {'one', 'two', 'three', 'four'};
fileSuffices            = {'leftTrials_rightSensors', 'rightTrials_rightSensors', ...
                           'leftTrials_leftSensors', 'rightTrials_leftSensors'};
seed_sensors.one        = right_sensors;
seed_sensors.two        = right_sensors;
seed_sensors.three      = left_sensors;
seed_sensors.four       = left_sensors;

for iii                 = 1:4           
    file_suffix         = fileSuffices{iii};
    TFR_coh_fpath       = [derivativesRoot filesep fNameRoot '_coh_' file_suffix '.mat'];
    TFR_coh.(fieldNames{iii}) ...
                        = load(TFR_coh_fpath);
end

%%

%% Normalized coh
TFR_coh_norm = TFR_coh;
for iii = 1:4
    thisPow = TFR_coh_norm.(fieldNames{iii}).TFR_coh.powspctrm;
    thisPowMean = mean(thisPow(:), "all", 'omitnan');
    thisPowStd = std(thisPow(:), [], 'all', 'omitnan');
    thisPowNorm = (thisPow - thisPowMean) / thisPowStd;
    TFR_coh_norm.(fieldNames{iii}).TFR_coh.powspctrm = thisPowNorm;
end
% freqbands                       = {'theta', 'alpha', 'beta'};
freqbands                       = {'beta'};
for ifx                         = 1:length(freqbands)
    freqband                    = freqbands{ifx};
    % ff                          = figure('Visible', 'off', 'Renderer','painters');
    ff                          = figure('Renderer', 'painters');
    tOnset                      = [-0.5 0.1 0.5 1  ];
    tOffset                     = [ 0   0.3 1   1.5];
    titleDict                   = {'Fixation', 'StimOn', 'Early Delay', 'Late Delay'};
    sgtitle(freqband)
    for iii                     = 1:4
        for idx                     = 1:length(tOnset)
            subplot(4, 4, (iii-1)*4 + idx)
            cfg                     = [];
            cfg.layout              = lay;
            cfg.figure              = 'gcf';
            cfg.xlim                = [tOnset(idx) tOffset(idx)];
            if strcmp(freqband, 'theta')
                cfg.ylim            = [5 8];
            elseif strcmp(freqband, 'alpha')
                cfg.ylim            = [8 12];
            elseif strcmp(freqband, 'beta')
                cfg.ylim            = [14 25];
            end
            % cfg.zlim                = [0 0.3];
            cfg.zlim                = [-1 1];
            cfg.marker              = 'off';
            cfg.parameter           = 'powspctrm';
            cfg.colormap            = '*RdBu';
            cfg.highlight           = 'on'; 
            cfg.highlightchannel    = seed_sensors.(fieldNames{iii});
            cfg.highlightsymbol     = '+';
            cfg.highlightcolor      = [0 1 0];
            cfg.hightlightsize      = 40;
            cfg.hightlightfontsize  = 30;
        
            cfg.title = titleDict{idx};
            ft_topoplotTFR(cfg, TFR_coh_norm.(fieldNames{iii}).TFR_coh)
        end
    end
%     fPath = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/group_plots/' freqband '_coh_topo_' file_suffix '.svg'];
%     saveas(ff, fPath)
% end
end