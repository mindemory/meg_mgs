% Initalizing variables
bidsRoot                = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
taskName                = 'mgs';
derivativesRoot         = [bidsRoot filesep 'derivatives/sub-12/meg'];
fNameRoot               = 'sub-12_task-mgs';
load('NYUKIT_helmet.mat');
load('/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-12/meg/sub-12_task-mgs_TFR_phase.mat')

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


TFRphase_chosen = TFRright_phase;
seed_indices = find(ismember(TFRphase_chosen.label, seed_sensors));
seed_phase_left = TFRphase_chosen.phaseangle(:, seed_indices, :, :);
% seed_phase_right = TFR_phase_right.phaseangle(:, seed_indices, :, :);
seed_phase_mean_left = angle(squeeze(mean(exp(1i * seed_phase_left), 2, 'omitnan'))); % [frequencies x time x trials]
% seed_phase_mean_right = angle(squeeze(mean(exp(1i * seed_phase_right), 2, 'omitnan')));

plv_left = zeros(numel(TFRphase_chosen.label), size(TFRphase_chosen.phaseangle, 3), size(TFRphase_chosen.phaseangle, 4)); % chan * freq * time
for ch = 1:numel(TFRphase_chosen.label)
    currChanPhase = squeeze(TFRphase_chosen.phaseangle(:, ch, :, :));
    phase_diff = currChanPhase - seed_phase_mean_left;
    plv_left(ch, :, :) = abs(squeeze(mean(exp(1i * phase_diff), 1, 'omitnan')));
end

TFR_plv_left = TFRphase_chosen;
TFR_plv_left.powspctrm = plv_left;
TFR_plv_left = rmfield(TFR_plv_left, 'phaseangle');
TFR_plv_left.dimord = 'chan_freq_time';

for iii                 = 1:4           
    file_suffix         = fileSuffices{iii};
    TFR_coh_fpath       = [derivativesRoot filesep fNameRoot '_coh_' file_suffix '.mat'];
    TFR_coh.(fieldNames{iii}) ...
                        = load(TFR_coh_fpath);
end

%%
% freqbands                       = {'theta', 'alpha', 'beta'};
freqbands                       = {'theta'};
for ifx                         = 1:length(freqbands)
    freqband                    = freqbands{ifx};
    % ff                          = figure('Visible', 'off', 'Renderer','painters');
    ff                          = figure;
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
        cfg.zlim                = [0 0.2];
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
        ft_topoplotTFR(cfg, TFR_coh.(fieldNames{iii}).TFR_coh)
    end
    end
%     fPath = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/group_plots/' freqband '_coh_topo_' file_suffix '.svg'];
%     saveas(ff, fPath)
% end
end