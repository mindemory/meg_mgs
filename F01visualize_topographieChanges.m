load('NYUKIT_helmet.mat')
freqband                                     = 'alpha'; % 'theta', 'alpha', 'beta', 'all'
timeOn                                       = [-0.3 0.1 0.5 1 1.5];
timeOff                                      = [0 0.3 1 1.5 1.7];
plttitles                                    = {'ITI', 'Stim On', ...
                                                'Early Delay', ...
                                                'Middle Delay', ...
                                                'Late Delay'};
% layNew = lay;
% layNew.pos = layNew.pos .* 2; %(:, 2) = layNew.pos(:, 2) + 0.05;
figure();
sgtitle(freqband)
for tIdx                                     = 1:length(timeOff)
    subplot(2, 3, tIdx)
    if strcmp(freqband, 'theta')
        freqBounds                           = [4 8];
    elseif strcmp(freqband, 'alpha')
        freqBounds                           = [8 12];
    elseif strcmp(freqband, 'beta')
        freqBounds                           = [12 35];
    elseif strcmp(freqband, 'all')
        freqBounds                           = [2 40];
    end
    timeBounds                               = [timeOn(tIdx) timeOff(tIdx)];
    freqIdx                                  = find((TFRdiff.freq >= freqBounds(1)) & (TFRdiff.freq <= freqBounds(2)));
    timeIdx                                  = find((TFRdiff.time >= timeBounds(1)) & (TFRdiff.time <= timeBounds(2)));
    avgPowAcrossChan                         = mean(TFRdiff.powspctrm(:, freqIdx, timeIdx), [2, 3], 'omitnan');
    qt                                       = 0.1;
    lowBound                                 = quantile(avgPowAcrossChan, qt);
    highBound                                = quantile(avgPowAcrossChan, 1-qt);
    highModulChanIdx                         = find((avgPowAcrossChan <= lowBound | avgPowAcrossChan >= highBound));
    highModulChan                            = arrayfun(@(x) TFRdiff.label{x}, highModulChanIdx, 'UniformOutput', false);
    if tIdx == 2
    leftModulChanIdx                         = find((avgPowAcrossChan <= lowBound));
    leftModulChan                            = arrayfun(@(x) TFRdiff.label{x}, leftModulChanIdx, 'UniformOutput', false);
    end
    
    cfg                                      = [];
    cfg.figure                               = 'gcf';
    cfg.xlim                                 = timeBounds; %[0.5 1];
    cfg.ylim                                 = freqBounds; %[8 12];
    cfg.zlim                                 = [-0.04 0.04];
    cfg.marker                               = 'off';
    cfg.layout                               = layNew;
    cfg.colormap                             = '*RdBu';
    cfg.colorbar                             = 'yes';
    cfg.highlight                            = 'on'; 
    cfg.highlightchannel                     = highModulChan;
    cfg.highlightsymbol                      = '+';
    cfg.highlightcolor                       = [1 1 0];
    cfg.hightlightsize                       = 40;
    cfg.hightlightfontsize                   = 20;
    cfg.title                                = plttitles{tIdx};
    ft_topoplotTFR(cfg, TFRdiff)
end
