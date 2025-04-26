clear; close all; clc;

subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, ...
    18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
metaTFRpath = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-meta/sub-meta_task-mgs_TFRbyCond_lineremoved.mat';
load('NYUKIT_helmet.mat');
load(metaTFRpath);

%%
cfg                                = [];
cfg.keepindividual                 = 'no';
TFRbaseline                        = ft_freqgrandaverage(cfg, TFR1{:}, TFR2{:}, TFR3{:}, TFR4{:}, TFR5{:}, ...
    TFR6{:}, TFR7{:}, TFR8{:}, TFR9{:}, TFR10{:});
TFRbaseline.grad                   = TFR1{1,1}.grad;

%%
freqbands = {'theta', 'alpha', 'beta'};
delayCats = {'Fixation', 'StimOn', 'Early', 'Middle'};
% freqbands = {'theta'};
% delayCats = {'StimOn'};
for freqband = freqbands
    for delayCat = delayCats
        % freqband = 'beta';
        % delayCat = 'StimOn';

        fig = figure('visible','off', 'Renderer', 'painters');
        sgtitle([freqband ': ' delayCat])
        pltVec                             = [18 11 4 3 8 13 20 27 28 23];
        for ii                             = [1 2 3 4 5 6 7 8 9 10]
            subplot(5, 6, pltVec(ii))

            cfg                            = [];
            cfg.keepindividual             = 'no';
            eval(['TFRgrandavg = ft_freqgrandaverage(cfg, TFR' num2str(ii) '{:});']);
            TFRgrandavg.grad               = TFR1{1,1}.grad;

            cfg                            = [];
            cfg.parameter                  = 'powspctrm';  % Specify the parameter to operate on
            % cfg.operation                  = '(10^(x1/10) - 10^(x2/10)) / (10^(x1/10) + 10^(x2/10))';
            cfg.operation = '(10^(x1/10) ) / (10^(x2/10))';
            TFRdiff                        = ft_math(cfg, TFRgrandavg, TFRbaseline);

            cfg                            = [];
            cfg.figure                     = 'gcf';
            if strcmp(delayCat, 'Fixation')
                cfg.xlim                   = [-0.5 0];
            elseif strcmp(delayCat, 'StimOn')
                cfg.xlim                   = [0.1 0.3];
            elseif strcmp(delayCat, 'Early')
                cfg.xlim                   = [0.5 1];
            elseif strcmp(delayCat, 'Middle')
                cfg.xlim                   = [1 1.5];
            elseif strcmp(delayCat, 'Late')
                cfg.xlim                   = [1.5 1.7];
            end

            if strcmp(freqband, 'theta')
                cfg.ylim                   = [4 8];
            elseif strcmp(freqband, 'alpha')
                cfg.ylim                   = [8 12];
            elseif strcmp(freqband, 'beta')
                cfg.ylim                   = [15 25];
            end
            cfg.marker                     = 'off';
            cfg.layout                     = lay;
            cfg.interplimits               = 'electrodes';
            cfg.colormap                   = '*RdBu';
            ft_topoplotTFR(cfg, TFRdiff);

            clearvars TFRgrandavg TFRdiff;
        end
        fPath = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/group_plots/' freqband{1} '_' delayCat{1} '_indivLocations_topo.svg'];
        saveas(fig, fPath)
    end
end

%% For 6 locations
freqbands = {'theta', 'alpha', 'beta'};
% delayCats = {'Fixation', 'StimOn', 'Early', 'Middle'};
delayCats = {'Fixation', 'Middle'};
% freqbands = {'theta'};
% delayCats = {'StimOn'};
for freqband = freqbands
    for delayCat = delayCats
        % freqband = 'beta';
        % delayCat = 'StimOn';

        fig = figure('visible','off', 'Renderer', 'painters');
        sgtitle([freqband ': ' delayCat])
        pltVec                             = [8 3 2 5 10 11];
        for ii                             = [1 2 3 4 5 6]
            subplot(3, 4, pltVec(ii))
            
            cfg                            = [];
            cfg.keepindividual             = 'no';
            if ii                          == 1
                TFRgrandavg                = ft_freqgrandaverage(cfg, TFR1{:});
            elseif ii                      == 2
                TFRgrandavg                = ft_freqgrandaverage(cfg, TFR2{:}, TFR3{:});
            elseif ii                      == 3
                TFRgrandavg                = ft_freqgrandaverage(cfg, TFR4{:}, TFR5{:});
            elseif ii                      == 4
                TFRgrandavg                = ft_freqgrandaverage(cfg, TFR6{:});
            elseif ii                      == 5
                TFRgrandavg                = ft_freqgrandaverage(cfg, TFR7{:}, TFR8{:});
            elseif ii                      == 6
                TFRgrandavg                = ft_freqgrandaverage(cfg, TFR9{:});
            end
            % eval(['TFRgrandavg = ft_freqgrandaverage(cfg, TFR' num2str(ii) '{:});']);
            TFRgrandavg.grad               = TFR1{1,1}.grad;

            cfg                            = [];
            cfg.parameter                  = 'powspctrm';  % Specify the parameter to operate on
            % cfg.operation                  = '(10^(x1/10) - 10^(x2/10)) / (10^(x1/10) + 10^(x2/10))';
            cfg.operation = '(10^(x1/10) ) / (10^(x2/10))';
            TFRdiff                        = ft_math(cfg, TFRgrandavg, TFRbaseline);

            cfg                            = [];
            cfg.figure                     = 'gcf';
            if strcmp(delayCat, 'Fixation')
                cfg.xlim                   = [-0.5 0];
            elseif strcmp(delayCat, 'StimOn')
                cfg.xlim                   = [0.1 0.3];
            elseif strcmp(delayCat, 'Early')
                cfg.xlim                   = [0.5 1];
            elseif strcmp(delayCat, 'Middle')
                cfg.xlim                   = [1 1.5];
            elseif strcmp(delayCat, 'Late')
                cfg.xlim                   = [1.5 1.7];
            end

            if strcmp(freqband, 'theta')
                cfg.ylim                   = [4 8];
            elseif strcmp(freqband, 'alpha')
                cfg.ylim                   = [8 12];
            elseif strcmp(freqband, 'beta')
                cfg.ylim                   = [15 25];
            end
            cfg.marker                     = 'off';
            cfg.layout                     = lay;
            % cfg.interplimits               = 'electrodes';
            cfg.colormap                   = '*RdBu';
            ft_topoplotTFR(cfg, TFRdiff);

            clearvars TFRgrandavg TFRdiff;
        end
        fPathsvg = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/group_plots/topoByLoc/svg/' freqband{1} '_' delayCat{1} '_locGroups_topo.svg'];
        fPathpng = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/group_plots/topoByLoc/png/' freqband{1} '_' delayCat{1} '_locGroups_topo.png'];
        saveas(fig, fPathsvg)
        saveas(fig, fPathpng)
    end
end