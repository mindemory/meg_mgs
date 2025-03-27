clear; close all; clc;
warning('off', 'all');

addpath('/d/DATD/hyper/software/fieldtrip-20220104/');
ft_defaults;
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'));

subList = [1 2 3 4 5 6 7 8 9 10 11 12 13 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32];

metaTFRpath = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-meta/sub-meta_task-mgs_TFRbyCond.mat';
load('/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-01/meg/sub-01_task-mgs_run-01_raw.mat', 'lay');
load(metaTFRpath);

cfg                                = [];
cfg.keepindividual                 = 'no';
TFRbaseline                        = ft_freqgrandaverage(cfg, TFR1{:}, TFR2{:}, TFR3{:}, TFR4{:}, ...
                                        TFR6{:}, TFR7{:}, TFR8{:}, TFR9{:}, TFR10{:});
TFRbaseline.grad                   = TFR1{1,1}.grad;

figure();
pltVec                             = [28 20 12 10 16 22 30 38 40 34];
for ii                             = [1 2 3 4 6 7 8 9 10]
    subplot(7, 7, pltVec(ii))

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
    % cfg.baseline                   = [-1.0 -0.5];
    % cfg.baselinetype               = 'absolute';
    cfg.xlim                       = [0.1 0.4];
    cfg.ylim                       = [4 8];
    cfg.marker                     = 'off';
    cfg.layout                     = lay;
    cfg.colormap                   = '*RdBu';
    ft_topoplotTFR(cfg, TFRdiff);

    clearvars TFRgrandavg TFRdiff;
end