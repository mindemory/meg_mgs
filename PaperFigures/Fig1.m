clear; close all; clc;
warning('off', 'all');

addpath('/d/DATD/hyper/software/fieldtrip-20220104/');
ft_defaults;
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'));

%% Figure 1B: plot of gaze in 2D
pthName = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-29/eyetracking/run-01/sub-29_task-mgs_run-01_eyetracking.mat';
load(pthName, 'ii_data');
valIdx = find(ismember(ii_data.XDAT, [4, 5]));
Tar = unique([ii_data.TarX(valIdx), ii_data.TarY(valIdx)], 'rows');
figure();
plot(Tar(:, 1), Tar(:, 2), 'ks', 'MarkerFaceColor', 'k', 'MarkerSize', 10);
hold on;
for ii = 1:length(Tar)
    Tar(ii)
    thisX = ii_data.X((ii_data.TarX == Tar(ii, 1)) & ...
                      (ii_data.TarY == Tar(ii, 2)) & ...
                      (ismember(ii_data.XDAT, [4, 5])));
    thisY = ii_data.Y((ii_data.TarX == Tar(ii, 1)) & ...
                      (ii_data.TarY == Tar(ii, 2)) & ...
                      (ismember(ii_data.XDAT, [4, 5])));
    plot(thisX, thisY, 'o-', 'MarkerFaceColor','auto', 'MarkerSize', 2);
end
plot(9.*cos(0:0.01:2*pi), 9.*sin(0:0.01:2*pi), 'm--', 'LineWidth',2)
xlim([-10, 10])
ylim([-10, 10])

%% Figure 1B (MEG part)
% subList = [1 2 3 4 5 6 7 8 9 10 11 12 13 15 16 17 18 19 20 22 23 24 25 26 27 28 29 30 31 32];
subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, ...
               18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
metaTFRpath = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-meta/sub-meta_task-mgs_TFRbyCond.mat';
load('NYUKIT_helmet.mat');
if ~exist(metaTFRpath, 'file')
    for i                                              = 1:10
        eval(['TFR' num2str(i) ' = {};']);
    end
    % for subjID = subList
    for sIdx = 1:length(subList)
    
        disp(['Running ' num2str(sIdx) ' of ' num2str(length(subList)) ' subjects.'])
        subjID = subList(sIdx);
    
        % Initalizing variables
        bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
        taskName = 'mgs';
    
    
        derivativesRoot = [bidsRoot filesep 'derivatives/sub-' num2str(subjID, '%02d') '/meg'];
        subName = ['sub-' num2str(subjID, '%02d')];
        megRoot = [bidsRoot filesep subName filesep 'meg'];
        stimRoot = [bidsRoot filesep subName filesep 'stimfiles'];
        fNameRoot = [subName '_task-' taskName];
    
        if sIdx == subList(1)
            rawdata_path = [derivativesRoot filesep fNameRoot '_run-01_raw.mat'];
            load(rawdata_path, 'lay');
        end
    
        stimLocked_fpath = [derivativesRoot filesep fNameRoot '_stimlocked.mat'];
        TFR_fpath = [derivativesRoot filesep fNameRoot '_TFR.mat'];
        load(TFR_fpath);
        if sIdx == 1
            TFRtemp = TFRleft_power;
        end
    
        [TFR1{sIdx}, TFR2{sIdx}, TFR3{sIdx}, TFR4{sIdx}, TFR5{sIdx}, TFR6{sIdx}, ...
         TFR7{sIdx}, TFR8{sIdx}, TFR9{sIdx}, TFR10{sIdx}] = ...
                        extractPowByCond(TFRleft_power, TFRright_power);
        
        clearvars TFRleft_power TFRright_power;
    
    end

    save(metaTFRpath, 'TFR1', 'TFR2', 'TFR3', 'TFR4', 'TFR5', 'TFR6', 'TFR7', ...
                      'TFR8', 'TFR9', 'TFR10');
else
    load(metaTFRpath);
end

%%
cfg                                = [];
cfg.keepindividual                 = 'no';
TFRbaseline                        = ft_freqgrandaverage(cfg, TFR1{:}, TFR2{:}, TFR3{:}, TFR4{:}, TFR5{:}, ...
                                        TFR6{:}, TFR7{:}, TFR8{:}, TFR9{:}, TFR10{:});
TFRbaseline.grad                   = TFR1{1,1}.grad;
%%
freqband = 'beta';
delayCat = 'Middle';
% 
% layNew = lay;
% 
% layNew.mask{1, 1} = lay.mask{1, 1} .* 1.7;
% layNew.outline{1, 5} = lay.mask{1, 1};
% layNew.pos = lay.pos .* 2;

figure();
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
    % cfg.baseline                   = [-1.0 -0.5];
    % cfg.baselinetype               = 'absolute';
    if strcmp(delayCat, 'ITI')
        cfg.xlim                   = [-0.3 0];
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
        cfg.ylim                   = [14 30];
    end
    cfg.marker                     = 'on';
    cfg.layout                     = layNew;
    % cfg.layout = 'CTF157_helmet';
    cfg.interplimits               = 'electrodes';
    cfg.colormap                   = '*RdBu';
    ft_topoplotTFR(cfg, TFRdiff);

    clearvars TFRgrandavg TFRdiff;
    % pause(0.1);
end

%%
TFRtemp = TFR1{1};
cfg = [];
cfg.keepindividual = 'no';
TFRg = ft_freqgrandaverage(cfg, TFR1{:});
TFRg.grad = TFRtemp.grad;

TFRbase = ft_freqgrandaverage(cfg, TFR1{:}, TFR2{:}, TFR3{:}, TFR4{:}, TFR5{:},...
                             TFR6{:}, TFR7{:}, TFR8{:}, TFR9{:}, TFR10{:});
TFRbase.grad = TFRtemp.grad;

cfg = [];
cfg.parameter = 'powspctrm';  % Specify the parameter to operate on
% cfg.operation = 'x1 / x2'; %/(x1+x2)';   % Element-wise subtraction
cfg.operation = '(10^(x1/10) - 10^(x2/10)) / (10^(x1/10) + 10^(x2/10))';
% cfg.operation = '(10^(x1/10) ) / (10^(x1/10) + 10^(x2/10))';
TFRdiff = ft_math(cfg, TFRg, TFRl);

cfg = [];
cfg.parameter = 'powspctrm';  % Specify the parameter to operate on
cfg.operation = '(10^(x1/10) ) / (10^(x1/10) + 10^(x2/10))';
TFRdiff = ft_math(cfg, TFRg, TFRl);


cfg                            = [];
% cfg.figure                     = 'gcf';
% cfg.baseline                   = [-1.0 0];
% cfg.baselinetype               = 'relative';
cfg.xlim                       = [0.5 1];
cfg.ylim                       = [8 12];
cfg.marker                     = 'on';
cfg.layout                     = lay;
cfg.colormap                   = '*RdBu';
% cfg.interpolatenan             = 'no';
% cfg.interpolation              = 'linear';
% cfg.interplimits = 'head';
ft_topoplotTFR(cfg, TFRdiff);

%% Figure 1C
% Perform cluster-stats to identify top-channels that capture alpha topography
TFRleft                                  = cell(1, length(subList));
TFRright                                 = cell(1, length(subList));
for ii                                   = 1:length(subList) % numsubs
    leftTargs                            = [4 5 6 7 8];
    rightTargs                           = [1 2 3 9 10];
    % leftTargs                            = [7 8];
    % rightTargs                           = [9 10];
    for jj                               = 1:length(leftTargs)
        
        if jj                            == 1
            eval(['TFRleft{ii} = TFR' num2str(leftTargs(jj)) '{' num2str(ii) '};']);
            eval(['TFRright{ii} = TFR' num2str(rightTargs(jj)) '{' num2str(ii) '};']);
        else
            eval(['powTempLeft = TFR' num2str(leftTargs(jj)) '{' num2str(ii) '}.powspctrm;']);
            eval(['powTempRight = TFR' num2str(rightTargs(jj)) '{' num2str(ii) '}.powspctrm;']);
            TFRleft{ii}.powspctrm        = cat(4, TFRleft{ii}.powspctrm, powTempLeft);
            TFRright{ii}.powspctrm       = cat(4, TFRright{ii}.powspctrm, powTempRight);
            clearvars powTempLeft powTempRight;
        end
    end
    TFRleft{ii}.powspctrm                = mean(TFRleft{ii}.powspctrm, 4, 'omitnan');
    TFRright{ii}.powspctrm               = mean(TFRright{ii}.powspctrm, 4, 'omitnan');
end

cfg                                      = [];
cfg.keepindividual                       = 'no';
TFRleft_grandavg                         = ft_freqgrandaverage(cfg, TFRleft{:});
TFRright_grandavg                        = ft_freqgrandaverage(cfg, TFRright{:});
TFRleft_grandavg.grad                    = TFRleft{1}.grad;
TFRright_grandavg.grad                   = TFRright{1}.grad;

left_sensors = {'AG013', 'AG014', 'AG015', 'AG016', 'AG023', 'AG025', 'AG026', ...
                'AG027', 'AG028', 'AG041', 'AG042', 'AG043', 'AG059', 'AG060', ...
                'AG066', 'AG092'};
leftModulChanIdx = find(ismember(TFRleft_grandavg.label, left_sensors));
% Visualize powerspectra
figure();
timeOn                                       = [-0.3 0.1 0.5 1   1.5];
timeOff                                      = [ 0   0.3 1   1.5 1.7];
plttitles                                    = {'ITI', 'Stim On', 'Early Delay', ...
                                                'Middle Delay', 'Late Delay'};
for tIdx = 1:5
subplot(2, 3, tIdx)
timeIdx                                  = find((TFRleft_grandavg.time >= timeOn(tIdx)) & (TFRleft_grandavg.time <= timeOff(tIdx)));

hold on;
plot(TFRleft_grandavg.freq, 10.^(squeeze(mean(TFRleft_grandavg.powspctrm(leftModulChanIdx, :, timeIdx), [1, 3], 'omitnan'))./10), 'LineWidth', 2);
plot(TFRright_grandavg.freq, 10.^(squeeze(mean(TFRright_grandavg.powspctrm(leftModulChanIdx, :, timeIdx), [1, 3], 'omitnan'))./10), 'LineWidth', 2);
legend('Ipsi', 'Contra');
xlabel('Frequency (Hz)')
ylabel('Power (fT^2)')
title(plttitles{tIdx})
% ylim([0.98 1.01]);
% yline(1, '--')
% suptitle('Power spectra for occipital channels')
end



cfg                                      = [];
cfg.parameter                            = 'powspctrm';
cfg.operation                            = '(10^(x1/10) - 10^(x2/10)) / (10^(x1/10) + 10^(x2/10))';
% cfg.operation                           = '(10^(x2/10) ) / (10^(x1/10) + 10^(x2/10))';
TFRdiff                                  = ft_math(cfg, TFRright_grandavg, TFRleft_grandavg);
cfg                                      = [];
cfg.parameter                            = 'powspctrm';
cfg.operation                            = '(10^(x1/10) ) / (10^(x1/10) + 10^(x2/10))';
TFRleft_basecorr                         = ft_math(cfg, TFRleft_grandavg, TFRright_grandavg);
TFRright_basecorr                        = ft_math(cfg, TFRright_grandavg, TFRleft_grandavg);

cfg = [];
cfg.layout = lay;
cfg.xlim = [-0.5 1.7];
cfg.ylim = [4 40];
cfg.colormap                             = '*RdBu';

ft_multiplotTFR(cfg, TFRleft_basecorr)


freqband                                 = 'theta'; % 'theta', 'alpha', 'beta', 'all'
if strcmp(freqband, 'theta')
    freqBounds                           = [4 8];
elseif strcmp(freqband, 'alpha')
    freqBounds                           = [8 12];
elseif strcmp(freqband, 'beta')
    freqBounds                           = [12 35];
elseif strcmp(freqband, 'all')
    freqBounds                           = [2 40];
end
timeBounds                               = [1 1.5];
freqIdx                                  = find((TFRdiff.freq >= freqBounds(1)) & (TFRdiff.freq <= freqBounds(2)));
timeIdx                                  = find((TFRdiff.time >= timeBounds(1)) & (TFRdiff.time <= timeBounds(2)));
avgPowAcrossChan                         = mean(TFRdiff.powspctrm(:, freqIdx, timeIdx), [2, 3], 'omitnan');
qt                                       = 0.1;
lowBound                                 = quantile(avgPowAcrossChan, qt);
highBound                                = quantile(avgPowAcrossChan, 1-qt);
highModulChanIdx                         = find((avgPowAcrossChan <= lowBound | avgPowAcrossChan >= highBound));
highModulChan                            = arrayfun(@(x) TFRdiff.label{x}, highModulChanIdx, 'UniformOutput', false);


cfg                                      = [];
cfg.xlim                                 = timeBounds; %[0.5 1];
cfg.ylim                                 = freqBounds; %[8 12];
cfg.zlim                                 = [-0.04 0.04];
cfg.marker                               = 'off';
cfg.layout                               = lay;
cfg.colormap                             = '*RdBu';
cfg.colorbar                             = 'yes';
cfg.highlight                            = 'on'; 
cfg.highlightchannel                     = highModulChan;
cfg.highlightsymbol                      = '+';
cfg.highlightcolor                       = [1 1 0];
cfg.hightlightsize                       = 40;
cfg.hightlightfontsize                   = 20;
ft_topoplotTFR(cfg, TFRdiff)

%% Visualize
left_chans                               = {'AG009', 'AG010', 'AG011', 'AG012', 'AG013', ...
                                            'AG014', 'AG015', 'AG016', 'AG025', 'AG026', ...
                                            'AG027', 'AG028', 'AG029', 'AG030', 'AG032', ...
                                            'AG041', 'AG042', 'AG043', 'AG045', 'AG048', ...
                                            'AG057', 'AG059', 'AG060', 'AG062', 'AG063', ...
                                            'AG073', 'AG075', 'AG078', 'AG079', 'AG091', ...
                                            'AG092', 'AG106', 'AG110', 'AG111'};
right_chans                              = {'AG001', 'AG002', 'AG003', 'AG004', 'AG006', ...
                                            'AG007', 'AG008', 'AG018', 'AG019', 'AG020', ...
                                            'AG021', 'AG022', 'AG024', 'AG034', 'AG036', ...
                                            'AG037', 'AG038', 'AG039', 'AG040', 'AG049', ...
                                            'AG050', 'AG051', 'AG052', 'AG053', 'AG054', ...
                                            'AG055', 'AG056', 'AG058', 'AG065', 'AG081', ...
                                            'AG082', 'AG085', 'AG097', 'AG098', 'AG103', ...
                                            'AG104', 'AG113'};

chan_groups                              = {'left', 'right'};
target_groups                            = {'left', 'right', 'diff'};
figIdxHolder                             = [1 2 3;
                                            4 5 6];
% tSeriesIdxHolder                         = [4 5 6;
%                                             10 11 12];

qt                                       = 0.4;
figure();
for ii                                   = 1:length(chan_groups)
    ch_group                             = chan_groups{ii};
    if strcmp(ch_group, 'left')
        chan_labels                      = left_chans;
        titleHolder                      = {'Left chans, left target', ...
                                            'Left chans, right target', ...
                                            'Left chans, difference'};
    else
        chan_labels                      = right_chans;
        titleHolder                      = {'Right chans, left target', ...
                                            'Right chans, right target', ...
                                            'Right chans, difference'};
    end
    for jj                               = 1:length(target_groups)
        targ_group                       = target_groups{jj};
        if strcmp(targ_group, 'left')
            data                         = TFRleft_basecorr;
        elseif strcmp(targ_group, 'right')
            data                         = TFRright_basecorr;
        else
            data                         = TFRdiff;
        end

        chIdx                            = find(ismember(data.label, chan_labels));
        freqIdx                          = find((data.freq >= 5) & (data.freq <= 40));
        timeIdx                          = find((data.time >= -1) & (data.time <= 2));

        thisPowMat                       = data.powspctrm(chIdx, freqIdx, timeIdx);
        thisPowMat                       = thisPowMat(:);
        qtThresh                         = 0.1;
        lowPow                           = quantile(thisPowMat, qtThresh);
        highPow                          = quantile(thisPowMat, 1-qtThresh);
        % PowScale                         = max(abs(lowPow), abs(highPow));

        subplot(2, 3, figIdxHolder(ii, jj))
        cfg                              = [];
        cfg.figure                       = 'gcf';
        cfg.channel                      = chan_labels;
        cfg.xlim                         = [-1 2];
        cfg.ylim                         = [5 40];
        if jj                            <= 2
            cfg.zlim                     = [0.47 0.53];
        else
            cfg.zlim                     = [-0.02 0.02];
        end
        cfg.colormap                     = '*RdBu';
        cfg.title                        = titleHolder{jj};
        cfg.fontsize                     = 12;
        ft_singleplotTFR(cfg, data)

        % subplot(4, 3, tSeriesIdxHolder(ii, jj))
        % tPlotIdx = find((data.time>=-1) & (data.time<=2));
        % plot(data.time(tPlotIdx), squeeze(mean(data.powspctrm(chIdx, freqIdx, tPlotIdx), [1, 2], 'omitnan')));

    end
end
% cfg                                      = [];
% % Left channels
% % cfg.channel                              = 
% % Right channels
% cfg.channel                               = right_chans;
% cfg.xlim                                  = [-1 2];
% cfg.ylim                                  = [5 40];
% cfg.colormap                              = '*RdBu';
% ft_singleplotTFR(cfg, TFRleft_basecorr)

%% Visualize
left_chans                               = {'AG009', 'AG010', 'AG011', 'AG012', 'AG013', ...
                                            'AG014', 'AG015', 'AG016', 'AG025', 'AG026', ...
                                            'AG027', 'AG028', 'AG029', 'AG030', 'AG032', ...
                                            'AG041', 'AG042', 'AG043', 'AG045', 'AG048', ...
                                            'AG057', 'AG059', 'AG060', 'AG062', 'AG063', ...
                                            'AG073', 'AG075', 'AG078', 'AG079', 'AG091', ...
                                            'AG092', 'AG106', 'AG110', 'AG111'};
right_chans                              = {'AG001', 'AG002', 'AG003', 'AG004', 'AG006', ...
                                            'AG007', 'AG008', 'AG018', 'AG019', 'AG020', ...
                                            'AG021', 'AG022', 'AG024', 'AG034', 'AG036', ...
                                            'AG037', 'AG038', 'AG039', 'AG040', 'AG049', ...
                                            'AG050', 'AG051', 'AG052', 'AG053', 'AG054', ...
                                            'AG055', 'AG056', 'AG058', 'AG065', 'AG081', ...
                                            'AG082', 'AG085', 'AG097', 'AG098', 'AG103', ...
                                            'AG104', 'AG113'};



data                                     = TFRdiff;
leftchIdx                                = find(ismember(data.label, left_chans));
rightchIdx                               = find(ismember(data.label, right_chans));
freqIdx                                  = find((data.freq >= 5) & (data.freq <= 40));
timeIdx                                  = find((data.time >= -1) & (data.time <= 2));

leftPowMat                               = data.powspctrm(leftchIdx, :, :);
rightPowMat                              = data.powspctrm(rightchIdx, :, :);

data2                                    = TFRdiff;
meanLeft                                 = mean(leftPowMat, 1, 'omitnan');
meanRight                                = mean(rightPowMat, 1, 'omitnan');
meanDiff                                 = squeeze(meanLeft - meanRight);
fakePowMat                               = NaN(size(data.powspctrm));
for ijk                                  = 1:length(leftchIdx)
    data2.powspctrm(ijk, :, :)              = meanDiff;
end

figure();


subplot(1, 3, 1)
cfg                                      = [];
cfg.figure                               = 'gcf';
cfg.channel                              = left_chans;
cfg.xlim                                 = [-1 2];
cfg.ylim                                 = [5 40];
if jj                                    <= 2
    cfg.zlim                             = [0.47 0.53];
else
    cfg.zlim                             = [-0.02 0.02];
end
cfg.colormap                             = '*RdBu';
cfg.title                                = 'Left chans';
cfg.fontsize                             = 12;
ft_singleplotTFR(cfg, data)

subplot(1, 3, 2)
cfg                                      = [];
cfg.figure                               = 'gcf';
cfg.channel                              = right_chans;
cfg.xlim                                 = [-1 2];
cfg.ylim                                 = [5 40];
if jj                                    <= 2
    cfg.zlim                             = [0.47 0.53];
else
    cfg.zlim                             = [-0.02 0.02];
end
cfg.colormap                             = '*RdBu';
cfg.title                                = 'Right chans';
cfg.fontsize                             = 12;
ft_singleplotTFR(cfg, data)

subplot(1, 3, 3)
cfg                                      = [];
cfg.figure                               = 'gcf';
cfg.channel                              = left_chans;
cfg.xlim                                 = [-1 2];
cfg.ylim                                 = [5 40];
if jj                                    <= 2
    cfg.zlim                             = [0.47 0.53];
else
    cfg.zlim                             = [-0.04 0.04];
end
cfg.colormap                             = '*RdBu';
cfg.title                                = 'All chans';
cfg.fontsize                             = 12;
ft_singleplotTFR(cfg, data2)


%% Perform cluster stats
cfg                       = [];
cfg.spmversion            = 'spm12';
cfg.method                = 'montecarlo';
cfg.latency               = [0.5 1];
cfg.frequency             = [8 12];
cfg.statistic             = 'ft_statfun_depsamplesT';
cfg.correctm              = 'cluster';
cfg.clusteralpha          = 0.1;
cfg.clusterstatistic      = 'maxsum';
cfg.numrandomization      = 100;
cfg.minnbchan             = 2;
cfg.tail                  = 0; % two-tailed
cfg.alpha                 = 0.05;
% cfg.correcttail           = 'alpha';
numSubj                   = numel(subList);
% cfg.design                = [ones(1, numSubj) 2*ones(1, numSubj);
%                              1:numSubj 1:numSubj];
cfg.design                = [1:numSubj 1:numSubj;
                             ones(1, numSubj) 2*ones(1, numSubj)];
cfg.ivar                  = 2;
cfg.uvar                  = 1;
cfg_neighbor              = [];
cfg_neighbor.method       = 'triangulation';
cfg.neighbours            = ft_prepare_neighbours(cfg_neighbor, TFRleft{1});
stats                     = ft_freqstatistics(cfg, TFRright{:}, TFRleft{:});

% Significant clusters
pos_clusters              = find([stats.posclusters.prob] < cfg.alpha);
neg_clusters              = find([stats.negclusters.prob] < cfg.alpha);
% Mask
stats.mask                = false(size(stats.stat));
for cl                    = pos_clusters
    stats.mask            = stats.mask | (stats.posclusterslabelmat == cl);
end
for cl                    = neg_clusters
    stats.mask            = stats.mask | (stats.negclusterslabelmat == cl);
end

% Visualize the topography with clusters
cfg                       = [];
cfg.parameter             = 'stat';
cfg.maskparameter         = 'mask';
cfg.xlim                  = [0.5 1];
cfg.ylim                  = [8 12];
cfg.layout                = lay;
cfg.colormap              = '*RdBu';
cfg.colorbar              = 'yes';
cfg.highlight             = 'on';
cfg.highlightchannel      = find(any(stats.mask, [2 3]));
cfg.markersymbol          = '*';
cfg.markercolor           = [0 0 0];
ft_topoplotTFR(cfg, stats);