subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, ...
               18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
% subList = [1, 2, 3, 4];
metaTFRpath = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-meta/sub-meta_task-mgs_TFRbyCond_lineremoved.mat';
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
   
        stimLocked_fpath = [derivativesRoot filesep fNameRoot '_stimlocked_lineremoved.mat'];
        TFR_fpath = [derivativesRoot filesep fNameRoot '_TFR_lineremoved.mat'];
        load(TFR_fpath);
        if sIdx == 1
            TFRtemp = powHigh_left;
        end
    
        [TFR1{sIdx}, TFR2{sIdx}, TFR3{sIdx}, TFR4{sIdx}, TFR5{sIdx}, TFR6{sIdx}, ...
         TFR7{sIdx}, TFR8{sIdx}, TFR9{sIdx}, TFR10{sIdx}] = ...
                        extractPowByCond(powHigh_left, powHigh_right);
        
        clearvars powHigh_left powHigh_right;
    
    end

    save(metaTFRpath, 'TFR1', 'TFR2', 'TFR3', 'TFR4', 'TFR5', 'TFR6', 'TFR7', ...
                      'TFR8', 'TFR9', 'TFR10');
else
    load(metaTFRpath);
end

%%
TFRleft                                  = cell(1, length(subList));
TFRright                                 = cell(1, length(subList));
for ii                                   = 1:length(subList) % numsubs
% TFRleft = cell(1, length(subList));
% TFRright = cell(1, 4);
% for ii =1:4
    leftTargs                            = [4 5 6 7 8];
    rightTargs                           = [1 2 3 9 10];
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
TFRbase_grandavg                         = ft_freqgrandaverage(cfg, TFRleft{:}, TFRright{:});
TFRleft_grandavg.grad                    = TFRleft{1}.grad;
TFRright_grandavg.grad                   = TFRright{1}.grad;
TFRbase_grandavg.grad                    = TFRleft{1}.grad;

%% Create Medendorp Like Figures
% right_sensors                            = {'AG001', 'AG003', 'AG004', 'AG005', 'AG006', 'AG007', 'AG008', ...
%                                             'AG018', 'AG019', 'AG020', 'AG022', 'AG024', 'AG034', 'AG036', ...
%                                             'AG038', 'AG039', 'AG040', 'AG050', 'AG051', 'AG052', 'AG053', ...
%                                             'AG055', 'AG065', 'AG081', 'AG098', 'AG103', 'AG104'};
% left_sensors                             = find_opposite_sensors(lay, right_sensors, 'right');
right_sensors               = {'AG001', 'AG002', 'AG007', 'AG008', 'AG020', 'AG022', 'AG023', ...
                               'AG024', 'AG034', 'AG036', 'AG050', 'AG055', 'AG065', 'AG066', ...
                               'AG098', 'AG103'};
left_sensors                = {'AG013', 'AG014', 'AG015', 'AG016', 'AG023', 'AG025', 'AG026', ...
                               'AG027', 'AG028', 'AG041', 'AG042', 'AG043', 'AG059', 'AG060', ...
                               'AG066', 'AG092'};
% left_idx                                 = find(ismember(left_sensors, TFRleft_grandavg.label));
% right_idx                                = find(ismember(right_sensors, TFRleft_grandavg.label));
% 
% cfg                                      = [];
% cfg.parameter                            = 'powspctrm';
% cfg.operation                            = '(10^(x1/10) - 10^(x2/10)) / (10^(x1/10) + 10^(x2/10))';
% TFRleftSensorsDiff                       = ft_math(cfg, TFRright_grandavg, TFRleft_grandavg);
% TFRrightSensorsDiff                      = ft_math(cfg, TFRleft_grandavg, TFRright_grandavg);
%%
TFRleft_grandavg_rawPow                  = TFRleft_grandavg;
TFRright_grandavg_rawPow                 = TFRright_grandavg;
TFRleft_grandavg_rawPow.powspctrm        = 10.^(TFRleft_grandavg_rawPow.powspctrm ./10);
TFRright_grandavg_rawPow.powspctrm       = 10.^(TFRright_grandavg_rawPow.powspctrm ./10);

% Define baseline time window
cfg                                      = [];
cfg.baseline                             = [-0.5 -0.1];
cfg.baselinetype                         = 'relative';  
cfg.parameter                            = 'powspctrm';
TFRleft_basecorr                         = ft_freqbaseline(cfg, TFRleft_grandavg_rawPow);
TFRright_basecorr                        = ft_freqbaseline(cfg, TFRright_grandavg_rawPow);

cfg                                      = [];
cfg.parameter                            = 'powspctrm';
% cfg.operation                            = '10^(x1/10) - 10^(x2/10)';
cfg.operation                            = 'x1 - x2';
TFRleftMinusright                        = ft_math(cfg, TFRleft_basecorr, TFRright_basecorr);
TFRrightMinusleft                        = ft_math(cfg, TFRright_basecorr, TFRleft_basecorr);

figure('Renderer','painters')
% Plot left trials
subplot(3, 3, 1)
cfg                                      = [];
cfg.figure                               = 'gcf';
cfg.layout                               = lay;
cfg.xlim                                 = [-0.3 1.7];
cfg.ylim                                 = [4 40];
% cfg.zlim                                 = [-0.06 0.06];
cfg.zlim                                 = [0.8 1.2];
cfg.colormap                             = '*RdBu';%'jet'; %'*RdBu';
cfg.channel                              = left_sensors;
cfg.title                                = 'Left Hemisphere';
ft_singleplotTFR(cfg, TFRleft_basecorr)
subplot(3, 3, 3)
cfg.channel                              = right_sensors;
cfg.title                                = 'Right hemisphere';
ft_singleplotTFR(cfg, TFRleft_basecorr)
% Plot right trials
subplot(3, 3, 4)
cfg.channel                              = left_sensors;
cfg.title                                = 'Left Hemisphere';
ft_singleplotTFR(cfg, TFRright_basecorr)
subplot(3, 3, 6)
cfg.channel                              = right_sensors;
cfg.title                                = 'Right Hemisphere';
ft_singleplotTFR(cfg, TFRright_basecorr)
% Plot the difference
cfg.zlim                                 = [-.1 0.1];
subplot(3, 3, 7)
cfg.channel                              = left_sensors;
cfg.title                                = 'Left Hemisphere';
ft_singleplotTFR(cfg, TFRrightMinusleft)
subplot(3, 3, 9)
cfg.channel                              = right_sensors;
cfg.title                                = 'Right Hemisphere';
ft_singleplotTFR(cfg, TFRleftMinusright)
% Plotting topographies
subplot(3, 3, 2)
cfg                                      = [];
cfg.figure                               = 'gcf';
cfg.layout                               = lay;
cfg.xlim                                 = [1 1.5];
cfg.ylim                                 = [15 25];
% cfg.zlim                                 = [-0.08 0.08];
cfg.zlim                                 = [0.8 1.2];
cfg.colormap                             = '*RdBu';%'jet'; %'*RdBu';
cfg.interpolation                        = 'v4';
cfg.comment                              = 'no';
cfg.marker                               = 'off';
cfg.highlight                            = 'on';
cfg.highlightchannel                     = [right_sensors, left_sensors];
cfg.highlightcolor                       = 'k';
cfg.highlightsymbol                      = 'o';
cfg.highlightsize                        = 6;
cfg.title                                = 'Left Targets';
ft_topoplotTFR(cfg, TFRleft_basecorr)
subplot(3, 3, 5)
cfg.title                                = 'Right Targets';
ft_topoplotTFR(cfg, TFRright_basecorr)