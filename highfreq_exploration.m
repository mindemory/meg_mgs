subList = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, ...
               18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
subList = [1, 2, 3, 4];
metaTFRpath = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-meta/sub-meta_task-mgs_TFRhighFreqbyCond_noiseremoved.mat';
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
        TFR_fpath = [derivativesRoot filesep fNameRoot '_TFR_highfreq_noiseremoved.mat'];
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
% TFRleft                                  = cell(1, length(subList));
% TFRright                                 = cell(1, length(subList));
% for ii                                   = 1:length(subList) % numsubs
TFRleft = cell(1, 4);
TFRright = cell(1, 4);
for ii =1:4
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
TFRleft_grandavg.grad                    = TFRleft{1}.grad;
TFRright_grandavg.grad                   = TFRright{1}.grad;

%%
right_sensors                            = {'AG001', 'AG003', 'AG004', 'AG005', 'AG006', 'AG007', 'AG008', ...
                                            'AG018', 'AG019', 'AG020', 'AG022', 'AG024', 'AG034', 'AG036', ...
                                            'AG038', 'AG039', 'AG040', 'AG050', 'AG051', 'AG052', 'AG053', ...
                                            'AG055', 'AG065', 'AG081', 'AG098', 'AG103', 'AG104'};
left_sensors                             = find_opposite_sensors(lay, right_sensors, 'right');

left_idx                                 = find(ismember(left_sensors, TFRleft_grandavg.label));
right_idx                                = find(ismember(right_sensors, TFRleft_grandavg.label));

cfg                                      = [];
cfg.parameter                            = 'powspctrm';
cfg.operation                            = '(10^(x1/10) - 10^(x2/10)) / (10^(x1/10) + 10^(x2/10))';
TFRleftSensorsDiff                       = ft_math(cfg, TFRright_grandavg, TFRleft_grandavg);
TFRrightSensorsDiff                      = ft_math(cfg, TFRleft_grandavg, TFRright_grandavg);

figure;
for i                                    = 1:2
    subplot(1, 3, i)
    
    cfg                                  = [];
    cfg.figure                           = 'gcf';
    cfg.layout                           = lay;
    cfg.xlim                             = [-0.5 1.7];
    cfg.colormap                         = 'jet'; %'*RdBu';
    % cfg.zlim                             = [-0.01 0.01];
    cfg.ylim                             = [40 100];
    if i                                 == 1
        cfg.channel                      = left_sensors;
        cfg.title                        = 'Left sensors';
        ft_singleplotTFR(cfg, TFRleftSensorsDiff)
    elseif i                             == 2
        cfg.channel                      = right_sensors;
        cfg.title                        = 'Right sensors';
        ft_singleplotTFR(cfg, TFRrightSensorsDiff)
    % elseif i                             == 3
    %     left_sensPow                     = TFRleftSensorsDiff.powspctrm(left_idx, :, :);
    %     right_sensPow                    = TFRrightSensorsDiff.powspctrm(right_idx, :, :);
    %     combinedPow                      = 

    end
end

figure;
plot(TFRright_grandavg.freq, squeeze(mean(TFRright_grandavg.powspctrm, [1, 3], 'omitnan')));