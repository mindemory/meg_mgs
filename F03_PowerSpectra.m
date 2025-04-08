clear; close all; clc;
warning('off', 'all');
%% Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/'); 

addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
%%
subList                             = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 17, ...
                                       18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
subList = [1 2 3 4 5];
right_sensors                       = {'AG001', 'AG002', 'AG007', 'AG008', 'AG020', 'AG022', 'AG023', ...
                                       'AG024', 'AG034', 'AG036', 'AG050', 'AG055', 'AG065', 'AG066', ...
                                       'AG098', 'AG103'};
left_sensors                        = {'AG013', 'AG014', 'AG015', 'AG016', 'AG023', 'AG025', 'AG026', ...
                                       'AG027', 'AG028', 'AG041', 'AG042', 'AG043', 'AG059', 'AG060', ...
                                       'AG066', 'AG092'};
% right_sensors                       = {'AG036', 'AG007', 'AG008', 'AG001', 'AG051', ...
%                                        'AG050', 'AG019', 'AG039', 'AG052', 'AG053', ...
%                                        'AG022', 'AG055'};
% left_sensors                        = {'AG043', 'AG060', 'AG026', 'AG014', 'AG032', ...
%                                        'AG015', 'AG062', 'AG010', 'AG078', 'AG073', ...
%                                        'AG059', 'AG092'};

tOnset                              = [-0.5 0.0 0.5 1  ];
tOffset                             = [ 0   0.3 1   1.5];
titleDict                           = {'Fixation', 'StimOn', 'Early Delay', 'Late Delay'};
% freqList                            = 4:0.5:150;
freqList                            = logspace(log10(4), log10(150), 300);

ipsiSpectra                         = NaN(length(subList), length(tOnset), 100);% length(freqList));
contraSpectra                       = NaN(length(subList), length(tOnset), 100); %length(freqList));
ipsiFreq                            = NaN(length(subList), length(tOnset), 100);
contraFreq                          = NaN(length(subList), length(tOnset), 100);

for sIdx                            = 1:length(subList)

    disp(['Running ' num2str(sIdx) ' of ' num2str(length(subList)) ' subjects.'])
    subjID                          = subList(sIdx);

    % Initalizing variables
    bidsRoot                        = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
    taskName                        = 'mgs';


    derivativesRoot                 = [bidsRoot filesep 'derivatives/sub-' ...
                                       num2str(subjID, '%02d') '/meg'];
    subName                         = ['sub-' num2str(subjID, '%02d')];
    megRoot                         = [bidsRoot filesep subName filesep 'meg'];
    stimRoot                        = [bidsRoot filesep subName filesep 'stimfiles'];
    fNameRoot                       = [subName '_task-' taskName];

    stimLocked_fpath                = [derivativesRoot filesep fNameRoot ...
                                       '_stimlocked_lineremoved.mat'];
    load(stimLocked_fpath);

    % Load the data
    epocThis                        = epocStimLocked;

    % Logical mask to find epochs matching your criteria
    trial_criteria_left             = (epocThis.trialinfo(:,2) == 4) | ...
                                      (epocThis.trialinfo(:,2) == 5) | ...
                                      (epocThis.trialinfo(:,2) == 6) | ...
                                      (epocThis.trialinfo(:,2) == 7) | ...
                                      (epocThis.trialinfo(:,2) == 8);
    trial_criteria_right            = (epocThis.trialinfo(:,2) == 1) | ...
                                      (epocThis.trialinfo(:,2) == 2) | ...
                                      (epocThis.trialinfo(:,2) == 3) | ...
                                      (epocThis.trialinfo(:,2) == 9) | ...
                                      (epocThis.trialinfo(:,2) == 10);
    has_no_nans                     = cellfun(@(x) ~any(isnan(x(:))), epocThis.trial)';
    % Combine both criteria
    valid_trialsLeft                = find(trial_criteria_left & has_no_nans);
    valid_trialsRight               = find(trial_criteria_right & has_no_nans);

    % Select Left & Right trials
    cfg                             = [];
    cfg.trials                      = valid_trialsLeft;
    epocLeft                        = ft_selectdata(cfg, epocThis);
    cfg.trials                      = valid_trialsRight;
    epocRight                       = ft_selectdata(cfg, epocThis);
    
    for tt                          = 1:length(tOnset)
        % Select desired time of interest
        cfg                         = [];
        cfg.latency                 = [tOnset(tt) tOffset(tt)];
        epocLeftSelected            = ft_selectdata(cfg, epocLeft);
        epocRightSelected           = ft_selectdata(cfg, epocRight);

        % Compute power spectrum
        cfg                         = [];
        cfg.method                  = 'mtmfft';
        % cfg.taper                   = 'dpss';
        cfg.taper                   = 'hanning';
        cfg.foi                     = freqList;
        % cfg.tapsmofrq               = 2.5;
        spectrumLeft                = ft_freqanalysis(cfg, epocLeftSelected);
        spectrumRight               = ft_freqanalysis(cfg, epocRightSelected);

        % Average the spectrum for ipsi and contra sensors
        right_indices               = find(ismember(spectrumLeft.label, right_sensors));
        left_indices                = find(ismember(spectrumLeft.label, left_sensors));
        ipsiSpectrum                = (mean(spectrumLeft.powspctrm(left_indices, :), 1, 'omitnan') + ...
                                       mean(spectrumRight.powspctrm(right_indices, :), 1, 'omitnan')) ./ 2;
        contraSpectrum              = (mean(spectrumLeft.powspctrm(right_indices, :), 1, 'omitnan') + ...
                                       mean(spectrumRight.powspctrm(left_indices, :), 1, 'omitnan')) ./ 2;
        
        numIpsiFreq                 = length(spectrumLeft.freq);
        numContraFreq               = length(spectrumRight.freq);

        ipsiSpectra(sIdx, tt, 1:numIpsiFreq) ...
                                    = ipsiSpectrum;
        contraSpectra(sIdx, tt, 1:numContraFreq) ...
                                    = contraSpectrum;
        ipsiFreq(sIdx, tt, 1:numIpsiFreq) ...
                                    = spectrumLeft.freq;
        contraFreq(sIdx, tt, 1:numContraFreq) ...
                                    = spectrumRight.freq;
    end

    
    %%%%% Average reference
    % cfg = [];
    % cfg.reref = 'yes';
    % cfg.refchannel = 'all';
    % cfg.refmethod = 'avg';
    % epocLeft = ft_preprocessing(cfg, epocLeft);
    % epocRight = ft_preprocessing(cfg, epocRight);

   
end

%%
ipsiSpectraDB                       = 10*log10(ipsiSpectra);
contraSpectraDB                     = 10*log10(contraSpectra);
% ipsiSpectraDB = ipsiSpectra;
% contraSpectraDB = contraSpectra;
custom_ticks                        = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150];
figure;
for tt                              = 1:length(tOnset)
    subplot(2, 2, tt)
    thisFreqIpsi                    = squeeze(mean(ipsiFreq(:, tt, :), 1, 'omitnan'));
    thisPowIpsiMean                 = squeeze(mean(ipsiSpectraDB(:, tt, :), 1, 'omitnan'));
    thisPowIpsiSEM                  = squeeze(std(ipsiSpectraDB(:, tt, :), [], 1, 'omitnan')) ./ sqrt(length(subList)-1);
    thisFreqContra                  = squeeze(mean(contraFreq(:, tt, :), 1, 'omitnan'));
    thisPowContraMean               = squeeze(mean(contraSpectraDB(:, tt, :), 1, 'omitnan'));
    thisPowContraSEM                = squeeze(std(contraSpectraDB(:, tt, :), [], 1, 'omitnan')) ./ sqrt(length(subList)-1);
    
    plot(thisFreqIpsi, thisPowIpsiMean, 'DisplayName', 'Ipsi', 'LineWidth', 2)
    hold on;
    plot(thisFreqContra, thisPowContraMean, 'DisplayName', 'Contra', 'LineWidth', 2)
    
    xticks(custom_ticks);
    
    xticklabels(cellstr(num2str(custom_ticks')));
    xlim([5 50])
    % ylim([0 6e-28])
    legend;
    xlabel('Frequency (Hz)')
    ylabel('Power (fT^2)')
    if tt                           == 1
        title('Fixation')
    elseif tt                       == 2
        title('Stim On')
    elseif tt                       == 3
        title('Early Delay')
    elseif tt                       == 4
        title('Late Delay')
    end
end