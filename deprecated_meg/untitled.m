
clearvars -except epocStimLocked grad;
data                          = epocStimLocked;
data.grad = grad;




% cfg = [];
% cfg.channel = data.label;%(1:157);
% cfg.method = 'triangulation';
% cfg.grad = grad;
% cfg.feedback = 'yes';
% neighbors = ft_prepare_neighbours(cfg, data);
% 
% cfg    = [];
% cfg.dftreplace = 'neighbour';
% cfg.dftfilter = 'yes';
% cfg.dftfreq = [60 120];
% % cfg.dft
% data = ft_preprocessing(cfg, data);
% 
% cfg = [];
% cfg.planarmethod = 'sincos';
% cfg.neighbours = neighbors;
% cfg.feedback = 'yes';
% dataPlanar = ft_megplanar(cfg, data);
% cfg = [];
% cfg.combinemethod = 'sum';
% data = ft_combineplanar(cfg, dataPlanar);


% Logical mask to find epochs matching your criteria
trial_criteria_left               = (data.trialinfo(:,2) == 4) | ...
                                    (data.trialinfo(:,2) == 5) | ...
                                    (data.trialinfo(:,2) == 6) | ...
                                    (data.trialinfo(:,2) == 7) | ...
                                    (data.trialinfo(:,2) == 8);

trial_criteria_right              = (data.trialinfo(:,2) == 1) | ...
                                    (data.trialinfo(:,2) == 2) | ...
                                    (data.trialinfo(:,2) == 3) | ...
                                    (data.trialinfo(:,2) == 9) | ...
                                    (data.trialinfo(:,2) == 10);

% Logical mask to find epochs without NaNs
has_no_nans                       = cellfun(@(x) ~any(isnan(x(:))), data.trial)';

% Combine both criteria
valid_trialsLeft                  = find(trial_criteria_left & has_no_nans);
valid_trialsRight                 = find(trial_criteria_right & has_no_nans);

% Select Left trials
cfg                               = [];
cfg.latency                       = [-0.5 1.5];
cfg.trials                        = valid_trialsLeft;
dataLeft                          = ft_selectdata(cfg, data);
cfg.trials                        = valid_trialsRight;
dataRight                         = ft_selectdata(cfg, data);



cfg = [];
cfg.method = 'mtmfft';
cfg.taper = 'hanning';
cfg.foilim = [5 150];
spectrum_clean = ft_freqanalysis(cfg, dataRight);
% spectrum   = ft_freqanalysis(cfg, data);

cfg = [];
% cfg.ylim = [0 0.5*1e-27];
ft_singleplotER(cfg, spectrum_clean); 
ft_singleplotER(cfg, spectrum);  


%% Visualize spectra by epochs and conditions
% Sensors
right_sensors               = {'AG001', 'AG002', 'AG007', 'AG008', 'AG020', 'AG022', 'AG023', ...
                               'AG024', 'AG034', 'AG036', 'AG050', 'AG055', 'AG065', 'AG066', ...
                               'AG098', 'AG103'};
left_sensors                = {'AG013', 'AG014', 'AG015', 'AG016', 'AG023', 'AG025', 'AG026', ...
                               'AG027', 'AG028', 'AG041', 'AG042', 'AG043', 'AG059', 'AG060', ...
                               'AG066', 'AG092'};

right_indices               = find(ismember(dataLeft.label, right_sensors));
left_indices                = find(ismember(dataLeft.label, left_sensors));

ff                          = figure;
tOnset                      = [-0.5 0.0 0.5 1  ];
tOffset                     = [ 0   0.3 1   1.5];
titleDict                   = {'Fixation', 'StimOn', 'Early Delay', 'Late Delay'};

for ii                      = 1:4
    subplot(2, 2, ii)
    if ii == 1
        sgtitle('Fixation')
    elseif ii == 2
        sgtitle('StimOn')
    elseif ii == 3
        sgtitle('Early Delay')
    elseif ii == 4
        sgtitle('Late Delay')
    end
    % title(titleDict{ii})
    cfg = [];
    cfg.latency = [tOnset(ii) tOffset(ii)];
    dataLeftSelected = ft_selectdata(cfg, dataLeft);
    dataRightSelected = ft_selectdata(cfg, dataRight);

    cfg = [];
    cfg.method = 'mtmfft';
    cfg.taper = 'hanning';
    % cfg.foilim = [0 150];
    cfg.foi = 2:0.5:150;
    spectrumLeft = ft_freqanalysis(cfg, dataLeftSelected);
    spectrumRight = ft_freqanalysis(cfg, dataRightSelected);
    
    ipsiSpectrum = (mean(spectrumLeft.powspctrm(left_indices, :), 1, 'omitnan') + ...
                    mean(spectrumRight.powspctrm(right_indices, :), 1, 'omitnan')) ./ 2;
    contraSpectrum = (mean(spectrumLeft.powspctrm(right_indices, :), 1, 'omitnan') + ...
                    mean(spectrumRight.powspctrm(left_indices, :), 1, 'omitnan')) ./ 2;
    ipsiSpectrum = log(ipsiSpectrum);
    contraSpectrum = log(contraSpectrum);
    semilogx(spectrumLeft.freq, ipsiSpectrum, 'DisplayName', 'Ipsi', 'LineWidth', 2)
    hold on;
    semilogx(spectrumRight.freq, contraSpectrum, 'DisplayName', 'Contra', 'LineWidth', 2)
    custom_ticks = [2, 5, 10, 20, 30, 40, 50, 100, 150];
    xticks(custom_ticks);
    
    xticklabels(cellstr(num2str(custom_ticks')));
    xlim([5 150])
    legend;
    xlabel('Frequency (Hz)')
    ylabel('Power (dB)')
    % ylim([-290 -265])
    clearvars dataLeftSelected dataRightSelected spectrumLeft spectrumRight;
end
%%
cfg = [];
cfg.method    = 'wavelet';
cfg.foi       = 2:1:150; % 1 Hz resolution
cfg.width     = 5; % Cycles per frequency (adjusts window: 5/frequency)
cfg.toi       = -0.5:0.02:1.5; % 50 ms steps (customize to epoch limits)
cfg.pad       = 'maxperlen'; % Minimize edge effects
TFRLeft = ft_freqanalysis(cfg, dataLeft);
TFRRight = ft_freqanalysis(cfg, dataRight);

cfg                               = [];
cfg.parameter                     = 'powspctrm';  % Specify the parameter to operate on
cfg.operation                     = '(x1-x2)/(x1+x2)';
TFRdiff                           = ft_math(cfg, TFRLeft, TFRRight);

load('NYUKIT_helmet.mat')
cfg = [];
cfg.layout = lay;
cfg.xlim = [-0.5 1.5];
cfg.ylim = [4 40];
ft_multiplotTFR(cfg, TFRdiff);