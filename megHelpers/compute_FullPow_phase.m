function outputTFR                             = compute_FullPow_phase(data, output)
% Created by Mrugank (10/04/2023)
% Creates a complex fourier spectrogram and extract absolute power, ITC and
% phase information.
if nargin < 2
    % Valid arguments are 'lowFreqPhase' or 'highFreqPow'
    output                                     = 'lowFreqPhase'; 
end


if strcmp(output, 'lowFreqPhase') 
    % Define frequencies, cycles and timepoints
    frequencies                                = linspace(2, 40, 53);
    cycles                                     = linspace(4, 15, numel(frequencies));
    time_points                                = linspace(min(data.time{1}), max(data.time{1}), 200);
    
    % Compute the full Fourier spectrogram for all trials
    cfg                                        = [];
    cfg.method                                 = 'wavelet';
    cfg.output                                 = 'fourier';
    cfg.foi                                    = frequencies;
    cfg.width                                  = cycles;
    cfg.toi                                    = time_points;
    cfg.keeptrials                             = 'yes';
    cfg.polyremoval                            = -1;
    TFR_fourier                                = ft_freqanalysis(cfg, data);
    
    % Compute phase
    TFR_phase                                  = TFR_fourier;
    TFR_phase.phaseangle                       = angle(TFR_fourier.fourierspctrm);
    TFR_phase                                  = rmfield(TFR_phase, 'fourierspctrm');
    TFR_phase                                  = rmfield(TFR_phase, 'cumtapcnt');

    outputTFR                                  = TFR_phase;

elseif strcmp(output, 'highFreqPow')
    % Remove line noise
    % cfg                                        = [];
    % cfg.channel                                = 'all';
    % cfg.dftfilter                              = 'yes';
    % cfg.dftfreq                                = [60 120 180];
    % data                                       = ft_preprocessing(cfg, data);

    % frequencies                                = [41:55 65:1:115 125:1:175];
    frequencies                                = 41:180;
    time_points                                = linspace(min(data.time{1}), max(data.time{1}), 200);

    % Compute the full Fourier spectrogram for all trials
    cfg                                        = [];
    cfg.method                                 = 'mtmconvol';
    cfg.output                                 = 'pow';
    cfg.taper                                  = 'dpss';
    % cfg.taper                                  = 'dpss';
    cfg.tapsmofrq                              = 3;
    cfg.foi                                    = frequencies;
    % cfg.width                                  = 15;              
    cfg.t_ftimwin                              = ones(length(cfg.foi), 1) * 0.5;
    cfg.toi                                    = time_points;
    cfg.keeptrials                             = 'yes';
    cfg.polyremoval                            = -1;
    TFR_power                                  = ft_freqanalysis(cfg, data);
    TFR_power.powspctrm                        = 10*log10(TFR_power.powspctrm);
    % TFR_fourier                                = ft_freqanalysis(cfg, data);
    % TFR_power                                  = TFR_fourier;
    % TFR_power.powspctrm                        = 10*log10(abs(TFR_fourier.fourierspctrm).^2);
    % TFR_power                                  = rmfield(TFR_power, 'fourierspctrm');
    % TFR_power                                  = rmfield(TFR_power, 'cumtapcnt');
    outputTFR                                  = TFR_power;
end
end