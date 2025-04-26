% function [TFR_power, TFR_itc, TFR_phase]       = compute_TFRs(data, base_corr)
function TFR_fourier                           = compute_fullFourier(data)
% Created by Mrugank (10/04/2023)
% Creates a complex fourier spectrogram and extract absolute power, ITC and
% phase information.


% Define frequencies, cycles and timepoints
frequencies                                    = linspace(2, 40, 53);
cycles                                         = linspace(4, 15, numel(frequencies));
time_points                                    = linspace(min(data.time{1}), max(data.time{1}), 200);

% Compute the full Fourier spectrogram for all trials
cfg                                            = [];
cfg.method                                     = 'wavelet';
cfg.output                                     = 'fourier';
cfg.foi                                        = frequencies;
cfg.width                                      = cycles;
cfg.toi                                        = time_points;
cfg.keeptrials                                 = 'yes';
cfg.polyremoval                                = 1;
TFR_fourier                                    = ft_freqanalysis(cfg, data);
end