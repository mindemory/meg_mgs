% function [TFR_power, TFR_itc, TFR_phase]       = compute_TFRs(data, base_corr)
function TFR_power = compute_TFRs(data, base_corr)
% Created by Mrugank (10/04/2023)
% Creates a complex fourier spectrogram and extract absolute power, ITC and
% phase information.
if nargin < 2
    base_corr                                  = 0;
end

% Define frequencies, cycles and timepoints
frequencies                                    = linspace(2, 40, 53);
cycles                                         = linspace(4, 15, numel(frequencies));
time_points                                    = linspace(min(data.time{1}), max(data.time{1}), 200);

% Compute the full Fourier spectrogram for all trials
cfg                                            = [];
cfg.method                                     = 'wavelet';
% cfg.output                                     = 'fourier';
cfg.output                                     = 'pow';
cfg.foi                                        = frequencies;
cfg.width                                      = cycles;
cfg.toi                                        = time_points;
% cfg.keeptrials                                 = 'yes';
cfg.keeptrials                                 = 'no';
cfg.polyremoval                                = 1;
TFR_fourier                                    = ft_freqanalysis(cfg, data);

% Power
TFR_power                                      = TFR_fourier;
% TFR_power.powspctrm                            = 10*log10(abs(TFR_fourier.fourierspctrm).^2);
% TFR_power                                      = rmfield(TFR_power, 'fourierspctrm');
% TFR_power                                      = rmfield(TFR_power, 'cumtapcnt');
if base_corr                                   == 1
    baseline_time_indices                      = find(time_points >= -1 & time_points < 0);
    baseline_mean                              = mean(TFR_power.powspctrm(:, :, :, baseline_time_indices), 4, 'omitnan');
    baseline_mean_expanded                     = repmat(baseline_mean, [1, 1, 1, size(TFR_power.powspctrm, 4)]);
    TFR_power.powspctrm                        = TFR_power.powspctrm - baseline_mean_expanded;
end

% ITC
% itc_complex                                    = mean(exp(1i * angle(TFR_fourier.fourierspctrm)), 1);
% itc                                            = abs(itc_complex);
% TFR_itc                                        = TFR_fourier;
% TFR_itc.itcspctrm                              = squeeze(itc);  
% TFR_itc                                        = rmfield(TFR_itc, 'fourierspctrm'); 
% TFR_itc                                        = rmfield(TFR_itc, 'cumtapcnt'); 
% smoothing_window_time                          = 3; 
% smoothed_itc                                   = movmean(TFR_itc.itcspctrm, smoothing_window_time, 2);  
% TFR_itc.itcspctrm                              = smoothed_itc;
% smoothing_window_freq                          = 3; 
% smoothed_itc                                   = movmean(TFR_itc.itcspctrm, smoothing_window_freq, 1);  
% TFR_itc.itcspctrm                              = smoothed_itc;
% 
% % Phase
% TFR_phase                                      = TFR_fourier;
% TFR_phase.phaseangle                           = angle(TFR_fourier.fourierspctrm);
% TFR_phase                                      = rmfield(TFR_phase, 'fourierspctrm');
% TFR_phase                                      = rmfield(TFR_phase, 'cumtapcnt');
end