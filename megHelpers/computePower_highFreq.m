function powHigh                               = computePower_highFreq(data)
% Created by Mrugank (10/04/2023)
% Creates a complex fourier spectrogram and extract absolute power, ITC and
% phase information.


time_points                                    = linspace(min(data.time{1}), max(data.time{1}), 200);
% cfg                                            = [];
% cfg.resamplefs                                 = 250;
% cfg.demean                                     = 'no';
% cfg.detrend                                    = 'no';
% data                                           = ft_resampledata(cfg, data);

cfg = [];
cfg.dftfilter = 'yes';
cfg.dftfreq = [60 120];
cfg.dftreplace = 'neighbour';
cfg.dftbandwidth = [1 2];
data = ft_preprocessing(cfg, data);
% Compute high frequency power
cfg                                            = [];
cfg.method                                     = 'mtmconvol';
cfg.output                                     = 'pow';
cfg.taper                                      = 'dpss';
cfg.foi                                        = 40:2:120;
cfg.t_ftimwin                                  = ones(length(cfg.foi),1) * 0.4;
cfg.tapsmofrq                                  = ones(length(cfg.foi),1) * 7.5;
% cfg.toi                                        = 'all';
cfg.toi                                        = time_points;
% cfg.pad                                        = 'maxperlen';
cfg.keeptrials                                 = 'yes';
cfg.polyremoval                                = 1;
powHigh                                        = ft_freqanalysis(cfg, data);
powHigh.powspctrm                              = 10*log10(powHigh.powspctrm);

% % Define frequencies, cycles and timepoints
% frequencies                                    = linspace(40, 120, 41);
% cycles                                         = linspace(15, 30, numel(frequencies));
% time_points                                    = linspace(min(data.time{1}), max(data.time{1}), 200);
% 
% % Compute the full Fourier spectrogram for all trials
% cfg                                            = [];
% cfg.method                                     = 'wavelet';
% cfg.output                                     = 'fourier';
% cfg.foi                                        = frequencies;
% cfg.width                                      = cycles;
% cfg.toi                                        = time_points;
% cfg.keeptrials                                 = 'yes';
% cfg.polyremoval                                = 1;
% TFR_fourier                                    = ft_freqanalysis(cfg, data);
% 
% % Power
% powHigh                                      = TFR_fourier;
% powHigh.powspctrm                            = 10*log10(abs(TFR_fourier.fourierspctrm).^2);
% powHigh                                      = rmfield(powHigh, 'fourierspctrm');
% powHigh                                      = rmfield(powHigh, 'cumtapcnt');

% % Define frequencies, cycles and timepoints
% frequencies                                    = linspace(2, 40, 53);
% cycles                                         = linspace(4, 15, numel(frequencies));
% time_points                                    = linspace(min(data.time{1}), max(data.time{1}), 200);
% 
% % Compute the full Fourier spectrogram for all trials
% cfg                                            = [];
% cfg.method                                     = 'wavelet';
% cfg.output                                     = 'fourier';
% cfg.foi                                        = frequencies;
% cfg.width                                      = cycles;
% cfg.toi                                        = time_points;
% cfg.keeptrials                                 = 'yes';
% cfg.polyremoval                                = 1;
% TFR_fourier                                    = ft_freqanalysis(cfg, data);
% 
% % Power
% TFR_power                                      = TFR_fourier;
% TFR_power.powspctrm                            = 10*log10(abs(TFR_fourier.fourierspctrm).^2);
% TFR_power                                      = rmfield(TFR_power, 'fourierspctrm');
% TFR_power                                      = rmfield(TFR_power, 'cumtapcnt');
% if base_corr                                   == 1
%     baseline_time_indices                      = find(time_points >= -1 & time_points < 0);
%     baseline_mean                              = mean(TFR_power.powspctrm(:, :, :, baseline_time_indices), 4, 'omitnan');
%     baseline_mean_expanded                     = repmat(baseline_mean, [1, 1, 1, size(TFR_power.powspctrm, 4)]);
%     TFR_power.powspctrm                        = TFR_power.powspctrm - baseline_mean_expanded;
% end

end