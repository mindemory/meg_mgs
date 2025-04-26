function powHigh                               = computePower_highFreq(data)
% Created by Mrugank (10/04/2023)
% Creates a complex fourier spectrogram and extract absolute power, ITC and
% phase information.


time_points                                    = linspace(min(data.time{1}), max(data.time{1}), 200);

% Compute high frequency power
cfg                                            = [];
cfg.method                                     = 'mtmconvol';
cfg.output                                     = 'pow';
cfg.taper                                      = 'dpss';
cfg.foi                                        = 40:2:150;
cfg.t_ftimwin                                  = ones(length(cfg.foi),1) * 0.4;
cfg.tapsmofrq                                  = ones(length(cfg.foi),1) * 7.5;
% cfg.toi                                        = 'all';
cfg.toi                                        = time_points;
% cfg.pad                                        = 'maxperlen';
cfg.keeptrials                                 = 'yes';
cfg.polyremoval                                = 1;
powHigh                                        = ft_freqanalysis(cfg, data);
powHigh.powspctrm                              = 10*log10(powHigh.powspctrm);

end