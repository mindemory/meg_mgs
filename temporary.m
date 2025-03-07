cfg = [];
cfg.avgoverrpt = 'yes';
TFRleft = ft_freqdescriptives(cfg, TFRleft_power);
TFRright = ft_freqdescriptives(cfg, TFRright_power);

cfg = [];
cfg.parameter = 'powspctrm'; % Specify the parameter to operate on
cfg.operation = '(10^(x1/10) - 10^(x2/10)) / (10^(x1/10) + 10^(x2/10))';
TFRdiff = ft_math(cfg, TFRleft, TFRright);

cfg = [];
cfg.layout = lay;
cfg.ylim = [125 175];
cfg.colormap = '*RdBu';
ft_multiplotTFR(cfg, TFRdiff);

figure(); 
plot(TFRleft.freq, squeeze(mean(TFRleft.powspctrm, [1, 3], 'omitnan')))
%cfg = [];
% cfg.baseline = [-1.0 0];
% cfg.baselinetype = 'absolute';
% cfg.maskparameter = 0;
cfg.xlim = 0:0.5:2;
% cfg.xlim = [0 0.5];
cfg.ylim = [65 115];
cfg.marker = 'on';
cfg.layout = lay;
cfg.colormap = '*RdBu';
ft_topoplotTFR(cfg, TFRdiff)


% Visualize power spectrum
pwrspctrm = mean(TFRleft_power, [1, 2, 4], 'omitnan');
figure();
plot(TFRleft_power.freq, pwrspctrm);
