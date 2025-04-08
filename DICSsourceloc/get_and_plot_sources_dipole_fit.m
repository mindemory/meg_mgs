function  [dipPos]= get_and_plot_sources_dipole_fit(fidmri,fidmeg, data, grid, vol, latency)




cfgave=[];
cfgave.latency=latency;
%cfgave.baseline=baseline;

[x, cmat, jacobian, rms, transpoints] = calc_transform_lm_seq(fidmri, fidmeg, zeros(1, 6))
grad=transform_sens(cmat, data.grad);


avg=timelockanalysis(cfgave, data);

avg.grad=grad;
cfg.grad=grad;
cfg.numdipoles=1;
cfg.grid=grid;
cfg.vol=vol;
dipPos=dipolefitting(cfg, avg);
% cfg=[];
% cfg.downsample=2;
% sourcePostInt= sourceinterpolate(cfg, sourcePost, segmentedmri);
% %plot sources
% cfg = [];
% cfg.method        = 'ortho';
% cfg.funparameter  = 'avg.pow';
% cfg.maskparameter = cfg.funparameter;
% cfg.interactive='yes';
% cfg.opacitymap    = 'rampup'; 
% figure
% %sourceplot(cfg, sourceDiffInt);
% sourceplot(cfg, sourcePostInt);