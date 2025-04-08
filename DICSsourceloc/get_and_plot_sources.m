function  [sourcePre, sourcePost]= get_and_plot_sources(freqPre, freqPost, grid, vol, segmentedmri)

cfg = [];
cfg.frequency    = 11;
cfg.method       = 'dics';
cfg.projectnoise = 'yes';
cfg.grid         = grid;
cfg.vol          = vol;
cfg.lambda       = 0;
sourcePre  = sourceanalysis(cfg, freqPre );
sourcePost = sourceanalysis(cfg, freqPost);


%compare pre and post stimulus interval
sourceDiff = sourcePost;
sourceDiff.avg.pow = (sourcePost.avg.pow - sourcePre.avg.pow) ./ sourcePre.avg.pow;
cfg = [];
cfg.downsample=2;
sourceDiffInt = sourceinterpolate(cfg, sourceDiff,segmentedmri);
%sourcePostInt= sourceinterpolate(cfg, sourcePost, segmentedmri);




%plot sources
cfg = [];
cfg.method        = 'ortho';
cfg.funparameter  = 'avg.pow';
cfg.maskparameter = cfg.funparameter;
cfg.interactive='yes';
cfg.opacitymap    = 'rampup'; 
figure
sourceplot(cfg, sourceDiffInt);
%sourceplot(cfg, sourcePostInt);