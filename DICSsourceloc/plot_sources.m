function  []= plot_sources(sourcePre, sourcePost)


%compare pre and post stimulus interval
sourceDiff = sourcePost;
sourceDiff.avg.pow = (sourcePost.avg.pow - sourcePre.avg.pow) ./ sourcePre.avg.pow;
cfg = [];
cfg.downsample = 2;
sourceDiffInt = sourceinterpolate(cfg, sourceDiff , segmentedmri);




%plot sources
cfg = [];
cfg.method        = 'ortho';
cfg.funparameter  = 'avg.pow';
cfg.maskparameter = cfg.funparameter;
cfg.interactive='yes';
cfg.opacitymap    = 'rampup';
figure
sourceplot(cfg, sourceDiffInt);