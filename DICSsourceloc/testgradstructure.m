cfg.grad=ftdata.grad
cfg.dip.pos=[0.01 0.01 0.01];
cfg.dip.mom=[1 0 0];
cfg.vol.r=5;
cfg.vol.o=[0 0 0];
cfg.dip.signal=[ 0 0 0 1 1 1 0 0 0];
data=dipolesimulation(cfg);



figure;
cfgplot=[];
avg=timelockanalysis(cfgplot, data);
topoplotER(cfgplot, avg)



cfg.grad=righttrials.grad
cfg.dip.pos=[0.01 0.01 0.01];
cfg.dip.mom=[-1 0 0];
cfg.vol.r=5;
cfg.vol.o=[0 0 0];
cfg.dip.signal=[ 0 0 0 1 1 1 0 0 0];
data=dipolesimulation(cfg);


figure;
cfgplot=[];
avg=timelockanalysis(cfgplot, data);
topoplotER(cfgplot, avg)


cfg.grad=righttrials.grad
cfg.dip.pos=[.1 .1 .1];
cfg.dip.mom=[0 1 0];
cfg.vol.r=5;
cfg.vol.o=[0 0 0];
cfg.dip.signal=[ 0 0 0 1 1 1 0 0 0];
data=dipolesimulation(cfg);


figure;
cfgplot=[];
avg=timelockanalysis(cfgplot, data);
topoplotER(cfgplot, avg)


cfg.grad=righttrials.grad
cfg.dip.pos=[.1 .1 .1];
cfg.dip.mom=[0 -1 0];
cfg.vol.r=5;
cfg.vol.o=[0 0 0];
cfg.dip.signal=[ 0 0 0 1 1 1 0 0 0];
data=dipolesimulation(cfg);


figure;
cfgplot=[];
avg=timelockanalysis(cfgplot, data);
topoplotER(cfgplot, avg)


cfg.grad=righttrials.grad
cfg.dip.pos=[.1 .1 .1];
cfg.dip.mom=[0 0 1];
cfg.vol.r=5;
cfg.vol.o=[0 0 0];
cfg.dip.signal=[ 0 0 0 1 1 1 0 0 0];
data=dipolesimulation(cfg)


figure;
cfgplot=[];
avg=timelockanalysis(cfgplot, data);
topoplotER(cfgplot, avg)

cfg.grad=righttrials.grad
cfg.dip.pos=[.1 .1 .1];
cfg.dip.mom=[0 0 -1];
cfg.vol.r=5;
cfg.vol.o=[0 0 0];
cfg.dip.signal=[ 0 0 0 1 1 1 0 0 0];
data=dipolesimulation(cfg);



figure;
cfgplot=[];
avg=timelockanalysis(cfgplot, data);
topoplotER(cfgplot, avg)






