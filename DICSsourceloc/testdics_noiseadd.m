% [x, cmat, jacobian, rms, transpoints] = calc_transform_lm_seq(fidmri, fidmeg, zeros(1, 6))
% gradtrans=transform_sens(cmat, allleft.grad);
cfg=[];

cfg.grad=lefttrials.grad;
cfg.dip.pos=[mean(vol.bnd.pnt(:,1)) mean(vol.bnd.pnt(:,2)) mean(vol.bnd.pnt(:,3))];

cfg.dip.mom=[0 1 0];
cfg.triallength=7;
cfg.dip.fsample=500;
cfg.dip.phase=0;
cfg.vol=vol;

cfg.dip.frequency=10;
cfg.ntrials=size(lefttrials.cfg.trl, 1);

trialtest=dipolesimulation(cfg);
trialtest.cfg.trl=lefttrials.cfg.trl;

cfg=[];
cfg.grad=righttrials.grad;
cfg.dip.pos=[mean(vol.bnd.pnt(:,1)) mean(vol.bnd.pnt(:,2)) mean(vol.bnd.pnt(:,3))]
cfg.dip.mom=[0 1 0];
cfg.triallength=7;
cfg.dip.fsample=500;
cfg.vol=vol; 

cfg.dip.frequency=20;
cfg.dip.phase=0;
cfg.ntrials=size(righttrials.cfg.trl, 1);

trialbase=dipolesimulation(cfg);
                              
trialbase.cfg.trl=righttrials.cfg.trl;

for t=1:length(trialtest.trial)
    trialtest.trial{t}=trialtest.trial{t}+randn(size(trialtest.trial{t})).*(.00001*max(max(trialtest.trial{t})));
end

for t=1:length(trialbase.trial)
    trialbase.trial{t}=trialbase.trial{t}+randn(size(trialbase.trial{t})).*(.00001*max(max(trialbase.trial{t})));
end