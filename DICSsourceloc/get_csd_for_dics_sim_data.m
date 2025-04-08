
function [freqPre, freqPost, grid]=get_csd_for_dics_sim_data(trialbase, trial, fidmri, fidmeg, vol, channelsel)


cfg = [];                                           
cfg.toilim = [.2 .7];                       
dataPre = redefinetrial(cfg, trialbase);
   
cfg.toilim = [.2 .7];                       
dataPost = redefinetrial(cfg, trial);


cfg = [];
cfg.method    = 'mtmfft';
cfg.output    = 'powandcsd';
cfg.tapsmofrq = 2;
cfg.foilim    = [10 10];
freqPre = freqanalysis(cfg, dataPre);

cfg = [];
cfg.method    = 'mtmfft';
cfg.output    = 'powandcsd';
cfg.tapsmofrq = 2;
cfg.foilim    = [10 10];
freqPost = freqanalysis(cfg, dataPost);

% 
%   [x, cmat, jacobian, rms, transpoints] = calc_transform_lm_seq(fidmri, fidmeg, zeros(1, 6))
%   freqPre.grad=transform_sens(cmat, freqPre.grad);
%   freqPost.grad=transform_sens(cmat, freqPost.grad);
%  

cfg = [];
cfg.grad            = freqPre.grad;
cfg.vol             = vol;
cfg.reducerank      = 2;
cfg.channel         = channelsel;
%cfg.grid.resolution = 2;


cfg.grid.xgrid=floor((min(vol.bnd.pnt(:,1)))):.5:ceil(max(vol.bnd.pnt(:,1)));
floor((min(vol.bnd.pnt(:,1))))
ceil(max(vol.bnd.pnt(:,1)))
floor((min(vol.bnd.pnt(:,1)))):ceil(max(vol.bnd.pnt(:,1)))


cfg.grid.ygrid=floor((min(vol.bnd.pnt(:,2)))):.5:ceil(max(vol.bnd.pnt(:,2)));
floor((min(vol.bnd.pnt(:,2))))
ceil(max(vol.bnd.pnt(:,2)))
floor((min(vol.bnd.pnt(:,2)))):ceil(max(vol.bnd.pnt(:,2)))


cfg.grid.zgrid=floor((min(vol.bnd.pnt(:,3)))):.5:ceil(max(vol.bnd.pnt(:,3)));
floor((min(vol.bnd.pnt(:,3)))) 
ceil(max(vol.bnd.pnt(:,3)));
floor((min(vol.bnd.pnt(:,3)))):ceil(max(vol.bnd.pnt(:,3)))


[grid] = prepare_leadfield(cfg, freqPre);
scatter3(vol.bnd.pnt(:,1), vol.bnd.pnt(:,2), vol.bnd.pnt(:,3));
hold on;
scatter3(fidmri(:,1), fidmri(:, 2), fidmri(:, 3), 'r');

scatter3(grid.pos(grid.inside,1), grid.pos(grid.inside,2), grid.pos(grid.inside,3), 'g')
scatter3(grid.pos(grid.outside,1), grid.pos(grid.outside,2), grid.pos(grid.outside,3), 'k')
scatter3(freqPre.grad.pnt(:,1), freqPre.grad.pnt(:, 2), freqPre.grad.pnt(:,3), 'b');


