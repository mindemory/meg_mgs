
function [grid, data]=prepare_leadfield_dipolefit(data, fidmri, fidmeg, vol, channelsel)


[x, cmat, jacobian, rms, transpoints] = calc_transform_lm_seq(fidmri, fidmeg, zeros(1, 6))
data.grad=transform_sens(cmat, data.grad);




cfg = [];
cfg.grad            = data.grad;
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


[grid] = prepare_leadfield(cfg, data);
scatter3(vol.bnd.pnt(:,1), vol.bnd.pnt(:,2), vol.bnd.pnt(:,3));
hold on;
scatter3(fidmri(:,1), fidmri(:, 2), fidmri(:, 3), 'r');

scatter3(grid.pos(grid.inside,1), grid.pos(grid.inside,2), grid.pos(grid.inside,3), 'g')
scatter3(grid.pos(grid.outside,1), grid.pos(grid.outside,2), grid.pos(grid.outside,3), 'k')



