function [ realignedset ] = transform_set_to_new_grad_positions(datatorealign,templateset,radius, origin )
cfg.vol.o=origin;
cfg.vol.r=radius;
cfg.inwardshift=2.5;
cfg.template{1}=templateset;
realignedset=megrealign(cfg, datatorealign);


end

