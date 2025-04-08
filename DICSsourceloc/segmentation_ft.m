mri=read_mri('Jason_ANALYZE+02+t1_mprage_sag.img');




%mri = read_mri('Sangi+12+t1mprage.img');

cfg = [];
cfg.write        = 'no';
cfg.coordinates  = '';
% The Analyze coordinate system is defined by and used in the Analyze software developed by the Mayo Clinic (see also this pdf). The orientation is according to radiological conventions, and uses a left-handed coordinate system. The definition of the Analyze coordinate system is 
% 
% the x-axis goes from right to left
%  the y-axis goes from posterior to anterior
%  the z-axis goes from inferior to superior



cfg.template     = '/deathstar/usr/local-mac/toolbox/spm2/templates/T1.mnc'; 
[segmentedmri] = volumesegment(cfg, mri);


segmentedmriF = segmentedmri;
segmentedmriF.gray  = flipdim(flipdim(flipdim(segmentedmriF.gray,3),2),1);
segmentedmriF.white = flipdim(flipdim(flipdim(segmentedmriF.white,3),2),1);
segmentedmriF.csf   = flipdim(flipdim(flipdim(segmentedmriF.csf,3),2),1);


% 
% mri = read_mri('Jason_ANALYZE+02+t1_mprage_sag.img');
% segmentedmriF.transform = mri.transform;
% segmentedmriF.anatomy   = mri.anatomy;

figure
cfg = [];
sourceplot(cfg,segmentedmriF); %only mri
figure
cfg.funparameter = 'gray';
sourceplot(cfg,segmentedmriF); %segmented gray matter on top
figure
cfg.funparameter = 'white';
sourceplot(cfg,segmentedmriF); %segmented white matter on top
figure
cfg.funparameter = 'csf';
sourceplot(cfg,segmentedmriF); %segmented csf matter on top


cfg = [];
vol = prepare_singleshell(cfg, segmentedmriF);
