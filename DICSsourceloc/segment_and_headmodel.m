%addpath(genpath('/home/sangi/matlab/fieldtrip_recent/'));
addpath(genpath('/home/sangi/matlab/fieldtrip/'))
%addpath(genpath('/usr/local/toolbox/spm2'))
 addpath(genpath('/home/sangi/matlab/spm8/'))
addpath(genpath('/home/sangi/matlab/yokogawa/'))
 addpath(genpath('/home/sangi/matlab/MGS_scripts_MEG/'))
addpath(genpath('/home/sangi/matlab/MGS_scripts/'))
%cd /deathstar/data/MEG_MGS/01MGS/hi_res/analyze


%mrifilename='Jason_ANALYZE+02+t1_mprage_sag.img';
%segfilename=mrifilename;

%mrifilename='Jason_anat+18+t1_mprage_sag.img'; %this file has one fiducial on the right 


%Sub 1
%mrifilename='FourpointStructural.img'; %this file has one fiducial on the
%right and is flipped in prep for segmentation (older?)
mrifilename='Structural.img'; %original flipped in preparation for segmentation 
segfilename='Structural.img';%gray, white, and csf segmentations should be saved with this suffix, with prefixes c1, c2, and c3 respectively


%Sub 6
%mrifilename='Structural.img'; %original flipped in preparation for segmentation 
%segfilename='ibStructural.img';
%segfilename='Structural.img';%gray, white, and csf segmentations should be saved with this suffix, with prefixes c1, c2, and c3 respectively



mri=read_mri(mrifilename); %has 3 fiducial points
mri.anatomy=log(mri.anatomy);


cfg=[];
cfg.method='interactive';
['Right is to the right in coronal image and to the right in axial image']
fidmri=get_mri_fiducials(cfg, mri);


cfg = [];

mri=read_mri(['c1' segfilename]);
segmentedmri.gray=mri.anatomy;
mri=read_mri(['c2' segfilename]);
segmentedmri.white=mri.anatomy;

segmentedmri.dim=mri.dim;

mri=read_mri(['c3' segfilename]);
segmentedmri.csf=mri.anatomy;
mri=read_mri(mrifilename);
segmentedmri.anatomy=mri.anatomy;
segmentedmri.transform=eye(4); %change?

% sourceplot(cfg,segmentedmriF); %only mri
% figure
% cfg.funparameter = 'gray';
% sourceplot(cfg,segmentedmriF); %segmented gray matter on top
% figure
% cfg.funparameter = 'white';
% sourceplot(cfg,segmentedmriF); %segmented white matter on top
% figure
% cfg.funparameter = 'csf';
% sourceplot(cfg,segmentedmriF); %segmented csf matter on top


cfg=[];
cfg.sourceunits='cm';
vol = prepare_singleshell(cfg, segmentedmri);
