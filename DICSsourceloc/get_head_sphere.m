function headsphere= get_head_sphere(xlfile);
%xlfile- output from MEG160 sfp file saved in Excel
%Markers 3-5 (rows 3-5 in output) are fiducials on the forehead which
%should be marked with vitamin E caplets in MRI.  3- in center of forehead,
%above nasion, 4- on left of forehead 5- on right of forehead
numeric=xlsread(xlfile);
headsphere=numeric(6:end, :)./10; %convert to cm as sensors are
% headsphere(:,1)=-headsphere(:,1);
% headsphere(:,2)=-headsphere(:,2);