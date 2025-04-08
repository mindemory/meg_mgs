function [h] = mri_to_yokogawa_headcoordinates(abovenas, leftforehead, rtforehead)

% HEADCOORDINATES returns the homogenous coordinate transformation matrix
% that converts the specified fiducials in any coordinate system (e.g. MRI)
% into the rotated and translated headccordinate system.
%
% [h] = headcoordinates(abovenas, leftforehead, rtforehead) 
% See also WARPING, WARP3D

% Copyright (C) 2003 Robert Oostenveld
%
%
% $Log: headcoordinates.m,v $
% Revision 1.1  2004/09/27 16:00:04  roboos
% initial submission
%
%Modified by Sangi for yokogawa coordinates 11/2009


% ensure that they are row vectors
leftforehead = leftforehead(:)';
rtforehead = rtforehead(:)';
abovenas = abovenas(:)';


origin = [leftforehead+rtforehead]/2*10; %in between left and right, and convert transformation to mm, which is what MRI is in
dirx = rtforehead-leftforehead;
dirx = dirx/norm(dirx);
diry=abovenas-origin;
diry= diry/norm(diry);
dirz = cross(dirx,leftforehead-rtforehead);
dirz = dirz/norm(dirz);




% compute the rotation matrix
rot = eye(4);
rot(1:3,1:3) = inv(eye(3) / [dirx; diry; dirz]);
% compute the translation matrix
tra = eye(4);
tra(1:4,4)   = [-origin(:); 1];
% compute the full homogenous transformation matrix from these two
h = rot * tra;
