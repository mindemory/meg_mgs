function [pt] = get_mri_fiducials(cfg, mri)


fieldtripdefs

cfg = checkconfig(cfg, 'trackconfig', 'on');

% check if the input data is valid for this function
mri = checkdata(mri, 'datatype', 'volume', 'feedback', 'yes');

% set the defaults
if ~isfield(cfg, 'fiducial'),  cfg.fiducial = [];         end
if ~isfield(cfg, 'parameter'), cfg.parameter = 'anatomy'; end
if ~isfield(cfg, 'clim'),      cfg.clim      = [];        end

if ~isfield(cfg, 'method')
  if ~isempty(cfg.fiducial)
    cfg.method = 'realignfiducial';
  else
    cfg.method = 'interactive';
  end
end

% select the parameter that should be displayed
cfg.parameter = parameterselection(cfg.parameter, mri);
if iscell(cfg.parameter)
  cfg.parameter = cfg.parameter{1};
end

switch cfg.method
  case 'realignfiducial'
    % do nothing
  case 'interactive'
    dat = getsubfield(mri, cfg.parameter);
    pt=zeros(5, 3);
    x = 1:mri.dim(1);
    y = 1:mri.dim(2);
    z = 1:mri.dim(3);
     xc = round(mri.dim(1)/2);
     yc = round(mri.dim(2)/2);
     zc = round(mri.dim(3)/2);




    while(1) % break when 'q' is pressed
      fprintf('============================================================\n');
      fprintf('click with mouse button to reslice the display to a new position\n');
      fprintf('press 1/2/3/4/5 on keyboard to record the current position as fiducial location 1:lpa, 2:rpa, 3:center forehead, 4:upperleft forehead, 5: upperright forehead\n');
      fprintf('press q on keyboard to quit interactive mode\n');
      xc = round(xc);
      yc = round(yc);
      zc = round(zc);
      volplot(x, y, z, dat, [xc yc zc], cfg.clim);
      drawnow;
      try, [d1, d2, key] = ginput(1); catch, key='q'; end
      if key=='q'
        break;
      elseif key=='1'
        pt(1, :, :) = [xc yc zc]./10;
      elseif key=='2'
        pt(2, :, :) = [xc yc zc]./10;
      elseif key=='3'
        pt(3,:, :)= [xc yc zc]./10;
      elseif key=='4'
        pt(4,:, :)= [xc yc zc]./10;
      elseif key=='5'
        pt(5, :, :)=[xc yc zc]./10;
      else
        % update the view to a new position
        l1 = get(get(gca, 'xlabel'), 'string');
        l2 = get(get(gca, 'ylabel'), 'string');
        switch l1,
          case 'i'
            xc = d1;
          case 'j'
            yc = d1;
          case 'k'
            zc = d1;
        end
        switch l2,
          case 'i'
            xc = d2;
          case 'j'
            yc = d2;
          case 'k'
            zc = d2;
        end
      end
      if sum(pt(1, :)>0), fprintf('lpa = [%f %f %f]\n', pt(1, :, :)); else fprintf('pt1 = undefined\n'); end
      if sum(pt(2, :)>0), fprintf('rpa = [%f %f %f]\n', pt(2, :, :)); else fprintf('pt2 = undefined\n'); end
      if sum(pt(3, :)>0), fprintf('center forehead  = [%f %f %f]\n', pt(3, :, :)); else fprintf('pt3 = undefined\n'); end
      if sum(pt(4, :)>0), fprintf('upper left forehead = [%f %f %f]\n', pt(4, :, :)); else fprintf('pt4 = undefined\n'); end
      if sum(pt(5, :)>0), fprintf('upper right forehead = [%f %f %f]\n', pt(5, :, :)); else fprintf('pt5 = undefined\n'); end
      
    end

    cfg.fiducial.pt=pt;
    

      
% origin = (lpa+rpa)/2
% dirx = rpa-lpa
% dirx = dirx/norm(dirx)
% diry=nas-origin
% diry= diry/norm(diry)
% dirz = cross(dirx, diry)
% dirz = dirz/norm(dirz)



  otherwise
    error('unsupported method');
end



% get the output cfg
cfg = checkconfig(cfg, 'trackconfig', 'off', 'checksize', 'yes'); 

% add version information to the configuration
try
  % get the full name of the function
  cfg.version.name = mfilename('fullpath');
catch
  % required for compatibility with Matlab versions prior to release 13 (6.5)
  [st, i] = dbstack;
  cfg.version.name = st(i);
end
cfg.version.id = '$Id: volumerealign.m,v 1.12 2009/07/31 13:43:36 jansch Exp $';

% remember the configuration
mri.cfg = cfg;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% helper function to show three orthogonal slices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function volplot(x, y, z, dat, c, cscale);
xi = c(1);
yi = c(2);
zi = c(3);

% manual color scaling of anatomy data is usefull in case of some pixel noise
if nargin<6 || isempty(cscale)
  cmin = min(dat(:));
  cmax = max(dat(:));
else
  cmin = cscale(1);
  cmax = cscale(2);
end

h1 = subplot(2,2,1);
h2 = subplot(2,2,2);
h3 = subplot(2,2,3);

subplot(h1);
imagesc(x, z, squeeze(dat(:,yi,:))'); set(gca, 'ydir', 'normal')
axis equal; axis tight;
xlabel('i'); ylabel('k');
caxis([cmin cmax]);
crosshair([x(xi) z(zi)], 'color', 'yellow');

subplot(h2);
imagesc(y, z, squeeze(dat(xi,:,:))'); set(gca, 'ydir', 'normal')
axis equal; axis tight;
xlabel('j'); ylabel('k');
caxis([cmin cmax]);
crosshair([y(yi) z(zi)], 'color', 'yellow');

subplot(h3);
imagesc(x, y, squeeze(dat(:,:,zi))'); set(gca, 'ydir', 'normal')
axis equal; axis tight;
xlabel('i'); ylabel('j');
caxis([cmin cmax]);
crosshair([x(xi) y(yi)], 'color', 'yellow');

colormap gray
