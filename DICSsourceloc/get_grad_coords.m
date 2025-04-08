function ft_dataout= get_grad_coords(ft_data, sqdfile, xlfile);
%xlfile- output from MEG160 pos file saved in Excel
%sqdfile-squid file from current set of runs


ft_dataout=ft_data;

hdr = read_yokogawa_header(sqdfile)


if isfield(hdr, 'orig')
  hdr = hdr.orig; % use the original header, not the FieldTrip header
end

handles    = definehandles;
isgrad     = (hdr.channel_info(:,2)==handles.AxialGradioMeter | hdr.channel_info(:,2)==handles.PlannerGradioMeter);



numeric=xlsread(xlfile);

grad.pnt(1:157, :)=numeric(1:157, 1:3)./10; %convert to cm as sensors are, in yokogawa2grad this was grad.pnt   = hdr.channel_info(isgrad,3:5)*100;    % cm
grad.pnt(158:314, :)=numeric(1:157, 4:6)./10;
grad.ori(1:157, :)= numeric(1:157, 7:9);

ori_1st=grad.ori; %in yokogawa2grad, this was ori_1st   = hdr.channel_info(find(isgrad),[6 8]);

% Get orientation from the 1st to 2nd coil for gradiometer
ori_1st_to_2nd   = hdr.channel_info(find(isgrad),[7 9]);
% polar to x,y,z coordinates
ori_1st_to_2nd = ...
  [sin(ori_1st_to_2nd(:,1)/180*pi).*cos(ori_1st_to_2nd(:,2)/180*pi) ...
  sin(ori_1st_to_2nd(:,1)/180*pi).*sin(ori_1st_to_2nd(:,2)/180*pi) ...
  cos(ori_1st_to_2nd(:,1)/180*pi)];
% Get baseline
baseline = hdr.channel_info(isgrad,size(hdr.channel_info,2));

% Define the location and orientation of 2nd coil
for i=1:sum(isgrad)
  if hdr.channel_info(i,2) == handles.AxialGradioMeter
    grad.pnt(i+sum(isgrad),:) = [grad.pnt(i,:)+ori_1st(i,:)*baseline(i)*100];
    grad.ori(i+sum(isgrad),:) = -ori_1st(i,:);
  elseif hdr.channel_info(i,2) == handles.PlannerGradioMeter
    grad.pnt(i+sum(isgrad),:) = [grad.pnt(i,:)+ori_1st_to_2nd(i,:)*baseline(i)*100];
    grad.ori(i+sum(isgrad),:) = ori_1st(i,:);
  end
end


% Define the pair of 1st and 2nd coils for each gradiometer
grad.tra = repmat(diag(ones(1,size(grad.pnt,1)/2),0),1,2);

% Make the matrix sparse to speed up the multiplication in the forward
% computation with the coil-leadfield matrix to get the channel leadfield
grad.tra = sparse(grad.tra);


tmp = hdr.channel_info(isgrad,1);
grad.label = cell(size(tmp, 1), 1);% cell-array containing strings, Nchan X 1 (This is one modification to yokogawa2grad.m -Sangi)
for i=1:size(tmp,1)
  grad.label{i} = num2str(tmp(i)+1);
end
grad.unit='cm';

ft_dataout.grad=grad;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% this defines some usefull constants
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function handles = definehandles;
handles.output = [];
handles.sqd_load_flag = false;
handles.mri_load_flag = false;
handles.NullChannel         = 0;
handles.MagnetoMeter        = 1;
handles.AxialGradioMeter    = 2;
handles.PlannerGradioMeter  = 3;
handles.RefferenceChannelMark = hex2dec('0100');
handles.RefferenceMagnetoMeter       = bitor( handles.RefferenceChannelMark, handles.MagnetoMeter );
handles.RefferenceAxialGradioMeter   = bitor( handles.RefferenceChannelMark, handles.AxialGradioMeter );
handles.RefferencePlannerGradioMeter = bitor( handles.RefferenceChannelMark, handles.PlannerGradioMeter );
handles.TriggerChannel      = -1;
handles.EegChannel          = -2;
handles.EcgChannel          = -3;
handles.EtcChannel          = -4;
handles.NonMegChannelNameLength = 32;
handles.DefaultMagnetometerSize       = (4.0/1000.0);		% ????4.0mm???????`
handles.DefaultAxialGradioMeterSize   = (15.5/1000.0);		% ???a15.5mm???~??
handles.DefaultPlannerGradioMeterSize = (12.0/1000.0);		% ????12.0mm???????`
handles.AcqTypeContinuousRaw = 1;
handles.AcqTypeEvokedAve     = 2;
handles.AcqTypeEvokedRaw     = 3;
handles.sqd = [];
handles.sqd.selected_start  = [];
handles.sqd.selected_end    = [];
handles.sqd.axialgradiometer_ch_no      = [];
handles.sqd.axialgradiometer_ch_info    = [];
handles.sqd.axialgradiometer_data       = [];
handles.sqd.plannergradiometer_ch_no    = [];
handles.sqd.plannergradiometer_ch_info  = [];
handles.sqd.plannergradiometer_data     = [];
handles.sqd.nullchannel_ch_no   = [];
handles.sqd.nullchannel_data    = [];
handles.sqd.selected_time       = [];
handles.sqd.sample_rate         = [];
handles.sqd.sample_count        = [];
handles.sqd.pretrigger_length   = [];
handles.sqd.matching_info   = [];
handles.sqd.source_info     = [];
handles.sqd.mri_info        = [];
handles.mri                 = [];