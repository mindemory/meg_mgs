% create a concentric 3-sphere volume conductor, the radius is the  
same as for the electrodes
vol = [];
vol.r = [0.88 0.92 1.00]; % radii of spheres
vol.c = [1 1/80 1];       % conductivity
vol.o = [0 0 0];          % center of sphere

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%
% create a dipole simulation with one dipole and a 10Hz sine wave
cfg = [];
cfg.vol  = vol;             % see above
cfg.elec = elec;            % see above
cfg.dip.pos = [0 0.5 0.3];
cfg.dip.mom = [1 0 0]';     % note, it should be transposed
cfg.dip.frequency = 10;
cfg.ntrials = 10;
cfg.triallength = 1;        % seconds
cfg.fsample = 250;          % Hz
raw1 = dipolesimulation(cfg);
avg1 = timelockanalysis([], raw1);
plot(avg1.time, avg1.avg);  % plot the timecourse

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%
% create a dipole simulation with one dipole and a custom timecourse
cfg = [];
cfg.vol  = vol;               % see above
cfg.elec = elec;              % see above
cfg.dip.pos = [0 0.5 0.3];
cfg.dip.mom = [1 0 0]';       % note, it should be transposed
cfg.fsample = 250;            % Hz
time = (1:250)/250;           % manually create a time axis
signal = sin(10*time*2*pi);   % manually create a signal
cfg.dip.signal = {signal, signal, signal};  % three trials
raw2 = dipolesimulation(cfg);
avg2 = timelockanalysis([], raw2);
plot(avg2.time, avg2.avg);    % plot the timecourse

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%%%%
% create a dipole simulation with two dipoles and a custom timecourse
cfg = [];
cfg.vol  = vol;    % see above
cfg.elec = elec;   % see above
cfg.dip.pos = [
   0  0.5 0.3       % dipole 1
   0 -0.5 0.3       % dipole 2
   ];
cfg.dip.mom = [     % each row represents [qx1 qy1 qz1 qx2 qy2 qz2]
   1 0 0 0 0 0       % this is how signal1 contributes to the 6  
dipole components
   0 0 0 1 0 0       % this is how signal2 contributes to the 6  dipole components
   ]';               % note, it should be transposed
time = (1:250)/250;
signal1 = sin(10*time*2*pi);
signal2 = cos(15*time*2*pi);
cfg.dip.signal = {[signal1; signal2]}; % one trial only
cfg.fsample = 250;                     % Hz
raw3 = dipolesimulation(cfg);
avg3 = timelockanalysis([], raw3);

