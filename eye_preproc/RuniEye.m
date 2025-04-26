function ii_sess = RuniEye(subjID, derivativesRoot, stimRoot, nRuns)
%%%%%%%%%%%%%%%%%%%%%%% NOTE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Updated on Jan 24th 2025:
% Adding a XDAT == 5 as the onset of a feedback epoch.
% response epoch is for 1000ms and feedback epoch is for the remaining
% 1500ms
% XDAT=5 natively does not exist in the edf files, but adding it manually.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
stimFiles = dir([stimRoot '/*.mat']);

% List of epochs in the task
% XDAT 1: Fixation window =  1s
% XDAT 10: Sample window =  0.5s (would correspond to 'S 11', 'S 12', 'S
% 13', 'S 14' in EEG flags depending on the stimulus VF and block type
% XDAT 2: Delay = 1.5/3.5s
% XDAT 4: Response window = 1 s
% XDAT 5: Feedback window = 1 s
% XDAT 6: ITI window = 2s or 3s (50-50 split)
ifgFile = 'meg_eye1000Hz.ifg';
ii_params = ii_loadparams; % load default set of analysis parameters, only change what we have to
%ii_params.valid_epochs =[1 2 3 11 12 13 14 15 16 17 18 19 20 6]; % Updated epochs (Mrugank: 04/05/2023)
ii_params.valid_epochs =[1 2 3 4 5 6]; % Updated epochs (Mrugank: 04/05/2023)
ii_params.trial_start_value = 1; %XDAT value for trial start
ii_params.trial_end_value = 5;   % XDAT value for trial end
ii_params.drift_epoch = [1 2 3]; % XDAT values for drift correction (
ii_params.drift_fixation_mode  = 'mode';
%ii_params.calibrate_epoch = [11 12 13 14 15 16 17 18 19 20]; % XDAT value for when we calibrate (feedback stim)
ii_params.calibrate_epoch = [5]; % XDAT value for when we calibrate (feedback stim)

ii_params.calibrate_select_mode = 'nearest_longdwell'; % how do we select fixation with which to calibrate?
ii_params.calibrate_mode = 'scale'; % scale: trial-by-trial, rescale each trial; 'run' - run-wise polynomial fit
ii_params.blink_thresh = 0.1;
ii_params.blink_window = [100 100]; % how long before/after blink (ms) to drop?
ii_params.plot_epoch = [2 3 4 5];  % what epochs do we plot for preprocessing?
%ii_params.plot_epoch = [2 3 11 12 13 14 15 16 17 18 19 20];  % what epochs do we plot for preprocessing?

ii_params.calibrate_limits = [2.5]; % when amount of adj exceeds this, don't actually calibrate (trial-wise); ignore trial for polynomial fitting (run)

% Mrugank (04/05/2023): Could possibly be updated later?
excl_criteria.i_dur_thresh = 850; % must be shorter than 150 ms
excl_criteria.i_amp_thresh = 2;   % must be longer than 5 dva [if FIRST saccade in denoted epoch is not at least this long and at most this duration, drop the trial]
excl_criteria.i_err_thresh = 10;   % i_sacc must be within this many DVA of target pos to consider the trial

excl_criteria.drift_thresh = 2.5;     % if drift correction norm is this big or more, drop
excl_criteria.delay_fix_thresh = 2.5; % if any fixation is this far from 0,0 during delay (epoch 3)

if subjID == 19
    runList = [1 2 3 4 5 7 8 9];
elseif subjID == 23
    runList = 2:9;
elseif subjID == 31
    runList = [1 3 5 6 7 8];
else
    runList = 1:nRuns;
end

for run = runList
    disp(['Preprocessing run ' num2str(run, '%02d')])
    runDir = [derivativesRoot '/run-' num2str(run, '%02d')];
    
    % Loading display parameters for the run
    
    stimulusfPath = [stimRoot filesep stimFiles(run).name];
    load(stimulusfPath, 'myscreen', 'stimulus');
    pixSize_hori = myscreen.displaySize(1) / myscreen.screenWidth;
    pixSize_verti = myscreen.displaySize(2) / myscreen.screenHeight;
    myscreen.pixSize = mean([pixSize_hori, pixSize_verti]);
    
    ii_params.resolution = [myscreen.screenWidth myscreen.screenHeight];
    ii_params.ppd = myscreen.displayDistance * tand(1) / myscreen.pixSize;
    
    edfList = dir([runDir '/*.edf']);
    edfFile = [runDir filesep edfList(1).name];
    % what is the output filename?
    preproc_fn = edfFile(1:end-4);
    
    % run preprocessing!
    [ii_data, ii_cfg, ii_sacc] = run_iipreproc(edfFile, ifgFile, preproc_fn, ii_params, [], stimulus);

    % score trials
    % default parameters should work fine - but see docs for other
    % arguments you can/should give when possible
    [ii_trial{run},~] = ii_scoreMGS(ii_data,ii_cfg,ii_sacc,[],4,[],excl_criteria,[],'lenient');
    ii_trial{run}.tarloc = stimulus.tarloc';
    ii_trial{run}.tarlocCode = stimulus.tarlocCode';
    ii_trial{run}.x = stimulus.x';
    ii_trial{run}.y = stimulus.y';
end

disp('Combining runs')
if subjID == 19
    rnum = 6;
    ntrls = 35;
    ii_trial{rnum}.X                = cell(ntrls, 1);
    ii_trial{rnum}.Y                = cell(ntrls, 1);
    ii_trial{rnum}.Pupil            = cell(ntrls, 1);
    ii_trial{rnum}.XDAT             = cell(ntrls, 1);
    ii_trial{rnum}.Velocity         = cell(ntrls, 1);
    ii_trial{rnum}.i_sacc_raw       = NaN(ntrls, 2);
    ii_trial{rnum}.f_sacc_raw       = NaN(ntrls, 2);
    ii_trial{rnum}.i_sacc           = NaN(ntrls, 2);
    ii_trial{rnum}.f_sacc           = NaN(ntrls, 2);
    ii_trial{rnum}.i_sacc_err       = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_err       = NaN(ntrls, 1);
    ii_trial{rnum}.n_sacc           = NaN(ntrls, 1);
    ii_trial{rnum}.n_sacc_epoch     = NaN(ntrls, 5);
    ii_trial{rnum}.i_sacc_rt        = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_rt        = NaN(ntrls, 1);
    ii_trial{rnum}.i_sacc_trace     = cell(ntrls, 1);
    ii_trial{rnum}.f_sacc_trace     = cell(ntrls, 1);
    ii_trial{rnum}.i_sacc_peakvel   = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_peakvel   = NaN(ntrls, 1);
    ii_trial{rnum}.calib_amt        = NaN(ntrls, 2);
    ii_trial{rnum}.calib_adj        = NaN(ntrls, 1);
    ii_trial{rnum}.calib_err        = NaN(ntrls, 1);
    ii_trial{rnum}.drift_amt        = NaN(ntrls, 2);
    ii_trial{rnum}.excl_trial       = cell(ntrls, 1);
    ii_trial{rnum}.params           = ii_trial{1}.params;
    ii_trial{rnum}.targ             = NaN(ntrls, 2);
    ii_trial{rnum}.tarloc           = NaN(ntrls, 1);
    ii_trial{rnum}.tarlocCode       = NaN(ntrls, 1);
    ii_trial{rnum}.x                = NaN(ntrls, 1);
    ii_trial{rnum}.y                = NaN(ntrls, 1);
elseif subjID == 23
    rnum = 1;
    ntrls = 35;
    ii_trial{rnum}.X                = cell(ntrls, 1);
    ii_trial{rnum}.Y                = cell(ntrls, 1);
    ii_trial{rnum}.Pupil            = cell(ntrls, 1);
    ii_trial{rnum}.XDAT             = cell(ntrls, 1);
    ii_trial{rnum}.Velocity         = cell(ntrls, 1);
    ii_trial{rnum}.i_sacc_raw       = NaN(ntrls, 2);
    ii_trial{rnum}.f_sacc_raw       = NaN(ntrls, 2);
    ii_trial{rnum}.i_sacc           = NaN(ntrls, 2);
    ii_trial{rnum}.f_sacc           = NaN(ntrls, 2);
    ii_trial{rnum}.i_sacc_err       = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_err       = NaN(ntrls, 1);
    ii_trial{rnum}.n_sacc           = NaN(ntrls, 1);
    ii_trial{rnum}.n_sacc_epoch     = NaN(ntrls, 5);
    ii_trial{rnum}.i_sacc_rt        = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_rt        = NaN(ntrls, 1);
    ii_trial{rnum}.i_sacc_trace     = cell(ntrls, 1);
    ii_trial{rnum}.f_sacc_trace     = cell(ntrls, 1);
    ii_trial{rnum}.i_sacc_peakvel   = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_peakvel   = NaN(ntrls, 1);
    ii_trial{rnum}.calib_amt        = NaN(ntrls, 2);
    ii_trial{rnum}.calib_adj        = NaN(ntrls, 1);
    ii_trial{rnum}.calib_err        = NaN(ntrls, 1);
    ii_trial{rnum}.drift_amt        = NaN(ntrls, 2);
    ii_trial{rnum}.excl_trial       = cell(ntrls, 1);
    ii_trial{rnum}.params           = ii_trial{2}.params;
    ii_trial{rnum}.targ             = NaN(ntrls, 2);
    ii_trial{rnum}.tarloc           = NaN(ntrls, 1);
    ii_trial{rnum}.tarlocCode       = NaN(ntrls, 1);
    ii_trial{rnum}.x                = NaN(ntrls, 1);
    ii_trial{rnum}.y                = NaN(ntrls, 1);
elseif subjID == 31
    rnum = 2;
    ntrls = 35;
    ii_trial{rnum}.X                = cell(ntrls, 1);
    ii_trial{rnum}.Y                = cell(ntrls, 1);
    ii_trial{rnum}.Pupil            = cell(ntrls, 1);
    ii_trial{rnum}.XDAT             = cell(ntrls, 1);
    ii_trial{rnum}.Velocity         = cell(ntrls, 1);
    ii_trial{rnum}.i_sacc_raw       = NaN(ntrls, 2);
    ii_trial{rnum}.f_sacc_raw       = NaN(ntrls, 2);
    ii_trial{rnum}.i_sacc           = NaN(ntrls, 2);
    ii_trial{rnum}.f_sacc           = NaN(ntrls, 2);
    ii_trial{rnum}.i_sacc_err       = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_err       = NaN(ntrls, 1);
    ii_trial{rnum}.n_sacc           = NaN(ntrls, 1);
    ii_trial{rnum}.n_sacc_epoch     = NaN(ntrls, 5);
    ii_trial{rnum}.i_sacc_rt        = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_rt        = NaN(ntrls, 1);
    ii_trial{rnum}.i_sacc_trace     = cell(ntrls, 1);
    ii_trial{rnum}.f_sacc_trace     = cell(ntrls, 1);
    ii_trial{rnum}.i_sacc_peakvel   = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_peakvel   = NaN(ntrls, 1);
    ii_trial{rnum}.calib_amt        = NaN(ntrls, 2);
    ii_trial{rnum}.calib_adj        = NaN(ntrls, 1);
    ii_trial{rnum}.calib_err        = NaN(ntrls, 1);
    ii_trial{rnum}.drift_amt        = NaN(ntrls, 2);
    ii_trial{rnum}.excl_trial       = cell(ntrls, 1);
    ii_trial{rnum}.params           = ii_trial{1}.params;
    ii_trial{rnum}.targ             = NaN(ntrls, 2);
    ii_trial{rnum}.tarloc           = NaN(ntrls, 1);
    ii_trial{rnum}.tarlocCode       = NaN(ntrls, 1);
    ii_trial{rnum}.x                = NaN(ntrls, 1);
    ii_trial{rnum}.y                = NaN(ntrls, 1);

    rnum = 4;
    ntrls = 35;
    ii_trial{rnum}.X                = cell(ntrls, 1);
    ii_trial{rnum}.Y                = cell(ntrls, 1);
    ii_trial{rnum}.Pupil            = cell(ntrls, 1);
    ii_trial{rnum}.XDAT             = cell(ntrls, 1);
    ii_trial{rnum}.Velocity         = cell(ntrls, 1);
    ii_trial{rnum}.i_sacc_raw       = NaN(ntrls, 2);
    ii_trial{rnum}.f_sacc_raw       = NaN(ntrls, 2);
    ii_trial{rnum}.i_sacc           = NaN(ntrls, 2);
    ii_trial{rnum}.f_sacc           = NaN(ntrls, 2);
    ii_trial{rnum}.i_sacc_err       = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_err       = NaN(ntrls, 1);
    ii_trial{rnum}.n_sacc           = NaN(ntrls, 1);
    ii_trial{rnum}.n_sacc_epoch     = NaN(ntrls, 5);
    ii_trial{rnum}.i_sacc_rt        = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_rt        = NaN(ntrls, 1);
    ii_trial{rnum}.i_sacc_trace     = cell(ntrls, 1);
    ii_trial{rnum}.f_sacc_trace     = cell(ntrls, 1);
    ii_trial{rnum}.i_sacc_peakvel   = NaN(ntrls, 1);
    ii_trial{rnum}.f_sacc_peakvel   = NaN(ntrls, 1);
    ii_trial{rnum}.calib_amt        = NaN(ntrls, 2);
    ii_trial{rnum}.calib_adj        = NaN(ntrls, 1);
    ii_trial{rnum}.calib_err        = NaN(ntrls, 1);
    ii_trial{rnum}.drift_amt        = NaN(ntrls, 2);
    ii_trial{rnum}.excl_trial       = cell(ntrls, 1);
    ii_trial{rnum}.params           = ii_trial{1}.params;
    ii_trial{rnum}.targ             = NaN(ntrls, 2);
    ii_trial{rnum}.tarloc           = NaN(ntrls, 1);
    ii_trial{rnum}.tarlocCode       = NaN(ntrls, 1);
    ii_trial{rnum}.x                = NaN(ntrls, 1);
    ii_trial{rnum}.y                = NaN(ntrls, 1);
end  

% Creating ii_sess only if ii_trial is valid
if ~exist("ii_trial", "var")
    ii_sess = [];
else
    ii_sess = ii_combineruns(ii_trial);
    disp(['Total trials = ', num2str(size(ii_sess.i_sacc_err, 1))])
    disp(['nan trials ii_sess.i_sacc_err = ', num2str(sum(isnan(ii_sess.i_sacc_err)))])
    disp(['nan trials ii_sess.f_sacc_err = ', num2str(sum(isnan(ii_sess.f_sacc_err)))])
    
    % Flag trials with bad drift correction
    ii_sess.bad_drift_correct = double(cell2mat(cellfun(@(x) ismember(11, x), ii_sess.excl_trial, 'UniformOutput', false)));
    % Flag trials with bad calibration
    ii_sess.bad_calibration = double(cell2mat(cellfun(@(x) ismember(12, x), ii_sess.excl_trial, 'UniformOutput', false)));
    % Flag trials with fixation breaks
    ii_sess.break_fix = double(cell2mat(cellfun(@(x) ismember(13, x), ii_sess.excl_trial, 'UniformOutput', false)));
    % Flag trials with no primary saccades
    ii_sess.no_prim_sacc = double(cell2mat(cellfun(@(x) ismember(20, x), ii_sess.excl_trial, 'UniformOutput', false)));
    % Flag trials with small or short saccades
    ii_sess.small_sacc = double(cell2mat(cellfun(@(x) ismember(21, x), ii_sess.excl_trial, 'UniformOutput', false)));
    % Flag trials with large MGS errors
    ii_sess.large_error = double(cell2mat(cellfun(@(x) ismember(22, x), ii_sess.excl_trial, 'UniformOutput', false)));

    % Put a reject trial flag: no primary saccade or a large saccade error
    ii_sess.rejtrials = double(cell2mat(cellfun(@(x) any(ismember([20, 22], x)), ii_sess.excl_trial, 'UniformOutput', false))); 
end
end