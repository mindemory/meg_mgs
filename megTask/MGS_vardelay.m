function myscreen = MGS_vardelay(subID, run)
    % addpath(genpath('/usr/local/toolbox/mgl/'))
    % This function is a functional FEF/IPS localizer (memory-guided saccade)
    % trialStart, target, delay, go, corrective, ITI
    % Eventually: TR = 2000 ms, ITI = 2 ~ 6 TRs
    % Example usage: MGSlocalizerA1('01', '01')
    
    %% History
    % 09/01/09: Initial version written
    % 10/06/09: Added MEG bbox & photodiode function
    
    %% Initialize the screen
    myscreen.background = [.2 .2 .2];
    myscreen = initScreen(myscreen);

    %% Task parameters
    task{1}.waitForBacktick = 1;
    task{1}.segmin = [1, .2, 1.5, 1, 1.5, 2]; % fixation, target, delay, go, corrective, ITI
    task{1}.segmax = [1, .2, 3.5, 1, 1.5, 3];
    task{1}.segquant = [0, 0, 2, 0, 0, 0];
    task{1}.loc = [0:25:50, 130:25:230, 310:25:350]; % target locations
    task{1}.ecc = 9; % target eccentricity
    task{1}.numTrials = 25;
    task{1}.do_el = 1; % EyeLink: 1 = Yes, 0 = No
    task{1}.do_MEG = 1; % MEG: 1 = Yes, 0 = No
    task{1}.do_photodiode = 1; % Photodiode: 1 = Display white dot, 0 = Regular

    %% Initialize the task
    for phaseNum = 1:length(task)
        [task{phaseNum}, myscreen] = ...
            initTask(task{phaseNum}, myscreen, @startSegmentCallback, @screenUpdateCallback);
    end
    [task{1}, myscreen] = addTraces(task{1}, myscreen, 'TrialEvents');

    %% Initialize stimulus
    global stimulus;
    myscreen = initStimulus('stimulus', myscreen);
    stimulus = myInitStimulus(task);

    %% Initialize global variables
    global trialCount;
    trialCount = 0;

    %% Initialize EyeLink (if enabled)
    if task{1}.do_el
        [myscreen, el] = elgInit(myscreen, 12);
        Eyelink('command', 'link_sample_data = LEFT,RIGHT,GAZE,AREA');
        Eyelink('openfile', [subID '_' run 'MGS.edf']); % File name must be <= 8 characters
        
        % Calibrate the eye tracker
        EyelinkDoTrackerSetup(el);
        EyelinkDoDriftCorrect(el);

        % Start recording
        Eyelink('StartRecording');
    end

    %% Initialize MEG (if enabled)
    global id;
    tnum = 1;
    if task{1}.do_MEG
        [id] = bboxid(); % BBox function
        resetingBBox(id);
    end

    %% Main display loop
    while (tnum <= length(task)) && ~myscreen.userHitEsc
        [task, myscreen, tnum] = updateTask(task, myscreen, tnum);
        myscreen = tickScreen(myscreen, task);
    end

    %% Close EyeLink (if enabled)
    if task{1}.do_el
        Eyelink('StopRecording');
        Eyelink('ReceiveFile', [subID '_' run 'MGS.edf'], [subID '_' run 'MGS.edf']);
        Eyelink('ShutDown');
    end

    %% End of block
    mglClearScreen;
    mglTextSet('Helvetica', 32, [0 0 0], 0, 0, 0, 0, 0, 0, 0);
    mglTextDraw('let''s take a break...', [0 0]);
    mglFlush;

    Response = [0 0];
    while isempty(find(Response > 0, 1))
        Response = mglGetKeys(50);
    end

    %% Save task and stimulus parameters
    myscreen = endTask(myscreen, task);
end
