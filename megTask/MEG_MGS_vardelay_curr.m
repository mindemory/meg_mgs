function myscreen = MEG_MGS_vardelay_curr(subID,run)
%this is version with longer delay but lower variance photo delay and no
%separate baseline trials
addpath(genpath('/Users/mobileeye/Desktop/Experiment/mglstimroom'));

% addpath(genpath('/usr/local/toolbox/mgl/'))
 addpath(genpath('/Users/mobileeye/matlab/toolbox/eyelink'))
% this fxn is a functional FEF/IPS localizer (memory-guided saccade)
% trialStart, target, delay, go, corrective, ITI
% eventually: TR = 2000 ms, ITI = 2 ~ 6 TRs
% 
% e.g. MGSlocalizerA1('01','01')
%
% history
% 09/01/09: wrote it
% 10/06/09: added MEG bbox & photodiode fxn 

% initalize the screen

myscreen.background = [.2 .2 .2];
myscreen = initScreen(myscreen);
%myscreen.flushMode = 1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% JUST CHANGE HERE
task{1}.waitForBacktick = 1;
task{1}.segmin = [1 .2 1.5 1 1 2]; % fixation, target, delay, go, corrective, ITI
task{1}.segmax = [1 .2 3.5 1 1 3];
task{1}.segquant= [0 0 2 0 0 0];
task{1}.loc = [0:25:50, 130:25:230, 310:25:350]; % target locations
task{1}.ecc = 9; % target eccentricity 
task{1}.numTrials = 35;
task{1}.do_el = 0; %Are we using EyeLink? 1 = Yes, 0 = No
task{1}.do_MEG = 0; % 1 = Yes, 0 = No
task{1}.do_photodiode = 1; % 1 = display white dot, 0 = regular
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize the task
for phaseNum = 1:length(task)
    [task{phaseNum} myscreen] = ...
        initTask(task{phaseNum},myscreen,@startSegmentCallback,@screenUpdateCallback);
end
[task{1} myscreen] = addTraces(task{1}, myscreen,'TrialEvents');

% init the stimulus
global stimulus;
myscreen = initStimulus('stimulus',myscreen);
stimulus = myInitStimulus(task);

global trialCount;
trialCount = 0;

%%%%%%%%%%%%%%%%
% initialize eyetracker
%%%%%%%%%%%%%%%%
if task{1}.do_el

    
    [myscreen el] = elgInit(myscreen,12);
    % make sure that we get gaze data from the Eyelink
    Eyelink('command', 'link_sample_data = LEFT,RIGHT,GAZE,AREA');
    Eyelink('openfile', [subID '_' run 'MGS.edf']); % file name has to be <= 8 characters
    
    % Calibrate the eye tracker
    EyelinkDoTrackerSetup(el);

    % Do a final check of calibration using driftcorrection
    EyelinkDoDriftCorrect(el);
    
    Eyelink('StartRecording'); % make 1 big edf file (save time)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main display loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

global id
tnum = 1;

if task{1}.do_MEG
    [id]=bboxid; % BBox function
    resetingBBox(id);
end

while (tnum <= length(task)) && ~myscreen.userHitEsc
    [task myscreen tnum] = updateTask(task,myscreen,tnum);   
    myscreen = tickScreen(myscreen,task);
end

if task{1}.do_el % close EyeLink
    Eyelink('Stoprecording')
    Eyelink('ReceiveFile',[subID '_' run 'MGS.edf'],[subID '_' run 'MGS.edf']);
    Eyelink('ShutDown');
end

% end of block
mglClearScreen;
mglTextSet('Helvetica',32,[0 0 0],0,0,0,0,0,0,0);
mglTextDraw('let''s take a break...',[0 0]);mglFlush;
Response=[0 0];
while isempty(find(Response>0, 1))
    Response=mglGetKeys(50);
end

% save task and stimulus parameters
myscreen = endTask(myscreen,task);
eval (['save ' subID 'MGS_' num2str(run) '.mat  myscreen task stimulus']); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function that gets called at the start of each segment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [task myscreen] = startSegmentCallback(task, myscreen)

global stimulus trialCount id
fixrad=0.1;
switch task.thistrial.thisseg
    case 1 % trialStart, 
        trialCount = trialCount+1;
%         mglClearScreen;
%         mglFillOval(0,0,[fixrad,fixrad],[1 1 1]); % white dot
%         myscreen = writeTrace(task,myscreen,'TrialEvents',1);
%         mglFlush;
%         
        if task.do_el % start EyeLink
         %   Eyelink('StartRecording'); 
            Eyelink('Message','3rdI XDAT %i', 1);
        end

        if task.do_MEG
            [err]=PsychHID('SetReport',id,2,hex2dec('32'),uint8([128 128])); % send event code to Bbox  
        end

    case 2 % target, 
         
%         mglClearScreen;
%         mglFillOval(0,0,[fixrad,fixrad],[1 1 1]);
%         if (stimulus.tarlocCode(trialCount)<11)
%         mglFillOval(stimulus.x(trialCount), stimulus.y(trialCount), [0.5,0.5], [0 0 0]); 
%         myscreen = writeTrace(task,myscreen,'TrialEvents',2);
%      
%         end
%         if task.do_photodiode
%             mglFillOval(-8,10,[0.75,0.75],[1 1 1]); % white dot
%          
%         end
%         mglFlush;
        
        if task.do_el
            Eyelink('Message','3rdI XDAT %i', 2);
         end
         
         if task.do_MEG
            [err]=PsychHID('SetReport',id,2,hex2dec('32'),uint8([64 64])); % send event code to Bbox  
        end
        
    case 3 % delay
     
%         mglClearScreen;
%         mglFillOval(0,0,[fixrad,fixrad],[1 1 1]); % white dot
%         myscreen = writeTrace(task,myscreen,'TrialEvents',3);
%         mglFlush;
%         
         if task.do_el
            Eyelink('Message','3rdI XDAT %i', 3);
         end
         
         if task.do_MEG
            [err]=PsychHID('SetReport',id,2,hex2dec('32'),uint8([32 32])); % send event code to Bbox  
         end
        
    case 4 % go
       
%         mglClearScreen;
%         if (stimulus.tarlocCode(trialCount)<11)
%         myscreen = writeTrace(task,myscreen,'TrialEvents',stimulus.tarlocCode(trialCount)+10);
%         else
%         mglFillOval(0,0,[fixrad,fixrad],[1 1 1]); % white dot
%         end
%         mglFlush;
             
        if task.do_el
            Eyelink('Message','3rdI XDAT %i', stimulus.tarlocCode(trialCount)+10);
        end
        
        if task.do_MEG
            [err]=PsychHID('SetReport',id,2,hex2dec('32'),uint8([16 16])); % send event code to Bbox  
        end
%          
    case 5 % corrective 
%         mglClearScreen;
%         if (stimulus.tarlocCode(trialCount)<11)
%         mglFillOval(stimulus.x(trialCount), stimulus.y(trialCount), [.5,.5], [0 1 0]); % green dot
%         myscreen = writeTrace(task,myscreen,'TrialEvents',stimulus.tarlocCode(trialCount)+10); % 11~20
%         else
%         mglFillOval(0,0,[fixrad,fixrad],[1 1 1]); % white dot
%         end
%         mglFlush;
        
        if task.do_el
            Eyelink('Message','3rdI XDAT %i', stimulus.tarlocCode(trialCount)+10);
         end
         
         if task.do_MEG
            [err]=PsychHID('SetReport',id,2,hex2dec('32'),uint8([8 8])); % send event code to Bbox  
        end
        
    case 6 % ITI
%         mglClearScreen;
%         mglFillOval(0,0,[fixrad,fixrad],[.4 .4 .4]); % grey dot
%         myscreen = writeTrace(task,myscreen,'TrialEvents',4);
%         mglFlush     
        if task.do_el
            Eyelink('Message','3rdI XDAT %i', 6);
           % Eyelink('Stoprecording')
         end
         
         if task.do_MEG
            [err]=PsychHID('SetReport',id,2,hex2dec('32'),uint8([4 4])); % send event code to Bbox  
        end
end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function that gets called to draw the stimulus each frame
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [task myscreen] = screenUpdateCallback(task, myscreen)
fixrad=0.1;
global stimulus trialCount
% 
switch task.thistrial.thisseg
    case 1 % trial start
        mglClearScreen;
        mglFillOval(0,0,[fixrad,fixrad],[1 1 1]); % white dot
        myscreen = writeTrace(task,myscreen,'TrialEvents',1);
        
    case 2 % target
        mglClearScreen;
        mglFillOval(0,0,[fixrad,fixrad],[1 1 1]);

         mglFillOval(stimulus.x(trialCount), stimulus.y(trialCount), [.5,.5], [0 0 0]); 
         myscreen = writeTrace(task,myscreen,'TrialEvents',2);
        
        if task.do_photodiode
            mglFillOval(-8,10,[0.75,0.75],[1 1 1]); % white dot
        end
    
    case 3 % delay
        mglClearScreen;
        mglFillOval(0,0,[fixrad,fixrad],[1 1 1]); % white dot
        myscreen = writeTrace(task,myscreen,'TrialEvents',3);
        
    case 4 % go
        mglClearScreen;
        myscreen = writeTrace(task,myscreen,'TrialEvents',stimulus.tarlocCode(trialCount)+10);
       
                
    case 5 % corrective
        mglClearScreen;
     
        mglFillOval(stimulus.x(trialCount), stimulus.y(trialCount), [.5,.5], [0 1 0]); % green dot
        myscreen = writeTrace(task,myscreen,'TrialEvents',stimulus.tarlocCode(trialCount)+10); % 11~20
        
    case 6 % ITI
        mglClearScreen;
        mglFillOval(0,0,[fixrad,fixrad],[.4 .4 .4]); % grey dot
        myscreen = writeTrace(task,myscreen,'TrialEvents',4);
        
        
end
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to init the stimulus
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function stimulus = myInitStimulus(task)

loc = task{1}.loc;
ecc = task{1}.ecc;
nlocs=length(task{1}.loc);
repsperloc=ceil((task{1}.numTrials)/nlocs);

% nT = 1:task{1}.numTrials;
% nTshuffle = Shuffle(nT);
% trialSeq = mod(nTshuffle,10) %0-9 rand order for ntrials
trials= [repmat(0:(nlocs-1), [1 repsperloc])];

[t, trialSeqind]=sort(rand(1, task{1}.numTrials));

trialSeq=trials(trialSeqind);

for i = 1:task{1}.numTrials
   
     stimulus.tarloc(i) = loc(trialSeq(i)+1);
     stimulus.tarlocCode(i) = trialSeq(i)+1;
     stimulus.x(i) = cosd(stimulus.tarloc(i))*ecc;
     stimulus.y(i) = sind(stimulus.tarloc(i))*ecc;


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% bboxid
function [id]=bboxid
fprintf('looking for ioLab BBox...\n');
devices=squeeze(struct2cell(PsychHID('devices')));
devicesid=cell2mat(devices(6,:));
id=find(devicesid==6588);
if isempty(id)==1
    fprintf('BBox not found!\n');return;
end;
fprintf('BBox is ready to go!\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% resetingBBox
function resetingBBox(id)
[err]=PsychHID('SetReport',1,2,hex2dec('48'),uint8([255 255 255 255])); % * echo '0x48,0xFF,0xFF' | ./usbbox
while 1,
    [report err]=PsychHID('GetReport',id,3,0,8);
    if(isempty(report)==1),
        break;
    end,
end,
