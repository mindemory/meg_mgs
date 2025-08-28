% This code gets metadata from all subjects in terms of experiment design
% and order
% Some things to keep in mind are:
%   Screen resolution is same throughout (768, 1024)
%   Eccentricity of the stimulus is same 9 degrees
%   displayDistance is same 57cm
%   Number of trials per block is either 25 (subs 1 - 9 (until block 7))
%   and 35 (sub 9 (blocks 8,9,10) - 32)
%   ntargs were 11 for sub1-7 and went to 10 for sub8-32
%   nruns varies between 8-12

bidsRoot = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';

subdirList = dir([bidsRoot '/sub-*']);

metaData = struct();
metaData.subID = [];
metaData.run = [];
metaData.screenHeight = [];
metaData.screenWidth = [];
metaData.displayDistance = [];
metaData.ntrials = [];
metaData.uniqtar = [];
metaData.ecc = [];
for sIdx = 1:length(subdirList)
    subdir = [bidsRoot filesep subdirList(sIdx).name];
    stimDir = [subdir filesep 'stimfiles'];
    stimdirList = dir([stimDir '/*.mat']);

    
    for run = 1:length(stimdirList)
        metaData.subID = [metaData.subID; sIdx];
        metaData.run = [metaData.run; run];

        stimfPath = [stimDir filesep stimdirList(run).name];
        load(stimfPath);

        metaData.screenHeight = [metaData.screenHeight; myscreen.screenHeight];
        metaData.screenWidth = [metaData.screenWidth; myscreen.screenWidth];
        metaData.displayDistance = [metaData.displayDistance; myscreen.displayDistance];
        metaData.ntrials = [metaData.ntrials; length(stimulus.x)];
        metaData.uniqtar = [metaData.uniqtar; length(unique(stimulus.tarlocCode))];
        metaData.ecc = [metaData.ecc; task{1}.ecc];
    
        clearvars myscreen stimulus task;
    end

end

metaTable = struct2table(metaData);