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