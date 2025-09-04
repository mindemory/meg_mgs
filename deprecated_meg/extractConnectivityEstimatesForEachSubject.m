clear; close all; clc;

subList                     = [ 1,  2,  3,  4,  5,  6,  7,  9, 10, 12, 13, 15, ...
                               17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
rootDir                     = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/allsubs_clayspace/connectivityMeasures';

% Initialize connectivity metric names and directories
connMetrics                 = {'coh', 'icoh', 'plv', 'pli'};
for mm                      = 1:length(connMetrics)
    thMetric                = connMetrics{mm};
    thMetricDir             = [rootDir filesep thMetric];
    if ~exist(thMetricDir, 'dir')
        mkdir(thMetricDir)
    end
end

for sIdx                    = 1:length(subList)
    subjID                  = subList(sIdx);
    disp(['Running ' num2str(subjID, '%02d')])
    connectivityRoot        = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/allsubs_clayspace/connectivityMeasures';
    connectivityfPath       = [connectivityRoot '/sub-' num2str(subjID, '%02d') '_task-mgs_source-clayspace.mat'];

    % Initialize paths for connectivity files
    pp                      = struct();
    flsExist                = 0;
    for mm                  = 1:length(connMetrics)
        thMetric            = connMetrics{mm};
        pp.(thMetric)       = [connectivityRoot filesep thMetric '/sub-' num2str(subjID, '%02d') '_task-mgs_' thMetric '.mat'];
        if exist(pp.(thMetric), 'file') == 2
            flsExist        = flsExist + 1;
        end
    end

    if flsExist             < length(connMetrics)
        disp('      Extracting connectivity measures ...')
        load(connectivityfPath);
    
        for mm                  = 1:length(connMetrics)
            thMetric            = connMetrics{mm};
            connectMat          = connectivityStruct.(thMetric);
            thMetricfPath       = pp.(thMetric);
            save(thMetricfPath, 'connectMat', '-v7.3');
        end
        clearvars connectivityStruct pp connectMat;
    end
end