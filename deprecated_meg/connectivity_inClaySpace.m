clear; close all; clc;
%% Initialization
addpath('/d/DATD/hyper/software/fieldtrip-20250318/'); % 2022 doesn't work well for sourerecon
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'))
ft_defaults;
ft_hastoolbox('spm12', 1);

%%
% subList                               = [ 1,  2,  3,  4,  5,  6,  7,  9, 10, 12, 13, 15, ...
%                                          17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
subList                               = [13, 18, 19];
% connMetrics                           = {'coh', 'icoh', 'pcoh', 'ipcoh', 'plv', 'pli'};
connMetrics                           = {'coh', 'icoh', 'plv', 'pli'};
nTrlGroups                            = 3; % 1:Left, 2:Right, 3:Combined
timeArray                             = -1:0.1:1.7;
tWindow                               = 0.15; % 150ms (in seconds) on each side
for sIdx                              = 1:length(subList)
    subjID                            = subList(sIdx);
    % subjID                             = 2;
    disp(['Running ' num2str(subjID, '%02d')])
    subDerivativesRoot                = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/sub-' num2str(subjID, '%02d') ...
                                         '/meg/sub-' num2str(subjID, '%02d') '_task-mgs_'];
    sourceSpacefPath                  = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/allsubs_clayspace/sub-' ...
                                         num2str(subjID, '%02d') '_task-mgs_source-clayspace.mat']; 
    connectivityfPath                 = ['/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives/allsubs_clayspace/connectivityMeasures/' ...
                                         'sub-' num2str(subjID, '%02d') '_task-mgs_source-clayspace.mat']; 
    load(sourceSpacefPath);

    
    nSeeds                            = length(find(source.inside));
    clearvars source;
    seedSources                       = 1:nSeeds;
    
    % Keep trlIndices for left and right Trials
    leftTrlIdx                        = find(sourcedata.trialinfo(:,6) == 1);
    rightTrlIdx                       = find(sourcedata.trialinfo(:,6) == 2);
    
    % Initialize arrays for each metric
    for mm                            = 1:length(connMetrics)
        thMetric                      = connMetrics{mm};
        connectivityStruct.(thMetric) = NaN(nTrlGroups, nSeeds, length(timeArray));
    end

    batchSize                         = 100;
    nBatches                          = ceil(length(seedSources) / batchSize);
    
    
    % Compute all connectivity measures for each time point using a
    % moving-window
    tic
    for tIdx                          = 1:length(timeArray)
        disp(['     Finished ' num2str(round(100 * tIdx/length(timeArray),2)) '%, time elapsed: ' num2str(toc-tic) ' s'])
        TOI                           = find(sourcedata.time{1} > timeArray(tIdx) - tWindow & ...
                                             sourcedata.time{1} < timeArray(tIdx) + tWindow);


        for b                         = 1:nBatches
            startIdx                  = (b - 1) * batchSize + 1;
            endIdx                    = min(b * batchSize, length(seedSources));
            batchSeeds                = seedSources(startIdx:endIdx);
    
            % [cohTrlWise, icohTrlWise, pcohTrlWise, ipcohTrlWise, plvTrlWise, pliTrlWise] ...
            [cohTrlWise, icohTrlWise, plvTrlWise, pliTrlWise] ...
                                      = computeConnectivitySourceSpace(sourcedata, batchSeeds, TOI);
    
            connData                  = struct('coh', cohTrlWise, 'icoh', icohTrlWise, ...
                                               'plv', plvTrlWise, 'pli', pliTrlWise);
    
            for mm                    = 1:length(connMetrics)
                thMetric              = connMetrics{mm};
                data                  = connData.(thMetric);
                connectivityStruct.(thMetric)(1,:,startIdx:endIdx,tIdx) ...
                                      = squeeze(mean(data, 1, 'omitnan'));
                connectivityStruct.(thMetric)(2,:,startIdx:endIdx,tIdx) ...
                                      = squeeze(mean(data(leftTrlIdx,:,:), 1, 'omitnan'));
                connectivityStruct.(thMetric)(3,:,startIdx:endIdx,tIdx) ...
                                          = squeeze(mean(data(rightTrlIdx,:,:), 1, 'omitnan'));
            end
            clearvars cohTrlWise icohTrlWise pcohTrlWise ipcohTrlWise plvTrlWise pliTrlWise;
        end
    end
    save(connectivityfPath, 'connectivityStruct', '-v7.3');
    clearvars connectivityStruct;
end