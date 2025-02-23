function [trl, event] = readEvents(cfg)

hdr = ft_read_header(cfg.dataset);
event = ft_read_event(cfg.dataset, ...
                       'chanindx', 161:166, ...
                       'threshold', 1e0);
if strcmp(cfg.dataset, ...
        '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-12/meg/sub-12_task-mgs_run-08_meg.sqd')
    % For sub12, run08 the first trial triggers are missing, fixing that
    % manually
    event = event(3:end);
elseif strcmp(cfg.dataset, ...
        '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-04/meg/sub-04_task-mgs_run-10_meg.sqd')
    % For sub04, run10, the first trial triggers are missing, fixing that
    % manually
    event = event(4:end);
elseif strcmp(cfg.dataset, ...
        '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-13/meg/sub-13_task-mgs_run-02_meg.sqd')
    % For sub13, run02, the first trial triggers are missing, fixin that
    % manually
    event = event(4:end);
elseif strcmp(cfg.dataset, ...
        '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS/sub-31/meg/sub-31_task-mgs_run-04_meg.sqd')
    % For sub31, run04, triggers are present only for first 18 trials and
    % missing after
    event = event(1:end-1);
end
% Triggers and epochs
%   161: fixation
%   162: sample
%   163: delay
%   164: response
%   165: feedback
%   166: ITI

% Creating a block trlMat
% startEvent = [event(strcmp('162', {event.type})).value]';
startSample = [event(strcmp('162', {event.type})).sample]';
% endEvent = [event(strcmp('166', {event.type})).value]';
endSample = [event(strcmp('165', {event.type})).sample]';


trl = [];
for j = 1:length(startSample)
    trlbegin  = startSample(j) - round(cfg.trialdef.prestim * hdr.Fs);
    trlend    = endSample(j) + round(cfg.trialdef.poststim * hdr.Fs);
    offset    = -round(cfg.trialdef.prestim * hdr.Fs);
    newtrl    = [trlbegin trlend offset];
    trl       = [trl; newtrl];
end


end