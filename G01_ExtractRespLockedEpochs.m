function G01_ExtractRespLockedEpochs(subjID)
%G01_EXTRACTRESPLOCKEDEPOCHS  Derive response-locked epochs from the
%   already-preprocessed allEpoc structure (A02_preprocMEG.m output),
%   without re-running raw load / artifact rejection / ICA.
%
%   Confirmed empirically: allEpoc trials already end exactly at +4.2s
%   (short-delay, trialinfo(:,5)==0) or +6.2s (long-delay,
%   trialinfo(:,5)==1) relative to stim onset (trigger 162). Both groups
%   therefore have well more than 5.0s of data available before their own
%   last sample (short: 4.2-(-1.5)=5.7s available; long: 6.2-(-1.5)=7.7s
%   available), so a single UNIFORM 5.0s window counted back from each
%   trial's own last sample works for every trial regardless of delay
%   length -- no need to split trials by long/short delay and apply
%   different window durations, and no truncation mismatch to work around.
%
%   Each trial is cropped to its last 5.0s (round(5.0*Fs) samples) and its
%   time axis re-referenced so 0 = trial end (last retained sample). If a
%   trial is unexpectedly shorter than 5.0s (timing jitter), the largest
%   available window is used instead and a warning is logged; trials are
%   never zero-padded.
%
%   Saves sub-XX_task-mgs_resplocked_lineremoved.mat (variable
%   epocRespLocked) alongside the existing *_epoched_lineremoved.mat /
%   *_stimlocked_lineremoved.mat files.

clearvars -except subjID; close all; clc;
warning('off', 'all');

addpath('/d/DATD/hyper/software/fieldtrip-20250318/');
ft_defaults;
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'));

bidsRoot         = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
derivativesRoot  = [bidsRoot filesep 'derivatives/sub-' num2str(subjID, '%02d') '/meg'];
taskName         = 'mgs';
subName          = ['sub-' num2str(subjID, '%02d')];
fNameRoot        = [subName '_task-' taskName];

epoch_fpath      = [derivativesRoot filesep fNameRoot '_epoched_lineremoved.mat'];
resplocked_fpath = [derivativesRoot filesep fNameRoot '_resplocked_lineremoved.mat'];

if exist(resplocked_fpath, 'file') > 0
    fprintf('%s already exists, skipping.\n', resplocked_fpath);
    return
end

fprintf('Loading %s\n', epoch_fpath);
load(epoch_fpath, 'allEpoc');

% Uniform window duration, seconds, applied to every trial regardless of
% long/short-delay condition (see header comment).
winDur = 5.0;

nTrials               = length(allEpoc.trial);
epocRespLocked        = allEpoc;             % carries over label/grad/fsample/trialinfo/etc.
epocRespLocked.trial  = cell(1, nTrials);
epocRespLocked.time   = cell(1, nTrials);
% sampleinfo referred to position within the original per-run continuous
% recording; after cropping to trial-end-relative windows it no longer
% has a well-defined meaning, so it is intentionally cleared rather than
% left silently stale.
epocRespLocked.sampleinfo = nan(nTrials, 2);

Fs             = allEpoc.fsample;
nSampRequested = round(winDur * Fs);

for i = 1:nTrials
    t          = allEpoc.time{i};
    trl        = allEpoc.trial{i};
    nSampAvail = size(trl, 2);

    if nSampRequested > nSampAvail
        warning('G01:shortTrial', ...
            ['Subject %d trial %d has only %d samples (%.3fs), fewer than ' ...
             'the %.1fs window (%d samples) -- using the full available ' ...
             'trial instead.'], ...
            subjID, i, nSampAvail, nSampAvail / Fs, winDur, nSampRequested);
        idx = 1:nSampAvail;
    else
        idx = (nSampAvail - nSampRequested + 1):nSampAvail;
    end

    epocRespLocked.trial{i} = trl(:, idx);
    % Re-reference time so 0 = trial end (last retained sample).
    epocRespLocked.time{i}  = t(idx) - t(idx(end));
end

fprintf('Saving %s\n', resplocked_fpath);
save(resplocked_fpath, 'epocRespLocked');

end
