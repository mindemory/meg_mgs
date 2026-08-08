function G04_BandAmplitudePhaseInSource(subjID, lockType, frequency_band, volumetric_resolution)
%G04_BANDAMPLITUDEPHASEINSOURCE  Per-band amplitude(+phase) extraction
%   from G03's broadband source-space output, adapted from
%   S03A_FrequencyPowerInSource.m's bandpass+Hilbert pattern (not
%   modified in place).
%
%   Inputs:
%     subjID                 - subject ID
%     lockType                - 'stim' or 'resp'
%     frequency_band          - 'theta' | 'alpha' | 'beta' | 'lowgamma' | 'highgamma'
%     volumetric_resolution   - volumetric grid resolution in mm (default: 8)
%
%   Canonical band table (new source-space pipeline only -- existing
%   Python connectivity/decoding scripts' independent band definitions
%   are left untouched):
%       theta      4-8   Hz   amplitude + phase
%       alpha      8-12  Hz   amplitude + phase
%       beta       15-25 Hz   amplitude + phase
%       lowgamma   30-50 Hz   amplitude only
%       highgamma  65-95 Hz   amplitude only  (clear of the 60Hz line-noise
%                                               notch and its filter skirt)
%
%   Unlike S03A_FrequencyPowerInSource.m, no complex analytic signal is
%   persisted -- amplitude (envelope, abs(hilbert)) and, where listed
%   above, phase (radians, angle(hilbert)) are saved directly as
%   real-valued single-precision arrays. This avoids the MATLAB
%   complex/h5py compound-dtype friction for Python consumers and matches
%   exactly what downstream analyses (PAC, connectivity, GLUE) need.
%
%   *** One shared storage rate across ALL bands (not per-band) ***
%   Every band is downsampled after the Hilbert transform to the SAME
%   TARGET_RATE (chosen to satisfy highgamma's Nyquist needs, the fastest
%   band), rather than a band-optimized rate. A per-band rate would save
%   some storage for the narrower, slower-fluctuating bands (theta/alpha/
%   beta), but it would leave each band on a different time axis --
%   directly at odds with the project's target unified array
%   ([subject,band,trial,parcel,time], one shared time axis across
%   bands), and it would force a resample-to-common-axis step before any
%   cross-band analysis (cross-frequency PAC, cross-band coherence, the
%   amplitude-vs-amplitude+phase capacity comparison across bands). One
%   shared rate costs more storage for theta/alpha/beta than they
%   strictly need, but keeps every band trivially stackable as-is.
%
%   *** Amplitude vs. phase downsampling (different treatment, on purpose) ***
%   Amplitude (envelope) is a smooth, non-wrapping real signal, so it is
%   anti-alias lowpass-filtered before decimation. Phase is a WRAPPED
%   circular quantity in [-pi,pi]; naively lowpass-filtering a wrapped
%   phase signal corrupts it at every +-pi wrap discontinuity, so phase
%   is decimated directly (no filtering) at the same sample indices used
%   for amplitude, keeping both on a shared time axis. This is safe here
%   because the instantaneous-frequency content of a band-limited
%   analytic signal is bounded by the band's own upper edge (e.g. <=25Hz
%   for beta), well below TARGET_RATE's Nyquist.

if nargin < 1 || isempty(subjID)
    error('Subject ID is required');
end
if nargin < 2 || isempty(lockType) || ~ismember(lockType, {'stim', 'resp'})
    error('lockType is required (''stim'' or ''resp'')');
end
if nargin < 3 || isempty(frequency_band)
    frequency_band = 'beta';
end
if nargin < 4 || isempty(volumetric_resolution)
    volumetric_resolution = 8;
end

% Canonical band edges + what to save
band_table = struct( ...
    'theta',     struct('range', [4  8],  'savePhase', true), ...
    'alpha',     struct('range', [8  12], 'savePhase', true), ...
    'beta',      struct('range', [15 25], 'savePhase', true), ...
    'lowgamma',  struct('range', [30 50], 'savePhase', false), ...
    'highgamma', struct('range', [65 95], 'savePhase', false) ...
);
valid_bands = fieldnames(band_table);
if ~ismember(frequency_band, valid_bands)
    error('Invalid frequency band: %s. Must be one of: %s', frequency_band, strjoin(valid_bands, ', '));
end
bandSpec  = band_table.(frequency_band);
freq_range = bandSpec.range;
savePhase  = bandSpec.savePhase;

% Single storage rate shared by ALL bands -- see header comment. 200Hz
% (Nyquist 100Hz) comfortably covers highgamma's 95Hz upper edge.
TARGET_RATE = 200;

restoredefaultpath;
clearvars -except subjID lockType frequency_band volumetric_resolution band_table valid_bands bandSpec freq_range savePhase TARGET_RATE;
close all; clc;

%% Setup paths (vader / local only -- no HPC)
fieldtrip_path = '/d/DATD/hyper/software/fieldtrip-20250318/';
project_path   = '/d/DATD/hyper/experiments/Mrugank/meg_mgs';
data_base_path = '/d/DATD/datd/MEG_MGS/MEG_BIDS/derivatives';

addpath(fieldtrip_path);
addpath(genpath(project_path));
ft_defaults;
ft_hastoolbox('spm12', 1);

fprintf('=== G04: Band Amplitude(+Phase) Extraction ===\n');
fprintf('Subject: %d | lockType: %s | band: %s (%.1f-%.1f Hz) | resolution: %dmm\n', ...
    subjID, lockType, frequency_band, freq_range(1), freq_range(2), volumetric_resolution);

%% Paths
source_data_path = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', ...
    sprintf('sub-%02d_task-mgs_sourceSpaceData_%d_%s.mat', subjID, volumetric_resolution, lockType));

output_dir = fullfile(data_base_path, sprintf('sub-%02d', subjID), 'sourceRecon', 'freqSpace');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
band_data_path = fullfile(output_dir, sprintf('sub-%02d_task-mgs_%s_allTargets_%d_%s.mat', ...
    subjID, frequency_band, volumetric_resolution, lockType));

if exist(band_data_path, 'file')
    fprintf('%s data already exists at: %s\nSkipping (delete to reprocess).\n', frequency_band, band_data_path);
    return;
end

if ~exist(source_data_path, 'file')
    error('Broadband source space data not found at: %s\nPlease run G03_SourceLocalizationBroadband first!', source_data_path);
end
fprintf('Loading broadband source space data from: %s\n', source_data_path);
load(source_data_path); % provides sourcedataCombined, inside_pos, volumetric_resolution, lockType

fprintf('Loaded source space data:\n');
fprintf('  Total trials: %d\n', length(sourcedataCombined.trial));
fprintf('  Inside sources: %d\n', length(inside_pos));

origFs = sourcedataCombined.fsample;
decimationFactor = max(1, round(origFs / TARGET_RATE));
actualRate = origFs / decimationFactor;
fprintf('Post-Hilbert storage rate (shared across all bands): requested %dHz -> decimation factor %d -> actual %.2fHz\n', ...
    TARGET_RATE, decimationFactor, actualRate);

%% Process each target location separately (same convention as S03A)
target_locations = 1:10;
ampDataByTarget = cell(10, 1);
if savePhase
    phaseDataByTarget = cell(10, 1);
end

for target = target_locations
    fprintf('Processing target location %d...\n', target);

    trial_criteria = sourcedataCombined.trialinfo(:,2) == target;
    valid_trials = find(trial_criteria);
    if isempty(valid_trials)
        fprintf('  No trials found for target %d\n', target);
        continue;
    end
    fprintf('  Found %d trials for target %d\n', length(valid_trials), target);

    cfg = [];
    cfg.trials = valid_trials;
    sourcedataTarget = ft_selectdata(cfg, sourcedataCombined);

    % Bandpass filter (same 4th-order Butterworth convention as S03A)
    cfg = [];
    cfg.bpfilter = 'yes';
    cfg.bpfreq = freq_range;
    cfg.bpfilttype = 'but';
    cfg.bpfiltord = 4;
    sourcedataTarget_freq = ft_preprocessing(cfg, sourcedataTarget);

    % Hilbert transform -> analytic signal (transient; not persisted)
    hilbert_compute = @(x) hilbert(x')';
    analytic = cellfun(hilbert_compute, sourcedataTarget_freq.trial, 'UniformOutput', false);

    amplitudeData = sourcedataTarget_freq;
    amplitudeData.trial = cellfun(@(x) abs(x), analytic, 'UniformOutput', false);

    if savePhase
        phaseTrialFull = cellfun(@(x) angle(x), analytic, 'UniformOutput', false);
    end
    clear analytic sourcedataTarget sourcedataTarget_freq;

    % Anti-alias lowpass on amplitude only (smooth, non-wrapping signal;
    % phase is NOT filtered -- see header comment on wrap artifacts).
    if decimationFactor > 1
        cfg = [];
        cfg.lpfilter = 'yes';
        cfg.lpfreq = 0.8 * (TARGET_RATE / 2); % margin below the new Nyquist
        amplitudeData = ft_preprocessing(cfg, amplitudeData);
    end

    % Decimate amplitude and phase at identical sample indices so both
    % land on the same shared time axis per band.
    nTrialsThisTarget = length(amplitudeData.trial);
    for i = 1:nTrialsThisTarget
        idx = 1:decimationFactor:size(amplitudeData.trial{i}, 2);
        amplitudeData.trial{i} = single(amplitudeData.trial{i}(:, idx));
        amplitudeData.time{i}  = amplitudeData.time{i}(idx);
        if savePhase
            phaseTrialFull{i} = single(phaseTrialFull{i}(:, idx));
        end
    end
    amplitudeData.fsample = actualRate;
    ampDataByTarget{target} = amplitudeData;

    if savePhase
        phaseData = amplitudeData; % reuse the already-decimated time/label/trialinfo/fsample
        phaseData.trial = phaseTrialFull;
        phaseDataByTarget{target} = phaseData;
        clear phaseTrialFull phaseData;
    end
    clear amplitudeData;

    fprintf('  Target %d processing complete\n', target);
end

fprintf('%s band processing complete for all targets\n', frequency_band);

%% Save
if savePhase
    fprintf('Saving %s amplitude and phase results for all targets...\n', frequency_band);
else
    fprintf('Saving %s amplitude results for all targets...\n', frequency_band);
end
if savePhase
    save(band_data_path, 'ampDataByTarget', 'phaseDataByTarget', 'target_locations', ...
        'subjID', 'volumetric_resolution', 'lockType', 'frequency_band', 'freq_range', 'actualRate', '-v7.3');
else
    save(band_data_path, 'ampDataByTarget', 'target_locations', ...
        'subjID', 'volumetric_resolution', 'lockType', 'frequency_band', 'freq_range', 'actualRate', '-v7.3');
end
fprintf('Saved to: %s\n', band_data_path);

end
