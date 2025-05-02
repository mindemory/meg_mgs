% function [cohMat, icohMat, pcohMat, ipcohMat, plvMat, pliMat] ...
function [cohMat, icohMat, plvMat, pliMat] ...
                                  = computeConnectivitySourceSpace(sourcedata, seedSources, TOI)
cohMat                            = NaN(length(sourcedata.trial), size(sourcedata.trial{1},1), length(seedSources));
icohMat                           = NaN(length(sourcedata.trial), size(sourcedata.trial{1},1), length(seedSources));
% pcohMat                           = NaN(length(sourcedata.trial), size(sourcedata.trial{1},1), length(seedSources));
% ipcohMat                          = NaN(length(sourcedata.trial), size(sourcedata.trial{1},1), length(seedSources));
plvMat                            = NaN(length(sourcedata.trial), size(sourcedata.trial{1},1), length(seedSources));
pliMat                            = NaN(length(sourcedata.trial), size(sourcedata.trial{1},1), length(seedSources));
for trlIdx                        = 1:length(sourcedata.trial)
    sourceThTrial                 = sourcedata.trial{trlIdx}(:, TOI);

    % Compute amplitude coherency
    sourceautoSpectra             = mean(abs(sourceThTrial).^2, 2); % [nsources x 1]
    seedAutoSpectra               = mean(abs(sourceThTrial(seedSources, :)).^2, 2); % [nseeds x 1]
    denominator                   = sqrt(sourceautoSpectra * seedAutoSpectra'); % [nsources x nseeds]
    coherencyThTrial              = (sourceThTrial * (sourceThTrial(seedSources, :)')) ./ ...
                                    (size(sourceThTrial,1) .* denominator);
    
    % Compute phase coherency
    % normSourceThTrial             = sourceThTrial ./ abs(sourceThTrial); % normalize to unit magnitude
    % sourceautoSpectraPhase        = mean(abs(normSourceThTrial).^2, 2); % should all be ~1, but just in case numerics
    % seedAutoSpectraPhase          = mean(abs(normSourceThTrial(seedSources, :)).^2, 2);
    % denominatorPhase              = sqrt(sourceautoSpectraPhase * seedAutoSpectraPhase'); % [nsources x nseeds]
    % phasecoherencyThTrial         = (normSourceThTrial * (normSourceThTrial(seedSources, :)')) ./ ...
    %                                 (size(normSourceThTrial,1) .* denominatorPhase);
    
    % Compute phase-difference
    sourcePhases                  = angle(sourceThTrial);                % [nsources x time]
    seedPhases                    = sourcePhases(seedSources, :);        % [nseeds x time]
    phaseDiff                     = permute(sourcePhases, [1, 3, 2]) - permute(seedPhases, [3, 1, 2]); % [nsources x nseeds x time]
    
    % Compute the coh, icoh, pcoh, ipcoh
    cohMat(trlIdx,:,:)            = abs(coherencyThTrial);
    icohMat(trlIdx,:,:)           = abs(imag(coherencyThTrial));
    % pcohMat(trlIdx,:,:)           = abs(phasecoherencyThTrial);
    % ipcohMat(trlIdx,:,:)          = abs(imag(phasecoherencyThTrial));
    plvMat(trlIdx,:,:)            = abs(mean(exp(1i * phaseDiff), 3));
    pliMat(trlIdx,:,:)            = abs(mean(sign(sin(phaseDiff)), 3));
end

end