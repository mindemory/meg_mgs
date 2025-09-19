
addpath(genpath('/d/DATD/hyper/toolboxes/CircStat2012a'));
TFR_left                   = TFRleft_power;
TFR_right                  = TFRright_power;


freq_band                  = [8 14];

% Select frequency indices within the band
freq_idx                   = find(TFR_left.freq >= freq_band(1) & ...
                                  TFR_left.freq <= freq_band(2));

% Compute circular mean phase across the frequency band
phase_left                 = TFR_left.phaseangle; % Extract phase from complex data
phase_right                = TFR_right.phaseangle;
% phase_avg                  = squeeze(circ_mean(phase_data, [], [1, 3]));
phase_avg_left             = squeeze(circ_mean(phase_left, [], [1, 3]));
phase_avg_right            = squeeze(circ_mean(phase_right, [], [1, 3]));
phase_diff                 = angle(exp(1i * (phase_avg_left - phase_avg_right))); 


% phase_avg                  = squeeze(circ)

tempTFR                    = TFR_left;
tempTFR.dimord             = 'chan_freq_time';
tempTFR                    = rmfield(tempTFR, {'phaseangle', 'trialinfo'});
% tempTFR.powspctrm          = phase_avg;
tempTFR.powspctrm          = repmat(reshape(phase_diff, [size(tempTFR.label,1), 1, length(TFR.time)]), [1, 53, 1]);

% Create a movie of phase topography over time
figure;
hold on;
dt                         = 0.1;
t_array                    = -0.5:dt:1.7;
for t                      = t_array
    cfg                    = [];
    cfg.figure             = 'gcf';
    cfg.parameter          = 'powspctrm'; % Although we're plotting phase, FieldTrip expects 'powspctrm
    cfg.layout             = lay;
    cfg.xlim               = [t t+dt];
    cfg.zlim               = [-pi pi];
    cfg.colormap           = jet;
    cfg.colorbar           = 'yes';

    ft_topoplotTFR(cfg, tempTFR);
    title(sprintf('Phase Topography at %.3f s', t));
    pause(0.01); % Adjust speed
end