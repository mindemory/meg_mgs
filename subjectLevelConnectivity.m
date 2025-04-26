clear; close all; clc;
%%%

for mm                      = 1:2
    for kk                  = 1:2
        if mm               == 1
            trlLocs         = 'left';
        elseif mm           == 2
            trlLocs         = 'right';
        end
        if kk               == 1
            seedLocs        = 'left';
        elseif kk           == 2
            seedLocs        = 'right';
        end
freqband                    = 'beta';
% trlLocs                     = 'left';
% seedLocs                    = 'right';
connectivityMetric          = 'coherence'; % Valid: coherence (coherency is computed for free),
                                           %        plv


load('NYUKIT_helmet.mat')
right_sensors               = {'AG001', 'AG007', 'AG008', 'AG020', 'AG022', 'AG024', ...
                               'AG034', 'AG036', 'AG050', 'AG055', 'AG065', 'AG098', ...
                               'AG103'};
left_sensors                = {'AG014', 'AG015', 'AG016', 'AG025', 'AG026', 'AG027', ...
                               'AG028', 'AG041', 'AG042', 'AG043', 'AG059', 'AG060', ...
                               'AG092'};
subList                   = [ 1,  2,  3,  4,  5,  6,  7,  9, 10, 12, 13, 15, ...
                             17, 18, 19, 23, 24, 25, 26, 27, 28, 29, 31, 32];
subList                   = [  23, 24, 25, 26, 27, 28, 29, 31, 32];
for sIdx                    = 1:length(subList)

    disp(['Running ' num2str(sIdx) ' of ' num2str(length(subList)) ' subjects.'])
    subjID                  = subList(sIdx);

    % Initalizing Paths
    bidsRoot                = '/System/Volumes/Data/d/DATD/datd/MEG_MGS/MEG_BIDS';
    taskName                = 'mgs';
    derivativesRoot         = [bidsRoot filesep 'derivatives/sub-' num2str(subjID, '%02d') '/meg'];
    subName                 = ['sub-' num2str(subjID, '%02d')];
    megRoot                 = [bidsRoot filesep subName filesep 'meg'];
    stimRoot                = [bidsRoot filesep subName filesep 'stimfiles'];
    fNameRoot               = [subName '_task-' taskName];
    stimLocked_fpath        = [derivativesRoot filesep fNameRoot '_stimlocked_lineremoved.mat'];
    TFR_fpath               = [derivativesRoot filesep fNameRoot '_TFRfull_evoked_lineremoved.mat'];
    connectivityRoot        = [derivativesRoot 'connectivity'];
    if ~exist(connectivityRoot, 'dir')
        mkdir(connectivityRoot)
    end
    connectivity_fPath      = [connectivityRoot filesep fNameRoot '_coh-' freqband '-' seedLocs 'sensors-' trlLocs '-trials.mat'];
    if ~exist(connectivity_fPath, 'file')
        disp('Connectivity does not exist, creating one ...')
    
        % Select frequency band of interest
        cfg                     = [];
        if strcmp(freqband, 'theta')
            cfg.frequency       = [5 8];
        elseif strcmp(freqband, 'alpha')
            cfg.frequency       = [8 12];
        elseif strcmp(freqband, 'beta')
            cfg.frequency       = [15 25];
        end
        if strcmp(trlLocs, 'left')
            load(TFR_fpath, 'TFR_fourier_left');
            TFR_this            = ft_selectdata(cfg, TFR_fourier_left);
            clearvars TFR_fourier_left;
        elseif strcmp(trlLocs, 'right')
            load(TFR_fpath, 'TFR_fourier_right');
            TFR_this            = ft_selectdata(cfg, TFR_fourier_right);
            clearvars TFR_fourier_right;
        end
    
        % Create channel combination
        if strcmp(seedLocs, 'left')
            seed_sensors        = left_sensors;
        elseif strcmp(seedLocs, 'right')
            seed_sensors        = right_sensors;
        end
        all_sensors             = TFR_this.label';
    
        cfg                     = [];
        cfg.channelcmb          = [repmat(seed_sensors', length(all_sensors), 1), ...
                                   repelem(all_sensors, length(seed_sensors))'];
        cfg.channelcmb          = cfg.channelcmb(~strcmp(cfg.channelcmb(:, 1), ...
                                                         cfg.channelcmb(:, 2)), :); % Remove the seed-seed pairs
        if strcmp(connectivityMetric, 'coherence')
            cfg.method          = 'coh';
            cfg.complex         = 'complex';
            cfg.partial         = 'no';
        end
    
        % Compute the connectivity
        connectivity            = ft_connectivityanalysis(cfg, TFR_this);
        % Convert the connectivity to required shape
        connectivity            = ft_checkdata(connectivity, 'cmbrepresentation', 'full');
    
        
        clearvars TFR_this;
        save(connectivity_fPath, 'connectivity', '-v7.3');
    else
        disp('Connectivity already exists, skipping this subject')
    end
    % clearvars TFR_fourier_left TFR_fourier_right;
end
    end
end
%%
% TFR_toRun                   = TFR_fourier_right;
% cfg                         = [];
% cfg.frequency               = [15 25];
% TFR_toRun                   = ft_selectdata(cfg, TFR_toRun);
% 
% seed_sensors                = right_sensors;
% all_sensors                 = TFR_toRun.label';
% 
% cfg                         = [];
% cfg.channelcmb = [repmat(seed_sensors', length(all_sensors), 1), ...
%                   repelem(all_sensors, length(seed_sensors))'];
% cfg.channelcmb = cfg.channelcmb(~strcmp(cfg.channelcmb(:,1), cfg.channelcmb(:,2)), :); % Remove seed-seed pairs
% cfg.method      = 'coh';       % Coherence magnitude (|coherency|)
% cfg.complex     = 'complex';   % Use 'complex' for coherency (retain phase)
% cfg.partial     = 'no';        % Use 'yes' for partial coherence
% 
% connectivity = ft_connectivityanalysis(cfg, TFR_toRun);
% 
% connectivity = ft_checkdata(connectivity, 'cmbrepresentation', 'full');
% coh_matrix = connectivity.cohspctrm;  % Dimensions: seeds × targets × freq × time
% 
% %%
% connectivityCoh  = connectivity;
% connectivityCoh.cohspctrm = imag(connectivity.cohspctrm);
% cfg = [];
% cfg.latency = [0 0.5]; 
% connectivity_short = ft_selectdata(cfg, connectivityCoh);
% 
% cfg = [];
% cfg.parameter = 'cohspctrm';
% cfg.refchannel = seed_sensors;
% cfg.layout = lay; 
% cfg.interactive = 'yes';
% cfg.colorbar = 'yes';
% cfg.colormap = '*RdBu';
% % cfg.zlim = [0.12 0.2];
% ft_topoplotER(cfg, connectivity_short);
