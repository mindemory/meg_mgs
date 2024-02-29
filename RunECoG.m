function RunECoG(p)
% Created by Jacob & Mrugank (06/29/2023): A crude pipeline to process
% time-series data. If certain steps are desired, putting a stop point
% where necessary and running the step is ideal. The code is optimized for
% run time and memory overload, hence make sure that steps desired are not
% attempting to produce files that already exist. The following steps are
% followed:
%         1. CONCATENATE ECoG data from multiple sessions if they exist for 
%            a given subject.
%         2. PREPROCESS ECoG data. The steps as of now (06/29/2023)
%            included are highpass filter at 1 Hz and a lowpass filter at 100
%            Hz.
%         3. Create EVENT TABLE from the DC channels flags and concatenate
%            event tables if multiple recordings.
%         4. EPOCH data with locking done at fixation.
%         5. ARTIFACT REJECTION to remove bad trials and channels for each
%            subject. Make sure to change the flagged trials and channels
%            which are hard-coded if any of the earlier steps are modified.
%            And remove these flagged trials and channels from epoched data
%            from the step above.
%         6. TIMELOCK ANLAYSIS: This is not yet implemented, goals for this
%            need to be clarified.
%         7. TIME-FREQUENCY ANALYSIS using wavelet convolution and hanning
%            taper separately for long and short trials.
%         8. PLOT DATA: Under progress, the number of options are
%            limitless, basically all crucial analysis would go in here.
% Trial structure:
%         - Fixation: 1s
%         - Sample: 0.2s
%         - Delay: 1.5s / 3.5s
%         - Response: 1s
%         - Feedback: 1s
%         - ITI: variable

% TODO (Mrugank): Add 
% File handler structure
fName.concatECoG           = [p.saveECoG filesep p.subjID '_concatECoG.edf'];
fName.rawECoG              = [p.saveECoG '/raw_ecog.mat'];
fName.DCECoG               = [p.saveECoG '/DC_ecog.mat'];
fName.evt_table            = [p.saveECoG '/evt_table.mat'];
fName.epochedECoG          = [p.saveECoG '/epoched_ecog.mat'];
fName.reepoc_data          = [p.saveECoG '/reepoc_data.mat'];
fName.ERP                  = [p.saveECoG '/ERP.mat'];
fName.ERPplot              = [p.saveECoG '/Figures'];
fName.ElecPlot             = [p.saveECoG '/Electrodes'];
fName.ElecERP              = [p.saveECoG '/ElecERP'];
fName.ElecTFA              = [p.saveECoG '/ElecTFA'];
fName.TFA                  = [p.saveECoG '/TFA.mat'];
fName.spectrum             = [p.saveECoG '/Spectrum'];
fName.spectrogram          = [p.saveECoG '/Spectrogram'];

% Loaders for edf vs nspike data
p.ECoGdir                      = [p.subjDIR filesep 'eegdata'];
if strcmp(p.subjID, 'NY098')
    dir_list                   = dir(p.ECoGdir);
    fNames_all                 = {dir_list.name};
    fNames_edf                 = fNames_all(endsWith(fNames_all, 'edf'));
    detect_not_at              = ~contains(fNames_edf, '@');
    fNames_edf                 = fNames_edf(detect_not_at);
    data_format                = 'clinical';
elseif strcmp(p.subjID, 'NY324') || strcmp(p.subjID, 'NY327') || strcmp(p.subjID, 'NY386') || strcmp(p.subjID, 'NY507') 
    dir_list                   = dir(p.ECoGdir);
    fNames_all                 = {dir_list.name};
    fNames_edf                 = fNames_all(endsWith(fNames_all, 'edf'));
    data_format                = 'clinical';
elseif strcmp(p.subjID, 'NY496') % EDF loading issue
    dir_list                   = dir(p.ECoGdir);
    fNames_all                 = {dir_list.name};
    fNames_first               = fNames_all(endsWith(fNames_all, 'edf'));
    fNames_clin1               = fNames_first(~contains(fNames_first, 'Clin2'));
    fNames_clin2               = fNames_first(contains(fNames_first, 'Clin2'));
    fNames_edf                 = [fNames_clin1, fNames_clin2];
    data_format                = 'clinical';
elseif strcmp(p.subjID, 'NY346') || strcmp(p.subjID, 'NY348') % Behavioral issues and EDF loading issues
    dir_list                   = dir(p.ECoGdir);
    block_list                 = {dir_list.name};
    detect_file                = contains(block_list, 'MGS_Block');
    block_files                = block_list(detect_file);
    fNames_edf                 = cell(1, numel(block_files));
    for bloc                   = 1:length(block_files)
        block_list             = dir([p.ECoGdir filesep block_files{bloc}]);
        block_names            = {block_list.name};
        block_final            = block_names(endsWith(block_names, 'edf'));
        fNames_edf{bloc}       = block_final;
    end
    data_format                = 'clinical';
elseif strcmp(p.subjID, 'NY442') % Issue with EDF file loading
    dir_list                   = dir(p.ECoGdir);
    block_list                 = {dir_list.name};
    detect_file                = contains(block_list, 'NY442');
    block_files                = block_list(detect_file);
    fNames_edf                 = {};
    for bloc                   = 1:length(block_files)
        block_list             = dir([p.ECoGdir filesep block_files{bloc}]);
        block_names            = {block_list.name};
        block_final            = block_names(endsWith(block_names, '.edf'));
        if ~isempty(block_final)
            fNames_edf         = [fNames_edf block_final];
        end
    end
    data_format                = 'clinical';
elseif strcmp(p.subjID, 'NY272')
    rec_blocks                 = [1, 2, 3, 4];
    fNames_rec                 = cell(length(rec_blocks), 1);
    fNames_flag                = cell(length(rec_blocks), 1);
    for ii = rec_blocks
        subdir_name            = num2str(ii, '%03d');
        subdir_list            = dir([p.ECoGdir filesep subdir_name]);
        flist                  = {subdir_list.name};
        rec_fname              = flist(endsWith(flist, ['_500hzMGS_' num2str(ii,'%02d'), '.low.nspike.dat']));
        flag_fname             = flist(endsWith(flist, ['_MGS_' num2str(ii,'%02d'), '-500Hz.dio.txt']));
        fNames_rec{ii}         = [subdir_name filesep rec_fname{1}];
        fNames_flag{ii}        = [subdir_name filesep flag_fname{1}];
    end
    data_format                = 'nspike';
elseif strcmp(p.subjID, 'NY276')
    rec_blocks                 = [1, 2];
    fNames_rec                 = cell(length(rec_blocks), 1);
    fNames_flag                = cell(length(rec_blocks), 1);
    for ii = rec_blocks
        subdir_name            = num2str(ii, '%03d');
        subdir_list            = dir([p.ECoGdir filesep subdir_name]);
        flist                  = {subdir_list.name};
        rec_fname              = flist(endsWith(flist, ['_500hzMGS_' num2str(ii,'%02d'), '.low.nspike.dat']));
        flag_fname             = flist(endsWith(flist, ['_MGS_' num2str(ii,'%02d'), '-500Hz.dio.txt']));
        fNames_rec{ii}         = [subdir_name filesep rec_fname{1}];
        fNames_flag{ii}        = [subdir_name filesep flag_fname{1}];
    end
    data_format                = 'nspike';
elseif strcmp(p.subjID, 'NY190')
    rec_blocks                 = [1, 2, 3, 4];
    fNames_rec                 = cell(length(rec_blocks), 1);
    fNames_flag                = cell(length(rec_blocks), 1);
    for ii = rec_blocks
        subdir_name            = num2str(ii, '%03d');
        subdir_list            = dir([p.ECoGdir filesep subdir_name]);
        flist                  = {subdir_list.name};
        rec_fname              = flist(endsWith(flist, ['_500hzMGS_' num2str(ii,'%02d'), '.low.nspike.dat']));
        flag_fname             = flist(endsWith(flist, ['_MGS_' num2str(ii,'%02d'), '-500Hz.dio.txt']));
        fNames_rec{ii}         = [subdir_name filesep rec_fname{1}];
        fNames_flag{ii}        = [subdir_name filesep flag_fname{1}];
    end
    data_format                = 'nspike';
elseif strcmp(p.subjID, 'NY297')
    rec_blocks                 = [2, 3, 4, 5];
    fNames_rec                 = cell(length(rec_blocks), 1);
    fNames_flag                = cell(length(rec_blocks), 1);
    for ii = rec_blocks
        subdir_name            = num2str(ii, '%03d');
        subdir_list            = dir([p.ECoGdir filesep subdir_name]);
        flist                  = {subdir_list.name};
        rec_fname              = flist(endsWith(flist, ['_500hz' num2str(ii,'%02d'), '.low.nspike.dat']));
        flag_fname             = flist(endsWith(flist, ['_mgs_' num2str(ii,'%02d'), '-500Hz.dio.txt']));
        fNames_rec{ii}         = [subdir_name filesep rec_fname{1}];
        fNames_flag{ii}        = [subdir_name filesep flag_fname{1}];
    end
    data_format                = 'nspike';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Concatenate data from multiple sessions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist(fName.concatECoG, 'file')
    if strcmp(data_format, 'clinical')
        P01_ConcatECoG(p, fNames_edf, fName);
    elseif strcmp(data_format, 'nspike')
        P01_ConcatECoG_nspike(p, fNames_rec, fName);
    end
else
    disp('Concat ECoG already exists. Skipping this step.')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2. Preprocess data (detrend, highpass filter and lowpass filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist(fName.rawECoG, 'file')
    raw_data                        = P02_PreprocECoG(fName, p);
else
    if ~exist(fName.reepoc_data, 'file')
        disp('Raw ECoG already exists. Loading existing data.')
        load(fName.rawECoG);
    else
        disp('Raw ECoG already exists, but not loading it.')
    end
end

% cfg = []; cfg.method = 'fastica';
% cfg.randomseed = 42;
% ica_comp = ft_componentanalysis(cfg, raw_data);

% cfg = []; 
% cfg.component = 1:length(ica_comp.label); 
% cfg.layout = ‘acticap-64_md.mat’; 
% cfg.comment = ‘no’; 
% ft_topoplotIC(cfg, ica_comp)

% cfg = []; 
% cfg.layout = ‘acticap-64_md.mat’; 
% cfg.viewmode = ‘component’; 
% cfg.ylim = [-200, 200]; 
% ft_databrowser(cfg, ica_comp)

% matcorr = corr(raw_data.trial{1}'); matcorr = matcorr.^2; figure(); imagesc(matcorr); xticks(1:length(raw_data.label));yticks(1:length(raw_data.label)); xticklabels(raw_data.label); yticklabels(raw_data.label);
% matcorr = corr(raw_data.trial{1}'); figure(); imagesc(matcorr); xticks(1:length(raw_data.label));yticks(1:length(raw_data.label)); xticklabels(raw_data.label); yticklabels(raw_data.label);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2a. Generate event-table 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist(fName.evt_table, 'file')
    if strcmp(data_format, 'clinical')
        DC_data                         = P02_ExtractDC(fName);
        evt_table                       = P03_EventExtraction(DC_data, p, fName);
    elseif strcmp(data_format, 'nspike')
        evt_table                       = P03_GenEvtTableNspike(fNames_rec, fNames_flag, p, fName);
    end
else
    disp('evt table already exists. Loading existing data.')
    load(fName.evt_table);
end

% cfg = []; cfg.viewmode = 'vertical'; ft_databrowser(cfg, epoc_data)

if ~exist(fName.reepoc_data, 'file')
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 3. Generate Event Table
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [bad_ch, bad_tr, raw_data]           = auto_reject(raw_data, evt_table, p);
    good_tr                              = setdiff(unique(evt_table(:, 7)), bad_tr);
    evt_table                            = evt_table(ismember(evt_table(:, 7), good_tr), :);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 4. Epoching Data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Epoched data trialinfo has 5 columns:
    % Column 1: Long (1) or short (1) trials
    % Column 2: Polarangle of stimulus (0-180)
    % Column 3: TarlocCode (1=0, 2=25, 3=50, 4=130, 5=155, 6=180, 7=205, 8=230,
    % 9=310, 10=335)
    % Column 4: Stimulus hemifield: Right (1) or Left (0)
    % Column 5: Stimulus hemifield: Top (1), Bottom (0), Horizontal meridian
    % (2)

%     if ~exist(fName.epochedECoG, 'file')
    epoc_data                       = struct;
    [~, epoc_data.all]              = P04_EpochECoG(p, evt_table, raw_data);
    [~, epoc_data.stimlocked]       = P04_EpochECoG(p, evt_table, raw_data, 'fixation', 'delay_1s');
    [~, epoc_data.responselocked]   = P04_EpochECoG(p, evt_table, raw_data, 'delay_end', 'iti');
%         save(fName.epochedECoG, "epoc_data", '-v7.3');
%     else
%         disp('Epoched data already exists. Loading existing epoched data.')
%         load(fName.epochedECoG)
%     end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 5. Average Reference
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Creating a common average reference electrode to rereference all other
    % electrodes excluding bad-channels, but leaving them in
    % Remove bad channels
%     cfg                                = [];
%     cfg.channel                        = setdiff(raw_data.label, bad_ch);
%     for epoc_type                      = ["all", "stimlocked", "responselocked"]
%         epoc_data.(epoc_type)          = ft_selectdata(cfg, epoc_data.(epoc_type));
%     end
% 
%     cfg                                 = []; 
%     cfg.reref                           = 'yes'; 
%     cfg.refchannel                      = setdiff(raw_data.label, bad_ch); 
%     for epoc_type                       = ["all", "stimlocked", "responselocked"]
%         epoc_data.(epoc_type)           = ft_preprocessing(cfg, epoc_data.(epoc_type));
%     end

    cfg                                  = [];
    cfg.trials                           = find(epoc_data.all.trialinfo(:,1) == 0);
    epoc_data.short                      = ft_selectdata(cfg, epoc_data.all);
    cfg.trials                           = find(epoc_data.all.trialinfo(:,1) == 1);
    epoc_data.long                       = ft_selectdata(cfg, epoc_data.all);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 6. Timelock analysis
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % TODO: Think about how to do this the best. The goal is to create
    % time-locked trial structured with stimulus-locked delay and
    % response-locked delay and combining these time-courses accounting for the
    % variable delay.
    % Timelocking preproccessed data
    timelock_data                         = struct();
    for epoc_type                         = ["short", "long", "stimlocked", "responselocked"]
        timelock_data.(epoc_type)         = P06_TimeLockData(epoc_data.(epoc_type));
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 7. Re-Epoching Data
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    reepoc_data                          = struct;
    for epoc_type                        = ["stimlocked", "responselocked"]
        [reepoc_conds, reepoc_data.(epoc_type)] = P05_ReepochECoG(timelock_data.(epoc_type));
    end
    % Creating Correlation matrices between electrode channels
    % matcorr_long = corr(epoc_long.trial{1}'); matcorr_long = matcorr_long.^2; figure(); imagesc(matcorr_long); xticks(1:length(epoc_long.label));yticks(1:length(epoc_long.label)); xticklabels(epoc_long.label); yticklabels(epoc_long.label);
    % matcorr_short = corr(epoc_short.trial{1}'); matcorr_short = matcorr_short.^2; figure(); imagesc(matcorr_short); xticks(1:length(epoc_short.label));yticks(1:length(epoc_short.label)); xticklabels(epoc_short.label); yticklabels(epoc_short.label);
    % matcorr_stim = corr(epoc_stimlocked.trial{1}'); matcorr_stim = matcorr_stim.^2; figure(); imagesc(matcorr_stim); xticks(1:length(epoc_stimlocked.label));yticks(1:length(epoc_stimlocked.label)); xticklabels(epoc_stimlocked.label); yticklabels(epoc_stimlocked.label);
    % matcorr_response = corr(epoc_responselocked.trial{1}'); matcorr_response = matcorr_response.^2; figure(); imagesc(matcorr_response); xticks(1:length(epoc_responselocked.label));yticks(1:length(epoc_responselocked.label)); xticklabels(epoc_responselocked.label); yticklabels(epoc_responselocked.label);
    save(fName.reepoc_data, 'reepoc_data', 'bad_ch', 'bad_tr', 'reepoc_conds', '-v7.3')
else
    if ~exist(fName.TFA, 'file')
        disp('Reepoched-data already exists. Loading it.')
        load(fName.reepoc_data)
    else
        disp('Reepoched-data already exists, but not loading it.')
        load(fName.reepoc_data, 'bad_ch');
        load(fName.reepoc_data, 'bad_tr');
        load(fName.reepoc_data, 'reepoc_conds');
    end
end

% % Testing out the power spectrum
% cfg = [];
% cfg.output  = 'pow';
% cfg.method  = 'mtmfft';
% cfg.taper   = 'boxcar';
% cfg.foi     = 0.5:1:100;
% TFA_info   = ft_freqanalysis(cfg, timelock_responselocked);
% 
% figure(); 
% plot(TFA_info.freq, 10*log10(TFA_info.powspctrm))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 7. Event-related Potential analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist(fName.ERP, 'file')
    ERP                                = struct;
    for epoc_type                      = ["stimlocked", "responselocked"]
        for cond                       = reepoc_conds
            ERP.(epoc_type).(cond{1})  = P07_CreateERP(reepoc_data.(epoc_type).(cond{1}), 0);
        end
    end
    save(fName.ERP, 'ERP', '-v7.3')
else
    disp('ERPs already exists. Loading existing file')
    load(fName.ERP)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 8. Plot ERP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Making ERP plot folder for individual condition folders
% if ~isfolder(fName.ERPplot)
%     mkdir(fName.ERPplot);
%     for cond = ["left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"]
%         condFile = fullfile(fName.ERPplot, cond);
%         if ~isfolder(condFile)
%             mkdir(condFile);
%             F01_ERPplot(condFile, ERP, cond);
%         end
%     end
% end

% % ERP plots for right visual hemifield
% if ~isfolder(fName.ERPplot_r)
%     mkdir(fName.ERPplot_r);
%     F01_ERPplot(fName.ERPplot_r, ERP_r);
% else
%     disp('Right ERP figures already exist.')
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 9. Time-frequency analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist(fName.TFA, 'file')
    POW                                = struct;
    for epoc_type                      = ["stimlocked", "responselocked"]
        for cond                       = reepoc_conds
            is_response                = strcmp(epoc_type, 'responselocked');
            [POW.(epoc_type).(cond{1}),~,~]= P07_RunTFA(reepoc_data.(epoc_type).(cond{1}), is_response, 'POW', 0);
        end
    end
    save(fName.TFA, 'POW', '-v7.3')
else
    disp('TFAs already exists. Loading existing file')
    load(fName.TFA)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 8. Plot Electrodes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Electrode plots for ERPs and TFAs
% if ~isfolder(fName.ElecPlot)
%     mkdir(fName.ElecPlot);
%     F01_ElecPlot(fName.ElecPlot, ERP, POW, p);
% else
%     disp('Electrode figures already exist.')
% end

% Electrode plots for ERPs
if ~isfolder(fName.ElecERP)
    mkdir(fName.ElecERP);
    F01_ElecERP(fName.ElecERP, ERP, POW, p);
else
    disp('ERP figures already exist.')
end

% Electrode plots for TFAs
if ~isfolder(fName.ElecTFA)
    mkdir(fName.ElecTFA);
    F01_ElecTFA(fName.ElecTFA, POW, p);
else
    disp('TFA figures already exist.')
end

temp = 4;
