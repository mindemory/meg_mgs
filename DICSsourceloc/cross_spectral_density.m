
% preprocess the data
cfg.channel   = {'MEG', '-46'};        % read all MEG channels except MLP31 and MLO12
                             % do baseline correction with the complete trial

dataFIC = preprocessing(cfg);
