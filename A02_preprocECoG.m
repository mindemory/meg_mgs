function A02_preprocECoG(subjID)
clearvars -except subjID; close all; clc;
warning('off', 'all');

%% Initialization
p.subjID          = subjID;
[p]               = initialization(p, 'ecog');

RunECoG(p);

end