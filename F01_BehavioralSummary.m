clear; close all; clc;
warning('off', 'all');

addpath('/d/DATD/hyper/software/fieldtrip-20220104/');
ft_defaults;
addpath(genpath('/d/DATD/hyper/experiments/Mrugank/meg_mgs'));

%% Figure 1B (behavioral part)
valIdx = find(ii_sess.tarlocCode ~= 11);
tarlocs = ii_sess.tarloc(valIdx);
errs = ii_sess.i_sacc_err(valIdx);
RTs = ii_sess.i_sacc_rt(valIdx);

unique_angs = unique(tarlocs);
errBinned = zeros(length(unique_angs));
RTBinned = zeros(length(unique_angs));

for i = 1:length(unique_angs)
    errBinned(i) = mean(errs(tarlocs==unique_angs(i)), "all", 'omitmissing');
    RTBinned(i) = mean(RTs(tarlocs==unique_angs(i)), "all", 'omitmissing');
end

figure();
subplot(1, 2, 1)
polarplot(deg2rad(unique_angs), errBinned, '*');
title('Memory error (dva)')
subplot(1, 2, 2)
polarplot(deg2rad(unique_angs), RTBinned, '*');
title('Saccade RT (ms)')

