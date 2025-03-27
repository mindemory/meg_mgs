function [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10] = extractPowByCond(TFRleft, TFRright)

for i                                              = 1:10
    eval(['p' num2str(i) ' = [];']);
end

for cond                                           = 4:8
    cfg                                            = [];
    cfg.trials                                     = find(TFRleft.trialinfo(:, 2) == cond);
    powData                                        = ft_selectdata(cfg, TFRleft);

    cfg                                            = [];
    cfg.avgoverrpt                                 = 'yes';  % This averages over the rpttap dimension (trials)
    eval(['p' num2str(cond) ' = ft_freqdescriptives(cfg, powData);']);
end

for cond                                           = [1 2 3 9 10]
    cfg                                            = [];
    cfg.trials                                     = find(TFRright.trialinfo(:, 2) == cond);
    powData                                        = ft_selectdata(cfg, TFRright);

    cfg                                            = [];
    cfg.avgoverrpt                                 = 'yes';  % This averages over the rpttap dimension (trials)
    eval(['p' num2str(cond) ' = ft_freqdescriptives(cfg, powData);']);
end

end