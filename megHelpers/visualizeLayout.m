
cfg                         = [];
cfg.layout                  = 'CTF151_helmet';
ctf151                      = ft_prepare_layout(cfg);

layNew = lay;
layNew.outline = ctf151.outline;
layNew.mask = ctf151.mask;
chIdx = 1:(length(lay.label)-2);
    
layNew.pos(chIdx, :) = lay.pos(chIdx, :) .* 1.05;
% layNew.mask{1, 1} = lay.mask{1, 1} .* 1.5;
% layNew.outline{1, 5} = lay.mask{1, 1};
% layNew.pos = lay.pos .* 1.0;
% layNew.pos(chIdx, 1) = layNew.pos(chIdx, 1) + 0.01;
layNew.pos(chIdx, 2) = layNew.pos(chIdx, 2) .* 1.08;%+ 0.02;

layNew = lay;
figure(); plot(layNew.mask{1, 1}(:, 1), layNew.mask{1, 1}(:, 2), 'b');
hold on;
for i = 1:5
    plot(layNew.outline{1, i}(:, 1), layNew.outline{1, i}(:, 2), 'b');
end

plot(layNew.pos(:, 1), layNew.pos(:, 2), 'ko')


lay = layNew;
save('NYUKIT_helmet.mat', "lay")