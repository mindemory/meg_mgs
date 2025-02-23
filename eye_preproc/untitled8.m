meanX = mean(ii_data.X(ismember(ii_data.XDAT, [1, 2, 3])), 'all', 'omitnan');
meanY = mean(ii_data.Y(ismember(ii_data.XDAT, [1, 2, 3])), 'all', 'omitnan');

demeanX = ii_data.X - meanX;
demeanY = ii_data.Y - meanY;

radialResp = ((ii_data.X - meanX).^2) + ((ii_data.Y - meanY).^2);
medianRadialresp = median(radialResp(ismember(ii_data.XDAT, [5])), 'all', 'omitnan');
% medianXresp = median(ii_data.X(ismember(ii_data.XDAT, [4, 5])), 'all', 'omitnan');
% medianYresp = median(ii_data.Y(ismember(ii_data.XDAT, [4, 5])), 'all', 'omitnan');
% medianTarXresp = median(iidata.TarX)
% 
% XX = ii_data.X - meanX + ii_params.resolution(1) / 2;
% YY = ii_data.Y - meanY + ii_params.resolution(2) / 2;
% 
% figure(); plot(XX); hold on; plot(YY); plot(ii_data.TarX); plot(ii_data.TarY)
meanTarX = mean(ii_data.TarX(ismember(ii_data.XDAT, [1, 2, 3])), 'all', 'omitnan');
meanTarY = mean(ii_data.TarY(ismember(ii_data.XDAT, [1, 2, 3])), 'all', 'omitnan');
radialTar = ((ii_data.TarX - meanTarX).^2) + ((ii_data.TarY - meanTarY).^2);
figure(); plot(radialTar); hold on; plot(radialResp)



%% Trying out affine estimation
X_center = mean(ii_data.X(ismember(ii_data.XDAT, [1, 2, 3])), 'all', 'omitnan');
Y_center = mean(ii_data.Y(ismember(ii_data.XDAT, [1, 2, 3])), 'all', 'omitnan');

feedbackSamps = find(ii_data.XDAT == 5);
targCoords = unique([ii_data.TarX(feedbackSamps) ii_data.TarY(feedbackSamps)], 'rows');
gazeCoords = [];
for i = 1:length(targCoords)
    relevSamps = find((abs(ii_data.TarX - targCoords(i, 1)) <= 0.01) & ...
                      (abs(ii_data.TarY - targCoords(i, 2)) <= 0.01) & ...
                      (ii_data.XDAT == 5));
    xData = ii_data.X(relevSamps);
    yData = ii_data.Y(relevSamps);
    gazeCoords = [gazeCoords; [median(xData, 'all', 'omitnan') median(yData, 'all', 'omitnan')]];
end
targCoords = [targCoords; ii_params.resolution./2];
fixSamps = find(ismember(ii_data.XDAT, [1, 2, 3]));
xFix = ii_data.X(fixSamps);
yFix = ii_data.Y(fixSamps);
gazeCoords = [gazeCoords; [median(xFix, 'all', 'omitnan') median(yFix, 'all', 'omitnan')]];

T = [targCoords ones(length(targCoords), 1)];
E = [gazeCoords ones(length(gazeCoords), 1)];

A = T' * pinv(E)';

oldData = [ii_data.X ii_data.Y ones(length(ii_data.X), 1)]';
newData = (A * oldData)';
newX = newData(:, 1);
newY = newData(:, 2);

figure(); 
subplot(1, 2, 1)
plot(ii_data.X, ii_data.Y, 'ro'); hold on; 
plot(targCoords(:, 1), targCoords(:, 2), 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');

subplot(1, 2, 2)
plot(newX, newY, 'ro'); hold on;
plot(targCoords(:, 1), targCoords(:, 2), 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
