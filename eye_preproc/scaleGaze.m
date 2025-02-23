function ii_data = scaleGaze(ii_data, ii_params, preproc_fn)

%%% Created by Mrugank (01/28/2025):
% The purpose is to fix gaze data if online calibration were off.
% The way it works 

% Extract relevant data
X                    = ii_data.X;
Y                    = ii_data.Y;
TarX                 = ii_data.TarX;
TarY                 = ii_data.TarY;
XDAT                 = ii_data.XDAT;
resolution           = ii_params.resolution;

% Flip TarY (this is a hack since Y is flipped, note the original TarY is
% not flipped here).
TarY                 = resolution(2)/2 - (TarY - resolution(2)/2);

feedbackSamps        = find(XDAT == 5);
targCoords           = unique([TarX(feedbackSamps) TarY(feedbackSamps)], 'rows');
gazeCoords           = [];
varGaze              = [];
for i = 1:length(targCoords)
    % Finding gaze coordinates during feedback epoch for this target
    % location
    relevSamps       = find((abs(TarX - targCoords(i, 1)) <= 0.01) & ...
                            (abs(TarY - targCoords(i, 2)) <= 0.01) & ...
                            (XDAT == 5));
    xData            = X(relevSamps);
    yData            = Y(relevSamps);
    rData            = (xData - mean(xData, 'all', 'omitnan')).^2 + ...
                        (yData - mean(yData, 'all', 'omitnan')).^2;
    gazeCoords       = [gazeCoords; [median(xData, 'all', 'omitnan') ...
                                        median(yData, 'all', 'omitnan')]];
    varGaze          = [varGaze; [var(rData, [], 'all', 'omitnan')]];
end
% Check if any of the tars have too high a variance, probably don't wanna
% use them
meanvarGaze          = median(varGaze);
goodIdx              = find(varGaze <= 10 * meanvarGaze);
length(goodIdx)
targCoords           = targCoords(goodIdx, :);
gazeCoords           = gazeCoords(goodIdx, :);


% Adding a mean estimate for gaze from fixation epochs
targCoords           = [targCoords; resolution./2];
fixSamps             = find(ismember(XDAT, ii_params.drift_epoch));
xFix                 = X(fixSamps);
yFix                 = Y(fixSamps);
gazeCoords           = [gazeCoords; [median(xFix, 'all', 'omitnan') ...
                                        median(yFix, 'all', 'omitnan')]];

% Project target and gaze estimates into 3D with z = 1
T                    = [targCoords ones(length(targCoords), 1)];
E                    = [gazeCoords ones(length(gazeCoords), 1)];

% Affine transformation in 2D is now simply rotation
A                    = T' * pinv(E)';

% Project oldData into 3D and apply the affine transformation to this data
oldData              = [X Y ones(length(X), 1)]';
newData              = (A * oldData)';
% Extract the x and y from this data and this will be the data we work with
newX                 = newData(:, 1);
newY                 = newData(:, 2);

ii_data.X            = newX;
ii_data.Y            = newY;

fig = figure('visible','off');
subplot(1, 2, 1)
plot(X, Y, 'ro', 'MarkerSize', 2); hold on; 
plot(targCoords(:, 1), targCoords(:, 2), 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
title('Original Data')

subplot(1, 2, 2)
plot(newX, newY, 'ro',  'MarkerSize', 2); hold on;
plot(targCoords(:, 1), targCoords(:, 2), 'bs', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
title('Transformed Data')

saveas(fig, [preproc_fn '_transformed_gaze.png'],'png');





end