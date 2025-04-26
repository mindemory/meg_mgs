clear; close all; clc;
Fs = 1000;                          % Sampling rate
t = 0:1/Fs:1;                       % Time vector (1 second)

% Signal parameters
f1 = 10;                            % Primary frequency (20 Hz)
f2 = 10;                            % Secondary frequency (20 Hz)
phase_shift1 = 0;                   % Phase shift for f1
amp1 = 1.0;                         % Amplitude of first sinusoid
amp2 = 1.0;                         % Amplitude of second sinusoid
noise_level = 0.3;                  % Noise level for signal y

% Compute metrics for each phase shift
ph2 = -2*pi:0.1:2*pi;
cohVec = NaN(size(ph2));
imcohVec = NaN(size(ph2));
corrVec = NaN(size(ph2));
plvVec = NaN(size(ph2));

for i = 1:length(ph2)
    phase_shift2 = ph2(i);
    % Create signals
    x = amp1*sin(2*pi*f1*t + phase_shift1 + noise_level * randn(size(t)));
    y = amp2*sin(2*pi*f2*t + phase_shift2 + noise_level * randn(size(t)));
    
    % Compute correlation
    corrVec(i) = corr(x', y');

    % Phase-locking Value (PLV)
    analytic_x = hilbert(x);
    analytic_y = hilbert(y);
    phase_diff = angle(analytic_y) - angle(analytic_x);
    plvVec(i) = abs(mean(exp(1i*phase_diff)));

    % Compute coherence metrics
    [Sxy, f] = cpsd(x, y, hann(500), 250, 1024, Fs);
    Sxx = pwelch(x, hann(500), 250, 1024, Fs);
    Syy = pwelch(y, hann(500), 250, 1024, Fs);
    Cxy = Sxy ./ sqrt(Sxx .* Syy);
    cxy = abs(Cxy);
    icoh = imag(Cxy);
    
    % Find the values at the frequency of interest
    [~, idx] = min(abs(f - f1));
    cohVec(i) = cxy(idx);
    imcohVec(i) = icoh(idx);
end

% Create figure with 2×4 layout
figure('Position', [100, 100, 1200, 600], 'Renderer','painters');

% Top row: Two instances of simulated signals
% Instance 1: Phase shift = 0
phase_shift2_1 = 0;
x1 = amp1*sin(2*pi*f1*t + phase_shift1 + noise_level * randn(size(t)));
y1 = amp2*sin(2*pi*f2*t + phase_shift2_1 + noise_level * randn(size(t)));
subplot(2,4,1:2);
plot(t, x1, 'Color', [0, 0.7, 0.9], 'LineWidth', 1.5);
hold on;
plot(t, y1, 'Color', [1, 0.65, 0], 'LineWidth', 1.5);
title('Signals with 0° Phase Shift');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Signal x', 'Signal y');
ylim([-1.5 1.5]);

% Instance 2: Phase shift = π/2
phase_shift2_2 = pi/2;
x2 = amp1*sin(2*pi*f1*t + phase_shift1 + noise_level * randn(size(t)));
y2 = amp2*sin(2*pi*f2*t + phase_shift2_2 + noise_level * randn(size(t)));
subplot(2,4,3:4);
plot(t, x2, 'Color', [0, 0.7, 0.9], 'LineWidth', 1.5);
hold on;
plot(t, y2, 'Color', [1, 0.65, 0], 'LineWidth', 1.5);
title('Signals with 90° Phase Shift');
xlabel('Time (s)');
ylabel('Amplitude');
legend('Signal x', 'Signal y');
ylim([-1.5 1.5]);

% Define radian x-ticks
xticks_rad = [-2*pi, -3*pi/2, -pi, -pi/2, 0, pi/2, pi, 3*pi/2, 2*pi];
xtick_labels = {'-2\pi', '-3\pi/2', '-\pi', '-\pi/2', '0', '\pi/2', '\pi', '3\pi/2', '2\pi'};

% Bottom row: Four measurement plots
% 1. Coherence
subplot(2,4,5);
plot(ph2, cohVec, 'LineWidth', 1.5, 'Color', [0, 0.7, 0.9]);
% hold on; 
% plot(ph2, ones(size(ph2))*0.4, 'r--', 'LineWidth', 1); 
% hold off;
title('Magnitude Coherence');
xlabel('Phase Shift (rad)');
ylabel('Coherence');
ylim([0 1.1]);
set(gca, 'XTick', xticks_rad, 'XTickLabel', xtick_labels);

% 2. Imaginary Coherency
subplot(2,4,6);
plot(ph2, imcohVec, 'LineWidth', 1.5, 'Color', [0, 0.7, 0.9]);
% hold on; 
% plot(ph2, sin(ph2), 'r--', 'LineWidth', 1); 
% hold off;
title('Imaginary Coherency');
xlabel('Phase Shift (rad)');
ylabel('Im(Coherency)');
ylim([-1.1 1.1]);
set(gca, 'XTick', xticks_rad, 'XTickLabel', xtick_labels);

% 3. Correlation
subplot(2,4,7);
plot(ph2, corrVec, 'LineWidth', 1.5, 'Color', [0, 0.7, 0.9]);
hold on; 
plot(ph2, cos(ph2), 'r--', 'LineWidth', 1); 
hold off;
title('Pearson Correlation');
xlabel('Phase Shift (rad)');
ylabel('Correlation');
ylim([-1.1 1.1]);
set(gca, 'XTick', xticks_rad, 'XTickLabel', xtick_labels);

% 4. PLV
subplot(2,4,8);
plot(ph2, plvVec, 'LineWidth', 1.5, 'Color', [0, 0.7, 0.9]);
% hold on; 
% plot(ph2, ones(size(ph2))*0.4, 'r--', 'LineWidth', 1); 
% hold off;
title('Phase-Locking Value (PLV)');
xlabel('Phase Shift (rad)');
ylabel('PLV');
ylim([0 1.1]);
set(gca, 'XTick', xticks_rad, 'XTickLabel', xtick_labels);

% Adjust spacing between subplots
set(gcf, 'Position', [100, 100, 1200, 500]);
