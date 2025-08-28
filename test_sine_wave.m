% Test script to plot a sine wave
% This script creates a simple sine wave plot to verify MATLAB IDE functionality

% Clear workspace and command window
clear all;
clc;

% Define time vector (0 to 2*pi with 1000 points)
t = linspace(0, 2*pi, 1000);

% Create sine wave with frequency of 1 Hz
frequency = 1;
amplitude = 1;
sine_wave = amplitude * sin(2 * pi * frequency * t);

% Create the plot
figure('Name', 'Sine Wave Test', 'NumberTitle', 'off');
plot(t, sine_wave, 'b-', 'LineWidth', 2);
grid on;
xlabel('Time (radians)');
ylabel('Amplitude');
title('Test Sine Wave - MATLAB IDE Verification');
axis tight;

% Add some text to confirm the plot was created
text(pi, 0.5, 'MATLAB IDE is working!', 'FontSize', 14, 'Color', 'red', ...
     'HorizontalAlignment', 'center', 'FontWeight', 'bold');

% Display some basic information
fprintf('Sine wave test completed successfully!\n');
fprintf('Frequency: %d Hz\n', frequency);
fprintf('Amplitude: %d\n', amplitude);
fprintf('Number of points: %d\n', length(t));

% Optional: Add a second sine wave with different frequency for comparison
hold on;
frequency2 = 2;
sine_wave2 = 0.5 * sin(2 * pi * frequency2 * t);
plot(t, sine_wave2, 'r--', 'LineWidth', 1.5);
legend('1 Hz', '2 Hz', 'Location', 'best');

fprintf('Second sine wave (2 Hz) added for comparison.\n');
