% Clear workspace
clear; close all; clc;

% Time vector
n = 0:199;

% Parameters
num_x1 = 10;              % Number of x1 series to generate
num_x2 = 10;              % Number of x2 series to generate
noise_level = 0.7;        % Moderate noise level
sine_amplitude = 0.3;     % Low sine wave amplitude
freq_sine = 0.05;         % Frequency of sine wave in Hz

% Initialize arrays to store autocorrelation, power spectra, and time-series
all_autocorr_x1 = [];
all_autocorr_x2 = [];
all_power_spectra_x1 = [];
all_power_spectra_x2 = [];
all_x1 = zeros(num_x1, length(n));   % Store x1 time-series
all_x2 = zeros(num_x2, length(n));   % Store x2 time-series

% Frequency axis for FFT (positive frequencies)
N = length(n);
freq = (0:N/2-1) / N;  % Frequency axis for first half

% Generate multiple x1's and x2's
for i = 1:num_x1
    % Generate x1 with sine wave and structured noise (random walk for long-term trends)
    x1 = sine_amplitude * sin(2 * pi * n * freq_sine) + noise_level * randn(size(n)); 
    
    % Add a random walk (integrated random noise) to introduce structured temporal variation
    random_walk_x1 = cumsum(randn(size(n)) * 0.1);  % Slow trend (random walk)
    x1 = x1 + random_walk_x1;  % Add random walk component to signal
    
    % Generate x2 with sine wave and noise, but no random walk (unstructured noise)
    x2 = sine_amplitude * sin(2 * pi * n * freq_sine) + noise_level * randn(size(n)); 
    
    % Store x1 and x2 time-series
    all_x1(i, :) = x1;
    all_x2(i, :) = x2;
    
    % Compute autocorrelation for x1 and x2
    autocorr_x1 = xcorr(x1, 'coeff');
    autocorr_x2 = xcorr(x2, 'coeff');
    
    % Store autocorrelation
    all_autocorr_x1 = [all_autocorr_x1; autocorr_x1];
    all_autocorr_x2 = [all_autocorr_x2; autocorr_x2];
    
    % Compute spectral power for x1 and x2
    power_x1 = abs(fft(x1)).^2;
    power_x2 = abs(fft(x2)).^2;
    
    % Store power spectra (only positive frequencies)
    all_power_spectra_x1 = [all_power_spectra_x1; power_x1(1:end/2)];
    all_power_spectra_x2 = [all_power_spectra_x2; power_x2(1:end/2)];
end

% Compute mean autocorrelation across all x1's and x2's
mean_autocorr_x1 = mean(all_autocorr_x1, 1);
mean_autocorr_x2 = mean(all_autocorr_x2, 1);

% Compute mean power spectrum across all x1's and x2's
mean_power_spectrum_x1 = mean(all_power_spectra_x1, 1);
mean_power_spectrum_x2 = mean(all_power_spectra_x2, 1);

% Compute average time-series for x1 and x2
avg_x1 = mean(all_x1, 1);
avg_x2 = mean(all_x2, 1);

% Compute standard deviation (variability) for time-series, autocorrelations, and power spectra
std_x1 = std(all_x1, 0, 1);    % Standard deviation of time-series for x1
std_x2 = std(all_x2, 0, 1);    % Standard deviation of time-series for x2

std_autocorr_x1 = std(all_autocorr_x1, 0, 1);    % Std for autocorrelation of x1
std_autocorr_x2 = std(all_autocorr_x2, 0, 1);    % Std for autocorrelation of x2

std_power_spectrum_x1 = std(all_power_spectra_x1, 0, 1);    % Std for power spectrum of x1
std_power_spectrum_x2 = std(all_power_spectra_x2, 0, 1);    % Std for power spectrum of x2

% Plot mean autocorrelation and power spectrum
figure;

% Plot mean time-series for x1 and x2
subplot(3, 2, 1);
plot(n, avg_x1, 'b', 'LineWidth', 1.5); hold on;
plot(n, avg_x2, 'r', 'LineWidth', 1.5);
title('Average Time-Series');
xlabel('Time');
ylabel('Amplitude');
legend({'x1', 'x2'}, 'Location', 'Best');

% Plot shaded region for variability in time-series for x1 and x2
subplot(3, 2, 2);
fill([n, fliplr(n)], [avg_x1 + std_x1, fliplr(avg_x1 - std_x1)], 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none'); hold on;
fill([n, fliplr(n)], [avg_x2 + std_x2, fliplr(avg_x2 - std_x2)], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(n, avg_x1, 'b', 'LineWidth', 1.5); hold on;
plot(n, avg_x2, 'r', 'LineWidth', 1.5);
title('Time-Series with Std Dev (Shaded Area)');
xlabel('Time');
ylabel('Amplitude');
legend({'x1', 'x2'}, 'Location', 'Best');

% Plot shaded region for variability in autocorrelation for x1 and x2
subplot(3, 2, 3);
lags = -N + 1:N - 1; % Lags for autocorrelation
fill([lags, fliplr(lags)], [mean_autocorr_x1 + std_autocorr_x1, fliplr(mean_autocorr_x1 - std_autocorr_x1)], 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none'); hold on;
fill([lags, fliplr(lags)], [mean_autocorr_x2 + std_autocorr_x2, fliplr(mean_autocorr_x2 - std_autocorr_x2)], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(lags, mean_autocorr_x1, 'b', 'LineWidth', 1.5); hold on;
plot(lags, mean_autocorr_x2, 'r', 'LineWidth', 1.5);
title('Mean Autocorrelation with Std Dev (Shaded Area)');
xlabel('Lag');
ylabel('Correlation');
legend({'x1', 'x2'}, 'Location', 'Best');

% Plot shaded region for variability in power spectrum for x1 and x2
subplot(3, 2, 4);
fill([freq, fliplr(freq)], [log(mean_power_spectrum_x1 + std_power_spectrum_x1), fliplr(log(mean_power_spectrum_x1 - std_power_spectrum_x1))], 'b', 'FaceAlpha', 0.3, 'EdgeColor', 'none'); hold on;
fill([freq, fliplr(freq)], [log(mean_power_spectrum_x2 + std_power_spectrum_x2), fliplr(log(mean_power_spectrum_x2 - std_power_spectrum_x2))], 'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none');
plot(freq, log(mean_power_spectrum_x1), 'b', 'LineWidth', 1.5); hold on;
plot(freq, log(mean_power_spectrum_x2), 'r', 'LineWidth', 1.5);
title('Mean Power Spectrum with Std Dev (Shaded Area)');
xlabel('Frequency');
ylabel('Log Power');
legend({'x1', 'x2'}, 'Location', 'Best');

% Adjust layout
set(gcf, 'Position', [100, 100, 1200, 800]);
