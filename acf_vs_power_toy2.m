% Parameters
n = 1:1000;  % Time vector
num_signals = 100;  % Number of time-series in each group
burst_duration = 50; % Duration of each burst
burst_interval = 200; % Interval between bursts
noise_level = 0.5; % Noise level

% Initialize arrays
group_1 = zeros(num_signals, length(n));
group_2 = zeros(num_signals, length(n));

% Generate Group 1 (with periodic bursts)
for i = 1:num_signals
    % Create a random signal with periodic bursts
    signal = zeros(1, length(n));
    for j = 1:5 % Number of bursts
        start_idx = (j - 1) * burst_interval + 1;
        burst = sin(2*pi*0.05*n(1:burst_duration)); % Periodic burst
        signal(start_idx:start_idx+burst_duration-1) = burst + noise_level * randn(1, burst_duration); % Add burst and noise
    end
    group_1(i, :) = signal;
end

% Generate Group 2 (random noise)
for i = 1:num_signals
    % Create random noise
    group_2(i, :) = noise_level * randn(1, length(n));
end

% Compute the autocorrelation and power spectra for both groups
lags = 0:length(n) - 1;  % Lags for autocorrelation
acf_group_1 = zeros(num_signals, length(lags));
acf_group_2 = zeros(num_signals, length(lags));
power_spectrum_group_1 = zeros(num_signals, length(n)/2);
power_spectrum_group_2 = zeros(num_signals, length(n)/2);

for i = 1:num_signals
    % Autocorrelation
    acf_group_1(i, :) = xcorr(group_1(i, :), 'coeff');
    acf_group_2(i, :) = xcorr(group_2(i, :), 'coeff');
    
    % Power Spectrum (FFT)
    power_spectrum_group_1(i, :) = abs(fft(group_1(i, :))).^2;
    power_spectrum_group_2(i, :) = abs(fft(group_2(i, :))).^2;
end

% Average autocorrelation and power spectrum
mean_acf_group_1 = mean(acf_group_1, 1);
mean_acf_group_2 = mean(acf_group_2, 1);
mean_power_spectrum_group_1 = mean(power_spectrum_group_1, 1);
mean_power_spectrum_group_2 = mean(power_spectrum_group_2, 1);

% Frequency axis for plotting power spectra
freq = (0:length(n)/2-1)/length(n);

% Plotting the results
figure;

% Plot autocorrelation
subplot(2, 2, 1);
plot(lags, mean_acf_group_1, 'b', 'LineWidth', 1.5); hold on;
plot(lags, mean_acf_group_2, 'r', 'LineWidth', 1.5);
title('Mean Autocorrelation');
xlabel('Lag');
ylabel('Autocorrelation');
legend({'Group 1', 'Group 2'});

% Plot power spectrum
subplot(2, 2, 2);
plot(freq, mean_power_spectrum_group_1(1:length(freq)), 'b', 'LineWidth', 1.5); hold on;
plot(freq, mean_power_spectrum_group_2(1:length(freq)), 'r', 'LineWidth', 1.5);
title('Mean Power Spectrum');
xlabel('Frequency');
ylabel('Power');
legend({'Group 1', 'Group 2'});

% Zoomed-in view of the low-frequency range
subplot(2, 2, 3);
plot(freq(1:50), mean_power_spectrum_group_1(1:50), 'b', 'LineWidth', 1.5); hold on;
plot(freq(1:50), mean_power_spectrum_group_2(1:50), 'r', 'LineWidth', 1.5);
title('Zoomed-in Power Spectrum (Low Frequencies)');
xlabel('Frequency');
ylabel('Power');
legend({'Group 1', 'Group 2'});

% Show ACF zoomed-in on a specific lag
subplot(2, 2, 4);
plot(lags(1:100), mean_acf_group_1(1:100), 'b', 'LineWidth', 1.5); hold on;
plot(lags(1:100), mean_acf_group_2(1:100), 'r', 'LineWidth', 1.5);
title('Zoomed-in ACF (Specific Lags)');
xlabel('Lag');
ylabel('Autocorrelation');
legend({'Group 1', 'Group 2'});
