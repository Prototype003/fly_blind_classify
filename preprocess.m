%% Description

%{

Data pre-processing

Line noise removal

%}

%% Common parameters

sample_rate = 1000;

params = struct();
params.Fs = sample_rate;
params.tapers = [5 9];
params.pad = 2;
params.removeFreq = [];

%% Load data

% Dror's data
source_file = 'data/split2250_bipolarRerefType1_postPuffpreStim.mat';
tmp = load(source_file); % training data
data_t = tmp.fly_data;

% Rhiannon's data
source_file = 'data/delabelled_data.mat';
tmp = load(source_file);
data_v = tmp.delabelled_data; % validation data (blind)

%% Make data formats consistent - 2.25s epochs

% Dror - time x channels x epochs x flies x conditions

% Rhiannon ->
% time x channels x repeats
data_v = cell2mat(struct2cell(data_v)); % channels x time x repeats;
data_v = permute(data_v, [2 1 3]);

% Number of epochs which fit into Rhiannon's epochs
nEpochs = floor(size(data_v, 1) / size(data_t, 1));
data_v = data_v(1:nEpochs*size(data_t, 1), :, :);
data_v = reshape(data_v, [size(data_t, 1), nEpochs, size(data_v, 2) size(data_v, 3)]); % time x split-epochs x channels x provided-epochs
data_v = permute(data_v, [1 3 2 4]);

%% Make data formats consistent - 18s epochs

% % Dror -> concatenate trials
% % time-trials x channels x flies x conditions
% data_t = permute(data_t, [1 3 2 4 5]); % time x trials x channels x flies x conditions
% dims = size(data_t);
% data_t = reshape(data_t, [dims(1)*dims(2) dims(3:end)]); % time-trials x channels x flies x conditions
% 
% % Rhiannon ->
% % time x channels x repeats
% data_v = cell2mat(struct2cell(data_v)); % channels x time x repeats;
% data_v = permute(data_v, [2 1 3]);
% data_v = data_v((1:size(data_t, 1)), :, :); % take same trial length as data_t

%% Power spectrum of each time-series
% No extra processing

tic;
% ~19 seconds (tapers [3 5]);
% ~40 seconds (tapers [5 9]; 2.25s epochs);
[powers_t, faxis_t] = getPower(data_t, params);
toc
tic;
% ~40 seconds (tapers [3 5])
% ~90 seconds (tapers [5 9]; 2.25s epochs)
[powers_v, faxis_v] = getPower(data_v, params);
toc

%% Plot and check power spectra for each fly

ch = 1;
tr = 1; % reference trial
ep = 1; % reference validation epoch

figure;
for fly = 1 : size(powers_t, 4)
    subplot(3, 5, fly);
    
    hold on;
    plot(faxis_t, log(powers_t(:, ch, tr, fly, 1)), 'r'); % wake
    plot(faxis_t, log(powers_t(:, ch, tr, fly, 2)), 'k'); % anest
    plot(faxis_v, log(powers_v(:, ch, tr, ep)), 'b'); % validation trial
    
    xlim([1 120]);
    
    title(['fly' num2str(fly) ' ch' num2str(ch)]);
    xlabel('Hz');
    ylabel('log(power)');
    legend('T_{W}', 'T_{A}', 'V_{?}');
end

%% Plot power spectra to help find validation trial with line noise

% 50Hz line noise seems to be more apparent for channel 15
% There seems to be another peak for channel 14 (not at 50Hz)?

ch = 15;
tr = 1;

figure;
for ep = 1 : size(powers_v, 4)
    subplot(6, 10, ep);
    plot(faxis_v, log(powers_v(:, ch, tr, ep)));
    
    xlim([40 60]);
    xlim([0 100]);
    title(num2str(ep));
end

%% Preprocessing

preprocess_string = ''; % To keep track of preprocessing pipeline

% Processed versions of the data structure
data_t_proc = data_t;
data_v_proc = data_v;

%% Subtract mean from each epoch

s_subtractMean = 1;

if s_subtractMean == 1
    tic;
    dims = size(data_t_proc);
    data_mean = mean(data_t_proc, 1);
    data_mean = repmat(data_mean, [dims(1) ones(1, length(dims)-1)]);
    data_t_proc = data_t_proc - data_mean;
    toc
    
    tic;
    dims = size(data_v_proc);
    data_mean = mean(data_v_proc, 1);
    data_mean = repmat(data_mean, [dims(1) ones(1, length(dims)-1)]);
    data_v_proc = data_v_proc - data_mean;
    toc
    
    preprocess_string = [preprocess_string '_subtractMean'];
end

%% Remove line noise

s_lineNoise = 1;

% Do extra processing
if s_lineNoise == 1
    tic;
    % ~103 seconds (tapers = [5 9]; 2.25s epochs)
    [data_t_proc, data_t_fit] = removeLineNoise(data_t_proc, params); % ~10 seconds (tapers = [3 5])
    toc
    tic;
    % ~227 seconds (tapers [5 9]; 2.25s epochs)
    [data_v_proc, data_v_fit] = removeLineNoise(data_v_proc, params); % ~21 seconds (tapers = [3 5])
    toc
    
    preprocess_string = [preprocess_string '_removeLineNoise'];
end

%% Plot and check power spectra after line noise removal
% Get power spectra

tic;
[powers_t_proc, faxis_t] = getPower(data_t_proc, params);
toc
tic;
[powers_v_proc, faxis_v] = getPower(data_v_proc, params);
toc

%% Plot and check power spectra for each fly

ch = 1;
tr = 1; % reference trial
ep = 1; % reference validation epoch

figure;
for fly = 1 : size(powers_t_proc, 4)
    subplot(3, 5, fly);
    
    hold on;
    plot(faxis_t, log(powers_t_proc(:, ch, tr, fly, 1)), 'r'); % wake
    plot(faxis_t, log(powers_t_proc(:, ch, tr, fly, 2)), 'k'); % anest
    plot(faxis_v, log(powers_v_proc(:, ch, tr, ep)), 'b'); % validation trial
    
    xlim([1 120]);
    
    title(['fly' num2str(fly) ' ch' num2str(ch)]);
    xlabel('Hz');
    ylabel('log(power)');
    legend('T_{W}', 'T_{A}', 'V_{?}');
end

%% Plot power spectra to help find validation trial with line noise

% 50Hz line noise seems to be more apparent for channel 15
% There seems to be another peak for channel 14 (not at 50Hz)?

ch = 15;
tr = 1;

figure;
for ep = 1 : size(powers_v_proc, 4)
    subplot(6, 10, ep);
    plot(faxis_v, log(powers_v_proc(:, ch, tr, ep)));
    
    xlim([40 60]);
    xlim([0 100]);
    title(num2str(ep));
end

%% Save

data = struct();
data.train = data_t_proc;
data.validate1 = data_v_proc;
data.preprocess_params = params;
data.preprocess_string = preprocess_string;

out_dir = 'data/preprocessed/';
out_file = ['fly_data' preprocess_string];
tic;
save([out_dir out_file], 'data', '-v7.3', '-nocompression');
toc
disp(['data saved: ' out_dir out_file]);
