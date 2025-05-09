%% Description

%{

Look at power spectrums in raw data

%}

%%

clear tmp

%% Common parameters

sample_rate = 1000;

params = struct();
params.Fs = sample_rate;
params.tapers = [5 9];
params.pad = 2;
params.removeFreq = [];

data = struct();

% For keeping track if the datasets have been rereferenced
%   0 = not rereferenced; 1 = rereferenced
%   Position 1 is for single-dosage set; position 2 is for sleep set
reref_flags = nan(2, 1);

%% Load data - Multi-dosage flies

% Multi-dosage flies
%   Note - dataset has 2 files
source_files = {'labelled_data_1_selected.mat', 'data/labelled_data_2_selected.mat'};
tmp = {};
tmp_ordered = {};
for f = 1 : length(source_files)
    tmp{f} = load(source_files{f});
    tmp_ordered{f} = tmp{f}.(['labelled_ordered_data_' num2str(f)]);
end
data.multidose = [tmp_ordered{:}]; % join parts together

%% Load data - Single-dosage flies

source_file = 'merge_table.mat';
tmp = load(source_file);
data.singledose = tmp.merge_table;
reref_flag(1) = 0;

%% Load data - Sleep flies

source_file = 'LFP_data.mat';
tmp = load(source_file);
data.sleep = tmp.LFP_data;
reref_flag(2) = 0;

%% Convert singledose trial_type label to string, instead of cell containing string

for row = 1 : length(data.singledose)
    data.singledose(row).trial_type = data.singledose(row).trial_type{1};
end

%% Preprocess - bipolar rereference
% Note - assumes channel 1 is the most central channel
% Assumes LFP data is always (ch x time)

% Which datasets need to be processed
process_sets = {'singledose', 'sleep'};

% Field names corresponding to LFP to be processed for each dataset
data_fields = {'pre_visual_lfp', 'LFP'};

for dset = 1 : length(process_sets)
    if reref_flag(dset) == 0
        for chunk = 1 : length(data.(process_sets{dset}))
            data.(process_sets{dset})(chunk).(data_fields{dset}) = ...
                data.(process_sets{dset})(chunk).(data_fields{dset})(1:end-1, :) - ...
                data.(process_sets{dset})(chunk).(data_fields{dset})(2:end, :);
        end
        reref_flag(dset) = 1;
    end
end

data_original = data;

%% Compute power spectra (fft)

process_sets = {'multidose', 'singledose', 'sleep'};
data_fields = {'data', 'pre_visual_lfp', 'LFP'};
chunkLength = 18000; % same duration epochs for all datasets

addpath('../');

for dset = length(process_sets) : -1 : 1
    tic;
    for chunk = 1 : length(data.(process_sets{dset}))
        
        Y = fft(data.(process_sets{dset})(chunk).(data_fields{dset})(:, 1:chunkLength)');
        P2 = abs(Y/chunkLength);
        P1 = P2(1:chunkLength/2+1, :);
        P1(2:end-1, :) = 2*P1(2:end-1, :);
        f = params.Fs*(0:(chunkLength/2))/chunkLength;
        data.(process_sets{dset})(chunk).power = P1;
        data.(process_sets{dset})(chunk).faxis = f;
        
    end
    toc
end

%% Compute power spectra (multitaper)

compute = 0; % If 0, load already computed spectra

process_sets = {'multidose', 'singledose', 'sleep'};
data_fields = {'data', 'pre_visual_lfp', 'LFP'};
chunkLength = 18000; % same duration epochs for all datasets

if compute == 1
    addpath('../');
    
    for dset = length(process_sets) : -1 : 1
        for chunk = 1 : length(data.(process_sets{dset}))
            tic;
            
            [...
                data.(process_sets{dset})(chunk).power,...
                data.(process_sets{dset})(chunk).faxis...
                ] = getPower(data.(process_sets{dset})(chunk).(data_fields{dset})(:, 1:chunkLength)', params);
            
            toc
        end
    end
    
    % Save the spectra
    spectra = data;
    delete_fields = {'data', 'pre_visual_lfp', 'LFP'};
    for dset = 1 : length(process_sets)
        spectra.(process_sets{dset}) = rmfield(spectra.(process_sets{dset}), delete_fields{dset});
    end
    save('multitaper_spectra.mat', 'spectra', '-v7.3', '-nocompression');
    
else % load the spectra
    
    load('multitaper_spectra.mat');
    for dset = 1 : length(process_sets)
        data.(process_sets{dset}).power = {spectra.(process_sets{dset}).power};
        data.(process_sets{dset}).faxis = {spectra.(process_sets{dset}).faxis};
    end
    
    clear spectra
    
end

%% Plot power spectra

ch = 6; % which channel to show for

process_sets = {'multidose', 'singledose', 'sleep'};
process_sets = {'multidose'};
fly_fields = {'fly_ID', 'fly_number', 'fly_num'};
cond_fields = {'TrialType', 'trial_type', 'status'};

for dset = 1 : length(process_sets)
    
    row_flies = [data.(process_sets{dset}).(fly_fields{dset})]';
    flies = unique(row_flies);
    
    figure;
    
    for fly = 1 : length(flies)
        fly_match = find(row_flies == fly);
        fly_rows = data.(process_sets{dset})(fly_match);
        
        row_conds = {fly_rows.(cond_fields{dset})}';
        conds = unique(row_conds);
        
        subplot(floor(sqrt(length(flies))), ceil(sqrt(length(flies))), fly);
        hold on;
        
        for cond = 1 : length(conds)
            cond_match = find(strcmp(row_conds, conds{cond}));
            
            % Average across epochs
            cond_rows = cat(3, fly_rows(cond_match).power);
            cond_vals = sum(log(cond_rows), 3) ./ length(cond_match);
            
            plot(fly_rows(1).faxis, cond_vals(:, ch));
            
        end
        
        %ylim([-6 6]);
        ylabel('log(power)');
        xlim([min(data.(process_sets{dset})(1).faxis) max(data.(process_sets{dset})(1).faxis)]);
        xlabel('f (Hz)');
        set(gca, 'xscale', 'log');
        title([process_sets{dset} ' fly' num2str(fly) ' ch' num2str(ch)]);
        
    end
end

%% Plot power spectra for multidose 8 + 4 flies

process_sets = {'multidose', 'singledose', 'sleep'};
fly_fields = {'fly_ID', 'fly_number', 'fly_num'};
cond_fields = {'TrialType', 'trial_type', 'status'};

dset = 1;
fly_colours = cat(1, repmat({[0 0 0 0.25]}, [8 1]), repmat({[1 0 0 0.25]}, [4 1]));

conds = unique({data.(process_sets{dset}).(cond_fields{dset})});

figure;
for cond = 1 : length(conds)
    
    cond_rows = strcmp(conds{cond}, {data.(process_sets{dset}).(cond_fields{dset})});
    
    row_flies = [data.(process_sets{dset}).(fly_fields{dset})]';
    flies = unique(row_flies);
    
    subplot(3, 2, cond); hold on
    
    legend_items = [];
    
    for fly = 1 : length(flies)
        fly_rows = [data.(process_sets{dset}).(fly_fields{dset})] == fly;
        
        % Get all rows for the condition and fly
        tmp_rows = cond_rows & fly_rows;
        
        % Average across epochs
        vals = cat(3, data.(process_sets{dset})(tmp_rows).power);
        vals = sum(log(vals), 3) ./ length(find(tmp_rows));
        vals = mean(vals(:, 6), 2); % Average across channels
        
        % Plot
        p = plot((data.(process_sets{dset})(1).faxis), vals, 'Color', fly_colours{fly});
        
        if fly == 1 || fly == 9
            legend_items = cat(1, legend_items, p);
        end
        
    end
    
    ylabel('log(power)');
    xlabel('f (Hz)');
    %set(gca, 'xscale', 'log');
    %set(gca, 'yscale', 'log');
    title([process_sets{dset} ' ' conds{cond}]);
    axis tight
    legend(legend_items, 'N=8 "normal"', 'N=4 "odd"', 'Location', 'southwest');
    
end

%% Discovery flies

load('split2250_bipolarRerefType1_postPuffpreStim.mat');

[nSamples, nChannels, nEpochs, nFlies, nConds] = size(fly_data);

% Reshape to concatenate epochs together
fly_data = permute(fly_data, [1 3 2 4 5]);
fly_data = reshape(fly_data, [nSamples*nEpochs nChannels nFlies, nConds]);

Y = fft(fly_data(1:chunkLength, :, :, :));
P2 = abs(Y/chunkLength);
P1 = P2(1:chunkLength/2+1, :, :, :);
P1(2:end-1, :, :, :) = 2*P1(2:end-1, :, :, :);
f = params.Fs*(0:(chunkLength/2))/chunkLength;

%% Plot

figure;
for fly = 1 : nFlies
    
    subplot(ceil(sqrt(nFlies)), ceil(sqrt(nFlies)), fly);
    hold on;
    
    for cond = 1 : nConds
        
        plot(f, log(P1(:, ch, fly, cond)));
        
    end
    
    ylim([-6 6]);
    ylabel('log(power)');
    xlabel('f (Hz)');
    
    title(['discovery fly' num2str(fly)]);
    
end
