%% Description

%{

Feature correlation matrices

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';
cond_strings = {'wake', 'anest'};

%% Get performance stats

perfs = get_stats(preprocess_string);

%%

data_sets = {'train', 'validate1'};

%% Get valid features

[ch_valid_features, ch_excluded, valid_perStage] = getValidFeatures_allChannels(data_sets{1}, preprocess_string);

%% Load hctsa values

nChannels = size(perfs.train.valid_features, 1);

hctsa = struct();

for d = 1 : length(data_sets)
    tic
    hctsa.(data_sets{d}) = cell(nChannels, 1);
    for ch = 1 : nChannels
        hctsa.(data_sets{d}){ch} = hctsa_load(data_sets{d}, ch, preprocess_string);
    end
    toc;
end

%% Correlation data structure (for a single channel at a time)
% Storage of correlation coefficients between flies
% For 1 channel

dset = data_sets{1};

[nChannels, nFlies, nConditions, nEpochs] = getDimensions(dset);

% Feature correlation matrices
rs = cell(nFlies, 1);

%% Compute feature correlations at each fly

dset = data_sets{1};

condition = 2; % wake; anest; wake-anest
cond_string = cond_strings{condition};
ch = 6;
for fly = 1 : nFlies
    
    % Get relevant data
    keys = {['fly' num2str(fly)], ['condition' num2str(condition)], ['channel' num2str(ch)]};
    [logical_ids, row_ids] = getIds(keys, hctsa.(dset){ch}.TimeSeries);
    fly_data = hctsa.(dset){ch}.TS_DataMat(row_ids, :);
    
    % Compute correlations
    rs{fly} = corr(fly_data(:, find(ch_valid_features(ch, :))));
    
end

%% Compute feature correlations at each fly
% wake minus anest

dset = data_sets{1};

cond_string = 'W-A';
ch = 6;
for fly = 1 : nFlies
    
    % Get relevant data
    keys = {['fly' num2str(fly)], ['condition1'], ['channel' num2str(ch)]};
    [logical_ids, row_ids] = getIds(keys, hctsa.(dset){ch}.TimeSeries);
    fly_data_wake = hctsa.(dset){ch}.TS_DataMat(row_ids, :);
    
    % Get relevant data
    keys = {['fly' num2str(fly)], ['condition2'], ['channel' num2str(ch)]};
    [logical_ids, row_ids] = getIds(keys, hctsa.(dset){ch}.TimeSeries);
    fly_data_anest = hctsa.(dset){ch}.TS_DataMat(row_ids, :);
    
    fly_data = fly_data_wake - fly_data_anest;
    
    % Compute correlations
    rs{fly} = corr(fly_data(:, find(ch_valid_features(ch, :))));
    
end

%% Compute between fly correlations using feature correlations
% Compute correlations using upper triangle of correlation matrix
%   Exclude the diagonal

ids = ones(size(rs{1})); % assumes same size r matrix across all flies
ids = logical(triu(ids, 1));

% Number of possible fly pairs is nFlies choose 2 (13 choose 2)
pair_rs = nan(nFlies, nFlies);

% Get feature correlations for each pair of flies
for f1 = 1 : nFlies-1
    vals_f1 = rs{f1}(ids);
    
    disp(['fly' num2str(f1)]);
    for f2 = f1+1 : nFlies
        vals_f2 = rs{f2}(ids);
        tic;
        pair_rs(f1, f2) = corr(vals_f1, vals_f2, 'Rows', 'pairwise');
        toc
    end
end

figure;
subplot(1, 2, 1);
imagesc(pair_rs);
xlabel('fly'); ylabel('fly'); title(cond_string);
c = colorbar; title(c, 'r');
subplot(1, 2, 2);
histogram(pair_rs);
xlabel('r'); ylabel('fly pairs');

%% Compute between fly correlations using all channels
% Can't store feature correlation matrices for all flies and channels
%   Too big
% So, compute and store only for 2 flies at a time

% Clear memory
clear rs

% Number of possible fly pairs is nFlies choose 2 (13 choose 2)
pair_rs = nan(nFlies, nFlies);

condition = 1;
for f1 = 1 : nFlies - 1
    disp(['fly' num2str(f1)]);
    
    fly_rs = cell(nChannels, 1);
    tic;
    for ch = 1 : nChannels % might be worth turning into function as this is repeated in inner loop
        
        % Get relevant data
        keys = {['fly' num2str(f1)], ['condition' num2str(condition)], ['channel' num2str(ch)]};
        [logical_ids, row_ids] = getIds(keys, hctsa.(dset){ch}.TimeSeries);
        fly_data = hctsa.(dset){ch}.TS_DataMat(row_ids, :);
        
        % Compute correlations
        feature_rs = corr(fly_data(:, find(ch_valid_features(ch, :))));
        
        % Store upper triangle
        ids = ones(size(feature_rs)); % assumes same size r matrix across all flies
        ids = logical(triu(ids, 1));
        fly_rs{ch} = feature_rs(ids);
        
    end
    clear feature_rs;
    rs_f1 = cat(1, fly_rs{:}); % concatenate channel correlation values together
    clear fly_rs;
    toc
    
    for f2 = f1+1 : nFlies
        disp(['fly' num2str(f2)]);
        tic;
        for ch = 1 : nChannels % might be worth turning into function as this is repeated
            
            % Get relevant data
            keys = {['fly' num2str(f2)], ['condition' num2str(condition)], ['channel' num2str(ch)]};
            [logical_ids, row_ids] = getIds(keys, hctsa.(dset){ch}.TimeSeries);
            fly_data = hctsa.(dset){ch}.TS_DataMat(row_ids, :);
            
            % Compute correlations
            feature_rs = corr(fly_data(:, find(ch_valid_features(ch, :))));
            
            % Store upper triangle
            ids = ones(size(feature_rs)); % assumes same size r matrix across all flies
            ids = logical(triu(ids, 1));
            fly_rs{ch} = feature_rs(ids);
            
        end
        clear feature_rs;
        rs_f2 = cat(1, fly_rs{:}); % concatenate channel correlation values together
        clear fly_rs;
        toc
        
        disp(['fly' num2str(f1) '+fly' num2str(f2)]);
        tic
        
        rs_f1 = single(rs_f1);
        rs_f2 = single(rs_f2);
        
        % Get rid of nan values
        invalid = isnan(rs_f1) | isnan(rs_f2);
        rs_f1_clean = rs_f1(~invalid); rs_f2_clean = rs_f2(~invalid);
        
        % Compute pearson correlation coefficient
        r = sum((rs_f1_clean - mean(rs_f1_clean)) .* (rs_f2_clean - mean(rs_f2_clean))) / ...
            (sqrt(sum((rs_f1_clean - mean(rs_f1_clean)).^2)) .* sqrt(sum((rs_f2_clean - mean(rs_f2_clean)).^2)));
        pair_rs(f1, f2) = r;
        toc
        
%         % Compute correlation between the flies (corr() goes out of memory)
%         disp(['fly' num2str(f1) '+fly' num2str(f2)]);
%         tic;
%         pair_rs(f1, f2) = corr(rs_f1, rs_f2, 'Rows', 'pairwise');
%         toc
        
    end
end

%%

cond_string = cond_strings{condition};
figure;
subplot(1, 2, 1);
imagesc(pair_rs);
xlabel('fly'); ylabel('fly'); title(cond_string);
c = colorbar; title(c, 'r');
subplot(1, 2, 2);
histogram(pair_rs);
xlabel('r'); ylabel('fly pairs');