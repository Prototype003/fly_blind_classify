%% Description

%{

Visualise differences in normalised value (DNV)
    Figure 3A in stage 1 registered report

Compare to consistency measure
    Line plots of average DNV

%}

%% Settings

perf_type = 'consis'; % 'nearestMedian'; 'consis'
data_set = 'train'; % 'train'; 'validate1';

preprocess_string = '_subtractMean_removeLineNoise';

%% Get valid features

[ch_valid_features, ch_excluded] = getValidFeatures_allChannels(data_set, preprocess_string);
nValid = sum(ch_valid_features, 2);
nValid_mean = mean(nValid)
nValid_min = min(nValid)
nValid_max = max(nValid)

% Treat all features as valid
%ch_valid_features = ones(size(ch_valid_features));

ch_valid_features = logical(ch_valid_features);

%% Get performance values and significance

[performances, performances_random, sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr] = get_sig_features(perf_type, data_set, ch_valid_features, preprocess_string);
consistencies = performances(ch, ch_valid_features(ch, :));

%% Load HCTSA values

ch = 6;

preprocess_string = '_subtractMean_removeLineNoise';
source_prefix = data_set;

source_dir = ['../hctsa_space' preprocess_string '/'];
source_file = ['HCTSA_' source_prefix '_channel' num2str(ch) '.mat']; % HCTSA_train.mat; HCTSA_validate1.mat;

tic;
hctsa = load([source_dir source_file]);
toc

% Note - even with mixedSigmoid, feature 976 (870th valid feature) scales
%   from mostly close to 0 values (and 1 much bigger value) to NaNs and 0s
hctsa.TS_Normalised = BF_NormalizeMatrix(hctsa.TS_DataMat, 'mixedSigmoid');

%% Compute differences in normalised values

%valid_features = getValidFeatures(hctsa.TS_DataMat);
valid_features = ch_valid_features(ch, :);

% Get data dimensions
[nChannels, nFlies, nConditions, nEpochs] = getDimensions(data_set);

% Get rows corresponding to each condition
class1 = getIds({'condition1'}, hctsa.TimeSeries);
classes = {class1, ~class1}; % two conditions only

%diff_mat = nan(nEpochs*nEpochs*nFlies, size(hctsa.TS_DataMat, 2));
diff_mat = [];

for fly = 1 : nFlies
    % Find rows corresponding to the fly
    fly_rows = getIds({['fly' num2str(fly)]}, hctsa.TimeSeries);
    
    % Get rows for each class for this fly
    rows = cell(size(classes));
    for class = 1 : length(classes)
        rows{class} = find(classes{class} & fly_rows);
    end
    
    % Subtract anest from wake for every pair of epochs
    %vals = nan(nEpochs*nEpochs, 1);
    vals = [];
    for epoch1 = 1 : nEpochs
        epoch_vals = nan(nEpochs, length(find(valid_features)));
        for epoch2 = 1 : nEpochs
            epoch_vals(epoch2, :) = hctsa.TS_Normalised(rows{1}(epoch1), valid_features) - hctsa.TS_Normalised(rows{2}(epoch2), valid_features);
        end
%         if any(isnan(epoch_vals(:)))
%             keyboard;
%         end
        vals = cat(1, vals, epoch_vals);
    end
    diff_mat = cat(1, diff_mat, vals);
end

% Replace nans for that one feature which gets nans after scaling
nan_features = any(isnan(diff_mat), 1);
diff_mat(:, nan_features) = [];
consistencies(:, nan_features) = [];

% Sort features by similarity across rows
tic;
fOrder_diff = clusterFeatures(diff_mat);
toc

%% Compare consistencies with mean DNVs

r = corr(consistencies', abs(mean(diff_mat, 1))');
disp(['consistencies vs abs(mean(DNV)) r = ' num2str(r)]);

figure;
set(gcf, 'Color', 'w');
scatter(consistencies, abs(mean(diff_mat, 1)), '.');
xlabel('mean consistency across flies');
ylabel('abs(mean DNV) across flies');
title([data_set ' consistency vs DNV r=' num2str(r)]);