%% Description

%{

Plot and get summary stats for training set cross-validation

%}

%% Settings

perf_type = 'nearestMedian'; % 'nearestMedian'; 'nearestMean'; 'consis'
data_set = 'validate1'; % 'train', 'validate1'

ref_type = 'consis';
ref_set = 'train';

preprocess_string = '_subtractMean_removeLineNoise';

%% Get valid features

[ch_valid_features, ch_excluded] = getValidFeatures_allChannels('train', preprocess_string);
nValid = sum(ch_valid_features, 2);
nValid_mean = mean(nValid)
nValid_min = min(nValid)
nValid_max = max(nValid)

% Treat all features as valid
%ch_valid_features = ones(size(ch_valid_features));

ch_valid_features = logical(ch_valid_features);

%% Get performance and significance of reference sets, for comparison

% Same performance type, different data_set
[performances_ref_set, performances_random_ref_set, sig_ref_set, ps_ref_set, ps_fdr_ref_set, sig_thresh_ref_set, sig_thresh_fdr_ref_set] =...
    get_sig_features(perf_type, ref_set, ch_valid_features, preprocess_string);

% Different performance type, same data_set
[performances_ref_type, performances_random_ref_type, sig_ref_type, ps_ref_type, ps_fdr_ref_type, sig_thresh_ref_type, sig_thresh_fdr_ref_type] =...
    get_sig_features(ref_type, data_set, ch_valid_features, preprocess_string);

% Different performance type, different data_set
[performances_ref_both, performances_random_ref_both, sig_ref_both, ps_ref_both, ps_fdr_ref_both, sig_thresh_ref_both, sig_thresh_fdr_ref_both] =...
    get_sig_features(ref_type, ref_set, ch_valid_features, preprocess_string);

%% Get performance values and significance

[performances, performances_random, sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr] = get_sig_features(perf_type, data_set, ch_valid_features, preprocess_string);

%% Get number of features which are significant in multiple sets

% Significant in both datasets and also significant in other performance in
% both datasets
sig_both = sig & sig_ref_set & sig_ref_type & sig_ref_both;

% Significant in both datasets
sig_both = sig & sig_ref_set;
perc_extend = sum(sig_both, 2) ./ sum(sig_ref_set, 2)

% Significant in the selected dataset but not in the reference set
sig_only = sig & ~sig_ref_set;

% Significant in the other performance type, for both datasets
sig_other = sig_ref_type & sig_ref_both;
perc_extend = sum(sig_other, 2) ./ sum(sig_ref_type, 2)

% Significant in the other performance type, but not in the reference set
sig_other_only = sig_ref_type & ~sig_ref_both;

%% Plot distribution of validation performance, after limiting

sig_both = sig_ref_set & sig;

figure;

% Channel colours
r = linspace(0, 1, size(ch_valid_features, 1))';
g = linspace(0, 0, size(ch_valid_features, 1))';
b = linspace(1, 0, size(ch_valid_features, 1))';
ch_colours = cat(2, r, g, b);

for ch = 1 : size(performances, 1)
    
    features_filtered = sig_both(ch, :);
    
    if ~isempty(performances(ch, features_filtered))
        h = cdfplot(performances(ch, features_filtered)); hold on;
        set(h, 'Color', ch_colours(ch, :));
        
        % Highlight particular channel
        if ch == 6
            set(h, 'LineWidth', 2);
            h.Annotation.LegendInformation.IconDisplayStyle = 'on';
        end
    end
    
    
end

%% Plot number of significant features after each stage
% Features are restricted by features which were significant through all
% previous stages

% Cross-validation

%% Get best feature for each channel
ch_best_perf = cell(size(performances, 1), 3);

for ch = 1 : size(performances)
    
    hctsa = hctsa_load(data_set, ch, preprocess_string);
    
    % Find best feature
    [sorted, order] = sort(performances(ch, :), 'descend');
    
    % Look up the feature
    perf = sorted(1);
    fID = hctsa.Operations{order(1), 'ID'};
    fName = hctsa.Operations{order(1), 'Name'};
    
    % Look up master feature
    mID = hctsa.Operations{order(1), 'MasterID'};
    mName = hctsa.MasterOperations{mID, 'Label'};
    
    % Store/print
    ch_best_perf{ch, 1} = ch;
    ch_best_perf{ch, 2} = perf;
    ch_best_perf{ch, 3} = fID;
    ch_best_perf{ch, 4} = fName{1};
    ch_best_perf{ch, 5} = mID;
    ch_best_perf{ch, 6} = mName{1};
    
end

% Convert to more readable table
ch_best_perf = cell2table(ch_best_perf, 'VariableNames', {'ch', 'perf', 'featureID', 'featureName', 'masterID', 'masterFeature'});
ch_best_perf

%% Get best feature after averaging across channels

% Consider features which are valid for ALL channels
valid_all = sum(ch_valid_features, 1);
valid_all = valid_all == size(ch_valid_features, 1);

% Average performance across channels
performances_mean = mean(performances, 1);
[perf, location] = max(performances_mean);

% Get feature details
fID = hctsa.Operations{location, 'ID'};
fName = hctsa.Operations{location, 'Name'};
mID = hctsa.Operations{location, 'MasterID'};
mName = hctsa.MasterOperations{mID, 'Label'};

% Display
disp([num2str(perf) ' ' num2str(fID) ' ' fName{1} ' ' num2str(mID) ' ' mName{1}]);

%% Detailed feature analysis

ch = 5;
topN = sum(sig_both(ch, :));
topN = 100;

hctsa = hctsa_load('train', ch, preprocess_string);

% Get sorted list of top N performing features
[perf_sorted, order] = sort(performances(ch, :), 'descend');
valid_sorted = ch_valid_features(ch, order);
valid_top = find(valid_sorted, topN, 'first');
perf_sorted = perf_sorted(:, valid_top);
order = order(:, valid_top);

% Correlate feature values
fValues = hctsa.TS_DataMat(:, order);
fValues(isinf(fValues)) = NaN; % Remove Inf for correlation
fCorr = corr(fValues, 'Rows', 'complete'); % Ignore NaNs

% Plot correlation matrix, sorted by performance
figure;
subplot(1, 2, 1); plot(perf_sorted);
subplot(1, 2, 2); imagesc(fCorr); colorbar;

% Plot correlation matrix, sorted by correlation distance
corrOrder = clusterFeatures(fValues);
figure;
subplot(1, 2, 1); plot(perf_sorted(corrOrder));
subplot(1, 2, 2); imagesc(fCorr(corrOrder, corrOrder)); colorbar;

% List of top features
f_perfs = cell(topN, 6);
for f = 1 : length(perf_sorted)
    
    % Look up the feature
    perf = perf_sorted(f);
    fID = hctsa.Operations{order(f), 'ID'};
    fName = hctsa.Operations{order(f), 'Name'};
    
    % Look up master feature
    mID = hctsa.Operations{order(f), 'MasterID'};
    mName = hctsa.MasterOperations{mID, 'Label'};
    
    % Store/print
    f_perfs{f, 1} = ch;
    f_perfs{f, 2} = perf;
    f_perfs{f, 3} = fID;
    f_perfs{f, 4} = fName{1};
    f_perfs{f, 5} = mID;
    f_perfs{f, 6} = mName{1};
    
end
f_perfs = cell2table(f_perfs, 'VariableNames', {'ch', 'perf', 'featureID', 'featureName', 'masterID', 'masterFeature'});