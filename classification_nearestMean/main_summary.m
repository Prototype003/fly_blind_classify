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

% Significant in the other performance type for both datasets, and in this
sig_otherBoth_andThis = sig & sig_ref_both & sig_ref_type;

% Significant in the selected dataset but not in the reference set
sig_only = sig & ~sig_ref_set;
perc_only = sum(sig_only, 2) ./ sum(~sig_ref_set, 2)

% Significant in the other performance type, for both datasets
sig_other = sig_ref_type & sig_ref_both;
perc_extend = sum(sig_other, 2) ./ sum(sig_ref_type, 2)

% Significant in the other performance type, but not in the reference set
sig_other_only = sig_ref_type & ~sig_ref_both;

%% Find feature which has good consistency but low classification accuracy

ch = 6;

% Sort features by consistency, and list accuracy as well
[sorted, order] = sort(performances_ref_type(ch, :));

acc_consis = cat(1, sorted, performances(ch, order));

%% Portion of features which generalised

figure;
set(gcf, 'Color', 'w');

subplot(1, 2, 1);
% Portion of sig features out of features which were sig in the ref_set
sig_both = sig & sig_ref_set;
extend_portion = sum(sig_both, 2) ./ sum(sig_ref_set, 2);

% Portion of sig features out of features which were not sig in the ref_set
sig_only = sig & ~sig_ref_set;
extend_bad = sum(sig_only, 2) ./ sum(~sig_ref_set, 2);

plot(extend_portion);
hold on;
plot(extend_bad);
title('A class.');
xlabel('channel')
ylabel('portion');
xlim([1 length(extend_bad)]);

subplot(1, 2, 2);
% Portion of sig features out of features which were sig in the ref_set
sig_both = sig_ref_type & sig_ref_both;
extend_portion = sum(sig_both, 2) ./ sum(sig_ref_both, 2);

% Portion of sig features out of features which were not sig in the ref_set
sig_only = sig_ref_type & ~sig_ref_both;
extend_bad = sum(sig_only, 2) ./ sum(~sig_ref_both, 2);

plot(extend_portion);
hold on;
plot(extend_bad);
title('B consis.');
xlabel('channel')
ylabel('portion');
xlim([1 length(extend_bad)]);

%% Plot number of significant features after each stage
% Features are restricted by features which were significant through all
% previous stages

% Cross-validation

%% Get best feature for each channel
ch_best_perf = cell(size(performances, 1), 3);
hctsa = cell(size(performances, 1), 1);

for ch = 1 : size(performances, 1)
    
    hctsa{ch} = hctsa_load(data_set, ch, preprocess_string);
    
    % Find best feature
    [sorted, order] = sort(performances_ref_type(ch, :), 'descend');
    
    % Look up the feature
    perf = sorted(1);
    fID = hctsa{ch}.Operations{order(1), 'ID'};
    fName = hctsa{ch}.Operations{order(1), 'Name'};
    
    % Look up master feature
    mID = hctsa{ch}.Operations{order(1), 'MasterID'};
    mName = hctsa{ch}.MasterOperations{mID, 'Label'};
    
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
fID = hctsa{1}.Operations{location, 'ID'};
fName = hctsa{1}.Operations{location, 'Name'};
mID = hctsa{1}.Operations{location, 'MasterID'};
mName = hctsa{1}.MasterOperations{mID, 'Label'};

% Display
disp([num2str(perf) ' ' num2str(fID) ' ' fName{1} ' ' num2str(mID) ' ' mName{1}]);

%% Check performance of features with Inf values

hasInf = nan(size(ch_valid_features));
hctsa = cell(size(performances, 1), 1);

figure;

fIDs = cell(size(ch_valid_features, 1), 1);
ch_vals = cell(size(fIDs));
for ch = 1 : length(hctsa)
    tic;
    hctsa{ch} = hctsa_load(ref_set, ch, preprocess_string);
    
    % Find which features have infinity values & are considered valid
    hasInf(ch, :) = any(isinf(hctsa{ch}.TS_DataMat), 1);
    match = hasInf(ch, :) & ch_valid_features(ch, :);% & sig_ref_set(ch, :);
    
    vals = performances_ref_set(ch, match);
    
    % Plot feature performance
    if ~isempty(vals)
        subplot(4, 4, ch);
        cdfplot(vals);
        title(['ch' num2str(ch)]);
        ylabel('prop. of Inf features');
        xlabel('accuracy');
    end
    
    if ~isempty(vals) % store to write to file
        [ch_vals{ch}, order] = sort(vals, 'descend');
        
        fID = find(match);
        fIDs{ch} = fID(order);
    end
    
    toc
end

%% Create excel file of valid Inf feature performances

out_dir = [pwd filesep 'results' preprocess_string '_inf' filesep];
out_file = [perf_type 'nearestMedian_discovery']; % file with cluster details

for ch = 1 : length(hctsa)
    
    if ~isempty(fIDs{ch})
        
        % Get feature names from ids
        fNames = hctsa{ch}.Operations{fIDs{ch}, 'Name'};
        fCodes = hctsa{ch}.Operations{fIDs{ch}, 'CodeString'};
        mIds = hctsa{ch}.Operations{fIDs{ch}, 'MasterID'};
        mNames = hctsa{ch}.MasterOperations{mIds, 'Label'};
        mCodes = hctsa{ch}.MasterOperations{mIds, 'Code'};
        fSigs = sig_ref_set(ch, fIDs{ch});
        
        % Make table
        headings = {...
            'fSig', 'fPerf', 'fID',...
            'fName', 'fCodeString', 'mID', 'mName', 'mCode'};
        fTable = table(...
            fSigs', ch_vals{ch}', fIDs{ch}',...
            fNames, fCodes, mIds, mNames, mCodes,...
            'VariableNames', headings);
        
        % Write to file
        writetable(fTable, [out_dir out_file '.xlsx'],...
            'Sheet', ['ch' num2str(ch)],...
            'WriteVariableNames', 1);
        
    end
    
end

%%
% NaN value proportion per feature
ch = 6;
nNan = sum(isnan(hctsa{ch}.TS_DataMat), 1);
propNan = nNan / size(hctsa{ch}.TS_DataMat, 1);
figure;
plot(propNan);

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