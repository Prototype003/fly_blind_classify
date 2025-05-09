%% Description

%{

Produce nearest mean classifier using all training data
Check for cases where class medians are equal

%}

%% Settings

%class_type = 'nearestMean'; % nearest mean classification
class_type = 'nearestMedian'; % nearest median classification

source_prefix = 'HCTSA_train';

preprocess_string = '_subtractMean_removeLineNoise';

out_dir = ['results' preprocess_string '/'];
out_file = ['class_' class_type '_thresholds'];
source_dir = ['../hctsa_space' preprocess_string '/'];

addpath('../');
here = pwd;
cd('../'); add_toolbox; cd(here);

%% Load

% Get dimensions
tic;
tmp = load('../data/preprocessed/fly_data_removeLineNoise.mat');
nChannels = size(tmp.data.train, 2);
nEpochs = size(tmp.data.train, 3);
nFlies = size(tmp.data.train, 4);
nConditions = size(tmp.data.train, 5);
toc

tic;
tmp = load([source_dir source_prefix '_channel' num2str(1) '.mat']);
nFeatures = size(tmp.TS_DataMat, 2);
toc

%% 

hctsas = cell(nChannels, 1);

%% Create classifier at each channel

% Results structure
% channels x features
%   Note - number of valid features can vary across channels
% Need to store - thresholds, directions

thresholds = NaN(nChannels, nFeatures);
directions = NaN(nChannels, nFeatures);

class_labels = [1 0]; % 1 = wake; 0 = anest
centres = NaN(length(class_labels), nChannels, nFeatures);

for ch = 1 : nChannels
    
    % Load HCTSA values for channel
    hctsa = load([source_dir source_prefix '_channel' num2str(ch) '.mat']);
    hctsas{ch} = hctsa;
    
    % Get valid features
    %valid_features = getValidFeatures(hctsa.TS_DataMat);
    valid_features = ones(1, size(hctsa.TS_DataMat, 2)); % do for all features
    feature_ids = find(valid_features);
    
    % Find rows corresponding to each class
    %   (assumes 2 classes only)
    %classes{1} = getIds({'condition1'}, hctsa.TimeSeries);
    %classes{2} = ~classes{1};
    class1 = getIds({'condition1'}, hctsa.TimeSeries);
    class2 = ~class1;
    classes = {class1, class2};
    
    for f = feature_ids
        tic;
        disp(f);
        
        % Get means for each class
        for c = 1 : length(classes)
            class_rows = classes{c};
            if strcmp(class_type, 'nearestMean')
                centres(c, ch, f) = mean(hctsa.TS_DataMat(class_rows, f), 1);
            elseif strcmp(class_type, 'nearestMedian')
                centres(c, ch, f) = median(hctsa.TS_DataMat(class_rows, f), 1);
            end
        end
        
        if f == 451
            disp('here');
        end
        
        % If centres are the same, get new centres based on another method
        if centres(1, ch, f) == centres(2, ch, f)
            switch class_type
                case 'nearestMedian'
                    % Replace Inf with max (non-inf val)
                    % Replace -Inf with min (non-inf val)
                    % Then use average as centre
                    for c = 1 : length(classes)
                        class_rows = classes{c};
                        values = hctsa.TS_DataMat(class_rows, f);
                        values_noInf = values(~isinf(values));
                        values(values==Inf) = max(values_noInf);
                        values(values==-Inf) = min(values_noInf);
                        centres(c, ch, f) = mean(values);
                    end
                case 'nearestMean'
                    % We're not using this method
            end
        end
        
        % Get threshold and direction based on centres
        %   direction: 1 means class 1 centre >= class 2 centre
        %   direction: 0 means class 1 centre < class 2 centre
        direction = centres(1, ch, f) >= centres(2, ch, f);
        threshold = sum(centres(:, ch, f)) / numel(centres(:, ch, f));
        
        thresholds(ch, f) = threshold;
        directions(ch, f) = direction;
        
        toc
    end
    
end

%%

preprocess_string = '_subtractMean_removeLineNoise';

dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};

%% Load stats
% Get all performances and stats using
%   get_stats_evaluation(preprocess_string)

stats_dir = ['results' preprocess_string filesep];
stats_file = 'stats_multidoseSplit.mat';

% Should have variable stats
load([stats_dir stats_file]);

%% Features which are valid in all datasets

valid_all = ones(size(stats.train.valid_features));
for d = 1 : length(dsets)
    
    disp(['====']);
    
    tmp = stats.(dsets{d}).valid_features;
    
    disp(['ch' num2str(ch) '-' dsets{d} ': ' num2str(numel(find(tmp(ch, :))))]);
    
    valid_all = valid_all & tmp;
    
    disp(['total ' num2str(numel(find(valid_all(ch, :)))) ' valid across datasets']);
end

%% Check where the feature is valid, but a class center is nan

% Interpretation - all values are Inf, and same number of +Inf and -Inf

tmp = squeeze(isnan(centres(1, :, :)) | isnan(centres(2, :, :))) &...
    stats.train.valid_features;

disp([num2str(numel(find(tmp(:)))) ' cases where a class center is nan in valid features']);

%% Check where both class centres are Inf -> threshold is nan

tmp = squeeze(all(isinf(centres), 1)) & stats.train.valid_features;
disp([num2str(numel(find(tmp(:)))) ' cases where both class centers are inf in valid features']);

%% Check proportion of features where centres are equal

equal_centres = centres(1, :, :) == centres(2, :, :);
equal_centres = permute(equal_centres, [2 3 1]);

% Intersect with valid features
equal_centres = equal_centres & stats.train.valid_features;
%equal_centres = equal_centres & valid_all;

% Breakdown per channel
perChannel_count = sum(equal_centres, 2);

% Percentage
perChannel_prop = sum(equal_centres, 2) ./ sum(stats.train.valid_features, 2);
%perChannel_prop = sum(equal_centres, 2) ./ sum(valid_all, 2);

%% Plot proportions

figure;

plot(perChannel_prop*100);
ylabel('% equal centres');
xlabel('channel');
title(['proportion of valid features with equal centres' newline 'median -> trunc-mean']);
xlim([1 15]);

%% Check (for after adding a second method for getting centres)
% Are equal centres due to constant feature values?
% Note - looks like diff doesn't always give 0s, but rather, values less
%   than eps (which essentially means 0, right?)
% So there may be features which are actually constant which haven't been
%   excluded

check = cell(nChannels, 1);
for ch = 1 : nChannels
    
    equal_features = find(equal_centres(ch, :));
    
    check{ch} = nan(size(equal_features));
    
    for f = 1 : length(equal_features)
        vals = hctsas{ch}.TS_DataMat(:, equal_features(f));
        val_diffs = diff(vals);
        check{ch}(f) = all(val_diffs < eps);
    end
    
end

% Check whether the constant values scenario applies to all features
cellfun(@all, check)

% Proportion of features where this (values with deviations <eps) is not
% the case
prop = cellfun(@(x) 1-(sum(x)/length(x)), check);

%% Save

tic;
save([out_dir out_file], 'thresholds', 'directions');
toc

disp(['saved - ' out_dir out_file '.mat']);