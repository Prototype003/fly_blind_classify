%% Description

%{

Apply thresholds obtained from test dataset to validation dataset(s)

%}

%% Settings

%class_type = 'nearestMean'; % nearest mean classification
class_type = 'nearestMedian'; % nearest median classification

preprocess_string = '_subtractMean_removeLineNoise';

source_dir = ['../hctsa_space' preprocess_string '/'];
source_prefix = 'HCTSA_validate1';

out_dir = ['results' preprocess_string '/'];
out_file = ['class_' class_type '_validate1BatchNormalised'];

thresh_dir = ['results' preprocess_string '/'];
thresh_file = ['class_' class_type '_thresholds'];

addpath('../');
here = pwd;
cd('../'); add_toolbox; cd(here);

%% Load thresholds

load([thresh_dir thresh_file]);

% Get dimensions
tic;
tmp = load([source_dir source_prefix '_channel1.mat']);
nRows = size(tmp.TS_DataMat, 1);
nFeatures = size(tmp.TS_DataMat, 2);
toc

%% Get labels
% Note - this classification with batch correction is not blinded

val_set = 'validate1';

label_dir = ['../hctsa_space' preprocess_string '/'];
label_file = [val_set '_labels.mat'];
data_labels = load([label_dir label_file]);

%% Get number of validation flies

keyword_cell = cellfun(@split, data_labels.keywords, repmat({','}, size(data_labels.keywords)), 'UniformOutput', false);

fly_key_pos = cellfun(@contains, keyword_cell, repmat({'fly'}, size(keyword_cell)), 'UniformOutput', false);
fly_key = cellfun(@(x, y) x(y), keyword_cell, fly_key_pos, 'UniformOutput', false);
fly_key = cellfun(@(x) x{1}, fly_key, 'UniformOutput', false); % get rid of the outer cell array

nFlies = numel(unique(fly_key));

%% Get parameters for normalisation from the discovery flies

ref_prefix = 'HCTSA_train';

means = nan(size(thresholds)); % channels x features
stds = nan(size(thresholds)); % channels x features
hctsa = cell(size(thresholds, 1), 1);
for ch = 1 : size(thresholds, 1)
    tic;
    
    hctsa{ch} = load([source_dir ref_prefix '_channel' num2str(ch) '.mat']);
    
    % replace Infs with nans, for omission
    hctsa{ch}.TS_DataMat(isinf(hctsa{ch}.TS_DataMat)) = nan;
    
    means(ch, :) = mean(hctsa{ch}.TS_DataMat, 1, 'omitnan');
    stds(ch, :) = std(hctsa{ch}.TS_DataMat, [], 1, 'omitnan');
    
    toc
end
stds(stds == 0) = 1; % if values are constant, only transform the mean

%% Normalise each validation fly using the other validation fly/flies

% Load raw data
disp('loading');
hctsa_raw = cell(size(thresholds, 1), 1);
for ch = 1 : size(thresholds, 1)
    tic;
    
    hctsa_raw{ch} = load([source_dir source_prefix '_channel' num2str(ch) '.mat']);
    
    % Replace (blinded) keywords with actual (nonblinded) labels
    hctsa_raw{ch}.TimeSeries.Keywords = data_labels.keywords;
    
    toc
end

% Transform each fly
disp('transforming');
hctsa_trans = hctsa_raw;
for ch = 1 : size(thresholds, 1)
    tic;
    
    % Normalise values based on training flies and other validation
    % fly/flies
    for fly = 1 : nFlies
        
        key = {['fly' num2str(fly)]};
        fly_rows = getIds(key, hctsa_raw{ch}.TimeSeries);
        
        % Transform to z-scores using mean+std of remaining fly/flies
        fly_raw = hctsa_raw{ch}.TS_DataMat(fly_rows, :);
        others_raw = hctsa_raw{ch}.TS_DataMat(~fly_rows, :);
        others_raw(isinf(others_raw)) = nan; % replace Infs with nans, for omission
        m = mean(others_raw, 1, 'omitnan');
        s = std(others_raw, [], 1, 'omitnan');
        s(s == 0) = 1; % if no std, only transform the mean
        fly_z = (fly_raw - m) ./ s; % "z-score"
        
        % Back-transform, but use mean+std of the training flies
        fly_trans = (fly_z .* stds(ch, :)) + means(ch, :);
        
        % Store
        hctsa_trans{ch}.TS_DataMat(fly_rows, :) = fly_trans;
    end
    
    toc
end

%% Classify validation data

predictions = zeros(nRows, nFeatures, size(thresholds, 1));

for ch = 1 : size(thresholds, 1)
    tic;
    for f = 1 : size(thresholds, 2)
        
        % Make predictions
        prediction = hctsa_trans{ch}.TS_DataMat(:, f) >= thresholds(ch, f);
        if directions(ch, f) == 0
            % flip if class 1 centre < class 2 centre
            prediction = ~prediction;
        end
        
        % Store predictions
        predictions(:, f, ch) = prediction;
    end
    toc
end

%% Save

tic;
save([out_dir out_file], 'predictions');
toc