%% Description

%{

Apply thresholds obtained from test dataset to validation dataset(s)

%}

%% Settings

%class_type = 'nearestMean'; % nearest mean classification
class_type = 'nearestMedian'; % nearest median classification

val_set = 'multidose'; % multidose only
fly_groups = {(1:8), (9:12)};
group_names = {'multidose8', 'multidose4'};

preprocess_string = '_subtractMean_removeLineNoise';

source_dir = ['../hctsa_space' preprocess_string '/'];
source_prefix = ['HCTSA_' val_set];

out_dir = ['results' preprocess_string '/'];
out_file = ['class_' class_type '_' val_set 'BatchNormalised_mdSplit'];

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

%% Load labelled data to get labels

tic;
label_dir = ['../hctsa_space' preprocess_string '/'];
label_file = [source_prefix '_channel1.mat']; % assumes all channels have same order of epochs

labelled_matfile = matfile([label_dir label_file]);
data_labelled = labelled_matfile.TimeSeries;
data_labels = data_labelled.Keywords;
toc

%% Get number of validation flies

keyword_cell = cellfun(@split, data_labels, repmat({','}, size(data_labels)), 'UniformOutput', false);

fly_key_pos = cellfun(@contains, keyword_cell, repmat({'fly'}, size(keyword_cell)), 'UniformOutput', false);
fly_key = cellfun(@(x, y) x(y), keyword_cell, fly_key_pos, 'UniformOutput', false);
fly_key = cellfun(@(x) x{1}, fly_key, 'UniformOutput', false); % get rid of the outer cell array

all_flies = unique(fly_key);

flyIDs_unsorted = str2double(cellfun(@(x) num2str(x(4:end)), all_flies, 'UniformOutput', false));
[sorted, sort_order] = sort(flyIDs_unsorted);

all_flies = all_flies(sort_order);

%{
% Get the flies in each fly group
flyIDs = cell(size(fly_groups));
for group = 1 : length(fly_groups)
    
    tmp = cellfun(@(x) regexp(x, '\d+$'), all_flies(fly_groups{group}));
    flyIDs{group} = cellfun(@(x) x(4:end), all_flies(fly_groups{group}), 'UniformOutput', false);
    flyIDs{group} = str2double(cell2mat(tmp));
    
end
%}

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

%% Load raw data

disp('loading');
hctsa_raw = cell(size(thresholds, 1), 1);
for ch = 1 : size(thresholds, 1)
    tic;
    
    hctsa_raw{ch} = load([source_dir source_prefix '_channel' num2str(ch) '.mat']);
    
    toc
end

hctsa_trans = cell(size(thresholds, 1), 1);
for ch = 1 : size(thresholds, 1)
    
    hctsa_trans{ch} = struct();
    hctsa_trans{ch}.TS_DataMat = hctsa_raw{ch}.TS_DataMat;
    
end

%% Normalise each validation fly using the other validation fly/flies

% Transform each fly
disp('transforming');
for ch = 1 : size(thresholds, 1)
    tic;
    
    for group = 1 : length(fly_groups)
        
        for f = 1 : length(fly_groups{group})
            fly = (fly_groups{group}(f));
            remaining_flies = fly_groups{group};
            remaining_flies(f) = [];
            remaining_keys = all_flies(fly_groups{group});
            remaining_keys(f) = [];
            
            key = {['fly' num2str(fly)]};
            fly_rows = getIds(key, hctsa_raw{ch}.TimeSeries);
            
            remaining_flies_rows = getIds(remaining_keys, hctsa_raw{ch}.TimeSeries, 'or');
            
            % Transform to z-scores using mean+std of remaining fly/flies
            fly_raw = hctsa_raw{ch}.TS_DataMat(fly_rows, :);
            others_raw = hctsa_raw{ch}.TS_DataMat(remaining_flies_rows, :);
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
        
    end
    
    toc;
end

%% Classify validation data

predictions = true(nRows, nFeatures, size(thresholds, 1));

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

%predictions = int8(predictions); % save space - values are binary anyway

%% Save
% Fly groups will be separated out when extracting accuracies

tic;
save([out_dir out_file], 'predictions');
toc