%% Description

%{

Plot feature matrix for given dataset

Note - sleep dataset - feature 977 has nans after normalising

Steps:
    Get list of features which are valid across all datasets
    Reorder features based on similarity in training set
        To do this, we have to normalise the training set first
    Normalise feature values from all datasets together
        (Or use normalisation parameters from the training set?)
    Reorder features based on ordering in training set

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

data_sources = {'train', 'multidose', 'singledose', 'sleep'};

channel = 6;

source_dir = ['hctsa_space' preprocess_string '/'];

%% Load files

hctsas = cell(size(data_sources));
for d = 1 : length(data_sources)
    
    source_file = ['HCTSA_' data_sources{d} '_channel' num2str(channel) '.mat'];
    
    disp(['loading ' source_file]);
    tic;
    hctsas{d} = load([source_dir source_file]);
    t = toc;
    disp(['loaded in ' num2str(t) 's']);
end

%% Exclude any rows

valid_rows = cell(size(data_sources));
for d = 1 : length(data_sources)
    
    switch data_sources{d}
        case 'train'
            keywords = {};
        case 'multidose'
            keywords = {};
        case 'singledose' % exclude recovery condition
            keywords = {'conditionRecovery'};
        case 'sleep'
            keywords = {};
    end
    
    % Find corresponding rows
    match = zeros(size(hctsas{d}.TimeSeries, 1), 1);
    for kw = keywords
        match = match | getIds(kw, hctsas{d}.TimeSeries);
    end
    
    valid_rows{d} = ~match;
    
end

%% Determine which feature columns hold valid features

valid_features = cell(size(data_sources));
for d = 1 : length(data_sources)
    valid_features{d} = getValidFeatures(hctsas{d}.TS_DataMat(valid_rows{d}, :));
end

%% Find which features are valid across all datasets

valid_all = ones(size(valid_features{1}));
for d = 1 : length(data_sources)
    
    valid_all = valid_all & valid_features{d};
    
end

%% Normalise reference dataset for reordering

ref_set = 1;

disp(['scaling reference dataset ' data_sources{ref_set}]);
tic;
hctsas{ref_set}.TS_Normalised = BF_NormalizeMatrix(hctsas{ref_set}.TS_DataMat(:, valid_all), 'mixedSigmoid');
t = toc;
disp(['scaled in ' num2str(t) 's']);

%% Reorder features based on similarity

% Sort features by similarity across time series
disp(['reordering features in reference dataset ' data_sources{ref_set}]);
tic;
fOrder = clusterFeatures(hctsas{ref_set}.TS_Normalised);
t = toc;
disp(['reordered in ' num2str(t) 's']);

% Sort rows by similarity across features
%{
tic;
rOrder = clusterFeatures(vis_matrix');
toc
%}

%% Order rows by condition
% Or check that they are already ordered by condition

conditions = cell(size(data_sources));
row_orders = cell(size(hctsas));
for d = 1 : length(hctsas)
    
    switch data_sources{d}
        case 'train'
            conditions{d}.conditions = {'condition1', 'condition2'};
            conditions{d}.cond_labels = {'D-W', 'D-A'};
        case 'multidose'
            conditions{d}.conditions = {'conditionWake', 'conditionIsoflurane_0.6', 'conditionIsoflurane_1.2', 'conditionPost_Isoflurane', 'conditionRecovery'};
            conditions{d}.cond_labels = {'E1-W1', 'E1-A0.6', 'E1-A1.2', 'E1-W2', 'E1-WR'};
        case 'singledose'
            conditions{d}.conditions = {'conditionWake', 'conditionIsoflurane', 'conditionPostIsoflurane'};
            conditions{d}.cond_labels = {'E2-W1', 'E2-A', 'E2-W2'};
        case 'sleep'
            conditions{d}.conditions = {'conditionActive', 'conditionInactive'};
            conditions{d}.cond_labels = {'E3-W', 'E3-S'};
    end
    
    % Get the rows corresponding to each condition
    % And then concatenate them together
    cond_rows = cell(length(conditions{d}.conditions), 1);
    for c = 1 : length(conditions{d}.conditions)
        [~, cond_rows{c}] = getIds(conditions{d}.conditions(c), hctsas{d}.TimeSeries);
    end
    
    row_orders{d} = cat(1, cond_rows{:});
    
end

%% Concatenate hctsa matrices and normalise across all datasets
% Note - multidose dataset has thousands of rows
%   Meanwhile, discovery, singledose, and sleep sets have hundreds of rows

% Which datasets to concatenate together
cat_sets = [1 2 3 4];

% Set to 0 if other datasets are to be scaled based on the first dataset
% Note, if scaling with a reference dataset, values in other datasets may
%   be outside the range of 0-1
scale_together = 0;

hctsas_all = [];
keywords_all = {};
for d = cat_sets
    hctsas_all = cat(1, hctsas_all, hctsas{d}.TS_DataMat(row_orders{d}, :));
    keywords_all = cat(1, keywords_all, hctsas{d}.TimeSeries.Keywords(row_orders{d}));
end

% Normalise concatenated hctsa matrix
disp('scaling concatenated hctsa matrix');
tic;
if scale_together == 1
    hctsas_all_normalised = BF_NormalizeMatrix(hctsas_all(:, valid_all), 'mixedSigmoid');
elseif scale_together == 0
    reference_rows = zeros(size(hctsas_all, 1), 1);
    reference_rows(1:size(hctsas{1}.TS_DataMat(row_orders{1}, :), 1)) = 1;
    hctsas_all_normalised = BF_NormalizeMatrix(hctsas_all(:, valid_all), 'mixedSigmoid', reference_rows);
end
t = toc;
disp(['scaled in ' num2str(t) 's']);

%% Look for nans after scaling
% For if we want to reorder features based on similarity across ALL
%   the datasets

% Keep track of any features which get nans for some reason after
%   scaling
nan_features = any(isnan(hctsas_all_normalised), 1);
if any(nan_features)
    disp('Note, excluding features with nan after scaling - features:');
    disp(find(nan_features));
end
valid_afterScale = logical(ones(size(hctsas_all_normalised, 2), 1));
valid_afterScale(nan_features) = 0; % keep track of columns with nan

%% Reorder features based on similarity across ALL the datasets

% Sort features by similarity across time series
disp(['reordering features in concatendated set']);
tic;
fOrder2 = clusterFeatures(hctsas_all_normalised(:, valid_afterScale));
t = toc;
disp(['reordered in ' num2str(t) 's']);

%% Get matrix which will be visualised

vis_matrix = hctsas_all_normalised(:, fOrder);

%% Create figure

clim = [0 1];

figure;
imagesc(vis_matrix, clim);
title([strjoin(data_sources(cat_sets), ';')], 'Interpreter', 'none');
xlabel('feature');

%% Find axis tick locations and create tick labels

% Get all the keywords for each row
%kw = split(keywords_all, ',');
%kw = kw(:, 4);

kw = cellfun(@(x) split(x, ','), keywords_all, 'UniformOutput', false);

% Get just conditions (assumes condition is fourth keyword)
for k = 1 : length(kw)
    kw{k} = kw{k}{4};
end

% Get first occurrence of each string
[kw_unique, kw_row] = unique(kw, 'stable');

% Shorten strings so they take up less space (as axis ticks)
kw_short = kw_unique;
for d = cat_sets
    for c = 1 : length(conditions{d}.conditions)
        kw_short = strrep(kw_short, conditions{d}.conditions{c}, conditions{d}.cond_labels{c});
    end
end

%% Add axis ticks and labels

xlabel('feature');

yticks(kw_row);
yticklabels(kw_short);

set(gca, 'TickDir', 'out');

%% Other details

c = colorbar;
ylabel(c, 'norm. value');

colormap inferno

set(gcf, 'Color', 'w');

%% Normalise each hctsa dataset separately

% Note - even with mixedSigmoid, some features scale to NaNs and 0s
%   discovery flies - feature 976 (870th valid feature) scales
%       to NaN and 0s
%   sleep flies - feature 977 scales to NaN and a 0

for d = 1 : length(hctsas)
    disp(['scaling dataset ' data_sources{d}]);
    tic;
    hctsas{d}.TS_Normalised = BF_NormalizeMatrix(hctsas{d}.TS_DataMat(valid_rows{d}, valid_all), 'mixedSigmoid');
    t = toc;
    disp(['scaled in ' num2str(t) 's']);
end

% Keep track of any features which get nans for some reason after
%   scaling
%       feature 976 in the discovery flies dataset
%       feature 977 in the sleep flies dataset
%   Features seem to get nans if all values are 0 except for 1
%       This might be occurring due to machine error (non-0 value tends to
%       be very small, very close to 0 in any case)
nan_features = any(isnan(hctsa.TS_Normalised), 1) & valid_features;
if any(nan_features)
    disp('Note, excluding features with nan after scaling - features:');
    disp(find(nan_features));
end
valid_features(find(nan_features)) = 0; % exclude these features from visualisation

%% Plot figure

% Can use feature order from raw values
% Only works if there's only one feature in nan_features
fOrder_removed = fOrder;
fOrder_removed(nan_features) = [];
fOrder_removed(fOrder_removed > find(nan_features)) = fOrder_removed(fOrder_removed > find(nan_features)) - 1;

figure;
imagesc(diff_mat(:, fOrder_removed));
title([source_file(1:end-4) ' ' strjoin(keywords, ',')], 'Interpreter', 'none');
colorbar;

%% Figure details

yticks((1 : nEpochs*nEpochs : nEpochs*nEpochs*nFlies));
ystrings = cell(nFlies, 1);
for fly = 1 : nFlies
    ystrings{fly} = ['F' num2str(fly)];
end
yticklabels(ystrings);
set(gca, 'TickDir', 'out');

neg = viridis(256);
pos = inferno(256);
negPos_map = cat(1, flipud(neg(1:128, :)), pos(129:end, :));
negPos_map = flipud(cbrewer('div', 'RdBu', 100));
colormap(negPos_map);