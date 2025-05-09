%% Description

%{

Plot feature matrix for given dataset

Note - sleep dataset - feature 977 has nans after normalising

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';
source_prefix = 'sleep';
ch = 6;

source_dir = ['hctsa_space' preprocess_string '/'];
source_file = ['HCTSA_' source_prefix '_channel' num2str(ch) '.mat']; % HCTSA_train.mat; HCTSA_validate1.mat;

%% Load

tic;
hctsa = load([source_dir source_file]);
toc

%% Normalise hctsa matrix

% Note - even with mixedSigmoid, some features scale to NaNs and 0s
%   discovery flies - feature 976 (870th valid feature) scales
%       to NaN and 0s
%   sleep flies - feature 977 scales to NaN and a 0
hctsa.TS_Normalised = BF_NormalizeMatrix(hctsa.TS_DataMat, 'mixedSigmoid');

%% Keywords to visualise for
% Which set of time series to visualise for
% Maybe it's better to specify keywords to exclude, instead...?

keywords = {}; % everything

switch source_prefix
    case 'multidose'
        keywords = {};
    case 'singledose' % exclude recovery condition
        keywords = {'conditionWake', 'conditionIsoflurane', 'conditionPostIsoflurane'};
    case 'sleep'
        keywords = {};
end

% Get corresponding rows
match = zeros(size(hctsa.TimeSeries, 1), 1);
for kw = keywords
    match = match | getIds(kw, hctsa.TimeSeries);
end

% If nothing matches (i.e. nothing selected, choose everything
if isempty(find(match))
    match = logical(ones(size(match)));
end

%% Determine which feature columns are valid features

valid_features = true(size(hctsa.TS_DataMat, 2), 1); % everything

valid_features = getValidFeatures(hctsa.TS_DataMat(match, :));

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

%% Order rows by condition
% Or check that they are already ordered by condition

conditions = {};

switch source_prefix
    case 'multidose'
        conditions = {'conditionWake', 'conditionIsoflurane_0.6', 'conditionIsoflurane_1.2', 'conditionPost_Isoflurane', 'conditionRecovery'};
        cond_labels = {'W1', 'A0.6', 'A1.2', 'W2', 'WR'};
    case 'singledose'
        conditions = {'conditionWake', 'conditionIsoflurane', 'conditionPostIsoflurane'};
        cond_labels = {'W1', 'A', 'W2'};
    %case 'sleep'
     %   conditions = {'conditionActive', 'conditionInactive'};
      %  cond_labels = {'W', 'S'};
    case 'sleep'
        conditions = {'conditionwakeEarly', 'conditionwake', 'conditionsleepEarly', 'conditionsleepLate'};
        cond_labels = {'WE', 'W', 'S', 'SL'};
end

% Get the rows corresponding to each condition
% And then concatenate them together
cond_rows = cell(length(conditions), 1);
for c = 1 : length(conditions)
    [~, cond_rows{c}] = getIds(conditions(c), hctsa.TimeSeries);
end

row_order = cat(1, cond_rows{:});

%% Get matrix which will be visualised

vis_matrix = hctsa.TS_Normalised(row_order, find(valid_features));
rOrder = (1:size(vis_matrix, 1));
fOrder = (1:size(vis_matrix, 2));

%% Sort features/rows

% Sort features by similarity across time series
tic;
fOrder = clusterFeatures(vis_matrix);
toc
%%
% Sort rows by similarity across features
tic;
rOrder = clusterFeatures(vis_matrix');
toc

%% Get reference feature order from discovery dataset

%{

% Load discovery flies

% Normalise and re-order through clustering

% Take the order

% Load discovery flies

% Normalise discovery flies (separately? or together with evaluation
flies?)

% Concatenate with evaluation flies

% Re-order through clustering

%}

%% Create figure

figure;
imagesc(vis_matrix(:, fOrder));
title([source_file(1:end-4) ' ' strjoin(keywords, ';')], 'Interpreter', 'none');
xlabel('feature');

%% Find axis tick locations and create tick labels

% Get all the keywords for each row
kw = split(hctsa.TimeSeries.Keywords(row_order), ',');

% Get just fly IDs and conditions
% Note - assumes flyID and condition are first and fourth keywords
kw = kw(:, [1 4]);
kw = join(kw, 2);

% Get first occurrence of each string
% Note - assumes epochs for each fly are grouped together
[kw_unique, kw_row] = unique(kw, 'stable');

% Shorten strings so they take up less space (as axis ticks)
kw_short = strrep(kw_unique, 'fly', 'F');
for c = 1 : length(conditions)
    kw_short = strrep(kw_short, conditions{c}, cond_labels{c});
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
