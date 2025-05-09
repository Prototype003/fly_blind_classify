%% Description

%{

Show correlations among specific features

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

%% Get performance stats

perfs = get_stats(preprocess_string);

%%

data_sets = {'train', 'validate1'};

%% Get raw values for each channel

fValues_all = cell(size(perfs.train.valid_features, 1), 1);
fValues_each = cell(size(perfs.train.valid_features, 1), 1);
hctsa = cell(size(fValues_all));
for ch = 1 : size(fValues_all, 1)
    tic;
    % Load and concatenat TS_DataMat for each dataset
    ds = cell(1);
    for d = 1 : length(data_sets)
        hctsa{ch} = hctsa_load(data_sets{d}, ch, preprocess_string);
        ds{d} = hctsa{ch}.TS_DataMat;
    end
    fValues_all{ch} = cat(1, ds{:});
    fValues_each{ch}.train = ds{1};
    fValues_each{ch}.validate1 = ds{2};
    toc
end

%% Get wanted features

feature_string = '^AC_\d'; % regex
ch = 6; % which channel to get features from (should be the same regardless of channel)

%features = hctsa{1}.Operations.Name(find(contains(hctsa{1}.Operations.Name, feature_string)));
feature_matches = regexp(hctsa{ch}.Operations.Name, feature_string);
noMatch = cellfun('isempty', feature_matches);
matchIDs = find(~noMatch);
features = hctsa{ch}.Operations.Name(matchIDs);

%% Correlate features
% Correlate across all flies, conditions, at one channel

ch = 6;
dataset = 2; % 1=discovery; 2=validate1

cmat = corr(fValues_each{ch}.(data_sets{dataset})(:, matchIDs));

f = figure;
set(f, 'Color', 'w');
imagesc(cmat);
c = colorbar;
title(c, 'r');

% Axis ticks for autocorrelation features
axis square;
title('autocorrelation features');
set(gca, 'TickLabelInterpreter', 'none');
set(gca, 'XTick', (5:5:length(matchIDs)));
set(gca, 'XTickLabel', features(5:5:end));
xtickangle(45);
set(gca, 'YTick', (5:5:length(matchIDs)));
set(gca, 'YTickLabel', features(5:5:end));

