%% Description

%{

% Plot feature matrix for given dataset

%}

%% Setup for hctsa matrix

preprocess_string = '_subtractMean_removeLineNoise';
source_prefix = 'train';

source_dir = ['../hctsa_space' preprocess_string '/'];
source_file = ['HCTSA_' source_prefix '_channel6.mat']; % HCTSA_train.mat; HCTSA_validate1.mat;

tic;
hctsa = load([source_dir source_file]);
toc

% Note - even with mixedSigmoid, feature 976 (870th valid feature) scales
%   from mostly close to 0 values (and 1 much bigger value) to NaNs and 0s
hctsa.TS_Normalised = BF_NormalizeMatrix(hctsa.TS_DataMat, 'mixedSigmoid');

% Manually rescale that weird feature
hctsa.TS_Normalised(:, 976) = ~isnan(hctsa.TS_Normalised(:, 976));

%% Setup for performance plots

perf_types = {'nearestMedian', 'consis'};
data_sets = {'train', 'validate1'};

preprocess_string = '_subtractMean_removeLineNoise';

addpath('../');

% Get valid features
[ch_valid_features1, ch_excluded] = getValidFeatures_allChannels('train', preprocess_string);
[ch_valid_features2, ch_excluded2] = getValidFeatures_allChannels('validate1', preprocess_string);

ch_valid_features = ch_valid_features1 & ch_valid_features2;

nValid = sum(ch_valid_features, 2);
nValid_mean = mean(nValid)
nValid_min = min(nValid)
nValid_max = max(nValid)

% Get performances
perfs = get_stats(preprocess_string);

%% Common plotting variables

ch = 6;

%% New figure

figure;
set(gcf, 'Color', 'w');

subplot_rows = 10;
subplot_cols = 2;

subplot_rows = 7;
subplot_cols = 2;

%% Get hctsa matrix

% Which set of time series to visualise for
keywords = {'fly1'};
keywords = {}; % everything

% Get corresponding rows
match = getIds(keywords, hctsa.TimeSeries);

% Get valid feature columns
valid_features = true(size(hctsa.TS_DataMat, 2), 1); % everything
valid_features = getValidFeatures(hctsa.TS_DataMat);

valid_cols = hctsa.TS_DataMat(find(match), find(valid_features));

% Sort features by similarity across time series
tic;
fOrder = clusterFeatures(valid_cols);
toc

% Sort rows by similarity across features
tic;
rOrder = clusterFeatures(valid_cols');
toc

% Normalise (note - nan values can occur during normalisation - so cluster
% first!
tic;
vis_rows = hctsa.TS_Normalised(:, valid_features);
toc

tmp = BF_NormalizeMatrix(hctsa.TS_DataMat(:, 976), 'robustSigmoid'); % problem feature?

%% Visualise hctsa matrix

subplot(subplot_rows, subplot_cols, [1 6]);
imagesc(vis_rows(:, fOrder));
%title([source_file(1:end-4) ' ' strjoin(keywords, ',')], 'Interpreter', 'none');

%% Manually add axis ticks to delineate groups
% Find a good way of doing this programmatically?

xlabel('feature');

% 13 flies x 8 epochs x 2 conditions
yticks((1 : 8 : 13*8*2));
ystrings = cell(size(yticks));
conds = {'W', 'A'};
y = 1;
for c = 1 : 2
    for f = 1 : 13
        ystrings{y} = ['D' num2str(f) ' ' conds{c}];
        y = y + 1;
    end
end
yticklabels(ystrings);

set(gca, 'TickDir', 'out');

% Other details
c = colorbar;
ylabel(c, 'norm. value');
colormap(gca, inferno);
%%
% Add markers for key features
key_features = [551 4529];
valid_features_ids = find(valid_features);
valid_features_ids = valid_features_ids(fOrder);
xticks_new = xticks;
xticklabels_new = xticklabels;
for k = key_features
    k_pos = find(valid_features_ids == k);
    xticks_new = [xticks_new k_pos];
    xticklabels_new = [xticklabels_new; {num2str(k)}];
end
[xticks_new, order] = sort(xticks_new);
xticklabels_new = xticklabels_new(order);
xticks(xticks_new)
xticklabels(xticklabels_new);

%% Plot number of sig features per channel

% Plot train vs validate1
subplot_pos = [9 13];
ptype = 1;
h = summary_plots(2, ch, ptype, [subplot_rows, subplot_cols], subplot_pos, data_sets, ch_valid_features, perfs, hctsa);
h.Title.String = ['b ' h.Title.String];
%%
% Plot train vs validate1BatchNormalised
subplot_pos = [15 19];
subplot_pos = [9 13];
ptype = 1;
h = summary_plots(2, ch, ptype, [subplot_rows, subplot_cols], subplot_pos, {'train', 'validate1BatchNormalised'}, ch_valid_features, perfs, hctsa);
h.Title.String = ['b ' h.Title.String];

%% Plot number of sig features per channel (3 datasets)
% Show train, validate1, and validate1BatchNormalised

% subplot_pos = [9 14];
% ptype = 1;
% h = summary_plots(3, ch, ptype, [subplot_rows, subplot_cols], subplot_pos, {'train', 'validate1', 'validate1BatchNormalised'}, ch_valid_features, perfs, hctsa);
% h.Title.String = ['b ' h.Title.String];

%% Plot discovery vs evaluation, for single channel

% Plot train vs validate1
subplot_pos = [10 14];
ptype = 1;
h = summary_plots(1, ch, ptype, [subplot_rows, subplot_cols], subplot_pos, data_sets, ch_valid_features, perfs, hctsa);
h.Title.String = ['c ' h.Title.String];
axis square
%%
% Plot train vs validate1BatchNormalised
subplot_pos = [16 20];
subplot_pos = [10 14];
ptype = 1;
h = summary_plots(1, ch, ptype, [subplot_rows, subplot_cols], subplot_pos, {'train', 'validate1BatchNormalised'}, ch_valid_features, perfs, hctsa);
h.Title.String = ['c ' h.Title.String];
axis square

%% Print figure

figure_name = '../figures/fig2_raw_compare_normed';

set(gcf, 'PaperOrientation', 'Portrait');

print(figure_name, '-dsvg', '-painters'); % SVG
print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
print(figure_name, '-dpng'); % PNG

%% New figure

figure;
set(gcf, 'Color', 'w');

subplot_rows = 7;
subplot_cols = 2;

%% Plot wake-anesthesia consistency matrix

% Get data dimensions
[nChannels, nFlies, nConditions, nEpochs] = getDimensions(source_prefix);

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

% Sort features by similarity across rows
tic;
fOrder_diff = clusterFeatures(diff_mat);
toc

%% Plot figure

% Can use feature order from raw values
% Only works if there's only one feature in nan_features
fOrder_removed = fOrder;
%fOrder_removed(nan_features) = [];
%fOrder_removed(fOrder_removed > find(nan_features)) = fOrder_removed(fOrder_removed > find(nan_features)) - 1;

subplot(subplot_rows, subplot_cols, [1 6]);
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
negPos_map = flipud(cbrewer('div', 'RdBu', 100)); negPos_map(negPos_map < 0) = 0;
colormap(negPos_map);

% Add markers for key features
key_features = [7702 16];
valid_features_ids = find(valid_features);
valid_features_ids = valid_features_ids(fOrder);
xticks_new = xticks;
xticklabels_new = xticklabels;
for k = key_features
    k_pos = find(valid_features_ids == k);
    xticks_new = [xticks_new k_pos];
    xticklabels_new = [xticklabels_new; {num2str(k)}];
end
[xticks_new, order] = sort(xticks_new);
xticklabels_new = xticklabels_new(order);
xticks(xticks_new)
xticklabels(xticklabels_new);

%% Plot number of sig features per channel

subplot_pos = [9 13];
ptype = 2;
h = summary_plots(2, ch, ptype, [subplot_rows, subplot_cols], subplot_pos, data_sets, ch_valid_features, perfs, hctsa);
h.Title.String = ['b ' h.Title.String];

%% Plot discovery vs evaluation, for single channel

subplot_pos = [10 14];
ptype = 2;
h = summary_plots(1, ch, ptype, [subplot_rows, subplot_cols], subplot_pos, data_sets, ch_valid_features, perfs, hctsa);
h.Title.String = ['c ' h.Title.String];
axis square

%% Print figure

figure_name = '../figures/fig3_raw_compare';

set(gcf, 'PaperOrientation', 'Portrait');

print(figure_name, '-dsvg', '-painters'); % SVG
print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
print(figure_name, '-dpng'); % PNG
