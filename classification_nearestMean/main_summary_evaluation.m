%% Description

%{

% Plot feature matrix for given dataset

%}

%%

val_set = 'sleep'; % 'multidose'; 'singledose'; 'sleep'

source_prefix = [val_set]; % 'xxxxBatchNormalised'

preprocess_string = '_subtractMean_removeLineNoise';

source_dir = ['results' preprocess_string '/'];
source_file = ['class_nearestMedian_' source_prefix '_accuracy'];

tic;
accuracies = load([source_dir source_file]);
toc

%% Get valid features

addpath('../');
[ch_valid_features, ch_excluded] = getValidFeatures_allChannels(val_set, preprocess_string);
ch_valid_features = logical(ch_valid_features);

%% Test getting stats

%[performances, performances_random, sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr] = get_sig_features_evaluation('nearestMedian', 'singledose', ch_valid_features, preprocess_string);

%% Plot overall correctness and distribution of accuracies

ch = 6; % which channel to plot for

figure;
set(gcf, 'color', 'w');

subplot(1, 2, 1);
imagesc(accuracies.correct(:, ch_valid_features(ch, :), ch)); % TODO - check how the rows are ordered
colorbar;
title([source_prefix '_allConds ch' num2str(ch) ' correct'], 'interpreter', 'none');
xlabel('feature');
ylabel('epoch');

subplot(1, 2, 2);
values = accuracies.accuracies(ch, ch_valid_features(ch, :));
diffs = diff(sort(values, 'ascend'));
min_diff = max(diffs(diffs > eps))
edges = (0 : min_diff : 1);
histogram(values, edges);
title([source_prefix '_allConds ch' num2str(ch) ' accuracies'], 'interpreter', 'none');
xlabel('accuracy');
xlim([min(values)-min_diff max(values)+min_diff]);

%% Plot correctness and distribution of accuracies per class

ch = 6;

figure;
set(gcf, 'color', 'w');

% Figure out how many conditions there are
[conditions, condition_ids] = get_dataset_conditions(val_set);

% For each condition, plot correctness and accuracies
for cond = 1 : length(conditions)
    % Assume that order of conditions corresponds to the
    % correctness/accuracies matrices
    
    subplot(2, length(conditions), cond);
    imagesc(accuracies.correct_perCondition{cond}(:, ch_valid_features(ch, :), ch)); % TODO - check how the rows are ordered
    colorbar;
    title([source_prefix newline conditions{cond} ' ch' num2str(ch) ' correct'], 'interpreter', 'none');
    xlabel('feature');
    ylabel('epoch');
    
    subplot(2, length(conditions), cond+length(conditions));
    values = accuracies.accuracies_perCondition{cond}(ch, ch_valid_features(ch, :));
    diffs = diff(sort(values, 'ascend'));
    min_diff = max(diffs(diffs > eps))
    edges = (0 : min_diff : 1);
    histogram(accuracies.accuracies_perCondition{cond}(ch, ch_valid_features(ch, :)), edges);
    title([source_prefix newline conditions{cond} ' ch' num2str(ch) ' accuracies'], 'interpreter', 'none');
    xlabel('accuracy');
    xlim([min(values)-min_diff max(values)+min_diff]);
    
    % Check that correctness and accuracies actually correspond with the
    % histograms
    allCorrect = sum(all(accuracies.correct_perCondition{cond}(:, ch_valid_features(ch, :), ch) == 1, 1))
    allIncorrect = sum(all(accuracies.correct_perCondition{cond}(:, ch_valid_features(ch, :), ch) == 0, 1))
end

%% Average two conditions together at a time
% Average "unconscious" with "conscious" so that number of epochs is
% balanced

% multidose
%   1-3; 1-4; 1-5; 2-3; 2-4; 2-5
% singledose
%   1-2; 1-3; 1-4
% sleep
%   1-2

ch = 6;

% Figure out how many conditions there are
[conditions, condition_ids] = get_dataset_conditions(val_set);

% Note - can get pairings from get_dataset_conditionPairs();
switch val_set
    case 'multidose'
        cond_pairs = {[1 3], [1 4], [1 5], [2 3], [2 4], [2 5]};
        sp_rows = 3;
        sp_cols = 4;
        sp_pos = {[1 2], [5 6], [9 10], [3 4], [7 8], [11 12]};
        sp_mat_pos = [1 10];
        sp_hist_pos = [3 4 7 8 11 12];
    case 'singledose'
        cond_pairs = {[1 2], [1 3], [1 4]};
        sp_rows = 3;
        sp_cols = 2;
        sp_pos = {[1 2], [3 4], [5 6]};
        sp_mat_pos = [1 5];
        sp_hist_pos = [2 4 6];
    case 'sleep'
        cond_pairs = {[1 3], [1 4], [2 3], [2 4]};
        sp_rows = 4;
        sp_cols = 2;
        sp_pos = {[1 2]};
        sp_mat_pos = [1 7];
        sp_hist_pos = [2 4 6 8];
end

figure;
set(gcf, 'Color', 'w');

% store all xlims, for setting common xlims across plots
xlims_all = nan(length(cond_pairs), 2);
h = cell(size(cond_pairs));

for cpair = 1 : length(cond_pairs)
    
    subplot(sp_rows, sp_cols, sp_hist_pos(cpair));
    
    acc_a = accuracies.accuracies_perCondition{cond_pairs{cpair}(1)}(ch, ch_valid_features(ch, :));
    acc_b = accuracies.accuracies_perCondition{cond_pairs{cpair}(2)}(ch, ch_valid_features(ch, :));
    pair_accuracies = (acc_a + acc_b) ./ 2;
    
    % Determine histogram bin edges
    diffs = diff(sort(pair_accuracies, 'ascend'));
    min_diff = max(diffs(diffs > eps))
    edges = (0 : min_diff : 1);
    %edges = (0 : 0.01 : 1);
    
    histogram(pair_accuracies, edges);
    h{cpair} = gca;
    xlims_all(cpair, :) = [min(pair_accuracies)-min_diff max(pair_accuracies)+min_diff];%xlim;
    
    title([source_prefix '_' conditions{cond_pairs{cpair}(1)} '_' conditions{cond_pairs{cpair}(2)} ' ch' num2str(ch)], 'interpreter', 'none');
    xlabel('accuracy');
    
end

xlims_extreme = [min(xlims_all(:, 1)) max(xlims_all(:, 2))];
for hi = 1 : length(h)
    set(h{hi}, 'XLim', xlims_extreme);
end

% Concatenate and show correctness matrices together
subplot(sp_rows, sp_cols, sp_mat_pos);
plot_predictions = 0; % plot predictions or correctness

if plot_predictions == 1
    % Get matrix of predictions (reorder existing predictions matrix)
    
    allConds = nan(size(accuracies.predictions));
    row = 1;
    cond_starts = nan(length(accuracies.labels), 1);
    for cond = 1 : length(accuracies.condition_ids)
        ids = find(accuracies.labels == accuracies.condition_ids(cond));
        allConds(row:row+length(ids)-1, :, :) = accuracies.predictions(ids, :, :);
        cond_starts(cond) = row;
        row = row + length(ids);
    end
    
    title_string = [source_prefix ' ch' num2str(ch) ' predictions'];
    
else % plot_predictions == 0 % plot correctness
    
    allConds = cat(1, accuracies.correct_perCondition{:});
    cond_starts = nan(length(accuracies.correct_perCondition), 1);
    row = 1;
    for cond = 1 : length(accuracies.correct_perCondition)
        cond_starts(cond) = row;
        row = row + size(accuracies.correct_perCondition{cond}, 1);
    end
    
    title_string = [source_prefix ' ch' num2str(ch) ' correct'];
    
end

imagesc(allConds(:, :, ch));
colorbar;
title(title_string, 'interpreter', 'none');
set(gca, 'YTick', cond_starts, 'YTickLabel', conditions, 'TickLabelInterpreter', 'none');
xlabel('feature');

%% BELOW IS REFERENCE CODE

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
negPos_map = flipud(cbrewer('div', 'RdBu', 100));
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

