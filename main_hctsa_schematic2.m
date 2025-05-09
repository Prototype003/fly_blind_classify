%% Description

%{

% Plot feature matrix for given dataset

%}

%% Settings

plot_ch = 6;

preprocess_string = '_subtractMean_removeLineNoise';

data_sets = {'train', 'validate1'};

source_dir = ['hctsa_space' preprocess_string '/'];

addpath('classification_nearestMean/');

%% Load

% HCTSA
hctsa = cell(size(data_sets));
for d = 1 : length(data_sets)
    source_file = ['HCTSA_' data_sets{d} '_channel' num2str(plot_ch) '.mat'];
    tic;
    hctsa{d} = load([source_dir source_file]);
    toc
end

% LFPs
fly_data = load(['data/preprocessed/fly_data' preprocess_string]);

%% Which time-series/feature to show in detail

plot_fly = 1; epoch = 1; cond = 1;
keywords = {'fly1', 'epoch1', 'condition1'};

perf_type = 'class_nearestMedian';
ref_set = 'train';
%[values, fDetails, perf] = get_best_from_set(plot_ch, perf_type, ref_set, 0, preprocess_string);

fID = 567; % which feature to show
fID_raw = 551; % best performing (nearestMedian) feature in training set
fID_diff = 7702; % best performing (consis) feature in training set
scale_values = 1;
values = struct(); fDetails = struct();
[values.raw, fDetails.raw] = get_fly_values(plot_ch, fID_raw, 'class_nearestMedian', scale_values, '_subtractMean_removeLineNoise');
[values.diff, fDetails.diff] = get_fly_values(plot_ch, fID_diff, 'class_nearestMedian', scale_values, '_subtractMean_removeLineNoise');

%% Get dataset to plot HCTSA mat values
% Either get specific dataset, or join matrices of datasets

d = 1;

TS_DataMat = hctsa{d}.TS_DataMat;

%% Scale HCTSA matrix for visualisation

valid_features = getValidFeatures(TS_DataMat);

hctsa_mat = TS_DataMat(:, find(valid_features));
hctsa_mat = BF_NormalizeMatrix(hctsa_mat, 'mixedSigmoid');

%%

figure;

set(gcf, 'Color', 'w');

colormap inferno

subplot_rows = 13;
subplot_cols = 10;

%% Visualise time-series

ts = fly_data.data.train(:, plot_ch, epoch, plot_fly, cond);

subplot(subplot_rows, subplot_cols, (1:10));

plot(ts);
ylabel('V'); xlabel('t');
axis tight

%% Reorder and plot feature values
% Note - assumes equal number of epochs per fly and condition

train_flies = (1:13);

% Order to show wake flies first, then anest flies
vis_col = cell2mat(values.raw(train_flies, :)); % concatenate flies together
dims = size(vis_col);
vis_col = vis_col(:); % concatenate conditions together

% Scale values for visualisation
vis_col = BF_NormalizeMatrix(vis_col, 'mixedSigmoid');

% Separate conditions
vis_col = reshape(vis_col, dims);

% Visualise each condition
plot_pos = {[21 25], [26 30]};
ticklabels = strcat('D', cellfun(@num2str, num2cell((1:13)), 'UniformOutput', 0));
for cond = 1 : size(vis_col, 2)
    subplot(subplot_rows, subplot_cols, plot_pos{cond});
    imagesc(vis_col(:, cond)', [0 1]);%[min(vis_col(:)) max(vis_col(:))]);
    xticks((1 : 8 : 13*8));
    xticklabels(ticklabels);
    yticks([]);
    set(gca, 'TickDir', 'out');
end
colorbar;
colormap inferno

%% Reorder and plot feature raw values (reference for diff vector)

train_flies = (1:13);

% Order to show wake flies first, then anest flies
vis_col = cell2mat(values.diff(train_flies, :)); % concatenate flies together
dims = size(vis_col);
vis_col = vis_col(:); % concatenate conditions together

% Scale values for visualisation
vis_col = BF_NormalizeMatrix(vis_col, 'mixedSigmoid');

% Separate conditions
vis_col = reshape(vis_col, dims);

% Visualise each condition
plot_pos = {[71 75], [76 80]};
ticklabels = strcat('D', cellfun(@num2str, num2cell((1:13)), 'UniformOutput', 0));
for cond = 1 : size(vis_col, 2)
    subplot(subplot_rows, subplot_cols, plot_pos{cond});
    imagesc(vis_col(:, cond)', [0 1]);%[min(vis_col(:)) max(vis_col(:))]);
    xticks((1 : 8 : 13*8));
    xticklabels(ticklabels);
    yticks([]);
    set(gca, 'TickDir', 'out');
end
colorbar;
colormap inferno

%% Visualise individual datapoints for each fly

% Colours
c = BF_GetColorMap('redyellowblue', 10);
cond_colours = {c(3, :), c(end-2, :)}; % red = wake; blue = anest
cond_colours_lines = {c(1, :), c(end, :)}; % red = wake; blue = anest
cond_offsets = [0 0]; % x-axis offsets for each violin
extraParams = struct();
extraParams.offsetRange = 0.5; % width of violins

subplot(subplot_rows, subplot_cols, [41 60]); hold on;

trainTicks = strcat('D', cellfun(@num2str, num2cell((1:13)), 'UniformOutput', 0));
validTicks = strcat('E', cellfun(@num2str, num2cell((14:15)), 'UniformOutput', 0));

% Trained threshold for the feature
thresh = load(['classification_nearestMean/results' preprocess_string '/' 'class_nearestMedian_thresholds.mat']); % note no thresholds for consistency
threshold = thresh.thresholds(plot_ch, fDetails.raw.fID);

% Plot violins for each fly
values_tmp = values.raw(:); % plot all wake, then all anest
extraParams.customOffset = cond_offsets(cond);
extraParams.theColors = repmat(cond_colours, [size(values.raw, 1) 1]);
BF_JitteredParallelScatter_custom(values_tmp, 1, 1, 0, extraParams);

% Plot medians for each condition
cond_pos = {[0.5 13.5], [0.5 13.5]+size(values.raw, 1)};
meds = nan(size(values.raw, 2), 1);
for cond = 1 : size(values.raw, 2)
    tmp = cell2mat(values.raw(train_flies, cond));
    meds(cond) = median(tmp);
    line(cond_pos{cond}, [meds(cond) meds(cond)], 'Color', cond_colours_lines{cond}, 'LineWidth', 2, 'LineStyle', '-');
end
if scale_values == 1
    % Threshold (compute from scaled values)
    line([0 numel(values.raw)+1], [mean(meds) mean(meds)], 'Color', 'k', 'LineWidth', 2, 'LineStyle', '-');
else
    % Threshold (if values are not scaled)
    line([0 numel(values.raw)+1], [threshold threshold], 'Color', 'k', 'LineWidth', 2, 'LineStyle', '-');
end
xlim([0 numel(values.raw)+1]);
xticks((1:numel(values.raw)));
xticklabels(cat(2, trainTicks, validTicks, trainTicks, validTicks));
ylabel('norm. value');

%% Plot wake - anest

% Pair every trial with every other trial
values_diff = cell(size(values.diff, 1), 1);
for fly = 1 : length(values.diff)
    values_diff{fly} = nan(length(values.diff{fly, 2}), length(values.diff{fly, 1}));
    for epoch1 = 1 : size(values.diff{fly, 1}, 1)
        values_diff{fly}(:, epoch1) = values.diff{fly, 1}(epoch1) - values.diff{fly, 2};
    end
    values_diff{fly} = values_diff{fly}(:); % convert from matrix to vector
end

% Binary colourmap
neg = viridis(256);
pos = inferno(256);
negPos_map = cat(1, flipud(neg(1:128, :)), pos(129:end, :));
negPos_map = flipud(cbrewer('div', 'RdBu', 100)); negPos_map(negPos_map < 0) = 0;

% Visualise difference values (vector)
plot_pos = [91 100];
subplot(subplot_rows, subplot_cols, plot_pos);
imagesc(cell2mat(values_diff(train_flies))', [-1 1]);
xticks((1 : 8*8 : 8*8*13));
xticklabels(ticklabels);
yticks([]);
colorbar;
set(gca, 'ColorMap', negPos_map);
set(gca, 'TickDir', 'out');

% Plot violins for each fly
subplot(subplot_rows, subplot_cols, [111 130]); hold on;
extraParams = struct();
extraParams.theColors = repmat({[0.3 0.3 0.3]}, size(values_diff));
BF_JitteredParallelScatter_custom(values_diff, 1, 1, 0, extraParams);
% 0 line
line([0 size(values_diff, 1)+1], [0 0], 'Color', 'k', 'Linewidth', 2, 'LineStyle', '-');
xlim([0 size(values_diff, 1)+1]);
xticks((1:numel(values_diff)));
xticklabels(cat(2, trainTicks, validTicks));
yticks([-1 0 1]);
ylabel('W-A');

%% Print

figure_name = 'figures/schematic2';

set(gcf, 'PaperOrientation', 'Landscape');

print(figure_name, '-dsvg', '-painters'); % SVG
print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
print(figure_name, '-dpng'); % PNG
