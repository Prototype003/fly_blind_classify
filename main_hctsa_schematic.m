%% Description

%{

% Plot feature matrix for given dataset

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

source_dir = ['hctsa_space' preprocess_string '/'];
source_file = 'HCTSA_train_channel6.mat'; % HCTSA_train.mat; HCTSA_validate1.mat;

addpath('classification_nearestMean/');

%% Which time-series/feature to show in detail

fly = 1; epoch = 1; cond = 1; ch = 6; % make sure ch matches HCTSA file
keywords = {'fly1', 'epoch1', 'condition1'};

perf_type = 'class_nearestMedian';
ref_set = 'train';
[values, fDetails, perf] = get_best_from_set(ch, perf_type, ref_set, 0, preprocess_string);

%% Load

% HCTSA
tic;
load([source_dir source_file]);
toc

% LFPs
fly_data = load(['data/preprocessed/fly_data' preprocess_string]);

%%

figure;

set(gcf, 'Color', 'w');

colormap inferno

%% Visualise time-series

ts = fly_data.data.train(:, ch, epoch, fly, cond);

subplot(10, 10, (1:4.5));

plot(ts);
ylabel('V'); xlabel('t');
axis tight

%% Scale HCTSA matrix for visualisation

valid_features = getValidFeatures(TS_DataMat);

hctsa_mat = TS_DataMat(:, find(valid_features));
hctsa_mat = BF_NormalizeMatrix(hctsa_mat, 'mixedSigmoid');

%% Visualise feature vector for time-series

match = getIds(keywords, TimeSeries);

vis_rows = hctsa_mat(find(match), :);

subplot(10, 10, (5.5:10));
imagesc(vis_rows);
xlabel('feature');

c = colorbar;
ylabel(c, 'norm. value');
colormap inferno

%% Visualise HCTSA matrix

subplot(10, 10, (21:60));
imagesc(hctsa_mat);
xlabel('feature');

% Manually add axis ticks to delineate groups
% Find a good way of doing this programmatically?
% 13 flies x 8 epochs x 2 conditions
yticks((1 : 8 : 13*8*2));
ystrings = cell(size(yticks));
conds = {'W', 'A'};
y = 1;
for c = 1 : 2
    for f = 1 : 13
        ystrings{y} = ['F' num2str(f) ' ' conds{c}];
        y = y + 1;
    end
end
yticklabels(ystrings);
set(gca, 'TickDir', 'out');

c = colorbar;
ylabel(c, 'norm. value');

%% Visualise datapoint vector for one feature

fIDs = find(valid_features);
vis_cols = hctsa_mat(:, fIDs == fDetails.fID); % 1834 - best from cross-validation set

subplot(10, 10, [71 91]);
imagesc(vis_cols);
xticks([]);
yticks((1 : 8 : 13*8*2));
yticklabels([]);

%% Visualise individual datapoints for each fly

train_flies = (1:13);
val_flies = (14:15);

% Colours
c = BF_GetColorMap('redyellowblue', 6);
cond_colours = {c(1, :), c(end, :)}; % red = wake; blue = anest
cond_offsets = [-0.1 0.1]; % x-axis offsets for each violin
extraParams = struct();
extraParams.offsetRange = 0.5; % width of violins

subplot(10, 10, [72.5 97.5]); hold on;

% Trained threshold for the feature
thresh = load(['classification_nearestMean/results' preprocess_string '/' 'class_nearestMedian_thresholds.mat']); % note no thresholds for consistency
threshold = thresh.thresholds(ch, fDetails.fID);

% Plot violins for training set
for cond = 1 : size(values, 2)
    values_tmp = values(train_flies, :);
    extraParams.customOffset = cond_offsets(cond);
    extraParams.theColors = repmat(cond_colours(cond), [size(values_tmp, 1) 1]);
    BF_JitteredParallelScatter_custom(values_tmp(:, cond), 1, 1, 0, extraParams);
    
    % Plot median for condition
    tmp = cell2mat(values(train_flies, cond));
    med = median(tmp);
    line([0 size(values_tmp, 1)+1], [med med], 'Color', cond_colours{cond}, 'LineWidth', 1, 'LineStyle', ':');
    
    % Threshold
    line([0 size(values_tmp, 1)+1], [threshold threshold], 'Color', 'k', 'LineWidth', 1, 'LineStyle', ':');
end
axis tight
ylims = ylim(gca);
xticks(1:length(train_flies));

subplot(10, 10, [79 100]); hold on;

% Plot violins for the val set
for cond = 1 : size(values, 2)
    values_tmp = values(val_flies, :);
    extraParams.customOffset = cond_offsets(cond);
    extraParams.theColors = repmat(cond_colours(cond), [size(values_tmp, 1) 1]);
    BF_JitteredParallelScatter_custom(values_tmp(:, cond), 1, 1, 0, extraParams);
    
    % Threshold
    line([0 size(values_tmp, 1)+1], [threshold threshold], 'Color', 'k', 'LineWidth', 1, 'LineStyle', ':');
end
axis tight
ylim(ylims);
xticks(1:length(val_flies));

%% Print

figure_name = 'figures/schematic';

set(gcf, 'PaperOrientation', 'Landscape');

print(figure_name, '-dsvg', '-painters'); % SVG
print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
print(figure_name, '-dpng'); % PNG
