%% Description

%{

Cluster significant features based on similarity in feature values across
all epochs

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

%% Get performance stats

perfs = get_stats(preprocess_string);

%%

data_sets = {'train', 'validate1'};

%% Get raw values for each channel

fValues_all = cell(size(perfs.train.valid_features, 1), 1);
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
    toc
end

%%

perf_type = 'nearestMedian'; % nearestMedian; consis
stage = 'validate1'; % train; validate1; validate1BatchNormalised
%stage = 'validate1BatchNormalised';

switch stage
    case 'train'
        stage_string = 'discovery';
    case 'validate1'
        stage_string = 'pilotEvaluation';
end

topN = 40;

%% Get significant features at each channel

% Significant in both datasets
sig_all = perfs.train.(perf_type).sig & perfs.validate1.(perf_type).sig;

% Valid in both datasets, and sig. in all datasets
valid_all = perfs.train.valid_features & perfs.validate1.valid_features;

%% Get significant features at each channel

sig_all = perfs.train.(perf_type).sig;

switch stage
    case 'train'
        sig_all = perfs.train.(perf_type).sig;
        valid_all = perfs.train.valid_features;
    case 'validate1' % sig and valid in train + validate1
        sig_all = perfs.train.(perf_type).sig & perfs.validate1.(perf_type).sig;
        valid_all = perfs.train.valid_features & perfs.validate1.valid_features;
    case 'validate1BatchNormalised'
        sig_all = perfs.train.(perf_type).sig & perfs.validate1BatchNormalised.(perf_type).sig;
        valid_all = perfs.train.valid_features & perfs.validate1BatchNormalised.valid_features;
end

%% Filter features

% Keep features which are valid and sig. in all sets
fValues = cell(size(fValues_all));
fIds = cell(size(fValues));
for ch = 1 : size(sig_all, 1)
    fIds{ch} = find(valid_all(ch, :) & sig_all(ch, :));
    fValues{ch} = fValues_all{ch}(:, fIds{ch});
end

% Take top N
if topN > 0
    for ch = 1 : size(sig_all, 1)
        [perfs_sorted, order] = sort(perfs.(stage).(perf_type).performances(ch, fIds{ch}), 'descend');
        
        if numel(fIds{ch}) > topN
            
            fIds{ch} = fIds{ch}(order);
            fIds{ch} = fIds{ch}(1:topN);
            
            fValues{ch} = fValues{ch}(:, order);
            fValues{ch} = fValues{ch}(:, 1:topN);
        end
    end
end

%% Generate dendrogram

clusterDistance_method = 'average';

trees = cell(size(fValues));
distances = cell(size(fValues));

for ch = 1 : length(fValues)
    tic;
    
    if length(fValues{ch}) > 1 % can't really cluster when there's only 1 feature
        
        % Use correlations among features as distance (manual)
        values = fValues{ch};
        values(isinf(values)) = NaN; % Remove Infs for correlation
        fCorr = (corr(values, 'Type', 'Spearman', 'Rows', 'complete')); % Ignore NaNs
        %fCorr = abs(fCorr + fCorr.') / 2; % because corr output isn't symmetric for some reason (?)
        distances_m = 1 - fCorr; % higher correlation -> more similar -> less distance
        distances_m = squareform(distances_m); % convert to pdist vector form
        
        % Use (one minus) spearman correlation as distance
        distances_p = pdist(values', 'spearman'); % Can't deal with nan/inf?
        
        % Note - distances must be pdist vector (treats matrix as data instead of distances
        trees{ch} = linkage(distances_m, clusterDistance_method);
        distances{ch} = distances_m;
    end
    
    toc
end

%% Get labels/colours by master operation

ch = 6;

masters = hctsa{ch}.Operations.MasterID(fIds{ch});
% Group master operation list by broad category
%   Broad category determined by starting letters
master_strings = hctsa{ch}.MasterOperations.Code(masters);

feature_string = repmat({'[a-zA-Z0-9]+_?[a-zA-Z]+[a-zA-Z0-9]*'}, size(master_strings));
[starts, ends] = cellfun(@regexp, master_strings, feature_string, 'UniformOutput', false);
groups = cellfun(@(x,y,z) x(y:z), master_strings, starts, ends, 'UniformOutput', false);

groups_unique = sort(unique(groups));
group_colours = (1:length(groups_unique));
group_colourmapping = containers.Map(groups_unique, group_colours);
colours = cellfun(@(x) group_colourmapping(x), groups);

[sorted, order] = sort(groups);

colour_scheme = jet(numel(unique(colours)));
colour_scheme = cbrewer('qual', 'Set3', numel(unique(colours)));

%% Get labels/colours by master operation
% Use same colour scheme across all features and channels

ch = 6;

masters = hctsa{ch}.Operations.MasterID;
% Group master operation list by broad category
%   Broad category determined by starting letters
master_strings = hctsa{ch}.MasterOperations.Code(masters);

feature_string = repmat({'[a-zA-Z0-9]+_?[a-zA-Z]+[a-zA-Z0-9]*'}, size(master_strings));
[starts, ends] = cellfun(@regexp, master_strings, feature_string, 'UniformOutput', false);
groups = cellfun(@(x,y,z) x(y:z), master_strings, starts, ends, 'UniformOutput', false);

groups_unique = sort(unique(groups));
group_colours = (1:length(groups_unique));
group_colourmapping = containers.Map(groups_unique, group_colours);
colours = cellfun(@(x) group_colourmapping(x), groups);

[sorted, order] = sort(groups);

colour_scheme = jet(numel(unique(colours)));
colour_scheme = cbrewer('qual', 'Set3', numel(unique(colours)));

% Limit to only valid and significant features
groups = groups(fIds{ch});
feature_labels = hctsa{ch}.Operations.Name(fIds{ch});
colours = colours(fIds{ch});

%%

figure;
set(gcf, 'Color', 'w');
subplot_rows = 1;
subplot_cols = 100;

dend_orientation = 'left'; % orientation of dendrogram

%% Plot dendrogram

subplot(subplot_rows, subplot_cols, [1 30]);

% Compare these numbers to unspecified labels to check proper order of
%   the labels
labels = num2cell((1:size(trees{ch}, 1)+1));
labels = cellfun(@num2str, labels, 'UniformOutput', false);

labels = groups; % master operations as labels
labels = feature_labels; % feature names as labels

% Add performances to labels
p = num2cell(perfs.validate1.nearestMedian.performances(ch, fIds{ch}));
l = max(cellfun(@length, labels)); % length of longest label
pad_length = cellfun(@(x,l) l-length(x), labels, repmat({l}, size(labels)), 'UniformOutput', false);
spaces = repmat({' '}, size(labels));
pad_spaces = cellfun(@(s,pl) repmat(s, [1 pl]), spaces, pad_length, 'UniformOutput', false);
labels = cellfun(@(x,y,z) sprintf('%s%s %2.0f%%',x,y,z*100), labels, pad_spaces, p(:), 'UniformOutput', false);

[H, T, outperm] = dendrogram(trees{ch}, size(trees{ch}, 1)+1, 'Orientation', dend_orientation, 'ColorThreshold', 'default', 'Labels', labels);
title({perf_type, [stage ' ch' num2str(ch) ' top' num2str(topN)]});
set(gca, 'TickLabelInterpreter', 'none')

%set(gca, 'YTick', []);
%set(gca, 'XTick', []);
box on

h = gca;
h.YAxis.FontName = 'FixedWidth';

% Make the leaves line up with the colour plot
if strcmp(dend_orientation, 'left')
    ylim([0.5 size(trees{ch}, 1)+1.5]);
else
    
    xlim([0.5 size(trees{ch}, 1)+1.5]);
end

% Replace default grouping colours
lineColours = cell2mat(get(H,'Color'));
colourList = unique(lineColours, 'rows');

myColours = cbrewer('qual', 'Set1', length(colourList));

for colour = 2:size(colourList,1)
    %// Find which lines match this colour
    idx = ismember(lineColours, colourList(colour,:), 'rows');
    %// Replace the colour for those lines
    lineColours(idx, :) = repmat(myColours(colour-1,:),sum(idx),1);
end
%// Apply the new colours to the chart's line objects (line by line)
for line = 1:size(H,1)
    set(H(line), 'Color', lineColours(line,:));
end

xlim([0 0.6]);
set(gca, 'TickLength', [0 0]);

%% Show theoretical colours (use colour scheme across all features)

subplot(subplot_rows, subplot_cols, [31 85]);

colormap(colour_scheme);

image(permute(colour_scheme(flipud(colours(outperm)), :), [1 3 2]));

set(gca, 'YTick', [], 'XTick', []);

axis off

%% Custom colorbar (show colour scheme for plotted features)

subplot(subplot_rows, subplot_cols, [91 92]);

image(permute(colour_scheme(sort(unique(colours)), :), [1 3 2])); % Only feature categories which were plotted

set(gca, 'YTick', (1:length(group_colours)), 'YTickLabel', unique(groups));
set(gca, 'TickLabelInterpreter', 'none')
set(gca, 'YAxisLocation', 'right');
set(gca, 'YTickLabelRotation', 60);
set(gca, 'TickDir', 'out');
set(gca, 'XTick', []);

%% Show theoretical colours (scale colours across plotted features)

subplot(subplot_rows, subplot_cols, [31 35]);

colormap(colour_scheme);

imagesc(flipud(colours(outperm)));

set(gca, 'YTick', [], 'XTick', []);

%c = colorbar;
%c.Ticks = (1:length(colours)+1);

box off

%% Custom colorbar (show colour scheme across all features)

subplot(subplot_rows, subplot_cols, [38 39]);

imagesc((group_colours)'); % Everything

set(gca, 'YTick', (1:length(group_colours)), 'YTickLabel', groups_unique);
set(gca, 'TickLabelInterpreter', 'none')
set(gca, 'YAxisLocation', 'right');
set(gca, 'YTickLabelRotation', 60);
set(gca, 'TickDir', 'out');
set(gca, 'XTick', []);
