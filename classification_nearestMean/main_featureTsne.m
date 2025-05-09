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
stage = 'validate1'; % train; validate1;

switch stage
    case 'train'
        stage_string = 'discovery';
    case 'validate1'
        stage_string = 'pilotEvaluation';
end

%% Get significant features at each channel

sig_all = perfs.train.(perf_type).sig;

switch stage
    case 'train'
        sig_all = perfs.train.(perf_type).sig;
        valid_all = perfs.train.valid_features;
    case 'validate1_only'
        sig_all = perfs.validate1.(perf_type).sig;
        valid_all = perfs.validate1.valid_features;
    case 'validate1' % significant in both train and validate
        sig_all = perfs.train.(perf_type).sig & perfs.validate1.(perf_type).sig;
        valid_all = perfs.train.valid_features & perfs.validate1.valid_features;
end

%% Filter features

% Keep features which are valid and sig. in all sets
fValues = cell(size(fValues_all));
fIds = cell(size(fValues));
for ch = 1 : size(sig_all, 1)
    fIds{ch} = find(valid_all(ch, :) & sig_all(ch, :));
    fValues{ch} = fValues_all{ch}(:, fIds{ch});
end

%% Take top N

topN = 0;

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

%% Scale for tSNE
% Seems like tsne doesn't work with Inf values

scale = 1;

if scale == 1
    for ch = 1 : length(fValues)
        fValues{ch} = BF_NormalizeMatrix(fValues{ch}, 'mixedSigmoid');
    end
end

%% t-SNE

ch = 6;

tsne_space = tsne(fValues{ch}', 'Distance', 'spearman', 'numDimensions', 2);

%% Plot, colour by performance

colour_dset = 1;

colours = perfs.(data_sets{colour_dset}).(perf_type).performances(ch, fIds{ch});

figure;
colormap inferno
scatter(tsne_space(:, 1), tsne_space(:, 2), [], colours, '.', 'MarkerEdgeAlpha', 0.75);
colorbar;

%% Plot, colour by master operation

%% Plot, colour by master operation, size by performance

masters = hctsa{ch}.Operations.MasterID(fIds{ch});
% Group master operation list by broad category
%   Broad category determined by starting letters
master_strings = hctsa{ch}.MasterOperations.Code(masters);
groups = cellfun(@(x) x(1:2), master_strings, 'UniformOutput', false);

feature_string = repmat({'[a-zA-Z0-9]+_?[a-zA-Z]+[a-zA-Z0-9]*'}, size(master_strings));
[starts, ends] = cellfun(@regexp, master_strings, feature_string, 'UniformOutput', false);
groups = cellfun(@(x,y,z) x(y:z), master_strings, starts, ends, 'UniformOutput', false);

sizes = 10*perfs.(data_sets{colour_dset}).(perf_type).performances(ch, fIds{ch}); % sizes not very important if all features are significant

groups_unique = sort(unique(groups));
group_colours = (1:length(groups_unique));
group_colourmapping = containers.Map(groups_unique, group_colours);
colours = cellfun(@(x) group_colourmapping(x), groups);

[sorted, order] = sort(groups);

%%

figure;
subplot(1, 5, [1 4]);

h = nan(length(groups_unique), 1);
hold on;
for g = 1 : length(groups_unique)
    plot_groups = strcmp(groups, groups_unique{g});
    h(g) = scatter(tsne_space(plot_groups, 1), tsne_space(plot_groups, 2), [], repmat(group_colourmapping(groups_unique{g}), size(find(plot_groups))), 'o', 'filled', 'MarkerEdgeAlpha', 0.75);
end

title(['Feature t-SNE: ch ' num2str(ch)]);
xlabel('x');
ylabel('y');

% Matlab legend displays maximum of 50 items
h_half = floor(length(h)/2);
legend(h(1:h_half), groups_unique(1:h_half), 'Location', 'eastoutside', 'Orientation', 'vertical', 'NumColumns', 2, 'Interpreter', 'none')
s = subplot(1, 5, 5, 'visible', 'off');
legend(s, h(h_half+1:end), groups_unique(h_half+1:end), 'Location', 'westoutside', 'Orientation', 'vertical', 'NumColumns', 2, 'Interpreter', 'none')

%%
gscatter(tsne_space(:, 1), tsne_space(:, 2), groups);
legend('Location', 'eastoutside', 'Orientation', 'vertical', 'NumColumns', 2, 'Interpreter', 'none')
xlabel('x');
ylabel('y');

%% Cluster from tSNE

