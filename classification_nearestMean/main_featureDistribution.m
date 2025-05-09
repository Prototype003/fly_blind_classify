%% Description

%{

Plot feature value distribution for a single feature

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

%% Get performance stats

perfs = get_stats(preprocess_string);

%%

data_sets = {'train', 'validate1'};
perf_types = {'nearestMedian', 'consis'};

%% Get raw values for each channel

fValues_all = cell(size(perfs.train.valid_features, 1), 1);
fValues_each = cell(size(perfs.train.valid_features, 1), 1);
hctsa = cell(size(fValues_all));
for ch = 1 : size(fValues_all, 1)
    tic;
    % Load and concatenat TS_DataMat for each dataset
    ds = cell(1);
    for d = 1 : length(data_sets)
        hctsa{ch}.(data_sets{d}) = hctsa_load(data_sets{d}, ch, preprocess_string);
        ds{d} = hctsa{ch}.(data_sets{d}).TS_DataMat;
    end
    fValues_all{ch} = cat(1, ds{:});
    fValues_each{ch}.train = ds{1};
    fValues_each{ch}.validate1 = ds{2};
    toc
end

%% Get wanted features

ch = 6;
dset = data_sets{1};

% Limit SP_Summaries to handpicked features
featureIDs = [...
    find(contains(hctsa{ch}.(dset).Operations.Name, 'SP_Summaries') & (contains(hctsa{ch}.(dset).Operations.Name, '_area_5_1') | contains(hctsa{ch}.(dset).Operations.Name, '_logarea_5_1')));... power at particular bands (can only do broadband power)
    find(contains(hctsa{ch}.(dset).Operations.Name, 'SP_Summaries') & (contains(hctsa{ch}.(dset).Operations.Name, '_area_') | contains(hctsa{ch}.(dset).Operations.Name, '_logarea_')));...
    find(contains(hctsa{ch}.(dset).Operations.Name, '_centroid')); find(contains(hctsa{ch}.(dset).Operations.Name, '_wmax_95'));... spectral edge frequencies
    find(contains(hctsa{ch}.(dset).Operations.Name, '_spect_shann_ent'))... spectral entropy
    ];

features = hctsa{ch}.(dset).Operations.Name(featureIDs);

% Display feature names, IDs, and performances
perf_type = perf_types{1};
perf_train = perfs.train.(perf_type).performances(ch, featureIDs);
perf_vals = perfs.validate1.(perf_type).performances(ch, featureIDs);
[sorted, order] = sort(perf_vals, 'descend');
s = sprintf('%s\t\t%s\t%s\t\t%s',...
    'fID',...
    'train',...
    'val',...
    'name');
disp(s);
for f = 1 : length(featureIDs)
    fname = hctsa{ch}.(dset).Operations.Name(featureIDs(order(f)));
    s = sprintf('f%i\t%.4f\t%.4f\t%s',...
        featureIDs(order(f)),...
        perf_train(order(f)),...
        perf_vals(order(f)),...
        fname{1});
    disp(s);
end

%% Plot value distributions

fID = 4428;

figure;
histogram(fValues_each{ch}.train(:, fID));
hold on;
histogram(fValues_each{ch}.validate1(:, fID));

%% Scatter plot

fID = 4428;

cond_colours = {'r', 'b'};
dset_markers = {'o', 'x'};

figure;
for d = 1 : length(data_sets)
    dset = data_sets{d};
    
    for cond = 1 : 2
        keywords = {['condition' num2str(cond)]};
        match = getIds(keywords, hctsa{ch}.(dset).TimeSeries);
        
        y = hctsa{ch}.(dset).TS_DataMat(match, fID);
        x = (1:length(y));
        
        scatter(x, y, [], cond_colours{cond}, dset_markers{d});
        
        hold on;
    end
    
end

title(['f' num2str(fID) ': ' hctsa{ch}.(dset).Operations.Name{fID}], 'Interpreter', 'none');
legend({'dis-wake', 'dis-anest', 'val-wake', 'val-anest'});
xlabel('arbitrary');
ylabel('feature value');