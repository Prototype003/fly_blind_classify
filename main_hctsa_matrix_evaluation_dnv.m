%% Description

%{

Plot DNV matrix for given dataset

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
%data_sources = {'sleep'};

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
    hctsas{d}.TimeSeriesUnfiltered = hctsas{d}.TimeSeries(valid_rows{d}, :);
    hctsas{d}.TimeSeries = hctsas{d}.TimeSeries(valid_rows{d}, :);
end

%% Find which features are valid across all datasets

valid_all = ones(size(valid_features{1}));
for d = 1 : length(data_sources)
    
    valid_all = valid_all & valid_features{d};
    
end

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

%{
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
%}

%% Get differences in normalised values for each dataset

% 0 - don't compute (takes lots of memory)
% 1 - compute, save, then remove from workspace
compute_multidose = 0;

dnvs = cell(size(hctsas));
dnvs_rows = cell(size(dnvs));
dnv_pairs = cell(size(dnvs));
for d = 1 : length(hctsas)
    
    switch data_sources{d}
        case 'train'
            conditions{d}.conditions{1} = {'condition1'};
            conditions{d}.cond_labels{1} = {'W'};
            conditions{d}.conditions{2} = {'condition2'};
            conditions{d}.cond_labels{2} = {'A'};
        case 'multidose'
            conditions{d}.conditions{1} = {'conditionWake', 'conditionPost_Isoflurane', 'conditionRecovery'};
            conditions{d}.cond_labels{1} = {'E1W', 'E1PostIso', 'E1R'};
            conditions{d}.conditions{2} = {'conditionIsoflurane_0.6', 'conditionIsoflurane_1.2'};
            conditions{d}.cond_labels{2} = {'E1A0.6', 'E1A1.2'};
        case 'singledose'
            conditions{d}.conditions{1} = {'conditionWake', 'conditionPostIsoflurane'};
            conditions{d}.cond_labels{1} = {'E2W', 'E2PostIso'};
            conditions{d}.conditions{2} = {'conditionIsoflurane'};
            conditions{d}.cond_labels{2} = {'E2A'};
        case 'sleep'
            conditions{d}.conditions{1} = {'conditionwakeEarly', 'conditionwake'};
            conditions{d}.cond_labels{1} = {'E3WE', 'EW'};
            conditions{d}.conditions{2} = {'conditionsleepEarly', 'conditionsleepLate'};
            conditions{d}.cond_labels{2} = {'E3SE', 'E3SL'};
    end
    
    % Get all pairings of the two condition classes
    a = (1:length(conditions{d}.conditions{1}));
    b = (1:length(conditions{d}.conditions{2}));
    [A, B] = ndgrid(a, b);
    pairs = [A(:), B(:)]; % each row gives the pair indexes
    dnv_pairs{d} = pairs;
    
    for p = 1 : size(pairs, 1)
        
        if d == 2
            
            if compute_multidose == 1
                
                tic;
                [dnv_vals, dnv_rows] = dnv(hctsas{d}, [],...
                    conditions{d}.conditions{1}(pairs(p, 1)),...
                    conditions{d}.conditions{2}(pairs(p, 2)));
                t = toc;
                disp([conditions{d}.conditions{1}{pairs(p, 1)} ' ' conditions{d}.conditions{2}{pairs(p, 2)} ' ' num2str(t) 's']);
                
                dnv_pair = dnv_pairs{d}(p, :);
                
                figure_space = 'figure_workspace/';
                figure_name = ['hctsa_matrix_evaluation_dnv_multidose_' conditions{d}.conditions{1}{pairs(p, 1)} '-' conditions{d}.conditions{2}{pairs(p, 2)}];
                tic;
                save([figure_space figure_name '.mat'], 'dnv_vals', 'dnv_rows', 'dnv_pair', '-v7.3', '-nocompression');
                t = toc;
                disp(['saved in ' num2str(t) 's']);
                
            else
                disp(['ignoring multidose dnv']);
            end
            
        else
            
            tic;
            [dnvs{d}{p}, dnvs_rows{d}{p}] = dnv(hctsas{d}, [],...
                conditions{d}.conditions{1}(pairs(p, 1)),...
                conditions{d}.conditions{2}(pairs(p, 2)));
            t = toc;
            disp([conditions{d}.conditions{1}{pairs(p, 1)} ' ' conditions{d}.conditions{2}{pairs(p, 2)} ' ' num2str(t) 's']);
            
        end
    end
    
end

%% Load performance

perf_type = 'consis';
perf_string = 'consistencies'; % actual field name

out_dir = ['classification_nearestMean/results' preprocess_string '/'];
out_prefix = [perf_type '_nearestMedian_'];

perfs = cell(size(hctsas));
for d = 1 : length(hctsas)
    
    disp(['loading ' data_sources{d} ' ' perf_string]);
    tic;
    
    perf = load([out_dir out_prefix data_sources{d}]);
    
    perfs{d} = cell(size(dnv_pairs{d}, 1), 1);
    
    switch data_sources{d}
        case 'train'
            perfs{d}{1} = squeeze(perf.(perf_string)(channel, valid_all, :, :));
        case {'multidose', 'singledose', 'sleep'}
            for pair = 1 : size(dnv_pairs{d}, 1)
                perfs{d}{pair} = squeeze(perf.(perf_string){pair}(channel, valid_all, :, :));
            end
    end
    
    t = toc;
    disp(['loaded ' data_sources{d} ' ' perf_string ' in t=' num2str(t) 's']);
    
end

%% Reorder features based on similarity

ref_set = 4;

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

%% Get matrix which will be visualised

d = 4;
p = 4;

if strcmp(data_sources{d}, 'multidose') % need to load from file
    
    figure_space = 'figure_workspace/';
    figure_name = ['hctsa_matrix_evaluation_dnv_multidose_' conditions{d}.conditions{1}{dnv_pairs{d}(p, 1)} '-' conditions{d}.conditions{2}{dnv_pairs{d}(p, 2)} '.mat'];
    tic;
    tmp = load([figure_space figure_name]);
    t = toc;
    disp(['loaded ' figure_name ' in ' num2str(t) 's']);
    vis_matrix = tmp.dnv_vals(:, fOrder);
    
else
    vis_matrix = dnvs{d}{p}(:, fOrder);
end

%% Create figure (matrix)

figure;

subplot(3, 1, [1 2]);

imagesc(vis_matrix);
pair = dnv_pairs{d}(p, :);
title([data_sources{d} ' ' ...
    conditions{d}.conditions{1}{pair(1)} '-'...
    conditions{d}.conditions{2}{pair(2)} '; '...
    'features sorted by ' data_sources{ref_set} ' similarity'],...
    'Interpreter', 'none');
%xlabel('feature');

%% Find axis tick locations and create tick labels

% Get all the keywords for each row
%kw = split(keywords_all, ',');
%kw = kw(:, 4);

if d == 2
    kw = cellfun(@(x) split(x, ','), tmp.dnv_rows, 'UniformOutput', false);
else
    kw = cellfun(@(x) split(x, ','), dnvs_rows{d}{p}, 'UniformOutput', false);
end

% Get just fly IDs (assumes flies are first keyword)
for k = 1 : length(kw)
    kw{k} = kw{k}{1};
end

% Get first occurrence of each string
[kw_unique, kw_row] = unique(kw, 'stable');

%% Add axis ticks and labels

%xlabel('feature');

yticks(kw_row);
yticklabels(kw_unique);

set(gca, 'TickDir', 'out');

%% Other details

set(gcf, 'Color', 'w');

c = colorbar('northoutside');
ylabel(c, 'DNV');

neg = viridis(256);
pos = inferno(256);
negPos_map = cat(1, flipud(neg(1:128, :)), pos(129:end, :));
negPos_map = flipud(cbrewer('div', 'RdBu', 100));
colormap(negPos_map);

%% Plot performance measure

subplot(3, 1, 3);

yyaxis left

m = mean(mean(perfs{d}{p}, 3), 2);
m = m(fOrder);

plot(m);
axis tight

hold on
line([xlim], [0.5 0.5], 'Color', 'k');

ylabel(perf_type);

ylim([0 1]);

hold off;

%% Plot average DNV across rows

yyaxis right

plot(mean(vis_matrix, 1));
axis tight

hold on
line([xlim], [0 0], 'Color', 'k');

xlabel('feature');
ylabel('mean DNV');

%c = colorbar; % just to align the x axis with the matrix plot
%ylabel(c, 'just for aligning x-axis');

ylim([-1 1]);

hold off;

%% Saving figures doesn't work...
%{
if d == 2 % save fig
    figure_space = 'figure_workspace/';
    figure_name = ['hctsa_matrix_evaluation_dnv_multidose_' conditions{d}.conditions{1}{dnv_pairs{d}(p, 1)} '-' conditions{d}.conditions{2}{dnv_pairs{d}(p, 2)} '.fig'];
    tic;
    saveas(gcf, [figure_space figure_name]);
    t = toc;
    disp(['saved ' figure_space figure_name ' in ' numstr(t) 's']);
end
%}

%% Compare within-dataset consistency to consistency relative to discovery
% Use mean dnv as within-dataset consistency

d = 4;
p = 4;

consis_disc = mean(mean(perfs{d}{p}, 3), 2);
consis_within = abs(mean(dnvs{d}{p}, 1)') + 0.5; % shift scale up

consis_diff = consis_within - consis_disc;

% Find feature which has large difference in mean(dnv) and consistency
big_diff = find(consis_disc < 0.3 & consis_within > 0.65);
[max_diff, max_idx] = max(consis_diff);

feature_list = find(valid_all);
max_feature = feature_list(max_idx);
