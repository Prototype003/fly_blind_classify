%% Description

%{

% Plot feature matrix for given dataset

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
dset_ids = (1:length(dsets));

addpath('../');

%% Load stats
% Get all performances and stats using
%   get_stats_evaluation(preprocess_string)

stats_dir = ['results' preprocess_string filesep];
stats_file = 'stats_multidoseSplit.mat';

% Should have variable stats
load([stats_dir stats_file]);

%% Features which are valid in all datasets

ch = 6;

valid_all = ones(size(stats.train.valid_features));
for d = 1 : length(dsets)
    
    disp(['====']);
    
    tmp = stats.(dsets{d}).valid_features;
    
    disp(['ch' num2str(ch) '-' dsets{d} ': ' num2str(numel(find(tmp(ch, :))))]);
        
    valid_all = valid_all & tmp;
    
    disp(['total ' num2str(numel(find(valid_all(ch, :)))) ' valid across datasets']);
end

%% Load hctsa files
% And join channels together

data_sources = {'train', 'multidose', 'singledose', 'sleep'};

source_dir = ['../hctsa_space' preprocess_string '/'];

hctsas = cell(size(data_sources));

for d = 1 : length(data_sources)
    
    ch = 1;
    source_file = ['HCTSA_' data_sources{d} '_channel' num2str(ch) '.mat'];
    disp(['loading ' source_file]);
    tic;
    hctsas{d} = load([source_dir source_file]);
    t = toc;
    disp(['loaded in ' num2str(t) 's']);
    
    tmp = nan([size(hctsas{d}.TS_DataMat) size(valid_all, 1)-1]);
    hctsas{d}.TS_DataMat = cat(3, hctsas{d}.TS_DataMat, tmp);
    
    for ch = 2 : size(valid_all, 1)
        
        source_file = ['HCTSA_' data_sources{d} '_channel' num2str(ch) '.mat'];
        disp(['loading ' source_file]);
        tic;
        hctsa_ch = load([source_dir source_file]);
        t = toc;
        disp(['loaded in ' num2str(t) 's']);
        
        % For now, only copy feature values, ignore the other stuff
        %   Because we are only plotting feature values
        hctsas{d}.TS_DataMat(:, :, ch) = hctsa_ch.TS_DataMat;
        
    end
    
end

%% Split multidose dataset into multidose8 and multidose4

disp('splitting multidose');
tic;
md_split = split_multidose(hctsas{2});

hctsas{5} = md_split{1};
hctsas{6} = md_split{2};

hctsas = hctsas([1 5 6 3 4]);
toc

dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
dset_bases = {'train', 'multidose', 'multidose', 'singledose', 'sleep'};

%% Replace values with nans for invalid features
% Only care about features which are valid across all datasets

for d = 1 : length(hctsas)
    
    for ch = 1 : size(hctsas{d}.TS_DataMat, 3)
        tic;
        for f = 1 : size(hctsas{d}.TS_DataMat, 2)
            
            if valid_all(ch, f) == 0
                hctsas{d}.TS_DataMat(:, f, ch) = nan;
            end
            
        end
        toc
    end
    
end

disp('invalid features at each channel replaced with nans');

%% Exclude rows corresponding to particular unused conditions

valid_rows = cell(size(dsets));
for d = 1 : length(dsets)
    
    switch dsets{d}
        case 'train'
            keywords = {};
        case {'multidose', 'multidose4', 'multidose8'}
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

%% Average performance across evalation flies together

perf_types = {'nearestMedian', 'consis'};
dset_suffixes = {{'', 'BatchNormalised'}, {''}};

% Store averaged performances in here
stats.evaluate = struct();
stats.evaluate.nearestMedian = struct();
stats.evaluate.nearestMedianBatchNormalised = struct();
stats.evaluate.consis = struct();

% We only will average one condition pair from each dataset
stats.evaluate.nearestMedian.performances = cell(1);
stats.evaluate.nearestMedianBatchNormalised.performances = cell(1);
stats.evaluate.consis.performances = cell(1);

disp(['averaging performances across evaluation flies' newline]);

for ptype = 1 : length(perf_types)
	perf_type = perf_types{ptype};
	for dsuffix = 1 : length(dset_suffixes{ptype})
		dset_suffix = dset_suffixes{ptype}{dsuffix};
		
		perf_string = [perf_type dset_suffixes{ptype}{dsuffix}];
		
		stats.evaluate.([perf_type dset_suffix]).performances = cell(1);
		stats.evaluate.([perf_type dset_suffix]).performances{1} =...
			zeros(size(stats.train.(perf_type).performances{1}));
		stats.evaluate.([perf_type dset_suffix]).sig =...
			ones(size(stats.train.(perf_type).sig));
		
		disp([perf_type ' ' dset_suffixes{ptype}{dsuffix}]);
		
		% Use the main condition pair of each dataset
		dset_pairings = cell(size(dsets));
		dset_flies = zeros(size(dsets));
		for d = 2 : length(dsets) % Exclude discovery flies from the average
			
			if strcmp(dsets{d}, 'train')
				dset_field = dsets{d}; % No batch-normalised classification for discovery flies
			else
				dset_field = [dsets{d} dset_suffix];
			end
			
			% Get main condition pair for the dataset
			[conds, cond_labels, cond_colours, stats_order, conds_main] =...
				getConditions(dsets{d});
			dset_pairings{d} = conds_main;
			pair_statsIds = find(ismember(stats_order, conds_main));
			statsPairId = find(ismember(sort(stats.(dset_field).(perf_type).condition_pairs, 2), sort(pair_statsIds), 'rows'));
			
			% Check the main pair by printing it out
			disp([dsets{d} ': '...
				stats.(dset_field).(perf_type).conditions{stats.(dset_field).(perf_type).condition_pairs(statsPairId, 1)}...
				' x '...
				stats.(dset_field).(perf_type).conditions{stats.(dset_field).(perf_type).condition_pairs(statsPairId, 2)}]);
			
			[~, nFlies] = getDimensionsFast(dsets{d});
			dset_flies(d) = nFlies;
			
			% Add the performances, weight by number of flies
			stats.evaluate.(perf_string).performances{1} =...
				stats.evaluate.(perf_string).performances{1} +...
				(stats.(dset_field).(perf_type).performances{statsPairId}*nFlies);
			
			% Add the significance decision
			stats.evaluate.(perf_string).sig =...
				stats.evaluate.(perf_string).sig &...
				stats.(dset_field).(perf_type).sig(:, :, statsPairId);
			
		end
		
		stats.evaluate.(perf_string).performances{1} = stats.evaluate.(perf_string).performances{1} ./ sum(dset_flies);
		
		disp(newline);
		
	end
end

stats.train.nearestMedianBatchNormalised = stats.train.nearestMedian;

%% Setup for performance plots

data_sets = {'train', 'evaluate'};

% Average, max, min number of valid feaures across the channels
ch_valid_features = valid_all;
nValid = sum(valid_all, 2);
nValid_mean = mean(nValid)
nValid_min = min(nValid)
nValid_max = max(nValid)

% Get performances
perfs = stats;

%% Common plotting variables

ch = 1;

%% Create Figure 4 panels
% Classification results

% b - number of sig. features
% c - discovery performance against mean evaluation performance
% d,e - same as b,c but for classification after normalising values

ch = 1;

figure;
set(gcf, 'Color', 'w');

subplot_rows = 4;
subplot_cols = 4;

% B - number of sig. features per channel
subplot_pos = [1 6];
perf_type = 'nearestMedian'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
h = summary_plots_evaluate(2, ch, perf_type, [subplot_rows, subplot_cols], subplot_pos, {'train', 'evaluate'}, ch_valid_features, perfs, hctsas);
title(['b sig. features per channel per combination of dsets' newline 'means weighted by number of flies per dset grouping' newline perf_type]);
%h.Title.String = ['b ' h.Title.String]; % Doesn't seem to work with newline
%axis square

% C - discovery performance against mean evaluation performance
subplot_pos = [3 8];
perf_type = 'nearestMedian'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
h = summary_plots_evaluate(1, ch, perf_type, [subplot_rows, subplot_cols], subplot_pos, {'train', 'evaluate'}, ch_valid_features, perfs, hctsas);
h.Title.String = ['c ' h.Title.String];
%axis square

% D - number of sig. features per channel
subplot_pos = [9 14];
perf_type = 'nearestMedianBatchNormalised'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
h = summary_plots_evaluate(2, ch, perf_type, [subplot_rows, subplot_cols], subplot_pos, {'train', 'evaluate'}, ch_valid_features, perfs, hctsas);
title([perf_type]);
h.Title.String = ['d ' h.Title.String];
%axis square

% E - discovery performance against mean evaluation performance
subplot_pos = [11 16];
perf_type = 'nearestMedianBatchNormalised'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
h = summary_plots_evaluate(1, ch, perf_type, [subplot_rows, subplot_cols], subplot_pos, {'train', 'evaluate'}, ch_valid_features, perfs, hctsas);
h.Title.String = ['e ' h.Title.String];
%axis square

%% Print figure

print_fig = 0;

if print_fig == 1

	figure_name = '../figures_stage2/fig4b_raw';

	set(gcf, 'PaperOrientation', 'Portrait');

	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG

end

%% Create Figure 7 panels
% Consistency results

% b - number of sig. features
% c - discovery performance against mean evaluation performance
% d,e - same as b,c but for classification after normalising values

ch = 1;

figure;
set(gcf, 'Color', 'w');

subplot_rows = 4;
subplot_cols = 4;

% B - number of sig. features per channel
subplot_pos = [1 6];
perf_type = 'consis'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
h = summary_plots_evaluate(2, ch, perf_type, [subplot_rows, subplot_cols], subplot_pos, {'train', 'evaluate'}, ch_valid_features, perfs, hctsas);
title(['b sig. features per channel per combination of dsets' newline 'means weighted by number of flies per dset grouping' newline perf_type]);
%h.Title.String = ['b ' h.Title.String]; % Doesn't seem to work with newline
%axis square

% C - discovery performance against mean evaluation performance
subplot_pos = [3 8];
perf_type = 'consis'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
h = summary_plots_evaluate(1, ch, perf_type, [subplot_rows, subplot_cols], subplot_pos, {'train', 'evaluate'}, ch_valid_features, perfs, hctsas);
h.Title.String = ['c ' h.Title.String];
%axis square

%% Print figure

print_fig = 0;

if print_fig == 1

	figure_name = '../figures_stage2/fig7b_raw';

	set(gcf, 'PaperOrientation', 'Portrait');

	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG

end

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

%% Print figure

print_fig = 0;

if print_fig == 1

	figure_name = '../figures_stage2/fig4a_raw';

	set(gcf, 'PaperOrientation', 'Portrait');

	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG

end