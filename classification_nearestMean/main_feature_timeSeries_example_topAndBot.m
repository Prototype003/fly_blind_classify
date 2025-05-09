%% Description

%{

For selected feature(s), show example time-series
Show for highest value of the feature, and lowest value

Select two features - a "good" feature which discriminates, and one which
doesn't

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

%% Load feature thresholds and directions

thresh_dir = ['results' preprocess_string filesep];
thresh_file = 'class_nearestMedian_thresholds.mat';
load([thresh_dir thresh_file]);

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

%% Average performance across evaluation flies together

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

%% Get main condition pair for each dataset

dset_pairings = cell(size(dsets));
for d = 1 : length(dsets)
	[conds, cond_labels, cond_colours, stats_order, conds_main] =...
		getConditions(dsets{d});
	dset_pairings{d} = conds_main;
end

%% Select features
% Ch1 best classification feature - 5072 : NL_BoxCorrDim_50_ac_5_minr13
% Ch1 non-sig. classification feature - 4676 : SP_Summaries_fft_area_5_1
%	5_1 - divides spectrum into 5 bands, takes power from 1st band
%	(sig. perf in discovery flies, but not all evaluation flies)

% Note - for log transform, check for negative minimum values in fVals{feature}
%	And update transform_features to offset accordingly for log transform


% NL_BoxCorrDim_50_ac_5_minr13 - 5072
ch = 1;
fIDs = [5072 4676];
fNames = hctsas{1}.Operations.Name(fIDs);
perf_type = 'nearestMedian'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
transform_types = {'log', ''};
transform_strings = {'log(log(f1+12.2))', 'f2'}; % cross-ref/update transform_values() function as needed
%%
% AC_33 - 125
% AC_31 - 123
ch = 1;
fIDs = [123 4676];
fNames = hctsas{1}.Operations.Name(fIDs);
perf_type = 'nearestMedian'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
transform_types = {'', ''};
transform_strings = {'log(log(f1+1.31))', 'f2'}; % cross-ref/update transform_values() function as needed
%%
% ST_LocalExtrema_n50_stdmax - 3447
ch = 1;
fIDs = [3447 4676];
fNames = hctsas{1}.Operations.Name(fIDs);
perf_type = 'nearestMedian'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
transform_types = {'log', ''};
transform_strings = {'log(f1+min)', 'f2'}; % cross-ref/update transform_values() function as needed

% Select which fly and dataset to show from?
%	Or would it be better to show one time series from each fly?
%		Total 12+18+19 time-series? Or show just from one dataset?

%% 2 features which achieved sig classification but non-sig consis
ch = 1;
fIDs = [12 42];
fIDs = [133 319];
fNames = hctsas{1}.Operations.Name(fIDs);
perf_type = 'nearestMedian'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
transform_types = {'', ''};
transform_strings = {'f1', 'f2'}; % cross-ref/update transform_values() function as needed

%% Get all sig features

ch = 1;

% Note - ~4 seconds per feature, so keep feature selection low
sig_only = 1;

main_pairs = [1 1 1 1 3];

% Include specific features
include_features = zeros(1, size(valid_all, 2));
include_features(4676) = 1;

if sig_only == 1
	sig_all = stats.train.nearestMedian.sig(ch, :) & valid_all(ch, :);
	
	for d = 2 : length(dsets)
		
		sig_all = sig_all & stats.(dsets{d}).nearestMedian.sig(ch, :, main_pairs(d));
		
	end
	
	sig_all = sig_all | include_features(ch, :);
	
else
	
	% Consider all valid features
	
	sig_all = valid_all(ch, :);
	
end

fIDs = find(sig_all);


%% Get all values

values = cell(size(fIDs));
labels = cell(size(fIDs));
for f = 1 : length(fIDs)
	
	tic;
	[values{f}, labels{f}] = featureValueDistributionFromHctsas(hctsas, fIDs(f), ch);
	toc
	
	% Get only values from the main condition pair
	for d = 1 : length(dsets)
		values{f}{d} = values{f}{d}(:, :, dset_pairings{d});
		labels{f}{d} = labels{f}{d}(:, :, dset_pairings{d});
	end
	
end

%% Combine values together across all flies and epochs

values_together = cell(size(values));
labels_together = cell(size(labels));
dsets_together = cell(size(values));

for f = 1 : length(fIDs)
	
	together_v = [];
	together_l = {};
	together_d = {};
	
	for d = 1 : length(values{f})
		tmp = values{f}{d};
		tmp = reshape(tmp, [size(tmp, 1)*size(tmp, 2), size(tmp, 3)]);
		together_v = cat(1, together_v, tmp);
		
		tmp = labels{f}{d};
		tmp = reshape(tmp, [size(tmp, 1)*size(tmp, 2), size(tmp, 3)]);
		together_l = cat(1, together_l, tmp);
		
		dset_label = repmat({d}, [size(tmp, 1), size(tmp, 2)]);
		together_d = cat(1, together_d, dset_label);
	end
	
	values_together{f} = together_v;
	labels_together{f} = together_l;
	dsets_together{f} = together_d;
	
	
end
%values_together = cat(3, values_together{:});

%% Find the most extreme values for each feature
% Get extreme wake and unawake value

fID_directions = directions(ch, fIDs);

extreme_values = nan(length(fIDs), 2);
extreme_tseries = cell(size(values_together));

for f = 1 : length(fIDs)
	
	extreme_tseries{f} = []; % first pos holds position of largest value
	
	if directions(ch, fIDs(f)) == 1
		
		% wake > anaesthesia, so
		%	use greatest wake value and
		%	use lowest unawake value
		
		[greatest, greatest_pos] = max(values_together{f}(:, 1));
		
		[smallest, smallest_pos] = min(values_together{f}(:, 2));
		
		extreme_tseries{f}(1) = greatest_pos;
		extreme_tseries{f}(2) = smallest_pos;
		
		extreme_values(f, 1) = greatest;
		extreme_values(f, 2) = smallest;
		
	else % directions(ch, fIDs(f)) == 0
		
		% wake < anaesthesia, so
		%	use lowest wake value and
		%	use greatest anaest value
		
		[smallest, smallest_pos] = min(values_together{f}(:, 1));
		
		[greatest, greatest_pos] = max(values_together{f}(:, 2));
		
		extreme_tseries{f}(1) = greatest_pos;
		extreme_tseries{f}(2) = smallest_pos;
		
		extreme_values(f, 1) = greatest;
		extreme_values(f, 2) = smallest;
		
	end
	
end

%% Plot top timeseries for each feature

figure;
set(gcf, 'Color', 'w');

hilo_colours = {'r', 'b'};

offset = 0;
for f = 1 : length(fIDs)
	
	for hilo = 1 : 2
		
		% Get time series label
		labelString = labels_together{f}{extreme_tseries{f}(hilo), hilo};
		d = dsets_together{f}{extreme_tseries{f}(hilo), hilo};
		
		% Find time series with matching label
		rowID = find(cellfun(@(x) strcmp(x, labelString), hctsas{d}.TimeSeries.Name));
		series = hctsas{d}.TimeSeries.Data{rowID};
		
		plot(series + offset, hilo_colours{hilo});
		hold on;
		
	end
	offset = offset + 250;
end

axis tight


%% Plot for selected feature (line plots)
% Sort time-series by value and then plot top and bottom

figure;
set(gcf, 'Color', 'w');

% NL_BoxCorrDim_50_ac_5_minr13 - 5072
% AC_30 - 122
% SP_Summaries_fft_area_5_1 - 4676

f = find(fIDs == 122);
f = find(fIDs == 4676); % SP_Summaries...
f = find(fIDs == 5072);

[sorted, tseries_order] = sort(values_together{f}(:, 1), 'ascend');

for hilo = 1 : 2
	
	subplot(2, 1, hilo);
	
	offset = 0;
	for e = length(tseries_order) : -1 : length(tseries_order) - 19
		
		% Get time series label
		labelString = labels_together{f}{tseries_order(e), hilo};
		d = dsets_together{f}{tseries_order(e), hilo};
		
		% Find time series with matching label
		rowID = find(cellfun(@(x) strcmp(x, labelString), hctsas{d}.TimeSeries.Name));
		series = hctsas{d}.TimeSeries.Data{rowID};
		
		plot(series + offset, hilo_colours{hilo});
		hold on;
		
		offset = offset + 200;
	end
	
	axis tight
	
end

%% Plot for selected features (line plots)
% Sort time-series by value and then plot top and bottom

figure;
set(gcf, 'Color', 'w');

sp_counter = 1;
cond_strings = {'wake', 'unwake'};

% NL_BoxCorrDim_50_ac_5_minr13 - 5072
% AC_30 - 122
% SP_Summaries_fft_area_5_1 - 4676
% DN_CompareKSFit_uni_psx - 3235
% SB_TransitionpAlphabet_20_ac_trcov_jump - 1498

show_fs = [5072 122 4676];

nExamples = 10;

for f_counter = 1 : length(show_fs)
	
	f = find(fIDs == show_fs(f_counter));
	
	tseries = [];
	
	[sorted, tseries_order] = sort(values_together{f}(:, 1), 'ascend');
	
	for hilo = 1 : 2
		
		subplot(length(show_fs), 2, sp_counter);
		
		offset = 0;
		for e = length(tseries_order) : -1 : length(tseries_order) - (nExamples-1)
			
			% Get time series label
			labelString = labels_together{f}{tseries_order(e), hilo};
			d = dsets_together{f}{tseries_order(e), hilo};
			
			% Find time series with matching label
			rowID = find(cellfun(@(x) strcmp(x, labelString), hctsas{d}.TimeSeries.Name));
			series = hctsas{d}.TimeSeries.Data{rowID};
			
			plot(series + offset, hilo_colours{hilo});
			hold on;
			
			% The first one has the greatest value, so have it on top
			offset = offset - 200;
		end
		
		axis tight
		
		title([num2str(show_fs(f_counter)) ' ' cond_strings{hilo}]);
		
		ylabel(num2str(show_fs(f_counter)));
		xlabel('t (ms)');
		
		sp_counter = sp_counter + 1;
	end

end

%% Print figure

print_fig = 1;

if print_fig == 1
	
	figure_name = '../figures_stage2/tseriesExamples_raw';
	
	set(gcf, 'PaperOrientation', 'Portrait');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG
	
end


%% Plot for selected features (color plots)

figure;
set(gcf, 'Color', 'w');

% NL_BoxCorrDim_50_ac_5_minr13 - 5072
% AC_30 - 122
% SP_Summaries_fft_area_5_1 - 4676
show_fs = [5072 122 4676 3245];

nExamples = 20;

for f_counter = 1 : length(show_fs)
	
	subplot(length(show_fs), 1, f_counter);
	
	f = find(fIDs == show_fs(f_counter));
	
	tseries = [];
	
	[sorted, tseries_order] = sort(values_together{f}(:, 1), 'ascend');
	
	for hilo = 1 : 2
		
		for e = length(tseries_order) : -1 : length(tseries_order) - (nExamples-1)
			
			% Get time series label
			labelString = labels_together{f}{tseries_order(e), hilo};
			d = dsets_together{f}{tseries_order(e), hilo};
			
			% Find time series with matching label
			rowID = find(cellfun(@(x) strcmp(x, labelString), hctsas{d}.TimeSeries.Name));
			series = hctsas{d}.TimeSeries.Data{rowID};
			
			tseries = cat(1, tseries, series');
		end
		
	end
	
	% center colour axis around 0
	tmp = abs(tseries);
	tmp = prctile(tmp(:), 99);
	clim = [-tmp tmp];
	
	imagesc(tseries, clim);
	c = colorbar; title(c, 'V');
	title(num2str(show_fs(f_counter)));
	
	set(gca, 'YTick', [1 nExamples+1], 'YTickLabels', {'wake', 'unawake'});
	
	xlabel('t (ms)');
	
end


%% Print figure

print_fig = 0;

if print_fig == 1
	
	figure_name = '../figures_stage2/fig5_tseriesExample_raw';
	
	set(gcf, 'PaperOrientation', 'Portrait');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG
	
end

%%

function [scatter_ax, hist_ax] = fly_scatter(sp_scatter_ax, sp_hist_ax, values, dsets, ch, fID, fName, perf_type, thresholds, transform_type, transform_string)

% Convert cell array of values to a single matrix

% Equalise epoch dimension by adding nans
epochMax = max(cellfun(@(x) max(size(x)), values));
values_equalEpochs = cell(size(values));
for d = 1 : length(dsets)
	if size(values{d}, 1) < epochMax
		tmp = nan(epochMax, size(values{d}, 2), size(values{d}, 3));
		tmp(1:size(values{d}, 1), :, :) = values{d};
		values_equalEpochs{d} = tmp;
	else
		values_equalEpochs{d} = values{d};
	end

	% epochs x conditions x flies
	values_equalEpochs{d} = permute(values_equalEpochs{d}, [1 3 2]);

end

% Convert to one single matrix
values_equalEpochs = cat(3, values_equalEpochs{:});



% Number of flies per dataset
dset_nFlies = nan(size(dsets));
xtick_labels = cell(size(dset_nFlies));
for d = 1 : length(dsets)
	[nChannels, nFlies, nConditions, nEpochs] = getDimensionsFast(dsets{d});
	dset_nFlies(d) = nFlies;
	xtick_labels{d} = [num2str(nFlies) ' ' dsets{d} ' flies'];
end
xtick_pos = 1+cumsum(dset_nFlies);
xtick_pos = [1 xtick_pos(1:end-1)];

cond_offsets = linspace(-0.15, 0.15, size(values_equalEpochs, 2));
cond_colours = {'r', 'b'};
cond_colours_light = {[1 0.4 0.4], [0.4 0.4 1]};

a = -0.01; b = 0.01;
epoch_offsets = a + (b-a).*rand([size(values_equalEpochs, 1) size(values_equalEpochs, 3)]);

% Scatter plot of values
subplot(sp_scatter_ax);
%title(['ch' num2str(ch) newline 'fID ' num2str(fID) ' ' fName newline perf_type], 'Interpreter', 'none');

% Plot threshold
line([0 sum(dset_nFlies)+1], repmat(transform_values(thresholds(ch, fID), transform_type), [1 2]), 'Color', 'k');
hold on;

% Plot raw values
for c = 1 : size(values_equalEpochs, 2)
	% Plot raw values
	tmp = transform_values(permute(values_equalEpochs(:, c, :), [1 3 2]), transform_type); % epochs x flies
	scatter((1:sum(dset_nFlies))+cond_offsets(c)+epoch_offsets, tmp, 3, cond_colours{c}, 'o',...
		'filled',...
		'MarkerFaceAlpha', 0.3,...
		'MarkerEdgeAlpha', 0.3);

end

% Plot epoch meaned values
for c = 1 : size(values_equalEpochs, 2)
	tmp = mean(transform_values(values_equalEpochs(:, c, :), transform_type), 1, 'omitnan');
	tmp = permute(tmp, [3 1 2]);
	scatter((1:sum(dset_nFlies))+cond_offsets(c), tmp, 20, cond_colours_light{c}, 'o', 'LineWidth', 2);
	
end


set(gca, 'XTick', xtick_pos, 'XTickLabel', xtick_labels);
%xtickangle(350);
set(gca, 'TickDir', 'out');

axis tight
xlim([0 sum(dset_nFlies)+1]);
ylim_scatter = ylim; % for aligning the following histogram y-axis

box on

scatter_ax = gca;

% Histogram of values
subplot(sp_hist_ax);

for c = 1 : size(values_equalEpochs, 2)

	binWidth = (prctile(transform_values(values_equalEpochs, transform_type), 90, 'all') - prctile(transform_values(values_equalEpochs, transform_type), 10, 'all')) / 50;

	histogram(transform_values(values_equalEpochs(:, c, :), transform_type),...
		'BinWidth', binWidth,...
		'Orientation', 'horizontal',...
		'FaceColor', cond_colours{c},...
		'FaceAlpha', 0.5,...
		'EdgeAlpha', 0);
	hold on;

end

axis tight

% Plot threshold
line(xlim, repmat(transform_values(thresholds(ch, fID), transform_type), [1 2]), 'Color', 'k');

ylabel(transform_string);

ylim(ylim_scatter);

box on

hist_ax = gca;

end

%%

function [transformed] = transform_values(values, transform_type)

switch transform_type
	case 'log'
		% Values need to be >= 0 to take log
		%	So, if there are negatives, take the smallest value
		%		and add the absolute of it to make all values >= 0
		%	Then add 1 so all values >= 1
		transformed = log(values+abs(min(values))+1);
		% Take log again
		%transformed = log(transformed+1);

		%transformed = log(log(values+11.2)+1);
		transformed = log(log(values+12.2));
		transformed = log(log(values+2));
		
		% offset (negative) values by adding the lowest value + diff
		% between two lowest values
		%transformed = log(values+abs(min(values))+abs(diff(mink(values, 2))));

		%transformed = exp(abs(values));
		%transformed = values;
		%transformed = 1 ./ (values+abs(min(values))+1);
	case 'root'
		% For negative values - take absolute, then do root, then restore
		transformed = nthroot(abs(values), 2) .* sign(values);
		transformed = abs(values).^0.75 .* sign(values);
		transformed = log(values+1.31).^0.01;
	otherwise
		transformed = values;
end

end
