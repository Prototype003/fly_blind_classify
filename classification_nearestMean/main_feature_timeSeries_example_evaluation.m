%% Description

%{

For selected feature(s), show example time-series, and distribution of
values

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

ch = 1;
fIDs = [5072 4676];
fNames = hctsas{1}.Operations.Name(fIDs);

% Select which fly and dataset to show from?
%	Or would it be better to show one time series from each fly?
%		Total 12+18+19 time-series? Or show just from one dataset?

%% Get example time series
% Get max and min values from each condition, for every fly

epochs = cell(size(dsets));
max_nMainConds = 0;

for d = 1 : length(dsets)

	[nChannels, nFlies, nConditions, nEpochs] = getDimensionsFast(dsets{d});
	[conditions, cond_labels, cond_colours, cond_statsOrder, conds_main] = getConditions(dsets{d});
	
	if length(conds_main) > max_nMainConds
		max_nMainConds = length(conds_main);
	end
	
	epochs{d}.minEpochs = nan(nFlies, length(conds_main), length(fIDs));
	epochs{d}.maxEpochs = epochs{d}.minEpochs;
	epochs{d}.minEpochVals = epochs{d}.minEpochs;
	epochs{d}.maxEpochVals = epochs{d}.minEpochs;
	epochs{d}.tMin = nan(length(hctsas{d}.TimeSeries.Data{1}), nFlies, length(conds_main), length(fIDs));
	epochs{d}.tMax = epochs{d}.tMin;
	
	for cond = 1 : length(conds_main)

		cond_rows = getIds(conditions(conds_main(cond)), hctsas{d}.TimeSeries);

		for fly = 1 : nFlies
			
			if strcmp(dsets{d}, 'multidose4') % correct for offset fly ids
				fly_rows = getIds({['fly' num2str(fly+8)]}, hctsas{d}.TimeSeries);
			else
				fly_rows = getIds({['fly' num2str(fly)]}, hctsas{d}.TimeSeries);
			end
			
			rows = find(cond_rows & fly_rows);
			
			% Find the epoch with the lowest, and highest feature value for the
			% selected features
			for f = 1 : length(fIDs)
				
				fVals = hctsas{d}.TS_DataMat(rows, fIDs(f), ch);
				
				[fMin, rowMin] = min(fVals);
				[fMax, rowMax] = max(fVals);
				
				% Assumes that rows are already sorted by epoch
				epochs{d}.minEpochs(fly, cond, f) = rowMin;
				epochs{d}.maxEpochs(fly, cond, f) = rowMax;
				epochs{d}.minEpochVals(fly, cond, f) = fMin;
				epochs{d}.maxEpochVals(fly, cond, f) = fMax;
				
				% Just get the row number directly
				rowMin = rows(rowMin);
				rowMax = rows(rowMax);
				
				% Get the associated timeseries
				epochs{d}.tMin(:, fly, cond, f) = hctsas{d}.TimeSeries.Data{rowMin};
				epochs{d}.tMax(:, fly, cond, f) = hctsas{d}.TimeSeries.Data{rowMax};
				
			end

		end

	end

end

%% Show time-series, for each fly, grouped by condition
% Image plot

sp_rows = length(dsets);
sp_cols = max_nMainConds;

for f = 1 : length(fIDs)
	
	figure;
	
	for d = 1 : length(dsets)
		
		[nChannels, nFlies, nConditions, nEpochs] = getDimensionsFast(dsets{d});
		[conditions, cond_labels, cond_colours, cond_statsOrder, conds_main] = getConditions(dsets{d});
		
		for cond = 1 : max_nMainConds
			
			if cond <= length(conds_main)
				
				% Note - the first time-series is plotted at the smallest
				% y-position
				series = cat(2, epochs{d}.tMin(:, :, cond, f), epochs{d}.tMax(:, :, cond, f));
				
				% If sp_rows is number of conditions
				%subplot(sp_rows, sp_cols, d+((cond-1)*length(dsets)));
				
				% If sp_rows is number of datasets
				subplot(sp_rows, sp_cols, cond+((d-1)*max_nMainConds));
				
				imagesc(series'); colorbar;
			end
			title([dsets{d} ' cond' num2str(cond)]);
			xlabel('t');
			ylabel('<- fly^{minval}    fly^{maxval} ->');
		end
		
	end
end

%% Show time-series, for each fly, grouped by condition
% Line plot

for f = 1 : length(fIDs)
	
	figure;
	
	nDsets = length(dsets);
	sp_rows = max_nMainConds;
	sp_cols = nDsets;
	
	for d = 1 : nDsets
		
		[nChannels, nFlies, nConditions, nEpochs] = getDimensionsFast(dsets{d});
		[conditions, cond_labels, cond_colours, cond_statsOrder, conds_main] = getConditions(dsets{d});
		
		for cond = 1 : max_nMainConds
			
			if cond <= length(conds_main)
				series = cat(2, epochs{d}.tMin(:, :, cond, f), epochs{d}.tMax(:, :, cond, f));
				subplot(sp_rows, sp_cols, d+((cond-1)*nDsets));
				offset = 0;
				for t = 1 : size(series, 2)
					
					plot(series(:, t)+offset);
					hold on
					offset = offset + 300;
				end

			end
			title([dsets{d} ' cond' num2str(cond)]);
			xlabel('t');
			ylabel('<- fly^{minval}    fly^{maxval} ->');
		end
		
	end
end

%% Show histogram of values, grouped by condition
for f = 1 : length(fIDs)
	
	figure;
	
	nDsets = length(dsets);
	sp_rows = max_nMainConds;
	sp_cols = nDsets;
	
	for d = 1 : nDsets
		
		[nChannels, nFlies, nConditions, nEpochs] = getDimensionsFast(dsets{d});
		[conditions, cond_labels, cond_colours, cond_statsOrder, conds_main] = getConditions(dsets{d});
		
		for cond = 1 : max_nMainConds
			
			if cond <= length(conds_main)
				subplot(sp_rows, sp_cols, d+((cond-1)*nDsets));
			end
			
			vals = cat(2, epochs{d}.minEpochVals(:, cond, f), epochs{d}.maxEpochVals(:, cond, f));
			scatter(ones(numel(vals(:, 1)), 1), vals(:, 1));
			hold on;
			scatter(ones(numel(vals(:, 1)), 1)+1, vals(:, 1));
			
			hold off
			histogram(vals(:, 1) - vals(:, 2));
			
		end
		
	end
end

%% Show the time-series pair with the largest difference in value
% Get time-series with greatest value across flies, and lowest value
% across flies, between conditions

% Get time-series pairs:
%	1) highest in cond1, lowest in cond2
%	2) lowest in cond1, highest in cond2
% Pick the appropriate pair based on trained direction

% To do this, assume:
%	1) Number of (main) conditions is same across datasets
%	2) First condition is wake, second is unawake

% Combine datasets together
epochs_all = struct();
for d = 1 : length(dsets)

	if d == 1
		
		tmp = fieldnames(epochs{d});
		for field_counter = 1 : length(tmp)
			field = tmp{field_counter};
			epochs_all.(field) = epochs{d}.(field);
		end
		
	else % d > 1
		% Concatenate on fly dimension
		
		for field_counter = 1 : 4 % first 4 fields, fly dim is dim 1
			field = tmp{field_counter};
			epochs_all.(field) = cat(1, epochs_all.(field), epochs{d}.(field));
		end
		for field_counter = 5 : 6 % for these fields, fly dim is dim 2
			field = tmp{field_counter};
			epochs_all.(field) = cat(2, epochs_all.(field), epochs{d}.(field));
		end
		
	end
	
end

%% Show time-series examples which have the greatest/smallest values

% Find the pair of time-series with the greatest difference in feature
% value
nTop = 3; % number of greatest/smallest examples to show
series = nan(size(epochs_all.tMin, 1), 2, length(fIDs));
seriesMaxFlies = nan(nTop, length(fIDs));
seriesMinFlies = seriesMaxFlies;
seriesMaxVals = seriesMaxFlies;
seriesMinVals = seriesMinFlies;
for f = 1 : length(fIDs)
	
	% direction==1 means wake > unawake
	if directions(ch, fIDs(f)) == 1
		
		% Get largest values in wake
		[sortedDescend, order] = sort(epochs_all.maxEpochVals(:, 1, f), 'descend');
		series(:, 1:nTop, f) = epochs_all.tMax(:, order(1:nTop), 1, f);
		seriesMaxFlies(:, f) = order(1:nTop);
		seriesMaxVals(:, f) = sortedDescend(1:nTop);
		
		% Get smallest values in unawake
		[sortedAscend, order] = sort(epochs_all.maxEpochVals(:, 2, f), 'ascend');
		series(:, (1:nTop)+nTop, f) = epochs_all.tMax(:, order(1:nTop), 2, f);
		seriesMinFlies(:, f) = order(1:nTop);
		seriesMinVals(:, f) = sortedAscend(1:nTop);
		
		colours = cat(1, repmat({'r'}, [nTop, 1]), repmat({'b'}, [nTop 1])); % wake first
		
	else % vice versa
		
		% Get largest values in unawake
		[sortedDescend, order] = sort(epochs_all.maxEpochVals(:, 2, f), 'descend');
		series(:, 1:nTop, f) = epochs_all.tMax(:, order(1:nTop), 2, f);
		seriesMaxFlies(:, f) = order(1:nTop);
		seriesMaxVals(:, f) = sortedDescend(1:nTop);
		
		% Get smallest values in awake
		[sortedAscend, order] = sort(epochs_all.minEpochVals(:, 1, f), 'ascend');
		series(:, (1:nTop)+nTop, f) = epochs_all.tMin(:, order(1:nTop), 1, f);
		seriesMinFlies(:, f) = order(1:nTop);
		seriesMinVals(:, f) = sortedAscend(1:nTop);
		
		colours = cat(1, repmat({'r'}, [nTop, 1]), repmat({'b'}, [nTop 1])); % unawake first
		
	end
	
end

% Plot
% Show time-series with nTop highest values (highest value on top)
% Then show time-series with nTop lowest values (lowest value on bottom)
figure;

% Reverse the order of time-series with min values
series(:, nTop+(1:nTop), :) = series(:, nTop+(nTop:-1:1), :);

offset_size = -300;
for f = 1 : length(fIDs)
	
	subplot(1, length(fIDs), f);
	offset = 0;

	for t = 1 : size(series, 2)
		p(t) = plot(series(:, t, f)+offset, 'Color', colours{t});
		hold on
		offset = offset + offset_size;
	end
	
	axis tight
	
	title(['f' num2str(fIDs(f)) ' ' fNames{f}], 'Interpreter', 'none'); % Show feature name
	
	set(gca, 'YTick', fliplr((0:offset_size:(size(series, 2)-1)*offset_size)));
	
	% Note - smallest y-pos label needs to come first
	%	For seriesMinFlies, the first fly has the smallest value
	%	For seriesMaxFlies, the last fly has the smallest value - so flip
	set(gca, 'YTickLabel', cat(1, seriesMinFlies(:, f), flipud(seriesMaxFlies(:, f))));
	
	ylabel('fly');
	xlabel('t (ms)');
	
	legend([p(1) p(1+nTop)], {'consc.', 'unconsc.'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
	
	% Show feature values
	tmp = ylim;
	yyaxis right;
	ylim(tmp);
	set(gca, 'YTick', fliplr((0:offset_size:(size(series, 2)-1)*offset_size)));
	set(gca, 'YTickLabel', cat(1, seriesMinVals(:, f), flipud(seriesMaxVals(:, f))));
	y = ylabel('feature value');
	set(y, 'Rotation', -90);
	
end

%% Get all values

values = cell(size(fIDs));
labels = cell(size(fIDs));
for f = 1 : length(fIDs)
	
	tic;
	[values{f}, labels{f}] = featureValueDistribution(fIDs(f), ch);
	toc
	
	% Get only values from the main condition pair
	for d = 1 : length(dsets)
		values{f}{d} = values{f}{d}(:, :, dset_pairings{d});
		labels{f}{d} = labels{f}{d}(:, :, dset_pairings{d});
	end
	
end

%% Place values in single matrix for each feature
% Pad epochs with nans (MD flies have more epochs than other flies)

fVals = cell(length(fIDs), 1);
fVal_labels = cell(size(fVals)); % Note - should be the same across each f
fVal_dLabelStrings = cell(size(fVals)); % Note - should be the same across each f
fVal_dsets = cell(size(fVals)); % Note - should be the same across each f
for f = 1 : length(fIDs)
	
	% Convert cell array of values to a single matrix
	
	% Equalise epoch dimension by adding nans
	epochMax = max(cellfun(@(x) max(size(x)), values{f}));
	values_equalEpochs = cell(size(values{f}));
	labels_equalEpochs = cell(size(values{f}));
	dLabelStrings_equalEpochs = cell(size(values{f}));
	dsets_equalEpochs = cell(size(values{f}));
	for d = 1 : length(dsets)
		
		if size(values{f}{d}, 1) < epochMax
			
			% Values
			tmp = nan(epochMax, size(values{f}{d}, 2), size(values{f}{d}, 3));
			tmp(1:size(values{f}{d}, 1), :, :) = values{f}{d};
			values_equalEpochs{d} = tmp;
			
			% Labels
			tmp = cell(epochMax, size(values{f}{d}, 2), size(values{f}{d}, 3));
			tmp(1:size(values{f}{d}, 1), :, :) = labels{f}{d};
			tmp(size(values{f}{d}, 1)+1:end, :, :) = {'paddedEpoch'};
			labels_equalEpochs{d} = tmp;
			
			% dset label strings
			tmp = cell(epochMax, size(values{f}{d}, 2), size(values{f}{d}, 3));
			tmp(1:size(values{f}{d}, 1), :, :) = dsets(d);
			tmp(size(values{f}{d}, 1):end, :, :) = dsets(d);
			dLabelStrings_equalEpochs{d} = tmp;
			
			% dset number
			tmp = nan(epochMax, size(values{f}{d}, 2), size(values{f}{d}, 3));
			tmp(1:size(values{f}{d}, 1), :, :) = d;
			tmp(size(values{f}{d}, 1):end, :, :) = d;
			dsets_equalEpochs{d} = tmp;
			
		else
			values_equalEpochs{d} = values{f}{d};
			labels_equalEpochs{d} = labels{f}{d};
			
			tmp = cell(epochMax, size(values{f}{d}, 2), size(values{f}{d}, 3));
			tmp(:) = dsets(d);
			dLabelStrings_equalEpochs{d} = tmp;
			
			tmp = nan(epochMax, size(values{f}{d}, 2), size(values{f}{d}, 3));
			tmp(:) = d;
			dsets_equalEpochs{d} = tmp;
		end
		
		% epochs x conditions x flies
		values_equalEpochs{d} = permute(values_equalEpochs{d}, [1 3 2]);
		labels_equalEpochs{d} = permute(labels_equalEpochs{d}, [1 3 2]);
		dLabelStrings_equalEpochs{d} = permute(dLabelStrings_equalEpochs{d}, [1 3 2]);
		dsets_equalEpochs{d} = permute(dsets_equalEpochs{d}, [1 3 2]);
		
	end
	
	% Convert to one single matrix
	values_equalEpochs = cat(3, values_equalEpochs{:});
	labels_equalEpochs = cat(3, labels_equalEpochs{:});
	dLabelStrings_equalEpochs = cat(3, dLabelStrings_equalEpochs{:});
	dsets_equalEpochs = cat(3, dsets_equalEpochs{:});
	
	% Combine flies and epochs into one dimension
	values_equalEpochs = permute(values_equalEpochs, [1 3 2]);
	dims = size(values_equalEpochs);
	values_equalEpochs = reshape(values_equalEpochs, [dims(1)*dims(2) dims(3)]);
	
	labels_equalEpochs = permute(labels_equalEpochs, [1 3 2]);
	dims = size(labels_equalEpochs);
	labels_equalEpochs = reshape(labels_equalEpochs, [dims(1)*dims(2) dims(3)]);
	
	dLabelStrings_equalEpochs = permute(dLabelStrings_equalEpochs, [1 3 2]);
	dims = size(dLabelStrings_equalEpochs);
	dLabelStrings_equalEpochs = reshape(dLabelStrings_equalEpochs, [dims(1)*dims(2) dims(3)]);
	
	dsets_equalEpochs = permute(dsets_equalEpochs, [1 3 2]);
	dims = size(dsets_equalEpochs);
	dsets_equalEpochs = reshape(dsets_equalEpochs, [dims(1)*dims(2) dims(3)]);
	
	% Remove padded epochs (we now have the labels for each epoch)
	pad_epoch = isnan(values_equalEpochs(:, 1)); % Note - should be same for both conditions
	values_equalEpochs(pad_epoch, :) = [];
	labels_equalEpochs(pad_epoch, :) = [];
	dLabelStrings_equalEpochs(pad_epoch, :) = [];
	dsets_equalEpochs(pad_epoch, :) = [];
	
	fVals{f} = values_equalEpochs;
	fVal_labels{f} = labels_equalEpochs;
	fVal_dLabelStrings{f} = dLabelStrings_equalEpochs;
	fVal_dsets{f} = dsets_equalEpochs;
	
end

%% Find examples of series with high values in one feature but low values in the other (and vice versa)
% Assign rank (1-100), and use average rank
% Find examples from each condition
% Assume that fIDs only has two features
% Assume there are only two main conditions

% 2 x 2 x conds x N
%	F1 low/high
%	F2 low/high
%	Example for each condition
%	Number of examples

nTop = 1; % number of greatest/smallest examples to show

% Note - when sort 'ascend', smallest element has smallest rank
%	when sort 'descend', largest element has smallest rank
sort_types = {'ascend', 'descend'};
series_vals = nan(2, 2, max_nMainConds, length(fIDs), nTop);
series_ids = nan(2, 2, max_nMainConds, nTop);
for f1 = 1 : 2 % feature 1 low then high
	
	[f1_sorted, f1_order] = sort(fVals{1}, 1, sort_types{f1}, 'MissingPlacement', 'last');
	f1_rank = repmat((1:size(f1_order, 1))', [1 size(f1_order, 2)]);
	for cond = 1 : max_nMainConds
		f1_rank(f1_order(:, cond), cond) = f1_rank(:, cond);
	end
	f1_rank(isnan(f1_rank)) = nan;
	
	for f2 = 1 : 2 % feature 2 low then high
		
		[f2_sorted, f2_order] = sort(fVals{2}, 1, sort_types{f2}, 'MissingPlacement', 'last');
		f2_rank = repmat((1:size(f2_order, 1))', [1 size(f2_order, 2)]);
		for cond = 1 : max_nMainConds
			f2_rank(f2_order(:, cond), cond) = f2_rank(:, cond);
		end
		f2_rank(isnan(f2_rank)) = nan;
		
		% Average the rankings of each feature
		rank_mean = (f1_rank + f2_rank);
		
		% Find the lowest average rank
		[rank_mean_sorted, rank_mean_order] = sort(rank_mean, 1, 'ascend', 'MissingPlacement', 'last');
		
		for cond = 1 : max_nMainConds
			
			series_ids(f1, f2, cond, :) = rank_mean_order(1:nTop, cond);
			
			for f = 1 : length(fIDs)
				series_vals(f1, f2, cond, f, :) = fVals{f}(series_ids(f1, f2, cond, :), cond);
			end
			
		end
		
	end
	
end

%% Scatter plot of feature 1 against feature 2

figure;

cond_colours = {'r', 'b'}; % red for wake, blue for unawake
cond_colours_dark = {[0.6 0 0], [0 0 0.6]};
cond_legend = {'wake', 'unawake'};
s = [];

for cond = 1 : max_nMainConds
	
	hold on;
	
	xvals = transform_values(fVals{1}(:, cond));
	yvals = (fVals{2}(:, cond));
	
	s(cond) = scatter(xvals, yvals,...
		5, cond_colours{cond}, 'o', 'filled',...
		'MarkerFaceAlpha', 0.5,...
		'MarkerEdgeAlpha', 0.5);
	
	%xscale('log'); % Note - this will drop values which become infinity (log(0)) = inf
	
end

% Show the position of of selected time-series
circle_labels = {'  LL', '  LH'; '  HL', '  HH'};
for f1 = 1 : 2
	for f2 = 1 : 2
		for cond = 1 : max_nMainConds
			
			ids = series_ids(f1, f2, cond, :);
			
			xvals = transform_values(fVals{1}(ids, cond));
			yvals = fVals{2}(ids, cond);
			
			scatter(xvals, yvals,...
				50, cond_colours_dark{cond}, 'o',...
				'LineWidth', 1,...
				'MarkerFaceAlpha', 1,...
				'MarkerEdgeAlpha', 1);
			
			text(xvals, yvals, circle_labels{f1, f2})
		end
	end
end

legend(s, cond_legend, 'Location', 'southeast');

xlabel(['log(f1+12.2): ' fNames{1}], 'interpreter', 'none');
ylabel(['f2: ' fNames{2}], 'interpreter', 'none');

%% Plot time series

figure;

cond_colours = {'r', 'b'}; % red for wake, blue for unawake
offset_size = -300;

sp_counter = 1;

% Note order of plotting f1-f2
%	low-low -> low-high -> high-low -> high-high
%	bot-left -> top-left -> bot-right -> top-right
sp_pos = [3 1 4 2];
sp_labels = {'f1-low f2-low', 'f1-low f2-high', 'f1-high f2-low', 'f1-high f2-high'};

for f1 = 1 : 2
	
	for f2 = 1 : 2
		
		subplot(2, 2, sp_pos(sp_counter));
		hold on;
		offset = 0;
		
		for cond = 1 : max_nMainConds
			
			for t = 1 : nTop
				
				% Get time series label
				labelString = fVal_labels{1}{series_ids(f1, f2, cond, t), cond};
				d = fVal_dsets{1}(series_ids(f1, f2, cond, t), cond);
				
				% Find time series with matching label
				rowID = find(cellfun(@(x) strcmp(x, labelString), hctsas{d}.TimeSeries.Name));
				series = hctsas{d}.TimeSeries.Data{rowID};
				
				% Plot
				plot(series+offset, cond_colours{cond});
				
				offset = offset + offset_size;
				
			end
			
		end
		
		axis tight
		
		title([circle_labels{f1, f2} ': ' sp_labels{sp_counter}]);
		
		set(gca, 'YTick', fliplr((0:offset_size:(length(fIDs)-1)*offset_size)));
		set(gca, 'YTickLabel', flipud(squeeze(transform_values(series_vals(f1, f2, :, 1)))));
		ylabel(['log(f1+12.2)']);
		
		tmp = ylim;
		yyaxis right;
		ylim(tmp);
		set(gca, 'YTick', fliplr((0:offset_size:(length(fIDs)-1)*offset_size)));
		set(gca, 'YTickLabel', flipud(squeeze(transform_values(series_vals(f1, f2, :, 2)))));
		y = ylabel(['f2']);
		set(y, 'Rotation', -90);
		
		set(gca, 'YColor', 'k');
		
		xlabel('t (ms)');
		
		sp_counter = sp_counter + 1;
		
	end
	
end

%%

function [transformed] = transform_values(values)
	
	% Values need to be >= 0 to take log
	%	So, if there are negatives, take the smallest value
	%		and add the absolute of it to make all values >= 0
	%	Then add 1 so all values >= 1
	transformed = log(values+abs(min(values))+1);
	% Take log again
	%transformed = log(transformed+1);
	
	%transformed = log(log(values+11.2)+1);
	transformed = log(log(values+12.2));
	
	%transformed = exp(abs(values));
	%transformed = values;
	%transformed = 1 ./ (values+abs(min(values))+1);
end
