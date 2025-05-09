%% Description

%{

Cluster significant features based on similarity in feature values across
all epochs

Takes topN features across all channels
	e.g. the top 50 features from across the 15 channels

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

%% Get performance stats

tmp = load(['results' preprocess_string filesep 'stats_multidoseSplit.mat']);

stats = tmp.stats;

%%

data_sets = {'train', 'multidose', 'singledose', 'sleep'};
dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
dset_labels = {'D', 'MD8', 'MD4', 'SD', 'S'};
dset_mainPairs = [1 1 1 1 3];

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
            keywords = {'conditionIsoflurane_0.6', 'conditionPost_Isoflurane', 'conditionRecovery'};
        case 'singledose' % exclude recovery condition
            keywords = {'conditionRecovery', 'conditionPostIsoflurane'};
        case 'sleep'
            keywords = {'conditionwakeEarly', 'conditionsleepLate'};
    end
    
    % Find corresponding rows
    match = zeros(size(hctsas{d}.TimeSeries, 1), 1);
    for kw = keywords
        match = match | getIds(kw, hctsas{d}.TimeSeries);
    end
    
    valid_rows{d} = ~match;
    
end

%% Get raw values for each channel

fValues_all = cell(size(stats.train.valid_features, 1), 1);

for ch = 1 : size(fValues_all, 1)
	tic;
	ds = cell(1);
	for d = 1 : length(dsets)
		ds{d} = hctsas{d}.TS_DataMat(valid_rows{d}, :, ch);
	end
	fValues_all{ch} = cat(1, ds{:});
	toc
end


%%

perf_type = 'nearestMedian'; % nearestMedian; consis
stage = 'evaluateBatchNormalised'; % train; evaluate; evaluateBatchNormalised

topN = 50;

%% Get significant features at each channel

switch stage
    case 'train'
        sig_all = stats.train.(perf_type).sig;
        valid_all = stats.train.valid_features;
	case 'evaluate' % sig and valid in train + validate1
		valid_all = stats.train.valid_features;
		sig_all = stats.train.(perf_type).sig;
		for d = 1 : length(dsets)
			valid_all = valid_all & stats.(dsets{d}).valid_features;
			sig_all = sig_all & stats.(dsets{d}).(perf_type).sig(:, :, dset_mainPairs(d));
		end
	case 'evaluateBatchNormalised'
		valid_all = stats.train.valid_features;
		sig_all = stats.train.(perf_type).sig;
		for d = 2 : length(dsets)
			valid_all = valid_all & stats.([dsets{d} 'BatchNormalised']).valid_features;
			sig_all = sig_all & stats.([dsets{d} 'BatchNormalised']).(perf_type).sig(:, :, dset_mainPairs(d));
		end
end

%% Filter features

% Keep features which are valid and sig. in all sets
fValues = cell(size(fValues_all));
fIds = cell(size(fValues));
chIds = cell(size(fValues));
chLabels = repmat((1:size(sig_all, 1))', [1 size(sig_all, 2)]); % for keeping track of channels
for ch = 1 : size(sig_all, 1)
    fIds{ch} = find(valid_all(ch, :) & sig_all(ch, :));
	fValues{ch} = fValues_all{ch}(:, fIds{ch});
	chIds{ch} = repmat(ch, [1 length(fIds{ch})]);
end

% Average performance across evaluation datasets
dset_nFlies = nan(length(dsets), 1);
for d = 1 : length(dsets)
	[~, nFlies] = getDimensionsFast(dsets{d});
	dset_nFlies(d) = nFlies;
end

fMeanPerfs = cell(size(fIds));
for ch = 1 : size(sig_all, 1)

	perfs_meaned = zeros(size(fIds{ch}));
	for d = 1 : length(dsets)
		perfs_meaned = perfs_meaned + dset_nFlies(d)*stats.(dsets{d}).(perf_type).performances{dset_mainPairs(d)}(ch, fIds{ch});
	end
	perfs_meaned = perfs_meaned ./ sum(dset_nFlies);
	
	fMeanPerfs{ch} = perfs_meaned;

end

% Collapse across channels
fIds = cat(2, fIds{:});
chIds = cat(2, chIds{:});
fMeanPerfs = cat(2, fMeanPerfs{:});

% Take top N across all the channels
%if topN > 0 && numel(fIds) >= topN
	
	% Sort features from all channels together
	[perfs_sorted, order] = sort(fMeanPerfs, 'descend');
	
	fIds = fIds(order);
	fMeanPerfs = fMeanPerfs(order);
	chIds = chIds(order);
	
	% Take top N across all the channels
	if topN > 0 && numel(fIds) >= topN
		fIds = fIds(1:topN);
		fMeanPerfs = fMeanPerfs(1:topN);
		chIds = chIds(1:topN);
	else
		fIds = fIds(1:end);
		fMeanPerfs = fMeanPerfs(1:end);
		chIds = chIds(1:end);
	end
	
	fValues = cell(length(fIds), 1);
	for f = 1 : length(fIds)
		fValues{f} = fValues_all{chIds(f)}(:, fIds(f));
	end
	fValues = cat(2, fValues{:});
	
%end

fIds = {fIds};
fValues = {fValues};
chIds = {chIds};
fMeanPerfs = {fMeanPerfs};

%% Generate dendrogram

clusterDistance_method = 'average';

trees = cell(size(fValues));
distances = cell(size(fValues));

tic;

if length(fValues{1}) > 1 % can't really cluster when there's only 1 feature

	% Use correlations among features as distance (manual)
	values = fValues{1};
	values(isinf(values)) = NaN; % Remove Infs for correlation
	fCorr = (corr(values, 'Type', 'Spearman', 'Rows', 'complete')); % Ignore NaNs
	%fCorr = abs(fCorr + fCorr.') / 2; % because corr output isn't symmetric for some reason (?)
	distances_m = 1 - abs(fCorr); % higher correlation -> more similar -> less distance
	distances_m = squareform(distances_m); % convert to pdist vector form

	% Use (one minus) spearman correlation as distance
	distances_p = pdist(values', 'spearman'); % Can't deal with nan/inf?

	% Note - distances must be pdist vector (treats matrix as data instead of distances
	trees{1} = linkage(distances_m, clusterDistance_method);
	distances{1} = distances_m;
end

toc

%% Create matrix of scaled values to visualise

% Select features + channels -> scale relative to discovery flies



% Scale values relative to discovery flies
% Get rows corresponding to discovery
ref_dset = 1;
reference_rows = zeros(size(fValues{1}, 1), 1);
reference_rows(1:size(hctsas{ref_dset}.TS_DataMat, 1)) = 1;
vis_mat_normalised = BF_NormalizeMatrix(fValues{1}, 'mixedSigmoid', reference_rows);

%%

% Average values across epochs, per fly
d_offset = 0; % for keeping track of the beginning of each dataset
vis_mats = cell(length(dsets), 2); % assumes only 2 conditions
for d = 1 : length(hctsas)
	
	dset = dsets{d};
	dbase = dset_bases{d};
	
	[nCh, nFlies, nConds, nEpochs] = getDimensionsFast(dbase);
	[conditions, cond_labels, cond_colours, stats_order, cond_idxs] = getConditions(dset);
	conditions = conditions(cond_idxs);
	cond_labels = cond_labels(cond_idxs);
	
	% Number of flies for multidose needs to be split
	if strcmp(dset, 'multidose8')
		flies = (1:8);
	elseif strcmp(dset, 'multidose4')
		flies = (9:12);
	else
		flies = (1:nFlies);
	end
	nFlies = length(flies);
	
	tmp = nan(nFlies, length(conditions), size(vis_mat_normalised, 2));
	
	for c = 1 : length(conditions)
		for f = 1 : length(flies) % average across epochs for the fly
			fly = flies(f);
			
			% Identify rows for the fly and condition
			keys = {['fly' num2str(fly)], conditions{c}};
            rows = find(getIds(keys, hctsas{d}.TimeSeries(valid_rows{d}, :)));
			
			% Average across epochs, for the fly
			tmp(f, c, :) = mean(vis_mat_normalised(rows+d_offset, :), 1);
			
		end
		
		vis_mats{d, c} = permute(tmp(:, c, :), [1 3 2]);
	end
	
	%vis_mats{d} = reshape(tmp, [nFlies*length(conditions) size(tmp, 3)]);
	
	d_offset = d_offset + (nFlies * length(conditions) * nEpochs);
	
end

% Join dsets together
%vis_mat = cat(1, vis_mats{:});

% Join dsets together, with same conditions grouped together
vis_mat = cat(1, vis_mats{:, 1}, vis_mats{:, 2});

% Get axis ticks and labels
ytickstrings = {};
ytickpos = [];
ycounter = 1;
for cond = 1 : 2% size(hctsas{1}.TS_NormalisedMean, 3)
	for d = 1 : length(hctsas)
		dset = dsets{d};
		dbase = dset_bases{d};

		[nCh, nFlies, nConds, nEpochs] = getDimensionsFast(dbase);
		[conditions, cond_labels, cond_colours, stats_order, cond_idxs] = getConditions(dset);
		conditions = conditions(cond_idxs);
		cond_labels = cond_labels(cond_idxs);

		if strcmp(dset, 'multidose8')
			flies = (1:8);
		elseif strcmp(dset, 'multidose4')
			flies = (9:12);
		else
			flies = (1:nFlies);
		end
		nFlies = length(flies);

		ytickpos = [ytickpos ycounter];
		ytickstrings = cat(2, ytickstrings, strcat(dset_labels{d}, '_', cond_labels(cond)));

		ycounter = ycounter + nFlies;
	end
end

%%

figure;
set(gcf, 'Color', 'w');
subplot_rows = 1;
subplot_cols = 100;

dend_orientation = 'left'; % orientation of dendrogram

% create colours based on selected features, or full feature list
%	'limited' = create colour scheme based on selected features
%	'full' = create colour scheme for all 7702 features
colour_type = 'full';

ch = 1; % This no longer corresponds to actual channels (they were collapsed across already)

% Need to get outperm first
f = figure('visible', 'off'); % we want the order, not the actual plot
[h, T, outperm] = dendrogram(trees{ch}, size(trees{ch}, 1)+1);
close(f);

%% Get labels/colours by master operation

switch colour_type
	
	case 'limited' % Create colours based on selected features
		
		masters = hctsas{1}.Operations.MasterID(fIds{ch});
		% Group master operation list by broad category
		%   Broad category determined by starting letters
		master_strings = hctsas{1}.MasterOperations.Code(masters);

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

	case 'full' % Use same colour scheme across all features and channels

		masters = hctsas{1}.Operations.MasterID;
		% Group master operation list by broad category
		%   Broad category determined by starting letters
		master_strings = hctsas{1}.MasterOperations.Code(masters);

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

		% cbrewer sometimes creates colours with values outside range [0 1]
		colour_scheme(colour_scheme>1) = 1;

		% Limit to only valid and significant features
		groups = groups(fIds{ch});
		feature_labels = hctsas{1}.Operations.Name(fIds{ch});
		colours = colours(fIds{ch});
end

%% Show theoretical colours

switch colour_type
	
	case 'limited' % use colours created from selected features
		
		subplot(subplot_rows, subplot_cols, [31 35]);

		colormap(colour_scheme);

		imagesc(flipud(colours(outperm)));

		set(gca, 'YTick', [], 'XTick', []);

		%c = colorbar;
		%c.Ticks = (1:length(colours)+1);

		box off
		
	case 'full' % use colours created from full feature list

		subplot(subplot_rows, subplot_cols, [31 85]);

		colormap(colour_scheme);

		image(permute(colour_scheme(flipud(colours(outperm)), :), [1 3 2]));

		set(gca, 'YTick', [], 'XTick', []);

		axis off
		
end

%% Custom colorbar

% 'full' - show colourbar for 7702 features
% 'limited' - show colourbar with only colours for selected features
cbar_type = 'limited';

switch cbar_type
	
	case 'limited' % show colour scheme for plotted features
		
		subplot(subplot_rows, subplot_cols, [91 92]);

		image(permute(colour_scheme(sort(unique(colours)), :), [1 3 2])); % Only feature categories which were plotted

		set(gca, 'YTick', (1:length(group_colours)), 'YTickLabel', unique(groups));
		set(gca, 'TickLabelInterpreter', 'none')
		set(gca, 'YAxisLocation', 'right');
		set(gca, 'YTickLabelRotation', 60);
		set(gca, 'TickDir', 'out');
		set(gca, 'XTick', []);
		
	case 'full' % show colour scheme across all features
		
		subplot(subplot_rows, subplot_cols, [38 39]);

		imagesc((group_colours)'); % Everything

		set(gca, 'YTick', (1:length(group_colours)), 'YTickLabel', groups_unique);
		set(gca, 'TickLabelInterpreter', 'none')
		set(gca, 'YAxisLocation', 'right');
		set(gca, 'YTickLabelRotation', 60);
		set(gca, 'TickDir', 'out');
		set(gca, 'XTick', []);
		
end

%% Plot dendrogram

subplot(subplot_rows, subplot_cols, [16 30]);

% Compare these numbers to unspecified labels to check proper order of
%   the labels
labels = num2cell((1:size(trees{ch}, 1)+1));
labels = cellfun(@num2str, labels, 'UniformOutput', false);

labels = groups; % master operations as labels
labels = feature_labels; % feature names as labels

%{
% Add performances to labels (weighted average across datasets)
pSum = zeros(size(labels));
flySum = 0;
for d = 2 : length(dsets)
	[~, nFlies, ~, ~] = getDimensionsFast(dsets{d});
	if d == 1
		pSum = pSum + nFlies*perfs.(dsets{d}).(perf_type).performances{dset_mainPairs(d)}(ch, fIds{ch})';
	else
		if strcmp(stage, 'evaluateBatchNormalised')
			pSum = pSum + nFlies*perfs.([dsets{d} 'BatchNormalised']).(perf_type).performances{dset_mainPairs(d)}(ch, fIds{ch})';
		else
			pSum = pSum + nFlies*perfs.(dsets{d}).(perf_type).performances{dset_mainPairs(d)}(ch, fIds{ch})';
		end
	end
	flySum = flySum + nFlies;
end
p = num2cell(pSum ./ flySum);
%p = num2cell(perfs.train.nearestMedian.performances{dset_mainPairs(1)}(ch, fIds{ch})); % show performance in a single dataset
%}

p = num2cell(fMeanPerfs{ch});
l = max(cellfun(@length, labels)); % length of longest label                                                                                                                                                                                        
pad_length = cellfun(@(x,l) l-length(x), labels, repmat({l}, size(labels)), 'UniformOutput', false);
spaces = repmat({' '}, size(labels));
pad_spaces = cellfun(@(s,pl) repmat(s, [1 pl]), spaces, pad_length, 'UniformOutput', false);
labels = cellfun(@(x,y,z,c) sprintf('%s%s %2.0f%% ch%d',x,y,z*100,c), labels, pad_spaces, p(:), num2cell(chIds{ch})', 'UniformOutput', false);

cluster_thresh = 0.7; % 0.7 or 'default'
[H, T, outperm] = dendrogram(trees{ch}, size(trees{ch}, 1)+1, 'Orientation', dend_orientation, 'ColorThreshold', cluster_thresh, 'Labels', labels);

if topN > 0
	title({perf_type, [stage ' ch' num2str(ch) ' top' num2str(topN)]});
else
	title({perf_type, [stage ' ch' num2str(ch) ' nSig=' num2str(length(labels))]})
end
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

xlim([0 1]);
set(gca, 'TickDir', 'out');
%set(gca, 'TickLength', [0 0]);

%% Plot hctsa values

subplot(subplot_rows, subplot_cols, [1 15]);

imagesc(vis_mat(:, outperm)');
%imagesc(vis_mat(1:62, outperm)' - vis_mat(63:end, outperm)')
c = colorbar;

colormap inferno;
set(gca, 'YDir', 'normal');

set(gca, 'TickLabelInterpreter', 'none');
xticks(ytickpos);
xticklabels(ytickstrings);
xlabel('fly');

%% Print

print_fig = 0;

if print_fig == 1

	switch perf_type
		case 'nearestMedian'
			switch stage
				case 'evaluate'
					figure_name = '../figures_stage2/nearestMedian_dendrogram_allCh_raw';
					%figure_name = '../figures_stage2/fig6_nearestMedian_dendrogram_raw';
				case 'evaluateBatchNormalised'
					figure_name = '../figures_stage2/fig6BNdendrogram_raw';
			end
		case 'consis'
			figure_name = '../figures_stage2/fig7dendrogram_raw';
	end

	set(gcf, 'PaperOrientation', 'Portrait');

	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG

end