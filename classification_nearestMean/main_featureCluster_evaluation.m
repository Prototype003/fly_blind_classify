%% Description

%{

Cluster significant features based on similarity in feature values across
all epochs

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

%% Get performance stats

tmp = load(['results' preprocess_string filesep 'stats_multidoseSplit.mat']);

stats = tmp.stats;

%%

data_sets = {'train', 'multidose', 'singledose', 'sleep'};
dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
dset_mainPairs = [1 1 1 1 3];

%% Get raw values for each channel
% Note problem - all values are used (from all conditions)
%	Use main_featuresCluster_allCh_evaluation.m, which removes unused
%	conditions

fValues_all = cell(size(stats.train.valid_features, 1), 1);
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
stage = 'evaluate'; % train; evaluate; evaluateBatchNormalised

topN = 0;

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
for ch = 1 : size(sig_all, 1)
    fIds{ch} = find(valid_all(ch, :) & sig_all(ch, :));
    fValues{ch} = fValues_all{ch}(valid_rows{d}, fIds{ch});
end

% Take top N
if topN > 0
	
	dset_nFlies = nan(length(dsets), 1);
	for d = 1 : length(dsets)
		[~, nFlies] = getDimensionsFast(dsets{d});
		dset_nFlies(d) = nFlies;
	end
	
    for ch = 1 : size(sig_all, 1)
		
		% Average performance across evaluation datasets
		perfs_meaned = zeros(size(fIds{ch}));
		dstart = 1; % no batch normalised performance in discovery flies
		if strcmp(stage, 'evaluateBatchNormalised')
			dstart = 2; 
		end
		for d = dstart : length(dsets)
			perfs_meaned = perfs_meaned + dset_nFlies(d)*stats.(dsets{d}).(perf_type).performances{dset_mainPairs(d)}(ch, fIds{ch});
		end
		perfs_meaned = perfs_meaned ./ sum(dset_nFlies);
		
        [perfs_sorted, order] = sort(perfs_meaned, 'descend');
        
        if numel(fIds{ch}) > topN
            
            fIds{ch} = fIds{ch}(order);
            fIds{ch} = fIds{ch}(1:topN);
            
            fValues{ch} = fValues{ch}(:, order);
            fValues{ch} = fValues{ch}(:, 1:topN);
        end
    end
end

%% Generate dendrogram
% Note - clustering here is done on values which in the previous section
%	were averaged across epochs per fly

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
        distances_m = 1 - abs(fCorr); % higher correlation -> more similar -> less distance
        distances_m = squareform(distances_m); % convert to pdist vector form
        
        % Use (one minus) spearman correlation as distance
        distances_p = pdist(values', 'spearman'); % Can't deal with nan/inf?
        
        % Note - distances must be pdist vector (treats matrix as data instead of distances
        trees{ch} = linkage(distances_m, clusterDistance_method);
        distances{ch} = distances_m;
    end
    
    toc
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

ch = 1;

% Need to get outperm first
f = figure('visible', 'off'); % we want the order, not the actual plot
[h, T, outperm] = dendrogram(trees{ch}, size(trees{ch}, 1)+1);
close(f);

%% Get labels/colours by master operation

switch colour_type
	
	case 'limited' % Create colours based on selected features
		
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

	case 'full' % Use same colour scheme across all features and channels

		masters = hctsa{1}.Operations.MasterID;
		% Group master operation list by broad category
		%   Broad category determined by starting letters
		master_strings = hctsa{1}.MasterOperations.Code(masters);

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
		feature_labels = hctsa{1}.Operations.Name(fIds{ch});
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

subplot(subplot_rows, subplot_cols, [1 30]);

% Compare these numbers to unspecified labels to check proper order of
%   the labels
labels = num2cell((1:size(trees{ch}, 1)+1));
labels = cellfun(@num2str, labels, 'UniformOutput', false);

labels = groups; % master operations as labels
labels = feature_labels; % feature names as labels

% Add performances to labels (weighted average across datasets)
pSum = zeros(size(labels));
flySum = 0;
for d = 1 : length(dsets)
	[~, nFlies, ~, ~] = getDimensionsFast(dsets{d});
	if d == 1
		pSum = pSum + nFlies*stats.(dsets{d}).(perf_type).performances{dset_mainPairs(d)}(ch, fIds{ch})';
	else
		if strcmp(stage, 'evaluateBatchNormalised')
			pSum = pSum + nFlies*stats.([dsets{d} 'BatchNormalised']).(perf_type).performances{dset_mainPairs(d)}(ch, fIds{ch})';
		else
			pSum = pSum + nFlies*stats.(dsets{d}).(perf_type).performances{dset_mainPairs(d)}(ch, fIds{ch})';
		end
	end
	flySum = flySum + nFlies;
end
p = num2cell(pSum ./ flySum);
%p = num2cell(perfs.train.nearestMedian.performances{dset_mainPairs(1)}(ch, fIds{ch})); % show performance in a single dataset
l = max(cellfun(@length, labels)); % length of longest label                                                                                                                                                                                        
pad_length = cellfun(@(x,l) l-length(x), labels, repmat({l}, size(labels)), 'UniformOutput', false);
spaces = repmat({' '}, size(labels));
pad_spaces = cellfun(@(s,pl) repmat(s, [1 pl]), spaces, pad_length, 'UniformOutput', false);
labels = cellfun(@(x,y,z) sprintf('%s%s %2.0f%%',x,y,z*100), labels, pad_spaces, p(:), 'UniformOutput', false);

cluster_thresh = 'default'; % 0.75 or 'default'
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
%set(gca, 'XTickLength', [0 0]);

%% Print

print_fig = 0;

if print_fig == 1

	switch perf_type
		case 'nearestMedian'
			switch stage
				case 'evaluate'
					figure_name = '../figures_stage2/fig6_nearestMedian_dendrogram_raw';
				case 'evaluateBatchNormalised'
					figure_name = '../figures_stage2/fig6_BN_dendrogram_raw';
			end
		case 'consis'
			figure_name = ['../figures_stage2/consis_dendrogram_ch' num2str(ch) '_raw'];
	end

	set(gcf, 'PaperOrientation', 'Portrait');

	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG

end