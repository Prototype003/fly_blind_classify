%%

%{

Create figure showing distribution of hctsa values across all time-series

1 - select (best performing) feature
2 - scatter plot of epoch values (and average across epochs) per fly
	equal spacing per fly
3 - hctsa matrix limited to significant features (flies*2conds x  features)
4 - scatter plot of difference in epoch values (and average) per fly
	equal spacing per fly
5 - difference / DNV matrix limited to significant features, show features for all
	channels (flies x channels*features)

%}

%%

preprocess_string = '_subtractMean_removeLineNoise';

dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
dset_ids = (1:length(dsets));

%% Load stats
% Get all performances and stats using
%   get_stats_evaluation(preprocess_string)

stats_dir = ['classification_nearestMean/results' preprocess_string filesep];
stats_file = 'stats_multidoseSplit.mat';

% Should have variable stats
load([stats_dir stats_file]);

%% Load feature thresholds and directions

thresh_dir = ['classification_nearestMean/results' preprocess_string filesep];
thresh_file = 'class_nearestMedian_thresholds.mat';
load([thresh_dir thresh_file]);

%% Feature and channel to plot for
% Feature with highest average accuracy across evaluation flies

ch = 1;
fID = 5072;
perf_type = 'nearestMedian';
plot_diff = 0;

fName = hctsas{1}.Operations.Name{fID};

%%

ch = 1;
fID = 4676;
perf_type = 'nearestMedian';
plot_diff = 0;

fName = hctsas{1}.Operations.Name{fID};


%% Feature and channel to plot for
% Feature with highest average consistency across evaluation flies

ch = 3;
fID = 7701;
perf_type = 'consis';
plot_diff = 1;

fName = hctsas{1}.Operations.Name{fID};

%% Get values for the feature
% ~15 seconds

tic;
values = featureValueDistribution(fID, ch);
toc

values_all = values; % backup

%% Get specific condition pairs

dset_pairings = cell(size(dsets));
for d = 1 : length(dsets)
	[conds, cond_labels, cond_colours, stats_order, conds_main] =...
		getConditions(dsets{d});
	dset_pairings{d} = conds_main;
end

%% Extract values for specific condition pairs

for d = 1 : length(values)
	values{d} = values{d}(:, :, dset_pairings{d});
end

%% Convert cell array of values to a single matrix

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

%% Scatter

if plot_diff == 0

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

	% For transforming feature values to make distributions more visible
	trans_exp = 1; % power transform (odd numbers preserve pos/neg sign
	log_offset_1 = 12;
	log_offset_2 = 1;

	figure;

	cond_offsets = linspace(-0.15, 0.15, size(values_equalEpochs, 2));
	cond_colours = {'r', 'b'};

	a = -0.01; b = 0.01;
	epoch_offsets = a + (b-a).*rand([size(values_equalEpochs, 1) size(values_equalEpochs, 3)]);

	% Scatter plot of values
	subplot(1, 5, [1 4]);
	title(['ch' num2str(ch) newline 'fID ' num2str(fID) ' ' fName newline perf_type], 'Interpreter', 'none');

	% Plot threshold
	line([0 sum(dset_nFlies)+1], repmat(log(log(thresholds(ch, fID).^trans_exp+log_offset_1)+log_offset_2), [1 2]), 'Color', 'k');
	hold on;

	for c = 1 : size(values_equalEpochs, 2)

		% Plot raw values
		tmp = log(log(permute(values_equalEpochs(:, c, :), [1 3 2])+log_offset_1)+log_offset_2); % epochs x flies
		scatter((1:sum(dset_nFlies))+cond_offsets(c)+epoch_offsets, tmp.^trans_exp, 7.5, cond_colours{c}, 'o',...
			'filled',...
			'MarkerFaceAlpha', 0.5,...
			'MarkerEdgeAlpha', 0.5);

		% Plot epoch meaned values
		tmp = mean(values_equalEpochs(:, c, :), 1, 'omitnan');
		tmp = log(log(permute(tmp, [3 1 2])+log_offset_1)+log_offset_2);
		scatter((1:sum(dset_nFlies))+cond_offsets(c), tmp.^trans_exp, [], cond_colours{c}, 'o');

	end
	set(gca, 'XTick', xtick_pos, 'XTickLabel', xtick_labels);
	ylabel(['log(log(feature value)' '^{' num2str(trans_exp) '}+' num2str(log_offset_1) ')+' num2str(log_offset_2) ')']); % Modify based on actual transformation
	xtickangle(350);
	set(gca, 'TickDir', 'out');	

	axis tight
	xlim([0 sum(dset_nFlies)+1]);
	ylim_scatter = ylim; % for aligning the following histogram y-axis

	% Histogram of values
	subplot(1, 5, 5);

	for c = 1 : size(values_equalEpochs, 2)

		binWidth = (prctile(log(log(values_equalEpochs.^trans_exp+log_offset_1)+log_offset_2), 90, 'all') - prctile(log(log(values_equalEpochs.^trans_exp+log_offset_1)+log_offset_2), 10, 'all')) / 50;

		histogram(log(log(values_equalEpochs(:, c, :).^trans_exp+log_offset_1)+log_offset_2),...
			'BinWidth', binWidth,...
			'Orientation', 'horizontal',...
			'FaceColor', cond_colours{c},...
			'FaceAlpha', 0.5,...
			'EdgeAlpha', 0);
		hold on;

	end

	axis tight

	% Plot threshold
	line(xlim, repmat(log(log(thresholds(ch, fID).^trans_exp+log_offset_1)+log_offset_2), [1 2]), 'Color', 'k');

	ylim(ylim_scatter);

end

%% Scatter
% Use a function to transform values

if plot_diff == 0

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

	% For transforming feature values to make distributions more visible
	trans_exp = 1; % power transform (odd numbers preserve pos/neg sign
	log_offset_1 = 12;
	log_offset_2 = 1;

	figure;

	cond_offsets = linspace(-0.15, 0.15, size(values_equalEpochs, 2));
	cond_colours = {'r', 'b'};

	a = -0.01; b = 0.01;
	epoch_offsets = a + (b-a).*rand([size(values_equalEpochs, 1) size(values_equalEpochs, 3)]);

	% Scatter plot of values
	subplot(1, 5, [1 4]);
	title(['ch' num2str(ch) newline 'fID ' num2str(fID) ' ' fName newline perf_type], 'Interpreter', 'none');

	% Plot threshold
	line([0 sum(dset_nFlies)+1], repmat(transform_values(thresholds(ch, fID)), [1 2]), 'Color', 'k');
	hold on;

	for c = 1 : size(values_equalEpochs, 2)

		% Plot raw values
		tmp = transform_values(permute(values_equalEpochs(:, c, :), [1 3 2])); % epochs x flies
		scatter((1:sum(dset_nFlies))+cond_offsets(c)+epoch_offsets, tmp.^trans_exp, 7.5, cond_colours{c}, 'o',...
			'filled',...
			'MarkerFaceAlpha', 0.5,...
			'MarkerEdgeAlpha', 0.5);

		% Plot epoch meaned values
		tmp = mean(transform_values(values_equalEpochs(:, c, :)), 1, 'omitnan');
		tmp = permute(tmp, [3 1 2]);
		scatter((1:sum(dset_nFlies))+cond_offsets(c), tmp.^trans_exp, [], cond_colours{c}, 'o');

	end
	set(gca, 'XTick', xtick_pos, 'XTickLabel', xtick_labels);
	ylabel(['log(log(feature value)' '^{' num2str(trans_exp) '}+' num2str(log_offset_1) ')+' num2str(log_offset_2) ')']); % Modify based on actual transformation
	xtickangle(350);
	set(gca, 'TickDir', 'out');	

	axis tight
	xlim([0 sum(dset_nFlies)+1]);
	ylim_scatter = ylim; % for aligning the following histogram y-axis

	% Histogram of values
	subplot(1, 5, 5);

	for c = 1 : size(values_equalEpochs, 2)

		binWidth = (prctile(transform_values(values_equalEpochs), 90, 'all') - prctile(transform_values(values_equalEpochs), 10, 'all')) / 50;

		histogram(transform_values(values_equalEpochs(:, c, :)),...
			'BinWidth', binWidth,...
			'Orientation', 'horizontal',...
			'FaceColor', cond_colours{c},...
			'FaceAlpha', 0.5,...
			'EdgeAlpha', 0);
		hold on;

	end

	axis tight

	% Plot threshold
	line(xlim, repmat(transform_values(thresholds(ch, fID)), [1 2]), 'Color', 'k');

	ylim(ylim_scatter);

end

%% Scatter (difference)

% TODO - show all epoch-pair combinations
%	instead of 1-1, 2-2, 3-3, etc.

if plot_diff == 1
	
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

	% For transforming feature values to make distributions more visible
	trans_exp = 15; % power transform (odd numbers preserve pos/neg sign

	figure;

	a = -0.01; b = 0.01;
	epoch_offsets = a + (b-a).*rand([size(values_equalEpochs, 1) size(values_equalEpochs, 3)]);

	% Scatter plot of values
	subplot(1, 5, [1 4]);
	title(['ch' num2str(ch) newline 'fID ' num2str(fID) ' ' fName newline perf_type], 'Interpreter', 'none');
	
	% Plot 0 (no difference) line
	line([0 sum(dset_nFlies)+1], [0 0], 'Color', 'k');
	hold on;
	
	values_equalEpochs_diff = values_equalEpochs(:, 1, :).^trans_exp - values_equalEpochs(:, 2, :).^trans_exp;
	
	% Plot raw values
	tmp = permute(values_equalEpochs_diff, [1 3 2]); % epochs x flies
	scatter((1:sum(dset_nFlies))+epoch_offsets, tmp.^trans_exp, 7.5, 'b', 'o',...
		'filled',...
		'MarkerFaceAlpha', 0.5,...
		'MarkerEdgeAlpha', 0.5);

	% Plot epoch meaned values
	tmp = mean(values_equalEpochs_diff, 1, 'omitnan');
	tmp = permute(tmp, [3 1 2]);
	scatter((1:sum(dset_nFlies)), tmp.^trans_exp, [], 'b', 'o');

	xlim([0 sum(dset_nFlies)+1]);
	
	set(gca, 'XTick', xtick_pos, 'XTickLabel', xtick_labels);
	ylabel(['(feature value)' '^{' num2str(trans_exp) '}' ' wake-unawake']);
	xtickangle(350);
	set(gca, 'TickDir', 'out');	

	axis tight
	xlim([0 sum(dset_nFlies)+1]);
	ylim_scatter = ylim; % for aligning the following histogram y-axis

	% Histogram of values
	subplot(1, 5, 5);
	
	binWidth = (prctile(values_equalEpochs_diff, 90, 'all') - prctile(values_equalEpochs_diff, 10, 'all')) / 50;

	histogram(values_equalEpochs_diff,...
		'BinWidth', binWidth,...
		'Orientation', 'horizontal',...
		'FaceColor', 'b',...
		'FaceAlpha', 0.5,...
		'EdgeAlpha', 0);
	hold on;

	axis tight
	
	% Plot 0 (no difference) line
	line(xlim, [0 0], 'Color', 'k');

	ylim(ylim_scatter);
	
	
end

%%

function [transformed] = transform_values(values)
	
	transformed = log(log(values+12)+1);
	%transformed = exp(abs(values));
	transformed = values;

end