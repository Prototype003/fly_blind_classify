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

%% Feature and channel to plot for
% Feature with highest average consistency across evaluation flies

ch = 3;
fID = 7701;
perf_type = 'consis';
plot_diff = 1;

fName = hctsas{1}.Operations.Name{fID};

%% Get values for the feature
% Assumes the file only has differences for the desired condition pair

source_dir = 'figure_workspace/';

tic;
values = cell(length(dsets), 1);
for d = 1 : length(dsets)
	tic;
	
	source_file = ['dnvs_' dsets{d} '_ch' num2str(ch)];
	tmp = load([source_dir source_file '.mat']);
	
	values{d} = tmp.dnvs(:, fID);
	
	% Reshape to create dimension for flies
	% Assumes contiguous rows in tmp.dnvs are grouped by flies
	%	i.e. rows 1-n for fly 1, etc.
	[nChannels, nFlies, nConditions, nEpochs] = getDimensionsFast(dsets{d});
	
	values{d} = reshape(values{d}, [length(values{d})/nFlies nFlies]);
	
	toc
end

values_all = values; % backup

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

figure;

cond_offsets = linspace(-0.15, 0.15, size(values_equalEpochs, 2));
cond_colours = {'r', 'b'};

a = -0.01; b = 0.01;
epoch_offsets = a + (b-a).*rand([size(values_equalEpochs, 1) size(values_equalEpochs, 3)]);

% Scatter plot of values
subplot(1, 5, [1 4]);
title(['ch' num2str(ch) newline 'fID ' num2str(fID) ' ' fName newline perf_type], 'Interpreter', 'none');

% Plot 0 (no difference) line
line([0 sum(dset_nFlies)+1], [0 0], 'Color', 'k');
hold on;

% Plot raw values
tmp = permute(values_equalEpochs(:, 1, :), [1 3 2]); % epochs x flies
scatter((1:sum(dset_nFlies))+epoch_offsets, tmp.^trans_exp, 7.5, 'b', 'o',...
	'filled',...
	'MarkerFaceAlpha', 0.5,...
	'MarkerEdgeAlpha', 0.5);

% Plot epoch meaned values
tmp = mean(values_equalEpochs(:, 1, :), 1, 'omitnan');
tmp = permute(tmp, [3 1 2]);
scatter((1:sum(dset_nFlies)), tmp.^trans_exp, [], 'b', 'o');

set(gca, 'XTick', xtick_pos, 'XTickLabel', xtick_labels);
ylabel(['(DNV)' '^{' num2str(trans_exp) '}']);
xtickangle(350);
set(gca, 'TickDir', 'out');

axis tight
xlim([0 sum(dset_nFlies)+1]);
ylim_scatter = ylim; % for aligning the following histogram y-axis

% Histogram of values
subplot(1, 5, 5);

binWidth = (prctile(values_equalEpochs.^trans_exp, 90, 'all') - prctile(values_equalEpochs.^trans_exp, 10, 'all')) / 50;

histogram(values_equalEpochs(:, 1, :).^trans_exp,...
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

