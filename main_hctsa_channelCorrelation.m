%% Description

%{

Check correlations in feature values among channels for each dataset

%}

%%

preprocess_string = '_subtractMean_removeLineNoise';

dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
dset_ids = (1:length(dsets));

%% Load stats
% Stats include feature validity
% Get all performances and stats using
%   get_stats_evaluation(preprocess_string)

stats_dir = ['classification_nearestMean/results' preprocess_string filesep];
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

source_dir = ['hctsa_space' preprocess_string '/'];

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

	% Remove extra recovery condition in singledose
	if strcmp(data_sources{d}, 'singledose')
		remove_rows = getIds({'conditionRecovery'}, hctsas{d}.TimeSeries);
		hctsas{d}.TimeSeries(remove_rows, :) = [];
		hctsas{d}.TS_DataMat(remove_rows, :, :) = [];
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

%% Set load or compute correlations

load_corrs = 1;

source_dir = 'channel_correlation';
source_file = 'corrs.mat';
if load_corrs == 1
	load([source_dir filesep source_file]);
	compute_corrs = 0;
else
	compute_corrs = 1;
end

%% Compute correlations in values between channels
% Compute correlations per fly? Or across all flies together?
% Target figure
%   Correlation matrix (using all flies together) per dataset

if compute_corrs == 1
	% Correlate all flies and epochs together
	corrs_pearson = cell(size(hctsas));
	corrs_spearman = cell(size(corrs_pearson));
	for d = 1 : length(hctsas)
		tic;

		tmp = hctsas{d}.TS_DataMat;

		dims = size(tmp);
		tmp = reshape(tmp, [dims(1)*dims(2) dims(3)]);

		corrs_spearman{d} = corr(tmp, 'Type', 'Spearman', 'rows', 'pairwise');

		% replace infs with nan if using Pearson
		tmp(isinf(tmp)) = nan;
		corrs_pearson{d} = corr(tmp, 'Type', 'Pearson', 'rows', 'pairwise');

		toc
		pause(0.01);
	end
end

%%

figure;
for d = 1 : length(hctsas)
	subplot(1, length(hctsas), d);
	imagesc(corrs_spearman{d});
	%imagesc(corrs_pearson{d});
	c = colorbar;
	title(c, 'r');
	title(dsets{d})
	axis square
end

%% Compute correlations in values between channels
% Compute correlations per fly? Or across all flies together?
% Target figure
%   Correlation matrix per fly

if compute_corrs == 1
	%corrs_perFly_pearson = cell(size(hctsas));
	%corrs_perFly_spearman = cell(size(hctsas));
	for d = [1 2 4 5]% : length(hctsas)
		
		% Get number of individual flies
		if strcmp(dsets{d}, 'multidose8')
			[nCh, nFlies, nConditions, nEpochs] = getDimensionsFast('multidose');
			flies = (1:8);
		elseif strcmp(dsets{d}, 'multidose4')
			[nCh, nFlies, nConditions, nEpochs] = getDimensionsFast('multidose');
			flies = (9:12);
		else
			[nCh, nFlies, nConditions, nEpochs] = getDimensionsFast(dsets{d});
			flies = (1:nFlies);
		end

		corrs_perFly_pearson{d} = nan(nCh, nCh, length(flies));

		for fly = flies
			tic;
			% Get hctsa values associated with the fly_rows = 1;
			fRows = find(getIds({['fly' num2str(fly)]}, hctsas{d}.TimeSeries));
			tmp = hctsas{d}.TS_DataMat(fRows, :, :);

			dims = size(tmp);
			tmp = reshape(tmp, [dims(1)*dims(2) dims(3)]);

			% Compute correlation among channels
			corrs_perFly_spearman{d}(:, :, fly) = corr(tmp, 'Type', 'Spearman', 'rows', 'pairwise');

			% Replace infs with nan if using Pearson
			tmp(isinf(tmp)) = nan;
			corrs_perFly_pearson{d}(:, :, fly) = corr(tmp, 'Type', 'Pearson', 'rows', 'pairwise');
			toc
			pause(0.01);
		end
	end

end

%% Remove fake flies in MD4 set

corrs_perFly_spearman{d} = corrs_perFly_spearman{d}(:, :, 9:end);
corrs_perFly_pearson{d} = corrs_perFly_pearson{d}(:, :, 9:end);

%% Plot correlation matrices per fly
%{
for d = 1 : length(hctsas)
	figure;
	nFlies = size(corrs_perFly_spearman{d}, 3);
	for fly = 1 : nFlies
		subplot(ceil(sqrt(nFlies)), ceil(sqrt(nFlies)), fly);
		imagesc(corrs_perFly_spearman{d}(:, :, fly));
		c = colorbar;
		title(c, 'r');
		title([dsets{d} ' fly' num2str(fly)]);
		xlabel('channel');
		ylabel('channel');
		axis square
	end
end
%}

%% Plot scatter of ch-X vs ch-Y for a fly
%{
d = 1;
fly = 1;
fRows = find(getIds({['fly' num2str(fly)]}, hctsas{d}.TimeSeries));
tmp = hctsas{d}.TS_DataMat(fRows, :, :);
dims = size(tmp);
tmp = reshape(tmp, [dims(1)*dims(2) dims(3)]);
nCh = dims(3);

figure;
sp_counter = 1;
for chX = 1 : size(tmp, 2)

	x = tiedrank(tmp(:, chX));

	for chY = chX+1 : size(tmp, 2)

		y = tiedrank(tmp(:, chY));
		
		subplot(floor(sqrt((nCh^2)/2)), ceil(sqrt((nCh^2)/2)), sp_counter);

		scatter(x, y, 0.01, '.', 'MarkerEdgeAlpha', 0.1, 'MarkerEdgeAlpha', 0.1);
		title(['ch' num2str(chX) '-ch' num2str(chY)]);
		axis tight

		sp_counter = sp_counter + 1;

	end
end
%}

%% Check Check distribution of rankings
% Is there some rank with a lot of ties which is causing the
%	weird structure in the scatter plot of ranks?

d = 5;
fly = 1;
fRows = find(getIds({['fly' num2str(fly)]}, hctsas{d}.TimeSeries));
tmp = hctsas{d}.TS_DataMat(fRows, :, :);
dims = size(tmp);
tmp = reshape(tmp, [dims(1)*dims(2) dims(3)]);
nCh = dims(3);

figure;
sp_counter = 1;
for chX = 1 : size(tmp, 2)
	
	x = tiedrank(tmp(:, chX));
	
	subplot(ceil(sqrt(nCh)), ceil(sqrt(nCh)), sp_counter);
	histogram(x, (0:1:numel(x)));
	title([dsets{d} ' fly' num2str(fly) ' ch' num2str(chX)]);
	xlabel('rank (bin size 1)');
	xlim tight
	ylabel('count');
	ylim([0 2000]);

	sp_counter = sp_counter + 1;
end

%% Average correlation matrix across all flies

corrMean = zeros(nCh, nCh);
fly_count = 0;
for d = 1 : length(corrs_perFly_spearman)
	tmp = corrs_perFly_spearman{d};
	for fly = 1 : size(tmp, 3)
		corrMean = corrMean + tmp(:, :, fly);
		fly_count = fly_count + 1;
	end
end
corrMean = corrMean ./ fly_count;

figure;
imagesc(corrMean);
c = colorbar;
axis square
title('mean r (all flies)');
title(c, 'r');
xlabel('channel');
ylabel('channel');

%% Distribution of correlation values across all flies

val_idxs = triu(true(nCh), 1);
vals_perFly = cell(1, fly_count); % obtained from averaging corr matrices
fly_counter = 1;
for d = 1 : length(corrs_perFly_spearman)
	
	for fly = 1 : size(corrs_perFly_spearman{d}, 3)
		tmp = corrs_perFly_spearman{d}(:, :, fly);
		vals_perFly{fly_counter} = tmp(val_idxs);
		fly_counter = fly_counter + 1;
	end

end
vals = cat(1, vals_perFly{:});

figure;
histogram(vals);
title('r distribution (all flies, 105 channel pairs)');
xlabel('r');

%% Save

save_corrs = 0;

out_dir = 'channel_correlation';
out_file = 'corrs.mat';
if save_corrs == 1
	save([out_dir filesep out_file], 'corrs_pearson', 'corrs_spearman', 'corrs_perFly_pearson', 'corrs_perFly_spearman');
	disp(['saved correlations to ' out_dir filesep out_file]);
end

%% Channel clustering - per dataset

% Use the correlations computed from using all flies+epochs at a time

mds_dims = 2;
cluster_on = 'significances'; % 'significances' or 'values'

figure;
for d = 1 : length(dsets)
	
	% correlation distance
	switch cluster_on
		case 'values'
			corr_dist = 1 - abs(corrs_spearman{d});
			channels = (1 : size(corr_dist, 1));
		case 'significances'
			
			perf_type = 'consis'; % 'nearestMedian' or 'consis'
			batchNorm = 0;
			
			if batchNorm == 1
				dset_tail = 'BatchNormalised';
			else
				dset_tail = '';
			end
			
			[conds, cond_labels, cond_colours, stats_order, conds_main] = getConditions(dsets{d});
			
			% Get main condition pair ID in stats structure
			cond_idxs = (1 : length(conds));
			stats_idxs = cond_idxs(stats_order);
			stats_cond_idxs = find(ismember(stats_idxs, conds_main)); % condition idxs of main pair in the stats structure
			stats_pair = find(all(ismember(stats.([dsets{d} dset_tail]).(perf_type).condition_pairs, stats_cond_idxs), 2)); % pair idx in the stats structure
			
			% Obtain pattern of significance for the main condition pair
			pattern = stats.([dsets{d} dset_tail]).(perf_type).sig(ch, all(valid_all(ch, :), 1), stats_pair); % Limit to features which are valid for all channels
			pattern = pattern'; % features x ch
			
			% Remove channels with no significant features (correlation
			% undefined)
			channels = (1 : size(pattern, 2));
			nosig_ch = all(pattern==0, 1);
			channels(nosig_ch) = [];
			pattern(:, nosig_ch) = [];
			
			corr_dist = 1 - corr(pattern);
	end

	% MDS visualisation
	Y = mdscale(corr_dist, mds_dims);
	
	subplot(2, length(dsets), d);
	
	switch mds_dims
		case 2
			scatter(Y(:, 1), Y(:, 2));
			hold on
			text(Y(:, 1), Y(:, 2), cellstr(num2str(channels')));
			axis square
		case 3
			scatter3(Y(:, 1), Y(:, 2), Y(:, 3));
			hold on
			text(Y(:, 1), Y(:, 2), Y(:, 3), cellstr(num2str(channels')));
			axis vis3d
	end
	
	title(dsets{d});
	%xlim([-0.06 0.08]);
	%ylim([-0.06 0.08]);
	xlabel('dim 1');
	ylabel('dim 2');
	
	% Clustering dendrogram
	subplot(2, length(dsets), d+length(dsets))
	
	tree = linkage(squareform(corr_dist), 'average'); % note - distances must be pdist vector (treats matrix as data instead of distances)
	
	% Sorted features
	[h, T, order] = dendrogram(tree, 0, 'Labels', cellfun(@num2str, num2cell(channels), 'UniformOutput', false));
	xlabel('channel');
	xtickangle(90);
	axis square
	
end

%% Dataset clustering on average feature values or significance

% Compute correlation distances among datasets, using all values
% Note - different number of epochs, flies, conditions across datasets
%	So, first identify main condition pair (wake+unawake)
%	Then, average across epochs and flies each of the two chosen conditions
% Conduct using all channels together, or one at a time?

ch = (1:15);
cluster_on = 'significance'; % 'significance' or 'values'

switch cluster_on
	case 'values' % cluster based on averaged feature values
		corr_type = 'Spearman';
		
		hctsas_epochMean = cell(size(hctsas));
		for d = 1 : length(dsets)
			% Get main condition pairs for each dataset
			[conds, cond_labels, cond_colours, stats_order, conds_main] = getConditions(dsets{d});
			
			dims = size(hctsas{d}.TS_DataMat);
			hctsas_epochMean{d} = nan(length(conds_main), dims(2), dims(3)); % conds x features x channels
			% Get hctsa values associated with each condition
			for cond = 1 : length(conds_main)
				
				condRows = find(getIds({conds{conds_main(cond)}}, hctsas{d}.TimeSeries));
				
				% Average across epochs, then flies per condition
				% Note - grand average is same as averaging across epochs, then
				%	flies, assuming number of epochs is the same for all flies
				hctsas_epochMean{d}(cond, :, :) = mean(hctsas{d}.TS_DataMat(condRows, :, :), 1); % conds x features x channels
				
			end
			
			% Limit to valid features (across all datasets) of the channel(s)
			hctsas_epochMean{d} = hctsas_epochMean{d}(:, all(valid_all(ch, :), 1), ch);
			
			% Reshape into 1D vector holding all values to correlate with
			% Order doesn't matter, so long as other datasets follow the same order
			hctsas_epochMean{d} = hctsas_epochMean{d}(:);
			
		end
		
		% Concatenate dataset vectors into a matrix
		dset_vals = cat(2, hctsas_epochMean{:});
		%dset_vals = BF_NormalizeMatrix(dset_vals, 'mixedSigmoid');
		
	case 'significance' % cluster based on pattern of significance across features
		corr_type = 'Pearson';
		
		perf_type = 'consis'; % 'nearestMedian' or 'consis'
		batchNorm = 0;
		
		if batchNorm == 1
			dset_tail = 'BatchNormalised';
		else
			dset_tail = '';
		end
		
		patterns = cell(size(dsets));
		for d = 1 : length(dsets)
			
			[conds, cond_labels, cond_colours, stats_order, conds_main] = getConditions(dsets{d});
			
			% Get main condition pair ID in stats structure
			cond_idxs = (1 : length(conds));
			stats_idxs = cond_idxs(stats_order);
			stats_cond_idxs = find(ismember(stats_idxs, conds_main)); % condition idxs of main pair in the stats structure
			stats_pair = find(all(ismember(stats.([dsets{d} dset_tail]).(perf_type).condition_pairs, stats_cond_idxs), 2)); % pair idx in the stats structure
			
			% Obtain pattern of significance for the main condition pair
			patterns{d} = stats.([dsets{d} dset_tail]).(perf_type).sig(ch, all(valid_all(ch, :), 1), stats_pair); % Limit to features which are valid for all channels
			
			% Reshape into 1D vector
			% Order doesn't matter, so long as other datasets follow the
			% same order
			patterns{d} = patterns{d}(:);
		end
		
		% Concatenate dataset vectors into a matrix
		dset_vals = cat(2, patterns{:});
		
end

% Correlate averaged feature values between datasets
corrs_dset = corr(dset_vals, 'Type', corr_type, 'rows', 'pairwise');

switch cluster_on
	case 'values'
		% Treat negatively correlated values as similar
		corr_dist = 1 - abs(corrs_dset);
	case 'significance'
		% Treat different pattern of significance as dissimilar
		corr_dist = 1 - corrs_dset;
end

figure;

subplot(1, 3, 1);
imagesc(corrs_dset);
c = colorbar;
title('correlations between datasets');
title(c, 'r');
xlabel('dset');
ylabel('dset');
set(gca, 'XTick', (1:length(dsets)), 'XTickLabel', dsets);
set(gca, 'YTick', (1:length(dsets)), 'YTickLabel', dsets);
axis square

% Conduct MDS on correlation distances
mds_dims = 2;
opt = statset('MaxIter', 200);
Y = mdscale(single(corr_dist), mds_dims, 'Options', opt);
subplot(1, 3, 2);
scatter(Y(:, 1), Y(:, 2));
hold on;
text(Y(:, 1), Y(:, 2), dsets);
ylim(xlim);
xlabel('dim 1');
ylabel('dim 2');
title('MDS across datasets');
axis square

% Clustering on correlation distances
subplot(1, 3, 3);
tree = linkage(squareform(corr_dist), 'average'); % note - distances must be pdist vector (treats matrix as data instead of distances)
[h, T, order] = dendrogram(tree, 0, 'Labels', dsets);
xlabel('dataset');
axis square
