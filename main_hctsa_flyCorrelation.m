%% Description

%{

Check correlations among flies/conditions
Do flies with same conditions cluster together?
Do conditions form distinct clusters?

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
	
	task1 = stats.(dsets{d}).valid_features;
	
	disp(['ch' num2str(ch) '-' dsets{d} ': ' num2str(numel(find(task1(ch, :))))]);
	
	valid_all = valid_all & task1;
	
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
	
	task1 = nan([size(hctsas{d}.TS_DataMat) size(valid_all, 1)-1]);
	hctsas{d}.TS_DataMat = cat(3, hctsas{d}.TS_DataMat, task1);
	
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

%% Convert into matrix

% Each row corresponds to one fly, one condition
% Values hold feature values averaged across epochs

ch = 1;

% Determine total number of fly-condition instances
nRows = 0;
for d = 1 : length(dsets)
	[nCh, nFlies, ~, nEpochs] = getDimensionsFast(dsets{d}); % note nConditions for singledose includes the extra recovery condition for some flies
	[conditions, cond_labels] = getConditions(dsets{d});
	nRows = nRows + (nFlies*length(conditions));
end

flyVals = nan(nRows, length(find(all(valid_all(ch, :), 1))));
row_labels = cell(nRows, 1);
row_dsets = nan(nRows, 1);
row_conds = nan(nRows, 1);
row_mainCond = nan(nRows, 1);
row_flies = nan(nRows, 1);
row_colours = cell(nRows, 1); % colours for each fly/condition
row_shapes = cell(nRows, 1); % shapes for each dataset

row_counter = 1;
for d = 1 : length(dsets)
	[nCh, nFlies, ~, nEpochs] = getDimensionsFast(dsets{d});
	[conditions, cond_labels, cond_colours, stats_order, isMain] = getConditions(dsets{d})
	
	% If you want to group conditions together, have condition as the outer
	%	loop
	for cond = 1 : length(conditions)
		tic;
		for fly = 1 : nFlies
			fly_id = fly;
			if strcmp(dsets{d}, 'multidose4')
				fly_id = fly_id + 8;
			end
			
			% Get values for the condition and fly
			key = {conditions{cond}, ['fly' num2str(fly_id)]};
			[~, key_rows] = getIds(key, hctsas{d}.TimeSeries, 'and');
			
			% Average across relevant rows (epochs)
			task1 = hctsas{d}.TS_DataMat(key_rows, all(valid_all(ch, :), 1), ch); % epochs x features x channels
			task1 = permute(task1, [2 1 3]); % features x epochs x channels
			task1 = reshape(task1, [size(task1, 1), size(task1, 2)*size(task1, 3)]); % features x epochs*channels
			% Replace infs with nans, to ignore when averaging
			task1(isinf(task1)) = nan;
			flyVals(row_counter, :) = mean(task1, 2, 'omitnan')';
			
			row_labels{row_counter} = [cond_labels{cond} '_fly' num2str(fly)];
			row_dsets(row_counter) = d;
			row_conds(row_counter) = cond;
			row_mainCond(row_counter) = any(isMain == cond);
			row_flies(row_counter) = fly;
			row_counter = row_counter + 1;
		end
		t = toc;
		disp([conditions{cond} ' t=' num2str(t) 's']);
	end
	
end

row_mainCond = logical(row_mainCond);

% Subtract each condition from wake
deltaWake = 0;

if deltaWake == 1
	% Reference (wake) condition of each dataset
	dset_ref_conds = [1 1 1 1 2]; % note sleep has wakeEarly, then wake
	
	flyDiffs = flyVals;
	
	for d = 1 : length(dsets)
		[conditions, cond_labels, cond_colours] = getConditions(dsets{d});
		
		for cond = 1 : length(conditions)
			
			% Note - presumes fly order is the same (which it should be)
			flyDiffs(row_dsets==d & row_conds==cond, :) =...
				flyVals(row_dsets==d & row_conds==dset_ref_conds(d), :) -...
				flyVals(row_dsets==d & row_conds==cond, :);
		end
		
	end

	% Remove wake conditions
	toDelete = nan(size(row_dsets));
	for d = 1 : length(dsets)
		toDelete(row_dsets==d & row_conds==dset_ref_conds(d)) = 1;
	end
	flyDiffs(toDelete==1, :) = [];
	row_dsets(toDelete==1) = [];
	row_conds(toDelete==1) = [];
	row_mainCond(toDelete==1) = []
	row_flies(toDelete==1) = [];
	row_labels(toDelete==1) = [];
	
	flyVals = flyDiffs;
	
end

% Scaled values
flyVals_scaled = BF_NormalizeMatrix(flyVals, 'mixedSigmoid');
[coeff, score, latent, tsquared, explained] = pca(flyVals_scaled);

%% Compute correlations between flies
% Maybe try PCA to reduce redundant/correlated features first?

tic;
corrs_s = corr(flyVals', 'Type', 'Spearman');
corrs = corr(flyVals_scaled', 'Type', 'Pearson');
toc

corr_dist = 1 - (corrs);

%% Check distribution of values
% Why negative Pearson correlation despite high Spearman correlation?
% row 66 and 69 have high spearman correlation but negative pearson
% row 58 and 82 have high spearman correlation but close to 0 pearson

row1 = 66;
row2 = 69;

%row1 = 58;
%row2 = 82;

label1 = [dsets{row_dsets(row1)} ' cond' num2str(row_conds(row1)) ' fly' num2str(row_flies(row1))];
label2 = [dsets{row_dsets(row2)} ' cond' num2str(row_conds(row2)) ' fly' num2str(row_flies(row2))];

figure;

subplot(1, 3, 1);
scatter(flyVals(row1, :), flyVals(row2, :), '.');
xlabel(label1);
ylabel(label2);
title(['unscaled epoch-meaned values' newline...
	'r_s=' num2str(corrs_s(row1, row2)) ' r_p=' num2str(corrs(row1, row2))],...
	'interpreter', 'none');
axis square

subplot(1, 3, 2);
scatter(tiedrank(flyVals(row1, :)), tiedrank(flyVals(row2, :)), '.');
xlabel(label1);
ylabel(label2);
title(['rank of epoch-meaned values' newline...
	'r_s=' num2str(corrs_s(row1, row2)) ' r_p=' num2str(corrs(row1, row2))],...
	'interpreter', 'none');
axis square
axis tight

subplot(1, 3, 3);
scatter(flyVals_scaled(row1, :), flyVals_scaled(row2, :), '.');
xlabel(label1);
ylabel(label2);
title(['mixed-sigmoid-normed epoch-mean values' newline...
	'r_s=' num2str(corrs_s(row1, row2)) ' r_p=' num2str(corrs(row1, row2))],...
	'interpreter', 'none');
axis square
axis tight

%% Correlation matrix

% corrs = pearson; corrs_s = spearman
% row_mainCond - restrict to only rows with main conditions
plot_corrType = corrs_s(row_mainCond, row_mainCond);

% Get first row of each dataset
% 13 flies, 8 flies, 4 flies, 18 flies, 19 flies
tick_pos = [1 14 27 35 43 47 51 69 87 106];
tickLabels = row_labels(row_mainCond);

figure;
imagesc((plot_corrType));
colormap inferno

c = colorbar;
title(c, 'r');

axis square

set(gca, 'XTick', tick_pos, 'XTickLabel', tickLabels(tick_pos));
set(gca, 'YTick', tick_pos, 'YTickLabel', tickLabels(tick_pos));
ytickangle(25);
xtickangle(75);
set(gca, 'TickLabelInterpreter', 'None');

%% Clustering dendrogram

figure;
subplot(10, 1, [1 5]);

% Clustering dendrogram
tree = linkage(squareform(corr_dist), 'average'); % note - distances must be pdist vector (treats matrix as data instead of distances)

% Sorted flies
[h, T, order] = dendrogram(tree, 0, 'Labels', row_labels);
xtickangle(75);
set(gca, 'TickLabelInterpreter', 'None', 'FontSize', 8);

% Label leaves with colours for conditions
%{
cond rgb

1 wake 1 0 0
2 postiso 0.8 0 0.8
3 recovery 0.8 0 0.4

4 iso0.6 0 0 0.8
5 iso1.2 0 0 1

6 sleepEarly 0 1 0
7 sleepLate 0 0.8 0
%}
cmap_conds = [...
	1 0 0;...
	0.8 0 0.8;...
	0.5 0 1;...
	0 0 1;...
	0 0 0.5;
	0 1 0;
	0 0.8 0];
cmap_conds_labels = {...
	'wake',...
	'recover',...
	'postiso',...
	'iso0.6',...
	'iso1.2',...
	'sleep2',...
	'sleep5'};

cond_image = nan(1, length(row_conds));

% d = 1
cond_image(row_dsets == 1 & row_conds == 1) = 1;
cond_image(row_dsets == 1 & row_conds == 2) = 4;
% d = 2
cond_image(row_dsets == 2 & row_conds == 1) = 1;
cond_image(row_dsets == 2 & row_conds == 2) = 4;
cond_image(row_dsets == 2 & row_conds == 3) = 5;
cond_image(row_dsets == 2 & row_conds == 4) = 3;
cond_image(row_dsets == 2 & row_conds == 5) = 2;
% d = 3
cond_image(row_dsets == 3 & row_conds == 1) = 1;
cond_image(row_dsets == 3 & row_conds == 2) = 4;
cond_image(row_dsets == 3 & row_conds == 3) = 5;
cond_image(row_dsets == 3 & row_conds == 4) = 3;
cond_image(row_dsets == 3 & row_conds == 5) = 2;
% d = 4
cond_image(row_dsets == 4 & row_conds == 1) = 1;
cond_image(row_dsets == 4 & row_conds == 2) = 4;
cond_image(row_dsets == 4 & row_conds == 3) = 3;
% d = 5
cond_image(row_dsets == 5 & row_conds == 1) = 1;
cond_image(row_dsets == 5 & row_conds == 2) = 1;
cond_image(row_dsets == 5 & row_conds == 3) = 6;
cond_image(row_dsets == 5 & row_conds == 4) = 7;

subplot(10, 1, [7 8]);
imagesc(cond_image(order));
c = colorbar('Location', 'SouthOutside');
% Note - colormap axis goes from 1 to however many colours specified (N)
% So the length of the colormap is (N-1)
% So the spacing of colours is (N-1)/N, i.e. length divided by N
cmap_spacing = (size(cmap_conds, 1)-1)/size(cmap_conds, 1);
set(c, 'XTick', (1:cmap_spacing:size(cmap_conds, 1))+(cmap_spacing/2), 'XTickLabel', cmap_conds_labels);
colormap(gca, cmap_conds);

% Label leaves with colours for datasets
cmap_dsets = repmat(linspace(1, 0, 5)', [1 3]);
cmap_dsets = [...
	1 0 0;...
	0.8 0 0.8;...
	0.5 0.3 0;...
	0 0 1;...
	0 1 0];
cmap_dsets_labels = dsets;
subplot(10, 1, [9 10]);
imagesc(row_dsets(order)');
c = colorbar('Location', 'SouthOutside');
cmap_spacing = (size(cmap_dsets, 1)-1)/size(cmap_dsets, 1);
set(c, 'XTick', (1:cmap_spacing:size(cmap_dsets, 1))+(cmap_spacing/2), 'XTickLabel', cmap_dsets_labels);
colormap(gca, cmap_dsets);

%% MDS visualisation

mds_dims = 2;

Y = mdscale(single(corr_dist), mds_dims);

figure;
hold on;

dset_shapes = {'o', 'hexagram', 'pentagram', '^', 'square'};

for d = 1 : length(dsets)
	[conditions, cond_labels, cond_colours] = getConditions(dsets{d});
	conditions = unique(row_conds(row_dsets == d)); % if deltaWake, the wake condition was removed
	
	for c = 1 : length(conditions)
		cond = conditions(c);
		rows = row_dsets == d & row_conds == cond;
		scatter(Y(rows, 1), Y(rows, 2), [], cmap_conds(cond_image(rows), :), dset_shapes{d});
	end

end
xlabel('dim1');
ylabel('dim2');
c = colorbar;
colormap(gca, cmap_conds);
cmap_spacing = 1/size(cmap_conds, 1);
set(c, 'YTick', (0:cmap_spacing:size(cmap_conds, 1))+(cmap_spacing/2), 'YTickLabel', cmap_conds_labels);

% Text labels
%{
text(Y(:, 1), Y(:, 2), row_labels, 'Interpreter', 'None');
%}


%% tSNE visualisation

Y = tsne(flyVals_scaled);

figure;
hold on;

dset_shapes = {'o', 'hexagram', 'pentagram', '^', 'square'};

for d = 1 : length(dsets)
	[conditions, cond_labels, cond_colours] = getConditions(dsets{d});
	conditions = unique(row_conds(row_dsets == d)); % if deltaWake, the wake condition was removed
	
	for c = 1 : length(conditions)
		cond = conditions(c);
		rows = row_dsets == d & row_conds == cond;
		scatter(Y(rows, 1), Y(rows, 2), [], cmap_conds(cond_image(rows), :), dset_shapes{d});
	end
	
end

xlabel('dim1');
ylabel('dim2');
c = colorbar;
colormap(gca, cmap_conds);
cmap_spacing = 1/size(cmap_conds, 1);
set(c, 'YTick', (0:cmap_spacing:size(cmap_conds, 1))+(cmap_spacing/2), 'YTickLabel', cmap_conds_labels);

%% Find mean correlation
% 1 - across datasets, within conscious
% 2 - across datasets, within unconscious
% 3 - within dataset, between conscious and unconscious

nTasks = 3;
tasks = cell(nTasks, 1);
task_text = {'consc. across dsets', 'unconsc. across dsets', 'wake*unconsc. within dset'};

dset_conscious = {[1], [1], [1], [1], [2]};
dset_unconscious = {[2], [3], [3], [2], [3]};

% 1 - across datasets, within conscious
tasks{1} = [];
for d1 = 1 : length(dsets)
	rows1 = row_dsets == d1 & ismember(row_conds, dset_conscious{d1});
	
	% correlations between d1 and other datasets
	tmpd = [];
	for d2 = d1+1 : length(dsets)
		rows2 = row_dsets == d2 & ismember(row_conds, dset_conscious{d2});
		
		tmpd = cat(2, tmpd, corrs(rows1, rows2));
		
	end
	
	% Concatenate together, filling mismatched dimensions with nans
	% d1==1 should always have the most columns
	if d1 > 1
		nCols1 = size(tasks{1}, 2);
		nCols2 = size(tmpd, 2);
		if nCols2 < nCols1
			pad_cols = nan(size(tmpd, 1), nCols1-nCols2);
		end
		tmpd = cat(2, pad_cols, tmpd);
	end
	tasks{1} = cat(1, tasks{1}, tmpd);
end


% 2 - across datasets, within conscious
tasks{2} = [];
for d1 = 1 : length(dsets)
	rows1 = row_dsets == d1 & ismember(row_conds, dset_unconscious{d1});
	
	% correlations between d1 and other datasets
	tmpd = [];
	for d2 = d1+1 : length(dsets)
		rows2 = row_dsets == d2 & ismember(row_conds, dset_unconscious{d2});
		
		tmpd = cat(2, tmpd, corrs(rows1, rows2));
		
	end
	
	% Concatenate together, filling mismatched dimensions with nans
	% d1==1 should always have the most columns
	if d1 > 1
		nCols1 = size(tasks{2}, 2);
		nCols2 = size(tmpd, 2);
		if nCols2 < nCols1
			pad_cols = nan(size(tmpd, 1), nCols1-nCols2);
		end
		tmpd = cat(2, pad_cols, tmpd);
	end
	tasks{2} = cat(1, tasks{2}, tmpd);
end

% 3 - within dataset, between conscious and unconscious
tasks{3} = [];
for d = 1 : length(dsets)
	
	for cond = 1 : 2
		rows1 = row_dsets == d & ismember(row_conds, dset_conscious{d});
		rows2 = row_dsets == d & ismember(row_conds, dset_unconscious{d});
		
		tmp = corrs(rows1, rows2);
	end
	
	% Concatenate together
	if d > 1
		nCols1 = size(tasks{3}, 2);
		nCols2 = size(tmp, 2);
		
		pad_cols1 = nan(size(tasks{3}, 1), nCols2);
		pad_cols2 = nan(size(tmp, 1), nCols1);
		
		tasks{3} = cat(2, tasks{3}, pad_cols1);
		tmp = cat(2, pad_cols2, tmp);
	end
	
	tasks{3} = cat(1, tasks{3}, tmp);
	
end

global_min = min(tasks{1}(:));
global_max = max(tasks{1}(:));
for task = 2 : 3
	
	newmin = min(tasks{task}(:));
	if newmin < global_min
		global_min = newmin;
	end
	
	newmax = max(tasks{task}(:));
	if newmax > global_max
		global_max = newmax;
	end
end

figure;
for task = 1 : 3
	
	subplot(nTasks, 2, (task*2)-1);
	imagesc(tasks{task}, [global_min global_max]);
	colormap inferno;
	c = colorbar;
	title(c, 'r');
	title(task_text{task});
	axis square
	xlabel('fly');
	ylabel('fly');
	%set(gca,'XTick', [], 'YTick', []);
	
	subplot(nTasks, 2, task*2);
	histogram(tasks{task}, 'BinWidth', 0.01); xlim([global_min global_max]);
	%cdfplot(tasks{task}(:));
	
	tmp1 = mean(tasks{task}(:), 'omitnan');
	tmp2 = median(tasks{task}(:), 'omitnan');
	ytmp = ylim;
	hold on;
	line([tmp1 tmp1], [ytmp(1) ytmp(end)], 'Color', 'r', 'LineWidth', 1.5);
	line([tmp2 tmp2], [ytmp(1) ytmp(end)], 'Color', 'b', 'LineWidth', 1.5);
	title(['mean=' num2str(tmp1) newline 'median=' num2str(tmp2)]);
	xlabel('r');
	ylabel('cum. prop.');
end
