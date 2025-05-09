%% Description

%{

Plot DNV matrix for all flies from all datasets
	Average across epoch-pairs and channels

plot flies x features (averaged across channels)

Note - sleep dataset - feature 977 has nans after normalising

Note - RUN ./main_hctsa_matrix_evaluation_channelAverage.m first
	To get order of features
	Make sure cluster_features = 1

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

dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
dset_ids = (1:length(dsets));

%% Load stats
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
    
end

clear tmp
clear hctsa_ch

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

clear md_split

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

%% Concatenate hctsa matrices and normalise across all datasets, per channel
% Note - multidose dataset has thousands of rows
%   Meanwhile, discovery, singledose, and sleep sets have hundreds of rows

% Which datasets to concatenate together
cat_sets = (1:length(hctsas));

% Set to 0 if other datasets are to be scaled based on the first dataset
% Note, if scaling with a reference dataset, values in other datasets may
%   be outside the range of 0-1
% Set to -1 if datasets are to be scaled independently
scale_together = 0;

tic;
nRows = 0;
for d = cat_sets
    hctsas{d}.TS_Normalised = nan(size(hctsas{d}.TS_DataMat));
    nRows = nRows + size(hctsas{d}.TS_DataMat, 1);
end

for ch = 1 : 1% size(hctsas{1}.TS_DataMat, 3)
    
    hctsas_all = nan(nRows, size(hctsas{1}.TS_DataMat, 2));
    
    dset_rowStarts = nan(length(cat_sets)+1, 1);
    row_counter = 1;
    dims = nan(length(cat_sets), 2); % Assumes TS_DataMat has 3 dimensions, we ignore channels
    for d = cat_sets
        tic;
        tmp = hctsas{d}.TS_DataMat(valid_rows{d}, :, ch);
        dims(d, :) = size(tmp);
        hctsas_all(row_counter : row_counter+numel(find(valid_rows{d}))-1, :) = tmp;
        dset_rowStarts(d) = row_counter;
        row_counter = row_counter + numel(find(valid_rows{d}));
        toc
    end
    dset_rowStarts(end) = row_counter;

    % Normalise concatenated hctsa matrix
    disp('scaling concatenated hctsa matrix');
    tic;
	switch scale_together
		case 1
			hctsas_all_normalised = BF_NormalizeMatrix(hctsas_all, 'mixedSigmoid');
		case 0
			ref_dset = 1;
			reference_rows = zeros(size(hctsas_all, 1), 1);
			reference_rows(1:size(hctsas{ref_dset}.TS_DataMat, 1)) = 1;
			hctsas_all_normalised = BF_NormalizeMatrix(hctsas_all, 'mixedSigmoid', reference_rows);
		case -1
			for d = cat_sets
				hctsas{d}.TS_Normalised = BF_NormalizeMatrix(hctsas{d}.TS_DataMat(valid_rows{d}, :, ch), 'mixedSigmoid');
			end
	end
    % Separate concatenated, matrix back out
	switch scale_together
		case {1, 0}
			for d = cat_sets
				tmp = hctsas_all_normalised(dset_rowStarts(d):dset_rowStarts(d+1)-1, :);
				hctsas{d}.TS_Normalised(valid_rows{d}, :, ch) = tmp;
			end
		case -1
			% Do nothing
	end
    
    t = toc;
    disp(['ch' num2str(ch) ' scaled in ' num2str(t) 's']);
end

clear hctsas_all
clear hctsas_all_normalised
clear tmp

%% Get differences in normalised values for each dataset

% 0 - don't compute (takes lots of memory)
% 1 - compute, save, then remove from workspace
compute_multidose = 0;

out_dir = 'figure_workspace/';
out_prefix = 'dnvs';

if compute_multidose == 1
	%parpool(2);
	for d = 1 : length(hctsas)
		
		% Need to do it one channel at a time due to memory constraints
		for ch = 1 : 1% size(hctsas{d}.TS_Normalised, 3)
			
			% Restrict to given channel
			hctsa_tmp = hctsas{d};
			
			% Convert from doubles to single to reduce memory
			hctsa_tmp.TS_DataMat = single(hctsa_tmp.TS_DataMat(:, :, ch));
			hctsa_tmp.TS_Normalised = single(hctsa_tmp.TS_Normalised(:, :, ch));

			% Get main condition pair for the dataset
			[conds, cond_labels, cond_colours, stats_order, conds_main] =...
				getConditions(dsets{d});

			% Compute DNV
			tic;
			[dnvs, dnvs_rows] = dnv_parallel(hctsa_tmp, [],...
				conds{conds_main(1)},...
				conds{conds_main(2)});
			t = toc;
			dnvs = cellfun(@single, dnvs, 'UniformOutput', false);
			disp(['ch' num2str(ch) ' ' conds{conds_main(1)} ' ' conds{conds_main(2)} ' ' num2str(t) 's']);
			
			% Average across epoch pairs
			for fly = 1 : length(dnvs)
				dnvs{fly} = mean(dnvs{fly}, 1);
			end

			% Save (takes a long time for multidose if saving all epoch-pair differences)
			save([out_dir out_prefix '_' dsets{d} '_ch' num2str(ch) '.mat'], 'dnvs', 'dnvs_rows', '-v7.3', '-nocompression');

		end

	end
	
end

%% Load previously computed consistencies
% Assumes fIds, fValues, and trees variables from ../main_featureCluster_evaluation.m
%	are in the workspace, for feature selection at all channels

nCh = size(hctsas{d}. TS_Normalised, 3);

ch_values = cell(nCh, 1);
ch_rowLabels = cell(nCh, 1); % channels should share the same row labels
ch_featureOrders = cell(nCh, 1);
for ch = 1 : nCh
	tic;
	
	dset_values = cell(length(dsets), 1);
	dset_rowLabels = cell(length(dsets), 1);
	for d = 1 : length(dsets)
		
		% Load the DNVs for the dataset
		source_file = ['dnvs_' dsets{d} '_ch' num2str(ch)];
		tmp = load([out_dir source_file '.mat']);
		
		% Concatenate flies together
		dset_values{d} = cat(1, tmp.dnvs{:});
		
	end
	
	% Concatenate datasets together
	ch_values{ch} = cat(1, dset_values{:});
	
	%{
	% Reorder features based on pre-clustering
	f = figure('visible', 'off'); % we want the order, not the actual plot
	[h, T, outperm] = dendrogram(trees{ch}, size(trees{ch}, 1)+1);
	close(f);
	ch_featureOrders{ch} = outperm;
	ch_values{ch} = ch_values{ch}(:, outperm);
	%}
	
	t = toc;
	disp(['ch' num2str(ch) ' loaded in t=' num2str(t) 's']);
end

% Concatenate channels into one matrix along third dimension
vis_mat_allCh = cat(3, ch_values{:});

%% Average across channels

vis_mat_all = mean(vis_mat_allCh, 3);

%% Reset vis_mat

vis_mat = vis_mat_all(:, any(valid_all, 1), :);

% HACK - replace nan features (which occurred after scaling) with 0
vis_mat(isnan(vis_mat)) = 0;

%% Get yticks to delineate datasets and remove MD flies if desired
% And get xticks to delineate channels

remove_md = 0;

% if epoch-pairs were averaged across, no point in removing MD flies
%	because every fly takes up only one row
epochPair_mean = 1;
if epochPair_mean == 1
	remove_md = 0;
end

% Y ticks
ytickpos = ones(size(dsets));
for d = 1 : length(dsets) - 1
	
	[~, nFlies, ~, nEpochs] = getDimensionsFast(dsets{d});
	
	if epochPair_mean == 0
		nRows = nFlies*nEpochs*nEpochs;
	else
		nRows = nFlies;
	end
	ytickpos(d+1) = ytickpos(d) + nRows;
	
end
ytickstrings = dsets;

if remove_md == 1
	keep_rows = [(ytickpos(1):ytickpos(2)-1) (ytickpos(4):size(vis_mat, 1))];
	vis_mat = vis_mat_all(keep_rows, :);
	ytickstrings = dsets([1 4 5]);
	
	% Reobtain updated ytickpos
	ytickpos = ones(size(ytickstrings));
	for d = 1 : length(ytickstrings) - 1

		[~, nFlies, ~, nEpochs] = getDimensionsFast(ytickstrings{d});

		nRows = nFlies*nEpochs*nEpochs;
		ytickpos(d+1) = ytickpos(d) + nRows;

	end

end

%% Create figure (matrix)

figure;
set(gcf, 'Color', 'w');

imagesc(vis_mat(:, fOrder));
title(['differences in normalised values' newline 'mean across epoch-pairs and channels'], 'interpreter', 'none');
c = colorbar;
title(c, 'DNV');

set(gca, 'YTick', ytickpos, 'YTickLabel', ytickstrings);
xlabel('feature');

set(gca,'TickDir', 'out')

neg = viridis(256);
pos = inferno(256);
negPos_map = cat(1, flipud(neg(1:128, :)), pos(129:end, :));
negPos_map = flipud(cbrewer('div', 'RdBu', 100));
negPos_map(negPos_map < 0) = 0; % for some reason cbrewer is giving negative values...?
colormap(negPos_map);

%% Add ticks for specific features to highlight

% best performing feature in discovery - MD_rawHRVmeas_SD2
% best performing feature in pilot - rms
% best performing feature in evaluation - rms/MD_rawHRVmeas_SD2 @ ch1,
%	MD_rawHRVmeas_SD1 @ ch3
highlight_names = {'rms', 'MD_rawHRVmeas_SD2', 'NL_BoxCorrDim_50_ac_5_minr13'}';
highlight_ids = [16 7702 5072];

% Adjust ids to account for removed features
tmp = any(valid_all, 1);
highlight_vec = zeros(size(tmp));
highlight_vec(highlight_ids) = highlight_ids;
highlight_vec(~tmp) = [];
highlight_vec(removed) = [];
highlight_vec = highlight_vec(fOrder); % re-order features based on clustering
highlight_pos = find(highlight_vec);
[a, b] = ismember(highlight_vec(highlight_pos), highlight_ids); % re-order the highlight features based on clustering
highlight_names = highlight_names(b);

best_fId = 5072; % highest average performance across evaluation flies from excel pivot table
best_fName = 'NL_BoxCorrDim_50_ac_5_minr13';

old_ticks = xticks;
old_labels = xticklabels;

[new_ticks, tick_order] = sort([xticks highlight_pos]);
new_labels = cat(1, old_labels, highlight_names);
new_labels = new_labels(tick_order);

xticks(new_ticks);
xticklabels(new_labels);

set(gca, 'TickLabelInterpreter', 'none');

%% Print

print_fig = 0;

if print_fig == 1

	figure_name = 'figures_stage2/fig7_dnv_raw';

	set(gcf, 'PaperOrientation', 'Portrait');

	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG

end