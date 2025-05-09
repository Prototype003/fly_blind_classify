%% Description

%{

Plot feature matrix for all datasets

Average across epochs and channels, per fly
OR
Average epochs and flies, per channel

Note - sleep dataset - feature 977 has nans after normalising

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

%% Normalise data per channel (per dataset)
% Normalise across datasets or per dataset?
% Per dataset - all datasets will have the same range of values
% Across datasets - all datasets will be relative to discovery flies

% Normalisation per dataset

% Note - even with mixedSigmoid, some features scale to NaNs and 0s
%   discovery flies - feature 976 (870th valid feature) scales
%       to NaN a nd 0s
%   sleep flies - feature 977 scales to NaN and a 0

% Time to scale every channel, multidose8 - ~836s

%{

scale_each = 0;

if scale_each == 1

	for d = 1 : length(hctsas)
		disp(['scaling dataset ' dsets{d}]);

		hctsas{d}.TS_Normalised = nan(size(hctsas{d}.TS_DataMat));

		for ch = 1 : size(hctsas{d}.TS_Normalised, 3)
			tic;

			hctsas{d}.TS_Normalised(valid_rows{d}, :, ch) =...
				BF_NormalizeMatrix(hctsas{d}.TS_DataMat(valid_rows{d}, :, ch), 'mixedSigmoid');

			t = toc;
			disp(['ch' num2str(ch) ' scaled in ' num2str(t) 's']);
		end

	end

end

%}

%% Concatenate hctsa matrices and normalise across all datasets, per channel
% Note - multidose dataset has thousands of rows
%   Meanwhile, discovery, singledose, and sleep sets have hundreds of rows

% Which datasets to concatenate together
cat_sets = (1:length(hctsas));

% Set to 0 if other datasets are to be scaled based on the first dataset
% Note, if scaling with a reference dataset, values in other datasets may
%   be outside the range of 0-1
scale_together = 0;

tic;
nRows = 0;
for d = cat_sets
    hctsas{d}.TS_Normalised = nan(size(hctsas{d}.TS_DataMat));
    nRows = nRows + numel(find(valid_rows{d})); %size(hctsas{d}.TS_DataMat, 1);
end

for ch = 1 : size(hctsas{1}.TS_DataMat, 3)
    
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
    if scale_together == 1
        hctsas_all_normalised = BF_NormalizeMatrix(hctsas_all, 'mixedSigmoid');
    elseif scale_together == 0
        ref_dset = 1;
        reference_rows = zeros(size(hctsas_all, 1), 1);
        reference_rows(1:size(hctsas{ref_dset}.TS_DataMat, 1)) = 1;
        hctsas_all_normalised = BF_NormalizeMatrix(hctsas_all, 'mixedSigmoid', reference_rows);
    end
    % Separate concatenated, matrix back out
    for d = cat_sets
        tmp = hctsas_all_normalised(dset_rowStarts(d):dset_rowStarts(d+1)-1, :);
        hctsas{d}.TS_Normalised(valid_rows{d}, :, ch) = tmp;
    end
    
    t = toc;
    disp(['ch' num2str(ch) ' scaled in ' num2str(t) 's']);
end

%% Average across epochs

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
    
    hctsas{d}.TS_NormalisedMean = nan(nFlies, size(hctsas{d}.TS_Normalised, 2), length(conditions), nCh);
    
    for c = 1 : length(conditions)
        tic;
        for f = 1 : length(flies) % Average across epochs, for the fly
            fly = flies(f);
            
            % Identify rows for the fly and condition
            keys = {['fly' num2str(fly)], conditions{c}};
            rows = getIds(keys, hctsas{d}.TimeSeries);
            
            % Average across epochs, for the fly
            hctsas{d}.TS_NormalisedMean(f, :, c, :) =...
                mean(hctsas{d}.TS_Normalised(rows, :, :), 1);
            
        end
        toc
    end
    
end


%% Average across channels
% Average AFTER replacing invalid features per channel with NaNs and
%   after flipping feature values based on direction

channel_mean = 1;

channel_selection = (1:15); % (1:15)

switch channel_mean
	case 1
		% Scale before or after normalising values?
		for d = 1 : length(hctsas)
			tic;
			hctsas{d}.TS_NormalisedMean = mean(hctsas{d}.TS_NormalisedMean(:, :, :, channel_selection), 4, 'omitnan');
			toc
		end
		
		ch = 1;
		disp(['averaged across channels: ' num2str(channel_selection)]);
	otherwise
		disp('didn''t average across channels');
		channel_selection = (1:15);
end


%% Join datasets together

% Row order
%ordering = 'dset_first'; % order by dataset, then condition within dataset
ordering = 'cond_first'; % order by condition, then dataset within condition

vis_mat = [];
dset_labels = {'D', 'MD8', 'MD4', 'SD', 'S'};
switch ordering
    case 'dset_first'
        
        % Concatenate averaged values across datasets
        for d = 1 : length(hctsas)
            for c = 1 : size(hctsas{d}.TS_NormalisedMean, 3)
                vis_mat = cat(1, vis_mat, hctsas{d}.TS_NormalisedMean(:, :, c, :));
            end
        end
        
        % Get axis ticks and labels
        ytickstrings = {};
        ytickpos = [];
        ycounter = 1;
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
            
            for cond = 1 : length(conditions)
                ytickpos = [ytickpos ycounter];
                ytickstrings = cat(2, ytickstrings, strcat(dset_labels{d}, '_', cond_labels(cond)));
                
                ycounter = ycounter + nFlies;
                
            end
            
        end

    case 'cond_first'
        % Note - assumes same number of conditions for all datasets
        
        % Concatenate averaged values across datasets
        for c = 1 : size(hctsas{1}.TS_NormalisedMean, 3)
            for d = 1 : length(hctsas)
                vis_mat = cat(1, vis_mat, hctsas{d}.TS_NormalisedMean(:, :, c, :));
            end
        end
        
        % Get axis ticks and labels
        ytickstrings = {};
        ytickpos = [];
        ycounter = 1;
        for cond = 1 : size(hctsas{1}.TS_NormalisedMean, 3)
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
        
    otherwise
        warning('datasets not joined');
end

vis_mat_all = vis_mat; % Keep this unaltered, clustering can take a while (~20 minutes)

%% Cluster features

cluster_features = 1;

if cluster_features == 1
	
	% Switch between all and any as needed, for clustering purpose
	%	all() - valid in all datasets across ALL channels
	%	any() - valid in all datasets for at least one channel
	vis_mat = vis_mat_all(:, any(valid_all, 1));
	
	% Cluster features
	
	disp('clustering features');
	tic;
	[fOrder, removed] = clusterFeatures(vis_mat);
	toc
	
	vis_mat(:, removed) = [];
	
end

%% Get positions of features to highlight

if cluster_features == 1
	
	%highlight_names = {'rms', 'SP_Summaries_welch_rect_area_2_1', 'NL_BoxCorrDim_50_ac_5_minr13'}';
	%highlight_ids = [16 5428 5072];
	
	% best performing feature in discovery - StatAvl250
	% best performing feature in pilot - SP_Summaries_welch_rect_area_2_1
	% best performing feature in pilot with norm. - DN_Moments_raw_4
	% best performing feature in evaluation - NL_BoxCorrDim_50_ac_5_minr13
	highlight_names = {'DN_Moments_raw_4', 'StatAvl250', 'SP_Summaries_welch_rect_area_2_1', 'NL_BoxCorrDim_50_ac_5_minr13'}';
	highlight_ids = [33 551 5428 5072];
	
	% Adjust ids to account for removed features
	tmp = any(valid_all, 1);
	highlight_vec = zeros(size(all(valid_all, 1)));
	highlight_vec(highlight_ids) = highlight_ids;
	highlight_vec(~tmp) = [];
	highlight_vec(removed) = [];
	highlight_vec = highlight_vec(fOrder); % re-order features based on clustering
	highlight_pos = find(highlight_vec);
	[a, b] = ismember(highlight_vec(highlight_pos), highlight_ids); % re-order the highlight features based on clustering
	highlight_names = highlight_names(b);
	
	best_fId = 5072; % highest average performance across evaluation flies from excel pivot table
	best_fName = 'NL_BoxCorrDim_50_ac_5_minr13';
	
end

%% Get feature selection variables from classification_nearestMean/main_featureCluster_evaluation.m
% There, features are clustered using raw values

%	fIds - feature selection
%	outperm - order of selected features

preclustered_features = 0;

switch preclustered_features
	case 1
		% Assumes variables from classification_nearestMean/main_featureCluster_evaluation.m
		%	are in the workspace
		% fIds, outperm
		
		% Channel selection (features would have been clustered per channel
		ch = 1;

		vis_mat = vis_mat_all(:, fIds{ch}(outperm));
		fOrder = 1 : length(outperm); % because vis_mat features are already order by outperm
		
		best_fId = 5072; % highest average performance across evaluation flies from excel pivot table
		best_fName = 'NL_BoxCorrDim_50_ac_5_minr13';
		
	case 2
		% Assumes variables from classification_nearestMean/main_featureCluster_allCh_evaluation.m
		%	are in the workspace
		% fIds, chIds, outperm
		
		vis_mat = nan(size(vis_mat_all, 1), length(fIds{1}));
		
		% Get selected features from their associated channels
		for f = 1 : length(fIds{1})
			fCh = fIds{1}(f);
			vis_mat(:, f) = vis_mat_all(:, fIds{1}(f), 1, chIds{1}(f));
		end
		
		fOrder = outperm;
		
end

%{
if preclustered_features == 1
	% Assumes variables from classification_nearestMean/main_featureCluster_evaluation.m
	%	are in the workspace

	ch = 1; % features would have been clustered per channel

	vis_mat = vis_mat_all(:, fIds{ch}(outperm));
	fOrder = 1 : length(outperm); % because vis_mat features are already order by outperm

	best_fId = 5072; % highest average performance across evaluation flies from excel pivot table
	best_fName = 'NL_BoxCorrDim_50_ac_5_minr13';

end
%}

%% Create figure

fig = figure;
set(fig, 'Color', 'w');
handle = imagesc(vis_mat(:, fOrder));
cbar = colorbar;

%% Add axis ticks and other details

colormap inferno

title(cbar, 'scaled value');

set(gca, 'TickLabelInterpreter', 'none');
yticks(ytickpos);
yticklabels(ytickstrings);
ylabel('fly');

xlabel('feature');

if channel_mean == 1
	title(['mean of normalised values, across epochs and channels' newline 'ch' num2str(channel_selection(1)) '-ch' num2str(channel_selection(end)) newline num2str(best_fId) ' ' best_fName], 'interpreter', 'none');
else
	title(['mean of normalised values, across epochs' newline 'ch' num2str(channel_selection(1)) '-ch' num2str(channel_selection(end)) newline num2str(best_fId) ' ' best_fName], 'interpreter', 'none');
end
%%
% Add ticks for features to highlight
if cluster_features == 1
	old_ticks = xticks;
	old_labels = xticklabels;
	
	[new_ticks, tick_order] = sort([xticks highlight_pos]);
	new_labels = cat(1, old_labels, highlight_names);
	new_labels = new_labels(tick_order);
	
	xticks(new_ticks);
	xticklabels(new_labels);
end

if preclustered_features == 1
	xticks([1 find(fIds{ch}(outperm) == best_fId) size(vis_mat, 2)]);
	xticklabels({'1', ['f' num2str(best_fId)], num2str(length(outperm))});
end

set(gca,'TickDir', 'out')

%% Print

print_fig = 0;

if print_fig == 1

	switch preclustered_features
		case 1
			figure_name = 'figures_stage2/fig6_hctsaMatChMean_raw';
		case 2
			figure_name = 'figures_stage2/fig7_hctsaMatCh_raw';
		otherwise % 0 - the figure shows all valid features
			figure_name = 'figures_stage2/fig4_hctsaMatChMean_raw';
	end
	
	set(gcf, 'PaperOrientation', 'Portrait');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG

end