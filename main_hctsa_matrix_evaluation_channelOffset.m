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

clear tmp
clear hctsas_all
clear hctsas_all_normalised

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

%% Match up channels
% In discovery flies, outermost electrode is outside the eye
% In other flies, outermost electrode is inside the eye
% Correlates SCALED values (no infs)

% 0: 1-to-1
% -1: 1-to-2 - corresponds to more likely case
% 1: 2-to-1 - reverse of the likely case

alignments = [-2 -1 0 1 2]; % shift in evaluation channels
alignments = [0 -1];

ref_dset = 1; % discovery flies
off_dsets = (2:5); % other flies

alignment_rs = cell(size(alignments));
for a = 1 : length(alignments)
	alignment = alignments(a);
	
	channels = (1:size(hctsas{1}.TS_DataMat, 3)); % channels in the discovery flies
	channels_aligned = channels + alignment; % offset channels in evaluation flies
	
	% Cut off extra extreme channels at the ends
	channels_aligned(channels_aligned > max(channels)) = nan;
	channels_aligned(channels_aligned < min(channels)) = nan;
	
	% Get feature values from the reference dataset
	ref_vals = hctsas{ref_dset}.TS_NormalisedMean;
	
	% Get feature values from the offset channel in the other datasets
	off_vals = [];
	for d = 1 : length(off_dsets)
		off_vals = cat(1, off_vals, hctsas{off_dsets(d)}.TS_NormalisedMean);
	end
	
	% Number of fly pairs
	nFlyPairs = size(ref_vals, 1) * size(off_vals, 1);
		
	% Correlation values for each fly pair
	rs = nan(nFlyPairs, length(channels));
	
	% Match each channel with the corresponding offset channel
	for ch = 1 : length(channels)
		tic;
		
		if isnan(channels(ch)) || isnan(channels_aligned(ch))
			% Do nothing (don't have value values to correlate)
		else
			
			% Features which are valid in both channels
			valid_offset = valid_all(channels(ch), :) & valid_all(channels_aligned(ch), :);
			
			% Get each pair of flies
			pair_counter = 1;
			for ref_fly = 1 : size(ref_vals, 1)
				for off_fly = 1 : size(off_vals, 1)
					
					% Get values for features which are valid in both channels
					tmpa = ref_vals(ref_fly, valid_offset, :, channels(ch));
					tmpb = off_vals(off_fly, valid_offset, :, channels_aligned(ch));
					
					% Correlate, ignore values which become nan from scaling
					r = corr(tmpa(:), tmpb(:), 'rows', 'pairwise', 'type', 'Spearman');
					
					rs(pair_counter, ch) = r;
					
					pair_counter = pair_counter + 1;
					
				end
			end
			
		end
		
		toc
		drawnow('update');
		
	end
	
	alignment_rs{a} = rs;
	
end

alignment_rs = cat(3, alignment_rs{:});

%% Plot histograms of correlation for each offset
% alignments
% alignment_rs

% Show histogram across all fly pairs and channels
figure; set(gcf, 'Color', 'w');
for a = 1 : size(alignment_rs, 3)
	subplot(length(alignments)+1, 1, a);
	tmp = alignment_rs(:, :, a);
	histogram(tmp);
	tmp_mean = mean(tmp(:), 'omitnan');
	line([tmp_mean tmp_mean], ylim, 'LineWidth', 2, 'Color', 'k');
	title(num2str(alignments(a)));
end
% Show together in one plot
subplot(length(alignments)+1, 1, length(alignments)+1);
hold on;
for a = 1 : size(alignment_rs, 3)
	histogram(alignment_rs(:, :, a));
end
legend(num2str(alignments(:)));
xlabel('r');
ylabel('count');
box on;

%%
% Show histogram across channels (mean across fly pairs)
alignment_rs_channelMean = mean(alignment_rs, 1);
figure; set(gcf, 'Color', 'w');
for a = 1 : size(alignment_rs, 3)
	subplot(length(alignments)+1, 1, a);
	tmp = alignment_rs_channelMean(:, :, a);
	histogram(tmp, (-1:0.05:1));
	tmp_mean = mean(tmp(:), 'omitnan');
	line([tmp_mean tmp_mean], ylim, 'LineWidth', 2, 'Color', 'k');
	title(num2str(alignments(a)));
end
% Show together in one plot
subplot(length(alignments)+1, 1, length(alignments)+1);
hold on;
for a = 1 : size(alignment_rs, 3)
	tmp = alignment_rs_channelMean(:, :, a);
	histogram(tmp, (-1:0.05:1));
end
legend(num2str(alignments(:)));
xlabel('r');
ylabel('channels');
box on;

%% Do stats test
% Fisher z transform, then t-test

tmp = fisher_rz(squeeze(alignment_rs_channelMean));

disp(['offset ' num2str(alignments(1)) ' vs 0']);
[h, p, ci, stats] = ttest(tmp(:, 1), tmp(:, 2), 'tail', 'right')
%{
disp(['offset ' num2str(alignments(3)) ' vs 0']);
[h, p, ci, stats] = ttest(tmp(:, 3), tmp(:, 2), 'tail', 'right')

disp(['offset ' num2str(alignments(1)) ' vs ' num2str(alignments(3))]);
[h, p, ci, stats] = ttest(tmp(:, 1), tmp(:, 3), 'tail', 'right')
%}

%% Show distribution of values per channel

% Fisher Z transform
zs = fisher_rz(alignment_rs);
zs_chMean = squeeze(mean(zs, 1)); % mean per channel

% Set y-offset for each point
%	Use same offsets across alignments (for the same fly pair)
y_offsets = (rand(size(zs, 1), size(zs, 2)) - 0.5) .* 0.5;
y_offsets = repmat(y_offsets, [1 1 size(zs, 3)]);

% Set ypos as channel+yoffset
y_channels = repmat((1:size(zs, 2)), [size(zs, 1) 1 size(zs, 3)]);
ys = y_channels + y_offsets;

% For when showing channel means
ys_chMean = [(1:size(zs, 2)) - 0.25; (1:size(zs, 2)) + 0.25];

figure;
set(gcf, 'Color', 'w');

for a = 1 : size(zs, 3)
	subplot(1, size(zs, 3), a);
	hold on;
	
	% Put channel on x-axis
	scatter(ys(:, :, a), zs(:, :, a), 2, 'k.');
	
	% Show means for each channel
	plot(ys_chMean, repmat(zs_chMean(:, a)', [2 1]), 'Color', 'r', 'LineWidth', 2);
	
	if a == 1
		
		ylabel('z');
		xlabel('channel (discovery)');
		
	end
	
	xlim([1 size(zs, 2)] + [-0.5 0.5]);
	ylim([min(zs(:)) max(zs(:))]);
	
	box on
	
	title(['offset ' num2str(alignments(a))]);
	
end

%% Conduct LME

z_table = matrixToTable(zs, {'flyPair', 'channel', 'offset', 'z'}, {'categorical', 'categorical', 'discrete', 'numeric'});

z_model = fitlme(z_table, 'z ~ offset + (1|flyPair) + (1|channel)');
null_model = fitlme(z_table, 'z ~ 1 + (1|flyPair) + (1|channel)');

results = compare(null_model, z_model);
disp(results);

%% Print

print_fig = 0;

if print_fig == 1
	
	figure_name = 'figures_stage2/channelOffset_raw';
	
	set(gcf, 'PaperOrientation', 'Portrait');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG

end