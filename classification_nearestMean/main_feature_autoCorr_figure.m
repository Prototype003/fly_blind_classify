%% Description

%{

Create plot illustrating the differences in wake/anesthesia+sleep for
a given value (autocorrelation)

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

%% Load feature thresholds and directions

thresh_dir = ['results' preprocess_string filesep];
thresh_file = 'class_nearestMedian_thresholds.mat';
load([thresh_dir thresh_file]);

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

% Note - for log transform, check for negative minimum values in fVals{feature}
%	And update transform_features to offset accordingly for log transform


% NL_BoxCorrDim_50_ac_5_minr13 - 5072
ch = 1;
fIDs = [5072 4676];
fNames = hctsas{1}.Operations.Name(fIDs);
perf_type = 'nearestMedian'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
transform_types = {'log', ''};
transform_strings = {'log(log(f1+12.2))', 'f2'}; % cross-ref/update transform_values() function as needed
%%
% AC_33 - 125
ch = 1;
fIDs = [125 4676];
fNames = hctsas{1}.Operations.Name(fIDs);
perf_type = 'nearestMedian'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
transform_types = {'log', ''};
transform_strings = {'log(log(f1+1.31))', 'f2'}; % cross-ref/update transform_values() function as needed
%%
% ST_LocalExtrema_n50_stdmax - 3447
ch = 1;
fIDs = [3447 4676];
fNames = hctsas{1}.Operations.Name(fIDs);
perf_type = 'nearestMedian'; % 'nearestMedian' | 'nearestMedianBatchNormalised' | 'consis'
transform_types = {'log', ''};
transform_strings = {'log(f1+min)', 'f2'}; % cross-ref/update transform_values() function as needed

% Select which fly and dataset to show from?
%	Or would it be better to show one time series from each fly?
%		Total 12+18+19 time-series? Or show just from one dataset?

%% Compute autocorrelation for each time-series

ch = 1;

fly_offsets = [0 0 8 0 0]; % multidose4 flies start from 9

nLags = 100; % autocorrelation computations starts from delay 0

acfs = cell(size(hctsas));
acfs_mean = cell(size(acfs));
acfs_std = cell(size(acfs));

for d = 1 : length(dsets)
	
	[conditions, ~, ~, ~, cond_pair] = getConditions(dsets{d});
	conditions = conditions(cond_pair);
	[~, nFlies, ~, nEpochs] = getDimensionsFast(dsets{d});
	
	acf_dset = nan(nLags+1, 2, nFlies, nEpochs); % autocorrelation computations starts from delay 0
	
	tic;
	for f = 1 : nFlies
		fly = f + fly_offsets(d);
		
		for c = 1 : 2
			
			% Identify rows for each fly and condition
            keys = {['fly' num2str(fly)], conditions{c}, ['channel' num2str(ch)]};
            rows = getIds(keys, hctsas{d}.TimeSeries);
			rows = find(rows);
			
			% Compute autocorrelation function for each time-series
			for r = 1 : length(rows)
				x = hctsas{d}.TimeSeries.Data{rows(r)};
				[acf, lags] = autocorr(x, nLags);
				
				acf_dset(:, c, f, r) = acf;
			end
			
		end
		
		toc
	end
	
	acfs{d} = acf_dset;
	
	% Average across epochs, then flies
	acfs_mean{d} = mean(mean(acfs{d}, 4), 3);
	acfs_std{d} = std(mean(acfs{d}, 4), [], 3);
	
end

%% Plot every fly

figure;
cond_colours = {'r', 'b'};
lags = (0:nLags);

ac_thresholds = thresholds(ch, 93:132); % thresholds for AC features (AC1-AC40)
ac_thresholds = [nan ac_thresholds nan(1, nLags-length(ac_thresholds))];

negPos_map = flipud(cbrewer('div', 'RdBu', 100));
negPos_map(negPos_map < 0) = 0; % for some reason cbrewer is giving negative values...?

for d = 1 : length(acfs)
	
	acfs_perFly = median(acfs{d}, 4);
	
	subplot(5, length(acfs), d);
	imagesc(squeeze(acfs_perFly(:, 1, :))');
	colorbar;
	title([dsets{d} ' median(W)']);
	xlabel('tau'); ylabel('fly');
	
	subplot(5, length(acfs), d + length(acfs));
	imagesc(squeeze(acfs_perFly(:, 2, :))');
	colorbar;
	title('median(U)');
	xlabel('tau'); ylabel('fly');
	
	subplot(5, length(acfs), d + 2*length(acfs));
	imagesc(squeeze(acfs_perFly(:, 1, :))' - squeeze(acfs_perFly(:, 2, :))', [-0.1 0.1]);
	colorbar;
	title('median(W)-median(U)');
	xlabel('tau'); ylabel('fly');
	colormap(gca, negPos_map);
	
	subplot(5, length(acfs), d + 3*length(acfs));
	imagesc(squeeze(acfs_perFly(:, 1, :))' - squeeze(acfs_perFly(:, 2, :))' > 0);
	%imagesc(squeeze(acfs_perFly(:, 1, :))' - squeeze(acfs_perFly(:, 2, :))' > repmat(ac_thresholds, [size(acfs_perFly, 3) 1]));
	colorbar;
	title('median(W)-median(U)>0');
	xlabel('tau'); ylabel('fly');
	
	subplot(5, length(acfs), d + 4*length(acfs));
	plot(lags, sum(squeeze(acfs_perFly(:, 1, :))' - squeeze(acfs_perFly(:, 2, :))' > 0));
	%plot(lags, median(acfs_perFly(:, 1, :), 3), 'r'); hold on;
	%plot(lags, median(acfs_perFly(:, 2, :), 3), 'b');
	title('nFlies med(W)>med(U)');
	xlabel('tau'); ylabel('fly count');
end

%% Plot, combining variance across epochs and flies

figure;
lags = (0:nLags);

for d = 1 : length(acfs)
	
	dims = size(acfs{d});
	acfs_flyEpoch = reshape(acfs{d}, [dims(1) dims(2) dims(3)*dims(4)]);
	acfs_flyEpoch_mean = mean(acfs_flyEpoch, 3);
	
	subplot(1, length(acfs), d);
	plot(lags, acfs_flyEpoch_mean(:, 1), 'r'); hold on;
	plot(lags, acfs_flyEpoch_mean(:, 2), 'b');
	
end

%% Plot, combining all flies across datasets

acfs_perFly = cell(size(acfs));
for d = 1 : length(acfs)
	acfs_perFly{d} = mean(acfs{d}, 4);
end

acfs_combined = cat(3, acfs_perFly{:});
acfs_combined_centre = mean(acfs_combined, 3);
acfs_combined_std = std(acfs_combined, [], 3);

figure;
lags = (0:nLags);

plot(lags, acfs_combined_centre(:, 1), 'r'); hold on;
plot(lags, acfs_combined_centre(:, 2), 'b');


%% Plot, combinding all flies across datasets
% Plot diff

figure;
lags = (0:nLags);

acfs_combined_diff = acfs_combined(:, 1, :) - acfs_combined(:, 2, :);
acfs_combined_diff_mean = mean(acfs_combined_diff, 3);
acfs_combined_diff_std = std(acfs_combined_diff, [], 3) ./ sqrt(size(acfs_combined_diff, 3));

ebar_x = [lags fliplr(lags)];
ebar_y = [acfs_combined_diff_mean' fliplr(acfs_combined_diff_mean')] + [-acfs_combined_diff_std' fliplr(acfs_combined_diff_std')];
patch(ebar_x, ebar_y, 'r', 'FaceAlpha', 0.1); hold on;
plot(lags, acfs_combined_diff_mean);

%% Plot, combining all epochs, flies across datasets

acfs_perFlyEpoch = cell(size(acfs));
for d = 1 : length(acfs)
	dims = size(acfs{d});
	acfs_perFlyEpoch{d} = reshape(acfs{d}, [dims(1) dims(2) dims(3)*dims(4)]);
end

acfs_combined = cat(3, acfs_perFlyEpoch{:});
acfs_combined_centre = mean(acfs_combined, 3);
acfs_combined_std = std(acfs_combined, [], 3);

figure;
lags = (0:nLags);


plot(lags, acfs_combined_centre(:, 1), 'r'); hold on;
plot(lags, acfs_combined_centre(:, 2), 'b');


%{
acfs_combined_diff = acfs_combined(:, 1, :) - acfs_combined(:, 2, :);
acfs_combined_diff_mean = mean(acfs_combined_diff, 3);
acfs_combined_diff_std = std(acfs_combined_diff, [], 3);

ebar_x = [lags fliplr(lags)];
ebar_y = [acfs_combined_diff_mean' fliplr(acfs_combined_diff_mean')] + [-acfs_combined_diff_std' fliplr(acfs_combined_diff_std')];
patch(ebar_x, ebar_y, 'r', 'FaceAlpha', 0.1); hold on;
plot(lags, acfs_combined_diff_mean);
%}

%% Directly plot distributions
% After combining flies and epochs across datasets

acfs_perFlyEpoch = cell(size(acfs));
for d = 1 : length(acfs)
	dims = size(acfs{d});
	acfs_perFlyEpoch{d} = reshape(acfs{d}, [dims(1) dims(2) dims(3)*dims(4)]);
end

acfs_combined = cat(3, acfs_perFlyEpoch{:});

%acfs_combined = log(acfs_combined + abs(min(acfs_combined(:))));
%acfs_combined = log(log(acfs_combined + 1.55));

figure;
cond_colours = {'r', 'b'};

nPanels = size(acfs_combined, 1);
for t = 2 : size(acfs_combined, 1)
	
	subplot(floor(sqrt(nPanels)), ceil(sqrt(nPanels)), t);
	title(num2str(t-1));
	
	% Remove extreme values, for visualisation purposes
	tmp = acfs_combined(t, :, :);
	upper_lim = prctile(tmp, 99, 'all');
	lower_lim = prctile(tmp, 1, 'all');
	tmp(tmp>upper_lim) = nan;
	tmp(tmp<lower_lim) = nan;
	acfs_combined(t, :, :) = tmp;
	
	for c = 1 : size(acfs_combined, 2)
		binWidth = (prctile(acfs_combined(t, :, :), 90, 'all') - prctile(acfs_combined(t, :, :), 10, 'all')) / 20;
		histogram(acfs_combined(t, c, :),...
			'BinWidth', binWidth,...
			'Orientation', 'horizontal',...
			'FaceColor', cond_colours{c},...
			'FaceAlpha', 0.5,...
			'EdgeAlpha', 0);
		hold on;
	end
	title(num2str(t-1));
	
end

%% Plot conditions for each dataset

figure;

cond_colours = {'r', 'b'};
lags = (0:nLags);

% Give idea of trained threshold
tmp = acfs{1};
dims = size(tmp);
tmp = reshape(tmp, [dims(1) dims(2) dims(3)*dims(4)]);
diff_thresh = median(tmp, 3);
diff_thresh = mean(diff_thresh, 2);

for d = 1 : length(acfs)
	
	%subplot(length(acfs), 1, d);
	subplot(floor(sqrt(length(acfs))), ceil(sqrt(length(acfs))), d);
	
	% Difference for each fly
	acfs_perFly = median(acfs{d}, 4);
	
	for c = 1 : size(acfs_perFly, 2)
		
		acfs_perFly_mean = mean(acfs_perFly(:, c, :), 3)';
		acfs_perFly_std = std(acfs_perFly(:, c, :), [], 3)';% ./ sqrt(size(acfs_perFly, 3));
		
		ebar_x = [lags fliplr(lags)];
		ebar_y = [acfs_perFly_mean fliplr(acfs_perFly_mean)] + [-acfs_perFly_std fliplr(acfs_perFly_std)];
		
		l = plot(lags, acfs_perFly_mean, cond_colours{c}); hold on;
		patch(ebar_x, ebar_y, l.Color, 'FaceAlpha', 0.1, 'EdgeAlpha', 0, 'HandleVisibility', 'off');
		
	end
	
	%plot(lags, diff_thresh, 'k');
	legend('wake', 'unawake', 'thresh');
	
	
	title(dsets{d});
	xlabel('tau');
	ylabel('r');
	
end

%% Plot difference for all datasets together

figure; hold on;

cond_colours = {'r', 'b'};
lags = (0:nLags);

use_flies = {...
	(1:size(acfs{1}, 3)),...
	(1:size(acfs{2}, 3)),...
	(1:size(acfs{3}, 3)),...
	(1:size(acfs{4}, 3))...
	(1:size(acfs{5}, 3))};

use_flies{4}([3 12 16]) = [];

acfs_perFly = cell(size(acfs));
for d = 1 : length(acfs)
	acfs_perFly{d} = median(acfs{d}(:, :, use_flies{d}, :), 4);
end

acfs_perFly = cat(3, acfs_perFly{:});

acfs_diff = acfs_perFly(:, 1, :) - acfs_perFly(:, 2, :);

% Mean + stderr
acfs_diff_mid = mean(acfs_diff, 3)';
acfs_diff_std = std(acfs_diff, [], 3)' ./ sqrt(size(acfs_diff, 3));

% Median with 5-95%tile
acfs_diff_mid = median(acfs_diff, 3)';
acfs_diff_prc95 = prctile(acfs_diff, 95, 3)';
acfs_diff_prc5 = prctile(acfs_diff, 5, 3)';

ebar_x = [lags fliplr(lags)];
ebar_y = [acfs_diff_mid fliplr(acfs_diff_mid)] + [-acfs_diff_std fliplr(acfs_diff_std)];
ebar_y = [acfs_diff_prc95 fliplr(acfs_diff_prc5)];

l = plot(lags, acfs_diff_mid);
patch(ebar_x, ebar_y, l.Color, 'FaceAlpha', 0.1, 'EdgeAlpha', 0);

% Significant ACs are AC_29 to AC_34
scatter((29:34), 0, [], 'k', '*', 'HandleVisibility', 'off');

% Show end of range of hctsa features
line([40 40], ylim, 'Color', 'k');

legend(['mean (N=' num2str(size(acfs_diff, 3)) ')'], 'stderr', 'hctsa \tau max');
legend(['median (N=' num2str(size(acfs_diff, 3)) ')'], '5-95%tile', 'hctsa \tau max');

xlabel('tau'); ylabel('\Deltar');

title(['autocorrelation (wake minus anes/sleep)' newline 'mean + stderr across N=' num2str(size(acfs_diff, 3)) ' flies']);
title(['autocorrelation (wake minus anes/sleep)' newline 'median with 5-95%tile across N=' num2str(size(acfs_diff, 3)) ' flies']);

%% Plot difference for each dataset

figure; hold on;
set(gcf, 'Color', 'w');

cond_colours = {'r', 'b'};
lags = (0:nLags);

use_flies = {...
	(1:size(acfs{1}, 3)),...
	(1:size(acfs{2}, 3)),...
	(1:size(acfs{3}, 3)),...
	(1:size(acfs{4}, 3))...
	(1:size(acfs{5}, 3))};

use_flies{4}([3 12 16]) = [];

for d = 1 : length(acfs)
	
	% Difference for each fly
	acfs_perFly = median(acfs{d}, 4);
	%acfs_diff = acfs_perFly(:, 1, :) - acfs_perFly(:, 2, :);
	acfs_diff = acfs_perFly(:, 1, use_flies{d}) - acfs_perFly(:, 2, use_flies{d});
	acfs_diff_mean = mean(acfs_diff, 3)';
	acfs_diff_std = std(acfs_diff, [], 3)' ./ sqrt(size(acfs_diff, 3));
	
	ebar_x = [lags fliplr(lags)];
	ebar_y = [acfs_diff_mean fliplr(acfs_diff_mean)] + [-acfs_diff_std fliplr(acfs_diff_std)];
	
	l = plot(lags, acfs_diff_mean);
	patch(ebar_x, ebar_y, l.Color, 'FaceAlpha', 0.1, 'EdgeAlpha', 0, 'HandleVisibility', 'off');
	
end

legend(dsets, 'AutoUpdate', 'off');

% Significant ACs are AC_29 to AC_34
scatter((29:34), 0, [], 'k', '*');

% Show end of range of hctsa features
line([40 40], ylim, 'Color', 'k', 'LineStyle', ':');

% Show zero
line(xlim, [0 0], 'Color', 'k');

xlabel('tau'); ylabel('\Deltar');

title(['autocorrelation (wake minus anes/sleep)' newline 'mean + stderr across flies']);
box on
axis tight

%% Print figure

print_fig = 0;

if print_fig == 1
	
	axis tight
	box on
	
	figure_name = '../figures_stage2/fig_autocorr_raw';
	
	set(gcf, 'PaperOrientation', 'Portrait');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters', '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG
	
end

%% Show all epochs for a single fly

d = 4;
fly = 4;

tmp = acfs{d}(:,:, 4, :);


tmp = permute(tmp, [1 4 2 3]);

figure;

subplot(1, 2, 1);
imagesc(tmp(:, :, 1)'); c = colorbar;
ylabel(c, 'r');
title([dsets{d} ' fly' num2str(fly) ' wake']);
xlabel('tau');
ylabel('epoch');

subplot(1, 2, 2);
imagesc(tmp(:, :, 2)'); c = colorbar;
ylabel(c, 'r');
title([dsets{d} ' fly' num2str(fly) ' unawake']);
xlabel('tau');
ylabel('epoch');

%% Get time-series

d = 5;

[conditions, ~, ~, ~, cond_pair] = getConditions(dsets{d});
conditions = conditions(cond_pair);
[~, nFlies, ~, nEpochs] = getDimensionsFast(dsets{d});

d_tseries = nan(hctsas{d}.TimeSeries.Length(1), length(cond_pair), nFlies, nEpochs);

for f = 1 : nFlies
	fly = f + fly_offsets(d);
	
	for c = 1 : 2
		
		% Identify rows for each fly and condition
		keys = {['fly' num2str(fly)], conditions{c}, ['channel' num2str(ch)]};
		rows = getIds(keys, hctsas{d}.TimeSeries);
		rows = find(rows);
		
		d_tseries(:, c, f, :) = cat(2, hctsas{d}.TimeSeries.Data{rows});
	end
	
end

%% Plot time-series

dims = size(d_tseries);
d_tseries_mat = reshape(permute(d_tseries, [1 4 2 3]), [dims(1) dims(4)*dims(2)*dims(3)]); % t x epochs*conds*flies

figure; imagesc(d_tseries_mat', [-50 50]); c = colorbar;
title(dsets{d});
ylabel(c, 'V');
set(gca, 'YTick', (1:dims(4)*dims(2):dims(4)*dims(2)*dims(3)), 'YTickLabel', (1:dims(3)));

xlabel('t');
ylabel('fly (wake then unawake epochs contiguous)');

%% Power spectra

sample_rate = 1000;

params = struct();
params.Fs = sample_rate;
params.tapers = [5 9];
params.pad = 2;
params.removeFreq = [];

[powers, faxis] = getPower(d_tseries, params);

%% Remove line noise

d_tseries_cleaned = removeLineNoise(d_tseries, params);
[powers_cleaned, faxis] = getPower(d_tseries_cleaned, params); % faxis should be same as before

%% Plot before/after

cond_colours = {'r', 'b'};

figure;

for f = 1 : nFlies
	subplot(ceil(sqrt(nFlies)), ceil(sqrt(nFlies)), f);
	hold on;
	
	for c = 1 : 2
		
		plot(faxis, mean(log(squeeze(powers(:, c, f, :))), 2), cond_colours{c});
		
	end
	
	title(['fly' num2str(f) 'cleaned']);
	ylabel('log(power)');
	xlabel('f (Hz)');
	
end

%{
%% Compare to raw(er) data

tic;
tmp = load('../data/preprocessed2/flyEvaluation_data_subtractMean.mat');
toc

tmp = tmp.data.singledose;

%%

% Fake hctsa.TimeSeries
TimeSeries_copy = struct();
TimeSeries_copy.Keywords = tmp.keywords;

% Get data for the channel
ch = 1;

d_tseries_orig = nan(hctsas{d}.TimeSeries.Length(1), length(cond_pair), nFlies, nEpochs);

for f = 1 : nFlies
	fly = f + fly_offsets(d);
	
	for c = 1 : 2
		
		% Identify rows for each fly and condition
		keys = {['fly' num2str(fly)], conditions{c}, ['channel' num2str(ch)]};
		rows = getIds(keys, TimeSeries_copy);
		rows = find(rows);
		
		d_tseries_orig(:, c, f, :) = cat(2, tmp.data{rows});
	end
	
end

%% Plot time-series

dims = size(d_tseries_orig);
d_tseries_orig_mat = reshape(permute(d_tseries_orig, [1 4 2 3]), [dims(1) dims(4)*dims(2)*dims(3)]); % t x epochs*conds*flies

figure; imagesc(d_tseries_orig_mat'); c = colorbar;
ylabel(c, 'V');
set(gca, 'YTick', (1:dims(4)*dims(2):dims(4)*dims(2)*dims(3)), 'YTickLabel', (1:dims(3)));

xlabel('t');
ylabel('fly (wake then unawake epochs contiguous)');

%% Get power spectra

[powers_orig, faxis] = getPower(d_tseries_orig, params);

%% Plot before and after

cond_colours = {'r', 'b'};

figure;

for f = 1 : nFlies
	subplot(floor(sqrt(nFlies)), ceil(sqrt(nFlies)), f);
	hold on;
	
	for c = 1 : 2
		
		plot(faxis, mean(log(squeeze(powers_orig(:, c, f, :))), 2), cond_colours{c});
		
	end
	
	title(['fly' num2str(f)]);
	ylabel('log(power)');
	xlabel('f (Hz)');
	
end

%% Compare 50 Hz power between conditions for each fly
% Conduct t-test/non-para across epochs per fly

% Find frequency closest to 50 Hz
[~, faxis50] = min(abs(50-faxis));

dims = size(powers);
cond_colours = {'r', 'b'};

% Plot for each fly
figure;

for f = 1 : nFlies
	subplot(floor(sqrt(nFlies)), ceil(sqrt(nFlies)), f);
	
	for c = 1 : dims(2)
		
		scatter((1:dims(4)), sort(squeeze(powers(faxis50, c, f, :))), [], cond_colours{c});
		%histogram(powers(faxis50, c, f, :), 'FaceColor', cond_colours{c});
		hold on;
		
	end
	
end

%% Wake - anaesthesia difference

diff_in_log = 0;

% Averaged across epochs (to avoid issue of pairing epochs for condition
% difference
switch diff_in_log
	case 0 % note - difference can be negative, so can't take log after
		powers_epochMean = mean((powers), 4);
		power_string = '\Deltapower';
		freq_bins = (-11:0.12:11);
	case 1
		powers_epochMean = mean(log(powers), 4);
		power_string = '\Deltalog(power)';
		freq_bins = (-1.8:0.09:1.8);
end

% Subtract anaesthesia cleaned from wake cleaned
powers_condDiff = powers_epochMean(:, 1, :) - powers_epochMean(:, 2, :);
powers_condDiff = (permute(powers_condDiff, [1 3 2])); % frequencies x flies

% Then, difference spectrum should be "flat"
figure;
for f = 1 : nFlies
	subplot(ceil(sqrt(nFlies)), ceil(sqrt(nFlies)), f);
	
	plot(faxis, powers_condDiff(:, f));
	title([dsets{d} ' fly' num2str(f)]);
	xlabel('f (Hz)');
	ylabel(power_string);
	%{
	switch diff_in_log
		case 00
			ylabel('\Deltapower')
		case 1
			ylabel('\Deltalog(power)');
	end
	%}
	xlim([1 150]);
	
end

%}

%% Plot for all frequencies

figure;

plot(faxis', powers_condDiff);
legend(cellfun(@num2str, num2cell((1:size(powers_condDiff, 2))), 'UniformOutput', false));
xlim([1 150]);

title(dsets{d});
xlabel('f (Hz)');
ylabel(power_string);

%% Then, look at distribution 50 Hz and other freq across flies to identify
%	individuals with high systematic difference in 50 Hz power between
%	conditions

switch diff_in_log
	case 0
		freq_bins = (-11:0.12:11);
	case 1
		freq_bins = (-2.5:0.09:2.5);
end

% Frequencies to look at
freq_points = [45 50 55];

% Stores specific faxis points which corresponds most closely to a
%	given frequency
faxis_points = nan(size(freq_points));

for freq = 1 : length(freq_points)
	[~, faxis_points(freq)] = min(abs(freq_points(freq)-faxis));
end

figure;

for freq = 1 : length(freq_points)
	subplot(length(freq_points), 1, freq);
	
	histogram(powers_condDiff(faxis_points(freq), :), freq_bins); hold on;
	scatter(powers_condDiff(faxis_points(freq), :), ones(1, size(powers_condDiff, 2)));
	
	%xlim([-2 2]);
	
	title([dsets{d} ' ' num2str(freq_points(freq)) 'Hz']);
	xlabel(power_string);
	ylabel('nFlies');
	
end

%% Look at all flies together


% Get time-series
d_tseries_all = cell(size(dsets));
dlabels = cell(size(dsets));
for d = 1 : length(dsets)
	[conditions, ~, ~, ~, cond_pair] = getConditions(dsets{d});
	conditions = conditions(cond_pair);
	[~, nFlies, ~, nEpochs] = getDimensionsFast(dsets{d});

	d_tseries = nan(hctsas{d}.TimeSeries.Length(1), length(cond_pair), nFlies, nEpochs);
	dlabels{d} = repmat(d, [1 nFlies]);

	for f = 1 : nFlies
		fly = f + fly_offsets(d);

		for c = 1 : 2

			% Identify rows for each fly and condition
			keys = {['fly' num2str(fly)], conditions{c}, ['channel' num2str(ch)]};
			rows = getIds(keys, hctsas{d}.TimeSeries);
			rows = find(rows);

			d_tseries(:, c, f, :) = cat(2, hctsas{d}.TimeSeries.Data{rows});
		end

	end
	
	d_tseries_all{d} = d_tseries;
end

dlabels_backup = dlabels;

%% Power spectra

sample_rate = 1000;

params = struct();
params.Fs = sample_rate;
params.tapers = [5 9];
params.pad = 2;
params.removeFreq = [];

powers_all = cell(size(dsets));
for d = 1 : length(dsets)
	tic;
	[powers, faxis] = getPower(d_tseries_all{d}, params);
	powers_all{d} = powers;
	toc
end

powers_all_backup = powers_all;

%% Mean across epochs, concatenate across fly dimension

powers_all = powers_all_backup;
dlabels = dlabels_backup;

diff_in_log = 1;

switch diff_in_log
	case 0
		powers_all = cellfun(@(x) mean(x, 4), powers_all, 'UniformOutput', false);
		power_string = '\Deltapower';
	case 1
		powers_all = cellfun(@(x) mean(log(x), 4), powers_all, 'UniformOutput', false);
		power_string = '\Deltalog(power)';
end

powers_all = cat(3, powers_all{:});
dlabels = cat(2, dlabels{:});

%% Wake - anaesthesia difference

nFlies = size(powers_all, 3);

% Subtract anaesthesia cleaned from wake cleaned
powers_condDiff = powers_all(:, 1, :) - powers_all(:, 2, :);
powers_condDiff = (permute(powers_condDiff, [1 3 2])); % frequencies x flies

% Then, difference spectrum should be "flat"
figure;
for f = 1 : nFlies
	subplot(ceil(sqrt(nFlies)), ceil(sqrt(nFlies)), f);
	
	plot(faxis, powers_condDiff(:, f));
	title(['fly' num2str(f)]);
	xlabel('f (Hz)');
	ylabel(power_string);
	%{
	switch diff_in_log
		case 00
			ylabel('\Deltapower')
		case 1
			ylabel('\Deltalog(power)');
	end
	%}
	xlim([1 150]);
	
end

%% Plot for all frequencies

figure;

plot(faxis', powers_condDiff);
legend(cellfun(@num2str, num2cell((1:size(powers_condDiff, 2))), 'UniformOutput', false));
xlim([1 150]);

title(dsets{d});
xlabel('f (Hz)');
ylabel(power_string);

%% Then, look at distribution 50 Hz and other freq across flies to identify
%	individuals with high systematic difference in 50 Hz power between
%	conditions

switch diff_in_log
	case 0
		freq_bins = (-10.5:0.12:10.5);
	case 1
		freq_bins = (-2.5:0.05:2.5);
end

cmap = cbrewer('qual', 'Set1', 5);

% Frequencies to look at
freq_points = [45 50 55];

% Stores specific faxis points which corresponds most closely to a
%	given frequency
faxis_points = nan(size(freq_points));

for freq = 1 : length(freq_points)
	[~, faxis_points(freq)] = min(abs(freq_points(freq)-faxis));
end

figure;
colormap(cmap);

for freq = 1 : length(freq_points)
	subplot(length(freq_points), 1, freq);
	
	% Histogram across all flies
	histogram(powers_condDiff(faxis_points(freq), :), freq_bins); hold on;
	ylabel('nFlies');
	
	% Show which dataset each point belongs to
	yyaxis right
	scatter(powers_condDiff(faxis_points(freq), :), dlabels, [], dlabels);
	ylabel('dset');
	set(gca, 'YTick', (1:length(dsets)), 'YTickLabel', dsets);
	
	title([num2str(freq_points(freq)) 'Hz']);
	xlabel(power_string);
	
end

%% Look at 50 vs mean(45, 55) Hz

figure;
set(gcf, 'Color', 'w');
colormap(cmap);

ref_freq = 2; % Which freq in freq_points is the one to focus on
ref_freq = (1:length(freq_points)) == ref_freq; % logical index

% Show 50 Hz
subplot(3, 1, 1);
vals = mean(powers_condDiff(faxis_points(ref_freq), :), 1);
histogram(vals, freq_bins); hold on;
ylabel('nFlies')
% Show which dataset each point belongs to
yyaxis right
scatter(vals, dlabels, [], dlabels);
ylabel('dset');
set(gca, 'YTick', (1:length(dsets)), 'YTickLabel', dsets);
title([num2str(freq_points(ref_freq)) 'Hz']);
xlabel(power_string);

% Show mean of comparison frequencies
subplot(3, 1, 2);
vals = mean(powers_condDiff(faxis_points(~ref_freq), :), 1);
histogram(vals, freq_bins); hold on;
ylabel('nFlies');
% Show which dataset each point belongs to
yyaxis right
scatter(vals, dlabels, [], dlabels);
ylabel('dset');
set(gca, 'YTick', (1:length(dsets)), 'YTickLabel', dsets);
title(['mean(' num2str(freq_points(~ref_freq)) ' Hz)']);
xlabel(power_string);

subplot(3, 1, 3);
% Show difference between 50 Hz vs comparison
vals = mean(powers_condDiff(faxis_points(ref_freq), :), 1) - mean(powers_condDiff(faxis_points(~ref_freq), :), 1);
histogram(vals, freq_bins); hold on;
ylabel('nFlies');
% Show which dataset each point belongs to
yyaxis right
scatter(vals, dlabels, [], dlabels);
ylabel('dset');
set(gca, 'YTick', (1:length(dsets)), 'YTickLabel', dsets);
title([num2str(freq_points(ref_freq)) 'Hz - ' 'mean(' num2str(freq_points(~ref_freq)) ' Hz)']);
xlabel([power_string '_{' num2str(freq_points(ref_freq)) '} - ' power_string '_{' num2str(freq_points(~ref_freq)) '}']);
% Calculate mean + SD
m = mean(vals);
sd = std(vals);
line([m m], ylim, 'Color', 'r', 'LineStyle', '-', 'Marker', 'none');
line([m+sd m+sd], ylim, 'Color', 'r', 'LineStyle', '--', 'Marker', 'none');
line([m-sd m-sd], ylim, 'Color', 'r', 'LineStyle', '--', 'Marker', 'none');
line([m+2*sd m+2*sd], ylim, 'Color', 'r', 'LineStyle', '--', 'Marker', 'none');
line([m-2*sd m-2*sd], ylim, 'Color', 'r', 'LineStyle', '--', 'Marker', 'none');
line([m+3*sd m+3*sd], ylim, 'Color', 'r', 'LineStyle', '--', 'Marker', 'none');
line([m-3*sd m-3*sd], ylim, 'Color', 'r', 'LineStyle', '--', 'Marker', 'none');

%% Print figure

print_fig = 0;

if print_fig == 1
	
	axis tight
	box on
	
	figure_name = '../figures_stage2/fig_50HzDiff';
	
	set(gcf, 'PaperOrientation', 'Portrait');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters');%, '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG
	
end


%%
find(vals>m+2*sd)
find(vals<m-2*sd)
