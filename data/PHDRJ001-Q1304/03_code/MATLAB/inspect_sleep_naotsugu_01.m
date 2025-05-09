%% Extract 18 seconds
% Extract 18 seconds before and after sleep start time to divide single
% sleep epochs into wake and sleep.
%% NOTES
% 18/06/2022 RJ Created new file.
% 28/08/2023 AL
%   Replace Z:\ with ..\..\
%   Extract 18s before sleep start time
%   Extract 18s, from 5 minutes after sleep start time

%% Script Start

fly_list = {'27072018_SponSleep_24hrs';
'14092018_SponSleep_LFP'; 
'17092018_SponSleep_LFP';
'19092018_SponSleep_LFP';
'30102018_SponSleep_LFP';
'01112018_SponSleep_LFP';
'03112018_SponSleep_LFP';
'13112018_SponSleep_LFP';
'21112018_SponSleep_LFP';
'22112018_SponSleep_LFP';
'28112018_SponSleep_LFP';
'11122018_SponSleep_LFP';
'13122018_SponSleep_LFP';
'18122018_SponSleep_LFP';
'10012019_SponSleep_LFP';
'22012019_SponSleep_LFP';
'20022019_SponSleep_LFP';
'06032019_SponSleep_LFP';
'13032019_SponSleep_LFP'}

blocks_to_extract = [2,2,1,5,5,3,5,6,2,3,5,4,5,4,3,3,8,8,7];

%% Extract 18 seconds per fly
LFP_data = struct;

count = 0;

times = cell(length(fly_list), 1);
LFPs = cell(size(times));

start_sleeps = nan(size(times));
end_sleeps = nan(size(times));
idx_starts = nan(size(times));
idx_ends = nan(size(times));

for fly = 1:length(fly_list)
    
    %% Load fly data
    datasetname = fly_list{fly}
    epoch_path = ['..\..\02_processed_data\LFP\' datasetname '\trimmed_LFP_200Hz.mat'];
    if strcmp(datasetname, '14092018_SponSleep_LFP')
        resample_path = ['..\..\02_processed_data\LFP\' datasetname '\LFP\Analyzed_Block-1\' datasetname '_chunk_0' num2str(blocks_to_extract(fly)) '_1000Hz.mat'];
    else
        resample_path = ['..\..\02_processed_data\LFP\' datasetname '\LFP\Analyzed_Block-' num2str(blocks_to_extract(fly)) '\' datasetname '_chunk_0' num2str(blocks_to_extract(fly)) '_1000Hz.mat'];
    end
    
    disp('Loading epoch data.');
    load_val = 0;
    while load_val == 0
        try
            epoch = load(epoch_path);
            load_val = 1;
        catch
            load_val = 0;
            warning('Error loading data. Trying again in thirty seconds.')
            pause(10)
        end % try
    end % while
    
    epoch = epoch.trimmed_LFP.epoch;
    
    for x = 1:length(epoch)
        epoch(x).block_num = epoch(x).unique_block_num(1,1);
    end % 
    
    blocks = extractfield(epoch, 'block_num');
    index = 1:length(blocks);
    sleep_block = index(...
        blocks == blocks_to_extract(fly) &...
        [epoch.epoch_dur_mins] >= 5 &... % at least 5 minutes
        strcmp('Inactive', {epoch.status}) &... % make sure it's sleeping
        cellfun(@(x) length(x)==1, {epoch.unique_block_num})); % make sure the block doesn't go into another block
    
    if strcmp(datasetname, '14092018_SponSleep_LFP') | strcmp(datasetname, '13112018_SponSleep_LFP') | strcmp(datasetname, '13032019_SponSleep_LFP')
        % RJ's selection
        block_select = sleep_block(2);
    else
        % Try get longest block which meets criteria
        [max_dur, b] = max([epoch(sleep_block).epoch_dur_mins]);
        block_select = sleep_block(b);
        
        % Get the first block which meets criteria
        %block_select = sleep_block(1);
    end
    
    epoch2 = epoch; % backup variable for debugging
    epoch = epoch(block_select);
    disp(['fly ' num2str(fly) ': selected block length ' num2str(epoch.epoch_dur_mins) 'mins']);
    
    %% remove fields
    fields_to_remove = {'start_unix_idx', 'end_unix_idx', 'vid_frames', 'contours','delta_pixels','delta_pixels_prop', 'mean_contours','mean_prop_pixels', 'delta_pixels','delta_pixels_prop', 'lfp_start','lfp_end','lfp_idx_start','lfp_idx_end'};
    epoch = rmfield(epoch,fields_to_remove);
    
    %% Load 1000Hz data    
    disp('Loading 1000Hz data')
    load_val = 0;
    while load_val == 0
        try
            LFP = load(resample_path);
            load_val = 1;
        catch
            load_val = 0;
            warning('Error loading data. Trying again in thirty seconds.')
            pause(10)
        end % try
    end % while
    
    times{fly} = LFP.EEG.epoch_times;
    LFPs{fly} = LFP.EEG.data;
    
    start_sleeps(fly) = epoch.start_unix;
    end_sleeps(fly) = epoch.end_unix;
    
    [val,idx] = min(abs(times{fly} - start_sleeps(fly)));
    idx_starts(fly) = idx;
    [val, idx_end] = min(abs(times{fly} - end_sleeps(fly)));
    idx_ends(fly) = idx_end;
    
    clear LFP epoch epoch2
    
end

%%

eighteen = 18*1000; % 18 seconds at 1000Hz
minute = 60 * 1000; % 1 minutes 1 1000Hz
fiveMinutes = 60 * 5 * 1000; % 5 minutes at 1000Hz
tenMinutes = 60 * 10 * 1000; % 10 minutes at 1000Hz

%%

LFPs_raw = LFPs;

%%
% reset
LFPs = LFPs_raw;

%% Bipolar rereference

for fly = 1 : length(fly_list)
    tic;
    LFPs{fly} = LFPs{fly}(1:end-1, :) - LFPs{fly}(2:end, :);
    toc
end

%% Sum squared voltages over 18s as proxy measure of movement

window_sum = 1;

if window_sum == 1
	for fly = 1 : length(fly_list)
		tic;
		LFPs{fly} = movsum(LFPs{fly}.^2, eighteen, 2);
		toc
	end
end
%%

for fly = 1 : length(fly_list)
	tic;
	LFPs{fly} = movsum(LFPs{fly}, eighteen, 2);
	toc
end

%% Plot data from each fly

%{

fig = figure('visible', 'on', 'Units', 'normalized', 'Position', [0 0 1 1], 'Color', 'w');

for fly = 1 : length(fly_list)
    
    idx = idx_starts(fly);
    idx_end = idx_ends(fly);
    
    periods = {'wakeEarly', 'wake', 'sleepEarly', 'sleepLate'};
    period_idx = {...
        idx-(1*minute)-eighteen:idx-(1*minute),...
        idx-eighteen:idx,...
        idx+(2*minute)-eighteen:idx+(2*minute),...
        idx_end-(1*minute)-eighteen:idx_end-(1*minute)};
    
    % If sleepLate period starts before 5 minutes,
    %   get period immediately from 5 minutes
    if idx_end-(1*minute)-eighteen < idx+5*minute
        period_idx{end} = idx+5*minute : idx+5*minute+eighteen;
    end
    
    figure(fig);
    subplot(5, 4, fly);
    imagesc(LFPs{fly}); c = colorbar;
    ylabel('ch');
    xlabel('t');
    title(c, '\SigmaV^2');
    title(['fly' num2str(fly) ' ' datasetname], 'interpreter', 'none')
    set(gca, 'XTick', [idx-fiveMinutes idx idx+fiveMinutes idx+tenMinutes]);
    set(gca, 'XTickLabel', {'-5mins', '0mins', '5mins', '10mins'});
    hold on;
    channels = [0.5 size(LFPs{fly}, 1)+0.5];
    lwidth = 1;
    for p = 1 : length(periods)
        line([period_idx{p}(1) period_idx{p}(1)], channels, 'LineWidth', lwidth, 'Color', 'r');
        line([period_idx{p}(end) period_idx{p}(end)], channels, 'LineWidth', lwidth, 'Color', 'r');
    end
    xtickangle(50);
    
end

%}

%% Check correlation among channels

fig = figure('visible', 'on', 'Units', 'normalized', 'Position', [0 0 1 1], 'Color', 'w');

for fly = 1 : length(fly_list)
    
    [r, p] = corr(LFPs{fly}');
    
    figure(fig);
    subplot(5, 4, fly);
    imagesc(r, [-1 1]); c = colorbar;
    ylabel('channel');
    xlabel('channel');
    ylabel(c, 'r');
    title(['fly' num2str(fly) ' ' datasetname], 'interpreter', 'none')
    axis square
end

%% Align time-series around t=0

% Shift everything to the latest t=0
offsets = max(idx_starts) - idx_starts;

% Figure out total duration needed for all blocks after shifting
endpoints = cellfun('size', LFPs, 2) + offsets;

% time x channels x flies
% false() is to reduce memory by not making a double-precision matrix
LFP_mat = single(false(max(endpoints), size(LFPs{1}, 1),  length(fly_list)));
LFP_mat(:) = nan;

for fly = 1 : length(fly_list)
    tic;
    LFP_mat(offsets(fly)+1 : endpoints(fly), :, fly) = single(LFPs{fly}');
    toc
end

%% PCA across channels
% Note - we can't do PCA using all flies at the same time
%   because duration of recordings are variable across flies
% So, need to do PCA per fly
% So, PCs (PC coefficients) may not be consistent across flies

% Be aware of memory limits

do_pca = 0;

if do_pca == 1

	nPCs = 2; % store only data for this number of PCs
	LFP_mat_pca = single(nan(max(endpoints), nPCs,  length(fly_list)));
	coeffs = nan(size(LFPs{1}, 1), size(LFPs{1}, 1), length(fly_list));
	latents = nan(size(LFPs{1}, 1), length(fly_list));
	for fly = 1 : length(fly_list)
		tic;
		[coeff, score, latent] = pca(LFP_mat(:, :, fly));
		toc
		LFP_mat_pca(offsets(fly)+1 : endpoints(fly), :, fly) =...
			score(offsets(fly)+1 : endpoints(fly), 1:nPCs);
		coeffs(:, :, fly) = coeff;
		latents(:, fly) = latent;
	end

end

%% Plot timeseries altogether as matrix in a image plot

fig = figure('visible', 'on', 'Units', 'normalized', 'Position', [0 0 1 1], 'Color', 'w');

flyOrder = (1:length(fly_list));
% Sort flies by when inactivity ends
[~, flyOrder] = sort(idx_ends + offsets);

plot_type = 'mean'; % 'ch'; 'mean'; 'pc'
switch plot_type
    case 'ch'
        ch = 6;
        imagesc(permute(LFP_mat(:, ch, flyOrder), [3 1 2]));
        title(['ch' num2str(ch)]);
    case 'mean'
        imagesc(permute(mean(LFP_mat(:, :, flyOrder), 2), [3 1 2]));
        title(['mean across channels']);
    case 'pc'
        pc = 1;
        imagesc(permute(LFP_mat_pca(:, pc, flyOrder), [3 1 2]));
        title(['PC' num2str(pc)]);
end

c = colorbar;
title(c, '\SigmaV^2');
hold on;

% Add lines

for f = 1 : length(flyOrder)
    fly = flyOrder(f);
    
    idx = idx_starts(fly);
    idx_end = idx_ends(fly);
    
    periods = {'wakeEarly', 'wake', 'sleepEarly', 'sleepLate'};
    period_idx = {...
        idx-(1*minute)-eighteen:idx-(1*minute),...
        idx-eighteen:idx,...
        idx+(2*minute)-eighteen:idx+(2*minute),...
        idx_end-(1*minute)-eighteen:idx_end-(1*minute)};
	
	periods = {'wake', 'sleepEarly'};
    period_idx = {...
        %idx-(1*minute)-eighteen:idx-(1*minute),...
        idx-eighteen:idx,...
        idx+(2*minute)-eighteen:idx+(2*minute),...
        %idx_end-(1*minute)-eighteen:idx_end-(1*minute)
		};
    
	if length(periods) == 4
    	% If sleepLate period starts before 5 minutes,
    	%   get period immediately from 5 minutes
    	if idx_end-(1*minute)-eighteen < idx+5*minute
        	period_idx{end} = idx+5*minute : idx+5*minute+eighteen;
    	end
	end
	
    % Plot extracted periods
    ys = [0.5 1.5] + f-1;
    lwidth = 1;
    for p = 1 : length(periods)
        line([period_idx{p}(1) period_idx{p}(1)]+offsets(fly), ys, 'LineWidth', lwidth, 'Color', 'r');
        line([period_idx{p}(end) period_idx{p}(end)]+offsets(fly), ys, 'LineWidth', lwidth, 'Color', 'r');
    end
    
    % Plot end of fly inactivity
    line([idx_end idx_end]+offsets(fly), ys, 'LineWidth', lwidth, 'Color', 'm');
    
    % Plot start and end of blocks
    line([offsets(fly)+1 offsets(fly)+1], ys, 'LineWidth', lwidth, 'Color', 'm', 'LineStyle', ':');
    line([endpoints(fly) endpoints(fly)], ys, 'LineWidth', lwidth, 'Color', 'm', 'LineStyle', ':');
    
end

% Add axis labels
[~, ref_fly] = min(offsets);
fly = ref_fly;
idx = idx_starts(fly);
idx_end = idx_ends(fly);

periods = {'wakeEarly', 'wake', 'sleepEarly', 'sleepLate'};

set(gca, 'XTick', [idx-fiveMinutes idx idx+fiveMinutes idx+tenMinutes]);
set(gca, 'XTickLabel', {'-5mins', '0mins', '5mins', '10mins'});
xtickangle(-25);

set(gca, 'YTick', (1:length(flyOrder)), 'YTickLabel', flyOrder);

ylabel('fly (sorted by end of inactivity)');
xlabel('t');

%% Print figure

print_fig = 0;

if print_fig == 1
	
	axis tight
	box on
	
	figure_name = '../../../../figures_stage2/fig_sleepPeriods';
	
	set(gcf, 'PaperOrientation', 'Portrait');
	
	print(figure_name, '-dsvg', '-painters'); % SVG
	print(figure_name, '-dpdf', '-painters');%, '-bestfit'); % PDF
	print(figure_name, '-dpng'); % PNG
	
end

%% Plot timeseries altogether, as individual lines
% And plot the average (where there all overlap in time)

fig = figure('visible', 'on', 'Units', 'normalized', 'Position', [0 0 0.75 0.75], 'Color', 'w');
hold on;

flyOrder = (1:length(fly_list));
% Sort flies by when inactivity ends
[~, flyOrder] = sort(idx_ends + offsets);

plot_type = 'ch'; % 'ch'; 'mean'; 'pc'
switch plot_type
    case 'ch'
        ch = 6;
        plot_map = permute(LFP_mat(:, ch, flyOrder), [3 1 2]);
        title(['ch' num2str(ch)]);
    case 'mean'
        plot_map = permute(mean(LFP_mat(:, :, flyOrder), 2), [3 1 2]);
        title(['mean across channels']);
    case 'pc'
        pc = 1;
        plot_map = permute(LFP_mat_pca(:, pc, flyOrder), [3 1 2]);
        title(['PC' num2str(pc)]);
end

% Normalise each fly
%   So value ranges are the same across flies
%plot_map = (plot_map - mean(plot_map, 2, 'omitnan')) ./ std(plot_map, [], 2, 'omitnan');

% Plot average across flies
% mean() will drop times if any of the flies' time-series stops (ie is nan)
% nanmean() will drop times if ALL of the flies' time-series are nan
plot_map_mean = nanmean(plot_map, 1);
%plot_map_mean = trimmean(plot_map, 100*(2/length(fly_list)), 1);

% NOTE - plotting std as a patch takes a LONG TIME
%   Every update to the figure (from resizing the figure window) takes a
%       long time
%plot_map_std = nanstd(plot_map, [], 1);
%plot(plot_map_mean, 'k');
%plot(plot_map_mean+plot_map_std/2, 'Color', [0 0 0 0.1]);
%plot(plot_map_mean-plot_map_std/2, 'Color', [0 0 0 0.1]);
patch(...
    [(1:length(plot_map_mean)) (length(plot_map_mean):-1:1)],...
    [plot_map_mean+plot_map_std/2 fliplr(plot_map_mean-plot_map_std/2)],...
    'k', 'FaceAlpha', 0.3, 'EdgeColor', 'none');

%plot((1:length(plot_map_mean)), plot_map_mean, 'k', 'LineWidth', 3);

% Find time range where all flies have data
plot_map_all = find(all(~isnan(plot_map), 1));

% Plot for where all flies have data
plot(plot_map_all, plot_map_mean(plot_map_all), 'k', 'LineWidth', 3);
% Plot before and after
plot((1:plot_map_all(1)-1), plot_map_mean(1:plot_map_all(1)-1), 'Color', [0 0 0 0.5], 'LineWidth', 1);
plot((plot_map_all(end)+1:length(plot_map_mean)), plot_map_mean(plot_map_all(end)+1:end), 'Color', [0 0 0 0.5], 'LineWidth', 1);

[~, ref_fly] = min(offsets);
fly = ref_fly;
idx = idx_starts(fly);
idx_end = idx_ends(fly);
set(gca, 'XTick', [idx-fiveMinutes idx idx+fiveMinutes idx+tenMinutes]);
set(gca, 'XTickLabel', {'-5mins', '0mins', '5mins', '10mins'});

% Plot for each fly
fly_colours = parula(length(flyOrder));
for f = 1 : length(flyOrder)
    fly = flyOrder(f);
    
    idx_ref = idx_starts(ref_fly);
    idx = idx_starts(fly);
    idx_end = idx_ends(fly);
    
    % Plot from the start to the end of epoch
    range_start = idx_ref-(1*minute)-eighteen;
    range_end = idx_end+offsets(fly);
    plot((range_start:range_end), plot_map(f, range_start:range_end), 'Color', [fly_colours(f, :) 0.9])
    
    % Plot before the epoch (rest of block)
    range_start = 1;
    range_end = idx_ref-(1*minute)-eighteen-1;
    plot((range_start:range_end), plot_map(f, range_start:range_end), 'Color', [fly_colours(f, :) 0.2])
    
    % Plot after the epoch (rest of block)
    range_start = idx_end+1+offsets(fly);
    range_end = size(plot_map, 2);
    plot((range_start:range_end), plot_map(f, range_start:range_end), 'Color', [fly_colours(f, :) 0.2])
    
    % Indicate the end of marked inactivity
    scatter(idx_end+offsets(fly), 0, 100, fly_colours(f, :));
    line([idx_end+offsets(fly) idx_end+offsets(fly)], [0 plot_map(f, idx_end+offsets(fly))], 'Color', 'k');
    
    %plot((1:size(plot_map, 2)), plot_map(fly, :), 'Color', [0 0 0 0.2]);
end

axis tight