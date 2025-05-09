%%

%{

Create figure showing distribution of hctsa values across all time-series

Average (nanmean) across channels

Flip directions and combine in one plot

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

%% Load feature thresholds and directions

thresh_dir = ['classification_nearestMean/results' preprocess_string filesep];
thresh_file = 'class_nearestMedian_thresholds.mat';
load([thresh_dir thresh_file]);

%% Flip values based on direction

for d = 1 : length(hctsas)
    
    for ch = 1 : size(hctsas{d}.TS_DataMat, 3)
        tic;
        for f = 1 : size(hctsas{d}.TS_DataMat, 2)
            
            if directions(ch, f) == 0
                
                % Flip
                hctsas{d}.TS_DataMat(:, f, ch) = hctsas{d}.TS_DataMat(:, f, ch) * -1;
                
            end
            
        end
        toc
    end
    
end

disp('flipped values');
direction_labels = {'W > A'};

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

%% Average across channels
% Average AFTER replacing invalid features per channel with NaNs and
%   after flipping feature values based on direction

% Scale before or after normalising values?
for d = 1 : length(hctsas)
    tic;
    hctsas{d}.TS_Normalised = mean(hctsas{d}.TS_Normalised, 3, 'omitnan');
    toc
end

disp('averaged across channels');

%% Exclude rows corresponding to particular unused conditions                     
%{
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

%% Normalise data
% Normalisation per dataset

% Note - even with mixedSigmoid, some features scale to NaNs and 0s
%   discovery flies - feature 976 (870th valid feature) scales
%       to NaN a nd 0s
%   sleep flies - feature 977 scales to NaN and a 0

for d = 1 : length(hctsas)
    disp(['scaling dataset ' dsets{d}]);
    tic;
    hctsas{d}.TS_Normalised = nan(size(hctsas{d}.TS_DataMat));
    hctsas{d}.TS_Normalised(valid_rows{d}, :) =...
        BF_NormalizeMatrix(hctsas{d}.TS_DataMat(valid_rows{d}, :), 'mixedSigmoid');
    t = toc;
    disp(['scaled in ' num2str(t) 's']);
end

%}

%% Group main condition pair comparisons together
% Plot certain combinations of datasets and conditions

% 1 - discovery wake vs anest
% 2 - multidose8 iso1.2 + multidose4 iso1.2 + singledose iso0.6 vs wake
% 3 - sleep vs wake

pair_datasets = cell(3, 1);
pair_conditions = cell(3, 2);

pair_datasets{1} = {'train'};
pair_conditions{1, 1} = {'condition1'};
pair_conditions{1, 2} = {'condition2'};

pair_datasets{2} = {'multidose8', 'multidose4', 'singledose'};
pair_conditions{2, 1} = {'conditionWake', 'conditionWake', 'conditionWake'};
pair_conditions{2, 2} = {'conditionIsoflurane_1.2', 'conditionIsoflurane_1.2', 'conditionIsoflurane'};

pair_datasets{3} = {'sleep'};
pair_conditions{3, 1} = {'conditionwake'};
pair_conditions{3, 2} = {'conditionsleepEarly'};

% Get features which are signficant in discovery flies
%   significant in any of the channels
ref_dset = 'train';
perf_type = 'consis';
sig_features = any(stats.(ref_dset).(perf_type).sig, 1);
selected_features = sig_features & any(valid_all, 1);

fig = figure('Color', 'w');
hold on;

% x-positions of each condition in each condition grouping
cond_xs = {[1 2], [4 5], [7 8]};
cond_xticks = [cond_xs{:}];
cond_xtick_labels = {'train-wake', 'train-iso',...
    'md8:md4:sd-wake', 'md8:md4:sd-iso',...
    'sleep-wake', 'sleep-sleep'};

d_markers = {'.', 'v', '^', '.', 'o'};
cond_colours = {'r', 'b'};

for p = 1 : length(pair_datasets)
    
    for c = 1 : size(pair_conditions, 2)
        
        ys = nan(length(find(selected_features)), length(pair_datasets{p}));
        
        % Randomly spread points around a given x-position
        a = -1/4; b = 1/4;
        xspread = a + (b-a).*rand(size(ys, 1), 1);
        
        alpha_val = 1000/numel(ys);
        alpha_val = 0.1;
        
        % Get feature values from the specified datasets and conditions
        for d = 1 : length(pair_datasets{p})
            dset = pair_datasets{p}{d};
            dset_id = find(strcmp(dset, dsets));
            
            % Get normalised values
            values = hctsas{dset_id}.TS_Normalised(:, selected_features);
            
            % Identify rows for the condition
            keys = pair_conditions{p, c}{d};
            rows = getIds({keys}, hctsas{dset_id}.TimeSeries);
            
            ys(:, d) = median(values(rows, :), 1); % medians per feature
            
            % Show individual feature medians for all datasets
            scatter(cond_xs{p}(c)+xspread, ys(:, d), 10, cond_colours{c}, d_markers{dset_id},...
                'MarkerEdgeAlpha', alpha_val);
            
        end
        
        % Plot box-plot across feature medians
        h = boxplot(ys(:), 'Positions', cond_xs{p}(c), 'Widths', 0.5,...
            'Whisker', 100, 'Colors', 'k'....
            );
        set(h, {'linew'}, {1.5}); % box plot line widths
        
    end
    
end

xlim([min([cond_xs{:}])-0.5 max([cond_xs{:}])+0.5]);
ylim([0 1]); % possible range of scaled values

[~, xtick_order] = sort(cond_xticks);
set(gca, 'XTick', cond_xticks(xtick_order));
set(gca, 'XTickLabel', cond_xtick_labels(xtick_order));
xtickangle(gca, 15);

title(['ch-mean' newline...
    'sig. and ' direction_labels{1} ' (opp. flipped) in discovery flies' newline...
    'main comparisons']);

%% Group main condition pair comparisons together
% Plot certain combinations of datasets and conditions
% Plot feature points separately per dataset
%   but plot boxplots combining across datasets

% 1 - discovery wake vs anest
% 2 - multidose8 iso1.2 + multidose4 iso1.2 + singledose iso0.6 vs wake
% 3 - sleep vs wake

pair_datasets = cell(3, 1);
pair_conditions = cell(3, 2);

pair_datasets{1} = {'train'};
pair_conditions{1, 1} = {'condition1'};
pair_conditions{1, 2} = {'condition2'};

pair_datasets{2} = {'multidose8', 'multidose4', 'singledose'};
pair_conditions{2, 1} = {'conditionWake', 'conditionWake', 'conditionWake'};
pair_conditions{2, 2} = {'conditionIsoflurane_1.2', 'conditionIsoflurane_1.2', 'conditionIsoflurane'};

pair_datasets{3} = {'sleep'};
pair_conditions{3, 1} = {'conditionwake'};
pair_conditions{3, 2} = {'conditionsleepEarly'};

% Get features which are signficant in discovery flies
%   significant in any of the channels
ref_dset = 'train';
perf_type = 'consis';
sig_features = any(stats.(ref_dset).(perf_type).sig, 1);
selected_features = sig_features & any(valid_all, 1);

fig = figure('Color', 'w');
hold on;

% x-positions of each condition in each condition grouping
dist_xs = {[1 1.75], linspace(3, 6, 6), [7.25 8]};
cond_xs = {[1 1.75], [3.6 5.4], [7.25 8]};
box_widths = {[0.5 0.5], [1 1], [0.5 0.5]};
cond_xticks = [cond_xs{:}];
cond_xtick_labels = {'train-wake', 'train-iso',...
    'md8:md4:sd-wake', 'md8:md4:sd-iso',...
    'sleep-wake', 'sleep-sleep'};

d_markers = {'.', '.', '.', '.', '.'};
cond_colours = {'r', 'b'};

for p = 1 : length(pair_datasets)
    
    for c = 1 : size(pair_conditions, 2)
        
        ys = nan(length(find(selected_features)), length(pair_datasets{p}));
        
        % Randomly spread points around a given x-position
        a = -1/4; b = 1/4;
        xspread = a + (b-a).*rand(size(ys, 1), 1);
        
        alpha_val = 0.1;
        
        % Get feature values from the specified datasets and conditions
        for d = 1 : length(pair_datasets{p})
            dset = pair_datasets{p}{d};
            dset_id = find(strcmp(dset, dsets));
            
            % Get normalised values
            values = hctsas{dset_id}.TS_Normalised(:, selected_features);
            
            % Identify rows for the condition
            keys = pair_conditions{p, c}{d};
            rows = getIds({keys}, hctsas{dset_id}.TimeSeries);
            
            ys(:, d) = median(values(rows, :), 1); % medians per feature
            
            % Show individual feature medians for the dataset
            scatter(dist_xs{p}(d+((c-1)*length(pair_datasets{p})))+xspread, ys(:, d), 10, cond_colours{c}, d_markers{dset_id},...
                'MarkerEdgeAlpha', alpha_val);
            
        end
        
        % Plot box-plot across feature medians
        h = boxplot(ys(:), 'Positions', cond_xs{p}(c), 'Widths', box_widths{p}(c),...
            'Whisker', 100, 'Colors', 'k'....
            );
        set(h, {'linew'}, {1.5}); % box plot line widths
        
    end
    
end

xlim([min([cond_xs{:}])-0.5 max([cond_xs{:}])+0.5]);
ylim([0 1]); % possible range of scaled values

[~, xtick_order] = sort(cond_xticks);
set(gca, 'XTick', cond_xticks(xtick_order));
set(gca, 'XTickLabel', cond_xtick_labels(xtick_order));
xtickangle(gca, 15);

title(['ch-mean' newline...
    'sig. and ' direction_labels{1} ' (opp. flipped) in discovery flies' newline...
    'main comparisons']);

%% Group main condition pair comparisons together
% Plot differences in medians between conditions
% Plot certain combinations of datasets and conditions
% Plot feature points separately per dataset
%   but plot boxplots combining across datasets

% 1 - discovery wake vs anest
% 2 - multidose8 iso1.2 + multidose4 iso1.2 + singledose iso0.6 vs wake
% 3 - sleep vs wake

pair_datasets = cell(3, 1);
pair_conditions = cell(3, 2);

pair_datasets{1} = {'train'};
pair_conditions{1, 1} = {'condition1'};
pair_conditions{1, 2} = {'condition2'};

pair_datasets{2} = {'multidose8', 'multidose4', 'singledose'};
pair_conditions{2, 1} = {'conditionWake', 'conditionWake', 'conditionWake'};
pair_conditions{2, 2} = {'conditionIsoflurane_1.2', 'conditionIsoflurane_1.2', 'conditionIsoflurane'};

pair_datasets{3} = {'sleep'};
pair_conditions{3, 1} = {'conditionwake'};
pair_conditions{3, 2} = {'conditionsleepEarly'};

% Get features which are signficant in discovery flies
%   significant in any of the channels
ref_dset = 'train';
perf_type = 'consis';
sig_features = any(stats.(ref_dset).(perf_type).sig, 1);
selected_features = sig_features & any(valid_all, 1);

fig = figure('Color', 'w');
hold on;

% x-positions of each condition in each condition grouping
dist_xs = {[1] [3.4 4 4.6], [7]};
cond_xs = {[1], [4], [7]};
box_widths = {[0.5], [1], [0.5]};
cond_xticks = [cond_xs{:}];
cond_xtick_labels = {'train:wake-iso',...
    'md8/md4/sd:wake-iso',...
    'sleep:wake-sleep'};

d_markers = {'.', '.', '.', '.', '.'};
cond_colours = {'r', 'b'};

for p = 1 : length(pair_datasets)
    
    ys = nan(length(find(selected_features)), length(pair_datasets{p}));
    
    % Randomly spread points around a given x-position
    a = -1/4; b = 1/4;
    xspread = a + (b-a).*rand(size(ys, 1), 1);
    
    alpha_val = 0.1;
    
    % Get feature values from the specified datasets and conditions
    for d = 1 : length(pair_datasets{p})
        dset = pair_datasets{p}{d};
        dset_id = find(strcmp(dset, dsets));
        
        % Get normalised values
        values = hctsas{dset_id}.TS_Normalised(:, selected_features);
        
        rows = cell(size(pair_conditions, 2));
        for c = 1 : size(pair_conditions, 2)
            % Identify rows for the condition
            keys = pair_conditions{p, c}{d};
            rows{c} = getIds({keys}, hctsas{dset_id}.TimeSeries);
        end
        
        ys(:, d) = median(values(rows{1}, :), 1) - median(values(rows{2}, :), 1); % medians per feature
        
        % Show individual feature medians for the dataset
        scatter(dist_xs{p}(d)+xspread, ys(:, d), 10, 'k', d_markers{dset_id},...
            'MarkerEdgeAlpha', alpha_val);
        
    end
    
    % Plot box-plot across feature medians
    h = boxplot(ys(:), 'Positions', cond_xs{p}(1), 'Widths', box_widths{p},...
        'Whisker', 100, 'Colors', 'r'....
        );
    set(h, {'linew'}, {1.5}); % box plot line widths
    
end

axis tight
xlim([min([cond_xs{:}])-0.5 max([cond_xs{:}])+0.5]);

[~, xtick_order] = sort(cond_xticks);
set(gca, 'XTick', cond_xticks(xtick_order));
set(gca, 'XTickLabel', cond_xtick_labels(xtick_order));
xtickangle(gca, 15);

title(['ch-mean' newline...
    'sig. and ' direction_labels{1} ' (opp. flipped) in discovery flies' newline...
    'main comparisons (diff. in medians)']);

line(xlim, [0 0], 'Color', 'k');

%% Plot features and boxplots
% Plot all datasets + conditions, separately

% Find all features which are significant in the discovery flies
% Normalise these features
% Separate features based on trained direction
% Show distribution of normalised vales at each condition

perf_type = 'consis';
ref_dset = 'train';

% Find features which are significant in discovery flies
sig_features = stats.(ref_dset).(perf_type).sig(ch, :);
%TODO - significant in any of the channels

fig = figure('Color', 'w');

% x-positions of each condition in each dataset
%    discovery; multidose8; multidose4; singledose; sleep
cond_xs = {[1 2], [8 9 10 11 12], [14 15 16 17 18], [4 5 6], [20 21 22 23]};
%cond_xticks = cellfun(@mean, cond_xs);
cond_xticks = [cond_xs{:}];
cond_xtick_labels = cell(size(cond_xticks));
cond_xtick_counter = 1;
d_markers = {'.', '.', '.', '.', '.'};
dset_order = [1 4 2 3 5];

hold on;

% Get features which are sig
selected_features = sig_features & any(valid_all, 1);

% Randomly spread points around a given x-position
a = -1/4; b = 1/4;
xspread = a + (b-a).*rand(numel(find(selected_features)), 1);

% Get feature values - median across epochs and flies
for d = 1 : length(dsets)
    values = hctsas{d}.TS_Normalised(:, selected_features); % normalised values
    
    [conditions, cond_labels, cond_colours] = getConditions(dsets{d});
    
    for c = 1 : length(conditions)
        cond_xtick_labels{cond_xtick_counter} = conditions{c};
        cond_xtick_counter = cond_xtick_counter+1;
        
        % Identify rows for the condition
        keys = {conditions{c}};
        rows = getIds(keys, hctsas{d}.TimeSeries);
        
        ys = median(values(rows, :), 1); % medians per feature
        
        % Show individual medians for all features
        scatter(cond_xs{d}(c)+xspread, ys, 5, cond_colours{c}, d_markers{d},...
            'MarkerEdgeAlpha', 0.5);
        
        % Plot box-plot across feature medians
        h = boxplot(ys, 'Positions', cond_xs{d}(c), 'Widths', 0.5,...
            'Whisker', 100, 'Colors', 'k'....
            );
        set(h, {'linew'}, {1.5}); % box plot line widths
    end
    
end

axis tight;
xlim([min([cond_xs{:}])-0.5 max([cond_xs{:}])+0.5]);
ylim([0 1]); % possible range of scaled values

ylabel('median feature value');

[~, xtick_order] = sort(cond_xticks);
set(gca, 'XTick', cond_xticks(xtick_order));
set(gca, 'XTickLabel', cond_xtick_labels(xtick_order));
xtickangle(gca, 15);
xlabel(strjoin(dsets(dset_order), ' : '));

title(['ch-mean' newline...
    'sig. and ' direction_labels{1} ' (opp. flipped) in discovery flies' newline...
    strjoin(dsets(dset_order), ' : ')]);
