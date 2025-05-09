%%

%{

Create figure showing distribution of hctsa values across all time-series

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

data_sources = {'train', 'multidose', 'singledose', 'sleep'};

source_dir = ['hctsa_space' preprocess_string '/'];

hctsas = cell(size(data_sources));
for d = 1 : length(data_sources)
    
    source_file = ['HCTSA_' data_sources{d} '_channel' num2str(ch) '.mat'];
    
    disp(['loading ' source_file]);
    tic;
    hctsas{d} = load([source_dir source_file]);
    t = toc;
    disp(['loaded in ' num2str(t) 's']);
end

%% Split multidose dataset into multidose8 and multidose4
disp('splitting multidose');
tic;
md_split = split_multidose(hctsas{2});

hctsas{5} = md_split{1};
hctsas{6} = md_split{2};

hctsas = hctsas([1 5 6 3 4]);
toc

%% Load feature thresholds and directions

thresh_dir = ['classification_nearestMean/results' preprocess_string filesep];
thresh_file = 'class_nearestMedian_thresholds.mat';
load([thresh_dir thresh_file]);

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

%%

% Find all features which are significant in the discovery flies
% Normalise these features
% Separate features based on trained direction
% Show distribution of normalised vales at each condition

perf_type = 'consis';
ref_dset = 'train';

% Find features which are significant in discovery flies
sig_features = stats.(ref_dset).(perf_type).sig(ch, :);

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

direction_labels = {'W < A', 'W > A'};

sp_counter = 1;
for direction = unique(directions(:))'

    subplot(length(unique(directions(:))), 1, sp_counter);
    hold on;
    
    % Get features which are sig and match the direction
    selected_features = sig_features & valid_all(ch, :) & (directions(ch, :) == direction);
    
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
    
    ylabel('median feature value');
    
    [~, xtick_order] = sort(cond_xticks);
    set(gca, 'XTick', cond_xticks(xtick_order));
    set(gca, 'XTickLabel', cond_xtick_labels(xtick_order));
    xtickangle(gca, 15);
    xlabel(strjoin(dsets(dset_order), ' : '));
    
    title(['ch' num2str(ch) newline...
        direction_labels{direction+1} ' in discovery flies' newline...
        strjoin(dsets(dset_order), ' : ')]);
    
    sp_counter = sp_counter + 1;
end