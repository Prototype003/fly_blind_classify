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

%% Normalise data per channel, per dataset
% Normalise across datasets or per dataset?
% Per dataset - all datasets will have the same range of values
% Across datasets - all datasets will be relative to discovery flies

% Normalisation per dataset

% Note - even with mixedSigmoid, some features scale to NaNs and 0s
%   discovery flies - feature 976 (870th valid feature) scales
%       to NaN a nd 0s
%   sleep flies - feature 977 scales to NaN and a 0

% Time to scale every channel, multidose8 - ~836s

for d = 1 : 1%length(hctsas)
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
    nRows = nRows + size(hctsas{d}.TS_DataMat, 1);
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


%% Concatenate hctsa matrices and normalise across all datasets
% Note - multidose dataset has thousands of rows
%   Meanwhile, discovery, singledose, and sleep sets have hundreds of rows
%{
% Which datasets to concatenate together
cat_sets = (1:length(hctsas));

% Set to 0 if other datasets are to be scaled based on the first dataset
% Note, if scaling with a reference dataset, values in other datasets may
%   be outside the range of 0-1
scale_together = 0;
disp('concatenating');
tic;

hctsas_all = [];
dset_rowStarts = nan(length(cat_sets)+1, 1);
row_counter = 1;
dims = nan(length(cat_sets), 3); % Assumes TS_DataMat has 3 dimensions
for d = cat_sets
    tmp = hctsas{d}.TS_DataMat(valid_rows{d}, :, :);
    dims(d, :) = size(tmp);
    tmp = permute(tmp, [1 3 2]);
    tmp = reshape(tmp, [dims(d, 1)*dims(d, 3) dims(d, 2)]);
    hctsas_all = cat(1, hctsas_all, tmp);
    dset_rowStarts(d) = row_counter;
    row_counter = row_counter + numel(find(valid_rows{d}))*dims(d, 3);
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
    reference_rows((1:size(hctsas{ref_dset}.TS_DataMat, 1)*dims(ref_dset, 3))) = 1;
    hctsas_all_normalised = BF_NormalizeMatrix(hctsas_all, 'mixedSigmoid', reference_rows);
end
t = toc;
% Separate concatenated, matrix back out
for d = cat_sets
    tmp = hctsas_all_normalised(dset_rowStarts(d):dset_rowStarts(d+1)-1, :);
    tmp = reshape(tmp, [dims(d, 1) dims(d, 3) dims(d, 2)]);
    tmp = permute(tmp, [1 3 2]);
    hctsas{d}.TS_Normalised = nan(size(hctsas{d}.TS_DataMat));
    hctsas{d}.TS_Normalised(valid_rows{d}, :, :) = tmp;
end
disp(['scaled in ' num2str(t) 's']);
%}
%% Average across flies and epochs
% Average AFTER replacing invalid features per channel with NaNs

for d = 1 : length(hctsas)
    tic;
    
    dset = dsets{d};
    dbase = dset_bases{d};
    
    % Separate out flies into a separate dimension
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
    
    hctsas{d}.TS_NormalisedMean = nan(nCh, size(hctsas{d}.TS_Normalised, 2), length(conditions));
    
    % Because we're using all flies, we don't actually need to index
    % specific flies; also, number of epochs is same across flies, so can
    % use grand average
    %keys_flies = cellfun(@(x) strcat('fly', num2str(x)), num2cell(flies), 'UniformOutput', false);
    for c = 1 : length(conditions)
        keys = {conditions{c}};
        rows = getIds(keys, hctsas{d}.TimeSeries);
        
        % Average across epochs and flies
        hctsas{d}.TS_NormalisedMean(:, :, c) =...
            permute(...
                mean(hctsas{d}.TS_Normalised(rows, :, :), 1),...
                [3 2 1]); % channel x feature x condition
        
    end
    
    toc
end

disp('averaged across flies and epochs');

%% Join datasets together

% Row order
%ordering = 'dset_first'; % order by dataset, then condition within dataset
ordering = 'cond_first'; % order by condition, then dataset within condition

vis_mat = [];
dset_labels = {'D', 'MD8', 'MD4', 'SD', 'S'};
dset_cond_rows = cell(length(hctsas), 2);
switch ordering
    case 'dset_first'
        
        % Concatenate averaged values across datasets
        for d = 1 : length(hctsas)
            for c = 1 : size(hctsas{d}.TS_NormalisedMean, 3)
                vis_mat = cat(1, vis_mat, hctsas{d}.TS_NormalisedMean(:, :, c));
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

                ycounter = ycounter + nCh;
                
            end
            
        end
    
    case 'cond_first'
        % Note - assumes same number of conditions for all datasets
        
        % Concatenate averaged values across datasets
        for c = 1 : size(hctsas{1}.TS_NormalisedMean, 3)
            for d = 1 : length(hctsas)
                vis_mat = cat(1, vis_mat, hctsas{d}.TS_NormalisedMean(:, :, c));
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
                
                ycounter = ycounter + nCh;
            end
        end
        
    otherwise
        warning('datasets not joined');
end

vis_mat_all = vis_mat; % Keep this unaltered, clustering can take a while (~20 minutes)

%%

% Switch between all and any as needed, for clustering purpose
vis_mat = vis_mat_all(:, all(valid_all, 1));

%% Cluster features
% Cluster the averaged matrix

%{
disp('clustering features');
tic;
[fOrder, removed] = clusterFeatures(vis_mat);
toc

% Remove features from the full matrix
vis_mat(:, removed) = [];
%}

%% Cluster features
% Cluster a given dataset
% Note - can cluster including NaN values, but this takes a long time
%   This is because some features are valid for some channels but not
%       others
%   So, for faster clustering, cluster features which are valid across
%       all channels

cluster_dset = 1;

disp('clustering features');

tic;
cluster_data = hctsas{cluster_dset}.TS_Normalised(:, all(valid_all, 1), :);
dims = size(cluster_data);
[fOrder, removed] = clusterFeatures(...
    reshape(...
        permute(cluster_data, [1 3 2])...
        , [dims(1)*dims(3) dims(2)]...
    )...
);
toc

vis_mat(:, removed) = [];

%% Create figure

fig = figure;
handle = imagesc(vis_mat(:, fOrder));
cbar = colorbar;

%% Add axis ticks and other details

colormap inferno

title(cbar, 'value');

set(gca, 'TickLabelInterpreter', 'none');
yticks(ytickpos);
yticklabels(ytickstrings);
ylabel('channel 15-1');

xlabel('feature');

title('mean of normalised values, across flies and epochs');
