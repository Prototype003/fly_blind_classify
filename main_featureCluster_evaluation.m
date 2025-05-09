%% Description

%{

Cluster significant features based on similarity in feature values across
all epochs

Filter features based on significance 


Cluster per channel or across channels?
Note - different significant features per channel
But - very few, or very many, significant features per channel
(<100 across all channels for classification, but 100s per channel for
consistency)

If we cluster per channel - need to report per channel
If we cluster across all channels - only need to report one summary of
features

How can we cluster across channels? We can concatenate hctsa matrices
across channels - but what about if the feature is invalid for some
channel? Check - does this occur? Replace all values with nans for the
channel?

Correlate across all flies and epochs

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


%% Settings

perf_type = 'consis'; % 'nearestMedian', 'consis'

topN = 0; % 0 = don't restrict

%% Output location
%{
out_dir = [pwd filesep 'results' preprocess_string '_cluster_top' num2str(topN) filesep];
out_file = [perf_type '_clusters']; % file with cluster details

if ~exist(out_dir, 'dir'); mkdir(out_dir); end
%}

%% Get main comparison for each dataset

main_pairs = nan(size(dsets));

for d = 1 : length(dsets)
    [conds, cond_labels, cond_colours, stats_order, main] = getConditions(dsets{d});
    
    % account for different order/idxs in stats structure
    % wake condition always has lower index than wake
    main = sort(stats_order(main));
    disp(main);
    main = repmat(main, [size(stats.(dsets{d}).(perf_type).condition_pairs, 1) 1]);
    main_pairs(d) = find(all(main == stats.(dsets{d}).(perf_type).condition_pairs, 2));
end

%% Get number of significant features at each channel

figure;

sig_all = valid_all; % valid and significant

% Plot stepwise number of significant features as datasets are added
subplot(1, 2, 2);
for d = 1 : length(dsets)
    sig_all = sig_all & stats.(dsets{d}).(perf_type).sig(:, :, main_pairs(d));
	plot(sum(sig_all, 2)); hold on;
end
legend(dsets);
ylabel('N sig. features');
xlabel('channel');
xlim([1 15]);
title([perf_type newline 'stepwise']);

% Plot number of significant features for each dataset separately
subplot(1, 2, 1);
for d = 1 : length(dsets)
	plot(sum(stats.(dsets{d}).(perf_type).sig(:, :, main_pairs(d)), 2));
	hold on;
end
plot(sum(sig_all, 2), 'k');
legend(cat(2, dsets, {'all'}));
ylabel('N sig. features');
xlabel('channel');
xlim([1 15]);
title([perf_type newline 'sig. features per dataset']);

%%
% Plot proportion of significant features relative to discovery
figure;
ref_d = 1;
sig_ref = valid_all & stats.(dsets{ref_d}).(perf_type).sig(:, :, main_pairs(ref_d));
sig_ref_count = sum(sig_ref, 2);
for d = 1 : length(dsets)
	tmp = sig_ref & stats.(dsets{d}).(perf_type).sig(:, :, main_pairs(d));
	plot(sum(tmp, 2) ./ sig_ref_count); hold on;
end
legend(dsets);
ylabel('prop. sig. features');
xlabel('channel');
xlim([1 15]);
title([perf_type newline 'prop. sig. features relative to discovery']);
%%
% Plot weighted mean across datasets
% Weigh datasets by number of flies in the dataset
figure;
sig_sum = zeros(size(valid_all, 1), 1);
total_flies = 0;
for d = 1 : length(dsets)
	[nChannels, nFlies, nConds, nEpochs] = getDimensionsFast(dsets{d});
	sig_sum = sig_sum + sum(stats.(dsets{d}).(perf_type).sig(:, :, main_pairs(d)), 2) * nFlies;
	total_flies = total_flies + nFlies;
end
plot(sig_sum ./ total_flies);
ylabel('N sig. features');
xlabel('channel');
xlim([1 15]);
title([perf_type newline 'mean sig. features across datasets (weighted by n flies)']);

%% Filter features and concatenate datasets together

% Consider all channels together
fIds = find(any(sig_all, 1));

% Count how many total rows we need
hctsas_rows = nan(length(hctsas), 1);
for d = 1 : length(dsets)
    hctsas_rows(d) = size(hctsas{d}.TS_DataMat, 1);
end
nRows = sum(hctsas_rows);

% Concatenate TS_DataMats together, for the given features
hctsas_cat = nan(nRows, numel(fIds), size(sig_all, 1)); % epochs x features x channels
row = 1;
for d = 1 : length(dsets)
    hctsas_cat(row : sum(hctsas_rows(1:d)), :, :) = hctsas{d}.TS_DataMat(:, fIds, :);
    row = row + sum(hctsas_rows(d));
end

%% Average performances across datasets
% Use for selecting "top" N features

% Unweighted average - each dataset is weighted equally
perfs_mean = zeros(size(valid_all));
for d = 1 : length(dsets)
    perfs_mean = perfs_mean + stats.(dsets{d}).(perf_type).performances{main_pairs(d)};
end
perfs_mean = perfs_mean ./ d;


% Weighted average - performance of each dataset is weighted by the
%   number of epochs(*flies) in the dataset
%   Note - multidose datasets have the most epochs*flies by far
%       (an order of magnitude more)

%% Take top N features

if topN > 0
    
    % Sort features
    
    % Take top N features
    
end

for ch = 1 : size(sig_all, 1)
    [perfs_sorted, order] = sort(perfs.(stage).(perf_type).performances(ch, fIds{ch}), 'descend');
    
    if numel(fIds{ch}) > topN
        
        fIds{ch} = fIds{ch}(order);
        fIds{ch} = fIds{ch}(1:topN);
        
        fValues{ch} = fValues{ch}(:, order);
        fValues{ch} = fValues{ch}(:, 1:topN);
    end
end

%% Cluster (all channels together)
% Use features which are significant in at least one channel



%% Cluster per channel
% Get significant features for each channel and cluster them

% Set whether to reorder feature clusters based on performance
plot_order = 'corrOrder'; % 'corrOrder'; 'clusterPerfOrder'
clusterThresh = 0.7;

%{
% Close existing plots
if exist('cluster_plot', 'var')
    if ishandle(cluster_plot); close(cluster_plot); end
end
if exist('dend_plot', 'var')
    if ishandle(dend_plot); close(dend_plot); end
end
%}

cluster_plot = figure; set(cluster_plot, 'Color', 'w');
dend_plot = figure; set(cluster_plot, 'Color', 'w');

% Storage for feature clustering
cluster_details = cell(size(hctsas_cat, 3), 1);

for ch = 1 : 1 % size(hctsas_cat, 3)
    feature_orders = struct();
    
    % Can't cluster if only 1 feature
    if ~(size(hctsas_cat, 2) < 2)
        
        % Get valid and significant feature values
        fIds_ch = find(valid_all(ch, :) & sig_all(ch, :));
        values = nan(nRows, length(fIds_ch)); % epochs x features
        row = 1;
        for d = 1 : length(dsets)
            values(row : sum(hctsas_rows(1:d)), :) = hctsas{d}.TS_DataMat(:, fIds_ch, ch);
            row = row + sum(hctsas_rows(d));
        end
        
        values(isinf(values)) = NaN; % Remove Infs for correlation
        
        % Correlate feature values
        fCorr = (corr(values, 'Type', 'Spearman', 'Rows', 'pairwise')); % Ignore NaNs
        fCorr = abs(fCorr + fCorr.') / 2; % because corr output isn't symmetric for whatever reason
        
        % Order features by correlation distance
        %corrOrder = clusterFeatures(values);
        
        % Order features by correlation distances
        distances = 1 - fCorr;
        tree = linkage(squareform(distances), 'average'); % note - distances must be pdist vector (treats matrix as data instead of distances)
        feature_orders.corrOrder = optimalleaforder(tree, distances);
        %f = figure('visible', 'off'); % we want the order, not the actual plot
        %[h, T, corrOrder] = dendrogram(tree, 0);
        %close(f);
        
        % Cluster features
        T = cluster(tree, 'Cutoff', clusterThresh, 'Criterion', 'distance');
        clusters = cell(max(T), 1);
        cluster_fIds = cell(size(clusters));
        cluster_centers = nan(size(clusters));
        cluster_perfs = cell(size(clusters));
        cluster_center_perf = nan(size(clusters));
        cluster_mean_perf = nan(size(clusters));
        for c = 1 : length(clusters)
            clusters{c} = find(T==c);
            cluster_fIds{c} = fIds_ch(clusters{c})';
            
            % Get best performing feature
            %cluster_perfs{c} = perfs.(stage).(perf_type).performances(ch, fIds(clusters{c}))';
            cluster_perfs{c} = mean(perfs_mean(fIds_ch(clusters{c})), 1);
            [maxPerf, id] = max(cluster_perfs{c});
            cluster_centers(c) = clusters{c}(id);
            cluster_center_perf(c) = maxPerf;
            cluster_mean_perf(c) = mean(cluster_perfs{c});
            
            % Sort features in cluster by performance
            [~, order] = sort(cluster_perfs{c}, 'desc');
            clusters{c} = clusters{c}(order);
            cluster_fIds{c} = cluster_fIds{c}(order);
            cluster_perfs{c} = cluster_perfs{c}(order);
            
        end
        
        % Sort clusters by performance
        [~, order] = sort(cluster_center_perf, 'desc');
        feature_orders.clusterPerfOrder = cat(1, clusters{order});
        
        % Reorder clusters if plotting clusterPerfOrder
        if strcmp(plot_order, 'clusterPerfOrder')
            clusters = clusters(order);
            cluster_fIds = cluster_fIds(order);
            cluster_centers = cluster_centers(order);
            cluster_perfs = cluster_perfs(order);
            cluster_center_perf = cluster_center_perf(order);
            cluster_mean_perf = cluster_mean_perf(order);
        end
        
        % Store cluster information for saving to file
        cluster_details{ch} = struct();
        cluster_details{ch}.clusters = clusters;
        cluster_details{ch}.cluster_fIds = cluster_fIds;
        cluster_details{ch}.cluster_centers = cluster_centers;
        cluster_details{ch}.cluster_perfs = cluster_perfs;
        cluster_details{ch}.cluster_center_perf = cluster_center_perf;
        cluster_details{ch}.cluster_mean_perf = cluster_mean_perf;
        cluster_details{ch}.feature_orders = feature_orders;
        
        % Get cluster positions
        cluster_pos = cellfun(@(x)arrayfun(@(y)find(feature_orders.(plot_order)==y),x),clusters,...
            'UniformOutput',false);
        cluster_center_pos = arrayfun(@(y)find(feature_orders.(plot_order)==y),cluster_centers);
        
        figure(cluster_plot); set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
        %subplot(4, 4, ch);
        similarity_mat = fCorr(feature_orders.(plot_order), feature_orders.(plot_order));
        cluster_details{ch}.corrMat = similarity_mat;
        imagesc(similarity_mat);
        set(gca, 'FontSize', 8);
        c = colorbar;
        c.Location = 'southoutside';
        ylabel(c, 'abs(r)');
        title(['ch' num2str(ch) ' - ' num2str(length(fIds_ch)) ' features']);
        display_limit = inf;
        if size(similarity_mat, 1) > display_limit % Limit axis to up to N features
            xlim([1 display_limit]);
            ylim([1 display_limit]);
        end
        colormap inferno
        %colormap([flipud(BF_GetColorMap('blues',9,0));BF_GetColorMap('reds',9,0)])
        hold on;
        
        % Outline clusters
        %colour_range = bone;
        %cluster_colour_ids = floor(rescale(cluster_mean_perf, 1, size(colour_range, 1)));
        %center_colours_ids = floor(rescale(cluster_center_perf, 1, size(colour_range, 1)));
        
        rectColors = BF_GetColorMap('accent', 5, 1);
        for c = 1 : length(clusters)
            % Cluster border
            rectangle('Position',[min(cluster_pos{c})-0.5,min(cluster_pos{c})-0.5, ...
                length(cluster_pos{c}),length(cluster_pos{c})], ...
                'EdgeColor', rectColors{1},'LineWidth',3);
            
            % Cluster center
            rectangle('Position',[cluster_center_pos(c)-0.5,cluster_center_pos(c)-0.5,1,1], ...
                'EdgeColor',rectColors{5},'LineWidth',3);
        end
        hold off
        
        axis square
        
        % Add feature names as ticklabels
        fNames = hctsas{1}.Operations{cat(1, cluster_fIds{:}), 'Name'};
        %set(gca, 'XTick', (1:length(fNames)), 'XTickLabel', fNames);
        %xtickangle(90);
        set(gca, 'YTick', (1:length(fNames)), 'YTickLabel', fNames);
        set(gca, 'TickLabelInterpreter', 'none');
        
        % Add figure to cluster output file
        %xlswritefig(cluster_plot, [out_dir out_file '.xlsx'], ['ch' num2str(ch)], 'J1');
        
        % Reference hctsa function plot
        %BF_ClusterDown(distances);
        
        % Make dendrogram
        figure(dend_plot); %subplot(4, 4, ch);
        h_dend = dendrogram(tree, 0, 'Reorder', feature_orders.(plot_order));
        set(h_dend, 'Color', 'k', 'LineWidth', 1);
        title(['ch' num2str(ch)]);
        
    end
end










%% Write feature performances to file

cluster_tables = cell(size(cluster_details));
for ch = 1 : length(cluster_details)
    
    if ~isempty(cluster_details{ch})
        
        % Assign labels to clusters
        cluster_labels = cluster_details{ch}.clusters;
        mean_perfs = cell(size(cluster_labels));
        cId = 0;
        for c = 1 : size(cluster_labels, 1)
            cluster_labels{c} = ones(size(cluster_labels{c})) + cId;
            cId = cId + 1;
            
            % Also repeat cluster's mean performance for vector conversion
            mean_perfs{c} = repmat(cluster_details{ch}.cluster_mean_perf(c), size(cluster_labels{c}));
        end
        
        % Convert to vectors for table
        cluster_labels = cat(1, cluster_labels{:});
        cluster_fIds = cat(1, cluster_details{ch}.cluster_fIds{:});
        cluster_perfs = cat(1, cluster_details{ch}.cluster_perfs{:});
        mean_perfs = cat(1, mean_perfs{:});
        
        % Get feature names from ids
        fNames = hctsa{ch}.Operations{cluster_fIds, 'Name'};
        fCodes = hctsa{ch}.Operations{cluster_fIds, 'CodeString'};
        mIds = hctsa{ch}.Operations{cluster_fIds, 'MasterID'};
        mNames = hctsa{ch}.MasterOperations{mIds, 'Label'};
        mCodes = hctsa{ch}.MasterOperations{mIds, 'Code'};
        
        % Make table
        headings = {...
            'cluster', 'clusterMeanPerf',...
            'fPerf', 'fID',...
            'fName', 'fCodeString', 'mID', 'mName', 'mCode'};
        cluster_table = table(...
            cluster_labels, mean_perfs,...
            cluster_perfs, cluster_fIds,...
            fNames, fCodes, mIds, mNames, mCodes,...
            'VariableNames', headings);
        
        % Write to file
        writetable(cluster_table, [out_dir out_file '.xlsx'],...
            'Sheet', ['ch' num2str(ch)],...
            'WriteVariableNames', 1);
        
    end
    
    cluster_tables{ch} = cluster_table;
    
end

save([out_dir out_file], 'cluster_tables', 'cluster_details');
