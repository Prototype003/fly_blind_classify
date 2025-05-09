%% Description

%{

Cluster significant features based on similarity in feature values across
all epochs

%}

%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

%% Get performance stats

perfs = get_stats(preprocess_string);

%%

data_sets = {'train', 'validate1'};

%% Get raw values for each channel

fValues_all = cell(size(perfs.train.valid_features, 1), 1);
hctsa = cell(size(fValues_all));
for ch = 1 : size(fValues_all, 1)
    tic;
    % Load and concatenat TS_DataMat for each dataset
    ds = cell(1);
    for d = 1 : length(data_sets)
        hctsa{ch} = hctsa_load(data_sets{d}, ch, preprocess_string);
        ds{d} = hctsa{ch}.TS_DataMat;
    end
    fValues_all{ch} = cat(1, ds{:});
    toc
end

%%

perf_type = 'nearestMedian'; % nearestMedian; consis
stage = 'train'; % train; validate1

switch stage
    case 'train'
        stage_string = 'discovery';
    case 'validate1'
        stage_string = 'pilotEvaluation';
end

topN = 40;

out_dir = [pwd filesep 'results' preprocess_string '_cluster_top' num2str(topN) filesep];
out_file = [perf_type '_' stage_string '_clusters']; % file with cluster details

if ~exist(out_dir, 'dir'); mkdir(out_dir); end

%% Get significant features at each channel

% Significant in both datasets
sig_all = perfs.train.(perf_type).sig & perfs.validate1.(perf_type).sig;

% Valid in both datasets, and sig. in all datasets
valid_all = perfs.train.valid_features & perfs.validate1.valid_features;

%% Get significant features at each channel

sig_all = perfs.train.(perf_type).sig;

switch stage
    case 'train'
        sig_all = perfs.train.(perf_type).sig;
        valid_all = perfs.train.valid_features;
    case 'validate1'
        sig_all = perfs.train.(perf_type).sig & perfs.validate1.(perf_type).sig;
        valid_all = perfs.train.valid_features & perfs.validate1.valid_features;
end

%% Filter features

% Keep features which are valid and sig. in all sets
fValues = cell(size(fValues_all));
fIds = cell(size(fValues));
for ch = 1 : size(sig_all, 1)
    fIds{ch} = find(valid_all(ch, :) & sig_all(ch, :));
    fValues{ch} = fValues_all{ch}(:, fIds{ch});
end

%% Take top N

for ch = 1 : size(sig_all, 1)
    [perfs_sorted, order] = sort(perfs.(stage).(perf_type).performances(ch, fIds{ch}), 'descend');
    
    if numel(fIds{ch}) > topN
        
        fIds{ch} = fIds{ch}(order);
        fIds{ch} = fIds{ch}(1:topN);
        
        fValues{ch} = fValues{ch}(:, order);
        fValues{ch} = fValues{ch}(:, 1:topN);
    end
end

%%

% Kill open excel process (e.g. if an error occurs during a function and
% leaves excel open)
system('taskkill /F /IM EXCEL.EXE');

%% Cluster

plot_order = 'clusterPerfOrder'; % 'corrOrder'; 'clusterPerfOrder'
clusterThresh = 0.2;

if exist('cluster_plot', 'var')
    if ishandle(cluster_plot); close(cluster_plot); end
end
if exist('dend_plot', 'var')
    if ishandle(dend_plot); close(dend_plot); end
end

cluster_plot = figure; set(cluster_plot, 'Color', 'w');
dend_plot = figure; set(cluster_plot, 'Color', 'w');

% Storage for feature clustering
clusters_all = cell(length(fValues), 1);

for ch = (1 : length(fValues))
    feature_orders = struct();
    
    % Can't cluster if only 1 feature
    if ~(length(fValues{ch}) < 2)
        
        % Correlate feature values
        values = fValues{ch};
        values(isinf(values)) = NaN; % Remove Infs for correlation
        fCorr = (corr(values, 'Type', 'Spearman', 'Rows', 'complete')); % Ignore NaNs
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
            cluster_fIds{c} = fIds{ch}(clusters{c})';
            
            % Get best performing feature
            cluster_perfs{c} = perfs.(stage).(perf_type).performances(ch, fIds{ch}(clusters{c}))';
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
        clusters_all{ch} = struct();
        clusters_all{ch}.clusters = clusters;
        clusters_all{ch}.cluster_fIds = cluster_fIds;
        clusters_all{ch}.cluster_centers = cluster_centers;
        clusters_all{ch}.cluster_perfs = cluster_perfs;
        clusters_all{ch}.cluster_center_perf = cluster_center_perf;
        clusters_all{ch}.cluster_mean_perf = cluster_mean_perf;
        clusters_all{ch}.feature_orders = feature_orders;
        
        % Get cluster positions
        cluster_pos = cellfun(@(x)arrayfun(@(y)find(feature_orders.(plot_order)==y),x),clusters,...
            'UniformOutput',false);
        cluster_center_pos = arrayfun(@(y)find(feature_orders.(plot_order)==y),cluster_centers);
        
        figure(cluster_plot); set(gcf, 'units','normalized','outerposition',[0 0 1 1]);
        %subplot(4, 4, ch);
        similarity_mat = fCorr(feature_orders.(plot_order), feature_orders.(plot_order));
        clusters_all{ch}.corrMat = similarity_mat;
        imagesc(similarity_mat);
        set(gca, 'FontSize', 8);
        c = colorbar;
        c.Location = 'southoutside';
        title(['ch' num2str(ch) ' - ' num2str(size(fValues{ch}, 2)) ' features']);
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
        fNames = hctsa{ch}.Operations{cat(1, cluster_fIds{:}), 'Name'};
        %set(gca, 'XTick', (1:length(fNames)), 'XTickLabel', fNames);
        %xtickangle(90);
        set(gca, 'YTick', (1:length(fNames)), 'YTickLabel', fNames);
        set(gca, 'TickLabelInterpreter', 'none');
        
        % Add figure to cluster output file
        xlswritefig(cluster_plot, [out_dir out_file '.xlsx'], ['ch' num2str(ch)], 'J1');
        
        % Reference hctsa function plot
        %BF_ClusterDown(distances);
        
        % Make dendrogram
        figure(dend_plot); subplot(4, 4, ch);
        h_dend = dendrogram(tree, 0, 'Reorder', feature_orders.(plot_order));
        set(h_dend, 'Color', 'k', 'LineWidth', 1);
        title(['ch' num2str(ch)]);
        
    end
end

%% Write feature performances to file

cluster_tables = cell(size(clusters_all));
for ch = 1 : length(clusters_all)
    
    if ~isempty(clusters_all{ch})
        
        % Assign labels to clusters
        cluster_labels = clusters_all{ch}.clusters;
        mean_perfs = cell(size(cluster_labels));
        cId = 0;
        for c = 1 : size(cluster_labels, 1)
            cluster_labels{c} = ones(size(cluster_labels{c})) + cId;
            cId = cId + 1;
            
            % Also repeat cluster's mean performance for vector conversion
            mean_perfs{c} = repmat(clusters_all{ch}.cluster_mean_perf(c), size(cluster_labels{c}));
        end
        
        % Convert to vectors for table
        cluster_labels = cat(1, cluster_labels{:});
        cluster_fIds = cat(1, clusters_all{ch}.cluster_fIds{:});
        cluster_perfs = cat(1, clusters_all{ch}.cluster_perfs{:});
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

save([out_dir out_file], 'cluster_tables', 'clusters_all');
