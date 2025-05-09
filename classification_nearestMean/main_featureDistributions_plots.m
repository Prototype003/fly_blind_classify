%% Description

%{

Plot summary figure for feature clustering

%}

%%

preprocess_string = '_subtractMean_removeLineNoise';

perf_type = 'nearestMedian'; % nearestMedian; consis
plot_order = 'clusterPerfOrder'; % 'corrOrder'; 'clusterPerfOrder'
perf_types = {'nearestMedian', 'consis'};
data_sets = {'train', 'validate1'};

train_flies = (1:13);
val_flies = (14:15);

addpath('../');

%% Load

source_dir = ['results' preprocess_string];

% hctsa values and time series
hctsa = cell(1);
for ch = 1 : 15 % find way to get number of channels dynamically
    tic;
    % Load and concatenat TS_DataMat for each dataset
    ds = cell(1);
    hctsa{ch} = struct();
    for d = 1 : length(data_sets)
        hctsa{ch}.(data_sets{d}) = hctsa_load(data_sets{d}, ch, preprocess_string);
    end
    toc
end
% Get dimensions
dims = cell(size(data_sets));
for d = 1 : length(data_sets)
    [nChannels, nFlies, nConditions, nEpochs] = getDimensions(data_sets{d});
    dims{d}.nChannels = nChannels;
    dims{d}.nFlies = nFlies;
    dims{d}.nConditions = nConditions;
    dims{d}.nEpochs = nEpochs;
end

% Get performances
perfs = get_stats(preprocess_string);

%%

ch = 6; % channel to focus on in the figure

figure;
set(gcf, 'Color', 'w');

% Colours
c = BF_GetColorMap('redyellowblue', 6);
cond_colours = {c(1, :), c(end, :)}; % red = wake; blue = anest
cond_offsets = [-0.15 0.15]; % x-axis offsets for each violin
extraParams = struct();
extraParams.offsetRange = 0.5; % width of violins

%%

plot_IDs = [4529 16 932];

%% Plot feature values

scale_vals = 1;

for f = 1 : length(plot_IDs)
    pID = plot_IDs(f);
    
    subplot(length(plot_IDs), 1, f);
    
    fID = plot_IDs(f);
    
    % Get feature values for every fly
    fly_rows = ones(1, 2); % each row n holds the starting rows of the nth fly for each condition
    fly_vals = cell(1, 2); % vector for each condition (can we assume equal observations per condition?)
    f_counter = 1;
    for d = 1 : length(data_sets)
        nChannels = dims{d}.nChannels;
        nFlies = dims{d}.nFlies;
        nConditions = dims{d}.nConditions;
        nEpochs = dims{d}.nEpochs;
        
        % Get values per fly
        %   Keep track of what rows belong to which flies
        for fly = 1 : nFlies
            for c = 1 : 2
                fRows = find(getIds({['fly' num2str(fly)], ['condition' num2str(c)]}, hctsa{ch}.(data_sets{d}).TimeSeries));
                tmp = hctsa{ch}.(data_sets{d}).TS_DataMat(fRows, fID);
                fly_vals{c} = cat(1, fly_vals{c}, tmp);
                fly_rows(f_counter+1, c) = fly_rows(f_counter, c) + length(fRows);
            end
            f_counter = f_counter + 1;
        end
    end
    
    % Combine classes to scale altogether
    cond_rows = [0 cumsum(cellfun(@length, fly_vals))]; % each gives the last row in each class
    vals_all = cat(1, fly_vals{:});
    
    % Get trained threshold for the feature
    thresh = load([source_dir filesep 'class_nearestMedian_thresholds.mat']); % note no thresholds for consistency
    threshold = thresh.thresholds(ch, fID);
    
    % Get trained medians for conditions
    meds = nan(nConditions, 1);
    for cond = 1 : nConditions
        values_tmp = fly_vals{cond}(fly_rows(train_flies(1), cond):fly_rows(val_flies(1), cond)-1);
        meds(cond) = median(values_tmp);
    end
    
    % Scale values (but note the threshold will need to be scaled too)
    if scale_vals
        vals_all = cat(1, vals_all, threshold, meds);
        [vals_all] = BF_NormalizeMatrix(vals_all, 'mixedSigmoid');
        threshold = vals_all(end-2);
        meds = vals_all(end-1:end);
        vals_all = vals_all(1:end-3);
    end
    
    % Separate classes again
    for c = 1 : length(cond_rows)-1
        fly_vals{c} = vals_all(cond_rows(c)+1:cond_rows(c+1));
    end
    
    % Separate values per fly
    values = cell(f_counter-1, nConditions);
    for f = 1 : f_counter-1
        for c = 1 : size(fly_vals, 2)
            if f == size(fly_rows, 1)
                values{f, c} = fly_vals{c}(fly_rows(f, c):end);
            else
                values{f, c} = fly_vals{c}(fly_rows(f, c):fly_rows(f+1, c)-1);
            end
        end
    end
    
    % Plot violins
    for cond = 1 : size(values, 2)
        
        extraParams.customOffset = cond_offsets(cond);
        extraParams.theColors = repmat(cond_colours(cond), [size(values, 1) 1]);
        extraParams.offsetRange = 0.3;
        
        BF_JitteredParallelScatter_custom(values(:, cond), 1, 1, 0, extraParams);
        
    end
    axis tight
    
    % Plot trained threshold for the feature
    line([0 size(values, 1)+1], [threshold threshold], 'Color', 'k', 'LineWidth', 1, 'LineStyle', ':');
    
    % Plot median for each condition
    for cond = 1 : nConditions
        med = meds(cond);
        line([0 size(values, 1)+1], [med med], 'Color', cond_colours{cond}, 'LineWidth', 1, 'LineStyle', ':');
    end
    
    % Plot lines to separate datasets
    line([13.5 13.5], ylim, 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--');
    
    set(gca, 'XTick', (1:size(values, 1)));
    
    title([hctsa{ch}.train.Operations.Name{fID}], 'interpreter', 'none');
    
    if f == length(plot_IDs)
        xlabel('fly');
        ylabel('norm. val');
    end

end
