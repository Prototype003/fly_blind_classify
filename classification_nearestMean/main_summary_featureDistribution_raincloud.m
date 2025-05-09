%% Description

%{

Plot distribution of values for a hctsa feature:
    Best accuracy in training set
    Best accuracy in pilot validation set
    Best consistency in training set
    Best consistency in pilot validation set

%}

%% Settings

ch = 6; % which channel to use

source_dir = 'results/';
hctsa_prefixes = {'train', 'validate1'};
perf_types = {'class_nearestMean', 'consis_nearestMean'};

addpath('../');

%% 1 - get most accurate feature from training set

perf_type = 'class_nearestMean'; % 1=class_nearestMean; 2=consis_nearestMean
perf_type = 'consis_nearestMean';
%perf_type = 'class_nearestMedian';

if strcmp(perf_type, 'class_nearestMean') || strcmp(perf_type, 'class_nearestMedian') 
    perf_sets = {'crossValidation', 'validate1_accuracy'};
elseif strcmp(perf_type, 'consis_nearestMean')
    perf_sets = {'train', 'validate1'};
end

reference_set = perf_sets{1}; % 1=crossValidation; 2=validate1_accuracy

% Load reference performance data
source_file = [perf_type '_' reference_set '.mat'];
acc = load([source_dir source_file]);
acc.accuracies = mean(acc.consistencies, 4);
accuracies = mean(acc.accuracies, 3); % average accuracies across cross-validations

% Valid features
tic;
hctsa = hctsa_load(hctsa_prefixes{1}, ch);
[valid_features, valid] = getValidFeatures(hctsa.TS_DataMat);
toc

% Find best feature
[sorted, order] = sort(accuracies(ch, :), 'descend');
% Look up the feature
fID = hctsa.Operations{order(1), 'ID'};
fName = hctsa.Operations{order(1), 'Name'};
% Look up master feature
mID = hctsa.Operations{order(1), 'MasterID'};
mName = hctsa.MasterOperations{mID, 'Label'};
perf = nan(size(hctsa_prefixes));
perf(1) = sorted(1); % reference performance

% Get trained threshold for the feature
thresh = load([source_dir 'class_nearestMean_thresholds.mat']); % note no thresholds for consistency
threshold = thresh.thresholds(ch, fID);

% Get values for each fly
fly_rows = ones(1, 2); % each row n holds the starting rows of the nth fly for each condition
fly_vals = cell(1, 2); % vector for each condition (can we assume equal observations per condition?)
f_counter = 1;
for d = 1 : length(hctsa_prefixes)
    [nChannels, nFlies, nConditions, nEpochs] = getDimensions(hctsa_prefixes{d});
    
    % Load values
    hctsa = hctsa_load(hctsa_prefixes{d}, ch);
    
    % Get values per fly
    %   Keep track of what rows belong to which flies
    for f = 1 : nFlies
        for c = 1 : 2
            fRows = find(getIds({['fly' num2str(f)], ['condition' num2str(c)]}, hctsa.TimeSeries));
            tmp = hctsa.TS_DataMat(fRows, fID);
            fly_vals{c} = cat(1, fly_vals{c}, tmp);
            fly_rows(f_counter+1, c) = fly_rows(f_counter, c) + length(fRows);
        end
        f_counter = f_counter + 1;
    end
    
    % Get performance of the feature
    source_file = [perf_type '_' perf_sets{d} '.mat'];
    acc = load([source_dir source_file]);
    acc.accuracies = mean(acc.consistencies, 4);
    tmp = mean(acc.accuracies, 3); % average accuracies across cross-validations
    perf(d) = tmp(ch, fID);
    
end

% Combine classes to scale altogether
cond_rows = [0 cumsum(cellfun(@length, fly_vals))]; % each gives the last row in each class
vals_all = cat(1, fly_vals{:});

% Scale values (but note the threshold will need to be scaled too...)
%vals_all = BF_NormalizeMatrix(vals_all, 'mixedSigmoid');

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

%% Plot rainclouds

% Plot
%subplot(2, 1, 2);
%subplot(4, 2, 4);
h = rm_raincloud(values, [1 0 0; 0 0 1], 0); hold on; % note x-y by default are swapped
delete(h.l); % delete lines

% Format figure
cellfun(@(x) set(x, 'SizeData', 10), h.s); % reduce size of individual points
cellfun(@(x) set(x, 'MarkerFaceAlpha', 0.3), h.s); % transparency of individual points
arrayfun(@(x) set(x, 'SizeData', 25), h.m); % reduce size of means
arrayfun(@(x) set(x, 'MarkerFaceAlpha', 1), h.m); % transparency of means
arrayfun(@(x) set(x, 'LineWidth', 1), h.m); % reduce outline of means
cellfun(@(x) set(x, 'FaceAlpha', 0.4), h.p); % transparency of clouds

% shift the individual points of one class

% separate the clouds more (by moving the entire rainclouds)


axis tight

% Plot chance
line([threshold threshold], ylim, 'Color', 'k', 'LineWidth', 2, 'LineStyle', '--');

axis tight
%title([perf_type '-' reference_set ': ' num2str(fID) ' ' fName{1}], 'interpreter', 'none');
title(['best from ' reference_set ': ' num2str(fID)]);
ylabel('fly');
xlabel('raw val');

%% Manual repeated rainclouds

figure;
set(gcf, 'Color', [1 1 1]);
set(gcf, 'InvertHardCopy', 'off'); % For keeping the black background when printing
set(gcf, 'RendererMode', 'manual');
set(gcf, 'Renderer', 'painters');

%colours = cbrewer('qual', 'Dark2', size(values, 2)); % https://au.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/34087/versions/2/screenshot.jpg
colours = [1 0 0; 0 0 1];

base_offset = 2;
baselines = (0:-base_offset:-(size(values, 1)-1)*base_offset); % baselines of clouds
rain_spread = 0.1;
cloud_rain_dist = 0.3;
rain_offset = (rain_spread/2) + cloud_rain_dist; % middle of rain

handles = cell(size(values));

ax_counter = 1;
for fly = 1 : size(values, 1)
    for cond = 1 : size(values, 2)
        ax(ax_counter) = axes; % Not possible to set multiple baselines within same axis;
        set(gca, 'XAxisLocation', 'top');
        
        if ax_counter > 1
            ax(ax_counter).YLim = ax(1).YLim;
            ax(ax_counter).XLim = ax(1).XLim;
            hold on; % this locks the axes limits
        end
        
        % Create raincloud
        handles{fly, cond} = raincloud_plot(values{fly, cond}, 'box_on', 0, 'color', colours(cond, :), 'alpha', 0.5,...
            'box_dodge', 1, 'box_dodge_amount', 0, 'dot_dodge_amount', 0,...
            'box_col_match', 0);
        
        % Shift baseline (cloud)
        handles{fly, cond}{1}.BaseValue = baselines(fly); % move baseline
        handles{fly, cond}{1}.YData = handles{fly, cond}{1}.YData + baselines(fly); % move cloud
        handles{fly, cond}{1}.ShowBaseLine = 'off'; % hide baseline
        handles{fly, cond}{1}.EdgeAlpha = 0; % hide cloud outline
        
        % Shift baseline (rain)
        rain_scatter = (rand(length(values{fly, cond}), 1) - 0.5) * rain_spread;
        handles{fly, cond}{2}.YData = rain_scatter;
        handles{fly, cond}{2}.YData = handles{fly, cond}{2}.YData + baselines(fly) - rain_offset; % move rain
        handles{fly, cond}{2}.SizeData = 2; % raindrop size
        
        % Hide all axes except the first
        if ax_counter > 1
            axis off; % hide all axes except the first
        end
        
        view([-90 90]);
        
        ax_counter = ax_counter + 1;
    end
end

linkaxes(ax);
ylim([min(baselines)-base_offset/2 max(baselines)+base_offset]);

