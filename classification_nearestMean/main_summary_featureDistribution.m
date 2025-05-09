%% Description

%{

Plot distribution of values for a hctsa feature:
    Best accuracy in training set
    Best accuracy in pilot validation set
    Best consistency in training set
    Best consistency in pilot validation set

%}

%% Settings

ch = 6;
perf_type = 'class_nearestMedian';
ref_set = 'validate1';

source_dir = 'results/';

addpath('../');

%% Get values for best performing feature in the reference set

[values, fDetails, perf] = get_best_from_set(ch, perf_type, ref_set, 0);

%% Plotting settings

% Colours
c = BF_GetColorMap('redyellowblue', 6);
cond_colours = {c(1, :), c(end, :)}; % red = wake; blue = anest
cond_offsets = [-0.1 0.1]; % x-axis offsets for each violin

extraParams = struct();
extraParams.offsetRange = 0.5; % width of violins

%% Plot violins

figure;
hold on;

% Plot trained threshold for the feature
thresh = load([source_dir 'class_nearestMedian_thresholds.mat']); % note no thresholds for consistency
threshold = thresh.thresholds(ch, fDetails.fID);
line([0 size(values, 1)+1], [threshold threshold], 'Color', 'k', 'LineWidth', 1, 'LineStyle', ':');

% Plot violins
for cond = 1 : size(values, 2)
    
    extraParams.customOffset = cond_offsets(cond);
    extraParams.theColors = repmat(cond_colours(cond), [size(values, 1) 1]);
    
    BF_JitteredParallelScatter_custom(values(:, cond), 1, 1, 0, extraParams);
    
end

axis tight

% Plot lines to separate datasets
line([13.5 13.5], ylim, 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--');

set(gca, 'XTick', (1:size(values, 1)));

%title([perf_type '-' reference_set ': ' num2str(fDetails.fID) ' ' fDetails.fName{1}], 'interpreter', 'none');
title(['best from ' ref_set ': ' num2str(fDetails.fID)]);
xlabel('fly');
ylabel('raw val');

