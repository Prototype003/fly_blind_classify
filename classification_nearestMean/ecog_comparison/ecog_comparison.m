%% Description

%{

%}

%%

addpath('../');
ch = 6; % which fly channel to look at
perf_type = 'nearestMedian';

%% Load hctsa file

preprocess_string = '_subtractMean_removeLineNoise';
source_prefix = 'train';

source_dir = ['../../hctsa_space' preprocess_string '/'];
source_file = ['HCTSA_' source_prefix '_channel6.mat']; % HCTSA_train.mat; HCTSA_validate1.mat;

tic;
hctsa = load([source_dir source_file]);
toc

%% Load fly data

preprocess_string = '_subtractMean_removeLineNoise';
cond_strings = {'wake', 'anest'};
data_sets = {'train', 'validate1'};

source_dir = 'results';

%% Load performances

perfs = get_stats(preprocess_string);

%% Load human data

human = load('Accuracy_n_ID.mat');

%% Convert human data to use fly data format

valid_features_human = zeros(1, 7702);
valid_features_human(human.accuracy_n_id(:, 2)) = 1;

accuracies_human = nan(1, 7702);
accuracies_human(human.accuracy_n_id(:, 2)) = human.accuracy_n_id(:, 1);

%% Find which features are valid in all datasets

valid_features = perfs.train.valid_features & perfs.validate1.valid_features;
valid_features_flyHuman = valid_features(ch, :) & valid_features_human;

%%

figure;
scatter(...
    perfs.train.nearestMedian.performances(ch, valid_features(ch, :)),...
    perfs.validate1.nearestMedian.performances(ch, valid_features(ch, :)),...
    '.');

%% Get best feature

% For every point, find distance from (0.5 0.5 0.5) in top-right quadrant
dists = nan(size(valid_features_flyHuman));
fIDs = find(valid_features_flyHuman);
for f = 1 : length(fIDs)
    fID = fIDs(f);
    x = perfs.(data_sets{1}).(perf_type).performances(ch, fID);
    y = perfs.(data_sets{2}).(perf_type).performances(ch, fID);
    z = accuracies_human(fID);
    
    xdist = x - 0.5;
    ydist = y - 0.5;
    zdist = z - 0.5;
    
    dists(fID) = sqrt(xdist^2 + ydist^2 + zdist^2); % check 3d pythagoras theorem
    
%    if xdist <= 0 || ydist <= 0 || zdist <=0
%        dists(fID) = NaN; % In the wrong quadrant
%    end
end

[topDist, topID] = max(dists);
[dists_sorted, dist_order] = sort(dists(~isnan(dists)), 'descend');
fIDs_sorted = fIDs(dist_order);

%% List of best features by distance

topN = 10;
top_features = cell(topN, 2);
top_features(:, 1) = hctsa.Operations.Name(fIDs_sorted(1:topN));
top_features(:, 2) = num2cell(dists_sorted(1:topN));

%% List of features above some threshold

thresh = 0.7;
top_features = {};
above_thresh =...
    perfs.train.nearestMedian.performances(ch, :) > thresh &...
    perfs.validate1.nearestMedian.performances(ch, :) > thresh &...
    accuracies_human > thresh;

fIDs = find(above_thresh);
topN = length(fIDs);
top_features = cell(topN, 4);
top_features(:, 1) = hctsa.Operations.Name(fIDs);
top_features(:, 2) = num2cell(perfs.train.nearestMedian.performances(ch, fIDs));
top_features(:, 3) = num2cell(perfs.validate1.nearestMedian.performances(ch, fIDs));
top_features(:, 4) = num2cell(accuracies_human(fIDs));

%% Plot 3D

jitters = rand(2, length(valid_features_flyHuman))*0.005;

figure;
colormap plasma;
scatter3(...
    perfs.train.nearestMedian.performances(ch, valid_features_flyHuman) + jitters(1, valid_features_flyHuman),...
    perfs.validate1.nearestMedian.performances(ch, valid_features_flyHuman) + jitters(2, valid_features_flyHuman),...
    accuracies_human(valid_features_flyHuman),...
    (dists(valid_features_flyHuman)+1).^15,...
    dists(valid_features_flyHuman),...
    '.');
xlabel('discovery fly perf.');
ylabel('evaluation fly perf.');
zlabel('human perf.');

axis vis3d;
hold on;

% Highlight best features
topN = 10;
scatter3(...
    perfs.train.nearestMedian.performances(ch, fIDs_sorted(1:topN)) + jitters(1, fIDs_sorted(1:topN)),...
    perfs.validate1.nearestMedian.performances(ch, fIDs_sorted(1:topN)) + jitters(2, fIDs_sorted(1:topN)),...
    accuracies_human(fIDs_sorted(1:topN)),...
    100,...
    'ko', 'LineWidth', 1);

c = colorbar;
ylabel(c, 'distance from chance');

xlim_now = xlim;
ylim_now = ylim;
zlim_now = zlim;

% Chance planes
fill3([0 1 1 0], [0 0 1 1], [0.5 0.5 0.5 0.5], 'k', 'FaceAlpha', 0.05, 'EdgeColor', 'none');
fill3([0.5 0.5 0.5 0.5], [0 1 1 0], [0 0 1 1], 'k', 'FaceAlpha', 0.05, 'EdgeColor', 'none');
fill3([0 1 1 0], [0.5 0.5 0.5 0.5], [0 0 1 1], 'k', 'FaceAlpha', 0.05, 'EdgeColor', 'none');

xlim(xlim_now);
ylim(ylim_now);
zlim(zlim_now);

%% Plot 2D

figure;
colormap plasma
scatter(...
    perfs.train.nearestMedian.performances(ch, valid_features_flyHuman) + jitters(1, valid_features_flyHuman),...
    perfs.validate1.nearestMedian.performances(ch, valid_features_flyHuman) + jitters(2, valid_features_flyHuman),...
    (dists(valid_features_flyHuman)+1).^15,...
    accuracies_human(valid_features_flyHuman),...
    '.');

xlabel('discovery fly perf.');
ylabel('evaluation fly perf.');

c = colorbar;
ylabel(c, 'human perf.');

xlim_now = xlim;
ylim_now = ylim;

hold on;

% Highlight best features
topN = 10;
scatter(...
    perfs.train.nearestMedian.performances(ch, fIDs_sorted(1:topN)) + jitters(1, fIDs_sorted(1:topN)),...
    perfs.validate1.nearestMedian.performances(ch, fIDs_sorted(1:topN)) + jitters(2, fIDs_sorted(1:topN)),...
    100,...
    'ko', 'LineWidth', 1.5);

% Plot chance
line([0 1], [0.5 0.5], 'Color', [0 0 0 0.5]);
line([0.5 0.5], [0 1], 'Color', [0 0 0 0.5]);
xlim(xlim_now);
ylim(ylim_now);

%% Plot

figure; scatter(accuracies_flyMean(6, valid_features_flyHuman), accuracies_human(valid_features_flyHuman), '.');

%% Correlations

[r, p] = corr(accuracies_flyMean(6, valid_features_flyHuman)', accuracies_human(valid_features_flyHuman)')

%% Limit to only features which are valid

accuracies_human_valid = accuracies_human(valid_features_flyHuman);
accuracies_flyMean_valid = accuracies_flyMean(:, valid_features_flyHuman);

%% Order by performance, for each dataset

[sorted, sorted_human] = sort(accuracies_human_valid, 'descend');
[sorted, sorted_fly] = sort(accuracies_flyMean_valid(6, :), 'descend');
[sorted, sorted_fly] = sort(accuracies_flyEval_valid(6, :), 'descend');

figure; plot(accuracies_flyMean_valid(sorted_human)); hold on; plot(accuracies_human_valid(sorted_human));
figure; plot(accuracies_flyMean_valid(6, sorted_fly)); hold on; plot(accuracies_human_valid(sorted_fly));

%% Limit to features which successfully generalised from discovery to evaluation flies

%% Get best features

