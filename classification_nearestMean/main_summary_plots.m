%% Description

%{

Plot and get summary stats for training set cross-validation

%}

%%

perf_types = {'nearestMedian', 'consis'};
data_sets = {'train', 'validate1'};

preprocess_string = '_subtractMean_removeLineNoise';

addpath('../');

%% Get valid features

[ch_valid_features, ch_excluded] = getValidFeatures_allChannels('train', preprocess_string);
nValid = sum(ch_valid_features, 2);
nValid_mean = mean(nValid)
nValid_min = min(nValid)
nValid_max = max(nValid)

%%

perfs = get_stats(preprocess_string);

%% Common plotting variables

% Channel colours
r = linspace(0, 1, size(ch_valid_features, 1))';
g = linspace(0, 0, size(ch_valid_features, 1))';
b = linspace(1, 0, size(ch_valid_features, 1))';
ch_colours = cat(2, r, g, b);

% data_set colours
tmp = cbrewer('div', 'BrBG', 8);
dset_colours = {tmp(end:-1:end-1, :), tmp(2:3, :)};
dset_lineWidth = [1 1];
dset_alpha1 = [1 0.5];
dset_alpha2 = [0.5 0.5];
dset_alpha_mult = [1 0.9];

subplot_rows = 3;
subplot_cols = 2;

%%

figure;
set(gcf, 'Color', 'w');

%% Feature performance distributions

ptype = 1;
perf_type = perf_types{ptype};

% Reference data set
data_set = data_sets{1};
performances = perfs.(data_set).(perf_type).performances;
sig = perfs.(data_set).(perf_type).sig;

% Best channel (with largest number of sig. features)
ch_best = find(sum(sig, 2) == max(sum(sig, 2)));
ch_worst = find(sum(sig, 2) == min(sum(sig, 2)));
ch_best = 6;

% Sort reference data set
[sorted, order] = sort(performances(ch_best, :), 'descend');

% Get sig. features from reference data set
sig_features = sum(sig, 1);
sig_features = sig_features(order) > 0;

plot_counter = 1;
for dset = 1 : length(data_sets)
    data_set = data_sets{dset};
    
    % Get performance values and significances
    performances = perfs.(data_set).(perf_type).performances;
    
    % Sort by order of reference data set
    performances_sorted = performances(:, order);
    
    % Plot features sorted by performance
    subplot(subplot_rows, subplot_cols, [1 4]); hold on;
    plot(x, performances_sorted(ch_best, sig_features), 'Color', [dset_colours{dset}(1,:) dset_alpha(dset)], 'LineWidth', dset_lineWidth(dset)); % best ch
    
    % Plot mean+std across channels
    y = mean(performances_sorted(:, sig_features), 1);
    yerr = std(performances_sorted(:, sig_features), [], 1);
    x = (1:sum(sig_features));
    h = patch([x fliplr(x)], [y+yerr fliplr(y-yerr)], dset_colours{dset}(2, :), 'FaceAlpha', dset_alpha2(dset), 'EdgeColor', 'none'); % std
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    
    h = plot(x, y, 'Color', [dset_colours{dset}(2, :) dset_alpha(dset)*dset_alpha_mult(dset)]); % mean
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    
    axis tight
    %plot_counter = plot_counter + 1;
end

line([1 max(x)], [0.5 0.5], 'LineWidth', 1, 'Color', 'k'); % chance line
% plot sig. threshold
for dset = 1 : length(data_sets)
    data_set = data_sets{dset};
    sig_thresh = perfs.(data_set).(perf_type).sig_thresh_fdr(ch_best);
    line([1 max(x)], [sig_thresh sig_thresh], 'LineWidth', 2, 'LineStyle', ':', 'Color', dset_colours{dset}(1, :)*0.9);
end

legend([data_sets{1} ' ch' num2str(ch_best)], [data_sets{2} ' ch' num2str(ch_best)]);
title(['sig. features sorted by ch' num2str(ch_best)]);
ylabel('performance'); xlabel(['feature (sorted by ch' num2str(ch_best) ')']);
axis tight;

title([letters{plot_counter} ' ' 'sig. features sorted by ch' num2str(ch_best)]);
xlabel('feature');

%% Plot discovery vs evaluation, for single channel

subplot(subplot_rows, subplot_cols, [1 3]); hold on;

ch = 6;
ptype = 1;
perf_type = perf_types{ptype};

% Get features which are sig. in all datasets
data_set = data_sets{1};
sig = perfs.(data_sets{1}).(perf_type).sig;
for dset = 2 : length(data_sets)
    sig = sig & perfs.(data_sets{dset}).(perf_type).sig;
end
sig = sig(ch, :);
features = logical(ch_valid_features(ch, :));
fIDs = find(features);

% Plot class. vs consis.
scatter(...
    perfs.train.(perf_type).performances(ch, features),...
    perfs.validate1.(perf_type).performances(ch, features),...
    '.');
axis tight
xl = xlim;
yl = ylim;

% Plot chance
line([0.5 0.5], yl, 'Color', 'k');
line(xl, [0.5 0.5], 'Color', 'k');
% Plot significance thresholds
thresh = perfs.train.(perf_type).sig_thresh_fdr(ch);
line([thresh thresh], yl, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
thresh = perfs.validate1.(perf_type).sig_thresh_fdr(ch);
line(xl, [thresh thresh], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);

title(['ch ' num2str(ch) ' ' perf_type]);
xlabel('discovery');
ylabel('pilot evaluation');
axis square

% Show best feature in each dataset
disp('===');
for dset = 1 : length(data_sets)
    data_set = data_sets{dset};
    [p, f] = max(perfs.(data_set).(perf_type).performances(ch, features));
    scatter(...
        perfs.train.(perf_type).performances(ch, fIDs(f)),...
        perfs.validate1.(perf_type).performances(ch, fIDs(f)),...
        'o', 'k');
    disp(['best for ' data_set ': fID:' num2str(fIDs(f)) ' perf=' num2str(p)]);
end

% Show best overall feature (feature most towards the top-right)
%   For every point, find distance from (0.5 0.5) in top-right quadrant
dists = nan(size(fIDs));
for f = 1 : size(fIDs, 2)
    fID = fIDs(f);
    x = perfs.train.(perf_type).performances(ch, fID);
    y = perfs.validate1.(perf_type).performances(ch, fID);
    
    xdist = x - 0.5;
    ydist = y - 0.5;
    
    if xdist <= 0 || ydist <= 0
        dists(f) = NaN; % In the wrong quadrant
    else
        dists(f) = sqrt(xdist^2 + ydist^2);
    end
end
[dist, f] = max(dists);
fID = fIDs(f);
x = perfs.train.(perf_type).performances(ch, fID);
y = perfs.validate1.(perf_type).performances(ch, fID);
scatter(x, y, 'o', 'k');
disp(['best overall: fID:' num2str(fIDs(f)) ' perf(train)=' num2str(x) ' perf(validate1)=' num2str(y)]);

%% Plot discovery vs evaluation, after averaging across channels

subplot(subplot_rows, subplot_cols, [2 4]); hold on;

ch = 6;
ptype = 1;
perf_type = perf_types{ptype};

% Get features which are sig. in all datasets
data_set = data_sets{1};
sig = perfs.(data_sets{1}).(perf_type).sig;
for dset = 2 : length(data_sets)
    sig = sig & perfs.(data_sets{dset}).(perf_type).sig;
end
sig = sig(ch, :);

ch_valid_all = all(ch_valid_features, 1);
ch_valid_all = logical(ch_valid_all);

features = ch_valid_all;
fIDs = find(features);

% Plot class. vs consis.
scatter(...
    mean(perfs.train.(perf_type).performances(:, features), 1),...
    mean(perfs.validate1.(perf_type).performances(:, features), 1),...
    '.');
axis tight
xl = xlim;
yl = ylim;

% Plot chance
line([0.5 0.5], yl, 'Color', 'k');
line(xl, [0.5 0.5], 'Color', 'k');
% Plot significance thresholds
thresh = perfs.train.(perf_type).sig_thresh_fdr(ch);
line([thresh thresh], yl, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
thresh = perfs.validate1.(perf_type).sig_thresh_fdr(ch);
line(xl, [thresh thresh], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);

title(perf_type);
xlabel('discovery');
ylabel('pilot evaluation');
axis square

% Show best feature in each dataset
disp('===');
for dset = 1 : length(data_sets)
    data_set = data_sets{dset};
    [p, f] = max(mean(perfs.(data_set).(perf_type).performances(:, features), 1));
    scatter(...
        mean(perfs.train.(perf_type).performances(:, fIDs(f)), 1),...
        mean(perfs.validate1.(perf_type).performances(:, fIDs(f)), 1),...
        'o', 'k');
    disp(['best for ' data_set ': fID:' num2str(fIDs(f)) ' perf=' num2str(p)]);
end

% Show best overall feature (feature most towards the top-right)
%   For every point, find distance from (0.5 0.5) in top-right quadrant
dists = nan(size(fIDs));
for f = 1 : size(fIDs, 2)
    fID = fIDs(f);
    x = mean(perfs.train.(perf_type).performances(:, fID), 1);
    y = mean(perfs.validate1.(perf_type).performances(:, fID), 1);
    
    xdist = x - 0.5;
    ydist = y - 0.5;
    
    if xdist <= 0 || ydist <= 0
        dists(f) = NaN; % In the wrong quadrant
    else
        dists(f) = sqrt(xdist^2 + ydist^2);
    end
end
[dist, f] = max(dists);
fID = fIDs(f);
x = mean(perfs.train.(perf_type).performances(:, fID), 1);
y = mean(perfs.validate1.(perf_type).performances(:, fID), 1);
scatter(x, y, 'o', 'k');
disp(['best overall: fID:' num2str(fIDs(f)) ' perf(train)=' num2str(x) ' perf(validate1)=' num2str(y)]);

%% Plot number of sig features per channel

subplot(subplot_rows, subplot_cols, 5); hold on;

ptype = 1;
perf_type = perf_types{ptype};

sigs_all = [];

% Plot number of sig features for each dataset
for dset = 1 : length(data_sets)
    sig = perfs.(data_sets{dset}).(perf_type).sig & logical(ch_valid_features);
    sigs_all = cat(3, sigs_all, sig);
    nSig = sum(sig, 2);
    
    plot(nSig, 'Color', dset_colours{dset}(1,:));
end

% Plot number of features which were sig in both datasets
sigs_all = all(sigs_all, 3);
nSig = sum(sigs_all, 2);
plot(nSig, 'Color', 'k', 'LineStyle', '-.');

axis tight
yl = ylim;
ylim([0 yl(end)]);
xlim([1 length(nSig)]);

%title(perf_type);
xlabel('channel');
ylabel('N sig. features');

legend(data_sets{1}, data_sets{2}, 'both');

%% Plot highest performance per channel

subplot(subplot_rows, subplot_cols, 6); hold on;

ptype = 1;
perf_type = perf_types{ptype};

% Plot highest performance for each dataset
for dset = 1 : length(data_sets)
    best = max(perfs.(data_sets{dset}).(perf_type).performances, [], 2);
    
    plot(best, 'Color', dset_colours{dset}(1,:));
end

% Plot chance
line([1 length(best)], [0.5 0.5], 'Color', 'k', 'LineStyle', '-.');

axis tight
yl = ylim;
ylim([0.45 yl(end)]);
xlim([1 length(nSig)]);

%title(perf_type);
xlabel('channel');
ylabel('performance');

legend(data_sets{1}, data_sets{2}, 'chance');

%%

figure;
set(gcf, 'Color', 'w');

%% Plot class. vs consis. for a single channel

subplot(1, 2, 1); hold on;

ch = 5;
dset = 2;
data_set = data_sets{dset};

% Get features which are sig. in all datasets
sig = perfs.(data_set).(perf_types{1}).sig;
for ptype = 2 : length(perf_types)
    sig = sig & perfs.(data_set).(perf_types{ptype}).sig;
end
features = logical(ch_valid_features(ch, :)) & sig(ch, :);
features = logical(ch_valid_features(ch, :));
fIDs = find(features);

% Plot class. vs consis.
scatter(...
    perfs.(data_set).(perf_types{1}).performances(ch, features),...
    perfs.(data_set).(perf_types{2}).performances(ch, features),...
    '.');
axis tight
xl = xlim;
yl = ylim;

% Plot chance
line([0.5 0.5], yl, 'Color', 'k');
line(xl, [0.5 0.5], 'Color', 'k');
% Plot significance thresholds
thresh = perfs.(data_set).(perf_types{1}).sig_thresh_fdr(ch);
line([thresh thresh], yl, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
thresh = perfs.(data_set).(perf_types{1}).sig_thresh_fdr(ch);
line(xl, [thresh thresh], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);

title(['ch' num2str(ch) ' ' perf_types{1} ' vs ' perf_types{2}]);
xlabel(perf_types{1});
ylabel(perf_types{2});
axis square

% Show best feature in each dataset
disp('===');
for ptype = 1 : length(perf_types)
    perf_type = perf_types{ptype};
    [p, f] = max(perfs.(data_set).(perf_type).performances(ch, features));
    scatter(...
        perfs.(data_set).(perf_types{1}).performances(ch, fIDs(f)),...
        perfs.(data_set).(perf_types{2}).performances(ch, fIDs(f)),...
        'o', 'k');
    disp(['best for ' perf_type ': fID:' num2str(fIDs(f)) ' perf=' num2str(p)]);
end

% Show best overall feature (feature most towards the top-right)
%   For every point, find distance from (0.5 0.5) in top-right quadrant
dists = nan(size(fIDs));
for f = 1 : size(fIDs, 2)
    fID = fIDs(f);
    x = perfs.(data_set).(perf_types{1}).performances(ch, fID);
    y = perfs.(data_set).(perf_types{2}).performances(ch, fID);
    
    xdist = x - 0.5;
    ydist = y - 0.5;
    
    if xdist <= 0 || ydist <= 0
        dists(f) = NaN; % In the wrong quadrant
    else
        dists(f) = sqrt(xdist^2 + ydist^2);
    end
end
[dist, f] = max(dists);
fID = fIDs(f);
x = perfs.(data_set).(perf_types{1}).performances(ch, fID);
y = perfs.(data_set).(perf_types{2}).performances(ch, fID);
scatter(x, y, 'o', 'k');
disp(['best overall: fID:' num2str(fIDs(f)) ' perf(train)=' num2str(x) ' perf(validate1)=' num2str(y)]);

%% Plot class. vs consis. after averaging across channels

subplot(1, 2, 2); hold on;

ch = 6;
dset = 1;
data_set = data_sets{dset};

% Get features which are sig. in all datasets
sig = perfs.(data_set).(perf_types{1}).sig;
for ptype = 2 : length(perf_types)
    sig = sig & perfs.(data_set).(perf_types{ptype}).sig;
end
features = all(logical(ch_valid_features) & sig, 1);
features = all(logical(ch_valid_features), 1);
fIDs = find(features);

% Plot class. vs consis.
scatter(...
    mean(perfs.(data_set).(perf_types{1}).performances(:, features), 1),...
    mean(perfs.(data_set).(perf_types{2}).performances(:, features), 1),...
    '.');
axis tight
xl = xlim;
yl = ylim;

% Plot chance
line([0.5 0.5], yl, 'Color', 'k');
line(xl, [0.5 0.5], 'Color', 'k');
% Plot significance thresholds
thresh = perfs.(data_set).(perf_types{1}).sig_thresh_fdr(ch);
line([thresh thresh], yl, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
thresh = perfs.(data_set).(perf_types{1}).sig_thresh_fdr(ch);
line(xl, [thresh thresh], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);

title([perf_types{1} ' vs ' perf_types{2}]);
xlabel(perf_types{1});
ylabel(perf_types{2});
axis square

% Show best feature in each performance type
disp('===');
for ptype = 1 : length(perf_types)
    perf_type = perf_types{ptype};
    [p, f] = max(mean(perfs.(data_set).(perf_type).performances(:, features), 1));
    scatter(...
        mean(perfs.(data_set).(perf_types{1}).performances(:, fIDs(f)), 1),...
        mean(perfs.(data_set).(perf_types{2}).performances(:, fIDs(f)), 1),...
        'o', 'k');
    disp(['best for ' perf_type ': fID:' num2str(fIDs(f)) ' perf=' num2str(p)]);
end

% Show best overall feature (feature most towards the top-right)
%   For every point, find distance from (0.5 0.5) in top-right quadrant
dists = nan(size(fIDs));
for f = 1 : size(fIDs, 2)
    fID = fIDs(f);
    x = mean(perfs.(data_set).(perf_types{1}).performances(:, fID), 1);
    y = mean(perfs.(data_set).(perf_types{2}).performances(:, fID), 1);
    
    xdist = x - 0.5;
    ydist = y - 0.5;
    
    if xdist <= 0 || ydist <= 0
        dists(f) = NaN; % In the wrong quadrant
    else
        dists(f) = sqrt(xdist^2 + ydist^2);
    end
end
[dist, f] = max(dists);
fID = fIDs(f);
x = mean(perfs.(data_set).(perf_types{1}).performances(:, fID), 1);
y = mean(perfs.(data_set).(perf_types{2}).performances(:, fID), 1);
scatter(x, y, 'o', 'k');
disp(['best overall: fID:' num2str(fIDs(f)) ' perf(train)=' num2str(x) ' perf(validate1)=' num2str(y)]);

%% Portion of features which generalised

if strcmp(data_set, 'validate1')
    
    plot_positions = [4 5];
    letters = {'B', 'E'};
    
    plot_counter = 1;
    
    for ptype = 1 : length(perf_types)
        perf_type = perf_types{ptype};
        
        % Portion of sig features in the train set which generalised
        sig_both = perfs.train.(perf_type).sig & perfs.validate1.(perf_type).sig;
        extend_portion = sum(sig_both, 2) ./ sum(perfs.train.(perf_type).sig, 2);
        
        % Portion of sig features in the validate set which were not sig in
        % the train set
        sig_only = perfs.validate1.(perf_type).sig & ~perfs.train.(perf_type).sig;
        extend_bad = sum(sig_only, 2) ./ sum(~perfs.train.(perf_type).sig, 2);
        
        subplot(rows, columns, plot_positions(plot_counter));
        plot(extend_portion);
        hold on;
        plot(extend_bad);
        title([letters{ptype} ' ' perf_type ' generalisation']);
        xlabel('channel')
        ylabel('portion');
        xlim([1 length(extend_bad)]);
        
        plot_counter = plot_counter + 1;
    end
    
end
