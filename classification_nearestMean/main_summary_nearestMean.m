%% Description

%{

Plot and get summary stats for training set cross-validation

%}

%% Settings

class_set = 'validate1_accuracy'; % crossValidation; validate1_accuracy
%class_set = 'validate1_accuracy';

%class_type = 'nearestMean'; % nearest mean classification
class_type = 'nearestMedian'; % nearest median classification

source_file = ['class_' class_type '_' class_set '.mat'];
source_dir = 'results/';

% For determining valid features
hctsa_prefix = '../hctsa_space/HCTSA_train';
%hctsa_prefix = '../hctsa_space/HCTSA_validate1';

%% Load

% Accuracies
acc = load([source_dir source_file]);

%% Get average accuracy across cross-validations for each feature

% Average accuracies across cross-validations
accuracies = mean(acc.accuracies, 3);

%% Load chance distribution

rand_file = ['class_random_' class_set];
acc_random = load([source_dir rand_file]);
accuracies_random = acc_random.accuracies_random;

%% Plot distribution across all features for each channel

r = linspace(0, 1, size(accuracies, 1))';
g = linspace(0, 0, size(accuracies, 1))';
b = linspace(1, 0, size(accuracies, 1))';
ch_colours = cat(2, r, g, b);

figure; hold on;
for ch = 1 : size(accuracies, 1)
    
    % Plot cumulative histogram
    h = cdfplot(accuracies(ch, :));
    set(h, 'Color', ch_colours(ch, :));
    
    % Show in legend
    if ch ~= 1 && ch ~= size(accuracies, 1)
        h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    
end

legend('ch1', 'ch15', 'Location', 'northwest');

%% Get valid features per channel

ch_valid_features = nan(size(accuracies, 1), size(accuracies, 2));
ch_excluded = zeros(size(accuracies, 1), 2); % 2 exclusion stages

for ch = 1 : size(accuracies, 1)
    tic;
    hctsa = load([hctsa_prefix '_channel' num2str(ch) '.mat']);
    [valid_ids, valid] = getValidFeatures(hctsa.TS_DataMat);
    ch_valid_features(ch, :) = valid_ids; % store
    ch_excluded(ch, :) = valid;
    toc
end

%% Number of valid features

nValid = sum(ch_valid_features, 2);
nValid_mean = mean(nValid)
nValid_min = min(nValid)
nValid_max = max(nValid)

%% Find significantly performing features
% One-tailed, better than chance
% p-value from distribution: https://www.jwilber.me/permutationtest/

alpha = 0.05;
q = 0.05;

% Get threshold from chance distribution
chance_dist = accuracies_random(1, :);
sig_thresh = prctile(chance_dist, (1-alpha)*100); % 95%tile -> alpha = 0.05

% Compare each feature to threshold, get p-value
ps = nan(size(accuracies));
for ch = 1 : size(accuracies, 1)
    for f = 1 : size(accuracies, 2)
        % Find how many in chance dist are greater
        nBetter = sum(chance_dist > accuracies(ch, f));
        % p-value
        ps(ch, f) = nBetter / numel(chance_dist);
    end
end

% Conduct FDR correction per channel
ps_fdr = nan(size(accuracies, 1), 1);
sig_thresh_fdr = nan(size(ps_fdr));
for ch = 1 : size(accuracies, 1)
    
    % FDR
    [pID, pN] = FDR(ps(ch, find(ch_valid_features(ch, :))), q);
    ps_fdr(ch) = pID; % nonparametric
    
    % Get corresponding accuracy
    sig_thresh_fdr(ch) = prctile(chance_dist, (1-ps_fdr(ch))*100);
end

% Number of significantly performing features
sig = ps < repmat(ps_fdr, [1, size(ps, 2)]);

%% Plot performance of significant features

% Get accuracies where features are significant
perf = accuracies;
perf(~sig) = nan;

% Limit matrix to only show features which are sig. at least once
sig_features = sum(sig, 1);
sig_features = sig_features > 0;

figure;
imagesc(perf(:, sig_features)); colorbar;

%% Get best feature for each channel
ch_best_perf = cell(size(accuracies, 1), 3);

for ch = 1 : size(accuracies)
    
    % Find best feature
    [sorted, order] = sort(accuracies(ch, :), 'descend');
    
    % Look up the feature
    perf = sorted(1);
    fID = hctsa.Operations{order(1), 'ID'};
    fName = hctsa.Operations{order(1), 'Name'};
    
    % Look up master feature
    mID = hctsa.Operations{order(1), 'MasterID'};
    mName = hctsa.MasterOperations{mID, 'Label'};
    
    % Store/print
    ch_best_perf{ch, 1} = ch;
    ch_best_perf{ch, 2} = perf;
    ch_best_perf{ch, 3} = fID;
    ch_best_perf{ch, 4} = fName{1};
    ch_best_perf{ch, 5} = mID;
    ch_best_perf{ch, 6} = mName{1};
    
end

% Convert to more readable table
ch_best_perf = cell2table(ch_best_perf, 'VariableNames', {'ch', 'perf', 'featureID', 'featureName', 'masterID', 'masterFeature'});
ch_best_perf

%% Get best feature after averaging across channels

% Consider features which are valid for ALL channels
valid_all = sum(ch_valid_features, 1);
valid_all = valid_all == size(ch_valid_features, 1);

% Average performance across channels
accuracies_mean = mean(accuracies, 1);
[perf, location] = max(accuracies_mean);

% Get feature details
fID = hctsa.Operations{location, 'ID'};
fName = hctsa.Operations{location, 'Name'};
mID = hctsa.Operations{location, 'MasterID'};
mName = hctsa.MasterOperations{mID, 'Label'};

% Display
disp([num2str(perf) ' ' num2str(fID) ' ' fName{1} ' ' num2str(mID) ' ' mName{1}]);

%% Plot distribution across valid features for each channel
% Show distributions per channel
% Get valid features per channel

r = linspace(0, 1, size(accuracies, 1))';
g = linspace(0, 0, size(accuracies, 1))';
b = linspace(1, 0, size(accuracies, 1))';
ch_colours = cat(2, r, g, b);

figure; hold on;
%subplot(2, 2, 1); hold on;
subplot(4, 2, 1); hold on;
for ch = 1 : size(accuracies, 1)
    
    % Plot cumulative histogram
    h = cdfplot(accuracies(ch, find(ch_valid_features(ch, :))));
    set(h, 'Color', ch_colours(ch, :));
    
    % Show in legend
    if ch ~= 1 && ch ~= size(accuracies, 1)
        h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
    
    % Show significance cutoff
    match = single(abs(h.XData - sig_thresh_fdr(ch)));
    match = find(match == min(match)); 
    s = scatter(h.XData(match(2)), h.YData(match(2)), 50, ch_colours(ch, :), '>', 'filled');
    s.Annotation.LegendInformation.IconDisplayStyle = 'off';
    
    % Highlight particular channel
    if ch == 6
        set(h, 'LineWidth', 2);
        h.Annotation.LegendInformation.IconDisplayStyle = 'on';
    end
    
end

% Plot chance
chance_colour = [0 0.8 0];
h = cdfplot(accuracies_random(1, :)); % only need 1 "channel"
set(h, 'Color', chance_colour);
set(h, 'LineWidth', 2);
% Show significance cutoff
match = single(abs(h.XData - sig_thresh));
match = h.YData(find(match == min(match)));
scatter(sig_thresh, match(2), 50, chance_colour, '>', 'filled');

% Plot average across channels
chMean_colour = [0 0 0];
h = cdfplot(accuracies_mean(valid_all));
set(h, 'Color', chMean_colour);
set(h, 'LineWidth', 2);
% Show significance cutoff?

legend('ch1', 'ch6', 'ch15', 'chance', 'p=0.05', 'chMean', 'Location', 'northwest');

title(class_set, 'interpreter', 'none');
xlabel('accuracy');
ylabel('portion of valid features');

xlim([0.4 0.6]);

%% Sort features by performance

% Get best channel (with largest number of sig. features)
ch_best = find(sum(sig, 2) == max(sum(sig, 2)));
ch_worst = find(sum(sig, 2) == min(sum(sig, 2)));
ch_best = 6;

% Sort features by performance (use same order for all channels
[sorted, order] = sort(accuracies(ch_best, :), 'descend');
accuracies_sorted = accuracies(:, order);

% Limit to only show features which are sig. at least once
sig_features = sum(sig, 1);
sig_features = sig_features(order) > 0;

% Plot
%figure; hold on;
% subplot(2, 2, 2); hold off;
% for ch = 1 : size(accuracies, 1)
%     if ch ~= ch_best
%         h = plot(accuracies_sorted(ch, sig_features), 'Color', ch_colours(ch, :));
%         h.Annotation.LegendInformation.IconDisplayStyle = 'off';
%     end
% end
% % Plot reference channel on top of everything else
% h = plot(accuracies_sorted(ch_best, sig_features), 'Color', ch_colours(ch_best, :));
% h.LineWidth = 5;
% legend(['ch' num2str(ch_best)]);
% title(['sig. features sorted by ch' num2str(ch_best)]);
% ylabel('accuracy'); xlabel(['feature (sorted by ch' num2str(ch_best) ')']);

% Plot mean+std across channels
%figure; hold on;
%subplot(2, 2, 2); hold on;
subplot(4, 2, 7); hold on;
y = mean(accuracies_sorted(:, sig_features), 1);
yerr = std(accuracies_sorted(:, sig_features), [], 1);
x = (1:sum(sig_features));
h = patch([x fliplr(x)], [y+yerr fliplr(y-yerr)], 'k', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); % std
h.Annotation.LegendInformation.IconDisplayStyle = 'off';
h = plot(x, y, 'k'); % mean
h.Annotation.LegendInformation.IconDisplayStyle = 'off';
plot(x, accuracies_sorted(ch_best, sig_features), 'Color', ch_colours(ch_best, :), 'LineWidth', 2); % best ch
line([1 max(x)], [0.5 0.5]); % chance line
% plot(x, accuracies_sorted(ch_worst, sig_features), 'Color', ch_colours(ch_worst, :)); % worst ch
legend(['ch' num2str(ch_best)]);
title(['sig. features sorted by ch' num2str(ch_best)]);
ylabel('accuracy'); xlabel(['feature (sorted by ch' num2str(ch_best) ')']);
axis tight;

%% Get dimensions

% Get data dimensions
if strcmp(class_set, 'crossValidation')
    tic;
    tmp = load('../data/preprocessed/fly_data_removeLineNoise.mat');
    nChannels = size(tmp.data.train, 2);
    nEpochs = size(tmp.data.train, 3);
    nFlies = size(tmp.data.train, 4);
    nConditions = size(tmp.data.train, 5);
    toc
elseif strcmp(class_set, 'HCTSA_validate1')
    % Need to figure out dimensions based on labelled datafile
elseif strcmp(class_set, 'HCTSA_validate2')
    % Need to figure out dimensions based on labelled datafile
end

%% Distribution of best feature
% Distribution of raw values for best feature from the best channel

% Load feature values
tic
hctsa = load([hctsa_prefix '_channel' num2str(ch_best) '.mat']);
toc;

% Best feature
[perf, f_best] = max(accuracies(ch_best, :));

rain_format = cell(nFlies, nConditions);
for fly = 1 : nFlies
    for c = 1 : nConditions
        keys = {['fly' num2str(fly)], ['condition' num2str(c)]};
        ids = getIds(keys, hctsa.TimeSeries);
        rain_format{fly, c} = hctsa.TS_DataMat(ids, f_best);
    end
end

figure;
rm_raincloud(rain_format, [1 0 0; 0 0 1]);