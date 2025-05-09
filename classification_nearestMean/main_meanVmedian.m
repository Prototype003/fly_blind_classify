%% Description

%{

Compare nearest mean to nearest median classification performance

%}

%% Settings

class_set = 'crossValidation'; % crossValidation; validate1_accuracy
class_set = 'validate1_accuracy';

class_types = {'nearestMean', 'nearestMedian'};

source_dir = 'results/';

hctsa_prefix = '../hctsa_space/HCTSA_train';
%hctsa_prefix = '../hctsa_space/HCTSA_validate1';

%% Load

% Accuracies
source_file = ['class_' class_types{1} '_' class_set '.mat'];
acc = load([source_dir source_file]);
source_file = ['class_' class_types{2} '_' class_set '.mat'];
acc_new = load([source_dir source_file]);

%% Get average accuracy across cross-validations for each feature

% Average accuracies across cross-validations
accuracies = mean(acc.accuracies, 3);
accuracies_new = mean(acc_new.accuracies, 3);

%% Load chance distribution

rand_file = ['class_random_' class_set];
acc_random = load([source_dir rand_file]);
accuracies_random = acc_random.accuracies_random;

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

%% Sort features by performance

% Get best channel (with largest number of sig. features)
ch_best = find(sum(sig, 2) == max(sum(sig, 2)));
ch_worst = find(sum(sig, 2) == min(sum(sig, 2)));

% Sort features by performance (use same order for all channels
[sorted, order] = sort(accuracies(ch_best, :), 'descend');
accuracies_sorted = accuracies(:, order);
accuracies_new_sorted = accuracies_new(:, order);

% Limit to only show features which are sig. at least once
sig_features = sum(sig, 1);
sig_features = sig_features(order) > 0;

%% Plot difference in accuracies

r = linspace(0, 1, size(accuracies, 1))';
g = linspace(0, 0, size(accuracies, 1))';
b = linspace(1, 0, size(accuracies, 1))';
ch_colours = cat(2, r, g, b);

figure;
subplot(3, 2, [1 2]); hold on;
x = (1:sum(sig_features));
plot(x, accuracies_new_sorted(ch_best, sig_features), 'Color', ch_colours(ch_best, :), 'LineStyle', ':');
plot(x, accuracies_sorted(ch_best, sig_features), 'Color', ch_colours(ch_best, :), 'LineWidth', 2); % reference
legend({'median', 'mean'});
title([class_set ' sig. features sorted by ch' num2str(ch_best)], 'interpreter', 'none');
ylabel('accuracy'); xlabel(['feature (sorted by ch' num2str(ch_best) ')']);

subplot(3, 2, [3 4])
plot(x, accuracies_sorted(ch_best, sig_features)-accuracies_new_sorted(ch_best, sig_features), 'Color', ch_colours(ch_best, :)); % best ch
title('mean - median');

%% Plot correlation

subplot(3, 2, 5);
scatter(accuracies(ch_best, :), accuracies_new(ch_best, :), '.');
r = corr(accuracies(ch_best, :)', accuracies_new(ch_best, :)');
title(['ch' num2str(ch_best) ' r=' num2str(r)]);
xlabel('nearest-mean acc.');
ylabel('nearest-median acc.');

subplot(3, 2, 6);
scatter(accuracies(:), accuracies_new(:), '.');
r = corr(accuracies(:), accuracies_new(:));
title(['all channels r=' num2str(r)]);
xlabel('nearest-mean acc.');
ylabel('nearest-median acc.');