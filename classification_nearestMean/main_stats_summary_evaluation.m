%% Description

%%

preprocess_string = '_subtractMean_removeLineNoise';

dsets = {'train', 'multidose', 'singledose', 'sleep'};

%% Load stats
% Get all performances and stats using
%   get_stats_evaluation(preprocess_string)

stats_dir = ['results' preprocess_string filesep];
stats_file = 'stats.mat';

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

%% Todo - show performances: train vs 1 evaluation set
% what about train vs all evaluation sets?
%   different marker for each dataset? 3D plot?


%% Plot all performances

perf_type = 'consis'; % nearestMedian or consis

figure;
sp_counter = 1;
for d = 1 : length(dsets)
    for pair = 1 : size(stats.(dsets{d}).(perf_type).condition_pairs, 1)
        
        tmp = stats.(dsets{d}).(perf_type).performances{pair};
        tmp(~valid_all) = nan;
        
        subplot(4, 4, sp_counter);
        imagesc(tmp, [0 1]); c = colorbar;
        title([...
            perf_type ' '...
            dsets{d} newline...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 1)} 'x'...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 2)}],...
            'interpreter', 'none');
        
        xlabel('feature');
        ylabel('channel');
        
        ylabel(c, perf_type);
        
        sp_counter = sp_counter + 1;
        
    end
end

%% Plot all performances in histogram

ch = 6;
perf_type = 'consis'; % nearestMedian or consis
chance = 0.5;

figure;
sp_counter = 1;
for d = 1 : length(dsets)
    for pair = 1 : size(stats.(dsets{d}).(perf_type).condition_pairs, 1)
        
        tmp = stats.(dsets{d}).(perf_type).performances{pair}(ch, :);
        tmp_rand = stats.(dsets{d}).(perf_type).performances_random{pair}(1, :); % same rand distribution across channels
        tmp(~valid_all(ch, :)) = nan;
        tmp_rand(~valid_all(ch, :)) = nan;
        
        min_both = min([min(tmp) min(tmp_rand)]);
        max_both = max([max(tmp) max(tmp_rand)]);
        
        subplot(4, 4, sp_counter);
        hold on;
        
        % Histograms
        histogram(tmp_rand, (min_both:0.01:max_both), 'DisplayStyle', 'stairs');
        histogram(tmp, (min_both:0.01:max_both), 'DisplayStyle', 'stairs');
        title([...
            perf_type ' '...
            dsets{d} newline...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 1)} 'x'...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 2)} newline...
            'ch' num2str(ch)],...
            'interpreter', 'none');
        
        % Significance thresholds
        sig_thresh = stats.(dsets{d}).(perf_type).sig_thresh(pair);
        sig_thresh_fdr = stats.(dsets{d}).(perf_type).sig_thresh_fdr(ch, pair);
        line([chance chance], ylim, 'Color', 'k', 'LineStyle', '-');
        line([sig_thresh sig_thresh], ylim, 'Color', 'k', 'LineStyle', '--');
        line([sig_thresh_fdr sig_thresh_fdr], ylim, 'Color', 'k', 'LineStyle', ':');
        
        axis tight
        
        xlabel(perf_type);
        ylabel('count');
        
        sp_counter = sp_counter + 1;
        
    end
end

% Add legend in separate plot
subplot(4, 4, sp_counter);
hold on;
title(['legend']);
histogram(tmp, (0:1:1), 'DisplayStyle', 'stairs');
histogram(tmp, (0:1:1), 'DisplayStyle', 'stairs');
line([chance chance], ylim, 'Color', 'k', 'LineStyle', '-');
line([chance chance], ylim, 'Color', 'k', 'LineStyle', '--');
line([chance chance], ylim, 'Color', 'k', 'LineStyle', ':');
legend('random-dist', 'actual-dist', 'chance', 'p=.05', 'p_{fdr}=.05');

%% Plot all significances

perf_type = 'consis'; % nearestMedian or consis

figure;
sp_counter = 1;
for d = 1 : length(dsets)
    for pair = 1 : size(stats.(dsets{d}).(perf_type).condition_pairs, 1)
        
        tmp = single(stats.(dsets{d}).(perf_type).sig(:, :, pair));
        tmp(~valid_all) = nan;
        
        subplot(4, 4, sp_counter);
        imagesc(tmp); c = colorbar;
        title([...
            perf_type ' '...
            dsets{d} newline...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 1)} 'x'...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 2)}],...
            'interpreter', 'none');
        
        xlabel('feature');
        ylabel('channel');
        
        ylabel(c, 'p_{fdr} < 0.05');
        
        sp_counter = sp_counter + 1;
        
    end
end

%% Features with significant above-chance performance across all datasets
% Features need to achieve above-chance performance in all datasets
% For datasets with multiple condition pairings, must be significant in all
%   pairings

perf_type = 'nearestMedian'; % nearestMedian or consis

sig_all = ones(size(stats.train.(perf_type).performances{1}));
sig_all_count = zeros(size(sig_all)); % count how many datasets in which feature is significant

for d = 1 : length(dsets)
    for pair = 1 : size(stats.(dsets{d}).(perf_type).condition_pairs, 1)
        
        tmp = stats.(dsets{d}).(perf_type).sig(:, :, pair);
        sig_all = sig_all & tmp;
        sig_all_count = sig_all_count + tmp;
        
    end
end

sig_all(~valid_all) = 0;
sig_all_count(~valid_all) = 0;

% Plot dataset/pairing significance counts at various thresholds

figure;
sp_counter = 1;
m = max(sig_all_count(:));
mPlots = 12; % m plots, plus two more
for count = 1 : m
    
    subplot(floor(sqrt(mPlots)), ceil(sqrt(mPlots)), sp_counter);
    imagesc(sig_all_count >= count); c = colorbar;
    
    title([...
        perf_type ' '...
        num2str(numel(find(sig_all_count >= count))) ' sig.' newline...
        'in at least ' newline...
        num2str(count) ' dsets/cond-pairs']);
    xlabel('feature');
    ylabel('channel');
    ylabel(c, 'sig.');
    
    sp_counter = sp_counter + 1;
end

sp_counter = mPlots-1;
subplot(floor(sqrt(mPlots)), ceil(sqrt(mPlots)), sp_counter);
imagesc(sig_all); c = colorbar;
title([...
    perf_type ' ' num2str(numel(find(sig_all))) ' sig.' newline...
    'in all dsets/cond-pairs']);
xlabel('channel');
ylabel('feature');
ylabel(c, 'sig. in all');
sp_counter = sp_counter + 1;

subplot(floor(sqrt(mPlots)), ceil(sqrt(mPlots)), sp_counter);
imagesc(sig_all_count); c = colorbar;
title('sig. in N dsets/cond-pairs');
ylabel(c, 'N');
xlabel('feature');
ylabel('channel');

% Display indices of surviving features
[i, j] = ind2sub(size(sig_all), find(sig_all));
if numel([i j]) == 0
    disp('no features sig. in all datasets');
else
    fprintf('\t\tchannel\t\tfeature\n');
    disp([i j]);
end

%% Features with significant above-chance performance across specific datasets
% Features need to achieve above-chance performance in specified datasets
% For datasets with multiple condition pairings, must be significant in all
%   pairings
% TODO - this is pretty much a copy of the previous section
%   Consider turning into a function

perf_type = 'nearestMedian'; % nearestMedian or consis

% TODO - create function which gets condition pairs which include only the
%   specified conditions
dsets_specific = {'train', 'multidose', 'singledose', 'sleep'};
pairs_specific = {[1], [4], [1], [1]};

sig_all = ones(size(stats.train.(perf_type).performances{1}));
sig_all_count = zeros(size(sig_all)); % count how many datasets in which feature is significant

for d = 1 : length(dsets_specific)
    for pair = 1 : length(pairs_specific{d})%size(stats.(dsets{d}).(perf_type).condition_pairs, 1)
        
        tmp = stats.(dsets_specific{d}).(perf_type).sig(:, :, pairs_specific{d}(pair));
        sig_all = sig_all & tmp;
        sig_all_count = sig_all_count + tmp;
        
    end
end

sig_all(~valid_all) = 0;
sig_all_count(~valid_all) = 0;

% Plot dataset/pairing significance counts at various thresholds

figure;
sp_counter = 1;
m = max(sig_all_count(:));
mPlots = 12; % m plots, plus two more
for count = 1 : m
    
    subplot(floor(sqrt(mPlots)), ceil(sqrt(mPlots)), sp_counter);
    imagesc(sig_all_count >= count); c = colorbar;
    
    title([...
        perf_type ' '...
        num2str(numel(find(sig_all_count >= count))) ' sig.' newline...
        'in at least ' newline...
        num2str(count) ' dsets/cond-pairs']);
    xlabel('feature');
    ylabel('channel');
    ylabel(c, 'sig.');
    
    sp_counter = sp_counter + 1;
end

sp_counter = mPlots-1;
subplot(floor(sqrt(mPlots)), ceil(sqrt(mPlots)), sp_counter);
imagesc(sig_all); c = colorbar;
title([...
    perf_type ' ' num2str(numel(find(sig_all))) ' sig.' newline...
    'in specified dsets/cond-pairs']);
xlabel('channel');
ylabel('feature');
ylabel(c, 'sig. in all');
sp_counter = sp_counter + 1;

subplot(floor(sqrt(mPlots)), ceil(sqrt(mPlots)), sp_counter);
imagesc(sig_all_count); c = colorbar;
title('sig. in N dsets/cond-pairs');
ylabel(c, 'N');
xlabel('feature');
ylabel('channel');

%{
% Display indices of surviving features
[i, j] = ind2sub(size(sig_all), find(sig_all));
if numel([i j]) == 0
    disp('no features sig. in specified datasets');
else
    fprintf('\t\tchannel\t\tfeature\n');
    disp([i j]);
end
%}

%% Plot all performances (batchNormalised)

perf_type = 'nearestMedian'; % nearestMedian or consis
dset_suffixes = cat(2, {''}, repmat({'BatchNormalised'}, [1, length(dsets)]));

figure;
sp_counter = 2;
for d = 2 : length(dsets)
    for pair = 1 : size(stats.(dsets{d}).(perf_type).condition_pairs, 1)
        
        tmp = stats.([dsets{d} dset_suffixes{d}]).(perf_type).performances{pair};
        tmp(~valid_all) = nan;
        
        subplot(4, 4, sp_counter);
        imagesc(tmp, [0 1]); c = colorbar;
        title([...
            perf_type ' '...
            dsets{d} newline...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 1)} 'x'...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 2)}],...
            'interpreter', 'none');
        
        xlabel('feature');
        ylabel('channel');
        
        ylabel(c, perf_type);
        
        sp_counter = sp_counter + 1;
        
    end
end

%% Plot all performances in histogram (batchNormalised)

ch = 6;
perf_type = 'nearestMedian'; % nearestMedian
dset_suffixes = cat(2, {''}, repmat({'BatchNormalised'}, [1, length(dsets)]));

chance = 0.5;

figure;
sp_counter = 2;
for d = 2 : length(dsets)
    for pair = 1 : size(stats.(dsets{d}).(perf_type).condition_pairs, 1)
        
        tmp = stats.([dsets{d} dset_suffixes{d}]).(perf_type).performances{pair}(ch, :);
        tmp_rand = stats.([dsets{d} dset_suffixes{d}]).(perf_type).performances_random{pair}(1, :); % same rand distribution across channels
        tmp(~valid_all(ch, :)) = nan;
        tmp_rand(~valid_all(ch, :)) = nan;
        
        min_both = min([min(tmp) min(tmp_rand)]);
        max_both = max([max(tmp) max(tmp_rand)]);
        
        subplot(4, 4, sp_counter);
        hold on;
        
        % Histograms
        histogram(tmp_rand, (min_both:0.01:max_both), 'DisplayStyle', 'stairs');
        histogram(tmp, (min_both:0.01:max_both), 'DisplayStyle', 'stairs');
        title([...
            perf_type 'BN '...
            dsets{d} newline...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 1)} 'x'...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 2)} newline...
            'ch' num2str(ch)],...
            'interpreter', 'none');
        
        % Significance thresholds
        sig_thresh = stats.(dsets{d}).(perf_type).sig_thresh(pair);
        sig_thresh_fdr = stats.(dsets{d}).(perf_type).sig_thresh_fdr(ch, pair);
        line([chance chance], ylim, 'Color', 'k', 'LineStyle', '-');
        line([sig_thresh sig_thresh], ylim, 'Color', 'k', 'LineStyle', '--');
        line([sig_thresh_fdr sig_thresh_fdr], ylim, 'Color', 'k', 'LineStyle', ':');
        
        axis tight
        
        xlabel(perf_type);
        ylabel('count');
        
        sp_counter = sp_counter + 1;
        
    end
end

% Add legend in separate plot
subplot(4, 4, sp_counter);
hold on;
title(['legend']);
histogram(tmp, (0:1:1), 'DisplayStyle', 'stairs');
histogram(tmp, (0:1:1), 'DisplayStyle', 'stairs');
line([chance chance], ylim, 'Color', 'k', 'LineStyle', '-');
line([chance chance], ylim, 'Color', 'k', 'LineStyle', '--');
line([chance chance], ylim, 'Color', 'k', 'LineStyle', ':');
legend('random-dist', 'actual-dist', 'chance', 'p=.05', 'p_{fdr}=.05');

%% Plot all significances (batchNormalised)

perf_type = 'nearestMedian';
dset_suffixes = cat(2, {''}, repmat({'BatchNormalised'}, [1, length(dsets)]));

% Discovery dataset doesn't go through batch normalisation
%   Evaluation datasets are normalised based on discovery dataset
figure;
sp_counter = 2;
for d = 2 : length(dsets)
    for pair = 1 : size(stats.(dsets{d}).(perf_type).condition_pairs, 1)
        
        tmp = single(stats.([dsets{d} dset_suffixes{d}]).(perf_type).sig(:, :, pair));
        tmp(~valid_all) = nan;
        
        subplot(4, 4, sp_counter);
        imagesc(tmp); c = colorbar;
        title([...
            perf_type 'BN '...
            dsets{d} newline...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 1)} 'x'...
            stats.(dsets{d}).(perf_type).conditions{stats.(dsets{d}).(perf_type).condition_pairs(pair, 2)}],...
            'interpreter', 'none');
        
        xlabel('feature');
        ylabel('channel');
        
        ylabel(c, 'p_{fdr} < 0.05');
        
        sp_counter = sp_counter + 1;
        
    end
end

%% Features with significant above-chance classification (batch normalised)

perf_type = 'nearestMedian';
dset_suffixes = cat(2, {''}, repmat({'BatchNormalised'}, [1, length(dsets)]));

% 1 if significant across all evaluation datasets
%   "BN" - "batchNormalised"
sig_all_BN = ones(size(stats.train.(perf_type).performances{1}));
sig_all_count_BN = zeros(size(sig_all_BN)); % count how many datasets in which feature is significant

% Discovery dataset doesn't go through batch normalisation
%   Evaluation datasets are normalised based on discovery dataset
for d = 2 : length(dsets)
    for pair = 1 : size(stats.(dsets{d}).(perf_type).condition_pairs, 1)
        
        tmp = stats.([dsets{d} dset_suffixes{d}]).(perf_type).sig(:, :, pair);
        sig_all_BN = sig_all_BN & tmp;
        sig_all_count_BN = sig_all_count_BN + tmp;
        
    end
end

sig_all(~valid_all) = 0;
sig_all_count(~valid_all) = 0;

% Plot dataset/pairing significance counts at various thresholds

figure;
sp_counter = 2;
m = max(sig_all_count_BN(:));
mPlots = 12; % m plots, plus two more
for count = 1 : m
    
    subplot(floor(sqrt(mPlots)), ceil(sqrt(mPlots)), sp_counter);
    imagesc(sig_all_count_BN >= count); c = colorbar;
    
    title([...
        perf_type 'BN '...
        num2str(numel(find(sig_all_count_BN >= count))) ' sig.' newline...
        'in at least ' newline...
        num2str(count) ' dsets/cond-pairs']);
    xlabel('feature');
    ylabel('channel');
    ylabel(c, 'sig.');
    
    sp_counter = sp_counter + 1;
end

sp_counter = mPlots-1;
subplot(floor(sqrt(mPlots)), ceil(sqrt(mPlots)), sp_counter);
imagesc(sig_all); c = colorbar;
title([...
    perf_type ' ' num2str(numel(find(sig_all_BN))) ' sig.' newline...
    'in all dsets/cond-pairs']);
xlabel('channel');
ylabel('feature');
ylabel(c, 'sig. in all');
sp_counter = sp_counter + 1;

subplot(floor(sqrt(mPlots)), ceil(sqrt(mPlots)), sp_counter);
imagesc(sig_all_count_BN); c = colorbar;
title('sig. in N dsets/cond-pairs');
ylabel(c, 'N');
xlabel('feature');
ylabel('channel');

% Display indices of surviving features
[i, j] = ind2sub(size(sig_all_BN), find(sig_all_BN));
if numel([i j]) == 0
    disp('no features sig. in all datasets');
else
    fprintf('\t\tchannel\t\tfeature\n');
    disp([i j]);
end

%% Plot raw feature values

% high performing feature
fID = 6904;
ch = 4;

% high mean-DNV (in sleep flies), but low consistency
fID = 1246;
ch = 6;

%%

% Following code to go into
%   /fly_blind_classify/main_hctsa_featureValueDistribution

addpath('../');

values = featureValueDistribution(fID, ch);

%%
% Plot values
figure;
for d = 1 : length(dsets)
    
    subplot(length(dsets)+1, 1, d);
    hold on;
    
    xpos = 1;
    [nEpochs, nFlies, nConditions] = size(values{d});
    [conditions, cond_labels, cond_colours] = getConditions(dsets{d});
    x_cond_offsets = (0 : 1/length(conditions) : 0.8);
    
    a = -1/30; b = 1/30;
    xspread = a + (b-a).*rand(nEpochs, 1);
    
    for fly = 1 : nFlies
        
        for c = 1 : length(conditions)
            
            % Plot each epoch value
            xs = repmat(xpos+x_cond_offsets(c), [nEpochs 1]);
            scatter(xs+xspread, values{d}(:, fly, c), [], cond_colours{c}, '.', 'MarkerEdgeAlpha', 0.5);
            
        end
        
        % Plot line joining medians per condition
        xs = xpos+x_cond_offsets;
        plot(xs, squeeze(median(values{d}(:, fly, :), 1)), 'k');
        
        xpos = xpos + 1;
    end
    
    % Title with feature details and performance
    nPairs = length(stats.(dsets{d}).consis.performances);
    perf_labels = cell(nPairs, 1);
    for p = 1 : nPairs
        perf_labels{p} = [...
            strjoin(stats.(dsets{d}).consis.conditions(stats.(dsets{d}).consis.condition_pairs(p, :)), 'x') '='...
            num2str(stats.(dsets{d}).consis.performances{p}(ch, fID))...
            ];
    end
    
    title(['f' num2str(fID) ' ch' num2str(ch) newline 'consis ' strjoin(perf_labels, ' : ')], 'interpreter', 'none');
    
    set(gca, 'XTick', []);
    xlabel(strjoin(conditions, ' : '), 'interpreter', 'none');
    
    if d == 2
        set(gca, 'YScale', 'log');
        ylabel([dsets{d} newline 'log(value)']);
    else
        ylabel([dsets{d} newline 'value']);
    end
    
    axis tight;
    
end

% Plot all on one axis
subplot(length(dsets)+1, 1, d+1);
hold on;
xpos = 1;
for d = 1 : length(dsets)
    [nEpochs, nFlies, nConditions] = size(values{d});
    [conditions, cond_labels, cond_colours] = getConditions(dsets{d});
    x_cond_offsets = (0 : 1/length(conditions) : 0.8);
    
    a = -1/30; b = 1/30;
    xspread = a + (b-a).*rand(nEpochs, 1);
    
    for fly = 1 : nFlies
        
        for c = 1 : length(conditions)
            
            % Plot each epoch value
            xs = repmat(xpos+x_cond_offsets(c), [nEpochs 1]);
            scatter(xs+xspread, values{d}(:, fly, c), [], cond_colours{c}, '.', 'MarkerEdgeAlpha', 0.5);
            
        end
        
        % Plot line joining medians per condition
        xs = xpos+x_cond_offsets;
        plot(xs, squeeze(median(values{d}(:, fly, :), 1)), 'k');
        
        xpos = xpos + 1;
    end
    
    xpos = xpos + 0.5; % Add extra space between datasets
end
title(['f' num2str(fID) ' ch' num2str(ch)]);
set(gca, 'YScale', 'log');
ylabel(['all' newline 'log(value)']);
axis tight;

%% Plot performances of significant features

perf_type = 'nearestMedian';
ch = 1;

nFeatures = numel(find(sig_all(ch, :)));
values = nan(length(dsets_specific));

% Sort performances of one dataset/pair
%   Use this order for all the others
perfs = stats.(dsets_specific{1}).(perf_type).performances{1}(ch, sig_all(ch, :));
[B, I] = sort(perfs);

figure;
sp_counter = 1;
for d = 1 : length(dsets_specific)
    for pair = 1 : length(pairs_specific{d})
        tmp = stats.(dsets_specific{d}).(perf_type).performances{pairs_specific{d}(pair)}(ch, sig_all(ch, :));
        tmp = tmp(I);
        
        subplot(length(dsets), max(cellfun(@length, pairs_specific)), sp_counter);
        histogram(tmp);
        
        sp_counter = sp_counter + 1;
    end
end