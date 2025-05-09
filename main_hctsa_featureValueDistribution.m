%%

%{

Create figure showing distribution of hctsa values across all time-series

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

%% Load feature thresholds and directions

thresh_dir = ['classification_nearestMean/results' preprocess_string filesep];
thresh_file = 'class_nearestMedian_thresholds.mat';
load([thresh_dir thresh_file]);

%% Feature and channel to plot for

% Significance feature (classification)

fID = 7200;
ch = 6;
perf_type = 'nearestMedian';
%%
% high performing feature (consistency)
fID = 6904;
ch = 4;
perf_type = 'consis';
%%
% high mean-DNV (in sleep flies), but low consistency
%fID = 1246;
%ch = 6;
%perf_type = 'consis';
%%
% This feature achieves sig. classification performance in all dsets
% but not for batchNormalised or for consistency
fID = 7111;
ch = 6;
perf_type = 'consis';
%%
tic;
values = featureValueDistribution(fID, ch);
toc

%% Plot all values from every fly

% Discovery N=13 (2 conditions) | Singledose N=18 (3 conditions)
% Multidose N=12 (5 conditions)
% Sleep N=19 (4 conditions)

fig = figure;

% Subplot positions for each dataset
dset_positions = {...
    [3 80 1 26],... % 80 is (13*2=26) + (18*3=54)
    [3 120 1+120 80+120],...
    [3 120 81+120 120+120],...
    [3 80 27 80],... % same row as discovery flies
    [3 1 3]... % this row only shows one dataset
    };

% Determine global range of values
dset_min = (min(cellfun(@(x) min(x, [], 'all'), values)));
dset_max = (max(cellfun(@(x) max(x, [], 'all'), values)));

% Which condition pairings to highlight in each dataset
dset_pairings = {...
    [1 2],... % wake x anest
    [1 3],... % wake x iso1.2
    [1 3],... % wake x iso1.2
    [1 2],... % wake x iso0.6
    [2 3]}; % wake x sleepEarly
dset_pairIDs = [1 1 1 1 3];

for d = 1 : length(dsets)
    
    subplot(dset_positions{d}(1), dset_positions{d}(2), dset_positions{d}(3:end));
    hold on;
    
    xpos = 1;
    [nEpochs, nFlies, nConditions] = size(values{d});
    [conditions, cond_labels, cond_colours] = getConditions(dsets{d});
    x_cond_offsets = (0 : 0.9/length(conditions) : 0.9);
    x_cond_offsets = x_cond_offsets(1:length(conditions));
    
    a = -1/30; b = 1/30;
    xspread = a + (b-a).*rand(nEpochs, 1);
    
    for fly = 1 : nFlies
        
        for c = 1 : length(conditions)
            
            % Plot each epoch value
            xs = repmat(xpos+x_cond_offsets(c), [nEpochs 1]);
            scatter(xs+xspread, (values{d}(:, fly, c)), 50/log(size(values{d}, 1)), cond_colours{c}, 'o', 'MarkerEdgeAlpha', 0.5);
            
        end
        
        % Plot line joining medians per condition
        xs = xpos+x_cond_offsets;
        plot(xs, (squeeze(median(values{d}(:, fly, :), 1))), 'k:', 'LineWidth', 1);
        
        % Plot lines illustrating condition pairings
        for pair = 1 : size(dset_pairings{d}, 1)
            plot(xs(dset_pairings{d}(pair, :)),...
                (squeeze(median(values{d}(:, fly, dset_pairings{d}(pair, :)), 1))), 'k', 'LineWidth', 2);
        end
        
        xpos = xpos + 1;
    end
    xlim([1-x_cond_offsets(2) nFlies+length(x_cond_offsets)*x_cond_offsets(2)]);
    ylim([dset_min dset_max]);
    
    % Draw trained threshold
    x_ends = xlim;
    plot(x_ends, (repmat(thresholds(ch, fID), [1 2])), 'Color', [0.4660 0.6740 0.1880]);
    
    % Title with feature details and performance
    nPairs = length(stats.(dsets{d}).consis.performances);
    perf_labels = cell(nPairs, 1);
    for p = 1 : nPairs
        perf_labels{p} = [...
            strjoin(stats.(dsets{d}).(perf_type).conditions(stats.(dsets{d}).(perf_type).condition_pairs(p, :)), 'x') '='...
            num2str(stats.(dsets{d}).(perf_type).performances{p}(ch, fID))...
            ];
    end
    
    %title(['f' num2str(fID) ' ch' num2str(ch) newline 'consis' newline strjoin(perf_labels, newline)], 'interpreter', 'none');
    title(['f' num2str(fID) ' ch' num2str(ch) newline perf_type ' ' perf_labels{dset_pairIDs(d)}], 'interpreter', 'none');
    %title([dsets{d} ' f' num2str(fID) ' ch' num2str(ch)], 'interpreter', 'none');
    disp([dsets{d} ' ' perf_type newline strjoin(perf_labels, newline) newline]);
    
    set(gca, 'XTick', []);
    xlabel(strjoin(conditions, ' : '), 'interpreter', 'none');
end

%% Plot median value per fly, boxplots across flies

fig = figure('Color', 'w');
hold on;

% x-positions of each condition in each dataset
%    discovery; multidose8; multidose4; singledose; sleep
cond_xs = {[1 2], [7 8 9 10 11], [12.5 13.5 14.5 15.5 16.5], [3.5 4.5 5.5], [18 19 20 21]};
%cond_xticks = cellfun(@mean, cond_xs);
cond_xticks = [cond_xs{:}];
cond_xtick_labels = cell(size(cond_xticks));
cond_xtick_counter = 1;
d_markers = {'o', 'v', '^', 'o', 'o'};
dset_order = [1 4 2 3 5];

% Which condition pairings to highlight in each dataset
dset_pairings = {...
    [1 2],... % wake x anest
    [1 3],... % wake x iso1.2
    [1 3],... % wake x iso1.2
    [1 2],... % wake x iso0.6
    [2 3]}; % wake x sleepEarly
dset_pairIDs = [1 1 1 1 3];

title_strings = cell(size(dset_pairings));

for d = 1 : length(dsets)
    
    [nEpochs, nFlies, nConditions] = size(values{d});
    [conditions, cond_labels, cond_colours] = getConditions(dsets{d});
    %cond_xtick_labels{d} = strjoin(conditions, newline);
    
    % Randomly spread points around a given x-position
    a = -1/4; b = 1/4;
    xspread = a + (b-a).*rand(nFlies, 1);
    
    % Find the median value across epochs, per fly, per condition
    ys = permute(median(values{d}, 1), [2 3 1]);
    
    % Plot lines connecting medians for each fly
    for fly = 1 : nFlies
        
        plot(cond_xs{d}+xspread(fly), ys(fly, :), ':', 'Color', [0 0 0 0.2], 'LineWidth', 1);
        
        % Plot lines highlighting specific condition pairings
        for pair = 1 : size(dset_pairings{d}, 1)
            plot(cond_xs{d}(dset_pairings{d}(pair, :))+xspread(fly), ys(fly, dset_pairings{d}(pair, :)),...
                'Color', [0 0 0 0.5], 'LineWidth', 1);
        end
        
    end
    
    for c = 1 : length(conditions)
        cond_xtick_labels{cond_xtick_counter} = conditions{c};
        cond_xtick_counter = cond_xtick_counter+1;
        
        % Plot box-plot across fly medians
        boxplot(ys(:, c), 'Positions', cond_xs{d}(c), 'Widths', 0.5,...
            'Whisker', 2, 'Colors', 'k'....
            );
        
        % Plot median values for each fly
        scatter(cond_xs{d}(c)+xspread, ys(:, c), [], cond_colours{c}, d_markers{d});
        
    end
    
    % Add condition pairing, performance to title string
    nPairs = length(stats.(dsets{d}).consis.performances);
    perf_labels = cell(nPairs, 1);
    for p = 1 : nPairs
        perf_labels{p} = [...
            strjoin(stats.(dsets{d}).(perf_type).conditions(stats.(dsets{d}).(perf_type).condition_pairs(p, :)), 'x') '='...
            num2str(stats.(dsets{d}).(perf_type).performances{p}(ch, fID))...
            ];
    end
    title_strings{d} = perf_labels{dset_pairIDs(d)};
    
end

axis tight;
xlim([min([cond_xs{:}])-0.5 max([cond_xs{:}])+0.5]);

% Draw trained threshold
x_ends = xlim;
plot(x_ends, (repmat(thresholds(ch, fID), [1 2])), 'Color', [0.4660 0.6740 0.1880]);

title(['f' num2str(fID) ' ch' num2str(ch)], 'interpreter', 'none');
title(['f' num2str(fID) ' ch' num2str(ch) newline...
    strjoin(dsets(dset_order), ' : ') newline...
    perf_type newline...
    strjoin(title_strings(dset_order), ' : ')]);

ylabel('feature value');

[~, xtick_order] = sort(cond_xticks);
set(gca, 'XTick', cond_xticks(xtick_order));
set(gca, 'XTickLabel', cond_xtick_labels(xtick_order));
xtickangle(gca, 15);
xlabel(strjoin(dsets(dset_order), ' : '));
