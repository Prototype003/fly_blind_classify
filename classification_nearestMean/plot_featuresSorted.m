function [] = plot_featuresSorted(ch_best, performances, sig, ch_colours)
%PLOT_FEATURESSORTED
% Plot sorted feature performance (sorted by a reference channel)
%
% Inputs:
%   ch_best
%   performances
%   sig
%   ch_colours

hold on;

[sorted, order] = sort(performances(ch_best, :), 'descend');
performances_sorted = performances(:, order);

% Limit to only show features which are sig. at least once
sig_features = sum(sig, 1);
sig_features = sig_features(order) > 0;

% Plot mean+std across channels
y = mean(performances_sorted(:, sig_features), 1);
yerr = std(performances_sorted(:, sig_features), [], 1);
x = (1:sum(sig_features));
h = patch([x fliplr(x)], [y+yerr fliplr(y-yerr)], 'k', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); % std
h.Annotation.LegendInformation.IconDisplayStyle = 'off';

h = plot(x, y, 'k'); % mean
h.Annotation.LegendInformation.IconDisplayStyle = 'off';

plot(x, performances_sorted(ch_best, sig_features), 'Color', ch_colours(ch_best, :), 'LineWidth', 2); % best ch

line([1 max(x)], [0.5 0.5], 'LineWidth', 2); % chance line

% plot(x, performances_sorted(ch_worst, sig_features), 'Color', ch_colours(ch_worst, :)); % worst ch

legend(['ch' num2str(ch_best)]);
title(['sig. features sorted by ch' num2str(ch_best)]);
ylabel('performance'); xlabel(['feature (sorted by ch' num2str(ch_best) ')']);
axis tight;

end

