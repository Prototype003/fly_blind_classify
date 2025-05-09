function [h] = plot_perfDistribution2(ch_valid_features, ch_colours, performances, sig_thresh_fdr, performances_random, sig_thresh)
%PLOT_PERFDIST
% Plot distribution of performances across features, per channel
% Inputs:
%   ch_valid_features
%   ch_colours
%   performances
%   sig_thresh_fdr
%   performances_random
%   sig_thresh

hold on;

ptile_lines = [50 75 90 95 99 100];
ptile_lines = (10:10:100);
ptiles = (0:0.1:100);

% Percentile colours
r = linspace(0, 1, length(ptiles))';
g = linspace(0.5, 0.5, length(ptiles))';
b = linspace(1, 0, length(ptiles))';
pcolors = cat(2, r, g, b);
pcolors = ptiles;

cmap = inferno(1000);
colormap(cmap);

% Percentile performances
ptile_perf = prctile(performances, ptiles, 2);
% Get percentile performances for each channel
for ch = 1 : size(performances, 1)
    ptile_perf(ch, :) = prctile(performances(ch, logical(ch_valid_features(ch, :))), ptiles);
end

% x-coords stay constant
x = [(1:size(performances, 1)) fliplr((1:size(performances, 1)))];

% Draw background patch (to fill in if 0th percentile is not zero
y = [zeros(1, size(performances, 1)) fliplr(ptile_perf(:, 1)')];
c = repmat(pcolors(1), [size(performances, 1)*2 1]);
h = patch(x, y, c, 'EdgeColor', 'none');
h.Annotation.LegendInformation.IconDisplayStyle = 'off';

% Patch - from this percentile to the next
for p = 1 : length(ptiles)-1
    
    y = [ptile_perf(:, p)' fliplr(ptile_perf(:, p+1)')];
    %c = cat(1, repmat(pcolors(p, :), [size(performances, 1) 1]), repmat(pcolors(p+1, :), [size(performances, 1) 1]));
    %c = permute(c, [3 1 2]);
    c = cat(1, repmat(pcolors(p), [size(performances, 1), 1]), repmat(pcolors(p+1), [size(performances, 1) 1]));
    
    h = patch(x, y, c, 'EdgeColor', 'none');
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
    
end

% Plot lines for reference
for p = ptile_lines
    h = plot(ptile_perf(:, ptiles == p), ':', 'LineWidth', 1, 'Color', [0.5 0.5 0.5]);
    h.Annotation.LegendInformation.IconDisplayStyle = 'off';
end

% Plot significance threshold
h = plot(sig_thresh_fdr, 'b-', 'LineWidth', 2);
h.Annotation.LegendInformation.IconDisplayStyle = 'on';

xlim([1 size(performances, 1)]);
ylim([min(ptile_perf(:, ptiles == min(ptile_lines))) 1]);
xlabel('channel');
ylabel('performance');


%cbar = colorbar;
%title(cbar, '%tile');


end

