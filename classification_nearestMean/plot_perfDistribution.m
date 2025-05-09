function [h, l] = plot_perfDistribution(ch_valid_features, ch_colours, performances, sig_thresh_fdr, performances_random, sig_thresh)
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

% Consider features which are valid for ALL channels
valid_all = sum(ch_valid_features, 1);
valid_all = valid_all == size(ch_valid_features, 1);

% Plot average across channels
chMean_colour = [0 0 0];
performances_mean = mean(performances, 1);
h = cdfplot(performances_mean(valid_all));
set(h, 'Color', chMean_colour);
set(h, 'LineWidth', 2);

for ch = 1 : size(performances, 1)
    
    % Plot cumulative histogram
    h = cdfplot(performances(ch, find(ch_valid_features(ch, :))));
    set(h, 'Color', ch_colours(ch, :));
    
    % Show in legend
    if ch ~= 1 && ch ~= size(performances, 1)
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
h = cdfplot(performances_random(1, :)); % only need 1 "channel"
set(h, 'Color', chance_colour);
set(h, 'LineWidth', 2);
% Show significance cutoff
match = single(abs(h.XData - sig_thresh));
match = h.YData(find(match == min(match)));
scatter(sig_thresh, match(2), 50, chance_colour, '>', 'filled');

l = legend('chMean', 'ch1', 'ch6', 'ch15', 'chance', 'p=0.05', 'Location', 'southeast');
ylabel('portion of valid features');

end

