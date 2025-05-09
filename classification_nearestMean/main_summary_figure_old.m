%% Description

%{

Plot and get summary stats for training set cross-validation

%}

%%

perf_types = {'nearestMedian', 'consis'};
data_set = 'validate1'; % 'train', 'validate1'

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

%%

figure;
set(gcf, 'Color', 'w');

%% Feature performance distributions

% Subplot dimensions/positions and labels
columns = 3;
if strcmp(data_set, 'train')
    rows = 2;
    row_offset = 1*columns;
    letters = {'A', 'B', 'C', 'D'};
elseif strcmp(data_set, 'validate1')
    rows = 3;
    row_offset = 2*columns; % How many rows to skip for the last row
    letters = {'A', 'C', 'D', 'F'};
end

plot_counter = 1;
for ptype = 1 : length(perf_types)
    perf_type = perf_types{ptype};
    
    % Get performance values and significances
    %[performances, performances_random, sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr] = get_sig_features(perf_type, data_set, ch_valid_features, preprocess_string);
    performances = perfs.(data_set).(perf_type).performances;
    performances_random = perfs.(data_set).(perf_type).performances_random;
    sig = perfs.(data_set).(perf_type).sig;
    sig_thresh = perfs.(data_set).(perf_type).sig_thresh;
    sig_thresh_fdr = perfs.(data_set).(perf_type).sig_thresh_fdr;
    
    % Best channel (with largest number of sig. features)
    ch_best = find(sum(sig, 2) == max(sum(sig, 2)));
    ch_worst = find(sum(sig, 2) == min(sum(sig, 2)));
    ch_best = 6;
    
    % Plot distribution across valid features for each channel
    subplot(rows, columns, ptype);
    [h] = plot_perfDistribution2(ch_valid_features, ch_colours, performances, sig_thresh_fdr, performances_random, sig_thresh);
    title([letters{plot_counter} ' ' perf_type ' ' data_set], 'interpreter', 'none');
%     if strcmp(data_set, 'train')
%         xlim([0.4 0.8]);
%     elseif strcmp(data_set, 'validate1')
%         xlim([0.4 0.7]);
%         if ptype == 2
%             xlim([0.4 0.8]);
%         end
%     end
    plot_counter = plot_counter + 1;
    
    % Plot features sorted by performance
    subplot(rows, columns, ptype + row_offset);
    plot_featuresSorted(ch_best, performances, sig, ch_colours);
    title([letters{plot_counter} ' ' 'sig. features sorted by ch' num2str(ch_best)]);
    xlabel(perf_types{ptype});
    plot_counter = plot_counter + 1;
end

% Colorbar
subplot(rows, columns, columns); axis off
cbar = colorbar('westoutside');
title(cbar, '%tile');

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

%% Feature value violin plots
% 
% ch = 6;
% perf_types = {'class_nearestMedian', 'consis_nearestMedian'};
% ref_sets = {'train', 'validate1'};
% 
% source_dir = ['results' preprocess_string '/'];
% 
% % Colours
% c = BF_GetColorMap('redyellowblue', 6);
% cond_colours = {c(1, :), c(end, :)}; % red = wake; blue = anest
% cond_offsets = [-0.1 0.1]; % x-axis offsets for each violin
% extraParams = struct();
% extraParams.offsetRange = 0.5; % width of violins
% 
% %%
% 
% if strcmp(data_set, 'validate1')
%     
%     plot_positions = [3 5 4 6];
%     letters = {'B', 'C', 'F', 'G'};
%     plot_counter = 1;
%     for ptype = 1 : length(perf_types)
%         perf_type = perf_types{ptype};
%         for rset = 1 : length(ref_sets)
%             ref_set = ref_sets{rset};
%             
%             % Get values
%             [values, fDetails, perf] = get_best_from_set(ch, perf_type, ref_set, 0, preprocess_string);
%             
%             subplot(rows, columns, plot_positions(plot_counter));
%             
%             % Plot trained threshold for the feature
%             thresh = load([source_dir 'class_nearestMedian_thresholds.mat']); % note no thresholds for consistency
%             threshold = thresh.thresholds(ch, fDetails.fID);
%             line([0 size(values, 1)+1], [threshold threshold], 'Color', 'k', 'LineWidth', 1, 'LineStyle', ':');
%             
%             % Plot violins
%             for cond = 1 : size(values, 2)
%                 
%                 extraParams.customOffset = cond_offsets(cond);
%                 extraParams.theColors = repmat(cond_colours(cond), [size(values, 1) 1]);
%                 
%                 BF_JitteredParallelScatter_custom(values(:, cond), 1, 1, 0, extraParams);
%                 
%             end
%             axis tight
%             
%             % Plot lines to separate datasets
%             line([13.5 13.5], ylim, 'Color', 'k', 'LineWidth', 1, 'LineStyle', '--');
%             
%             set(gca, 'XTick', (1:size(values, 1)));
%             
%             %title([perf_type '-' reference_set ': ' num2str(fDetails.fID) ' ' fDetails.fName{1}], 'interpreter', 'none');
%             title([letters{plot_counter} ' ' 'best from ' ref_set ': ' num2str(fDetails.fID)]);
%             xlabel('fly');
%             ylabel('raw val');
%             
%             plot_counter = plot_counter + 1;
%             
%         end
%     end
%     
% end