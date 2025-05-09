function [h] = summary_plots(panel, ch, ptype, subplot_size, subplot_pos, data_sets, ch_valid_features, perfs, hctsa)
%SUMMARY_PLOTS Summary of this function goes here
%   Detailed explanation goes here
% Inputs:
%   panel = integer; which panel to plot
%   ch = integer; which channel to plot for
%   ptype = integer; performance type to plot
%       1 = nearestMedian
%       2 = consis
%   subplot_size = vector; [rows columns]
%   subplot_pos = subplot position
%   data_sets = cell array of strings
%       Which data sets to plot against each other
%       Should have length 2 for panel == 1 or 2
%       Should have length 3 for panel == 3
%       {'train', 'validate1', 'validateBatchNormalised'}
%   ch_valid_features = logical matrix; channels x features
%   perfs = struct; structure of performances
%   hctsa = struct; hctsa data
%
% Outputs:
%   h = axis handle

%% Common

perf_types = {'nearestMedian', 'consis'};

%data_sets = {'train', 'validate1'};

% data_set colours
tmp = cbrewer('div', 'BrBG', 8);
dset_colours = {tmp(end:-1:end-1, :), tmp(2:3, :)};
dset_lineWidth = [1 1];
dset_alpha1 = [1 0.5];
dset_alpha2 = [0.5 0.5];
dset_alpha_mult = [1 0.9];

%% Plot discovery vs evaluation, for single channel

if panel == 1
    h = subplot(subplot_size(1), subplot_size(2), subplot_pos); hold on;
    
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
        perfs.(data_sets{1}).(perf_type).performances(ch, features),...
        perfs.(data_sets{2}).(perf_type).performances(ch, features),...
        '.', 'MarkerEdgeColor', [0.8 0.8 0.8]);
    axis tight
    xl = xlim;
    yl = ylim;
    
    % Plot chance
    line([0.5 0.5], yl, 'Color', 'k');
    line(xl, [0.5 0.5], 'Color', 'k');
    % Plot significance thresholds
    thresh = perfs.(data_sets{1}).(perf_type).sig_thresh_fdr(ch);
    line([thresh thresh], yl, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
    thresh = perfs.(data_sets{2}).(perf_type).sig_thresh_fdr(ch);
    line(xl, [thresh thresh], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
    
    title(['ch ' num2str(ch) ' ' perf_type]);
    xlabel('discovery');
    ylabel('pilot evaluation');
    %axis square
    
    % Show best feature in each dataset
    disp('===');
    for dset = 1 : length(data_sets)
        data_set = data_sets{dset};
        [p, f] = max(perfs.(data_set).(perf_type).performances(ch, features));
        scatter(...
            perfs.(data_sets{1}).(perf_type).performances(ch, fIDs(f)),...
            perfs.(data_sets{2}).(perf_type).performances(ch, fIDs(f)),...
            'o', 'k',...
            'LineWidth', 1);
        fname = hctsa.Operations.Name(fIDs(f));
        disp(['best for ' data_set ': fID:' num2str(fIDs(f)) ' perf=' num2str(p) ' ' fname{1}]);
    end
    
    % Show best overall feature (feature most towards the top-right)
    %   For every point, find distance from (0.5 0.5) in top-right quadrant
    dists = nan(size(fIDs));
    for f = 1 : size(fIDs, 2)
        fID = fIDs(f);
        x = perfs.(data_sets{1}).(perf_type).performances(ch, fID);
        y = perfs.(data_sets{2}).(perf_type).performances(ch, fID);
        
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
    x = perfs.(data_sets{1}).(perf_type).performances(ch, fID);
    y = perfs.(data_sets{2}).(perf_type).performances(ch, fID);
    scatter(x, y, 'o', 'k', 'LineWidth', 1);
    fname = hctsa.Operations.Name(fIDs(f));
    disp(['best overall: fID:' num2str(fIDs(f)) ' perf(train)=' num2str(x) ' perf(validate1)=' num2str(y) ' ' fname{1}]);
    
    % Show reference features
    disp('===');
    [featureIDs, masterList] = getRefFeatures(hctsa);
    colours = cbrewer('qual', 'Set1', length(masterList));
    colormap(gca, colours);
    for m = 1 : length(masterList)
        disp([masterList{m} ': ' num2str(length(featureIDs{m})) ' features']);
        
        perf_train = perfs.(data_sets{1}).(perf_type).performances(ch, featureIDs{m});
        perf_vals = perfs.(data_sets{2}).(perf_type).performances(ch, featureIDs{m});
        % Mark out features
        scatter(...
            perf_train,...
            perf_vals,...
            [], repmat(colours(m, :), [length(featureIDs{m}) 1]), 'x',...
            'LineWidth', 1);
        
        % Display feature names, IDs, and performances
        [sorted, order] = sort(perf_vals, 'descend');
        s = sprintf('%s\t\t%s\t%s\t\t%s',...
            'fID',...
            'train',...
            'val',...
            'name');
        disp(s);
        for f = 1 : length(featureIDs{m})
            fname = hctsa.Operations.Name(featureIDs{m}(order(f)));
            s = sprintf('f%i\t%.4f\t%.4f\t%s',...
                featureIDs{m}(order(f)),...
                perf_train(order(f)),...
                perf_vals(order(f)),...
                fname{1});
            disp(s);
        end
        
    end
    cbar = colorbar;
    tickpos = linspace(0, 1, (length(masterList)*2)+1);
    tickpos = tickpos(2:2:end);
    set(cbar, 'YTick', tickpos);
    set(cbar, 'YTickLabel', masterList);
    set(cbar, 'TickLabelInterpreter', 'none');
end

%% Plot number of sig features per channel

if panel == 2
    h = subplot(subplot_size(1), subplot_size(2), subplot_pos); hold on;
    
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
    
    legend(data_sets{1}, data_sets{2}, 'both', 'Location', 'northoutside', 'Orientation', 'horizontal');
    
end

%% Plot number of sig features per channel (3 data sets)

if panel == 3
    h = subplot(subplot_size(1), subplot_size(2), subplot_pos); hold on;
    
    pair_styles = {'-.', ':'};
    tmp = cbrewer('qual', 'Set1', 8);
    dset_colours = {tmp(2, :), tmp(4, :), tmp(5, :)};
    
    perf_type = perf_types{ptype};
    
    sigs_all = [];
    
    % Plot number of sig features for each dataset
    for dset = 1 : length(data_sets)
        sig = perfs.(data_sets{dset}).(perf_type).sig & logical(ch_valid_features);
        sigs_all = cat(3, sigs_all, sig);
        nSig = sum(sig, 2);
        
        plot(nSig, 'Color', dset_colours{dset});
    end
    
    % Plot number of features which were sig in discovery+evaluation
    % datasets
    for dset = 2 : length(data_sets)
        sigs_both = all(sigs_all(:, :, [1 dset]), 3);
        nSig = sum(sigs_both, 2);
        plot(nSig, 'Color', dset_colours{dset}, 'Linestyle', '-.');
    end
    
    axis tight
    yl = ylim;
    ylim([0 yl(end)]);
    xlim([1 length(nSig)]);
    
    %title(perf_type);
    xlabel('channel');
    ylabel('N sig. features');
    
    legend(data_sets{1}, data_sets{2}, data_sets{3}, 'Location', 'northoutside', 'Orientation', 'horizontal');
    
end

end

function [featureIDs, masterList] = getRefFeatures(hctsa)
% Get IDs of references features to highlight
%   Features which are related to measures previously reported as
%   indicators of conscious level

masterList = {...
    %'CO_RM_AMInformation',...
    'ApEn',...
    'EN_MS_LZcomplexity',...
    'EN_PermEn',...
    'EN_SampEn',...
    'SP_Summaries'}; % keep SP_Summaries at the end, for handpicking features
masterIDs = cell(size(masterList));

% Get all master operations from the master list
for category = 1 : length(masterList)
    masterIDs{category} = find(contains(hctsa.MasterOperations.Label, masterList{category}));
end

featureIDs = cell(size(masterList));
% Get all features for each master operation
for category = 1 : length(masterIDs)
    matches = cell(length(masterIDs{category}));
    for mID = 1 : length(masterIDs{category})
        matches{mID} = find(hctsa.Operations.MasterID == masterIDs{category}(mID));
    end
    featureIDs{category} = cat(1, matches{:});
end

% Limit SP_Summaries to handpicked features
featureIDs{end} = [...
    find(contains(hctsa.Operations.Name, 'SP_Summaries') & (contains(hctsa.Operations.Name, '_area_5_1') | contains(hctsa.Operations.Name, '_logarea_5_1')));... power at particular bands (can only do broadband power)
    find(contains(hctsa.Operations.Name, 'SP_Summaries') & (contains(hctsa.Operations.Name, '_area_') | contains(hctsa.Operations.Name, '_logarea_')));...
    find(contains(hctsa.Operations.Name, '_centroid')); find(contains(hctsa.Operations.Name, '_wmax_95'));... spectral edge frequencies
    find(contains(hctsa.Operations.Name, '_spect_shann_ent'))... spectral entropy
    ];

end