function [h] = summary_plots(panel, ch, perf_type, subplot_size, subplot_pos, data_sets, ch_valid_features, perfs, hctsa)
%SUMMARY_PLOTS Summary of this function goes here
%   Detailed explanation goes here
% Inputs:
%   panel = integer; which panel to plot
%   ch = integer; which channel to plot for
%   perf_type = string; performance type to plot
%       'nearestMedian', 'nearestMedian', 'consis',
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
    
    % Get features which are sig. in all datasets
    sig = perfs.(data_sets{1}).(perf_type).sig;
    for dset = 2 : length(data_sets)
        sig = sig & perfs.(data_sets{dset}).(perf_type).sig;
    end
    sig = sig(ch, :);
    features = logical(ch_valid_features(ch, :));
    fIDs = find(features);
    
    % Plot performance in one dataset against the other
    scatter(...
        perfs.(data_sets{1}).(perf_type).performances{1}(ch, features),...
        perfs.(data_sets{2}).(perf_type).performances{1}(ch, features),...
        5, 'o', 'filled',...
		'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerFaceAlpha', 0.25,...
		'MarkerEdgeColor', [0.7 0.7 0.7], 'MarkerEdgeAlpha', 0);
    axis tight
    xl = xlim;
    yl = ylim;
	xlim([0.35 xl(end)]);
	ylim([0.35 yl(end)]);
	
	% Highlight the features which are significant across everything
	sig_features = perfs.(data_sets{1}).(perf_type).sig & perfs.(data_sets{2}).(perf_type).sig;
	scatter(...
        perfs.(data_sets{1}).(perf_type).performances{1}(ch, sig_features(ch, :)),...
        perfs.(data_sets{2}).(perf_type).performances{1}(ch, sig_features(ch, :)),...
        5, 'o', 'MarkerEdgeColor', [0 0 0]);
    
    % Plot chance
	xl = xlim;
    yl = ylim;
    line([0.5 0.5], yl, 'Color', 'k');
    line(xl, [0.5 0.5], 'Color', 'k');
    
	% Significance threshold in discovery flies
	thresh = perfs.(data_sets{1}).(perf_type).sig_thresh_fdr(ch); % assumes data_sets{1} is 'train'
    line([thresh thresh], yl, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
	
	%{
	% This is probably not meaningful? Because features may have average
	%	performance above this threshold, but not significant in all 
	%	evaluationflies (e.g., if it achieves very high performance in 
	%	just one evaluation set)
	% Indicate where the cutoff of significant features are
	thresh = min(... % smallest significant performance value
		perfs.(data_sets{2}).(perf_type).performances{1}(ch, perfs.(data_sets{2}).(perf_type).sig(ch, :)));
	line(xl, [thresh thresh], 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);
	%}
	%}
	
    title(['ch ' num2str(ch) ' ' perf_type]);
    xlabel('discovery');
    ylabel('evaluation');
    %axis square
    
    % Show best feature in each dataset
	% Really, it only makes sense to show the best feature overall (which
	% generalised across everything)
    disp('===');
    for dset = 2 : length(data_sets) % assumes first one is discovery set
        data_set = data_sets{dset};
        [p, f] = max(perfs.(data_set).(perf_type).performances{1}(ch, features));
        scatter(...
            perfs.(data_sets{1}).(perf_type).performances{1}(ch, fIDs(f)),...
            perfs.(data_sets{2}).(perf_type).performances{1}(ch, fIDs(f)),...
            'o', 'k',...
            'LineWidth', 1,...
			'MarkerEdgeColor', 'k');
        fname = hctsa{1}.Operations.Name(fIDs(f));
        disp(['best for ' data_set ': fID:' num2str(fIDs(f)) ' perf=' num2str(p) ' ' fname{1}]);
    end
    
	%{
    % Show best overall feature (feature most towards the top-right)
    %   For every point, find distance from (0.5 0.5) in top-right quadrant
    dists = nan(size(fIDs));
    for f = 1 : size(fIDs, 2)
        fID = fIDs(f);
        x = perfs.(data_sets{1}).(perf_type).performances{1}(ch, fID);
        y = perfs.(data_sets{2}).(perf_type).performances{1}(ch, fID);
        
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
    x = perfs.(data_sets{1}).(perf_type).performances{1}(ch, fID);
    y = perfs.(data_sets{2}).(perf_type).performances{1}(ch, fID);
    scatter(x, y, 'o', 'k', 'LineWidth', 1);
    fname = hctsa{1}.Operations.Name(fIDs(f));
    disp(['best overall: fID:' num2str(fIDs(f)) ' perf(train)=' num2str(x) ' perf(validate1)=' num2str(y) ' ' fname{1}]);
	%}
	
    % Show reference features
    disp('===');
    [featureIDs, masterList] = getRefFeatures(hctsa);
    colours = cbrewer('qual', 'Set1', length(masterList));
    colormap(gca, colours);
    for m = 1 : length(masterList)
        disp([masterList{m} ': ' num2str(length(featureIDs{m})) ' features']);
        
        perf_train = perfs.(data_sets{1}).(perf_type).performances{1}(ch, featureIDs{m});
        perf_vals = perfs.(data_sets{2}).(perf_type).performances{1}(ch, featureIDs{m});
        % Mark out features
        scatter(...
            perf_train,...
            perf_vals,...
            [], repmat(colours(m, :), [length(featureIDs{m}) 1]), 'x',...
            'LineWidth', 0.5);
        
        % Display feature names, IDs, and performances
        [sorted, order] = sort(perf_vals, 'descend');
        s = sprintf('%s\t\t%s\t%s\t\t%s',...
            'fID',...
            'train',...
            'val',...
            'name');
        disp(s);
        for f = 1 : length(featureIDs{m})
            fname = hctsa{1}.Operations.Name(featureIDs{m}(order(f)));
            s = sprintf('f%i\t%.4f\t%.4f\t%s',...
                featureIDs{m}(order(f)),...
                perf_train(order(f)),...
                perf_vals(order(f)),...
                fname{1});
            disp(s);
		end
		
		% Mark out features which are significant
		sig_features = perfs.(data_sets{1}).(perf_type).sig(ch, :) & perfs.(data_sets{2}).(perf_type).sig(ch, :);
		sig_ref = sig_features(featureIDs{m}); % get sigs of ref features
		sig_ref = featureIDs{m}(sig_ref); % Filter ref features by significance
		perf_train = perfs.(data_sets{1}).(perf_type).performances{1}(ch, sig_ref);
        perf_vals = perfs.(data_sets{2}).(perf_type).performances{1}(ch, sig_ref);
		% Mark out features thicker
		if ~isempty(perf_train) && ~isempty(perf_vals)
        	scatter(...
            	perf_train,...
            	perf_vals,...
            	[], repmat(colours(m, :), [length(sig_ref) 1]), 'x',...
            	'LineWidth', 1.5);
		end
        
    end
    cbar = colorbar;
    tickpos = linspace(0, 1, (length(masterList)*2)+1);
    tickpos = tickpos(2:2:end);
    set(cbar, 'YTick', tickpos);
    set(cbar, 'YTickLabel', masterList);
    set(cbar, 'TickLabelInterpreter', 'none');
	
	
	% Highlight the features which are significant across everything
	% Do it again, to plot on top of reference feature markers
	sig_features = perfs.(data_sets{1}).(perf_type).sig & perfs.(data_sets{2}).(perf_type).sig;
	scatter(...
        perfs.(data_sets{1}).(perf_type).performances{1}(ch, sig_features(ch, :)),...
        perfs.(data_sets{2}).(perf_type).performances{1}(ch, sig_features(ch, :)),...
        5, 'o', 'MarkerEdgeColor', [0 0 0], 'MarkerEdgeAlpha', 0.05);
	
	% Highlight previously highlighted features from Stage 1
	%	These were the previously best features in Channel 6,
	%	classification:
	%		StatAvl250, f551 (discovery flies)
	%		SP_Summaries_welch_rect_logarea_2_1, f4529 (pilot evaluation)
	%	classification (batch normalised; pilot evaluation flies only)
	%		StatAvl250, f551
	%		DN_Moments_raw_4, f33 (best across all channels, @ channel 5)
	%	consistency
	%		MD_rawHRVmeas_SD, f7702 (discovery flies)
	%		rms, f16  (pilot evaluation)
	%		
	switch perf_type
		case 'nearestMedian'
			previous_features = [551 4529];
		case 'nearestMedianBatchNormalised'
			previous_features = [33 551];
		case 'consis'
			previous_features = [7702 16];
	end
	circle_labels = hctsa{1}.Operations.Name(previous_features);
	scatter(...
            perfs.(data_sets{1}).(perf_type).performances{1}(ch, previous_features),...
            perfs.(data_sets{2}).(perf_type).performances{1}(ch, previous_features),...
            'o', 'k',...
            'LineWidth', 1.5,...
			'MarkerEdgeColor', 'b',...
			'LineWidth', 1);
	text(...
            perfs.(data_sets{1}).(perf_type).performances{1}(ch, previous_features),...
            perfs.(data_sets{2}).(perf_type).performances{1}(ch, previous_features),...
			circle_labels,...
			'interpreter', 'none');
end

%% Plot number of sig features per channel
% Show average number of significant features when considering
%	1 - each dataset independently
%	2 - each pair of datasets (5 choose 2)
%	3 - each group of three datasets (5 choose 3)
%	4 - each group of 4 datasets (5 choose 4)
%	5 - all datasets together (5 choose 5)
% Create option - for 2-5, always include discovery in the combination
% Create option - exclude a specific dataset (e.g. sleep)

if panel == 2
	
	perfs = updateStatsStructure(perfs);
	
	dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
	%dsets = {'train', 'multidose8', 'multidose4', 'singledose'};
	
	h = subplot(subplot_size(1), subplot_size(2), subplot_pos); hold on;
	
	% Get number of flies in each dataset
	dset_nFlies = nan(size(dsets));
	dset_pairings = cell(length(dsets), 1);
	dset_statsPairIds = nan(size(dsets));
	for d = 1 : length(dsets)
		
		% Get number of flies in each dataset
		[~, nFlies] = getDimensionsFast(dsets{d});
		dset_nFlies(d) = nFlies;
		
		% Get main condition pair for the dataset
		[conds, cond_labels, cond_colours, stats_order, conds_main] =...
			getConditions(dsets{d});
		dset_pairings{d} = conds_main;
		pair_statsIds = find(ismember(stats_order, conds_main));
		dset_statsPairIds(d) = find(ismember(sort(perfs.(dsets{d}).(perf_type).condition_pairs, 2), sort(pair_statsIds), 'rows'));
		
	end
	
	% Group size colours
	%group_colours = cbrewer('seq', 'YlGnBu', length(dsets));
	group_colours = flipud(turbo(length(dsets)));
	
	for dset_group_size = 1 : length(dsets)
		
		% Get all combinations of datasets
		groupings = nchoosek((1:length(dsets)), dset_group_size);
		
		nSig_mean = zeros(size(ch_valid_features, 1), 1);
		nFlies_total = 0;
		
		% Go through each grouping
		for group = 1 : size(groupings, 1)
			
			sigs_group = logical(ch_valid_features);
			group_nFlies = 0;
			
			for d = 1 : size(groupings, 2)
				
				sigs_group = sigs_group & perfs.(dsets{groupings(group, d)}).(perf_type).sig(:, :, dset_statsPairIds(groupings(group, d)));
				
				group_nFlies = group_nFlies + dset_nFlies(groupings(group, d));
				
			end
			
			% Plot number of features which were sig in all datasets in the
			% group
			nSig = sum(sigs_group, 2);
			plot(nSig, 'Color', group_colours(dset_group_size, :), 'LineWidth', 0.25);
			
			nSig_mean = nSig_mean + nSig;%(group_nFlies*nSig);
			nFlies_total = nFlies_total + group_nFlies;
			
		end
		
		% Average across groups (should we weight by number of flies?)
		nSig_mean = nSig_mean ./ size(groupings, 1);% ./ nFlies_total; %size(groupings, 1);
		
		% Plot average number of sig features across the dataset groupings
		label_lines(dset_group_size) = plot(nSig_mean, 'Color', [group_colours(dset_group_size, :) 0.8], 'LineWidth', 3);
		
	end
	
	legend(label_lines, cellfun(@(x) [x ' sets'], arrayfun(@num2str, (1:length(dsets)), 'UniformOutput', false), 'UniformOutput', false));
	
	axis tight
    yl = ylim;
    ylim([0 yl(end)]);
    xlim([1 length(nSig_mean)]);
	
	xlabel('channel');
    ylabel('N sig. features');
	
end

%% Plot number of sig features per channel

if panel == -2
	
    h = subplot(subplot_size(1), subplot_size(2), subplot_pos); hold on;
    
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

function [perfs] = updateStatsStructure(perfs)
% Reorganise stats structure so BatchNormalised classification doesn't need
%	to be treated in a special way
% perfs.multidose8BatchNormalised.nearestMedian ->
%	perfs.multidose8.nearestMedianBatchNormalised

perf_type = 'nearestMedian'; % BatchNormalised only applies to nearestMedian

% Get fields which have 'BatchNormalised' at the end of the name
fNames = fieldnames(perfs);

keyString = 'BatchNormalised';
for f = 1 : length(fNames)
	
	if length(fNames{f}) > length(keyString)
		
		if strcmp(fNames{f}(end-length(keyString)+1:end), keyString)
			
			prestring = fNames{f}(1:end-length(keyString));
			
			perfs.(prestring).([perf_type keyString]) =...
				perfs.(fNames{f}).(perf_type);
		end
		
	end
	
end

% Discovery fly batch-normalised performance is just a copy of original
% performance
perfs.train.([perf_type keyString]) = perfs.train.(perf_type);

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
    masterIDs{category} = find(contains(hctsa{1}.MasterOperations.Label, masterList{category}));
end

featureIDs = cell(size(masterList));
% Get all features for each master operation
for category = 1 : length(masterIDs)
    matches = cell(length(masterIDs{category}));
    for mID = 1 : length(masterIDs{category})
        matches{mID} = find(hctsa{1}.Operations.MasterID == masterIDs{category}(mID));
    end
    featureIDs{category} = cat(1, matches{:});
end

% Limit SP_Summaries to handpicked features
% SP_Summaries ... area_X_Y - divides the power spectrum into X frequency
% bands; Y=1 is the first frequency band (lowest frequency band), Y=5 is
% the last frequency band (highest frequency band)
featureIDs{end} = [...
    find(contains(hctsa{1}.Operations.Name, 'SP_Summaries') & (contains(hctsa{1}.Operations.Name, '_area_5_1') | contains(hctsa{1}.Operations.Name, '_logarea_5_1')));... power at particular bands (can only do broadband power)
    find(contains(hctsa{1}.Operations.Name, 'SP_Summaries') & (contains(hctsa{1}.Operations.Name, '_area_') | contains(hctsa{1}.Operations.Name, '_logarea_')));... note - if specifying area_X_Y previously, then this will double up on those features
    find(contains(hctsa{1}.Operations.Name, '_centroid')); find(contains(hctsa{1}.Operations.Name, '_wmax_95'));... spectral edge frequencies
    find(contains(hctsa{1}.Operations.Name, '_spect_shann_ent'))... spectral entropy
    ];

end