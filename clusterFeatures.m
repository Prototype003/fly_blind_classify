function [order, consts] = clusterFeatures(data)
%CLUSTERFEATURES
%   Clusters features based on correlation distance (correlation across
%   time-series
%
% Inputs:
%   data = TS_DataMat matrix (time-series x features)
% Outputs:
%   order = ordering of features based on correlation-distance clustering
%   consts = features (columns of data) which were removed, if any

% Remove columns with constant values
% (some features become constant after scaling; e.g. a feature with
%   mostly constant 0 values, but then one very small non-zero value)
nan_replaced = data; % treat nans as 0s when taking differences
nan_replaced(isnan(nan_replaced)) = 0;
consts = find(~any(diff(nan_replaced, 1), 1));
if ~isempty(consts)
    data(:, consts) = [];
    warning('some features removed due to constant values');
end

% Similarity measure - aboslute correlation across rows
%data(isinf(data)) = NaN; % Remove Infs for correlation
% Note - Spearman pairwise can take a long time
fCorr = (corr(data, 'Type', 'Spearman', 'rows', 'pairwise'));
%fCorr = abs(fCorr + fCorr.') / 2; % because corr output isn't symmetric for whatever reason

% Correlation distance
distances = 1 - (fCorr);

% Cluster tree
tree = linkage(squareform(distances), 'average'); % note - distances must be pdist vector (treats matrix as data instead of distances)

% Sorted features
f = figure('visible', 'on'); % we want the order, not the actual plot
[h, T, order] = dendrogram(tree, 0);
close(f);

end