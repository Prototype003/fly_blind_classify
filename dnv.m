function [diff_mat, kw_order, nan_features] = dnv(hctsa, valid_features, class1, class2)
% Compute difference in normalised values
%
% Inputs:
%   hctsa = hctsa struct (for 1 channel); must contain fields:
%       TS_DataMat = matrix containing hctsa values (epochs x features)
%       TS_TimeSeries = struct containing keywords for each epoch
%       TS_Normalised = matrix containing normalised hctsa values
%           (epochs x features)
%   valid_features = logical vector indicating which features are valid
%       If empty, assumes all provided feaures are valid
%   class1 = string; keyword for filtering epochs
%   class2 = string; keyword for filtering epochs to be subtracted from
%       the epochs in class1
%
% Outputs:
%   diff_mat = matrix holding differences in normalised values
%       (epochs^2 x features)
%   kw_order = cell array (epochs^2 x 1) describing the row order of
%       diff_mat
%   nan_features = logical vector indicating if any of the "valid" features
%       are nan

% valid_features = getValidFeatures(hctsa.TS_DataMat);

% If nothing, assume all features there are valid
if isempty(valid_features)
    valid_features = true(1, size(hctsa.TS_Normalised, 2));
end

% Get data dimensions from hctsa.TimeSeries keywords
kw = split(hctsa.TimeSeries.Keywords, ',');

% Number of individual flies (assumed as first keyword)
flies = unique(kw(:, 1), 'stable'); % first occurrence of each fly
nFlies = length(flies);

% Number of epochs (assumes same number of epochs across flies, conditions
%   Treat each epoch segment as individual epochs (for multidose set)
%   Assumes epoch information is in third and fifth keywords
epochs = unique(kw(:, 3), 'stable');
if size(kw, 2) == 5
    segments = unique(kw(:, 5), 'stable');
else
    segments = 1;
end
nEpochs = length(epochs) * length(segments);

% Get data dimensions
%[nChannels, nFlies, nConditions, nEpochs] = getDimensions(data_set);

% Get rows corresponding to each condition
class1 = getIds({class1}, hctsa.TimeSeries);
class2 = getIds({class2}, hctsa.TimeSeries);
classes = {class1, class2};

% % Get rows corresponding to each condition
% class1 = getIds({'condition1'}, hctsa.TimeSeries);
% classes = {class1, ~class1}; % two conditions only

%diff_mat = nan(nEpochs*nEpochs*nFlies, size(hctsa.TS_DataMat, 2));
diff_mat = single([]);
kw_order = cell(nFlies*nEpochs*nEpochs, 1);

row_counter = 1;
for fly = 1 : nFlies
    % Find rows corresponding to the fly
    fly_rows = getIds({flies{fly}}, hctsa.TimeSeries);
    
    % Get rows for each class for this fly
    rows = cell(size(classes));
    for class = 1 : length(classes)
        rows{class} = find(classes{class} & fly_rows);
    end
    
    % Subtract anest from wake for every pair of epochs
    %vals = nan(nEpochs*nEpochs, 1);
    vals = nan(length(rows{1})*length(rows{2}), length(find(valid_features)), size(hctsa.TS_Normalised, 3));
    pair_counter = 1;
    for epoch1 = 1 : length(rows{1})
        for epoch2 = 1 : length(rows{2})
            vals(pair_counter, :, :) =...
                hctsa.TS_Normalised(rows{1}(epoch1), valid_features, :) -...
                hctsa.TS_Normalised(rows{2}(epoch2), valid_features, :);
            kw_order{row_counter} = [flies{fly} ',epochW' num2str(epoch1) ',epochA' num2str(epoch2)];
            pair_counter = pair_counter + 1;
            row_counter = row_counter + 1;
        end
%         if (any(isnan(vals(:))))
%             keyboard;
%         end
		disp(['fly ' num2str(fly) ' epoch1 ' num2str(epoch1)]);
    end
    
%     vals = [];
%     for epoch1 = 1 : nEpochs
%         epoch_vals = nan(nEpochs, length(find(valid_features)));
%         for epoch2 = 1 : nEpochs
%             epoch_vals(epoch2, :) = hctsa.TS_Normalised(rows{1}(epoch1), valid_features) - hctsa.TS_Normalised(rows{2}(epoch2), valid_features);
%         end
% %         if any(isnan(epoch_vals(:)))
% %             keyboard;
% %         end
%         vals = cat(1, vals, epoch_vals);
%     end
    diff_mat = cat(1, diff_mat, vals);
end

%{
% Replace nans for any features which get nans for some reason after
%   scaling
%       feature 976 in the discovery flies dataset
%       feature 977 in the sleep flies dataset
%   Features seem to get nans if all values are 0 except for 1
%       This might be occurring due to machine error (non-0 value tends to
%       be very small, very close to 0 in any case)

nan_features = any(isnan(diff_mat), 1);
if any(nan_features)
    warning('dnv() - some "valid" features which have nans after scaling - columns:');
    disp(find(nan_features));
    %diff_mat(:, nan_features) = [];
end
%}

end

