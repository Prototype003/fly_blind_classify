function [values, labels] = featureValueDistributionFromHctsas(hctsas, fID, ch)
% Plot distribution of values for a given feature and channel across all
% flies
%
% Inputs:
%	hctsas = cell array, each cell contains hctsa struct for a dataset
%		Contains TS_DataMat (time series x features x channels)
%		Contains TimeSeries (table)
%	fID
%	ch
%
% Outputs:
%   values = cell array; each cell holds a matrix of feature values
%       (epochs x flies x conditions)

dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
fly_groups = {(1:13), (1:8), (9:12), (1:18), (1:19)};

values = cell(size(dsets));
labels = cell(size(dsets));

for d = 1 : length(hctsas)
    
	% Get values
	dValues = hctsas{d}.TS_DataMat(:, fID, ch);
	dTimeSeries = hctsas{d}.TimeSeries;
    
    % Get dimensions
    %   Don't use nConditions - as conditions might be excluded in a dset
    [nChannels, nFlies, nConditions, nEpochs] = getDimensionsFast(dsets{d});
    
    % Conditions in dataset
    [conditions, cond_labels] = getConditions(dsets{d});
    
    % Storage matrix
    values{d} = nan(nEpochs, length(fly_groups{d}), length(conditions));
	labels{d} = cell(nEpochs, length(fly_groups{d}), length(conditions));
	
    for f = 1 : length(fly_groups{d})
        for c = 1 : length(conditions)
            fly = fly_groups{d}(f);
            
            % Identify rows for each fly and condition
            keys = {['fly' num2str(fly)], conditions{c}};
            rows = getIds(keys, dTimeSeries);
            
            % Store row values
            values{d}(:, f, c) = dValues(rows);
			labels{d}(:, f, c) = dTimeSeries.Name(rows);
            
        end
    end
    
end

end

