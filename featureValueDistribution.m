function [values, labels] = featureValueDistribution(fID, ch)
% Plot distribution of values for a given feature and channel across all
% flies
%
% In principle, does the same thing as get_fly_values()
%   TODO - update get_fly_values to do this / work for evaluation flies?
%
% Inputs:
%	fID
%	ch
%
% Outputs:
%   values = cell array; each cell holds a matrix of feature values
%       (epochs x flies x conditions)

hctsa_dir = 'hctsa_space_subtractMean_removeLineNoise';
hctsa_prefix = 'HCTSA_';
hctsa_suffix = '.mat';

% note - data from the same dataset should be together/consecutive
dsets = {'train', 'multidose', 'multidose', 'singledose', 'sleep'};
dsets_actual = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
needs_loading = [1 1 0 1 1]; % 0 means it's loaded for a previous set
fly_groups = {(1:13), (1:8), (9:12), (1:18), (1:19)};

values = cell(size(dsets_actual));
labels = cell(size(dsets_actual));

for d = 1 : length(dsets)
    
    if needs_loading(d)
        
        % Get matfile
        filename = [fileparts(mfilename('fullpath')) filesep hctsa_dir filesep hctsa_prefix dsets{d} '_channel' num2str(ch) hctsa_suffix];
        dfile = matfile(filename);
        
        % Get values
        dValues = dfile.TS_DataMat(:, fID);
        dTimeSeries = dfile.TimeSeries;
        
        % Get dimensions
        %   Don't use nConditions - as conditions might be excluded in a dset
        [nChannels, nFlies, nConditions, nEpochs] = getDimensionsFast(dsets{d});
        
        % Conditions in dataset
        [conditions, cond_labels] = getConditions(dsets{d});
        
    end
    
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

