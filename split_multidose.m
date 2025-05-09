function [hctsa_split] = split_multidose(hctsa)
%SPLIT_MULTIDOSE
%   Split hctsa dataset (multidose flies) into two groups
%       multidose -> multidose8, multidose4
%
% Inputs:
%   hctsa = struct(); holds hctsa data for multidose dataset
%
% Outputs:
%   hctsa_split = cell; holds a struct of hctsa data for each of
%       multidose8 and multidose4 (in that order)

fly_groups = {(1:8), (9:12)};

% Get fly strings for each entry
data_labels = hctsa.TimeSeries.Keywords;
keyword_cell = cellfun(@split, data_labels, repmat({','}, size(data_labels)), 'UniformOutput', false);
fly_key_pos = cellfun(@contains, keyword_cell, repmat({'fly'}, size(keyword_cell)), 'UniformOutput', false);
fly_key = cellfun(@(x, y) x(y), keyword_cell, fly_key_pos, 'UniformOutput', false);
fly_key = cellfun(@(x) x{1}, fly_key, 'UniformOutput', false); % get rid of the outer cell array

% Get unique flies and sort ascending (1-12)
all_flies = unique(fly_key);
flyIDs_unsorted = str2double(cellfun(@(x) num2str(x(4:end)), all_flies, 'UniformOutput', false));
[sorted, sort_order] = sort(flyIDs_unsorted);
all_flies = all_flies(sort_order);

% In case there are more dimensions (e.g. when joining across channels etc)
trailing_dims = repmat({':'}, [1 length(size(hctsa.TS_DataMat))-1]);

fly_group_ids = cell(size(fly_groups));
hctsa_split = cell(size(fly_groups));
for g = 1 : length(fly_groups)
    
    % Find which entries to put in the group
    fly_group_ids{g} = getIds(all_flies(fly_groups{g}), hctsa.TimeSeries, 'or');
    
    % Extract the relevant data for the group
    hctsa_split{g} = hctsa;
    hctsa_split{g}.TS_CalcTime = hctsa.TS_CalcTime(fly_group_ids{g}, trailing_dims{:});
    hctsa_split{g}.TS_DataMat = hctsa.TS_DataMat(fly_group_ids{g}, trailing_dims{:});
    hctsa_split{g}.TS_Quality = hctsa.TS_Quality(fly_group_ids{g}, trailing_dims{:});
    hctsa_split{g}.TimeSeries = hctsa.TimeSeries(fly_group_ids{g}, :); % Note this is a table, can't add extra dimensions
    hctsa_split{g}.op_clust.ord = hctsa.op_clust.ord(fly_group_ids{g});
    hctsa_split{g}.ts_clust.ord = hctsa.ts_clust.ord(fly_group_ids{g});
    
end


end

