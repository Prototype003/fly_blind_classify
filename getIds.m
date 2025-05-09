function [logical_ids, row_ids] = getIds(keys, TimeSeries, join_type)
%getIds Get row-ids for certain keywords
%
% Inputs:
%   keys = 1D cell array; each cell contains a keyword to match
%       All keywords need to match (by default)
%   TimeSeries = matrix; TimeSeries matrix from HCTSA file
%   join_type = 'and' | 'or'; optional
%       Join keywords with either and (&) or or (|) operation
%
% Outputs:
%   logical_ids = 1D logical vector; 1s correspond to rows in TimeSeries
%       which have to required keywords
%   row_ids = 1D vector; holds rows in TimeSeries which have the required
%       keywords

% SUB_cell2cellcell() is a HCTSA function which converts table column of
%   csv strings into a column of cells, each cell containing a cell array
%   with each cell containing a csv value

if ~exist('join_type', 'var')
    join_type = 'and';
end

keywords = SUB_cell2cellcell(TimeSeries.Keywords);

switch join_type
    case 'and'
        ids = ones(size(keywords));
    case 'or'
        ids = zeros(size(keywords));
end

for k = 1 : length(keys)
    key = keys{k};
    
    switch join_type
        case 'and'
            ids = ids & cellfun(@(x) any(ismember(key, x)), keywords);
        case 'or'
            ids = ids | cellfun(@(x) any(ismember(key, x)), keywords);
    end
end

logical_ids = ids;
row_ids = find(ids);

end

