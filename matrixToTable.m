function T = matrixToTable(mat, fieldNames, fieldTypes)
    % Converts a multidimensional matrix into a table, with each dimension
    % treated as one column in the table. Allows customization of field names
    % and data types.
    %
    % Inputs:
    %   mat - A multidimensional matrix
    %   fieldNames - A cell array of strings specifying the column names.
    %                Must have ndims(mat) + 1 entries. Last entry is for
	%                the name of the matrix values.
    %   fieldTypes - A cell array of strings specifying the data type for
    %                each column. Must have ndims(mat) + 1 entries.
    %                Supported types: 'numeric', 'categorical', 'discrete'.
    %
    % Output:
    %   T - A table with the specified field names and data types
    
    % Get the size of the input matrix
    dims = size(mat);
    numDims = ndims(mat);

    % Validate the fieldNames input
    if nargin < 2 || isempty(fieldNames)
        % Generate default column names if none are provided
        fieldNames = [arrayfun(@(i) sprintf('Dim%d', i), 1:numDims, 'UniformOutput', false), {'Value'}];
    elseif length(fieldNames) ~= numDims + 1
        error('fieldNames must have %d entries (one for each dimension plus one for values)', numDims + 1);
    end

    % Validate the fieldTypes input
    if nargin < 3 || isempty(fieldTypes)
        % Default to 'numeric' for all fields
        fieldTypes = repmat({'numeric'}, 1, numDims + 1);
    elseif length(fieldTypes) ~= numDims + 1
        error('fieldTypes must have %d entries (one for each dimension plus one for values)', numDims + 1);
    end

    % Generate all linear indices for the matrix
    numElements = numel(mat);
    subs = cell(1, numDims); % Initialize cell array for subscripts
    [subs{:}] = ind2sub(dims, 1:numElements); % Generate subscripts

    % Combine the subscripts and the values into one array
    data = [cat(1, subs{:})' mat(:)];

    % Convert to a table
    T = array2table(data, 'VariableNames', fieldNames);

    % Apply field types
    for i = 1:length(fieldTypes)
        switch fieldTypes{i}
            case 'categorical'
                T.(fieldNames{i}) = categorical(T.(fieldNames{i}));
            case 'discrete'
                T.(fieldNames{i}) = uint8(T.(fieldNames{i}));
            case 'numeric'
                % Already numeric by default
                continue;
            otherwise
                error('Unsupported field type: %s. Supported types: numeric, categorical, discrete.', fieldTypes{i});
        end
    end
end