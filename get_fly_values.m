function [values, fDetails] = get_fly_values(ch, fID, perf_type, scale_vals, preprocess_string)
% Get from each fly the values for a given feature
%
% Inputs:
%   ch = integer; which channel to obtain values
%   fID = feature ID to get values for
%   perf_type = string; performance type to look at
%       'class_nearestMean';
%       'class_nearestMedian';
%       'consis_nearestMean'
%   scale_vals = 1/0; 1 to scale values; 0 otherwise
%   preprocess_string = string; preprocessing stream identifier
% Outputs:
%   values = cell array (flies x conditions); each holds a vector of hctsa
%       values for the 'best' feature
%   fDetails = struct containing details of the 'best' feature
%       .fID = feature ID
%       .fName = feature name
%       .mID = master ID
%       .mName = master name

% Files are relative to this function file
source_dir = ['classification_nearestMean/results' preprocess_string];
[filepath, filename, ext] = fileparts(mfilename('fullpath'));
source_dir = [filepath filesep source_dir filesep];
addpath([filepath filesep '..' filesep]);

hctsa_prefixes = {'train', 'validate1'};

if strcmp(perf_type, 'class_nearestMean') || strcmp(perf_type, 'class_nearestMedian') 
    perf_sets = {'crossValidation', 'validate1_accuracy'};
elseif strcmp(perf_type, 'consis_nearestMedian')
    perf_sets = {'train', 'validate1'};
end

% Get values for each fly
fly_rows = ones(1, 2); % each row n holds the starting rows of the nth fly for each condition
fly_vals = cell(1, 2); % vector for each condition (can we assume equal observations per condition?)
f_counter = 1;
for d = 1 : length(hctsa_prefixes)
    [nChannels, nFlies, nConditions, nEpochs] = getDimensions(hctsa_prefixes{d});
    
    % Load values
    hctsa = hctsa_load(hctsa_prefixes{d}, ch, preprocess_string);
    
    if d == 1
        % Look up the feature
        fID = hctsa.Operations{fID, 'ID'};
        fName = hctsa.Operations{fID, 'Name'};
        % Look up master feature
        mID = hctsa.Operations{fID, 'MasterID'};
        mName = hctsa.MasterOperations{mID, 'Label'};
        perf = nan(size(hctsa_prefixes));
        fDetails.fID = fID;
        fDetails.fName = fName;
        fDetails.mID = mID;
        fDetails.mName = mName;
    end
    
    % Get values per fly
    %   Keep track of what rows belong to which flies
    for f = 1 : nFlies
        for c = 1 : 2
            fRows = find(getIds({['fly' num2str(f)], ['condition' num2str(c)]}, hctsa.TimeSeries));
            tmp = hctsa.TS_DataMat(fRows, fID);
            fly_vals{c} = cat(1, fly_vals{c}, tmp);
            fly_rows(f_counter+1, c) = fly_rows(f_counter, c) + length(fRows);
        end
        f_counter = f_counter + 1;
    end
    
    % Get performance of the feature
    source_file = [perf_type '_' perf_sets{d} '.mat'];
    acc = load([source_dir source_file]);
    if strcmp(perf_type, 'consis_nearestMedian')
        acc.accuracies = mean(acc.consistencies, 4);
    end
    tmp = mean(acc.accuracies, 3); % average accuracies across cross-validations
    perf(d) = tmp(ch, fID);
    
end
fDetails.perf = perf;

% Combine classes to scale altogether
cond_rows = [0 cumsum(cellfun(@length, fly_vals))]; % each gives the last row in each class
vals_all = cat(1, fly_vals{:});

% Scale values (but note the threshold will need to be scaled too...
if scale_vals
    vals_all = BF_NormalizeMatrix(vals_all, 'mixedSigmoid');
end

% Separate classes again
for c = 1 : length(cond_rows)-1
    fly_vals{c} = vals_all(cond_rows(c)+1:cond_rows(c+1));
end

% Separate values per fly
values = cell(f_counter-1, nConditions);
for f = 1 : f_counter-1
    for c = 1 : size(fly_vals, 2)
        if f == size(fly_rows, 1)
            values{f, c} = fly_vals{c}(fly_rows(f, c):end);
        else
            values{f, c} = fly_vals{c}(fly_rows(f, c):fly_rows(f+1, c)-1);
        end
    end
end

end

