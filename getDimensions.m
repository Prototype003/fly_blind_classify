function [nChannels, nFlies, nConditions, nEpochs] = getDimensions(source_prefix)
%GETDIMENSIONS
%   Gets dimensions for the given dataset
%
% Inputs:
%   source_prefix = 'train'; 'validate1', 'multidose', 'singledose',
%       'sleep'
% Outputs:
%   nChannels
%   nFlies
%   nConditions
%   nEpochs

% Files are relative to this function file
[filepath, filename, ext] = fileparts(mfilename('fullpath'));

disp(['getting ' source_prefix ' dimensions']);

%source_prefix = ['HCTSA_' source_prefix];

switch source_prefix
    case 'train'
        tic;
        tmp = load([filepath filesep 'data/preprocessed/fly_data_removeLineNoise.mat']);
        nChannels = size(tmp.data.train, 2);
        nEpochs = size(tmp.data.train, 3);
        nFlies = size(tmp.data.train, 4);
        nConditions = size(tmp.data.train, 5);
        toc
    case 'validate1'
        % Assumes equal dimensions for all flies, etc.
        tic;
        tmp = load([filepath filesep 'data/labelled/labelled_data_01.mat']);
        nChannels = size(tmp.labelled_shuffled_data(1).data, 1);
        nFlies = numel(unique([tmp.labelled_shuffled_data.fly_ID]));
        nConditions = numel(unique({tmp.labelled_shuffled_data.TrialType})); % should be 2
        nChunks = numel(unique([tmp.labelled_shuffled_data.chunk_number]));
        tLength = size(tmp.labelled_shuffled_data(1).data, 2);
        tmp = load([filepath filesep 'data/preprocessed/fly_data_removeLineNoise.mat']);
        nEpochs = nChunks * floor(tLength/size(tmp.data.validate1, 1));
        toc
    case {'multidose', 'singledose', 'sleep'}
        tic;
        tmp = load([filepath filesep 'data/preprocessed/flyEvaluation_data_subtractMean_removeLineNoise.mat']);
        kw = split(tmp.data.(source_prefix).keywords, ',');
        nChannels = length(unique(kw(:, 2), 'stable'));
        nFlies = length(unique(kw(:, 1), 'stable'));
        nConditions = length(unique(kw(:, 4), 'stable'));
        epochs = unique(kw(:, 3), 'stable');
        if size(kw, 2) == 5
            segments = unique(kw(:, 5), 'stable');
        else
            segments = 1;
        end
        nEpochs = length(epochs) * length(segments);
        toc
end

end