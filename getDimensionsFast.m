function [nChannels, nFlies, nConditions, nEpochs] = getDimensionsFast(source_prefix)
%GETDIMENSIONS
%   Gets dimensions for the given dataset
%   Dimensions are hardcoded - use getDimensions() to get from file
%
% Inputs:
%   source_prefix = 'train', 'multidose' 'multidose8', 'multidose4',
%		'singledose', 'sleep'
% Outputs:
%   nChannels
%   nFlies
%   nConditions
%   nEpochs

switch source_prefix
    case 'train'
        nChannels = 15;
        nEpochs = 8;
        nFlies = 13;
        nConditions = 2;
    case 'multidose'
        nChannels = 15;
        nEpochs = 112;
        nFlies = 12;
        nConditions = 5;
	case 'multidose8'
		nChannels = 15;
		nEpochs = 112;
		nFlies = 8;
		nConditions = 5;
	case 'multidose4'
		nChannels = 15;
		nEpochs = 112;
		nFlies = 4;
		nConditions = 5;
    case 'singledose'
        nChannels = 15;
        nEpochs = 8;
        nFlies = 18;
        nConditions = 4;
    case 'sleep'
        nChannels = 15;
        nEpochs = 8;
        nFlies = 19;
        nConditions = 2;
end

end