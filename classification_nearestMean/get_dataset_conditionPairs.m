function [cond_pairs] = get_dataset_conditionPairs(val_set)
% Inputs:
%   val_set = string; 'multidose', 'singledose', or 'sleep'
%
% Outputs:
%   cond_pairs = cell array; each cell holds gives a pair of condition IDs
%       Each pair gives one "unconscious" and one "conscious" condition
%       Each value in the pair gives the index of a condition ID in
%           get_dataset_conditions()

switch val_set
    case 'train'
        cond_pairs = {[1 2]};
    case {'multidose', 'multidose8', 'multidose4', 'multidoseBatchNormalised', 'multidose8BatchNormalised', 'multidose4BatchNormalised'}
        cond_pairs = {[1 3], [1 4], [1 5], [2 3], [2 4], [2 5]};
    case {'singledose', 'singledoseBatchNormalised'}
        cond_pairs = {[1 2], [1 3]}; %, [1 4]};
    case {'sleep', 'sleepBatchNormalised'}
        cond_pairs = {[1 3], [1 4], [2 3], [2 4]};
end

end