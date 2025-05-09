function [conditions, cond_labels, cond_colours, stats_order, main] = getConditions(dset)
% Get condition names in a given dataset
%
% Inputs:
%   dset = string; 'train', 'multidose', 'multidose8', 'multidose4', 'singledose', 'sleep'
%
% Outputs:
%   conditions = cell array of strings; each string is a condition name
%   cond_labels = cell array of strings; each string is a shortened label
%   cond_colours = cell array of colours
%   stats_order = vector; use to reorder conditions to match order of
%       conditions in results stats structure,
%       e.g. stats.sleep.nearestMedian.conditions
%   main = vector; indices of conditions for "main" comparison;
%       wake first, then unconscious condition

switch dset
    case 'train'
        conditions = {'condition1', 'condition2'};
        cond_labels = {'W', 'A'};
        cond_colours = {'r', 'b'};
        stats_order = [1 2];
        main = [1 2];
    case {'multidose', 'multidose8', 'multidose4'}
        conditions = {'conditionWake', 'conditionIsoflurane_0.6', 'conditionIsoflurane_1.2', 'conditionPost_Isoflurane', 'conditionRecovery'};
        cond_labels = {'W1', 'A0.6', 'A1.2', 'W2', 'WR'};
        cond_colours = {'r', 'b', 'b', 'm', 'm'};
        stats_order = [3 2 1 4 5];
        main = [1 3];
    case 'singledose'
        conditions = {'conditionWake', 'conditionIsoflurane', 'conditionPostIsoflurane'};
        cond_labels = {'W1', 'A', 'W2'};
        cond_colours = {'r', 'b', 'm'};
        stats_order = [2 1 3];
        main = [1 2];
    case 'sleep'
        conditions = {'conditionwakeEarly', 'conditionwake', 'conditionsleepEarly', 'conditionsleepLate'};
        cond_labels = {'WE', 'W' 'SE', 'DS'};
        cond_colours = {'r', 'r', 'b', 'b'};
        stats_order = [4 3 2 1];
        main = [2 3];
end

end

