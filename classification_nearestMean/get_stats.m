function [stats, stats2] = get_stats(preprocess_string)
% Get classification performances and related stats for each dataset and
%   performance type
%
% Inputs:
%   ch_valid_features = logical matrix (channels x features); 1/0 for
%       valid/invalid feature for the channel
%   preprocess_string = string; preprocessing stream identifier
% Outputs:
%   stats = struct holding performance stats for each dataset
%       stats.(train|validate1).(nearestMedian|consis) has fields
%           performances, performances_random, sig, ps, ps_fdr, sig_thresh
%               sig_thresh_fdr
%   stats2 = same as stats, but includes performance stats for evaluation
%       datasets (stats.(train|multidose|singledose|sleep).(nearestMedian|consis)
%       Fields are same as for stats, but are cell arrays
%           Each cell array gives the stats for a given condition pairing
%       Includes fields conditions, condition_ids, condition_pairs

stats = struct();
perf_types = {'nearestMedian', 'consis'};
data_strings = {'train', 'validate1', 'validate1BatchNormalised'};
data_sets = {'train', 'validate1', 'validate1'};

for d = 1 : length(data_strings)
    stats.(data_strings{d}) = struct();
    
    % Get valid features
    [valid_features, ch_excluded] = getValidFeatures_allChannels(data_sets{d}, preprocess_string);
    stats.(data_strings{d}).valid_features = valid_features;
    
    for p = 1 : length(perf_types)
        stats.(data_strings{d}).(perf_types{p}) = struct();
        
        [performances, performances_random, sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr] =...
            get_sig_features(perf_types{p}, data_strings{d}, valid_features, preprocess_string);
        
        stats.(data_strings{d}).(perf_types{p}).performances = performances;
        stats.(data_strings{d}).(perf_types{p}).performances_random = performances_random;
        stats.(data_strings{d}).(perf_types{p}).sig = sig;
        stats.(data_strings{d}).(perf_types{p}).ps = ps;
        stats.(data_strings{d}).(perf_types{p}).ps_fdr = ps_fdr;
        stats.(data_strings{d}).(perf_types{p}).sig_thresh = sig_thresh;
        stats.(data_strings{d}).(perf_types{p}).sig_thresh_fdr = sig_thresh_fdr;
        
    end
end

end

