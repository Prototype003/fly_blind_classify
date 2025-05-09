function [stats] = get_stats_evaluation_multidoseSplit(preprocess_string)
% Get classification performances and related stats for each dataset and
%   performance type
%
% Inputs:
%   preprocess_string = string; preprocessing stream identifier
% Outputs:
%   stats = struct holding performance stats for each dataset
%       stats.(train|multidose|singledose|sleep).(nearestMedian|consis) has
%           fields:
%               cell arrays: each cell gives performances for a pairing
%                   performances, performances_random
%               matrices: hold all condition pairings
%                   sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr,
%                   conditions, condition_ids, condition_pairs

stats = struct();
perf_types = {'nearestMedian', 'consis'};
data_strings = {'train',...
    'multidose8', 'multidose8BatchNormalised',...
    'multidose4', 'multidose4BatchNormalised',...
    'singledose', 'singledoseBatchNormalised',...
    'sleep', 'sleepBatchNormalised'};
data_sets = {'train',...
    'multidose', 'multidose',...
    'multidose', 'multidose',...
    'singledose', 'singledose',...
    'sleep', 'sleep'};

% Avoid getting valid features twice for multidose dataset (because it
% takes time
multidose_valid_done = 0;

for d = 1 : length(data_strings)
    stats.(data_strings{d}) = struct();
    
    switch data_sets{d}
        case 'multidose'
            if multidose_valid_done == 0
                % Get valid features
                [valid_features, ch_excluded] = getValidFeatures_allChannels(data_sets{d}, preprocess_string);
                multidose_valid = valid_features;
            end
            stats.(data_strings{d}).valid_features = multidose_valid;
            multidose_valid_done = 1;
        otherwise
            % Get valid features
            [valid_features, ch_excluded] = getValidFeatures_allChannels(data_sets{d}, preprocess_string);
            stats.(data_strings{d}).valid_features = valid_features;
    end
    
    % Get conditions and condition pairings
    [conditions, condition_ids] = get_dataset_conditions(data_sets{d});
    cond_pairs = get_dataset_conditionPairs(data_sets{d});
    cond_pairs = cat(1, cond_pairs{:}); % (pairs x condition IDs) matrix
    
    for p = 1 : length(perf_types)
        stats.(data_strings{d}).(perf_types{p}) = struct();
        
        switch data_sets{d}
            case 'train' % convert to cell arrays to match evaluation formats
                
                [performances, performances_random, sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr] =...
                    get_sig_features(perf_types{p}, data_strings{d}, valid_features, preprocess_string);
                
                stats.(data_strings{d}).(perf_types{p}).performances = {performances};
                stats.(data_strings{d}).(perf_types{p}).performances_random = {performances_random};
                stats.(data_strings{d}).(perf_types{p}).sig = sig;
                stats.(data_strings{d}).(perf_types{p}).ps = ps;
                stats.(data_strings{d}).(perf_types{p}).ps_fdr = ps_fdr;
                stats.(data_strings{d}).(perf_types{p}).sig_thresh = sig_thresh;
                stats.(data_strings{d}).(perf_types{p}).sig_thresh_fdr = sig_thresh_fdr;
                stats.(data_strings{d}).(perf_types{p}).conditions = conditions;
                stats.(data_strings{d}).(perf_types{p}).condition_ids = condition_ids;
                stats.(data_strings{d}).(perf_types{p}).condition_pairs = cond_pairs;
                
            otherwise % multidose, singledose, sleep
                
                [performances, performances_random, sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr] =...
                    get_sig_features_evaluation(perf_types{p}, data_strings{d}, valid_features, preprocess_string);
                
                stats.(data_strings{d}).(perf_types{p}).performances = performances;
                stats.(data_strings{d}).(perf_types{p}).performances_random = performances_random;
                stats.(data_strings{d}).(perf_types{p}).sig = sig;
                stats.(data_strings{d}).(perf_types{p}).ps = ps;
                stats.(data_strings{d}).(perf_types{p}).ps_fdr = ps_fdr;
                stats.(data_strings{d}).(perf_types{p}).sig_thresh = sig_thresh;
                stats.(data_strings{d}).(perf_types{p}).sig_thresh_fdr = sig_thresh_fdr;
                stats.(data_strings{d}).(perf_types{p}).conditions = conditions;
                stats.(data_strings{d}).(perf_types{p}).condition_ids = condition_ids;
                stats.(data_strings{d}).(perf_types{p}).condition_pairs = cond_pairs;
                
        end
        
    end
end

% Save it as well
try
    out_file = ['results' preprocess_string filesep 'stats_multidoseSplit.mat'];
    save(out_file, 'stats');
    disp(['saved ' out_file]);
catch
    disp('stats file not saved');
end

end

