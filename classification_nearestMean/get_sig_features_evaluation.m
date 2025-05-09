function [performances, performances_random, sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr] = get_sig_features_evaluation(perf_type, data_set, valid_features, preprocess_string)
% Get features which perform significantly better than chance
% Conducts FDR correction for multiple corrections, on valid features
%
% Inputs:
%   perf_type = 'nearestMean' or 'nearestMedian' or 'consis'
%   data_type = 'train' or 'validate1'
%   valid_features = logical matrix (channels x features); 1/0 for
%       valid/invalid feature for the channel
%   preprocess_string = 'string'
% Outputs
%   performances = cell array;
%       Each entry holds a matrix (channels x features) of
%           accuracy/consistencies corresponding to one condition pairing
%   performances_random = cell array;
%       Each entry holds a matrix (channels x features) of random
%           accuracy/consistencies corresponding to one condition pairing
%   sig = logical matrix (channels x features x pairs)
%   ps = matrix (channels x features x pairs)
%       Holds uncorrected p-values
%   ps_fdr = matrix (channels x pairs)
%       Holds corrected p-value thresholds
%   sig_thresh = vector (pairs x 1)
%       Holds threshold values for p < .05, uncorrected across all
%       features
%   sig_thresh_fdr = matrix (channels x pairs)
%       % Holds threshold values for each channel for corrected p < .05

%% Deal with filenames

switch(perf_type)
    case {'nearestMedian', 'nearestMean'}
        perf_string = ['class_' perf_type];
        data_string = [data_set '_accuracy'];
    case 'consis'
        perf_string = 'consis_nearestMedian';
        switch(data_set)
            case {'multidose', 'singledose', 'sleep', 'multidose8', 'multidose4'}
                data_string = data_set;
            case {'multidoseBatchNormalised', 'singledoseBatchNormalised', 'sleepBatchNormalised', 'multidose8BatchNormalised', 'multidose4BatchNormalised'}
                data_string = data_set(1:end-15); % remove 'BatchNormalised'
        end
end

% File locations
source_dir = ['results' preprocess_string '/'];
source_file = [perf_string '_' data_string '.mat'];
[filepath, filename, ext] = fileparts(mfilename('fullpath')); % files relative to this function file
%hctsa_prefix = ['../hctsa_space/' hctsa_string];

%% Load performance files

%perf = load([source_dir source_file]);
perf = load(fullfile(filepath, source_dir, source_file));

% Get conditions and condition pairings
[conditions, condition_ids] = get_dataset_conditions(data_set);
cond_pairs = get_dataset_conditionPairs(data_set);
cond_pairs = cat(1, cond_pairs{:}); % (pairs x condition IDs) matrix

performances = cell(1, size(cond_pairs, 1));
switch(perf_type)
    case {'nearestMedian', 'nearestMean'}
        
        for pair = 1 : size(cond_pairs, 1)
            performances{pair} =...
                (perf.accuracies_perCondition{cond_pairs(pair, 1)} +...
                perf.accuracies_perCondition{cond_pairs(pair, 2)})...
                ./ 2;
        end
        
    case 'consis'
        
        for pair = 1 : size(cond_pairs, 1)
            % average across epochs, flies
            performances{pair} = mean(mean(perf.consistencies{pair}, 4), 3);
        end
        
end

%% Load chance distribution

switch(data_set)
    case {'multidose', 'singledose', 'sleep', 'multidose8', 'multidose4',}
        data_string = data_set;
    case {'multidoseBatchNormalised', 'singledoseBatchNormalised', 'sleepBatchNormalised', 'multidose8BatchNormalised', 'multidose4BatchNormalised'}
        data_string = data_set(1:end-15); % remove 'BatchNormalised'
end

switch(perf_type)
    case {'nearestMedian', 'nearestMean'}
        rand_string = 'class_random';
        data_string = [data_string '_accuracy'];
    case 'consis'
        % average across epochs, flies
        rand_string = 'consis_random';
end

rand_file = [rand_string '_' data_string];
perf_random = load(fullfile(filepath, source_dir, rand_file));
performances_random = cell(1, size(cond_pairs, 1));
switch(perf_type)
    case {'nearestMedian', 'nearestMean'}
        
        for pair = 1 : size(cond_pairs, 1)
            performances_random{pair} =...
                (perf_random.accuracies_perCondition{cond_pairs(pair, 1)} +...
                perf_random.accuracies_perCondition{cond_pairs(pair, 2)})...
                ./ 2;
        end
        
    case 'consis'
        
        for pair = 1 : size(cond_pairs, 1)
            % average across epochs, flies
            performances_random{pair} = mean(mean(perf_random.consistencies_random{pair}, 4), 3);
        end
        
end

%% Find significantly performing features
% One-tailed, better than chance
% p-value from distribution: https://www.jwilber.me/permutationtest/

alpha = 0.05;
q = 0.05;

% Note - same number of channels and features across all condition pairs
ps = nan([size(performances{1}) size(cond_pairs, 1)]); % (ch x feature x pair)
ps_fdr = nan([size(performances{1}, 1) size(cond_pairs, 1)]); % (ch x pair)
sig_thresh = nan(size(cond_pairs, 1), 1);
sig_thresh_fdr = nan(size(ps_fdr));
for pair = 1 : size(cond_pairs, 1)
    
    % Get threshold from chance distribution
    chance_dist = performances_random{pair}(1, :); % just use the same chance distribution for all channels
    sig_thresh(pair) = prctile(chance_dist, (1-alpha)*100); % 95%tile -> alpha = 0.05
    
    % Compare each feature to threshold, get p-value
    for ch = 1 : size(performances{pair}, 1)
        for f = 1 : size(performances{pair}, 2)
            % Find how many in chance distribution are greater
            nBetter = sum(chance_dist > performances{pair}(ch, f));
            % p-value
            ps(ch, f, pair) = nBetter / numel(chance_dist);
        end
    end
    
    % Conduct FDR correction per channel
    for ch = 1 : size(performances{pair}, 1)
        
        % FDR
        [pID, pN] = FDR(ps(ch, logical(valid_features(ch, :)), pair), q);
        ps_fdr(ch, pair) = pID;
        
        % Get corresponding accuracy
        sig_thresh_fdr(ch, pair) = prctile(chance_dist, (1-ps_fdr(ch, pair))*100);
        
    end
    
end

% Get which features are significant after FDR
ps_fdr_repeat = permute(ps_fdr, [1 3 2]);
ps_fdr_repeat = repmat(ps_fdr_repeat, [1 size(ps, 2) 1]);
sig = ps < ps_fdr_repeat;

end

