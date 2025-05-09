%% Description

%{

Compare number of significant features depending on preprocessing

%}

%%

perf_type = 'nearestMedian'; % 'nearestMedian'; 'nearestMean'; 'consis'
data_set = 'train'; % 'train', 'validate1'

ref_type = 'nearestMedian';
ref_set = 'train';

preprocess_strings = {'_removeLineNoise', '_subtractMean_removeLineNoise'};

%% Get valid features

[ch_valid_features, ch_excluded, valid_perStage] = getValidFeatures_allChannels('train', preprocess_strings{1});
nValid = sum(ch_valid_features, 2);
nValid_mean = mean(nValid)
nValid_min = min(nValid)
nValid_max = max(nValid)

valids = cell(1, 2);
valids{2} = valid_perStage(:, :, 2); % constant values
valids{3} = ch_valid_features; % NaN and constant values
valids{1} = ones(size(ch_valid_features)); % treat all features as valid

valid_labels = {'all', 'exclude_const_only', 'exclude'};

nChannels = size(ch_valid_features, 1);

%% Get performance and significance of reference sets, for comparison

sigs = [];
legend_labels = cell(1, length(preprocess_strings)*length(valids));

c = 1;
for prep = 1 : length(preprocess_strings)
    for v = 1 : length(valids)
        
        [performances, performances_random, sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr] =...
            get_sig_features(perf_type, data_set, valids{v}, preprocess_strings{prep});
        
        sigs = cat(2, sigs, sum(sig, 2));
        
        legend_labels{c} = [preprocess_strings{prep} '+' valid_labels{v}];
        c = c+1;
    end
end

%% Plot

figure;
bar(sigs);
xlabel('channel')
legend(legend_labels, 'interpreter', 'none');
title([perf_type ' ' data_set ' sig. features']);
