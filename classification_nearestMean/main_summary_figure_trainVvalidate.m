%% Description

%{

Plot and get summary figures for training set cross-validation

%}

%%

perf_type = 'consis'; % nearestMedian; consis
data_sets = {'train', 'validate1'};

preprocess_string = '_subtractMean_removeLineNoise';

addpath('../');

%% Get valid features

[ch_valid_features, ch_excluded] = getValidFeatures_allChannels('train', preprocess_string);
nValid = sum(ch_valid_features, 2);
nValid_mean = mean(nValid)
nValid_min = min(nValid)
nValid_max = max(nValid)

%% Common plotting variables

% Channel colours
r = linspace(0, 1, size(ch_valid_features, 1))';
g = linspace(0, 0, size(ch_valid_features, 1))';
b = linspace(1, 0, size(ch_valid_features, 1))';
ch_colours = cat(2, r, g, b);

%%

figure;
set(gcf, 'Color', 'w');

%% Feature performance distributions

rows = 1;
columns = 2;

plot_counter = 1;
for dset = 1 : length(data_sets)
    data_set = data_sets{dset};
    
    % Get performance values and significances
    [performances, performances_random, sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr] = get_sig_features(perf_type, data_set, ch_valid_features, preprocess_string);
    
    % Plot distribution across valid features for each channel
    subplot(rows, columns, dset);
    [h] = plot_perfDistribution(ch_valid_features, ch_colours, performances, sig_thresh_fdr, performances_random, sig_thresh);
    title([data_set], 'interpreter', 'none');
        if strcmp(data_set, 'train')
            xlim([0.4 0.8]);
        elseif strcmp(data_set, 'validate1')
            xlim([0.4 0.7]);
            if ptype == 2
                xlim([0.4 0.8]);
            end
        end
    
    %legend('sig. thres.');
    
    ylabel('accuracy');
    
    plot_counter = plot_counter + 1;

end
