%% Description

%{

View distributions of performances

View distribution of hctsa values for low (or high) performing features

%}

%% Settings

perf_type = 'nearestMedian'; % 'nearestMedian'; 'consis'
data_set = 'train'; % 'train', 'validate1'

preprocess_string = '_subtractMean_removeLineNoise';

%% Get valid features

[ch_valid_features, ch_excluded] = getValidFeatures_allChannels(data_set, preprocess_string);
nValid = sum(ch_valid_features, 2);
nValid_mean = mean(nValid)
nValid_min = min(nValid)
nValid_max = max(nValid)

% Treat all features as valid
%ch_valid_features = ones(size(ch_valid_features));

ch_valid_features = logical(ch_valid_features);

%% Get performance values and significance

[performances, performances_random, sig, ps, ps_fdr, sig_thresh, sig_thresh_fdr] = get_sig_features(perf_type, data_set, ch_valid_features, preprocess_string);

%% Plot distributions across features for each channel

figure;
set(gcf, 'Color', 'w');
sp_rows = 3;
sp_cols = 5;

for ch = 1 : size(consistencies, 1)
    subplot(sp_rows, sp_cols, ch);
    
    p = performances(ch, logical(ch_valid_features(ch, :)));
    
    h = histogram(p, 'DisplayStyle', 'stairs');
    hold on;
    
    steps = diff(sort(p, 'ascend'));
    steps(steps == 0) = [];
    minStep = min(steps);
    histBins = (0:minStep:1); % bins increase with smallest step difference in data
    %histBins = (0:0.01:1); % bins increase with arbitrary step size
    [histCounts] = histcounts(p, histBins);
    plot(histBins(1:end-1), histCounts, 'LineWidth', 2);
    
    pMean = mean(p);
    pMedian = median(p);
    ys = ylim;
    
    l(1) = line([pMean pMean], ys, 'LineStyle', '-', 'Color', 'k');
    l(2) = line([pMedian pMedian], ys, 'LineStyle', ':', 'Color', 'k');
    l(3) = line([0.5 0.5], ys, 'LineStyle', '--', 'Color', 'k');
    
    legend(l, 'mean', 'median', 'chance');
    legend('boxoff');
    
    ylim(ys);
    
    title([data_set ' ch' num2str(ch) newline ' mean=' num2str(pMean) ' median=' num2str(pMedian)]);
    xlabel(perf_type, 'interpreter', 'none');
end

%% Load HCTSA values

ch = 6;

preprocess_string = '_subtractMean_removeLineNoise';
source_prefix = 'train';

source_dir = ['../hctsa_space' preprocess_string '/'];
source_file = ['HCTSA_' source_prefix '_channel' num2str(ch) '.mat']; % HCTSA_train.mat; HCTSA_validate1.mat;

tic;
hctsa = load([source_dir source_file]);
toc

% Note - even with mixedSigmoid, feature 976 (870th valid feature) scales
%   from mostly close to 0 values (and 1 much bigger value) to NaNs and 0s
hctsa.TS_Normalised = BF_NormalizeMatrix(hctsa.TS_DataMat, 'mixedSigmoid');

%% Find lowest performing feature

ch = 6;

p = performances(ch, :);
p(~logical(ch_valid_features(ch, :))) = nan;

[pmin, feature] = min(p);

%% Sort performances
% Then you can take the Nth worst performing feature

p = performances(ch, :);
p(~logical(ch_valid_features(ch, :))) = nan;

[psorted, sortOrder] = sort(p, 'ascend');

N = 1;
pmin = psorted(N);
feature = sortOrder(N);

%% Visualise feature

fValues = hctsa.TS_DataMat(:, feature);

figure;
set(gcf, 'Color', 'w');
subplot(1, 2, 1);
plot(fValues);
xlabel('epoch');
ylabel('value');

subplot(1, 2, 2);
conds = (1:2);
cond_offsets = [-1 1];
cond_colours = {'r', 'b'};
for f = 1 : 13 % how to get number of flies?
    fly_rows = getIds({['fly' num2str(f)]}, hctsa.TimeSeries);
    
    for c = 1 : length(conds)
        cond_rows = getIds({['condition' num2str(conds(c))]}, hctsa.TimeSeries);
        
        xRand = cond_offsets(c)*0.2*rand(length(find(cond_rows & fly_rows)), 1);
        
        scatter(f+xRand, fValues(cond_rows & fly_rows), [cond_colours{c} 'x']);
        hold on;
    end
end

title([data_set ' ' perf_type ' ' num2str(pmin) ' f' num2str(feature)]);
xlabel('fly');
ylabel('value');