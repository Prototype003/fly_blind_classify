%% Settings

preprocess_string = '_subtractMean_removeLineNoise';

dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
dset_ids = (1:length(dsets));

%% Load stats
% Get all performances and stats using
%   get_stats_evaluation(preprocess_string)

stats_dir = ['classification_nearestMean/results' preprocess_string filesep];
stats_file = 'stats_multidoseSplit.mat';

% Should have variable stats
load([stats_dir stats_file]);

%% Features which are valid in all datasets

ch = 6;

valid_all = ones(size(stats.train.valid_features));
for d = 1 : length(dsets)
    
    disp(['====']);
    
    tmp = stats.(dsets{d}).valid_features;
    
    disp(['ch' num2str(ch) '-' dsets{d} ': ' num2str(numel(find(tmp(ch, :))))]);
        
    valid_all = valid_all & tmp;
    
    disp(['total ' num2str(numel(find(valid_all(ch, :)))) ' valid across datasets']);
end

%% Load hctsa files
% And join channels together

data_sources = {'train', 'multidose', 'singledose', 'sleep'};

source_dir = ['hctsa_space' preprocess_string '/'];

hctsas = cell(size(data_sources));

for d = 1 : length(data_sources)
    
    ch = 1;
    source_file = ['HCTSA_' data_sources{d} '_channel' num2str(ch) '.mat'];
    disp(['loading ' source_file]);
    tic;
    hctsas{d} = load([source_dir source_file]);
    t = toc;
    disp(['loaded in ' num2str(t) 's']);
    
    tmp = nan([size(hctsas{d}.TS_DataMat) size(valid_all, 1)-1]);
    hctsas{d}.TS_DataMat = cat(3, hctsas{d}.TS_DataMat, tmp);
    
    for ch = 2 : size(valid_all, 1)
        
        source_file = ['HCTSA_' data_sources{d} '_channel' num2str(ch) '.mat'];
        disp(['loading ' source_file]);
        tic;
        hctsa_ch = load([source_dir source_file]);
        t = toc;
        disp(['loaded in ' num2str(t) 's']);
        
        % For now, only copy feature values, ignore the other stuff
        %   Because we are only plotting feature values
        hctsas{d}.TS_DataMat(:, :, ch) = hctsa_ch.TS_DataMat;
        
    end
    
end

%% Split multidose dataset into multidose8 and multidose4

disp('splitting multidose');
tic;
md_split = split_multidose(hctsas{2});

hctsas{5} = md_split{1};
hctsas{6} = md_split{2};

hctsas = hctsas([1 5 6 3 4]);
toc

dsets = {'train', 'multidose8', 'multidose4', 'singledose', 'sleep'};
dset_bases = {'train', 'multidose', 'multidose', 'singledose', 'sleep'};

%% Replace values with nans for invalid features
% Only care about features which are valid across all datasets

for d = 1 : length(hctsas)
    
    for ch = 1 : size(hctsas{d}.TS_DataMat, 3)
        tic;
        for f = 1 : size(hctsas{d}.TS_DataMat, 2)
            
            if valid_all(ch, f) == 0
                hctsas{d}.TS_DataMat(:, f, ch) = nan;
            end
            
        end
        toc
    end
    
end

disp('invalid features at each channel replaced with nans');

%% Exclude rows corresponding to particular unused conditions

valid_rows = cell(size(dsets));
for d = 1 : length(dsets)
    
    switch dsets{d}
        case 'train'
            keywords = {};
        case {'multidose', 'multidose4', 'multidose8'}
            keywords = {};
        case 'singledose' % exclude recovery condition
            keywords = {'conditionRecovery'};
        case 'sleep'
            keywords = {};
    end
    
    % Find corresponding rows
    match = zeros(size(hctsas{d}.TimeSeries, 1), 1);
    for kw = keywords
        match = match | getIds(kw, hctsas{d}.TimeSeries);
    end
    
    valid_rows{d} = ~match;
    
end

%%

plot_dset = 1;

%% Normalise data per channel, per dataset
% Normalise across datasets or per dataset?
% Per dataset - all datasets will have the same range of values
% Across datasets - all datasets will be relative to discovery flies

% Normalisation per dataset

% Note - even with mixedSigmoid, some features scale to NaNs and 0s
%   discovery flies - feature 976 (870th valid feature) scales
%       to NaN a nd 0s
%   sleep flies - feature 977 scales to NaN and a 0

% Time to scale every channel, multidose8 - ~836s

for d = 1 : 1%length(hctsas)
    disp(['scaling dataset ' dsets{d}]);
    
    hctsas{d}.TS_Normalised = nan(size(hctsas{d}.TS_DataMat));
    
    for ch = 1 : size(hctsas{d}.TS_Normalised, 3)
        tic;
        
        hctsas{d}.TS_Normalised(valid_rows{d}, :, ch) =...
            BF_NormalizeMatrix(hctsas{d}.TS_DataMat(valid_rows{d}, :, ch), 'mixedSigmoid');
        
        t = toc;
        disp(['ch' num2str(ch) ' scaled in ' num2str(t) 's']);
    end
    
end

channels_separate = hctsas{plot_dset}.TS_Normalised;

%% Normalise data, per dataset
% Normalise across datasets or per dataset?
% Per dataset - all datasets will have the same range of values
% Across datasets - all datasets will be relative to discovery flies

% Normalisation per dataset

% Note - even with mixedSigmoid, some features scale to NaNs and 0s
%   discovery flies - feature 976 (870th valid feature) scales
%       to NaN a nd 0s
%   sleep flies - feature 977 scales to NaN and a 0

% Time to scale every channel, multidose8 - ~836s

dims = nan(length(cat_sets), 3); % Assumes TS_DataMat has 3 dimensions
for d = 1 : 1 %length(hctsas)
    disp(['scaling dataset ' dsets{d}]);

    hctsas{d}.TS_Normalised = nan(size(hctsas{d}.TS_DataMat));

    tic;

    tmp = hctsas{d}.TS_DataMat(valid_rows{d}, :, :);
    dims(d, :) = size(tmp);

    tmp = permute(tmp, [1 3 2]); % epoch/flies x features x channnels
    tmp = reshape(tmp, [dims(d, 1)*dims(d, 3) dims(d, 2)]); % epoch/flies/channels x features

    tmp = BF_NormalizeMatrix(tmp, 'mixedSigmoid');

    tmp = reshape(tmp, [dims(d, 1) dims(d, 3) dims(d, 2)]); % epoch/flies x channels x features
    tmp = permute(tmp, [1 3 2]); % epoch/flies x features x channels

    hctsas{d}.TS_Normalised(valid_rows{d}, :, :) = tmp;

    t = toc;
    disp(['ch' num2str(ch) ' scaled in ' num2str(t) 's']);

end

channels_together = hctsas{plot_dset}.TS_Normalised;

%% This plot looks nice (but it doesn't mean anything)
% Plots correlation between normalising all channels together vs
%   normalising each channel separately

figure;
nChannels = 15;

% Can use hsv2rgb() to specify colours with same saturation, value values
% Note for linspace - if from 0-360, first and last will be the same hue
%   Because 0 hue is the same as 360 hue
% Hue offset - which hue to start from (0-360)
offset = 220;
hues = mod(offset+floor(linspace(1, 360, nChannels+1)), 360) / 360;
%hues = hues(randperm(length(hues))); % shuffle colour order
sats = repmat(0.7, size(hues));
vals = repmat(1, size(hues));
colours = hsv2rgb([hues' sats' vals']);

%colours = (colormap(turbo(nChannels)));

dotSize = 1;

for ch = 1 : 15
    a = channels_separate(:, :, ch); % scale across all channels
    b = channels_together(:, :, ch); % scale per channel
    %scatter(a(:), b(:), dotSize, '.'); % default matlab colours
    scatter(a(:), b(:), dotSize, colours(ch, :), '.');
    hold on;
end

background_colour = 'k';

% set(gca, 'Color', background_colour);
axis off
set(gcf, 'Color', background_colour);

% Set paper size for exporting
%   Note - print '-r' option is resolution as dots per INCH
%   So, to set a particular resolution, just set '-r1' (DPI 1) and set
%       paper position in inches to however many pixels you want
set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 3840 2160]);
set(gcf, 'InvertHardCopy', false); % This is to keep the background colour

print_fig = 0;
out_name = 'test.png';
if print_fig == 1
    print(out_name, '-dpng', '-r1');
end
